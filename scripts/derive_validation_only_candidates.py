#!/usr/bin/env python3
"""Derive a validation-only candidate artifact without opening source essays.

The parent candidate batch is already strictly validated.  This transformer
uses only its authoritative routing map to create a split-scoped validation
artifact, so the later frozen-evaluation scorer never opens train candidates.
All outputs remain beneath the ignored restricted-data root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESTRICTED = ROOT / "data/processed/restricted/openai_rationale_batches"
BATCH = "openai-rationale-terra-full-20260719-001"
SCHEMA = "rationale-v3-validation-only-candidate-artifact-v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def derive(run_id: str) -> dict:
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for char in run_id):
        raise ValueError("derived run id is invalid")
    source = RESTRICTED / BATCH
    manifest_path, aggregate_path = source / "manifest.json", source / "validation_aggregate.json"
    candidates_path, source_map_path = source / "candidates.jsonl", source / "source_map.jsonl"
    if any(not path.is_file() or path.is_symlink() for path in (manifest_path, aggregate_path, candidates_path, source_map_path)):
        raise RuntimeError("validated parent lineage is unavailable")
    parent, aggregate = json.loads(manifest_path.read_text(encoding="utf-8")), json.loads(aggregate_path.read_text(encoding="utf-8"))
    expected = {split: int(parent["splits"][split]) * int(parent["candidates_per_essay"]) for split in ("train", "validation")}
    if (parent.get("status") != "validated" or parent.get("candidates_sha256") != sha(candidates_path) or
            parent.get("source_map_sha256") != sha(source_map_path) or aggregate.get("status") != "strict_validation_complete" or
            aggregate.get("candidates_sha256") != parent["candidates_sha256"] or aggregate.get("candidate_records") != sum(expected.values())):
        raise RuntimeError("parent manifest/aggregate does not prove the candidate population")

    routes: dict[str, tuple[str, int]] = {}
    split_counts = {"train": 0, "validation": 0}
    source_sets = {"train": set(), "validation": set()}
    with source_map_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            custom_id, source_id, split, candidate = row.get("custom_id"), row.get("source_id"), row.get("split"), row.get("candidate")
            if (not isinstance(custom_id, str) or not isinstance(source_id, (str, int)) or split not in split_counts or
                    not isinstance(candidate, int) or candidate not in (1, 2, 3) or custom_id in routes):
                raise RuntimeError("authoritative source map is invalid")
            routes[custom_id] = (split, candidate)
            split_counts[split] += 1
            source_sets[split].add(str(source_id))
    if (split_counts != expected or source_sets["train"] & source_sets["validation"] or
            len(routes) != sum(expected.values())):
        raise RuntimeError("source map does not prove disjoint complete split routing")

    final = source / "derived" / run_id
    if final.exists() or final.is_symlink():
        raise FileExistsError("refusing to overwrite validation-only artifact")
    final.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging = final.parent / f".{run_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    seen: set[str] = set()
    try:
        staging.mkdir(mode=0o700)
        output = staging / "candidates.validation.jsonl"
        out_count = 0
        with candidates_path.open("rb") as source_handle, output.open("xb") as destination:
            for line in source_handle:
                row = json.loads(line)
                custom_id, candidate, schema = row.get("custom_id"), row.get("candidate"), row.get("schema_version")
                route = routes.get(custom_id) if isinstance(custom_id, str) else None
                if (route is None or candidate != route[1] or schema != "rationale-v3-sentence-id" or
                        not isinstance(row.get("rationale"), dict) or custom_id in seen):
                    raise RuntimeError("candidate row does not match the authoritative routing map")
                seen.add(custom_id)
                if route[0] == "validation":
                    destination.write(line)
                    out_count += 1
            destination.flush(); os.fsync(destination.fileno())
        if len(seen) != sum(expected.values()) or out_count != expected["validation"]:
            raise RuntimeError("validation artifact routing is incomplete")
        proof = {
            "candidate_custom_id_duplicates": 0,
            "source_candidate_duplicates": {"train": 0, "validation": 0},
            "train_validation_source_id_overlap": 0,
            "train_validation_candidate_key_overlap": 0,
            "unmapped_or_mismatched_candidates": 0,
            "train_rows_in_new_artifact": 0,
            "train_requests_constructed": 0,
            "train_source_text_opened": 0,
        }
        payload = {
            "schema_version": SCHEMA, "status": "completed", "created_at": now(), "batch_run_id": BATCH,
            "split": "validation", "candidate_file": output.name, "candidate_file_sha256": sha(output),
            "row_count": out_count, "parent_manifest_sha256": sha(manifest_path),
            "parent_validation_aggregate_sha256": sha(aggregate_path), "parent_candidate_file_sha256": sha(candidates_path),
            "parent_source_map_sha256": sha(source_map_path), "input_candidate_counts": expected,
            "output_candidate_counts": {"train": 0, "validation": out_count},
            "input_source_counts": {split: len(source_sets[split]) for split in source_sets},
            "output_source_counts": {"train": 0, "validation": len(source_sets["validation"])},
            "proof": proof,
        }
        target = staging / "candidates.validation.manifest.json"
        with target.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        fsync(staging); os.replace(staging, final); fsync(final.parent)
    except Exception:
        if staging.exists(): shutil.rmtree(staging)
        raise
    return {"status": "completed", "derived_run_id": run_id, "split": "validation", "row_count": out_count,
            "candidate_file_sha256": payload["candidate_file_sha256"], "train_source_text_opened": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-run-id", required=True)
    print(json.dumps(derive(parser.parse_args().derived_run_id), sort_keys=True))
