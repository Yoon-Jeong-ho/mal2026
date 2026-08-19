#!/usr/bin/env python3
"""Atomically derive a train-only restricted candidate artifact.

This utility is intentionally a lineage transformer, not a prompt builder.  It
never opens evaluation source files, emits no record contents or identifiers,
and routes combined candidates only through the validated parent source map.
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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESTRICTED_ROOT = ROOT / "data/processed/restricted/openai_rationale_batches"
ARTIFACT_SCHEMA = "rationale-v3-train-only-candidate-artifact-v1"
CANDIDATE_SCHEMA = "rationale-v3-sentence-id"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_set_sha256(values: set[str]) -> str:
    """Commit to an identifier set without placing its members in a manifest."""
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_fsynced(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("required aggregate manifest is not an object")
    return value


def checked_parent(source_dir: Path, batch_run_id: str) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    manifest_path = source_dir / "manifest.json"
    aggregate_path = source_dir / "validation_aggregate.json"
    candidates_path = source_dir / "candidates.jsonl"
    source_map_path = source_dir / "source_map.jsonl"
    if any(not path.is_file() or path.is_symlink() for path in (manifest_path, aggregate_path, candidates_path, source_map_path)):
        raise RuntimeError("validated parent lineage inputs are unavailable")
    parent = read_json(manifest_path)
    aggregate = read_json(aggregate_path)
    if parent.get("run_id") != batch_run_id or parent.get("status") != "validated":
        raise RuntimeError("parent lineage is not the requested validated run")
    if not isinstance(parent.get("splits"), dict) or not isinstance(parent.get("candidates_per_essay"), int):
        raise RuntimeError("parent split metadata is unavailable")
    if parent.get("candidates_sha256") != sha256(candidates_path) or parent.get("source_map_sha256") != sha256(source_map_path):
        raise RuntimeError("parent artifact checksum does not match its manifest")
    expected = sum(parent["splits"].values()) * parent["candidates_per_essay"]
    if (set(parent["splits"]) != {"train", "validation"} or expected <= 0 or parent.get("accepted") != expected or
            aggregate.get("status") != "strict_validation_complete" or aggregate.get("batch_run_id") != batch_run_id or
            aggregate.get("candidates_sha256") != parent["candidates_sha256"] or
            aggregate.get("accepted_records") != expected or aggregate.get("candidate_records") != expected or
            aggregate.get("mapping_records") != expected or
            any(aggregate.get(field) != 0 for field in (
                "candidate_duplicate_records", "candidate_unknown_records", "candidate_schema_or_grounding_invalid_records",
                "mapping_duplicate_records", "missing_records", "rejected_records"))):
        raise RuntimeError("parent aggregate validation does not prove a complete clean candidate population")
    return parent, aggregate, candidates_path, source_map_path


def mapping_index(source_map_path: Path, expected: int) -> tuple[dict[str, tuple[str, str, int]], dict[str, set[str]]]:
    """Read only routing fields from the authoritative mapping, never source text."""
    mappings: dict[str, tuple[str, str, int]] = {}
    source_ids = {"train": set(), "validation": set()}
    tuples: set[tuple[str, int]] = set()
    with source_map_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise RuntimeError("source map contains a blank record")
            row = json.loads(line)
            custom_id, source_id, split, candidate = row.get("custom_id"), row.get("source_id"), row.get("split"), row.get("candidate")
            if (not isinstance(custom_id, str) or not isinstance(source_id, (str, int)) or split not in source_ids or
                    not isinstance(candidate, int) or candidate < 1):
                raise RuntimeError("source map routing schema is invalid")
            source_key = str(source_id)
            key = (source_key, candidate)
            if custom_id in mappings or key in tuples:
                raise RuntimeError("source map has duplicate routing keys")
            mappings[custom_id] = (source_key, split, candidate)
            source_ids[split].add(source_key)
            tuples.add(key)
    if len(mappings) != expected or source_ids["train"] & source_ids["validation"]:
        raise RuntimeError("source map does not prove complete disjoint split routing")
    return mappings, source_ids


def derive(parent_run_id: str, derived_run_id: str) -> dict[str, Any]:
    if not derived_run_id or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for ch in derived_run_id):
        raise ValueError("derived run id is invalid")
    source_dir = RESTRICTED_ROOT / parent_run_id
    parent, aggregate, candidates_path, source_map_path = checked_parent(source_dir, parent_run_id)
    expected = int(parent["accepted"])
    mappings, mapped_source_ids = mapping_index(source_map_path, expected)
    expected_counts = {split: parent["splits"][split] * parent["candidates_per_essay"] for split in ("train", "validation")}
    if any(not isinstance(value, int) or value <= 0 for value in expected_counts.values()):
        raise RuntimeError("parent split counts are invalid")

    parent_manifest_sha = sha256(source_dir / "manifest.json")
    parent_aggregate_sha = sha256(source_dir / "validation_aggregate.json")
    final_dir = source_dir / "derived" / derived_run_id
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError("derived run directory already exists; refusing overwrite")
    final_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if final_dir.parent.is_symlink():
        raise RuntimeError("derived artifact parent may not be a symlink")
    staging = final_dir.parent / f".{derived_run_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    seen_custom_ids: set[str] = set()
    seen_tuples: dict[str, set[tuple[str, int]]] = {"train": set(), "validation": set()}
    seen_source_ids: dict[str, set[str]] = {"train": set(), "validation": set()}
    input_counts = {"train": 0, "validation": 0}
    output_counts = {"train": 0, "validation": 0}
    try:
        staging.mkdir(mode=0o700)
        output_path = staging / "candidates.train.jsonl"
        with candidates_path.open("rb") as source, output_path.open("xb") as output:
            for raw_line in source:
                if not raw_line.strip():
                    raise RuntimeError("combined candidate artifact contains a blank record")
                # Parse only routing and schema fields; do not examine rationale text.
                row = json.loads(raw_line)
                custom_id, source_id, candidate, schema = row.get("custom_id"), row.get("source_id"), row.get("candidate"), row.get("schema_version")
                if (not isinstance(custom_id, str) or not isinstance(source_id, (str, int)) or not isinstance(candidate, int) or
                        schema != CANDIDATE_SCHEMA or not isinstance(row.get("rationale"), dict)):
                    raise RuntimeError("combined candidate routing/schema fields are invalid")
                mapped = mappings.get(custom_id)
                source_key = str(source_id)
                if mapped is None or mapped[0] != source_key or mapped[2] != candidate or custom_id in seen_custom_ids:
                    raise RuntimeError("combined candidate does not match authoritative source map uniquely")
                split = mapped[1]
                key = (source_key, candidate)
                if key in seen_tuples[split]:
                    raise RuntimeError("combined candidate has a duplicate source/candidate routing key")
                seen_custom_ids.add(custom_id)
                seen_tuples[split].add(key)
                seen_source_ids[split].add(source_key)
                input_counts[split] += 1
                if split == "train":
                    output.write(raw_line)
                    output_counts["train"] += 1
            output.flush()
            os.fsync(output.fileno())
        if (len(seen_custom_ids) != expected or input_counts != expected_counts or output_counts != {"train": expected_counts["train"], "validation": 0} or
                seen_source_ids != mapped_source_ids or seen_source_ids["train"] & seen_source_ids["validation"]):
            raise RuntimeError("routing counts, completeness, deduplication, or split isolation failed")
        if sha256(candidates_path) != parent["candidates_sha256"] or sha256(source_map_path) != parent["source_map_sha256"]:
            raise RuntimeError("parent inputs changed during derivation")
        manifest = {
            "schema_version": ARTIFACT_SCHEMA,
            "status": "completed",
            "created_at": now(),
            "batch_run_id": parent_run_id,
            "split": "train",
            "candidate_file": "candidates.train.jsonl",
            "candidate_file_sha256": sha256(output_path),
            "row_count": output_counts["train"],
            "parent_manifest_sha256": parent_manifest_sha,
            "parent_validation_aggregate_sha256": parent_aggregate_sha,
            "parent_candidate_file_sha256": parent["candidates_sha256"],
            "parent_source_map_sha256": parent["source_map_sha256"],
            "parent_candidate_schema": CANDIDATE_SCHEMA,
            "parent_validation_status": aggregate["status"],
            "routing_authority": "parent_source_map.jsonl",
            "input_candidate_counts": input_counts,
            "output_candidate_counts": output_counts,
            "input_source_counts": {split: len(seen_source_ids[split]) for split in ("train", "validation")},
            "output_source_counts": {"train": len(seen_source_ids["train"]), "validation": 0},
            "input_source_id_set_sha256": {split: canonical_set_sha256(seen_source_ids[split]) for split in ("train", "validation")},
            "output_source_id_set_sha256": {"train": canonical_set_sha256(seen_source_ids["train"]), "validation": None},
            "proof": {
                "candidate_schema_validation": "parent_strict_validation_inherited_and_routing_schema_rechecked",
                "candidate_custom_id_duplicates": 0,
                "source_candidate_duplicates": {"train": 0, "validation": 0},
                "train_validation_source_id_overlap": 0,
                "train_validation_candidate_key_overlap": 0,
                "unmapped_or_mismatched_candidates": 0,
                "validation_rows_in_new_artifact": 0,
                "validation_requests_constructed": 0,
                "validation_source_text_opened": 0,
            },
        }
        manifest_path = staging / "candidates.train.manifest.json"
        write_json_fsynced(manifest_path, manifest)
        fsync_path(staging)
        os.replace(staging, final_dir)
        fsync_path(final_dir.parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"status": "completed", "derived_run_id": derived_run_id, "row_count": output_counts["train"],
            "input_candidate_counts": input_counts, "output_candidate_counts": output_counts,
            "validation_rows_in_new_artifact": 0, "validation_requests_constructed": 0,
            "candidate_file_sha256": manifest["candidate_file_sha256"], "manifest_sha256": sha256(final_dir / "candidates.train.manifest.json")}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--parent-run-id", required=True)
    value.add_argument("--derived-run-id", required=True)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(derive(args.parent_run_id, args.derived_run_id), sort_keys=True))
