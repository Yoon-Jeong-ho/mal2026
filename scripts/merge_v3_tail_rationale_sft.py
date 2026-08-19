#!/usr/bin/env python3
"""Merge independently judged frozen-v3 rationale SFT handoffs.

The restricted target files remain score-free.  Labels and judge metadata stay
in separate restricted provenance files.  Exact duplicate rationales are
removed only within the same source and split, so identical wording attached
to different essays can never collapse two training examples.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "data/processed/restricted/rationale_v3_tail_sft"
PRIVATE_ROOT = ROOT / "data/processed/restricted/rationale_v3_tail_sft_merged"
OUTPUT_ROOT = ROOT / "outputs/rationale-v3-tail-sft-merged"
SPLITS = ("train", "validation")
VIEWS = ("valid", "quality_filtered")


class MergeError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise MergeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.chmod(path, 0o600)
    return count, file_sha(path)


def remapped_key(handoff: str, key: str) -> str:
    return sha256(f"{handoff}\0{key}".encode()).hexdigest()


def normalized_target(rationale: Mapping[str, Any]) -> str:
    return re.sub(r"\s+", "", json.dumps(rationale, ensure_ascii=False, sort_keys=True))


def load_handoff(run_id: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = INPUT_ROOT / run_id
    manifest_path = root / "manifest.json"
    aggregate_path = ROOT / "outputs/rationale-v3-tail-sft" / run_id / "aggregate.json"
    need(manifest_path.is_file() and aggregate_path.is_file(), f"handoff unavailable: {run_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    need(manifest.get("status") == aggregate.get("status") == "completed", f"handoff incomplete: {run_id}")
    need(manifest.get("run_id") == aggregate.get("run_id") == run_id, f"handoff identity differs: {run_id}")
    need(manifest.get("aggregate_sha256") == file_sha(aggregate_path), f"handoff aggregate differs: {run_id}")
    for name, metadata in manifest.get("files", {}).items():
        path = root / name
        need(path.is_file() and file_sha(path) == metadata.get("sha256"), f"handoff file differs: {run_id}/{name}")
    return manifest, aggregate, root


def merge(args: argparse.Namespace) -> dict[str, Any]:
    need(len(args.handoff_run) >= 2 and len(set(args.handoff_run)) == len(args.handoff_run), "at least two distinct handoffs are required")
    private = PRIVATE_ROOT / args.run_id
    public = OUTPUT_ROOT / args.run_id
    need(not private.exists() and not public.exists(), "merge output must be fresh")
    private.mkdir(parents=True, mode=0o700)
    public.mkdir(parents=True)

    inputs: list[dict[str, Any]] = []
    targets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    provenances: dict[str, list[dict[str, Any]]] = defaultdict(list)
    provenance_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    train_sources: set[str] = set()
    validation_sources: set[str] = set()

    for handoff in args.handoff_run:
        manifest, aggregate, root = load_handoff(handoff)
        inputs.append({
            "run_id": handoff,
            "judge_campaign": aggregate["judge_campaign"],
            "manifest_sha256": file_sha(root / "manifest.json"),
            "aggregate_sha256": file_sha(ROOT / "outputs/rationale-v3-tail-sft" / handoff / "aggregate.json"),
            "mechanically_valid_targets": aggregate["mechanically_valid_targets"],
            "quality_filtered_targets": aggregate["quality_filtered_targets"],
        })
        for split in SPLITS:
            provenance_rows = jsonl(root / f"provenance.{split}.jsonl")
            old_keys = {str(row["candidate_key"]) for row in provenance_rows}
            need(len(old_keys) == len(provenance_rows), f"provenance keys differ: {handoff}/{split}")
            for row in provenance_rows:
                old_key = str(row["candidate_key"])
                new_key = remapped_key(handoff, old_key)
                mapped = {**row, "candidate_key": new_key, "source_handoff": handoff, "source_candidate_key": old_key}
                provenances[split].append(mapped)
                provenance_by_key[(handoff, old_key)] = mapped
            for view in VIEWS:
                rows = jsonl(root / f"sft_targets.{split}.{view}.jsonl")
                need(all(str(row["candidate_key"]) in old_keys for row in rows), f"target has no provenance: {handoff}/{split}/{view}")
                for row in rows:
                    targets[(split, view)].append({
                        "candidate_key": remapped_key(handoff, str(row["candidate_key"])),
                        "source_id": str(row["source_id"]),
                        "rationale": row["rationale"],
                    })
            source_set = {str(row["source_id"]) for row in provenance_rows}
            (train_sources if split == "train" else validation_sources).update(source_set)

    need(train_sources.isdisjoint(validation_sources), "train and validation source IDs overlap")
    all_keys = [str(row["candidate_key"]) for split in SPLITS for row in provenances[split]]
    need(len(all_keys) == len(set(all_keys)), "remapped candidate keys differ")

    files: dict[str, dict[str, Any]] = {}
    duplicate_counts: Counter[tuple[str, str]] = Counter()
    merged_counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    teacher_counts: Counter[tuple[str, str, str]] = Counter()
    band_counts: Counter[tuple[str, str, int]] = Counter()

    for split in SPLITS:
        for row in provenances[split]:
            teacher_counts[(split, str(row["teacher_model"]), str(row["source_handoff"]))] += 1
            for axis, score in row["integer_scores"].items():
                band_counts[(split, str(axis), int(score))] += 1
        prov_name = f"provenance.{split}.jsonl"
        count, digest = write_jsonl(private / prov_name, provenances[split])
        files[prov_name] = {"records": count, "sha256": digest, "contains_scores": True}

        for view in VIEWS:
            seen: set[tuple[str, str]] = set()
            kept: list[dict[str, Any]] = []
            for row in sorted(targets[(split, view)], key=lambda value: (value["source_id"], value["candidate_key"])):
                fingerprint = (str(row["source_id"]), normalized_target(row["rationale"]))
                if fingerprint in seen:
                    duplicate_counts[(split, view)] += 1
                    continue
                seen.add(fingerprint)
                kept.append(row)
            name = f"sft_targets.{split}.{view}.jsonl"
            count, digest = write_jsonl(private / name, kept)
            files[name] = {"records": count, "sha256": digest, "contains_scores": False}
            merged_counts[split][view] = count

    summary = {
        "schema_version": "mal2026-rationale-v3-tail-sft-merged-aggregate-v1",
        "status": "completed",
        "run_id": args.run_id,
        "created_at": now(),
        "inputs": inputs,
        "merged_targets": merged_counts,
        "exact_within_source_duplicates_removed": {
            split: {view: duplicate_counts[(split, view)] for view in VIEWS} for split in SPLITS
        },
        "source_counts": {"train": len(train_sources), "validation": len(validation_sources)},
        "train_validation_source_overlap": 0,
        "teacher_handoff_candidate_counts": {
            split: {
                handoff: {
                    teacher: teacher_counts[(split, teacher, handoff)]
                    for teacher in ("gpt-5.6-luna", "gpt-5.6-terra")
                }
                for handoff in args.handoff_run
            }
            for split in SPLITS
        },
        "axis_band_candidate_counts": {
            split: {
                axis: {str(score): band_counts[(split, axis, score)] for score in range(1, 6)}
                for axis in ("content", "organization", "expression")
            }
            for split in SPLITS
        },
        "selection_policy": "union of each handoff view; exact target deduplication only within identical split+source; quality labels are not recomputed",
        "target_contract": "score-free target files; integer labels and judge metadata remain only in separate restricted provenance",
        "files": files,
        "privacy": "aggregate_only_no_source_ids_prompts_essays_rationales_or_judge_evidence",
    }
    atomic_json(public / "aggregate.json", summary)
    manifest = {
        "schema_version": "mal2026-rationale-v3-tail-sft-merged-manifest-v1",
        "status": "completed",
        "run_id": args.run_id,
        "created_at": summary["created_at"],
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "inputs": inputs,
        "files": files,
        "aggregate_sha256": file_sha(public / "aggregate.json"),
        "privacy": "restricted targets/provenance; aggregate output only",
    }
    atomic_json(private / "manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--handoff-run", action="append", required=True)
    args = parser.parse_args()
    need(bool(re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,120}", args.run_id)), "invalid run ID")
    setproctitle(f"mal2026:merge-v3-tail-rationale-sft:{args.run_id}"[:255])
    result = merge(args)
    print(json.dumps({"status": result["status"], "run_id": args.run_id, "merged_targets": result["merged_targets"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
