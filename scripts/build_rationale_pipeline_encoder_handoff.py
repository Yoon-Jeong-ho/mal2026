#!/usr/bin/env python3
"""Build train-only teacher plus score-blind student rationale encoder views."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import EXPECTED_ESSAYS, sha256_file  # noqa: E402
from mal2026.rationale_pipeline_prompts import AXES, normalize_rationales, routing  # noqa: E402


MERGED_PARENT = ROOT / "data/processed/restricted/rationale_v3_tail_sft_merged"
GENERATION_PARENT = ROOT / "data/processed/restricted/rationale_pipeline_v1/generation"
FINAL_PARENT = ROOT / "outputs/rationale-pipeline-final-generation-v1"
RESTRICTED_PARENT = ROOT / "data/processed/restricted/rationale_pipeline_v1/encoder_handoff"
OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-encoder-handoff-v1"
DIMENSIONS = ("score_rationale_consistency", "groundedness", "specificity", "domain_match")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def judge_total(value: Mapping[str, Any]) -> int:
    need(set(value) == set(AXES), "encoder teacher judge axes differ")
    total = 0
    for axis in AXES:
        need(set(value[axis]) == set(DIMENSIONS), "encoder teacher judge dimensions differ")
        for dimension in DIMENSIONS:
            cell = value[axis][dimension]
            score = cell.get("score") if isinstance(cell, Mapping) else cell
            need(type(score) is int and 1 <= score <= 5, "encoder teacher judge score differs")
            total += score
    return total


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(path, 0o600)
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--merged-run-id", default="rationale-v3-tail-sft-merged-20260807-001")
    parser.add_argument("--student-run-id", required=True)
    parser.add_argument("--student-candidate-key", required=True)
    args = parser.parse_args()
    setproctitle(f"mal2026:rationale-encoder-handoff:{args.run_id}"[:255])
    merged = MERGED_PARENT / args.merged_run_id
    merged_manifest_path = merged / "manifest.json"
    target_path = merged / "sft_targets.train.quality_filtered.jsonl"
    provenance_path = merged / "provenance.train.jsonl"
    need(all(path.is_file() for path in (merged_manifest_path, target_path, provenance_path)), "merged teacher rationale handoff unavailable")
    merged_manifest = json.loads(merged_manifest_path.read_text(encoding="utf-8"))
    need(merged_manifest.get("status") == "completed", "merged teacher rationale handoff incomplete")
    for path in (target_path, provenance_path):
        need(merged_manifest["files"][path.name]["sha256"] == sha256_file(path), "merged teacher rationale checksum differs")

    final_report_path = FINAL_PARENT / args.student_run_id / "aggregate.json"
    need(final_report_path.is_file(), "final student generation report unavailable")
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    need(final_report.get("status") == "completed" and final_report.get("candidate_key") == args.student_candidate_key, "final student generation report differs")
    need(final_report.get("score_conditioning") is False and final_report.get("human_or_reference_score_read_or_prompted") is False, "student rationale generation is not score-blind")
    student_train_path = Path(final_report["train_path"])
    student_validation_path = Path(final_report["validation_path"])
    need(final_report["train_sha256"] == sha256_file(student_train_path) and final_report["validation_sha256"] == sha256_file(student_validation_path), "student rationale checksum differs")

    restricted = RESTRICTED_PARENT / args.run_id
    output = OUTPUT_PARENT / args.run_id
    need(not restricted.exists() and not output.exists(), "encoder rationale handoff output must be fresh")
    restricted.mkdir(parents=True, mode=0o700); output.mkdir(parents=True)

    targets = {str(row["candidate_key"]): row for row in rows(target_path)}
    best: dict[str, tuple[int, str, dict[str, str]]] = {}
    for row in rows(provenance_path):
        if row.get("quality_filtered_included") is not True:
            continue
        key = str(row["candidate_key"]); source_id = str(row["source_id"])
        need(key in targets and str(targets[key]["source_id"]) == source_id, "teacher target/provenance linkage differs")
        rationales = normalize_rationales(targets[key]["rationale"])
        total = judge_total(row["judge_scores"])
        tie = sha256(f"2026080706:{source_id}:{key}".encode()).hexdigest()
        previous = best.get(source_id)
        if previous is None or (total, tie) > (previous[0], previous[1]):
            best[source_id] = (total, tie, rationales)
    need(len(best) == EXPECTED_ESSAYS["train"], "teacher rationale source coverage differs")
    teacher_rows = [{"source_id": source_id, "rationales": value[2]} for source_id, value in sorted(best.items())]

    def normalize_student(path: Path, expected: int) -> list[dict[str, Any]]:
        result = []
        seen: set[str] = set()
        for row in rows(path):
            source_id = str(row["source_id"])
            need(source_id not in seen and row.get("failure_category") is None, "student rationale row differs")
            seen.add(source_id)
            result.append({"source_id": source_id, "rationales": normalize_rationales(row["rationales"])})
        need(len(result) == expected, "student rationale population differs")
        return sorted(result, key=lambda row: row["source_id"])

    student_train = normalize_student(student_train_path, EXPECTED_ESSAYS["train"])
    student_validation = normalize_student(student_validation_path, EXPECTED_ESSAYS["validation"])
    need({row["source_id"] for row in teacher_rows} == {row["source_id"] for row in student_train}, "teacher/student train population differs")
    paths = {
        "teacher_train": restricted / "teacher.train.jsonl",
        "student_train": restricted / "student.train.jsonl",
        "student_validation": restricted / "student.validation.jsonl",
    }
    digests = {
        "teacher_train": write_jsonl(paths["teacher_train"], teacher_rows),
        "student_train": write_jsonl(paths["student_train"], student_train),
        "student_validation": write_jsonl(paths["student_validation"], student_validation),
    }
    report = {
        "schema_version": "mal2026-rationale-pipeline-encoder-handoff-v1",
        "status": "completed", "run_id": args.run_id, "completed_at": now(),
        "training_views": ["teacher_exact_q4_best_train_only", "student_score_blind_train"],
        "selection_dev_view": "student_score_blind_only",
        "validation_view": "student_score_blind_only",
        "source_disjoint_view_partition_required": True,
        "teacher_score_conditioned": True,
        "teacher_use": "train_only_label_aware_augmentation_never_validation_or_selection_dev",
        "student_score_conditioned": False,
        "records": {"teacher_train": 2000, "student_train": 2000, "student_validation": 400},
        "paths": {key: str(path.resolve()) for key, path in paths.items()},
        "sha256": digests,
        "merged_manifest_sha256": sha256_file(merged_manifest_path),
        "student_generation_report_sha256": sha256_file(final_report_path),
        "rationale_prompt_sha256": routing()["rationale_generation_training_evaluation"]["source_file_sha256"],
        "score_input_prompt_sha256": routing()["rationale_to_score_encoder"]["source_file_sha256"],
        "average_used": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights",
    }
    atomic_json(output / "aggregate.json", report)
    atomic_json(restricted / "manifest.json", {**report, "aggregate_sha256": sha256_file(output / "aggregate.json")})
    print(json.dumps({"status": "completed", "run_id": args.run_id, "teacher_train": 2000, "student_train": 2000, "student_validation": 400}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
