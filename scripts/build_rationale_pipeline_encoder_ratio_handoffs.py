#!/usr/bin/env python3
"""Build the corrected OpenAI:SFT 1:1, 1:2, and 1:3 encoder views."""
from __future__ import annotations

import argparse
from collections import Counter
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

from mal2026.api_rationale_data import EXPECTED_ESSAYS, load_writing_rows, sha256_file  # noqa: E402
from mal2026.rationale_pipeline_prompts import AXES, normalize_rationales, round_half_up_score, routing  # noqa: E402


MERGED_PARENT = ROOT / "data/processed/restricted/rationale_v3_tail_sft_merged"
RATIO_GENERATION_PARENT = ROOT / "outputs/rationale-pipeline-final-ratio-generation-v1"
QUALITY_PARENT = ROOT / "outputs/rationale-pipeline-ratio-quality-gate-v1"
ADMISSION_PARENT = ROOT / "outputs/rationale-pipeline-ratio-encoder-admission-v1"
FINAL_PARENT = ROOT / "outputs/rationale-pipeline-final-generation-v1"
RESTRICTED_PARENT = ROOT / "data/processed/restricted/rationale_pipeline_v1/encoder_ratio_handoff"
OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-encoder-ratio-handoff-v2"
DIMENSIONS = ("score_rationale_consistency", "groundedness", "specificity", "domain_match")
OPENAI_VIEWS = 11_342
RATIOS = (1, 2, 3)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(path, 0o600)
    return sha256_file(path)


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


def normalize_single_student(path: Path, expected: int) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        source_id = str(row["source_id"])
        need(source_id not in seen and row.get("failure_category") is None, "single student rationale row differs")
        seen.add(source_id)
        result.append({"source_id": source_id, "rationales": normalize_rationales(row["rationales"])})
    need(len(result) == expected, "single student rationale population differs")
    return sorted(result, key=lambda row: row["source_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--merged-run-id", default="rationale-v3-tail-sft-merged-20260807-001")
    parser.add_argument("--ratio-generation-run-id", required=True)
    parser.add_argument("--quality-run-id", required=True)
    parser.add_argument("--admission-run-id", required=True)
    parser.add_argument("--single-student-run-id", required=True)
    parser.add_argument("--student-candidate-key", required=True)
    args = parser.parse_args()
    setproctitle(f"mal2026:rationale-encoder-ratio-handoffs:{args.run_id}"[:255])

    merged = MERGED_PARENT / args.merged_run_id
    merged_manifest_path = merged / "manifest.json"
    target_path = merged / "sft_targets.train.quality_filtered.jsonl"
    provenance_path = merged / "provenance.train.jsonl"
    need(all(path.is_file() for path in (merged_manifest_path, target_path, provenance_path)), "merged teacher rationale handoff unavailable")
    merged_manifest = json.loads(merged_manifest_path.read_text(encoding="utf-8"))
    need(merged_manifest.get("status") == "completed", "merged teacher rationale handoff incomplete")
    for path in (target_path, provenance_path):
        need(merged_manifest["files"][path.name]["sha256"] == sha256_file(path), "merged teacher rationale checksum differs")

    ratio_report_path = RATIO_GENERATION_PARENT / args.ratio_generation_run_id / "aggregate.json"
    quality_report_path = QUALITY_PARENT / args.quality_run_id / "aggregate.json"
    admission_report_path = ADMISSION_PARENT / args.admission_run_id / "aggregate.json"
    need(ratio_report_path.is_file() and quality_report_path.is_file() and admission_report_path.is_file(), "ratio generation, quality, or admission report unavailable")
    ratio_report = json.loads(ratio_report_path.read_text(encoding="utf-8"))
    quality_report = json.loads(quality_report_path.read_text(encoding="utf-8"))
    admission_report = json.loads(admission_report_path.read_text(encoding="utf-8"))
    need(ratio_report.get("status") == "completed" and ratio_report.get("candidate_key") == args.student_candidate_key, "ratio generation report differs")
    need(ratio_report.get("student_pool_records") == 3 * OPENAI_VIEWS and ratio_report.get("score_conditioning") is False, "ratio generation population differs")
    need(quality_report.get("status") in {"passed", "failed"} and quality_report.get("generation_run_id") == args.ratio_generation_run_id, "ratio quality evidence differs")
    need(admission_report.get("status") == "admitted_for_deployment_view_encoder_training_not_rationale_quality_promotion" and admission_report.get("quality_run_id") == args.quality_run_id, "ratio encoder admission differs")
    need(admission_report.get("quality_report_sha256") == sha256_file(quality_report_path), "ratio encoder admission quality lineage differs")
    student_pool_path = Path(ratio_report["train_path"])
    need(student_pool_path.is_file() and sha256_file(student_pool_path) == ratio_report["train_sha256"], "student ratio pool checksum differs")

    final_report_path = FINAL_PARENT / args.single_student_run_id / "aggregate.json"
    need(final_report_path.is_file(), "single student generation report unavailable")
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    need(final_report.get("status") == "completed" and final_report.get("candidate_key") == args.student_candidate_key, "single student generation report differs")
    need(final_report.get("score_conditioning") is False and final_report.get("human_or_reference_score_read_or_prompted") is False, "single student rationale generation is not score-blind")
    student_train_path = Path(final_report["train_path"]); student_validation_path = Path(final_report["validation_path"])
    need(final_report["train_sha256"] == sha256_file(student_train_path) and final_report["validation_sha256"] == sha256_file(student_validation_path), "single student rationale checksum differs")

    restricted_root = RESTRICTED_PARENT / args.run_id
    output_root = OUTPUT_PARENT / args.run_id
    need(not restricted_root.exists() and not output_root.exists(), "encoder ratio handoff output must be fresh")
    restricted_root.mkdir(parents=True, mode=0o700); output_root.mkdir(parents=True)

    targets = read_jsonl(target_path)
    need(len(targets) == OPENAI_VIEWS and len({str(row["candidate_key"]) for row in targets}) == OPENAI_VIEWS, "teacher candidate population differs")
    teacher_all = [
        {"source_id": str(row["source_id"]), "candidate_key": str(row["candidate_key"]), "rationales": normalize_rationales(row["rationale"])}
        for row in targets
    ]
    base_counts = Counter(row["source_id"] for row in teacher_all)
    need(len(base_counts) == EXPECTED_ESSAYS["train"] and sum(base_counts.values()) == OPENAI_VIEWS and min(base_counts.values()) >= 1, "teacher source multiplicity differs")

    provenance = {str(row["candidate_key"]): row for row in read_jsonl(provenance_path) if row.get("quality_filtered_included") is True}
    need(set(provenance) == {row["candidate_key"] for row in teacher_all}, "teacher provenance linkage differs")
    canonical = {row.identifier: row for row in load_writing_rows("train", include_scores=True)}
    source_scores: dict[str, dict[str, int]] = {}
    best: dict[str, tuple[int, str, dict[str, str]]] = {}
    target_lookup = {row["candidate_key"]: row for row in teacher_all}
    for key, row in provenance.items():
        source_id = str(row["source_id"]); need(source_id in canonical and key in target_lookup and target_lookup[key]["source_id"] == source_id, "teacher source linkage differs")
        integer_scores = {axis: int(row["integer_scores"][axis]) for axis in AXES}
        expected_scores = {axis: round_half_up_score((canonical[source_id].scores or {})[axis]) for axis in AXES}
        need(integer_scores == expected_scores, "teacher integer score differs from canonical ROUND_HALF_UP")
        previous_scores = source_scores.setdefault(source_id, integer_scores); need(previous_scores == integer_scores, "teacher source score conflict")
        total = judge_total(row["judge_scores"]); tie = sha256(f"2026080706:{source_id}:{key}".encode()).hexdigest()
        previous = best.get(source_id); rationales = target_lookup[key]["rationales"]
        if previous is None or (total, tie) > (previous[0], previous[1]): best[source_id] = (total, tie, rationales)
    need(len(source_scores) == len(best) == EXPECTED_ESSAYS["train"], "teacher source coverage differs")
    teacher_single = [{"source_id": source_id, "rationales": value[2]} for source_id, value in sorted(best.items())]

    pool_rows = read_jsonl(student_pool_path)
    need(len(pool_rows) == 3 * OPENAI_VIEWS, "student ratio pool record count differs")
    pool_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in pool_rows:
        source_id = str(row["source_id"]); variant = row.get("variant_index")
        need(source_id in base_counts and type(variant) is int and row.get("failure_category") is None, "student ratio pool row differs")
        pool_by_source.setdefault(source_id, []).append({"source_id": source_id, "variant_index": variant, "rationales": normalize_rationales(row["rationales"])})
    need(set(pool_by_source) == set(base_counts), "student ratio pool source coverage differs")
    for source_id, values in pool_by_source.items():
        values.sort(key=lambda row: row["variant_index"])
        need([row["variant_index"] for row in values] == list(range(3 * base_counts[source_id])), "student ratio variants are not contiguous")

    student_single_train = normalize_single_student(student_train_path, EXPECTED_ESSAYS["train"])
    student_single_validation = normalize_single_student(student_validation_path, EXPECTED_ESSAYS["validation"])
    need({row["source_id"] for row in student_single_train} == set(base_counts), "single student train source coverage differs")

    teacher_band_counts = {
        axis: {str(score): sum(count * (source_scores[source_id][axis] == score) for source_id, count in base_counts.items()) for score in range(1, 6)}
        for axis in AXES
    }
    shared = {
        "teacher_train_all": teacher_all,
        "teacher_train_single_best": teacher_single,
        "student_train_single": student_single_train,
        "student_validation_single": student_single_validation,
    }
    arm_manifests = {}
    for ratio in RATIOS:
        name = f"1to{ratio}"
        restricted = restricted_root / name; output = output_root / name
        restricted.mkdir(mode=0o700); output.mkdir()
        student_ratio = [row for source_id in sorted(pool_by_source) for row in pool_by_source[source_id][: ratio * base_counts[source_id]]]
        need(len(student_ratio) == ratio * OPENAI_VIEWS, "student ratio subset population differs")
        values = {**shared, "student_train_ratio": student_ratio}
        paths = {key: restricted / f"{key.replace('_', '.')}.jsonl" for key in values}
        digests = {key: write_jsonl(paths[key], rows) for key, rows in values.items()}
        records = {key: len(rows) for key, rows in values.items()}
        report = {
            "schema_version": "mal2026-rationale-pipeline-encoder-ratio-handoff-v2",
            "status": "completed", "run_id": args.run_id, "arm": name, "completed_at": now(),
            "openai_to_student_ratio": f"1:{ratio}",
            "training_views": ["openai_quality_filtered_all_train_only", f"student_score_blind_nested_{ratio}x"],
            "selection_dev_view": "student_score_blind_single_only",
            "validation_view": "student_score_blind_single_only",
            "source_disjoint_view_partition_required": True,
            "teacher_score_conditioned": True,
            "teacher_use": "train_only_label_aware_augmentation_never_validation_or_selection_dev",
            "student_score_conditioned": False,
            "student_multiplicity": "exact_per_source_openai_multiplicity_times_ratio",
            "records": records,
            "train_records_total": OPENAI_VIEWS + ratio * OPENAI_VIEWS,
            "paths": {key: str(path.resolve()) for key, path in paths.items()}, "sha256": digests,
            "teacher_axis_score_band_counts": teacher_band_counts,
            "student_axis_score_band_counts": {axis: {score: ratio * count for score, count in bands.items()} for axis, bands in teacher_band_counts.items()},
            "combined_axis_score_band_counts": {axis: {score: (1 + ratio) * count for score, count in bands.items()} for axis, bands in teacher_band_counts.items()},
            "merged_manifest_sha256": sha256_file(merged_manifest_path),
            "ratio_generation_report_sha256": sha256_file(ratio_report_path),
            "quality_gate_report_sha256": sha256_file(quality_report_path),
            "quality_gate_status": quality_report["status"],
            "encoder_admission_report_sha256": sha256_file(admission_report_path),
            "encoder_admission_scope": admission_report["scope"],
            "single_student_generation_report_sha256": sha256_file(final_report_path),
            "rationale_prompt_sha256": routing()["rationale_generation_training_evaluation"]["source_file_sha256"],
            "score_input_prompt_sha256": routing()["rationale_to_score_encoder"]["source_file_sha256"],
            "average_used": False,
            "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights",
        }
        atomic_json(output / "aggregate.json", report)
        atomic_json(restricted / "manifest.json", {**report, "aggregate_sha256": sha256_file(output / "aggregate.json")})
        arm_manifests[name] = {"path": str((output / "aggregate.json").resolve()), "sha256": sha256_file(output / "aggregate.json"), "train_records_total": report["train_records_total"]}

    parent = {
        "schema_version": "mal2026-rationale-pipeline-encoder-ratio-handoff-family-v2",
        "status": "completed", "run_id": args.run_id, "completed_at": now(),
        "arms": arm_manifests, "openai_views": OPENAI_VIEWS,
        "student_views": {"1to1": OPENAI_VIEWS, "1to2": 2 * OPENAI_VIEWS, "1to3": 3 * OPENAI_VIEWS},
        "quality_gate_report_sha256": sha256_file(quality_report_path), "average_used": False,
        "quality_gate_status": quality_report["status"], "encoder_admission_report_sha256": sha256_file(admission_report_path),
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights",
    }
    atomic_json(output_root / "aggregate.json", parent)
    atomic_json(restricted_root / "manifest.json", {**parent, "aggregate_sha256": sha256_file(output_root / "aggregate.json")})
    print(json.dumps({"status": "completed", "run_id": args.run_id, "arms": {name: value["train_records_total"] for name, value in arm_manifests.items()}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
