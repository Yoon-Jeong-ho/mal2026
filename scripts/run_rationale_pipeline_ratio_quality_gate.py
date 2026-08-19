#!/usr/bin/env python3
"""Stratified exact-Q4 quality gate for the 3x student rationale pool."""
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
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from mal2026.api_rationale_data import load_writing_rows, sha256_file  # noqa: E402
from mal2026.official_writing_contract import JUDGE_DIMENSIONS  # noqa: E402
from mal2026.rationale_pipeline_prompts import AXES, round_half_up_score, routing  # noqa: E402
from run_rationale_pipeline_sft_evaluation import (  # noqa: E402
    GPUS, Q4_AGGREGATE, Q4_PORTS, Q4_RESTRICTED, band_metrics, jsonl,
    launch_q4, participant_file, run_q4, stop_owned, wait_released,
)


GENERATION_PARENT = ROOT / "outputs/rationale-pipeline-final-ratio-generation-v1"
RESTRICTED_PARENT = ROOT / "data/processed/restricted/rationale_pipeline_v1/ratio_quality_gate"
OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-ratio-quality-gate-v1"
SAMPLE_SIZE = 400


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def deterministic_key(source_id: str) -> str:
    return sha256(f"2026080803:{source_id}".encode()).hexdigest()


def tranche(source_id: str) -> int:
    return int(deterministic_key(source_id)[:8], 16) % 3 + 1


def select_sources() -> tuple[set[str], dict[str, dict[str, int]]]:
    writings = load_writing_rows("train", include_scores=True)
    bands = {
        row.identifier: {axis: round_half_up_score((row.scores or {})[axis]) for axis in AXES}
        for row in writings
    }
    selected = {source_id for source_id, value in bands.items() if 1 in value.values()}
    # Include at least 50 independently ordered sources from every 2/5 cell,
    # or the entire cell when support is smaller. All score-1 sources are kept.
    for axis in AXES:
        for score in (2, 5):
            candidates = sorted((source_id for source_id, value in bands.items() if value[axis] == score), key=deterministic_key)
            selected.update(candidates[: min(50, len(candidates))])
    for source_id in sorted(bands, key=deterministic_key):
        if len(selected) >= SAMPLE_SIZE:
            break
        selected.add(source_id)
    need(len(selected) == SAMPLE_SIZE, "ratio quality sample population differs")
    counts = {
        axis: {str(score): sum(bands[source_id][axis] == score for source_id in selected) for score in range(1, 6)}
        for axis in AXES
    }
    need(all(counts[axis]["1"] == sum(value[axis] == 1 for value in bands.values()) for axis in AXES), "ratio quality sample omitted score-1 source")
    return selected, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generation-run-id", required=True)
    parser.add_argument("--candidate-key", required=True)
    args = parser.parse_args()
    setproctitle(f"mal2026:rationale-ratio-quality-gate:{args.run_id}"[:255])

    generation_report_path = GENERATION_PARENT / args.generation_run_id / "aggregate.json"
    need(generation_report_path.is_file(), "ratio generation report unavailable")
    generation_report = json.loads(generation_report_path.read_text(encoding="utf-8"))
    need(generation_report.get("status") == "completed" and generation_report.get("candidate_key") == args.candidate_key, "ratio generation report differs")
    need(generation_report.get("student_pool_records") == 34_026 and generation_report.get("score_conditioning") is False, "ratio generation pool contract differs")
    generated_path = Path(generation_report["train_path"])
    need(generated_path.is_file() and sha256_file(generated_path) == generation_report["train_sha256"], "ratio generation pool checksum differs")

    restricted = RESTRICTED_PARENT / args.run_id
    output = OUTPUT_PARENT / args.run_id
    need(not restricted.exists() and not output.exists(), "ratio quality output must be fresh")
    restricted.mkdir(parents=True, mode=0o700); output.mkdir(parents=True)
    selected, sample_band_counts = select_sources()
    all_rows = jsonl(generated_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        source_id = str(row["source_id"])
        need(row.get("failure_category") is None and isinstance(row.get("variant_index"), int), "ratio quality generated row differs")
        grouped.setdefault(source_id, []).append(row)
    need(len(grouped) == 2_000 and sum(map(len, grouped.values())) == 34_026, "ratio quality generated population differs")

    # Cover all nested tranches: the end of the 1x prefix, the 2x-only added
    # tranche, or the 3x-only added tranche, deterministically by source.
    sample_rows = []
    sample_tranches: dict[str, int] = {}
    for source_id in sorted(selected):
        choices = sorted(grouped[source_id], key=lambda row: int(row["variant_index"]))
        need([int(row["variant_index"]) for row in choices] == list(range(len(choices))), "ratio quality variant sequence differs")
        need(len(choices) % 3 == 0 and len(choices) >= 3, "ratio quality source multiplicity differs")
        base = len(choices) // 3; selected_tranche = tranche(source_id)
        sample_rows.append(choices[selected_tranche * base - 1]); sample_tranches[source_id] = selected_tranche
    sample_path = restricted / "sample.generated_rationales.jsonl"
    with sample_path.open("x", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(sample_path, 0o600)
    candidate = {"key": args.candidate_key}
    participant = participant_file(args.run_id, candidate, "train", sample_path, restricted / "participants.sample400.jsonl")
    smoke_participant = restricted / "participants.smoke1.jsonl"
    smoke_participant.write_text(participant.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    os.chmod(smoke_participant, 0o600)

    processes = []
    try:
        processes, endpoints, attestation = launch_q4((0,), (Q4_PORTS[0],), output / "runtime-smoke")
        smoke_report = run_q4(f"{args.run_id}-smoke1", smoke_participant, "train", 1, endpoints, attestation, output / "telemetry-smoke.jsonl", (0,))
    finally:
        if processes:
            stop_owned(processes); wait_released((0,))
    processes = []
    try:
        processes, endpoints, attestation = launch_q4(GPUS, Q4_PORTS, output / "runtime-full")
        full_report = run_q4(args.run_id, participant, "train", SAMPLE_SIZE, endpoints, attestation, output / "telemetry-full.jsonl", GPUS)
    finally:
        if processes:
            stop_owned(processes); wait_released(GPUS)

    judge_report = json.loads(full_report.read_text(encoding="utf-8"))
    judge_records = Q4_RESTRICTED / args.run_id / "judge_records.jsonl"
    strata = band_metrics(participant, judge_records)
    tranche_cells: dict[int, list[int]] = {value: [] for value in (1, 2, 3)}
    for row in jsonl(judge_records):
        output_value = row.get("judge_output"); source_id = str(row["source_id"])
        need(output_value is not None and source_id in sample_tranches, "ratio quality tranche linkage differs")
        tranche_cells[sample_tranches[source_id]].extend(int(output_value[axis][dimension]["score"]) for axis in AXES for dimension in JUDGE_DIMENSIONS)
    tranche_means = {str(value): sum(cells) / len(cells) for value, cells in tranche_cells.items()}
    populated_band_means = [value for value in strata["reference_band_macro_means"].values() if value is not None]
    gates = {
        "exact_q4_completed": judge_report.get("status") == "completed" and judge_report.get("counts", {}).get("valid") == SAMPLE_SIZE,
        "macro_mean_at_least_4_8": float(judge_report.get("macro_mean", 0.0)) >= 4.8,
        "worst_cell_mean_at_least_4_5": float(judge_report.get("worst_cell_mean", 0.0)) >= 4.5,
        "low_score_cell_rate_at_most_1_percent": float(judge_report.get("score_1_or_2_rate", 1.0)) <= 0.01,
        "every_populated_reference_band_mean_at_least_4_5": bool(populated_band_means) and min(populated_band_means) >= 4.5,
        "every_nested_tranche_mean_at_least_4_5": min(tranche_means.values()) >= 4.5,
    }
    result = {
        "schema_version": "mal2026-rationale-pipeline-ratio-quality-gate-v1",
        "status": "passed" if all(gates.values()) else "failed",
        "run_id": args.run_id, "completed_at": now(),
        "generation_run_id": args.generation_run_id,
        "candidate_key": args.candidate_key,
        "sample": {"records": SAMPLE_SIZE, "selection": "all_score1_then_50_per_axis_score2_or5_then_hash_fill", "variant": "hash_partitioned_end_of_1x_or_2x_added_or_3x_added_tranche", "tranche_records": dict(sorted(Counter(sample_tranches.values()).items())), "axis_score_band_counts": sample_band_counts},
        "quality": {"macro_mean": judge_report["macro_mean"], "worst_cell_mean": judge_report["worst_cell_mean"], "score_1_or_2_rate": judge_report["score_1_or_2_rate"], "nested_tranche_macro_means": tranche_means, **strata},
        "gates": gates,
        "generation_report_sha256": sha256_file(generation_report_path),
        "sample_sha256": sha256_file(sample_path), "participant_sha256": sha256_file(participant),
        "judge_report_path": str(full_report.resolve()), "judge_report_sha256": sha256_file(full_report),
        "judge_records_sha256": sha256_file(judge_records), "smoke_report_sha256": sha256_file(smoke_report),
        "rationale_prompt_sha256": routing()["rationale_generation_training_evaluation"]["source_file_sha256"],
        "judge_prompt_sha256": routing()["rationale_reward_and_quality_judge"]["source_file_sha256"],
        "canonical_scores_projected_with_decimal_round_half_up": True,
        "gpu_scope": list(GPUS), "average_used": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights",
    }
    atomic_json(output / "aggregate.json", result)
    atomic_json(restricted / "manifest.json", {**result, "aggregate_sha256": sha256_file(output / "aggregate.json")})
    print(json.dumps({"status": result["status"], "run_id": args.run_id, "macro_mean": judge_report["macro_mean"], "worst_cell_mean": judge_report["worst_cell_mean"]}, sort_keys=True), flush=True)
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
