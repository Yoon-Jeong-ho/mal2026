#!/usr/bin/env python3
"""Admit score-blind student views for encoder training without hiding Q4 failure.

The strict rationale-quality gate includes score/rationale consistency against
the hidden gold label.  Filtering deployment score-blind rationales on that
label would leak the target into encoder-view selection.  This admission is
therefore limited to the encoder experiment and requires the three genuinely
score-blind quality dimensions to pass.  It never promotes the rationale model
as having passed the original strict gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import sha256_file  # noqa: E402
from mal2026.official_writing_contract import JUDGE_DIMENSIONS  # noqa: E402
from mal2026.rationale_pipeline_prompts import AXES, routing  # noqa: E402


QUALITY_PARENT = ROOT / "outputs/rationale-pipeline-ratio-quality-gate-v1"
Q4_RESTRICTED = ROOT / "data/processed/restricted/official_rationale_fidelity_v1/q4_judge"
OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-ratio-encoder-admission-v1"
NON_CONSISTENCY = ("groundedness", "specificity", "domain_match")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--quality-run-id", required=True)
    parser.add_argument("--user-authorization", required=True)
    args = parser.parse_args()
    setproctitle(f"mal2026:rationale-ratio-encoder-admission:{args.run_id}"[:255])
    need(bool(args.user_authorization.strip()), "ratio encoder admission lacks user authorization")
    quality_path = QUALITY_PARENT / args.quality_run_id / "aggregate.json"
    records_path = Q4_RESTRICTED / args.quality_run_id / "judge_records.jsonl"
    need(quality_path.is_file() and records_path.is_file(), "ratio quality evidence unavailable")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    need(quality.get("status") == "failed" and quality.get("gates", {}).get("exact_q4_completed") is True, "ratio strict quality result differs")
    failed = {key for key, passed in quality["gates"].items() if passed is not True}
    need(failed == {"low_score_cell_rate_at_most_1_percent"}, "ratio strict quality failure is broader than score-cell rate")

    cells: dict[str, list[int]] = {dimension: [] for dimension in JUDGE_DIMENSIONS}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line); output = row.get("judge_output")
        need(output is not None, "ratio encoder admission judge row invalid")
        for axis in AXES:
            for dimension in JUDGE_DIMENSIONS:
                cells[dimension].append(int(output[axis][dimension]["score"]))
    need(all(len(values) == 1_200 for values in cells.values()), "ratio encoder admission judge cell count differs")
    dimension_means = {dimension: statistics.fmean(values) for dimension, values in cells.items()}
    low_counts = {dimension: sum(value <= 2 for value in values) for dimension, values in cells.items()}
    blind_cells = [value for dimension in NON_CONSISTENCY for value in cells[dimension]]
    blind_low_rate = sum(value <= 2 for value in blind_cells) / len(blind_cells)
    gates = {
        "strict_gate_failure_only_hidden_score_consistency_rate": failed == {"low_score_cell_rate_at_most_1_percent"},
        "groundedness_mean_at_least_4_8": dimension_means["groundedness"] >= 4.8,
        "specificity_mean_at_least_4_8": dimension_means["specificity"] >= 4.8,
        "domain_match_mean_at_least_4_8": dimension_means["domain_match"] >= 4.8,
        "non_score_consistency_low_cell_rate_at_most_1_percent": blind_low_rate <= 0.01,
    }
    need(all(gates.values()), "score-blind encoder view admission gates failed")
    output = OUTPUT_PARENT / args.run_id; need(not output.exists(), "ratio encoder admission output must be fresh"); output.mkdir(parents=True)
    result = {
        "schema_version": "mal2026-rationale-pipeline-ratio-encoder-admission-v1",
        "status": "admitted_for_deployment_view_encoder_training_not_rationale_quality_promotion",
        "run_id": args.run_id, "completed_at": now(), "quality_run_id": args.quality_run_id,
        "strict_rationale_quality_status": quality["status"], "strict_failed_gates": sorted(failed),
        "decision": "retain_unfiltered_score_blind_student_outputs_for_encoder_training_to_avoid_gold_label_conditioned_view_selection",
        "scope": "encoder_training_views_only",
        "user_authorization": args.user_authorization,
        "dimension_means": dimension_means, "low_cell_counts": low_counts,
        "non_score_consistency_low_cell_rate": blind_low_rate,
        "gates": gates,
        "quality_report_sha256": sha256_file(quality_path), "judge_records_sha256": sha256_file(records_path),
        "rationale_prompt_sha256": routing()["rationale_generation_training_evaluation"]["source_file_sha256"],
        "judge_prompt_sha256": routing()["rationale_reward_and_quality_judge"]["source_file_sha256"],
        "average_used": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights",
    }
    atomic_json(output / "aggregate.json", result)
    print(json.dumps({"status": result["status"], "run_id": args.run_id, "non_score_consistency_low_cell_rate": blind_low_rate}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
