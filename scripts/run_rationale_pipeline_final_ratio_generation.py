#!/usr/bin/env python3
"""Generate the nested 1x/2x/3x score-blind student-rationale pool.

The per-source 1x multiplicity exactly copies the quality-filtered OpenAI
teacher pool.  The 2x and 3x arms add independently sampled student outputs
for the same per-source distribution, so their only planned difference is the
OpenAI:student view ratio.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from mal2026.api_rationale_data import load_writing_rows, sha256_file  # noqa: E402
from mal2026.rationale_pipeline_prompts import AXES, round_half_up_score, routing  # noqa: E402
from run_rationale_pipeline_sft_evaluation import (  # noqa: E402
    GENERATION_AGGREGATE, GENERATION_RESTRICTED, GEN_PORTS, GPUS,
    launch_vllm, run_generation_on_server, stop_owned, wait_released,
)


OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-final-ratio-generation-v1"
DEFAULT_REFERENCE = ROOT / "data/processed/restricted/rationale_v3_tail_sft_merged/rationale-v3-tail-sft-merged-20260807-001/sft_targets.train.quality_filtered.jsonl"
OPENAI_VIEWS = 11_342
MAX_SCALE = 3


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def reference_counts(path: Path) -> Counter[str]:
    result: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                need(isinstance(row, dict) and row.get("source_id") is not None, "ratio reference row differs")
                result[str(row["source_id"])] += 1
    need(len(result) == 2_000 and sum(result.values()) == OPENAI_VIEWS and min(result.values()) >= 1, "ratio reference population differs")
    return result


def distribution(counts: Mapping[str, int]) -> dict[str, dict[str, int]]:
    rows = {row.identifier: row for row in load_writing_rows("train", include_scores=True)}
    need(set(rows) == set(counts), "ratio distribution source coverage differs")
    result = {axis: {str(score): 0 for score in range(1, 6)} for axis in AXES}
    for source_id, multiplicity in counts.items():
        scores = rows[source_id].scores; need(scores is not None, "ratio distribution scores unavailable")
        for axis in AXES:
            result[axis][str(round_half_up_score(scores[axis]))] += multiplicity
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--training-completion", type=Path, required=True)
    parser.add_argument("--multiplicity-reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    setproctitle(f"mal2026:rationale-final-ratio-generation:{args.candidate_key}"[:255])

    need((args.base_model / "config.json").is_file() and (args.adapter / "adapter_model.safetensors").is_file(), "ratio rationale model artifact unavailable")
    completion = json.loads(args.training_completion.read_text(encoding="utf-8"))
    need(completion.get("status") == "completed" and completion.get("human_or_reference_score_read_or_prompted") is False, "ratio rationale model is not score-blind SFT")
    counts = reference_counts(args.multiplicity_reference)
    output = OUTPUT_PARENT / args.run_id
    need(not output.exists(), "ratio rationale output must be fresh")
    output.mkdir(parents=True)
    candidate = {
        "key": args.candidate_key,
        "base_model_path": str(args.base_model.resolve()),
        "adapter_path": str(args.adapter.resolve()),
        "training_completion_path": str(args.training_completion.resolve()),
    }
    processes = []
    expected = OPENAI_VIEWS * MAX_SCALE
    try:
        processes, endpoints, attestation, aliases = launch_vllm([candidate], GPUS, GEN_PORTS, output / "runtime", max_model_len=6144)
        alias = aliases[args.candidate_key]
        smoke = run_generation_on_server(args.run_id, candidate, "train", 1, endpoints, attestation, alias, output / "telemetry-smoke.jsonl", GPUS)
        generated = run_generation_on_server(
            args.run_id, candidate, "train", expected, endpoints, attestation, alias,
            output / "telemetry-ratio-pool.jsonl", GPUS,
            multiplicity_reference=args.multiplicity_reference, multiplicity_scale=MAX_SCALE,
        )
    finally:
        if processes:
            stop_owned(processes); wait_released(GPUS)

    prefix = f"{args.run_id}-{args.candidate_key}"
    smoke_report = GENERATION_AGGREGATE / f"{prefix}-train1" / "aggregate.json"
    pool_report = GENERATION_AGGREGATE / f"{prefix}-train{expected}" / "aggregate.json"
    report = json.loads(pool_report.read_text(encoding="utf-8"))
    need(report.get("status") == "completed" and report.get("counts", {}).get("valid") == expected, "ratio rationale generation report differs")
    need(report.get("reference_multiplicity") is True and report.get("multiplicity_scale") == MAX_SCALE, "ratio rationale multiplicity attestation differs")
    need(report.get("multiplicity_reference_sha256") == sha256_file(args.multiplicity_reference), "ratio rationale reference checksum differs")

    base_distribution = distribution(counts)
    payload = {
        "schema_version": "mal2026-rationale-pipeline-final-ratio-generation-v1",
        "status": "completed", "run_id": args.run_id, "completed_at": now(),
        "candidate_key": args.candidate_key,
        "openai_reference_views": OPENAI_VIEWS,
        "student_pool_records": expected,
        "source_records": 2_000,
        "nested_arms": {
            "1to1": {"openai": OPENAI_VIEWS, "student": OPENAI_VIEWS, "student_variant_rule": "variant_index < source_openai_count"},
            "1to2": {"openai": OPENAI_VIEWS, "student": 2 * OPENAI_VIEWS, "student_variant_rule": "variant_index < 2 * source_openai_count"},
            "1to3": {"openai": OPENAI_VIEWS, "student": 3 * OPENAI_VIEWS, "student_variant_rule": "variant_index < 3 * source_openai_count"},
        },
        "one_x_source_multiplicity": {"minimum": min(counts.values()), "maximum": max(counts.values()), "histogram": dict(sorted(Counter(counts.values()).items()))},
        "one_x_axis_score_band_counts": base_distribution,
        "two_x_axis_score_band_counts": {axis: {score: 2 * count for score, count in bands.items()} for axis, bands in base_distribution.items()},
        "three_x_axis_score_band_counts": {axis: {score: 3 * count for score, count in bands.items()} for axis, bands in base_distribution.items()},
        "score_conditioning": False,
        "scores_used_for_generation": False,
        "multiplicity_copied_from_quality_filtered_openai_pool": True,
        "sampling": {"temperature": 0.7, "top_p": 0.95, "seed_scheme": "sha256_source_variant"},
        "train_path": str(generated.resolve()), "train_sha256": sha256_file(generated),
        "generation_report_path": str(pool_report.resolve()), "generation_report_sha256": sha256_file(pool_report),
        "smoke_report_sha256": sha256_file(smoke_report), "smoke_records_sha256": sha256_file(smoke),
        "multiplicity_reference_path": str(args.multiplicity_reference.resolve()),
        "multiplicity_reference_sha256": sha256_file(args.multiplicity_reference),
        "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "training_completion_sha256": sha256_file(args.training_completion),
        "rationale_prompt_sha256": routing()["rationale_generation_training_evaluation"]["source_file_sha256"],
        "gpu_scope": list(GPUS), "average_used": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights",
    }
    atomic_json(output / "aggregate.json", payload)
    print(json.dumps({"status": "completed", "run_id": args.run_id, "student_pool_records": expected}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
