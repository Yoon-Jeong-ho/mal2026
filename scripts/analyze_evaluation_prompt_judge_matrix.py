#!/usr/bin/env python3
"""Create an aggregate-only paired analysis of exact-judge matrix records."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import sha256_file  # noqa: E402
from mal2026.official_writing_contract import AXES, JUDGE_DIMENSIONS, parse_judge_output  # noqa: E402


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    need(bool(separator) and bool(name.strip()) and bool(raw_path.strip()), "arm must be NAME=PATH")
    return name.strip(), Path(raw_path).resolve()


def load_arm(path: Path, expected: int) -> tuple[dict[str, float], dict[str, Any]]:
    restricted = (ROOT / "data/processed/restricted/official_prompt_alignment_v1/q4_judge").resolve()
    need(path.is_file() and not path.is_symlink() and path.is_relative_to(restricted), "judge records must be restricted")
    record_means: dict[str, float] = {}
    counts: Counter[int] = Counter()
    cells: dict[str, dict[str, list[int]]] = {
        axis: {dimension: [] for dimension in JUDGE_DIMENSIONS} for axis in AXES
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            need(isinstance(raw, dict) and isinstance(raw.get("source_id"), str), "judge row schema differs")
            source_id = raw["source_id"]
            need(source_id not in record_means and raw.get("failure_category") is None, "judge row failed or duplicated")
            parsed = parse_judge_output(raw.get("judge_output"))
            values: list[int] = []
            for axis in AXES:
                for dimension in JUDGE_DIMENSIONS:
                    score = int(parsed[axis][dimension]["score"])
                    values.append(score)
                    counts[score] += 1
                    cells[axis][dimension].append(score)
            record_means[source_id] = statistics.fmean(values)
    need(len(record_means) == expected, "judge population differs")
    total = sum(counts.values())
    need(total == expected * len(AXES) * len(JUDGE_DIMENSIONS), "judge cell population differs")
    cell_means = {
        axis: {dimension: statistics.fmean(cells[axis][dimension]) for dimension in JUDGE_DIMENSIONS}
        for axis in AXES
    }
    flat = [cell_means[axis][dimension] for axis in AXES for dimension in JUDGE_DIMENSIONS]
    aggregate = {
        "records": expected,
        "judge_cells": total,
        "macro_mean": statistics.fmean(flat),
        "worst_cell_mean": min(flat),
        "cell_score_counts": {str(score): counts[score] for score in range(1, 6)},
        "cell_score_rates": {str(score): counts[score] / total for score in range(1, 6)},
        "score_1_or_2_rate": (counts[1] + counts[2]) / total,
        "cell_means": cell_means,
        "judge_records_sha256": sha256_file(path),
    }
    return record_means, aggregate


def paired_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    need(set(left) == set(right), "paired judge populations differ")
    identifiers = sorted(left)
    differences = [left[key] - right[key] for key in identifiers]
    rng = random.Random(seed)
    samples = [
        statistics.fmean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(replicates)
    ]
    samples.sort()
    lower = samples[int(0.025 * replicates)]
    upper = samples[max(0, int(0.975 * replicates) - 1)]
    return {
        "records": len(differences),
        "mean_delta": statistics.fmean(differences),
        "bootstrap_95_ci": [lower, upper],
        "bootstrap_replicates": replicates,
        "seed": seed,
        "nonzero_record_differences": sum(value != 0 for value in differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True, help="NAME=restricted judge_records.jsonl")
    parser.add_argument("--compare", action="append", default=[], help="LEFT,RIGHT; reports LEFT minus RIGHT")
    parser.add_argument("--expected", type=int, default=400)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(args.expected > 0 and args.bootstrap_replicates >= 1000, "analysis bounds differ")
    output = args.output.resolve()
    need(output.is_relative_to((ROOT / "outputs").resolve()) and not output.exists(), "aggregate output must be fresh under outputs")

    raw_arms = [parse_named_path(value) for value in args.arm]
    names = [name for name, _ in raw_arms]
    need(len(names) == len(set(names)), "arm names must be unique")
    paired_values: dict[str, dict[str, float]] = {}
    arm_reports: dict[str, Any] = {}
    for name, path in raw_arms:
        paired_values[name], arm_reports[name] = load_arm(path, args.expected)

    comparisons: dict[str, Any] = {}
    for value in args.compare:
        left, separator, right = value.partition(",")
        left, right = left.strip(), right.strip()
        need(bool(separator) and left in paired_values and right in paired_values, "comparison arm differs")
        key = f"{left}_minus_{right}"
        need(key not in comparisons, "comparison duplicated")
        comparisons[key] = paired_bootstrap(
            paired_values[left], paired_values[right], seed=args.seed, replicates=args.bootstrap_replicates
        )

    payload = {
        "schema_version": "mal2026-evaluation-prompt-exact-judge-matrix-analysis-v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "expected_records_per_arm": args.expected,
        "arms": arm_reports,
        "paired_comparisons": comparisons,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_predictions_or_evidence",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "arms": len(arm_reports), "comparisons": len(comparisons)}, sort_keys=True))


if __name__ == "__main__":
    main()
