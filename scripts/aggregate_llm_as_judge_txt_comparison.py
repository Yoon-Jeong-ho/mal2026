#!/usr/bin/env python3
"""Aggregate the exact llm_as_judge.txt rerun without exposing restricted rows."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import statistics
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
RESTRICTED = ROOT / "data/processed/restricted/official_prompt_alignment_v1/q4_judge"
AGGREGATES = ROOT / "outputs/official-prompt-alignment-v1/q4-judge"
OUTPUT = (
    ROOT
    / "outputs/official-prompt-alignment-v1/llm-as-judge-txt-comparison"
    / "llm-as-judge-txt-comparison-20260728-001"
    / "aggregate_prompt_comparison.json"
)
PROMPT = ROOT / "llm_as_judge.txt"
AXES = ("content", "organization", "expression")
DIMENSIONS = ("domain_match", "score_rationale_consistency", "specificity", "groundedness")
RUNS = {
    "bundle": {
        "old": "official-q4-judge-ax4-bundle-validation-001",
        "new": "llm-as-judge-txt-q4-bundle-validation400-20260728-001",
    },
    "axis_triplet": {
        "old": "official-q4-judge-ax4-axis-triplet-validation-001",
        "new": "llm-as-judge-txt-q4-axis-triplet-validation400-20260728-001",
    },
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(run_id: str) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any]]:
    record_path = RESTRICTED / run_id / "judge_records.jsonl"
    report_path = AGGREGATES / run_id / "aggregate_judge_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records: dict[str, Mapping[str, Any]] = {}
    for line in record_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source_id = str(row["source_id"])
        if source_id in records:
            raise RuntimeError("duplicate source ID")
        if row.get("judge_output") is not None:
            records[source_id] = row["judge_output"]
    if len(records) != report["counts"]["valid"]:
        raise RuntimeError("valid record count differs from aggregate")
    return records, report


def values(output: Mapping[str, Any]) -> list[int]:
    return [int(output[axis][dimension]["score"]) for axis in AXES for dimension in DIMENSIONS]


def distribution(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    all_scores = [score for output in records.values() for score in values(output)]
    histogram = Counter(all_scores)
    perfect = sum(all(score == 5 for score in values(output)) for output in records.values())
    return {
        "valid_essays": len(records),
        "judge_cells": len(all_scores),
        "score_histogram": {str(score): histogram[score] for score in range(1, 6)},
        "score_5_rate": histogram[5] / len(all_scores),
        "perfect_12_of_12_essays": perfect,
        "perfect_12_of_12_rate": perfect / len(records),
    }


def paired(left: Mapping[str, Mapping[str, Any]], right: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    differences = [statistics.fmean(values(left[key])) - statistics.fmean(values(right[key])) for key in common]
    return {
        "paired_essays": len(common),
        "left_higher": sum(value > 0 for value in differences),
        "right_higher": sum(value < 0 for value in differences),
        "tie": sum(value == 0 for value in differences),
        "left_minus_right_macro_mean": statistics.fmean(differences),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("refusing to overwrite prompt comparison")
    loaded: dict[str, dict[str, tuple[dict[str, Mapping[str, Any]], Mapping[str, Any]]]] = {}
    for method, versions in RUNS.items():
        loaded[method] = {version: load_records(run_id) for version, run_id in versions.items()}
    prompt_sha = file_sha256(PROMPT)
    methods: dict[str, Any] = {}
    for method, versions in loaded.items():
        old_records, old_report = versions["old"]
        new_records, new_report = versions["new"]
        if new_report.get("judge_system_prompt_sha256") != prompt_sha:
            raise RuntimeError("exact-file prompt hash differs")
        if old_report["participant_sha256"] != new_report["participant_sha256"]:
            raise RuntimeError("participant population differs across prompts")
        methods[method] = {
            "old_prompt": {
                "macro_mean": old_report["macro_mean"],
                "worst_cell_mean": old_report["worst_cell_mean"],
                "status": old_report["status"],
                "distribution": distribution(old_records),
            },
            "llm_as_judge_txt": {
                "macro_mean": new_report["macro_mean"],
                "worst_cell_mean": new_report["worst_cell_mean"],
                "status": new_report["status"],
                "failure_categories": new_report["failure_categories"],
                "distribution": distribution(new_records),
            },
            "paired_new_minus_old": paired(new_records, old_records),
        }
    bundle_new = loaded["bundle"]["new"][0]
    axis_new = loaded["axis_triplet"]["new"][0]
    payload = {
        "schema_version": "mal2026-llm-as-judge-txt-prompt-comparison-v1",
        "status": "completed_with_preserved_bundle_schema_failure",
        "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "llm_as_judge_txt_sha256": prompt_sha,
        "old_frozen_proxy_prompt_sha256": "1a93a3a4c18d34318d6926871fa0a527bbaf422fe78dac8c4efb66345b222e34",
        "exact_prompt_equal": False,
        "same_participants_within_each_method": True,
        "candidate_score_kind": "actual_emitted_integer_prediction",
        "human_or_reference_score_read_or_prompted": False,
        "methods": methods,
        "new_prompt_axis_triplet_minus_bundle": paired(axis_new, bundle_new),
        "privacy": "aggregate_only_no_rows_ids_prompts_essays_rationales_evidence_or_predictions",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(OUTPUT), "sha256": file_sha256(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
