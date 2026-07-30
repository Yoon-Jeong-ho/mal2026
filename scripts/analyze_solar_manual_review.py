#!/usr/bin/env python3
"""Reconcile three independent blind agent reviews without publishing text.

The input files live under the ignored restricted tree.  The output contains
only aggregate agreement, score-error, and hard-fail counts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_axis_actual_label_v1"
OUTPUT_ROOT = ROOT / "outputs/solar-axis-actual-label-v1"
AXES = ("content", "organization", "expression")


class ReviewAnalysisError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewAnalysisError(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    need(rows, f"empty input: {path.name}")
    return rows


def keyed(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Mapping[str, Any]]:
    values: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        need(isinstance(value, str) and value not in values, f"invalid/duplicate {key}")
        values[value] = row
    return values


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def score_metrics(predicted: Sequence[int], reference: Sequence[int]) -> dict[str, Any]:
    need(len(predicted) == len(reference) and predicted, "score metric population differs")
    differences = [prediction - target for prediction, target in zip(predicted, reference)]
    mean_predicted = sum(predicted) / len(predicted)
    mean_reference = sum(reference) / len(reference)
    numerator = sum(
        (prediction - mean_predicted) * (target - mean_reference)
        for prediction, target in zip(predicted, reference)
    )
    denominator = math.sqrt(
        sum((prediction - mean_predicted) ** 2 for prediction in predicted) *
        sum((target - mean_reference) ** 2 for target in reference)
    )
    return {
        "n": len(predicted),
        "rmse": round(math.sqrt(sum(value * value for value in differences) / len(differences)), 6),
        "mae": round(sum(abs(value) for value in differences) / len(differences), 6),
        "mean_signed_error": round(sum(differences) / len(differences), 6),
        "exact": sum(value == 0 for value in differences),
        "within_one": sum(abs(value) <= 1 for value in differences),
        "pearson": round(numerator / denominator, 6) if denominator else None,
    }


def hard_fail_a(row: Mapping[str, Any]) -> bool:
    value = row["blind_assessment"]
    return any(value[key] == "issue" for key in (
        "grounding", "unverifiable_external_facts", "stance_reversal_or_inconsistency",
        "overedit", "new_duplication", "artifact_or_privacy",
    ))


def hard_fail_b(row: Mapping[str, Any]) -> bool:
    return (
        row["blind_grounding"] == "fail" or
        row["blind_unverifiable_external_facts"] == "present" or
        row["blind_stance_reversal"] == "present" or
        row["blind_overedit"] == "present" or
        row["blind_new_duplication"] == "present" or
        row["blind_artifact_privacy"] == "fail"
    )


def hard_fail_f(row: Mapping[str, Any]) -> bool:
    return (
        row["source_grounding"] == "fail" or
        row["external_fact"] == "unverifiable" or
        row["overedit"] == "overedit" or
        row["duplicate"] == "new" or
        row["artifact_privacy"] == "fail"
    )


def categorical_agreement(values: Sequence[Sequence[str]]) -> dict[str, Any]:
    need(values and all(len(row) == len(values[0]) for row in values),
         "categorical review population differs")
    all_equal = sum(len(set(items)) == 1 for items in zip(*values))
    return {"n": len(values[0]), "all_three_exact": all_equal}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    restricted = RESTRICTED_ROOT / args.run_id
    output = OUTPUT_ROOT / args.run_id
    paths = {
        "mapping": restricted / "manual_review_mapping_80.jsonl",
        "valid_candidates": restricted / "valid_candidates.jsonl",
        "review_feasibility": restricted / "review_feasibility_agent_80.jsonl",
        "review_a": restricted / "review_verifier_a_80.jsonl",
        "review_b": restricted / "review_verifier_b_80.jsonl",
    }
    need(all(path.is_file() for path in paths.values()), "manual review input is missing")
    mapping = keyed(read_jsonl(paths["mapping"]), "review_id")
    valid = keyed(read_jsonl(paths["valid_candidates"]), "candidate_id")
    reviewers = {
        "feasibility": keyed(read_jsonl(paths["review_feasibility"]), "review_id"),
        "a": keyed(read_jsonl(paths["review_a"]), "review_id"),
        "b": keyed(read_jsonl(paths["review_b"]), "review_id"),
    }
    review_ids = sorted(mapping)
    need(len(review_ids) == 80 and all(set(rows) == set(review_ids) for rows in reviewers.values()),
         "manual review ID population differs")

    solar_target: list[int] = []
    reviewer_scores: dict[str, list[int]] = {name: [] for name in reviewers}
    hard_fail_votes: list[int] = []
    hard_fail_votes_by_stratum: dict[str, list[int]] = {}
    overedit_values = [[], [], []]
    duplicate_values = [[], [], []]
    adherence_values = [[], [], []]
    axis_counts: Counter[str] = Counter()
    for review_id in review_ids:
        candidate_id = mapping[review_id]["candidate_id"]
        need(candidate_id in valid, "review candidate is absent from valid candidates")
        candidate = valid[candidate_id]
        axis = candidate["requested_target_axis"]
        need(axis in AXES, "review target axis differs")
        axis_counts[axis] += 1
        solar_target.append(int(candidate["score"][axis]))
        f = reviewers["feasibility"][review_id]
        a = reviewers["a"][review_id]
        b = reviewers["b"][review_id]
        reviewer_scores["feasibility"].append(int(f["scores"][axis]))
        reviewer_scores["a"].append(
            int(a["adherence_assessment"]["observed_requested_axis_score_1_to_5"])
        )
        reviewer_scores["b"].append(int(b["observed_target_axis_score_1to5"]))
        vote_count = sum((hard_fail_f(f), hard_fail_a(a), hard_fail_b(b)))
        hard_fail_votes.append(vote_count)
        stratum = mapping[review_id].get("stratum")
        need(isinstance(stratum, str), "manual review stratum differs")
        hard_fail_votes_by_stratum.setdefault(stratum, []).append(vote_count)
        overedit_values[0].append("issue" if f["overedit"] == "overedit" else "pass")
        overedit_values[1].append(a["blind_assessment"]["overedit"])
        overedit_values[2].append(
            {"present": "issue", "absent": "pass", "uncertain": "uncertain"}[
                b["blind_overedit"]
            ]
        )
        duplicate_values[0].append("issue" if f["duplicate"] == "new" else "pass")
        duplicate_values[1].append(a["blind_assessment"]["new_duplication"])
        duplicate_values[2].append(
            {"present": "issue", "absent": "pass"}[b["blind_new_duplication"]]
        )
        adherence_values[0].append(f["instruction_adherence"])
        adherence_values[1].append(a["adherence_assessment"]["overall_adherence"])
        adherence_values[2].append(b["adherence_overall"])

    pairwise: dict[str, Any] = {}
    names = list(reviewer_scores)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            pairwise[f"{left}_vs_{right}"] = score_metrics(
                reviewer_scores[left], reviewer_scores[right]
            )
    result = {
        "schema_version": "mal2026-solar-manual-review-reconciliation-v3",
        "run_id": args.run_id,
        "reviewed": len(review_ids),
        "requested_axis_counts": dict(sorted(axis_counts.items())),
        "target_axis_score_comparison_against_single_draw_solar": {
            name: score_metrics(scores, solar_target)
            for name, scores in reviewer_scores.items()
        },
        "target_axis_score_reviewer_pairwise": pairwise,
        "hard_fail_vote_counts": {
            str(votes): hard_fail_votes.count(votes) for votes in range(4)
        },
        "hard_fail_majority_two_or_more": sum(votes >= 2 for votes in hard_fail_votes),
        "hard_fail_unanimous": sum(votes == 3 for votes in hard_fail_votes),
        "hard_fail_by_sampling_stratum": {
            stratum: {
                "n": len(votes),
                "vote_counts": {str(value): votes.count(value) for value in range(4)},
                "majority_two_or_more": sum(value >= 2 for value in votes),
                "unanimous": sum(value == 3 for value in votes),
            }
            for stratum, votes in sorted(hard_fail_votes_by_stratum.items())
        },
        "all_three_exact_categorical_agreement": {
            "overedit": categorical_agreement(overedit_values),
            "new_duplication": categorical_agreement(duplicate_values),
            "overall_instruction_adherence": categorical_agreement(adherence_values),
        },
        "input_sha256": {name: digest(path) for name, path in paths.items()},
        "protocol": {
            "reviewers_blind_to_solar_scores": True,
            "requested_metadata_opened_only_after_blind_reviews_fixed": True,
            "solar_comparison_performed_only_by_posthoc_aggregate_script": True,
            "validation_used": False,
        },
        "privacy": "aggregate contains no essay, prompt, rationale, identifier, or individual row",
    }
    target = output / "manual_review_reconciliation_v3.json"
    need(not target.exists(), "review reconciliation output must be fresh")
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
