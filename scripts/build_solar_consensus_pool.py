#!/usr/bin/env python3
"""Build consensus/disagreement pools from Solar labels and train-only OOF scorers."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_score_matrix import AXES, file_sha256, score_metrics  # noqa: E402
from mal2026.solar_consensus_pilot import calibrated_quarter_threshold  # noqa: E402


PILOT_RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_consensus_pilot_v1"
OOF_RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_consensus_oof_v1"
OUTPUT_ROOT = ROOT / "outputs/solar-consensus-selection-v1"
RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_consensus_selection_v1"
MODELS = ("qwen3_embedding_8b", "kure_v1")
FOLDS = 5
CALIBRATION_QUANTILE = 0.8
CONTROL_FRACTION = 0.1
CONTROL_SEED = 2026073003


class ConsensusSelectionError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ConsensusSelectionError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    need(path.is_file() and not path.is_symlink(), f"input unavailable: {path.name}")
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    need(values and all(isinstance(value, dict) for value in values), f"input differs: {path.name}")
    return values


def keyed(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        need(isinstance(value, str) and value not in result, f"{key} population differs")
        result[value] = row
    return result


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    need(not path.exists(), "selection output must be fresh")
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return file_sha256(path)


def collect_predictions(
    oof_run_id: str, model: str, filename: str, key: str,
) -> tuple[dict[str, Mapping[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for fold in range(FOLDS):
        path = OOF_RESTRICTED_ROOT / oof_run_id / model / f"fold-{fold:02d}" / filename
        rows.extend(read_jsonl(path))
        paths.append(path)
    return keyed(rows, key), paths


def original_oof_metrics(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ordered = [rows[key] for key in sorted(rows)]
    labels = [[float(row["reference_score"][axis]) for axis in AXES] for row in ordered]
    continuous = [[float(row["continuous_prediction"][axis]) for axis in AXES] for row in ordered]
    integers = [[int(row["integer_prediction"][axis]) for axis in AXES] for row in ordered]
    violations = [[[False], [False], [False]] for _ in ordered]
    return score_metrics(labels, continuous, integers, violations)


def control_sample(rows: Sequence[Mapping[str, Any]], label: str) -> list[Mapping[str, Any]]:
    count = math.ceil(len(rows) * CONTROL_FRACTION)
    return sorted(
        rows,
        key=lambda row: sha256(
            f"{CONTROL_SEED}\0{label}\0{row['candidate_id']}".encode()
        ).digest(),
    )[:count]


def score_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        axis: {
            str(score): sum(int(row["score"][axis]) == score for row in rows)
            for score in range(1, 6)
        }
        for axis in AXES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pilot-run-id", required=True)
    parser.add_argument("--oof-run-id", required=True)
    args = parser.parse_args()
    output = OUTPUT_ROOT / args.run_id
    restricted = RESTRICTED_ROOT / args.run_id
    need(not output.exists() and not restricted.exists(), "selection outputs must be fresh")
    output.mkdir(mode=0o700, parents=True)
    restricted.mkdir(mode=0o700, parents=True)

    candidate_path = PILOT_RESTRICTED_ROOT / args.pilot_run_id / "stable_modal_candidates.jsonl"
    candidates = keyed(read_jsonl(candidate_path), "candidate_id")
    essay_hash_counts = Counter(
        str(candidate["candidate_essay_sha256"]) for candidate in candidates.values()
    )
    original: dict[str, dict[str, Mapping[str, Any]]] = {}
    predicted: dict[str, dict[str, Mapping[str, Any]]] = {}
    input_paths: dict[str, list[Path]] = {}
    for model in MODELS:
        original[model], original_paths = collect_predictions(
            args.oof_run_id, model, "original_oof_predictions.jsonl", "record_id"
        )
        predicted[model], candidate_paths = collect_predictions(
            args.oof_run_id, model, "candidate_predictions.jsonl", "candidate_id"
        )
        input_paths[f"{model}_original"] = original_paths
        input_paths[f"{model}_candidate"] = candidate_paths
        need(len(original[model]) == 2000, "original OOF population differs")
        need(set(predicted[model]) == set(candidates), "candidate OOF population differs")
    need(set(original[MODELS[0]]) == set(original[MODELS[1]]),
         "model OOF source populations differ")

    thresholds: dict[str, dict[str, float]] = {
        "encoder_pair": {}, "solar_vs_encoder_mean": {},
    }
    for axis in AXES:
        pair_differences = [
            abs(float(original[MODELS[0]][record_id]["continuous_prediction"][axis]) -
                float(original[MODELS[1]][record_id]["continuous_prediction"][axis]))
            for record_id in original[MODELS[0]]
        ]
        ensemble_errors = [
            abs(
                (float(original[MODELS[0]][record_id]["continuous_prediction"][axis]) +
                 float(original[MODELS[1]][record_id]["continuous_prediction"][axis])) / 2.0 -
                float(original[MODELS[0]][record_id]["reference_score"][axis])
            )
            for record_id in original[MODELS[0]]
        ]
        thresholds["encoder_pair"][axis] = calibrated_quarter_threshold(
            pair_differences, CALIBRATION_QUANTILE
        )
        thresholds["solar_vs_encoder_mean"][axis] = calibrated_quarter_threshold(
            ensemble_errors, CALIBRATION_QUANTILE
        )

    core: list[dict[str, Any]] = []
    disagreement: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for candidate_id in sorted(candidates):
        candidate = candidates[candidate_id]
        qwen = predicted[MODELS[0]][candidate_id]["continuous_prediction"]
        kure = predicted[MODELS[1]][candidate_id]["continuous_prediction"]
        solar = candidate["score"]
        reasons: list[str] = []
        if essay_hash_counts[str(candidate["candidate_essay_sha256"])] > 1:
            reasons.append("duplicate_candidate_essay")
        if candidate.get("strict_score_specific_edit_count_pass") is not True:
            reasons.append("relaxed_edit_count_only")
        per_axis: dict[str, Any] = {}
        for axis in AXES:
            q = float(qwen[axis])
            k = float(kure[axis])
            mean = (q + k) / 2.0
            pair_gap = abs(q - k)
            solar_gap = abs(float(solar[axis]) - mean)
            pair_ok = pair_gap <= thresholds["encoder_pair"][axis]
            solar_ok = solar_gap <= thresholds["solar_vs_encoder_mean"][axis]
            if not pair_ok:
                reasons.append(f"{axis}:encoder_pair")
            if not solar_ok:
                reasons.append(f"{axis}:solar_encoder")
            per_axis[axis] = {
                "qwen_oof": q,
                "kure_oof": k,
                "encoder_mean": mean,
                "encoder_pair_gap": pair_gap,
                "solar_encoder_gap": solar_gap,
                "encoder_pair_within_oof_threshold": pair_ok,
                "solar_within_oof_threshold": solar_ok,
            }
        value = dict(candidate)
        value["oof_encoder_consensus"] = per_axis
        value["selection_provenance"] = {
            "requested_target_used": False,
            "validation_used": False,
            "threshold_source": "real_train_only_five_fold_oof_80th_percentile_rounded_up_quarter",
            "reasons": reasons,
        }
        if reasons:
            disagreement.append(value)
            reason_counts.update(reasons)
        else:
            core.append(value)

    # Equalize total candidate weight per immutable source inside each pool.
    for pool in (core, disagreement):
        counts = Counter(row["source_id"] for row in pool)
        for row in pool:
            row["source_normalized_weight"] = 1.0 / counts[row["source_id"]]

    core_control = control_sample(core, "consensus-core")
    disagreement_control = control_sample(disagreement, "disagreement")
    control = [
        {**row, "control_stratum": "consensus_core"} for row in core_control
    ] + [
        {**row, "control_stratum": "disagreement"} for row in disagreement_control
    ]
    core_path = restricted / "consensus_core.jsonl"
    disagreement_path = restricted / "disagreement_pool.jsonl"
    control_path = restricted / "qwen36_random_control.jsonl"
    hashes = {
        "consensus_core": write_jsonl(core_path, core),
        "disagreement_pool": write_jsonl(disagreement_path, disagreement),
        "qwen36_random_control": write_jsonl(control_path, control),
    }
    result = {
        "schema_version": "mal2026-solar-consensus-selection-result-v1",
        "status": "completed",
        "completed_at": now(),
        "run_id": args.run_id,
        "pilot_run_id": args.pilot_run_id,
        "oof_run_id": args.oof_run_id,
        "stable_solar_candidates": len(candidates),
        "consensus_core": len(core),
        "disagreement_pool": len(disagreement),
        "consensus_rate": len(core) / len(candidates),
        "consensus_score_counts": score_counts(core),
        "disagreement_score_counts": score_counts(disagreement),
        "disagreement_reason_counts": dict(sorted(reason_counts.items())),
        "calibration": {
            "source": "canonical_real_train_only_five_fold_oof",
            "threshold_rows_include_validation": False,
            "encoder_fixed_epoch_provenance_may_reflect_prior_validation_selection": True,
            "quantile": CALIBRATION_QUANTILE,
            "rounding": "up_to_next_quarter_score_minimum_0.5",
            "thresholds": thresholds,
            "model_oof_metrics": {
                model: original_oof_metrics(original[model]) for model in MODELS
            },
        },
        "qwen36_control": {
            "fraction_per_stratum": CONTROL_FRACTION,
            "consensus_core": len(core_control),
            "disagreement": len(disagreement_control),
            "selection_uses_scores": False,
            "seed": CONTROL_SEED,
        },
        "artifact_sha256": hashes,
        "input_sha256": {
            "stable_solar_candidates": file_sha256(candidate_path),
            **{
                name: [file_sha256(path) for path in paths]
                for name, paths in input_paths.items()
            },
        },
        "protocol": {
            "validation_rows_used_for_calibration_or_selection": False,
            "encoder_fixed_epoch_provenance_may_reflect_prior_validation_selection": True,
            "candidate_solar_labels_used_to_train_oof_encoders": False,
            "requested_target_used_for_consensus": False,
            "actual_solar_modal_triplet_is_pseudo_label": True,
            "encoder_agreement_is_filter_confidence_not_ground_truth": True,
            "disagreement_rows_preserved": True,
            "duplicate_candidate_essays_excluded_from_core": True,
            "hard_rejections_preserved_in_pilot_artifact": True,
            "full_training_authorized_by_this_result": False,
        },
        "bindings": {
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "worktree_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()),
            "script_sha256": file_sha256(Path(__file__).resolve()),
        },
        "privacy": "aggregate contains no essay, prompt, rationale, identifier, or individual prediction",
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
