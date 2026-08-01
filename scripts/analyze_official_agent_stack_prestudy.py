#!/usr/bin/env python3
"""Aggregate-only adaptive prestudy for score-blind official Terra scores.

This analysis was run before V7 preregistration and therefore cannot be used
as confirmatory evidence.  It deliberately reads train only, writes no row
prediction, and records the full fixed ridge/cap grid rather than only its
best cell.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.iterative_tail_metrics import compute_iterative_tail_metrics  # noqa: E402
from mal2026.iterative_tail_runner import load_experiment_data  # noqa: E402
from mal2026.official_rationale_data import candidate_provenance, load_candidates  # noqa: E402


RUN_ID = "iterative-official-agent-stack-v7-prestudy-20260802-001"
OUTPUT = ROOT / "outputs/iterative-official-agent-stack-v7-prestudy" / RUN_ID / "aggregate.json"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
CAPS = (0.1, 0.2, 0.3, 0.5, 1.0)
AXES = ("content", "organization", "expression")


def feature_matrix(base: np.ndarray, source_ids: tuple[str, ...]) -> np.ndarray:
    index = {source_id: row for row, source_id in enumerate(source_ids)}
    values = np.empty((len(source_ids), 3, 3), dtype=np.float64)
    coverage = np.zeros((len(source_ids), 3), dtype=np.int8)
    for candidate in load_candidates():
        row = index.get(candidate.source_id)
        if row is None:
            raise RuntimeError("official candidate/train population differs")
        number = candidate.candidate_number - 1
        if coverage[row, number]:
            raise RuntimeError("duplicate official candidate")
        coverage[row, number] = 1
        values[row, number] = [candidate.scores[axis] for axis in AXES]
    if not np.all(coverage == 1):
        raise RuntimeError("official candidate coverage differs")
    mean, std = values.mean(1), values.std(1)
    minimum, maximum = values.min(1), values.max(1)
    agreement = np.concatenate(
        [(values[:, left] == values[:, right]).astype(np.float64) for left, right in ((0, 1), (0, 2), (1, 2))],
        axis=1,
    )
    repeated_base = np.repeat(base[:, None, :], 3, axis=1)
    return np.concatenate(
        (values.reshape(len(values), -1), mean, std, minimum, maximum, agreement,
         (values - repeated_base).reshape(len(values), -1)), axis=1,
    )


def ridge(train_x: np.ndarray, train_y: np.ndarray, predict_x: np.ndarray, alpha: float) -> np.ndarray:
    mean, std = train_x.mean(0), train_x.std(0)
    std[std < 1e-8] = 1.0
    x, z = (train_x - mean) / std, (predict_x - mean) / std
    target_mean = train_y.mean(0)
    centered = train_y - target_mean
    weights = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ centered)
    return z @ weights + target_mean


def main() -> None:
    data = load_experiment_data()
    features = feature_matrix(data.base.astype(np.float64), data.source_ids)
    base, targets = data.base.astype(np.float64), data.targets.astype(np.float64)
    baseline = compute_iterative_tail_metrics(targets, base)
    cells = []
    for alpha in ALPHAS:
        corrections = np.zeros_like(base)
        for fold in range(5):
            train, predict = data.folds != fold, data.folds == fold
            corrections[predict] = ridge(
                features[train], targets[train] - base[train], features[predict], alpha,
            )
        for cap in CAPS:
            prediction = np.clip(base + np.clip(corrections, -cap, cap), 1.0, 5.0)
            metrics = compute_iterative_tail_metrics(targets, prediction)
            cells.append({"alpha": alpha, "cap": cap, "metrics": metrics})
    payload = {
        "schema_version": "mal2026-iterative-official-agent-stack-prestudy-v7",
        "status": "completed",
        "run_id": RUN_ID,
        "adaptive_before_v7_preregistration": True,
        "claim_role": "candidate_design_evidence_only_not_outer_confirmation",
        "records": 2000,
        "folds": 5,
        "feature_dimensions": int(features.shape[1]),
        "feature_order": ["candidate_scores_3x3", "axis_mean", "axis_std", "axis_min", "axis_max", "pairwise_equal_3x3", "candidate_minus_r0_3x3"],
        "official_candidate_provenance": candidate_provenance(),
        "baseline_metrics": baseline,
        "grid": cells,
        "grid_alphas": list(ALPHAS),
        "grid_caps": list(CAPS),
        "validation_loaded": False,
        "average_target_used": False,
        "row_predictions_persisted": False,
        "external_api_calls": 0,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps({
        "run_id": RUN_ID,
        "status": "completed",
        "cells": len(cells),
        "best_macro_rmse": min(cell["metrics"]["macro"]["rmse"] for cell in cells),
        "output_sha256": sha256(OUTPUT.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
