"""Aggregate-only metrics and promotion gates for iterative three-axis scoring.

Equal-group RMSE gives each true-gold group ``{1,2}``, ``{3}``, ``{4}``, and
``{5}`` equal weight: RMSE is computed within each non-empty group and axis,
then averaged over groups within an axis and finally over the three axes.  It
therefore does not let the common middle bands dominate the tail bands.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np


AXES = ("content", "organization", "expression")
GROUPS = ((1, 2), (3,), (4,), (5,))
GROUP_NAMES = ("1_2", "3", "4", "5")


class IterativeMetricError(ValueError):
    """Raised when aggregate metric inputs violate the three-axis contract."""


def half_up_band(value: float) -> int:
    """Clamp a finite score to 1--5 and round ties upward."""
    number = float(value)
    if not np.isfinite(number):
        raise IterativeMetricError("scores must be finite")
    number = min(5.0, max(1.0, number))
    return int(Decimal(str(number)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _matrix(values: Sequence[Sequence[float]] | Mapping[str, Sequence[float]], name: str) -> np.ndarray:
    if isinstance(values, Mapping):
        if set(values) != set(AXES):
            raise IterativeMetricError(f"{name} must contain exactly the three axes; average is forbidden")
        columns = [np.asarray(values[axis], dtype=float) for axis in AXES]
        if any(column.ndim != 1 for column in columns):
            raise IterativeMetricError(f"{name} axis values must be one-dimensional")
        if len({len(column) for column in columns}) != 1:
            raise IterativeMetricError(f"{name} axes must have equal lengths")
        result = np.column_stack(columns)
    else:
        result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != len(AXES) or result.shape[0] == 0:
        raise IterativeMetricError(f"{name} must be a nonempty N x 3 matrix")
    if not np.all(np.isfinite(result)):
        raise IterativeMetricError(f"{name} scores must be finite")
    return result


def _bands(values: np.ndarray) -> np.ndarray:
    return np.asarray([[half_up_band(value) for value in row] for row in values], dtype=int)


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2:
        return None
    x, y = _rank(left), _rank(right)
    denominator = float(np.sqrt(np.sum(np.square(x - x.mean())) * np.sum(np.square(y - y.mean()))))
    return None if denominator == 0.0 else float(np.sum((x - x.mean()) * (y - y.mean())) / denominator)


def _mean_present(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _qwk(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    observed = np.zeros((5, 5), dtype=float)
    np.add.at(observed, (truth - 1, prediction - 1), 1.0)
    total = float(observed.sum())
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    indices = np.arange(5, dtype=float)
    weights = np.square(indices[:, None] - indices[None, :]) / 16.0
    weighted_expected = float(np.sum(weights * expected))
    weighted_observed = float(np.sum(weights * observed))
    if weighted_expected == 0.0:
        return 1.0 if weighted_observed == 0.0 else None
    return 1.0 - weighted_observed / weighted_expected


def _subset_rmse(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float | None:
    return _rmse(truth[mask], prediction[mask]) if np.any(mask) else None


def compute_iterative_tail_metrics(
    targets: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    predictions: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Compute deterministic, aggregate-only metrics for the three score axes."""
    truth = _matrix(targets, "targets")
    pred = _matrix(predictions, "predictions")
    if truth.shape != pred.shape:
        raise IterativeMetricError("targets and predictions must have identical shapes")
    gold_band, pred_band = _bands(truth), _bands(pred)
    by_axis: dict[str, Any] = {}
    for column, axis in enumerate(AXES):
        actual, estimate = truth[:, column], pred[:, column]
        actual_band, estimate_band = gold_band[:, column], pred_band[:, column]
        bands: dict[str, Any] = {}
        for band in range(1, 6):
            mask = actual_band == band
            count = int(mask.sum())
            bands[str(band)] = {
                "count": count,
                "rmse": _subset_rmse(actual, estimate, mask),
                "recall": float(np.mean(estimate_band[mask] == band)) if count else None,
                "one_off": float(np.mean(np.abs(estimate_band[mask] - band) <= 1)) if count else None,
            }
        group_rmse = {
            name: _subset_rmse(actual, estimate, np.isin(actual_band, group))
            for name, group in zip(GROUP_NAMES, GROUPS, strict=True)
        }
        mask_34 = np.isin(actual_band, (3, 4))
        recalls_34 = [
            float(np.mean(estimate_band[actual_band == band] == band)) if np.any(actual_band == band) else None
            for band in (3, 4)
        ]
        by_axis[axis] = {
            "rmse": _rmse(actual, estimate),
            "spearman": _spearman(actual, estimate),
            "qwk": _qwk(actual_band, estimate_band),
            "bands": bands,
            "low_tail_rmse": _subset_rmse(actual, estimate, np.isin(actual_band, (1, 2))),
            "high_tail_rmse": _subset_rmse(actual, estimate, actual_band == 5),
            "score1_descriptive_rmse": bands["1"]["rmse"],
            "equal_group_rmse": _mean_present(list(group_rmse.values())),
            "equal_group_components": group_rmse,
            "gold_3_4_balanced_accuracy": _mean_present(recalls_34) if np.any(mask_34) else None,
            "rate_3_to_4": float(np.mean(estimate_band[actual_band == 3] == 4)) if np.any(actual_band == 3) else None,
            "rate_4_to_3": float(np.mean(estimate_band[actual_band == 4] == 3)) if np.any(actual_band == 4) else None,
        }
    macro_keys = (
        "rmse", "spearman", "qwk", "low_tail_rmse", "high_tail_rmse",
        "score1_descriptive_rmse", "equal_group_rmse", "gold_3_4_balanced_accuracy",
        "rate_3_to_4", "rate_4_to_3",
    )
    macro = {key: _mean_present([by_axis[axis][key] for axis in AXES]) for key in macro_keys}
    return {
        "record_count": int(truth.shape[0]),
        "axes": by_axis,
        "macro": macro,
        "banding": "clamp_1_5_then_decimal_round_half_up",
        "equal_group_aggregation": "mean RMSE across nonempty true-gold groups {1,2}, {3}, {4}, {5} within each axis, then mean across axes",
        "score1_role": "standalone_descriptive_not_a_promotion_gate",
    }


def metric_improvements(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return deltas normalized so positive always means candidate improvement."""
    base_macro, cand_macro = baseline["macro"], candidate["macro"]
    lower_better = ("rmse", "equal_group_rmse", "low_tail_rmse", "high_tail_rmse")
    result: dict[str, Any] = {
        key: None if base_macro[key] is None or cand_macro[key] is None else float(base_macro[key] - cand_macro[key])
        for key in lower_better
    }
    for key in ("gold_3_4_balanced_accuracy", "spearman"):
        result[key] = None if base_macro[key] is None or cand_macro[key] is None else float(cand_macro[key] - base_macro[key])
    result["axis_rmse"] = {
        axis: float(baseline["axes"][axis]["rmse"] - candidate["axes"][axis]["rmse"]) for axis in AXES
    }
    return result


def promotion_decision(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen point-estimate promotion gates; score-1 is descriptive only."""
    delta = metric_improvements(baseline, candidate)
    gates = {
        "macro_rmse_improvement_at_least_0_005": delta["rmse"] is not None and delta["rmse"] >= 0.005,
        "equal_group_rmse_improvement_at_least_0_010": delta["equal_group_rmse"] is not None and delta["equal_group_rmse"] >= 0.010,
        "low_tail_improves": delta["low_tail_rmse"] is not None and delta["low_tail_rmse"] > 0.0,
        "high_tail_improves": delta["high_tail_rmse"] is not None and delta["high_tail_rmse"] > 0.0,
        "gold_3_4_balanced_accuracy_improvement_at_least_0_01": delta["gold_3_4_balanced_accuracy"] is not None and delta["gold_3_4_balanced_accuracy"] >= 0.01,
        "no_axis_rmse_worsens_more_than_0_01": all(value >= -0.01 for value in delta["axis_rmse"].values()),
        "macro_spearman_fall_at_most_0_005": delta["spearman"] is not None and delta["spearman"] >= -0.005,
    }
    return {"promote": all(gates.values()), "gates": gates, "improvements": delta, "score1_used_for_promotion": False}


def paired_bootstrap_delta_ci(
    targets: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    baseline_predictions: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    candidate_predictions: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    *,
    document_ids: Sequence[str | int] | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260801,
) -> dict[str, Any]:
    """Paired percentile bootstrap of promotion deltas by row or document cluster."""
    truth = _matrix(targets, "targets")
    baseline = _matrix(baseline_predictions, "baseline_predictions")
    candidate = _matrix(candidate_predictions, "candidate_predictions")
    if truth.shape != baseline.shape or truth.shape != candidate.shape:
        raise IterativeMetricError("all bootstrap score matrices must have identical shapes")
    if n_resamples < 1 or not 0.0 < confidence < 1.0:
        raise IterativeMetricError("n_resamples must be positive and confidence must be between zero and one")
    if document_ids is None:
        clusters = [np.asarray([index], dtype=int) for index in range(len(truth))]
        unit = "row"
    else:
        if len(document_ids) != len(truth):
            raise IterativeMetricError("document_ids length must match score rows")
        ordered_ids = list(dict.fromkeys(document_ids))
        clusters = [np.flatnonzero(np.asarray(document_ids, dtype=object) == value) for value in ordered_ids]
        unit = "document"
    point = metric_improvements(compute_iterative_tail_metrics(truth, baseline), compute_iterative_tail_metrics(truth, candidate))
    keys = ("rmse", "equal_group_rmse", "low_tail_rmse", "high_tail_rmse", "gold_3_4_balanced_accuracy", "spearman")
    draws: dict[str, list[float]] = {key: [] for key in keys}
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[index] for index in chosen])
        delta = metric_improvements(
            compute_iterative_tail_metrics(truth[indices], baseline[indices]),
            compute_iterative_tail_metrics(truth[indices], candidate[indices]),
        )
        for key in keys:
            if delta[key] is not None:
                draws[key].append(float(delta[key]))
    alpha = (1.0 - confidence) / 2.0
    intervals = {}
    for key in keys:
        values = draws[key]
        intervals[key] = {
            "estimate": point[key],
            "lower": float(np.quantile(values, alpha)) if values else None,
            "upper": float(np.quantile(values, 1.0 - alpha)) if values else None,
            "valid_resamples": len(values),
        }
    return {
        "unit": unit,
        "cluster_count": len(clusters),
        "n_resamples": n_resamples,
        "confidence": confidence,
        "seed": seed,
        "delta_direction": "positive_means_candidate_improvement",
        "intervals": intervals,
    }


# Compact alias for callers that already name the metric family in context.
compute_metrics = compute_iterative_tail_metrics
