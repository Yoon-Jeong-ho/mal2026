"""Dependency-light, aggregate-only regression and operational metrics."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_HALF_UP
from math import sqrt
from statistics import fmean
from typing import Any

from .constants import SCORE_FIELDS, SCORE_MAX, SCORE_MIN
from .data_contract import ScoreVector


class MetricError(ValueError):
    """Metric inputs are incomplete, non-finite, or could expose raw content."""


def _finite_number(value: Any) -> float:
    parsed = float(value)
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise MetricError("metric values must be finite")
    return parsed


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise MetricError("metrics need at least one value")
    return fmean(values)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    xm, ym = _mean(x), _mean(y)
    numerator = sum((a - xm) * (b - ym) for a, b in zip(x, y, strict=True))
    denom_x = sum((a - xm) ** 2 for a in x)
    denom_y = sum((b - ym) ** 2 for b in y)
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / sqrt(denom_x * denom_y)


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original, _ in ordered[index:end]:
            ranks[original] = rank
        index = end
    return ranks


def _round_half_up_bin(value: float) -> int:
    clamped = min(SCORE_MAX, max(SCORE_MIN, value))
    return int(Decimal(str(clamped)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def quadratic_weighted_kappa(y_true: Sequence[float], y_pred: Sequence[float]) -> float | None:
    if len(y_true) != len(y_pred) or not y_true:
        raise MetricError("QWK requires equally sized, nonempty inputs")
    classes = 5
    observed = [[0.0 for _ in range(classes)] for _ in range(classes)]
    for actual, predicted in zip(y_true, y_pred, strict=True):
        observed[_round_half_up_bin(actual) - 1][_round_half_up_bin(predicted) - 1] += 1.0
    row = [sum(line) for line in observed]
    col = [sum(observed[i][j] for i in range(classes)) for j in range(classes)]
    total = float(len(y_true))
    weighted_observed = weighted_expected = 0.0
    for i in range(classes):
        for j in range(classes):
            weight = ((i - j) ** 2) / ((classes - 1) ** 2)
            weighted_observed += weight * observed[i][j] / total
            weighted_expected += weight * (row[i] * col[j]) / (total * total)
    if weighted_expected == 0:
        return 1.0 if weighted_observed == 0 else None
    return 1.0 - weighted_observed / weighted_expected


def _values(rows: Sequence[ScoreVector | Mapping[str, float]], field: str) -> list[float]:
    result = []
    for row in rows:
        value = getattr(row, field) if isinstance(row, ScoreVector) else row.get(field)
        result.append(_finite_number(value))
    return result


def compute_regression_metrics(
    targets: Sequence[ScoreVector | Mapping[str, float]], predictions: Sequence[ScoreVector | Mapping[str, float]]
) -> dict[str, Any]:
    """Compute predeclared metrics without retaining rows or any free text."""
    if len(targets) != len(predictions) or not targets:
        raise MetricError("targets and predictions must be nonempty and equally sized")
    per_target: dict[str, dict[str, float | None]] = {}
    for field in SCORE_FIELDS:
        actual, predicted = _values(targets, field), _values(predictions, field)
        errors = [abs(a - b) for a, b in zip(actual, predicted, strict=True)]
        per_target[field] = {
            "mae": _mean(errors),
            "rmse": sqrt(_mean([(a - b) ** 2 for a, b in zip(actual, predicted, strict=True)])),
            "pearson_r": _pearson(actual, predicted),
            "spearman_rho": _pearson(_average_ranks(actual), _average_ranks(predicted)),
            "quadratic_weighted_kappa": quadratic_weighted_kappa(actual, predicted),
        }
    return {"record_count": len(targets), "primary_target": "average", "per_target": per_target}


def train_mean_vector(rows: Sequence[ScoreVector | Mapping[str, float]]) -> ScoreVector:
    if not rows:
        raise MetricError("train mean requires at least one row")
    return ScoreVector(**{field: _mean(_values(rows, field)) for field in SCORE_FIELDS})


def aggregate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Accept only score vectors, then return metrics; reject private raw fields."""
    forbidden = {"essay", "prompt", "id", "document_id", "feedback", "rationale", "text", "tokens", "raw_output"}
    targets: list[Mapping[str, float]] = []
    predictions: list[Mapping[str, float]] = []
    for row in rows:
        if forbidden.intersection(row):
            raise MetricError("aggregate metric input contains a forbidden raw-content field")
        if set(row) != {"target", "prediction"}:
            raise MetricError("prediction rows must contain only target and prediction score objects")
        targets.append(row["target"])
        predictions.append(row["prediction"])
    return compute_regression_metrics(targets, predictions)
