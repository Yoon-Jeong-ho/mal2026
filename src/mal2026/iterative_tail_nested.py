"""Leakage-guarded nested evaluation for iterative tail-score candidates.

Candidate callbacks receive only ``(fit_indices, predict_indices)`` and return
an ``N x 3`` prediction matrix.  Inner selection is confined to each outer
training partition; outer-fold gold is first consumed only after a candidate
has been fixed and its outer predictions have been produced.  Results contain
aggregates and selection decisions, never row-level predictions or indices.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .iterative_tail_metrics import (
    AXES,
    compute_iterative_tail_metrics,
    paired_bootstrap_delta_ci,
    promotion_decision,
)


Predictor = Callable[[np.ndarray, np.ndarray], Sequence[Sequence[float]]]


class NestedEvaluationError(ValueError):
    """Raised when folds, callbacks, or predictions violate nested evaluation."""


@dataclass(frozen=True)
class Candidate:
    """A named model-fitting callback evaluated in the declared order."""

    name: str
    fit_predict: Predictor


def make_validation_folds(indices: Sequence[int], n_splits: int, seed: int) -> tuple[np.ndarray, ...]:
    """Deterministically partition unique indices into nonempty validation folds."""
    values = np.asarray(indices, dtype=int)
    if values.ndim != 1 or len(values) < n_splits or n_splits < 2 or len(np.unique(values)) != len(values):
        raise NestedEvaluationError("fold indices must be unique and numerous enough for n_splits")
    shuffled = np.random.default_rng(seed).permutation(values)
    return tuple(np.sort(part.astype(int, copy=False)) for part in np.array_split(shuffled, n_splits))


def _targets(values: Sequence[Sequence[float]] | Mapping[str, Sequence[float]]) -> np.ndarray:
    if isinstance(values, Mapping):
        if set(values) != set(AXES):
            raise NestedEvaluationError("targets must contain exactly three axes; average is forbidden")
        columns = [np.asarray(values[axis], dtype=float) for axis in AXES]
        if any(column.ndim != 1 for column in columns) or len({len(column) for column in columns}) != 1:
            raise NestedEvaluationError("target axes must be one-dimensional and equally sized")
        result = np.column_stack(columns)
    else:
        result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 3 or not np.all(np.isfinite(result)):
        raise NestedEvaluationError("targets must be a finite, nonempty N x 3 matrix")
    return result


def _validate_partition(folds: Sequence[Sequence[int]], universe: np.ndarray, expected: int, label: str) -> tuple[np.ndarray, ...]:
    if len(folds) != expected:
        raise NestedEvaluationError(f"{label} must contain exactly {expected} folds")
    normalized = tuple(np.asarray(fold, dtype=int) for fold in folds)
    if any(fold.ndim != 1 or len(fold) == 0 or len(np.unique(fold)) != len(fold) for fold in normalized):
        raise NestedEvaluationError(f"{label} folds must be nonempty, one-dimensional, and internally unique")
    concatenated = np.concatenate(normalized)
    if len(np.unique(concatenated)) != len(concatenated):
        raise NestedEvaluationError(f"{label} folds overlap")
    if set(concatenated.tolist()) != set(universe.tolist()):
        raise NestedEvaluationError(f"{label} folds must partition only their declared universe")
    return normalized


def _predict(candidate: Candidate, fit_indices: np.ndarray, predict_indices: np.ndarray) -> np.ndarray:
    if np.intersect1d(fit_indices, predict_indices).size:
        raise NestedEvaluationError("fit and predict indices overlap")
    prediction = np.asarray(candidate.fit_predict(fit_indices.copy(), predict_indices.copy()), dtype=float)
    if prediction.shape != (len(predict_indices), 3) or not np.all(np.isfinite(prediction)):
        raise NestedEvaluationError(f"candidate {candidate.name!r} must return a finite len(predict_indices) x 3 matrix")
    return prediction


def _inner_oof(
    candidate: Candidate,
    outer_train: np.ndarray,
    outer_validation: np.ndarray,
    inner_folds: tuple[np.ndarray, ...],
    total_records: int,
) -> np.ndarray:
    prediction = np.empty((total_records, 3), dtype=float)
    filled = np.zeros(total_records, dtype=bool)
    outer_train_set = set(outer_train.tolist())
    outer_validation_set = set(outer_validation.tolist())
    for inner_validation in inner_folds:
        inner_train = np.asarray(sorted(outer_train_set - set(inner_validation.tolist())), dtype=int)
        # These assertions are deliberately adjacent to the callback boundary.
        if not set(inner_train.tolist()).issubset(outer_train_set) or not set(inner_validation.tolist()).issubset(outer_train_set):
            raise NestedEvaluationError("inner callback indices escaped the outer-training partition")
        if outer_validation_set.intersection(inner_train.tolist()) or outer_validation_set.intersection(inner_validation.tolist()):
            raise NestedEvaluationError("outer-fold gold indices reached inner selection")
        prediction[inner_validation] = _predict(candidate, inner_train, inner_validation)
        filled[inner_validation] = True
    if not np.all(filled[outer_train]):
        raise NestedEvaluationError("inner OOF predictions do not cover the outer-training partition")
    return prediction[outer_train]


def nested_tail_evaluation(
    targets: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    baseline: Candidate,
    candidates: Sequence[Candidate],
    *,
    outer_validation_folds: Sequence[Sequence[int]] | None = None,
    inner_validation_folds: Sequence[Sequence[Sequence[int]]] | None = None,
    seed: int = 20260801,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    """Run five-by-four nested strict-gate selection and aggregate evaluation.

    Candidates are considered sequentially on four-fold inner OOF predictions.
    A candidate replaces the current incumbent only when every strict gate from
    :func:`promotion_decision` passes.  If none passes (including an empty
    candidate sequence), the baseline is retained fail-closed for that fold.
    """
    truth = _targets(targets)
    all_indices = np.arange(len(truth), dtype=int)
    outer_raw = outer_validation_folds if outer_validation_folds is not None else make_validation_folds(all_indices, 5, seed)
    outer_folds = _validate_partition(outer_raw, all_indices, 5, "outer validation")
    candidate_list = tuple(candidates)
    names = [baseline.name, *(candidate.name for candidate in candidate_list)]
    if not baseline.name or any(not name for name in names) or len(set(names)) != len(names):
        raise NestedEvaluationError("baseline and candidate names must be nonempty and unique")
    if inner_validation_folds is not None and len(inner_validation_folds) != 5:
        raise NestedEvaluationError("inner_validation_folds must contain one four-fold partition per outer fold")

    baseline_outer = np.empty_like(truth)
    selected_outer = np.empty_like(truth)
    fold_audit: list[dict[str, Any]] = []
    for outer_number, outer_validation in enumerate(outer_folds):
        outer_train = np.asarray(sorted(set(all_indices.tolist()) - set(outer_validation.tolist())), dtype=int)
        raw_inner = (
            inner_validation_folds[outer_number]
            if inner_validation_folds is not None
            else make_validation_folds(outer_train, 4, seed + outer_number + 1)
        )
        inner_folds = _validate_partition(raw_inner, outer_train, 4, f"outer fold {outer_number + 1} inner validation")

        incumbent = baseline
        decisions: list[dict[str, Any]] = []
        if candidate_list:
            incumbent_prediction = _inner_oof(baseline, outer_train, outer_validation, inner_folds, len(truth))
            incumbent_metrics = compute_iterative_tail_metrics(truth[outer_train], incumbent_prediction)
            for candidate in candidate_list:
                candidate_prediction = _inner_oof(candidate, outer_train, outer_validation, inner_folds, len(truth))
                candidate_metrics = compute_iterative_tail_metrics(truth[outer_train], candidate_prediction)
                decision = promotion_decision(incumbent_metrics, candidate_metrics)
                decisions.append({"candidate": candidate.name, **decision})
                if decision["promote"]:
                    incumbent = candidate
                    incumbent_prediction = candidate_prediction
                    incumbent_metrics = candidate_metrics

        # Model choice is frozen before either callback predicts the outer fold.
        baseline_outer[outer_validation] = _predict(baseline, outer_train, outer_validation)
        selected_outer[outer_validation] = (
            baseline_outer[outer_validation]
            if incumbent is baseline
            else _predict(incumbent, outer_train, outer_validation)
        )
        fold_audit.append({
            "outer_fold": outer_number + 1,
            "outer_train_count": len(outer_train),
            "outer_validation_count": len(outer_validation),
            "inner_fold_count": 4,
            "selected_candidate": incumbent.name,
            "fell_back_to_identity_baseline": incumbent is baseline,
            "sequential_decisions": decisions,
            "outer_gold_used_for_inner_selection": False,
        })

    baseline_metrics = compute_iterative_tail_metrics(truth, baseline_outer)
    selected_metrics = compute_iterative_tail_metrics(truth, selected_outer)
    bootstrap = paired_bootstrap_delta_ci(
        truth,
        baseline_outer,
        selected_outer,
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    improvement_interval = bootstrap["intervals"]["rmse"]
    candidate_minus_baseline_ci = {
        "estimate": -improvement_interval["estimate"],
        "lower": -improvement_interval["upper"],
        "upper": -improvement_interval["lower"],
    }
    final_gates = {
        "macro_rmse_improvement_at_least_0_01": improvement_interval["estimate"] >= 0.01,
        "candidate_minus_baseline_rmse_ci_upper_below_zero": candidate_minus_baseline_ci["upper"] < 0.0,
    }
    return {
        "schema_version": "mal2026-iterative-tail-nested-v1",
        "record_count": len(truth),
        "outer_fold_count": 5,
        "inner_fold_count": 4,
        "baseline_name": baseline.name,
        "candidate_order": [candidate.name for candidate in candidate_list],
        "folds": fold_audit,
        "baseline_metrics": baseline_metrics,
        "selected_metrics": selected_metrics,
        "paired_row_bootstrap": bootstrap,
        "candidate_minus_baseline_rmse_ci": candidate_minus_baseline_ci,
        "final_gates": final_gates,
        "final_gate_pass": all(final_gates.values()),
        "predictions_returned": False,
    }


# Short alias for orchestration callers.
run_nested_evaluation = nested_tail_evaluation
