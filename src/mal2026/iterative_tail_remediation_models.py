"""Cold-start, train-only remediation models for iterative tail predictions.

The API consumes only three-axis gold, base, and challenger matrices.  It
never accepts an essay-average target and never writes row-level artifacts.
All model selection is deterministic over the predeclared grids below.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Literal, Mapping, Sequence

import numpy as np


Family = Literal[
    "gated_delta",
    "weighted_isotonic",
    "piecewise_5knot",
    "tail_boundary",
    "convex_blend",
]

FAMILIES: tuple[str, ...] = (
    "gated_delta",
    "weighted_isotonic",
    "piecewise_5knot",
    "tail_boundary",
    "convex_blend",
)

PREDECLARED_GRIDS: Mapping[str, tuple[Any, ...]] = {
    "gate_kind": ("hard", "sigmoid"),
    "gate_threshold": (2.5, 3.0, 3.5, 4.0),
    "gate_temperature": (0.10, 0.25, 0.50, 1.00),
    "delta_weight": (0.0, 0.25, 0.50, 0.75, 1.0),
    "low_identity_threshold": (None, 2.0, 2.5),
    "calibration_source": ("base", "challenger"),
    "tail_source": ("base", "challenger"),
    "low_offset": (-0.20, -0.10, 0.0, 0.10, 0.20),
    "high_offset": (-0.20, -0.10, 0.0, 0.10, 0.20),
    "boundary_nudge": (0.0, 0.05, 0.10, 0.20),
    "blend_weight": tuple(step / 20.0 for step in range(21)),
}

KNOTS_5 = np.asarray((1.0, 2.0, 3.0, 4.0, 5.0), dtype=np.float64)


@dataclass(frozen=True)
class RemediationSpec:
    """One predeclared remediation family and its fixed weighting policy."""

    family: Family
    equal_gold_band_weights: bool = False

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown remediation family: {self.family!r}")


@dataclass(frozen=True)
class RemediationResult:
    """In-memory fold predictions plus aggregate-safe selected parameters."""

    predictions: np.ndarray
    selected_parameters: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]
    train_objective: float
    family: str


def _matrix(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have exactly three axis columns; average is forbidden")
    if len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty and finite")
    if np.any(result < 1.0) or np.any(result > 5.0):
        raise ValueError(f"{name} must be within [1, 5]")
    return result


def gold_band_equal_weights(gold: Any) -> np.ndarray:
    """Return per-axis inverse-frequency weights for rounded gold bands 1..5."""
    target = _matrix(gold, "gold")
    bands = np.floor(target + 0.5).astype(int).clip(1, 5)
    result = np.empty_like(target)
    for axis in range(3):
        counts = np.bincount(bands[:, axis], minlength=6)[1:]
        observed = counts > 0
        per_class = np.zeros(5, dtype=np.float64)
        per_class[observed] = len(target) / (observed.sum() * counts[observed])
        result[:, axis] = per_class[bands[:, axis] - 1]
    return result


def _weights(gold: np.ndarray, equal_bands: bool) -> np.ndarray:
    return gold_band_equal_weights(gold) if equal_bands else np.ones_like(gold)


def _axis_mse(gold: np.ndarray, prediction: np.ndarray, weights: np.ndarray, axis: int) -> float:
    return float(np.sum(weights[:, axis] * np.square(gold[:, axis] - prediction)) / np.sum(weights[:, axis]))


def _macro_rmse(gold: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    return float(np.mean([
        np.sqrt(_axis_mse(gold, prediction[:, axis], weights, axis)) for axis in range(3)
    ]))


def apply_score_conditional_gate(
    base: Any,
    challenger: Any,
    *,
    kind: str,
    threshold: float,
    temperature: float,
    weight: float,
    low_identity_threshold: float | None = None,
) -> np.ndarray:
    """Blend challenger delta through a hard or sigmoid score gate."""
    base_values = np.asarray(base, dtype=np.float64)
    challenger_values = np.asarray(challenger, dtype=np.float64)
    if base_values.shape != challenger_values.shape:
        raise ValueError("base and challenger shapes must match")
    if kind not in PREDECLARED_GRIDS["gate_kind"] or temperature <= 0 or not 0 <= weight <= 1:
        raise ValueError("invalid gate parameters")
    if kind == "hard":
        gate = (base_values > threshold).astype(np.float64)
    else:
        z = np.clip((base_values - threshold) / temperature, -40.0, 40.0)
        gate = 1.0 / (1.0 + np.exp(-z))
    if low_identity_threshold is not None:
        gate = np.where(base_values <= low_identity_threshold, 0.0, gate)
    return np.clip(base_values + weight * gate * (challenger_values - base_values), 1.0, 5.0)


def weighted_pava(values: Sequence[float], weights: Sequence[float]) -> np.ndarray:
    """Weighted pool-adjacent-violators projection onto nondecreasing values."""
    y = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if y.ndim != 1 or w.shape != y.shape or len(y) == 0:
        raise ValueError("PAVA values and weights must be nonempty vectors of equal length")
    if not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("PAVA values must be finite and weights positive")
    means: list[float] = []
    masses: list[float] = []
    lengths: list[int] = []
    for value, mass in zip(y, w, strict=True):
        means.append(float(value)); masses.append(float(mass)); lengths.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            merged_mass = masses[-2] + masses[-1]
            merged_mean = (means[-2] * masses[-2] + means[-1] * masses[-1]) / merged_mass
            merged_length = lengths[-2] + lengths[-1]
            means[-2:] = [merged_mean]
            masses[-2:] = [merged_mass]
            lengths[-2:] = [merged_length]
    return np.concatenate([np.full(length, mean) for mean, length in zip(means, lengths, strict=True)])


def _fit_isotonic(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="mergesort")
    sorted_x, sorted_y, sorted_w = x[order], y[order], weights[order]
    unique_x, starts = np.unique(sorted_x, return_index=True)
    sums_w = np.add.reduceat(sorted_w, starts)
    sums_y = np.add.reduceat(sorted_w * sorted_y, starts)
    fitted = weighted_pava(sums_y / sums_w, sums_w)
    return unique_x, fitted


def _predict_isotonic(x: np.ndarray, fitted_x: np.ndarray, fitted_y: np.ndarray) -> np.ndarray:
    return np.clip(np.interp(x, fitted_x, fitted_y, left=fitted_y[0], right=fitted_y[-1]), 1.0, 5.0)


def _piecewise_design(x: np.ndarray) -> np.ndarray:
    """Linear-interpolation design for fixed knots 1,2,3,4,5."""
    value = np.clip(np.asarray(x, dtype=np.float64), 1.0, 5.0)
    left = np.floor(value).astype(int).clip(1, 4)
    fraction = value - left
    at_upper = value >= 5.0
    left[at_upper] = 4
    fraction[at_upper] = 1.0
    design = np.zeros((len(value), 5), dtype=np.float64)
    rows = np.arange(len(value))
    design[rows, left - 1] = 1.0 - fraction
    design[rows, left] = fraction
    return design


def _fit_five_knots(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exact bounded monotone weighted least-squares five-knot fit.

    With five ordered knot ordinates, the convex feasible set has only 16
    contiguous equality partitions and four lower/upper-bound states.  We
    enumerate those faces, solve weighted least squares on each face, reject
    infeasible solutions, and retain the global minimum.  This avoids a
    nearest-bin proxy and needs no external optimizer.
    """
    design = _piecewise_design(x)
    sqrt_weight = np.sqrt(np.asarray(weights, dtype=np.float64))
    weighted_design = design * sqrt_weight[:, None]
    weighted_y = np.asarray(y, dtype=np.float64) * sqrt_weight
    best: tuple[float, tuple[int, int, int], np.ndarray] | None = None

    # A set bit joins adjacent knot ordinates into one active equality block.
    for joined_mask in range(1 << 4):
        block_for_knot = [0]
        for boundary in range(4):
            block_for_knot.append(
                block_for_knot[-1] if joined_mask & (1 << boundary) else block_for_knot[-1] + 1
            )
        block_count = block_for_knot[-1] + 1
        expansion = np.zeros((5, block_count), dtype=np.float64)
        expansion[np.arange(5), block_for_knot] = 1.0
        face_design = weighted_design @ expansion
        for lower_active in (0, 1):
            for upper_active in (0, 1):
                fixed: dict[int, float] = {}
                if lower_active:
                    fixed[0] = 1.0
                if upper_active:
                    fixed[block_count - 1] = 5.0
                # One block cannot be fixed simultaneously to different bounds.
                if len(fixed) < lower_active + upper_active:
                    continue
                free = [index for index in range(block_count) if index not in fixed]
                residual = weighted_y.copy()
                if fixed:
                    fixed_columns = np.asarray(sorted(fixed), dtype=int)
                    fixed_values = np.asarray([fixed[index] for index in fixed_columns], dtype=np.float64)
                    residual -= face_design[:, fixed_columns] @ fixed_values
                blocks = np.empty(block_count, dtype=np.float64)
                for index, value in fixed.items():
                    blocks[index] = value
                if free:
                    solution, _, _, _ = np.linalg.lstsq(face_design[:, free], residual, rcond=None)
                    blocks[free] = solution
                tolerance = 1e-10
                if (
                    np.any(blocks < 1.0 - tolerance)
                    or np.any(blocks > 5.0 + tolerance)
                    or np.any(np.diff(blocks) < -tolerance)
                ):
                    continue
                knots = np.clip(expansion @ blocks, 1.0, 5.0)
                prediction = design @ knots
                objective = float(np.sum(weights * np.square(y - prediction)))
                key = (objective, (joined_mask, lower_active, upper_active), knots)
                if best is None or key[:2] < best[:2]:
                    best = key
    if best is None:  # pragma: no cover - the all-equal bounded face is feasible
        raise ValueError("five-knot constrained least-squares fit failed")
    return best[2]


def apply_tail_boundary_adjustment(
    values: Any,
    *,
    low_offset: float,
    high_offset: float,
    boundary_nudge: float,
) -> np.ndarray:
    """Apply tail offsets and a bounded away-from-3.5 middle nudge."""
    score = np.asarray(values, dtype=np.float64)
    adjusted = score.copy()
    adjusted += np.where(score <= 2.5, low_offset, 0.0)
    adjusted += np.where(score >= 4.5, high_offset, 0.0)
    proximity = np.clip(1.0 - 2.0 * np.abs(score - 3.5), 0.0, 1.0)
    direction = np.where(score < 3.5, -1.0, 1.0)
    adjusted += boundary_nudge * proximity * direction
    return np.clip(adjusted, 1.0, 5.0)


def _best(candidates: Sequence[tuple[float, tuple[Any, ...], np.ndarray]]) -> tuple[float, tuple[Any, ...], np.ndarray]:
    if not candidates:
        raise ValueError("candidate grid is empty")
    return min(candidates, key=lambda item: (item[0], tuple(str(value) for value in item[1])))


def fit_predict(
    spec: RemediationSpec,
    train_gold: Any,
    train_base: Any,
    train_challenger: Any,
    test_base: Any,
    test_challenger: Any,
) -> RemediationResult:
    """Select predeclared parameters on train only and predict held-out rows."""
    gold = _matrix(train_gold, "train_gold")
    base = _matrix(train_base, "train_base")
    challenger = _matrix(train_challenger, "train_challenger")
    infer_base = _matrix(test_base, "test_base")
    infer_challenger = _matrix(test_challenger, "test_challenger")
    if gold.shape != base.shape or gold.shape != challenger.shape:
        raise ValueError("training gold/base/challenger shapes must match")
    if infer_base.shape != infer_challenger.shape:
        raise ValueError("test base/challenger shapes must match")
    weights = _weights(gold, spec.equal_gold_band_weights)
    output = np.empty_like(infer_base)
    train_selected = np.empty_like(gold)
    parameters: list[Mapping[str, Any]] = []

    for axis in range(3):
        y, b, c, w = gold[:, axis], base[:, axis], challenger[:, axis], weights[:, axis]
        tb, tc = infer_base[:, axis], infer_challenger[:, axis]
        candidates: list[tuple[float, tuple[Any, ...], np.ndarray]] = []

        if spec.family == "gated_delta":
            for kind, threshold, temperature, delta_weight, low_identity in product(
                PREDECLARED_GRIDS["gate_kind"], PREDECLARED_GRIDS["gate_threshold"],
                PREDECLARED_GRIDS["gate_temperature"], PREDECLARED_GRIDS["delta_weight"],
                PREDECLARED_GRIDS["low_identity_threshold"],
            ):
                key = (kind, threshold, temperature, delta_weight, low_identity)
                pred = apply_score_conditional_gate(
                    b, c, kind=kind, threshold=threshold, temperature=temperature,
                    weight=delta_weight, low_identity_threshold=low_identity,
                )
                candidates.append((_axis_mse(gold, pred, weights, axis), key, pred))
            score, key, train_pred = _best(candidates)
            kind, threshold, temperature, delta_weight, low_identity = key
            test_pred = apply_score_conditional_gate(
                tb, tc, kind=kind, threshold=threshold, temperature=temperature,
                weight=delta_weight, low_identity_threshold=low_identity,
            )
            parameter = dict(kind=kind, threshold=threshold, temperature=temperature,
                             weight=delta_weight, low_identity_threshold=low_identity)
        elif spec.family in {"weighted_isotonic", "piecewise_5knot"}:
            fitted_by_key: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
            for source in PREDECLARED_GRIDS["calibration_source"]:
                x = b if source == "base" else c
                if spec.family == "weighted_isotonic":
                    fit_x, fit_y = _fit_isotonic(x, y, w)
                else:
                    fit_x, fit_y = KNOTS_5, _fit_five_knots(x, y, w)
                pred = _predict_isotonic(x, fit_x, fit_y)
                key = (source,)
                fitted_by_key[key] = (fit_x, fit_y)
                candidates.append((_axis_mse(gold, pred, weights, axis), key, pred))
            score, key, train_pred = _best(candidates)
            fit_x, fit_y = fitted_by_key[key]
            source_values = tb if key[0] == "base" else tc
            test_pred = _predict_isotonic(source_values, fit_x, fit_y)
            parameter = {"source": key[0], "x_knots": tuple(float(v) for v in fit_x),
                         "y_knots": tuple(float(v) for v in fit_y)}
        elif spec.family == "tail_boundary":
            for source, low, high, nudge in product(
                PREDECLARED_GRIDS["tail_source"], PREDECLARED_GRIDS["low_offset"],
                PREDECLARED_GRIDS["high_offset"], PREDECLARED_GRIDS["boundary_nudge"],
            ):
                x = b if source == "base" else c
                key = (source, low, high, nudge)
                pred = apply_tail_boundary_adjustment(x, low_offset=low, high_offset=high, boundary_nudge=nudge)
                candidates.append((_axis_mse(gold, pred, weights, axis), key, pred))
            score, key, train_pred = _best(candidates)
            source, low, high, nudge = key
            test_pred = apply_tail_boundary_adjustment(
                tb if source == "base" else tc, low_offset=low, high_offset=high, boundary_nudge=nudge,
            )
            parameter = dict(source=source, low_offset=low, high_offset=high, boundary_nudge=nudge)
        else:
            for blend_weight in PREDECLARED_GRIDS["blend_weight"]:
                key = (blend_weight,)
                pred = np.clip((1.0 - blend_weight) * b + blend_weight * c, 1.0, 5.0)
                candidates.append((_axis_mse(gold, pred, weights, axis), key, pred))
            score, key, train_pred = _best(candidates)
            blend_weight = key[0]
            test_pred = np.clip((1.0 - blend_weight) * tb + blend_weight * tc, 1.0, 5.0)
            parameter = {"weight": blend_weight}

        del score
        train_selected[:, axis] = train_pred
        output[:, axis] = test_pred
        parameters.append(parameter)

    return RemediationResult(
        predictions=np.asarray(np.clip(output, 1.0, 5.0), dtype=np.float32),
        selected_parameters=tuple(parameters),  # type: ignore[arg-type]
        train_objective=_macro_rmse(gold, train_selected, weights),
        family=spec.family,
    )


__all__ = [
    "FAMILIES", "KNOTS_5", "PREDECLARED_GRIDS", "RemediationResult", "RemediationSpec",
    "apply_score_conditional_gate", "apply_tail_boundary_adjustment", "fit_predict",
    "gold_band_equal_weights", "weighted_pava",
]
