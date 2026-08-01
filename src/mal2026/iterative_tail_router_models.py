"""Deterministic score-only routers over fixed train-OOF component predictions."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from mal2026.iterative_tail_metrics import compute_iterative_tail_metrics


FAMILIES = (
    "low_protected_sigmoid_stack", "four_zone_hard_stack", "boundary_hurdle_overlay",
    "sigmoid_four_expert_route", "formal_gate_lattice_stack",
)
COMPONENTS = ("soft", "direct", "hurdle", "r17")


@dataclass(frozen=True)
class RouterSpec:
    cycle: int
    family: str
    variant_id: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class RouterResult:
    train_predictions: np.ndarray
    selected_parameters: Mapping[str, Any]
    audit: Mapping[str, Any]


def _spec(cycle: int, family: str, variant: int, **parameters: Any) -> RouterSpec:
    return RouterSpec(cycle, family, f"{family}-v{variant}", MappingProxyType(parameters))


def router_specs() -> tuple[RouterSpec, ...]:
    return (
        _spec(1, FAMILIES[0], 1, temperature=.10, max_nonidentity=.15, step=.05),
        _spec(2, FAMILIES[0], 2, temperature=.20, max_nonidentity=.20, step=.05),
        _spec(3, FAMILIES[0], 3, temperature=.35, max_nonidentity=.25, step=.05),
        _spec(4, FAMILIES[0], 4, temperature=.50, max_nonidentity=.30, step=.05),
        _spec(5, FAMILIES[1], 1, identity_floors=(.90, .70, .70, .60), step=.10, passes=2),
        _spec(6, FAMILIES[1], 2, identity_floors=(.85, .60, .60, .50), step=.10, passes=2),
        _spec(7, FAMILIES[1], 3, identity_floors=(.80, .50, .50, .40), step=.10, passes=2),
        _spec(8, FAMILIES[1], 4, identity_floors=(.75, .40, .40, .30), step=.10, passes=2),
        _spec(9, FAMILIES[2], 1, window=.05, max_nonidentity=.15, step=.05),
        _spec(10, FAMILIES[2], 2, window=.10, max_nonidentity=.20, step=.05),
        _spec(11, FAMILIES[2], 3, window=.15, max_nonidentity=.25, step=.05),
        _spec(12, FAMILIES[2], 4, window=.20, max_nonidentity=.30, step=.05),
        _spec(13, FAMILIES[3], 1, temperature=.10, identity_floor=.80, pareto_margin=0.0, step=.05, passes=2),
        _spec(14, FAMILIES[3], 2, temperature=.20, identity_floor=.70, pareto_margin=0.0, step=.05, passes=2),
        _spec(15, FAMILIES[3], 3, temperature=.35, identity_floor=.60, pareto_margin=1e-4, step=.05, passes=2),
        _spec(16, FAMILIES[3], 4, temperature=.50, identity_floor=.50, pareto_margin=2.5e-4, step=.05, passes=2),
        _spec(17, FAMILIES[4], 1, temperature=.15, max_nonidentity=.20, correction_cap=.10, step=.10, passes=2),
        _spec(18, FAMILIES[4], 2, temperature=.25, max_nonidentity=.30, correction_cap=.15, step=.10, passes=2),
        _spec(19, FAMILIES[4], 3, temperature=.35, max_nonidentity=.25, correction_cap=.15, step=.05, passes=2),
        _spec(20, FAMILIES[4], 4, temperature=.50, max_nonidentity=.35, correction_cap=.20, step=.05, passes=2),
    )


_REGISTERED = {spec.variant_id: spec for spec in router_specs()}


def _validate_spec(spec: RouterSpec) -> None:
    registered = _REGISTERED.get(spec.variant_id)
    if (
        registered is None or spec.cycle != registered.cycle or spec.family != registered.family
        or dict(spec.parameters) != dict(registered.parameters)
    ):
        raise ValueError("router spec differs from the fixed inventory")


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have exactly three columns; average is forbidden")
    if len(result) == 0 or not np.isfinite(result).all() or np.any(result < 1) or np.any(result > 5):
        raise ValueError(f"{name} must be nonempty, finite, and bounded [1, 5]")
    return result


def _inputs(base: Any, r17: Any, direct: Any, hurdle: Any, soft: Any) -> tuple[np.ndarray, ...]:
    values = tuple(_matrix(value, name) for value, name in zip(
        (base, r17, direct, hurdle, soft), ("base", "r17", "direct", "hurdle", "soft"), strict=True,
    ))
    if len({value.shape for value in values}) != 1:
        raise ValueError("all route component matrices must have identical shapes")
    return values


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))


def _metric_state(gold: np.ndarray, base: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    baseline = compute_iterative_tail_metrics(gold, base)
    candidate = compute_iterative_tail_metrics(gold, prediction)
    bm, cm = baseline["macro"], candidate["macro"]
    improvement = {
        key: float(bm[key] - cm[key]) for key in ("rmse", "equal_group_rmse", "low_tail_rmse", "high_tail_rmse")
    }
    improvement["gold_3_4_balanced_accuracy"] = float(cm["gold_3_4_balanced_accuracy"] - bm["gold_3_4_balanced_accuracy"])
    improvement["spearman"] = float(cm["spearman"] - bm["spearman"])
    axis = {name: float(baseline["axes"][name]["rmse"] - candidate["axes"][name]["rmse"])
            for name in ("content", "organization", "expression")}
    slacks = (
        (improvement["rmse"] - .005) / .005,
        (improvement["equal_group_rmse"] - .010) / .010,
        improvement["low_tail_rmse"] / .005,
        improvement["high_tail_rmse"] / .005,
        (improvement["gold_3_4_balanced_accuracy"] - .010) / .010,
        (min(axis.values()) + .010) / .010,
        (improvement["spearman"] + .005) / .005,
    )
    eligible = (
        improvement["rmse"] >= .005 and improvement["equal_group_rmse"] >= .010
        and improvement["low_tail_rmse"] > 0 and improvement["high_tail_rmse"] > 0
        and improvement["gold_3_4_balanced_accuracy"] >= .010
        and min(axis.values()) >= -.010 and improvement["spearman"] >= -.005
    )
    return {"eligible": eligible, "normalized_gate_slack": float(min(slacks)),
            "macro_rmse": float(cm["rmse"]), "improvements": improvement, "axis_rmse": axis}


def _better(left: dict[str, Any], right: dict[str, Any] | None) -> bool:
    if right is None:
        return True
    return (left["normalized_gate_slack"], -left["macro_rmse"]) > (right["normalized_gate_slack"], -right["macro_rmse"])


def _weight_values(step: float, maximum: float) -> tuple[float, ...]:
    return tuple(round(index * step, 10) for index in range(int(round(maximum / step)) + 1))


def _zones(base: np.ndarray, temperature: float) -> np.ndarray:
    s1, s2, s3 = (_sigmoid((base - center) / temperature) for center in (2.5, 3.5, 4.5))
    return np.stack((1 - s1, s1 - s2, s2 - s3, s3), axis=-1)


def _component_map(r17: np.ndarray, direct: np.ndarray, hurdle: np.ndarray, soft: np.ndarray) -> dict[str, np.ndarray]:
    return {"r17": r17, "direct": direct, "hurdle": hurdle, "soft": soft}


def _low_protected(spec: RouterSpec, base: np.ndarray, components: dict[str, np.ndarray], weights: tuple[float, ...]) -> np.ndarray:
    gate = _sigmoid((base - 2.5) / float(spec.parameters["temperature"]))
    gate = np.where(base <= 2.0, 0.0, gate)
    correction = sum(weight * (components[name] - base) for weight, name in zip(weights, ("r17", "direct", "soft"), strict=True))
    return np.clip(base + gate * correction, 1, 5)


def _hard_prediction(base: np.ndarray, components: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    pairs = (("soft", "hurdle"), ("hurdle", "direct"), ("hurdle", "r17"), ("r17", "direct"))
    zone = np.select((base <= 2.5, base <= 3.5, base <= 4.5), (0, 1, 2), default=3)
    correction = np.zeros_like(base)
    for index, pair in enumerate(pairs):
        value = sum(weights[index, column] * (components[name] - base) for column, name in enumerate(pair))
        correction = np.where(zone == index, value, correction)
    return np.clip(base + correction, 1, 5)


def _boundary(spec: RouterSpec, base: np.ndarray, components: dict[str, np.ndarray], weights: tuple[float, ...]) -> np.ndarray:
    soft_weight, hurdle_weight, r17_weight = weights
    window = float(spec.parameters["window"])
    near = np.zeros_like(base, dtype=bool)
    for center in (2.5, 3.5, 4.5):
        near |= np.abs(base - center) <= window
    correction = soft_weight * (components["soft"] - base)
    correction += hurdle_weight * near * (components["hurdle"] - base)
    correction += r17_weight * (base >= 4.25) * (components["r17"] - base)
    return np.clip(base + correction, 1, 5)


def _routed_prediction(
    spec: RouterSpec, base: np.ndarray, components: dict[str, np.ndarray], weights: np.ndarray,
    *, cap: float | None = None,
) -> np.ndarray:
    zone = _zones(base, float(spec.parameters["temperature"]))
    correction = np.zeros_like(base)
    for route in range(4):
        for component, name in enumerate(COMPONENTS):
            correction += zone[:, :, route] * weights[route, component] * (components[name] - base)
    if cap is not None:
        correction = np.clip(correction, -cap, cap)
    return np.clip(base + correction, 1, 5)


def _pareto(gold: np.ndarray, base: np.ndarray, prediction: np.ndarray, margin: float) -> bool:
    band = np.floor(gold + .5).astype(int).clip(1, 5)
    for axis in range(3):
        for label in range(1, 6):
            mask = band[:, axis] == label
            if np.any(mask):
                base_mse = np.mean(np.square(gold[mask, axis] - base[mask, axis]))
                candidate_mse = np.mean(np.square(gold[mask, axis] - prediction[mask, axis]))
                if candidate_mse > base_mse + margin + 1e-12:
                    return False
    return True


def _coordinate_search(
    spec: RouterSpec, gold: np.ndarray, base: np.ndarray, components: dict[str, np.ndarray],
    *, hard: bool = False, pareto_margin: float | None = None, strict_fallback: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    weights = np.zeros((4, 2 if hard else 4), dtype=np.float64)
    prediction = base.copy()
    state = _metric_state(gold, base, prediction)
    step = float(spec.parameters["step"])
    floors = tuple(spec.parameters["identity_floors"]) if hard else (float(spec.parameters.get("identity_floor", 1 - spec.parameters.get("max_nonidentity", 0))),) * 4
    for _ in range(int(spec.parameters["passes"])):
        for route in range(4):
            for component in range(weights.shape[1]):
                maximum = 1.0 - floors[route]
                best_weights, best_prediction, best_state = weights, prediction, state
                for value in _weight_values(step, maximum):
                    trial = weights.copy(); trial[route, component] = value
                    if trial[route].sum() > maximum + 1e-12:
                        continue
                    candidate = _hard_prediction(base, components, trial) if hard else _routed_prediction(
                        spec, base, components, trial,
                        cap=float(spec.parameters["correction_cap"]) if "correction_cap" in spec.parameters else None,
                    )
                    if pareto_margin is not None and not _pareto(gold, base, candidate, pareto_margin):
                        continue
                    candidate_state = _metric_state(gold, base, candidate)
                    if _better(candidate_state, best_state):
                        best_weights, best_prediction, best_state = trial, candidate, candidate_state
                weights, prediction, state = best_weights.copy(), best_prediction, best_state
    if strict_fallback and not state["eligible"]:
        return np.zeros_like(weights), base.copy(), _metric_state(gold, base, base)
    return weights, prediction, state


def fit_route(
    spec: RouterSpec, gold: Any, base: Any, r17: Any, direct: Any, hurdle: Any, soft: Any,
) -> RouterResult:
    _validate_spec(spec)
    target = _matrix(gold, "gold")
    base_value, r17_value, direct_value, hurdle_value, soft_value = _inputs(base, r17, direct, hurdle, soft)
    if target.shape != base_value.shape:
        raise ValueError("gold and route components must have identical shapes")
    components = _component_map(r17_value, direct_value, hurdle_value, soft_value)
    fallback = False
    if spec.family == FAMILIES[0]:
        best = None
        values = _weight_values(float(spec.parameters["step"]), float(spec.parameters["max_nonidentity"]))
        for weights in product(values, repeat=3):
            if sum(weights) > float(spec.parameters["max_nonidentity"]) + 1e-12:
                continue
            prediction = _low_protected(spec, base_value, components, weights)
            state = _metric_state(target, base_value, prediction)
            key = (not state["eligible"], -state["normalized_gate_slack"], state["macro_rmse"], weights)
            if best is None or key < best[0]:
                best = (key, weights, prediction, state)
        assert best is not None
        selected, prediction, state = {"weights": best[1]}, best[2], best[3]
    elif spec.family == FAMILIES[1]:
        weights, prediction, state = _coordinate_search(spec, target, base_value, components, hard=True)
        selected = {"zone_weights": tuple(tuple(float(x) for x in row) for row in weights)}
    elif spec.family == FAMILIES[2]:
        best = None
        values = _weight_values(float(spec.parameters["step"]), float(spec.parameters["max_nonidentity"]))
        for weights in product(values, repeat=3):
            if sum(weights) > float(spec.parameters["max_nonidentity"]) + 1e-12:
                continue
            prediction = _boundary(spec, base_value, components, weights)
            state = _metric_state(target, base_value, prediction)
            key = (not state["eligible"], -state["normalized_gate_slack"], state["macro_rmse"], weights)
            if best is None or key < best[0]:
                best = (key, weights, prediction, state)
        assert best is not None
        selected, prediction, state = {"weights": best[1]}, best[2], best[3]
    elif spec.family == FAMILIES[3]:
        weights, prediction, state = _coordinate_search(
            spec, target, base_value, components, pareto_margin=float(spec.parameters["pareto_margin"]),
        )
        selected = {"zone_weights": tuple(tuple(float(x) for x in row) for row in weights)}
    else:
        weights, prediction, state = _coordinate_search(spec, target, base_value, components, strict_fallback=True)
        fallback = not state["eligible"]
        selected = {"zone_weights": tuple(tuple(float(x) for x in row) for row in weights)}
    audit = MappingProxyType({
        "cycle": spec.cycle, "family": spec.family, "variant_id": spec.variant_id,
        "eligible": state["eligible"], "normalized_gate_slack": state["normalized_gate_slack"],
        "train_macro_rmse": state["macro_rmse"], "improvements": state["improvements"],
        "identity_fallback": fallback, "score1_role": "descriptive_only", "average_target_used": False,
    })
    return RouterResult(np.asarray(prediction, dtype=np.float32), MappingProxyType(selected), audit)


def apply_route(
    spec: RouterSpec, selected_parameters: Mapping[str, Any], base: Any, r17: Any, direct: Any,
    hurdle: Any, soft: Any,
) -> np.ndarray:
    _validate_spec(spec)
    base_value, r17_value, direct_value, hurdle_value, soft_value = _inputs(base, r17, direct, hurdle, soft)
    components = _component_map(r17_value, direct_value, hurdle_value, soft_value)
    if spec.family == FAMILIES[0]:
        output = _low_protected(spec, base_value, components, tuple(selected_parameters["weights"]))
    elif spec.family == FAMILIES[1]:
        output = _hard_prediction(base_value, components, np.asarray(selected_parameters["zone_weights"], dtype=float))
    elif spec.family == FAMILIES[2]:
        output = _boundary(spec, base_value, components, tuple(selected_parameters["weights"]))
    else:
        output = _routed_prediction(
            spec, base_value, components, np.asarray(selected_parameters["zone_weights"], dtype=float),
            cap=float(spec.parameters["correction_cap"]) if "correction_cap" in spec.parameters else None,
        )
    return np.asarray(np.clip(output, 1, 5), dtype=np.float32)


__all__ = ["FAMILIES", "RouterResult", "RouterSpec", "apply_route", "fit_route", "router_specs"]
