"""Aggregate-only, fail-closed selection helpers for V5 tail learners.

This module accepts metric mappings only.  It never accepts targets,
predictions, row identifiers, or text, and score-1 is never a gate.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from .iterative_tail_metrics import AXES, metric_improvements


DEFAULT_INNER_GATE: dict[str, Any] = {
    "operator": "AND",
    "macro_rmse_min_improvement": 0.005,
    "equal_group_rmse_min_improvement": 0.010,
    "low_tail_must_improve": True,
    "high_tail_must_improve": True,
    "gold_3_4_balanced_accuracy_min_improvement": 0.010,
    "max_axis_rmse_worsening": 0.010,
    "max_macro_spearman_fall": 0.005,
    "score1_descriptive_only": True,
    "require_finite_metrics": True,
}

DEFAULT_FINAL_GATE: dict[str, Any] = {
    "operator": "AND",
    "macro_rmse_min_improvement": 0.010,
    "low_tail_must_improve": True,
    "high_tail_must_improve": True,
    "gold_3_4_balanced_accuracy_min_improvement": 0.010,
    "max_axis_rmse_worsening": 0.010,
    "max_macro_spearman_fall": 0.005,
    "score1_descriptive_only": True,
    "require_finite_metrics": True,
    "candidate_minus_baseline_ci_upper_bound": 0.0,
}

_FORBIDDEN_KEYS = {"rows", "targets", "predictions", "documents", "source_ids", "text"}


def _section(config: Mapping[str, Any], names: Sequence[str]) -> Mapping[str, Any] | None:
    for name in names:
        value = config.get(name)
        if isinstance(value, Mapping):
            return value
    return config


def _number(config: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(config[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if isfinite(value) and value >= 0.0 else None


def _valid_gate_config(config: Mapping[str, Any], *, final: bool) -> tuple[Mapping[str, Any], dict[str, float]] | None:
    names = ("final_evaluation", "outer_final_gate", "final_gate") if final else ("inner_promotion_gate", "promotion_gate")
    gate = _section(config, names)
    if not isinstance(gate, Mapping) or gate.get("operator", "AND") != "AND":
        return None
    required_true = ("low_tail_must_improve", "high_tail_must_improve", "score1_descriptive_only", "require_finite_metrics")
    if any(gate.get(key) is not True for key in required_true):
        return None
    keys = (
        "macro_rmse_min_improvement",
        *(("equal_group_rmse_min_improvement",) if not final else ()),
        "gold_3_4_balanced_accuracy_min_improvement",
        "max_axis_rmse_worsening",
        "max_macro_spearman_fall",
    )
    values = {key: _number(gate, key) for key in keys}
    if any(value is None for value in values.values()):
        return None
    return gate, {key: float(value) for key, value in values.items()}


def _improvements(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    if _FORBIDDEN_KEYS.intersection(baseline) or _FORBIDDEN_KEYS.intersection(candidate):
        return None
    try:
        delta = metric_improvements(baseline, candidate)
        flat = [delta[key] for key in ("rmse", "equal_group_rmse", "low_tail_rmse", "high_tail_rmse", "gold_3_4_balanced_accuracy", "spearman")]
        flat.extend(delta["axis_rmse"][axis] for axis in AXES)
        if any(value is None or not isfinite(float(value)) for value in flat):
            return None
        return delta
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def gate_decision(
    config: Mapping[str, Any], baseline_metrics: Mapping[str, Any], candidate_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the config-bound seven-gate inner conjunction to aggregates."""
    parsed = _valid_gate_config(config, final=False)
    delta = _improvements(baseline_metrics, candidate_metrics)
    if parsed is None or delta is None:
        return {
            "eligible": False,
            "finite_metrics": delta is not None,
            "config_valid": parsed is not None,
            "gates": {},
            "improvements": delta,
            "score1_used_for_promotion": False,
        }
    _, threshold = parsed
    gates = {
        "macro_rmse_improvement": delta["rmse"] >= threshold["macro_rmse_min_improvement"],
        "equal_group_rmse_improvement": delta["equal_group_rmse"] >= threshold["equal_group_rmse_min_improvement"],
        "low_tail_improves": delta["low_tail_rmse"] > 0.0,
        "high_tail_improves": delta["high_tail_rmse"] > 0.0,
        "gold_3_4_balanced_accuracy_improvement": delta["gold_3_4_balanced_accuracy"] >= threshold["gold_3_4_balanced_accuracy_min_improvement"],
        "axis_rmse_worsening_bound": all(value >= -threshold["max_axis_rmse_worsening"] for value in delta["axis_rmse"].values()),
        "macro_spearman_fall_bound": delta["spearman"] >= -threshold["max_macro_spearman_fall"],
    }
    return {
        "eligible": all(gates.values()),
        "finite_metrics": True,
        "config_valid": True,
        "gates": gates,
        "improvements": delta,
        "score1_used_for_promotion": False,
    }


def _spec_value(spec: Any, key: str) -> Any:
    return spec.get(key) if isinstance(spec, Mapping) else getattr(spec, key, None)


def _spec_id(spec: Any) -> str | None:
    value = _spec_value(spec, "variant_id")
    if value is None:
        value = _spec_value(spec, "id")
    return value if isinstance(value, str) and value else None


def select_candidate(
    specs: Sequence[Any],
    metrics_by_id: Mapping[str, Mapping[str, Any]],
    baseline_metrics: Mapping[str, Any],
    *,
    gate_config: Mapping[str, Any] = DEFAULT_INNER_GATE,
) -> dict[str, Any]:
    """Select minimum-RMSE eligible candidate; require all 20 registered cycles."""
    specs = tuple(specs)
    ids = [_spec_id(spec) for spec in specs]
    cycles = [_spec_value(spec, "cycle") for spec in specs]
    inventory_valid = (
        len(specs) == 20
        and all(identifier is not None for identifier in ids)
        and len(set(ids)) == 20
        and sorted(cycles) == list(range(1, 21))
        and set(metrics_by_id) == set(ids)
    )
    if not inventory_valid:
        return {
            "selected_id": "baseline",
            "selected_cycle": None,
            "fell_back_to_baseline": True,
            "inventory_valid": False,
            "eligible_ids": [],
            "decisions": [],
            "selection_rule": "baseline_relative_7gate_then_min_macro_rmse_then_cycle",
        }
    decisions = []
    eligible = []
    for spec, identifier, cycle in zip(specs, ids, cycles, strict=True):
        assert identifier is not None
        decision = gate_decision(gate_config, baseline_metrics, metrics_by_id[identifier])
        decisions.append({"variant_id": identifier, "cycle": int(cycle), **decision})
        if decision["eligible"]:
            try:
                rmse = float(metrics_by_id[identifier]["macro"]["rmse"])
            except (KeyError, TypeError, ValueError):
                continue
            if isfinite(rmse):
                eligible.append((rmse, int(cycle), identifier))
    if not eligible:
        selected_id, selected_cycle = "baseline", None
    else:
        _, selected_cycle, selected_id = min(eligible)
    return {
        "selected_id": selected_id,
        "selected_cycle": selected_cycle,
        "fell_back_to_baseline": selected_cycle is None,
        "inventory_valid": True,
        "eligible_ids": [identifier for _, _, identifier in sorted(eligible)],
        "decisions": decisions,
        "selection_rule": "baseline_relative_7gate_then_min_macro_rmse_then_cycle",
    }


def _candidate_minus_upper(bootstrap: Mapping[str, Any]) -> float | None:
    candidates = [bootstrap]
    for key in ("candidate_minus_baseline_ci", "paired_bootstrap", "bootstrap"):
        value = bootstrap.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
            nested = value.get("candidate_minus_baseline_ci")
            if isinstance(nested, Mapping):
                candidates.append(nested)
    for value in candidates:
        if "upper" in value:
            try:
                upper = float(value["upper"])
            except (TypeError, ValueError):
                continue
            return upper if isfinite(upper) else None
    return None


def final_decision(
    config: Mapping[str, Any], baseline: Mapping[str, Any], candidate: Mapping[str, Any], bootstrap: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply final macro/CI/tail/BA/axis/Spearman conjunction."""
    parsed = _valid_gate_config(config, final=True)
    delta = _improvements(baseline, candidate)
    upper = _candidate_minus_upper(bootstrap) if isinstance(bootstrap, Mapping) else None
    if parsed is None or delta is None or upper is None:
        return {
            "pass": False,
            "finite_metrics": delta is not None and upper is not None,
            "config_valid": parsed is not None,
            "gates": {},
            "improvements": delta,
            "candidate_minus_baseline_ci_upper": upper,
            "score1_used_for_promotion": False,
        }
    gate, threshold = parsed
    ci_bound = _number(gate, "candidate_minus_baseline_ci_upper_bound")
    if ci_bound is None:
        paired = gate.get("paired_bootstrap")
        if isinstance(paired, Mapping):
            raw_bound = paired.get("required_upper_bound_lt")
            try:
                ci_bound = float(raw_bound)
            except (TypeError, ValueError):
                ci_bound = None
        elif gate.get("paired_bootstrap_candidate_minus_baseline_ci_upper_below_zero") is True:
            ci_bound = 0.0
    if ci_bound is None or not isfinite(ci_bound):
        return {"pass": False, "finite_metrics": True, "config_valid": False, "gates": {}, "improvements": delta, "candidate_minus_baseline_ci_upper": upper, "score1_used_for_promotion": False}
    gates = {
        "macro_rmse_improvement": delta["rmse"] >= threshold["macro_rmse_min_improvement"],
        "candidate_minus_baseline_rmse_ci_upper_below_bound": upper < ci_bound,
        "low_tail_improves": delta["low_tail_rmse"] > 0.0,
        "high_tail_improves": delta["high_tail_rmse"] > 0.0,
        "gold_3_4_balanced_accuracy_improvement": delta["gold_3_4_balanced_accuracy"] >= threshold["gold_3_4_balanced_accuracy_min_improvement"],
        "axis_rmse_worsening_bound": all(value >= -threshold["max_axis_rmse_worsening"] for value in delta["axis_rmse"].values()),
        "macro_spearman_fall_bound": delta["spearman"] >= -threshold["max_macro_spearman_fall"],
    }
    return {
        "pass": all(gates.values()),
        "finite_metrics": True,
        "config_valid": True,
        "gates": gates,
        "improvements": delta,
        "candidate_minus_baseline_ci_upper": upper,
        "score1_used_for_promotion": False,
    }


def fold_direction_diagnostics(
    per_fold_baseline: Sequence[Mapping[str, Any]], per_fold_candidate: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Count fold-positive core improvements and material tail regressions."""
    if len(per_fold_baseline) != len(per_fold_candidate) or len(per_fold_baseline) != 5:
        return {"valid": False, "fold_count": 0, "positive_fold_counts": {}, "tail_risk_counts": {}}
    keys = ("rmse", "equal_group_rmse", "low_tail_rmse", "high_tail_rmse", "gold_3_4_balanced_accuracy")
    deltas = [_improvements(base, candidate) for base, candidate in zip(per_fold_baseline, per_fold_candidate, strict=True)]
    if any(delta is None for delta in deltas):
        return {"valid": False, "fold_count": 5, "positive_fold_counts": {}, "tail_risk_counts": {}}
    complete = [delta for delta in deltas if delta is not None]
    return {
        "valid": True,
        "fold_count": 5,
        "positive_fold_counts": {key: sum(delta[key] > 0.0 for delta in complete) for key in keys},
        "tail_risk_counts": {
            "low_tail_below_minus_0_005": sum(delta["low_tail_rmse"] < -0.005 for delta in complete),
            "high_tail_below_minus_0_005": sum(delta["high_tail_rmse"] < -0.005 for delta in complete),
        },
        "all_core_positive_at_least_4_of_5": all(sum(delta[key] > 0.0 for delta in complete) >= 4 for key in keys),
        "any_material_tail_risk": any(
            delta[key] < -0.005 for delta in complete for key in ("low_tail_rmse", "high_tail_rmse")
        ),
        "score1_used_for_promotion": False,
    }


__all__ = [
    "DEFAULT_FINAL_GATE", "DEFAULT_INNER_GATE", "final_decision", "fold_direction_diagnostics",
    "gate_decision", "select_candidate",
]
