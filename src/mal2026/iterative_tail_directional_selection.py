"""Aggregate-only directional selection helpers for the fixed V6 candidates."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from .iterative_tail_learner_selection import final_decision, gate_decision
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
    "macro_score5_recall_min_improvement": 0.010,
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
    "macro_score5_recall_min_improvement": 0.010,
    "score1_descriptive_only": True,
    "require_finite_metrics": True,
    "candidate_minus_baseline_ci_upper_bound": 0.0,
}

_RAW_KEYS = {
    "rows", "predictions", "targets", "id", "ids", "source_id", "source_ids",
    "document_id", "document_ids", "text",
}


def _contains_raw(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if _RAW_KEYS.intersection(value):
        return True
    return any(_contains_raw(child) for child in value.values())


def _gate_section(config: Mapping[str, Any], *, final: bool) -> Mapping[str, Any] | None:
    names = ("final_evaluation", "outer_final_gate", "final_gate") if final else ("inner_promotion_gate", "promotion_gate")
    for name in names:
        section = config.get(name)
        if isinstance(section, Mapping):
            return section
    return config if isinstance(config, Mapping) else None


def _score5_threshold(config: Mapping[str, Any], *, final: bool) -> float | None:
    section = _gate_section(config, final=final)
    if section is None or section.get("score1_descriptive_only") is not True:
        return None
    try:
        threshold = float(section["macro_score5_recall_min_improvement"])
    except (KeyError, TypeError, ValueError):
        return None
    return threshold if isfinite(threshold) and threshold >= 0.0 else None


def score5_macro_recall(metrics: Mapping[str, Any]) -> float | None:
    """Return the finite mean true-band-5 recall across the three axes."""
    if not isinstance(metrics, Mapping) or _contains_raw(metrics):
        return None
    try:
        values = [float(metrics["axes"][axis]["bands"]["5"]["recall"]) for axis in AXES]
    except (KeyError, TypeError, ValueError):
        return None
    return sum(values) / len(values) if all(isfinite(value) for value in values) else None


def inner_gate(
    config: Mapping[str, Any], baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the original seven inner gates plus score-5 recall improvement."""
    threshold = _score5_threshold(config, final=False)
    baseline_recall = score5_macro_recall(baseline)
    candidate_recall = score5_macro_recall(candidate)
    base = gate_decision(config, baseline, candidate)
    recall_gain = None if baseline_recall is None or candidate_recall is None else candidate_recall - baseline_recall
    config_valid = base["config_valid"] and threshold is not None
    finite_metrics = base["finite_metrics"] and recall_gain is not None and isfinite(recall_gain)
    gates = dict(base["gates"]) if config_valid and finite_metrics else {}
    if gates:
        gates["score5_macro_recall_improvement"] = recall_gain >= threshold - 1e-12
    return {
        **base,
        "eligible": bool(gates) and all(gates.values()),
        "config_valid": config_valid,
        "finite_metrics": finite_metrics,
        "gates": gates,
        "score5_macro_recall_gain": recall_gain,
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
    baseline: Mapping[str, Any],
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require all three registered candidates, then select RMSE/cycle or baseline."""
    specs = tuple(specs)
    identifiers = [_spec_id(spec) for spec in specs]
    cycles = [_spec_value(spec, "cycle") for spec in specs]
    inventory_valid = (
        len(specs) == 3
        and all(identifier is not None for identifier in identifiers)
        and len(set(identifiers)) == 3
        and sorted(cycles) == [1, 2, 3]
        and isinstance(metrics_by_id, Mapping)
        and set(metrics_by_id) == set(identifiers)
    )
    if not inventory_valid:
        return {
            "selected_id": "baseline", "selected_cycle": None, "fell_back_to_baseline": True,
            "inventory_valid": False, "eligible_ids": [], "decisions": [],
            "selection_rule": "baseline_relative_8gate_then_min_macro_rmse_then_cycle",
        }
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[float, int, str]] = []
    for spec, identifier, cycle in zip(specs, identifiers, cycles, strict=True):
        assert identifier is not None
        decision = inner_gate(gate_config, baseline, metrics_by_id[identifier])
        decisions.append({"variant_id": identifier, "cycle": int(cycle), **decision})
        if decision["eligible"]:
            try:
                rmse = float(metrics_by_id[identifier]["macro"]["rmse"])
            except (KeyError, TypeError, ValueError):
                continue
            if isfinite(rmse):
                eligible.append((rmse, int(cycle), identifier))
    if eligible:
        _, selected_cycle, selected_id = min(eligible)
    else:
        selected_id, selected_cycle = "baseline", None
    return {
        "selected_id": selected_id,
        "selected_cycle": selected_cycle,
        "fell_back_to_baseline": selected_cycle is None,
        "inventory_valid": True,
        "eligible_ids": [identifier for _, _, identifier in sorted(eligible)],
        "decisions": decisions,
        "selection_rule": "baseline_relative_8gate_then_min_macro_rmse_then_cycle",
    }


def final_gate(
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the original final seven gates plus score-5 recall improvement."""
    threshold = _score5_threshold(config, final=True)
    baseline_recall = score5_macro_recall(baseline)
    candidate_recall = score5_macro_recall(candidate)
    base = final_decision(config, baseline, candidate, bootstrap)
    recall_gain = None if baseline_recall is None or candidate_recall is None else candidate_recall - baseline_recall
    config_valid = base["config_valid"] and threshold is not None
    finite_metrics = base["finite_metrics"] and recall_gain is not None and isfinite(recall_gain)
    gates = dict(base["gates"]) if config_valid and finite_metrics else {}
    if gates:
        gates["score5_macro_recall_improvement"] = recall_gain >= threshold - 1e-12
    return {
        **base,
        "pass": bool(gates) and all(gates.values()),
        "config_valid": config_valid,
        "finite_metrics": finite_metrics,
        "gates": gates,
        "score5_macro_recall_gain": recall_gain,
        "score1_used_for_promotion": False,
    }


def fold_diagnostics(
    per_fold_baseline: Sequence[Mapping[str, Any]], per_fold_candidate: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Count positive core and score-5 directions across exactly five folds."""
    if len(per_fold_baseline) != 5 or len(per_fold_candidate) != 5:
        return {"valid": False, "fold_count": 0, "positive_fold_counts": {}, "tail_risk_counts": {}}
    records = []
    for baseline, candidate in zip(per_fold_baseline, per_fold_candidate, strict=True):
        if _contains_raw(baseline) or _contains_raw(candidate):
            return {"valid": False, "fold_count": 5, "positive_fold_counts": {}, "tail_risk_counts": {}}
        try:
            delta = metric_improvements(baseline, candidate)
        except (KeyError, TypeError, ValueError, OverflowError):
            return {"valid": False, "fold_count": 5, "positive_fold_counts": {}, "tail_risk_counts": {}}
        base_recall, candidate_recall = score5_macro_recall(baseline), score5_macro_recall(candidate)
        values = [delta[key] for key in ("rmse", "equal_group_rmse", "low_tail_rmse", "high_tail_rmse", "gold_3_4_balanced_accuracy")]
        if base_recall is None or candidate_recall is None or any(value is None or not isfinite(float(value)) for value in values):
            return {"valid": False, "fold_count": 5, "positive_fold_counts": {}, "tail_risk_counts": {}}
        records.append({**{key: float(delta[key]) for key in ("rmse", "equal_group_rmse", "low_tail_rmse", "high_tail_rmse", "gold_3_4_balanced_accuracy")}, "score5_macro_recall": candidate_recall - base_recall})
    keys = ("rmse", "equal_group_rmse", "low_tail_rmse", "high_tail_rmse", "gold_3_4_balanced_accuracy", "score5_macro_recall")
    return {
        "valid": True,
        "fold_count": 5,
        "positive_fold_counts": {key: sum(record[key] > 0.0 for record in records) for key in keys},
        "tail_risk_counts": {
            "low_tail_below_minus_0_005": sum(record["low_tail_rmse"] < -0.005 for record in records),
            "high_tail_below_minus_0_005": sum(record["high_tail_rmse"] < -0.005 for record in records),
        },
        "all_directions_positive_at_least_4_of_5": all(sum(record[key] > 0.0 for record in records) >= 4 for key in keys),
        "score1_used_for_promotion": False,
    }


__all__ = [
    "DEFAULT_FINAL_GATE", "DEFAULT_INNER_GATE", "final_gate", "fold_diagnostics",
    "inner_gate", "score5_macro_recall", "select_candidate",
]
