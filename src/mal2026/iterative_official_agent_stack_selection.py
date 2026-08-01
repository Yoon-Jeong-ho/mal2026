"""Aggregate-only exact-gate selection for three frozen V7 candidates."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from .iterative_tail_learner_selection import final_decision, fold_direction_diagnostics, gate_decision
from .iterative_tail_metrics import AXES


_RAW_KEYS = {"rows", "targets", "predictions", "source_id", "source_ids", "essay", "text", "rationale"}


def _contains_raw(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if _RAW_KEYS.intersection(value):
        return True
    return any(_contains_raw(child) for child in value.values())


def score5_macro_recall(metrics: Mapping[str, Any]) -> float | None:
    if _contains_raw(metrics):
        return None
    try:
        values = [float(metrics["axes"][axis]["bands"]["5"]["recall"]) for axis in AXES]
    except (KeyError, TypeError, ValueError):
        return None
    return sum(values) / 3.0 if all(isfinite(value) for value in values) else None


def select_candidate(
    specs: Sequence[Any],
    metrics_by_id: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    specs = tuple(specs)
    identifiers = [getattr(spec, "variant_id", None) for spec in specs]
    cycles = [getattr(spec, "cycle", None) for spec in specs]
    valid = (
        len(specs) == 3 and all(isinstance(value, str) and value for value in identifiers)
        and len(set(identifiers)) == 3 and sorted(cycles) == [1, 2, 3]
        and set(metrics_by_id) == set(identifiers) and not _contains_raw(baseline)
        and not any(_contains_raw(metrics) for metrics in metrics_by_id.values())
    )
    if not valid:
        return {
            "selected_id": "baseline", "selected_cycle": None, "fell_back_to_baseline": True,
            "inventory_valid": False, "eligible_ids": [], "decisions": [],
            "selection_rule": "baseline_relative_original_7gate_then_min_macro_rmse_then_cycle",
        }
    decisions, eligible = [], []
    for spec in specs:
        metrics = metrics_by_id[spec.variant_id]
        decision = gate_decision(config, baseline, metrics)
        base_recall, candidate_recall = score5_macro_recall(baseline), score5_macro_recall(metrics)
        recall_gain = None if base_recall is None or candidate_recall is None else candidate_recall - base_recall
        decisions.append({"variant_id": spec.variant_id, "cycle": spec.cycle, **decision,
                          "score5_macro_recall_gain_descriptive": recall_gain})
        if decision["eligible"]:
            eligible.append((float(metrics["macro"]["rmse"]), spec.cycle, spec.variant_id))
    selected = min(eligible) if eligible else None
    return {
        "selected_id": selected[2] if selected else "baseline",
        "selected_cycle": selected[1] if selected else None,
        "fell_back_to_baseline": selected is None,
        "inventory_valid": True,
        "eligible_ids": [identifier for _, _, identifier in sorted(eligible)],
        "decisions": decisions,
        "selection_rule": "baseline_relative_original_7gate_then_min_macro_rmse_then_cycle",
    }


def final_gate(
    config: Mapping[str, Any], baseline: Mapping[str, Any], candidate: Mapping[str, Any], bootstrap: Mapping[str, Any]
) -> dict[str, Any]:
    if any(_contains_raw(value) for value in (baseline, candidate, bootstrap)):
        return {"pass": False, "config_valid": False, "finite_metrics": False, "gates": {}}
    result = final_decision(config, baseline, candidate, bootstrap)
    base_recall, candidate_recall = score5_macro_recall(baseline), score5_macro_recall(candidate)
    return {**result, "score5_macro_recall_gain_descriptive": (
        None if base_recall is None or candidate_recall is None else candidate_recall - base_recall
    )}


__all__ = [
    "final_gate", "fold_direction_diagnostics", "score5_macro_recall", "select_candidate",
]
