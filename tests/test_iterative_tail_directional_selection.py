"""Aggregate-only tests for fixed V6 directional selection."""
from __future__ import annotations

from copy import deepcopy
import math
import unittest

from mal2026.iterative_tail_directional_selection import (
    DEFAULT_FINAL_GATE,
    DEFAULT_INNER_GATE,
    final_gate,
    fold_diagnostics,
    inner_gate,
    score5_macro_recall,
    select_candidate,
)


AXES = ("content", "organization", "expression")


def metrics(rmse=.60, equal=.80, low=1.0, high=1.0, ba=.50, spearman=.60, recall=.10, axes=None, score1=1.0):
    axes = axes or (.60, .60, .60)
    return {
        "macro": {
            "rmse": rmse, "equal_group_rmse": equal, "low_tail_rmse": low,
            "high_tail_rmse": high, "gold_3_4_balanced_accuracy": ba,
            "spearman": spearman, "score1_descriptive_rmse": score1,
        },
        "axes": {
            axis: {"rmse": value, "bands": {"5": {"recall": recall}}}
            for axis, value in zip(AXES, axes, strict=True)
        },
    }


def passing(rmse=.594, recall=.111):
    return metrics(rmse=rmse, equal=.789, low=.99, high=.99, ba=.511, spearman=.596, recall=recall, axes=(.594, .594, .594))


class DirectionalSelectionTests(unittest.TestCase):
    def test_score5_macro_recall_is_axis_mean_and_finite(self):
        value = metrics()
        for axis, recall in zip(AXES, (.0, .3, .6), strict=True):
            value["axes"][axis]["bands"]["5"]["recall"] = recall
        self.assertAlmostEqual(.3, score5_macro_recall(value))
        value["axes"]["content"]["bands"]["5"]["recall"] = math.nan
        self.assertIsNone(score5_macro_recall(value))

    def test_inner_gate_direction_boundary_config_and_score1(self):
        baseline = metrics(score1=0.0)
        candidate = passing(recall=.11); candidate["macro"]["score1_descriptive_rmse"] = 999.0
        decision = inner_gate(DEFAULT_INNER_GATE, baseline, candidate)
        self.assertTrue(decision["eligible"])
        self.assertEqual(8, len(decision["gates"])); self.assertFalse(decision["score1_used_for_promotion"])

        below = passing(recall=.109999)
        self.assertFalse(inner_gate(DEFAULT_INNER_GATE, baseline, below)["gates"]["score5_macro_recall_improvement"])
        drift = deepcopy(DEFAULT_INNER_GATE); drift.pop("macro_score5_recall_min_improvement")
        rejected = inner_gate(drift, baseline, candidate)
        self.assertFalse(rejected["eligible"]); self.assertFalse(rejected["config_valid"])

    def test_nonfinite_and_raw_inputs_fail_closed(self):
        candidate = passing(); candidate["macro"]["rmse"] = math.inf
        self.assertFalse(inner_gate(DEFAULT_INNER_GATE, metrics(), candidate)["eligible"])
        raw = {**passing(), "targets": [[1, 2, 3]], "source_id": "forbidden"}
        self.assertFalse(inner_gate(DEFAULT_INNER_GATE, metrics(), raw)["eligible"])

    def test_exact_three_barrier_selection_and_fallback(self):
        specs = [{"variant_id": f"v{cycle}", "cycle": cycle} for cycle in (1, 2, 3)]
        values = {"v1": passing(.590), "v2": passing(.590), "v3": passing(.589)}
        result = select_candidate(specs, values, metrics(), DEFAULT_INNER_GATE)
        self.assertEqual("v3", result["selected_id"])
        values["v3"] = metrics()
        self.assertEqual("v1", select_candidate(specs, values, metrics(), DEFAULT_INNER_GATE)["selected_id"])
        missing = select_candidate(specs[:2], {"v1": values["v1"], "v2": values["v2"]}, metrics(), DEFAULT_INNER_GATE)
        self.assertEqual("baseline", missing["selected_id"]); self.assertFalse(missing["inventory_valid"])
        none = select_candidate(specs, {key: metrics() for key in values}, metrics(), DEFAULT_INNER_GATE)
        self.assertTrue(none["fell_back_to_baseline"])

    def test_final_bootstrap_direction_and_score5_boundary(self):
        bootstrap = {"candidate_minus_baseline_ci": {"upper": -.0001}}
        self.assertTrue(final_gate(DEFAULT_FINAL_GATE, metrics(), passing(.589), bootstrap)["pass"])
        at_zero = final_gate(DEFAULT_FINAL_GATE, metrics(), passing(.589), {"candidate_minus_baseline_ci": {"upper": 0.0}})
        self.assertFalse(at_zero["pass"])
        recall_miss = final_gate(DEFAULT_FINAL_GATE, metrics(), passing(.589, recall=.109), bootstrap)
        self.assertFalse(recall_miss["gates"]["score5_macro_recall_improvement"])

    def test_fold_diagnostics_counts_score5_and_tail_risk(self):
        baseline = [metrics() for _ in range(5)]
        candidate = [passing() for _ in range(5)]
        candidate[2] = passing(recall=.09)
        candidate[2]["macro"]["low_tail_rmse"] = 1.006
        result = fold_diagnostics(baseline, candidate)
        self.assertTrue(result["valid"])
        self.assertEqual(4, result["positive_fold_counts"]["score5_macro_recall"])
        self.assertEqual(1, result["tail_risk_counts"]["low_tail_below_minus_0_005"])
        self.assertTrue(result["all_directions_positive_at_least_4_of_5"])


if __name__ == "__main__":
    unittest.main()
