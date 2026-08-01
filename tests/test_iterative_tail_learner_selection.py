"""Aggregate-only unit tests for V5 learner selection."""
from __future__ import annotations

from copy import deepcopy
import math
import unittest

from mal2026.iterative_tail_learner_selection import (
    DEFAULT_FINAL_GATE,
    DEFAULT_INNER_GATE,
    final_decision,
    fold_direction_diagnostics,
    gate_decision,
    select_candidate,
)


def metrics(rmse=.60, equal=.80, low=1.0, high=1.0, ba=.50, spearman=.60, axes=None, score1=1.0):
    axes = axes or (.60, .60, .60)
    return {
        "macro": {
            "rmse": rmse, "equal_group_rmse": equal, "low_tail_rmse": low,
            "high_tail_rmse": high, "gold_3_4_balanced_accuracy": ba,
            "spearman": spearman, "score1_descriptive_rmse": score1,
        },
        "axes": {axis: {"rmse": value} for axis, value in zip(("content", "organization", "expression"), axes, strict=True)},
    }


def passing(rmse=.594):
    return metrics(rmse=rmse, equal=.789, low=.99, high=.99, ba=.511, spearman=.596, axes=(.594, .594, .594))


class LearnerSelectionTests(unittest.TestCase):
    def test_gate_is_config_bound_strict_and_score1_blind(self):
        baseline = metrics(score1=0.0)
        candidate = passing(); candidate["macro"]["score1_descriptive_rmse"] = 999.0
        decision = gate_decision(DEFAULT_INNER_GATE, baseline, candidate)
        self.assertTrue(decision["eligible"])
        self.assertEqual(7, len(decision["gates"]))
        self.assertFalse(decision["score1_used_for_promotion"])

        stricter = deepcopy(DEFAULT_INNER_GATE); stricter["macro_rmse_min_improvement"] = .007
        self.assertFalse(gate_decision(stricter, baseline, candidate)["eligible"])
        invalid = deepcopy(DEFAULT_INNER_GATE); invalid["operator"] = "OR"
        rejected = gate_decision(invalid, baseline, candidate)
        self.assertFalse(rejected["eligible"]); self.assertFalse(rejected["config_valid"])

    def test_nonfinite_and_raw_shaped_inputs_fail_closed(self):
        candidate = passing(); candidate["macro"]["rmse"] = math.nan
        self.assertFalse(gate_decision(DEFAULT_INNER_GATE, metrics(), candidate)["eligible"])
        raw = {**passing(), "predictions": [[1, 2, 3]]}
        self.assertFalse(gate_decision(DEFAULT_INNER_GATE, metrics(), raw)["eligible"])

    def test_selection_requires_all_twenty_and_ties_by_cycle(self):
        specs = [{"variant_id": f"v{cycle}", "cycle": cycle} for cycle in range(1, 21)]
        values = {f"v{cycle}": metrics() for cycle in range(1, 21)}
        values["v7"] = passing(.590); values["v3"] = passing(.590); values["v12"] = passing(.589)
        result = select_candidate(specs, values, metrics())
        self.assertEqual("v12", result["selected_id"])
        values["v12"] = metrics()
        result = select_candidate(specs, values, metrics())
        self.assertEqual("v3", result["selected_id"])
        self.assertFalse(result["fell_back_to_baseline"])

        missing = select_candidate(specs[:-1], {key: value for key, value in values.items() if key != "v20"}, metrics())
        self.assertEqual("baseline", missing["selected_id"]); self.assertFalse(missing["inventory_valid"])

        stricter = deepcopy(DEFAULT_INNER_GATE)
        stricter["macro_rmse_min_improvement"] = .012
        rejected = select_candidate(specs, values, metrics(), gate_config=stricter)
        self.assertEqual("baseline", rejected["selected_id"])

    def test_none_eligible_falls_back_to_baseline(self):
        specs = [{"id": f"c{cycle}", "cycle": cycle} for cycle in range(1, 21)]
        result = select_candidate(specs, {f"c{cycle}": metrics() for cycle in range(1, 21)}, metrics())
        self.assertTrue(result["fell_back_to_baseline"]); self.assertEqual("baseline", result["selected_id"])

    def test_final_gate_uses_candidate_minus_baseline_upper_direction(self):
        candidate = passing(.589)
        good = final_decision(DEFAULT_FINAL_GATE, metrics(), candidate, {"candidate_minus_baseline_ci": {"upper": -.0001}})
        self.assertTrue(good["pass"]); self.assertEqual(7, len(good["gates"]))
        boundary = final_decision(DEFAULT_FINAL_GATE, metrics(), candidate, {"candidate_minus_baseline_ci": {"upper": 0.0}})
        self.assertFalse(boundary["pass"])
        self.assertFalse(boundary["gates"]["candidate_minus_baseline_rmse_ci_upper_below_bound"])

    def test_fold_direction_counts_core_and_tail_risk(self):
        baseline = [metrics() for _ in range(5)]
        candidate = [passing() for _ in range(5)]
        candidate[1] = metrics(rmse=.59, equal=.78, low=1.006, high=.99, ba=.52, spearman=.60, axes=(.59, .59, .59))
        result = fold_direction_diagnostics(baseline, candidate)
        self.assertTrue(result["valid"])
        self.assertEqual(4, result["positive_fold_counts"]["low_tail_rmse"])
        self.assertEqual(1, result["tail_risk_counts"]["low_tail_below_minus_0_005"])
        self.assertTrue(result["all_core_positive_at_least_4_of_5"])
        self.assertTrue(result["any_material_tail_risk"])


if __name__ == "__main__":
    unittest.main()
