from __future__ import annotations

from copy import deepcopy
import unittest

from mal2026.iterative_official_rationale_semantic_models import candidate_specs
from mal2026.iterative_official_rationale_semantic_selection import final_gate, select_candidate


INNER = {
    "operator": "AND", "macro_rmse_min_improvement": .005,
    "equal_group_rmse_min_improvement": .010, "low_tail_must_improve": True,
    "high_tail_must_improve": True, "gold_3_4_balanced_accuracy_min_improvement": .010,
    "max_axis_rmse_worsening": .010, "max_macro_spearman_fall": .005,
    "score1_descriptive_only": True, "require_finite_metrics": True,
}
FINAL = {
    "operator": "AND", "macro_rmse_min_improvement": .010,
    "low_tail_must_improve": True, "high_tail_must_improve": True,
    "gold_3_4_balanced_accuracy_min_improvement": .010,
    "max_axis_rmse_worsening": .010, "max_macro_spearman_fall": .005,
    "score1_descriptive_only": True, "require_finite_metrics": True,
    "candidate_minus_baseline_ci_upper_bound": 0.0,
}
CONFIG = {"inner_promotion_gate": INNER, "final_evaluation": FINAL}


def metrics(rmse=.60, equal=.80, low=1.0, high=1.0, ba=.50, spearman=.60, recall=.10):
    return {
        "macro": {"rmse": rmse, "equal_group_rmse": equal, "low_tail_rmse": low,
                  "high_tail_rmse": high, "gold_3_4_balanced_accuracy": ba, "spearman": spearman},
        "axes": {axis: {"rmse": rmse, "bands": {"5": {"recall": recall}}}
                 for axis in ("content", "organization", "expression")},
    }


class RationaleSemanticSelectionTest(unittest.TestCase):
    def setUp(self):
        self.specs = candidate_specs()
        self.baseline = metrics()
        self.good = metrics(.59, .78, .98, .98, .52, .61, recall=0.0)

    def test_unchanged_seven_gates_and_rmse_cycle_tie(self):
        values = {spec.variant_id: self.baseline for spec in self.specs}
        values[self.specs[0].variant_id] = self.good
        values[self.specs[1].variant_id] = self.good
        result = select_candidate(self.specs, values, self.baseline, CONFIG)
        self.assertEqual(self.specs[0].variant_id, result["selected_id"])
        self.assertEqual(7, len(result["decisions"][0]["gates"]))
        self.assertLess(result["decisions"][0]["score5_macro_recall_gain_descriptive"], 0)

    def test_config_drift_nonfinite_raw_and_incomplete_fail_closed(self):
        drift = deepcopy(CONFIG); drift["inner_promotion_gate"]["operator"] = "OR"
        values = {spec.variant_id: self.good for spec in self.specs}
        self.assertTrue(select_candidate(self.specs, values, self.baseline, drift)["fell_back_to_baseline"])
        nonfinite = metrics(.59, .78, .98, .98, .52, float("nan"))
        values[self.specs[0].variant_id] = nonfinite
        values[self.specs[1].variant_id] = self.baseline
        values[self.specs[2].variant_id] = self.baseline
        self.assertTrue(select_candidate(self.specs, values, self.baseline, CONFIG)["fell_back_to_baseline"])
        raw = dict(self.good); raw["rationale"] = "forbidden"
        self.assertFalse(select_candidate(self.specs, {spec.variant_id: raw for spec in self.specs}, self.baseline, CONFIG)["inventory_valid"])
        self.assertFalse(select_candidate(self.specs, {self.specs[0].variant_id: self.good}, self.baseline, CONFIG)["inventory_valid"])

    def test_no_eligible_fallback_and_final_ci_direction(self):
        result = select_candidate(
            self.specs, {spec.variant_id: self.baseline for spec in self.specs}, self.baseline, CONFIG
        )
        self.assertTrue(result["fell_back_to_baseline"])
        self.assertTrue(final_gate(CONFIG, self.baseline, self.good, {"candidate_minus_baseline_ci": {"upper": -.001}})["pass"])
        self.assertFalse(final_gate(CONFIG, self.baseline, self.good, {"candidate_minus_baseline_ci": {"upper": 0.0}})["pass"])


if __name__ == "__main__":
    unittest.main()
