import json
import unittest

from mal2026.iterative_official_balanced_boundary_models import candidate_specs
from mal2026.iterative_official_balanced_boundary_protocol import CONFIG_PATH
from mal2026.iterative_official_balanced_boundary_selection import final_gate, score5_macro_recall, select_candidate


def metrics(rmse=.57, equal=.70, low=.93, high=.89, ba=.64, spearman=.60):
    return {
        "macro": {"rmse": rmse, "equal_group_rmse": equal, "low_tail_rmse": low,
                  "high_tail_rmse": high, "gold_3_4_balanced_accuracy": ba, "spearman": spearman},
        "axes": {
            axis: {"rmse": rmse, "bands": {"5": {"recall": recall}}}
            for axis, recall in zip(("content", "organization", "expression"), (0.0, 0.1, 0.2), strict=True)
        },
    }


class OfficialBalancedBoundarySelectionTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.specs = candidate_specs()
        self.baseline = metrics()

    def test_original_seven_gate_selects_lowest_rmse_eligible(self):
        good = metrics(.55, .68, .90, .85, .66, .62)
        better = metrics(.54, .67, .89, .84, .66, .63)
        result = select_candidate(self.specs, {
            self.specs[0].variant_id: good, self.specs[1].variant_id: better,
            self.specs[2].variant_id: self.baseline,
        }, self.baseline, self.config)
        self.assertEqual(self.specs[1].variant_id, result["selected_id"])
        self.assertFalse(result["fell_back_to_baseline"])

    def test_incomplete_inventory_fails_closed(self):
        result = select_candidate(self.specs, {self.specs[0].variant_id: self.baseline}, self.baseline, self.config)
        self.assertTrue(result["fell_back_to_baseline"])
        self.assertFalse(result["inventory_valid"])

    def test_score5_recall_is_descriptive_and_raw_fields_fail_closed(self):
        candidate = metrics(.55, .68, .90, .85, .66, .62)
        for axis in candidate["axes"].values():
            axis["bands"]["5"]["recall"] = 0.0
        result = select_candidate(self.specs, {spec.variant_id: candidate for spec in self.specs}, self.baseline, self.config)
        self.assertFalse(result["fell_back_to_baseline"])
        self.assertLess(result["decisions"][0]["score5_macro_recall_gain_descriptive"], 0)
        candidate["source_ids"] = ["private"]
        result = select_candidate(self.specs, {spec.variant_id: candidate for spec in self.specs}, self.baseline, self.config)
        self.assertFalse(result["inventory_valid"])
        self.assertIsNone(score5_macro_recall(candidate))

    def test_final_gate_requires_strict_negative_ci_upper(self):
        candidate = metrics(.55, .68, .90, .85, .66, .62)
        self.assertTrue(final_gate(self.config, self.baseline, candidate, {"candidate_minus_baseline_ci": {"upper": -.001}})["pass"])
        self.assertFalse(final_gate(self.config, self.baseline, candidate, {"candidate_minus_baseline_ci": {"upper": 0.0}})["pass"])


if __name__ == "__main__":
    unittest.main()
