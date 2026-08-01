import copy
import json
import unittest

from mal2026.iterative_official_agent_stack_models import candidate_specs
from mal2026.iterative_official_agent_stack_protocol import CONFIG_PATH
from mal2026.iterative_official_agent_stack_selection import final_gate, score5_macro_recall, select_candidate


def metrics(rmse=0.57, equal=0.70, low=0.93, high=0.89, ba=0.64, spearman=0.60):
    return {
        "macro": {"rmse": rmse, "equal_group_rmse": equal, "low_tail_rmse": low,
                  "high_tail_rmse": high, "gold_3_4_balanced_accuracy": ba, "spearman": spearman},
        "axes": {
            axis: {"rmse": rmse, "bands": {"5": {"recall": recall}}}
            for axis, recall in zip(("content", "organization", "expression"), (0.0, 0.1, 0.2))
        },
    }


class OfficialAgentStackSelectionTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.specs = candidate_specs()
        self.baseline = metrics()

    def test_original_seven_gate_selects_lowest_rmse_eligible(self):
        good = metrics(0.55, 0.68, 0.90, 0.85, 0.66, 0.62)
        better = metrics(0.54, 0.67, 0.89, 0.84, 0.66, 0.63)
        result = select_candidate(self.specs, {
            self.specs[0].variant_id: good, self.specs[1].variant_id: better,
            self.specs[2].variant_id: self.baseline,
        }, self.baseline, self.config)
        self.assertEqual(result["selected_id"], self.specs[1].variant_id)
        self.assertFalse(result["fell_back_to_baseline"])

    def test_incomplete_inventory_falls_back(self):
        result = select_candidate(self.specs, {self.specs[0].variant_id: self.baseline}, self.baseline, self.config)
        self.assertTrue(result["fell_back_to_baseline"])
        self.assertFalse(result["inventory_valid"])

    def test_score5_recall_is_descriptive_not_gate(self):
        candidate = metrics(0.55, 0.68, 0.90, 0.85, 0.66, 0.62)
        for axis in candidate["axes"].values():
            axis["bands"]["5"]["recall"] = 0.0
        result = select_candidate(self.specs, {spec.variant_id: candidate for spec in self.specs}, self.baseline, self.config)
        self.assertFalse(result["fell_back_to_baseline"])
        self.assertLess(result["decisions"][0]["score5_macro_recall_gain_descriptive"], 0)

    def test_final_gate_requires_macro_and_strict_negative_ci_upper(self):
        candidate = metrics(0.55, 0.68, 0.90, 0.85, 0.66, 0.62)
        passed = final_gate(self.config, self.baseline, candidate, {"candidate_minus_baseline_ci": {"upper": -0.001}})
        self.assertTrue(passed["pass"])
        failed = final_gate(self.config, self.baseline, candidate, {"candidate_minus_baseline_ci": {"upper": 0.0}})
        self.assertFalse(failed["pass"])

    def test_raw_fields_fail_closed(self):
        candidate = metrics(0.55, 0.68, 0.90, 0.85, 0.66, 0.62)
        candidate["source_ids"] = ["private"]
        result = select_candidate(self.specs, {spec.variant_id: candidate for spec in self.specs}, self.baseline, self.config)
        self.assertFalse(result["inventory_valid"])
        self.assertIsNone(score5_macro_recall(candidate))


if __name__ == "__main__":
    unittest.main()
