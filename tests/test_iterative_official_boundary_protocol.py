import copy
import json
import unittest

from mal2026.iterative_official_boundary_protocol import (
    CONFIG_PATH, OfficialBoundaryProtocolError, load_protocol, validate_bound_inputs, validate_protocol_mapping,
)


class OfficialBoundaryProtocolTest(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_canonical_protocol_and_bound_inputs(self):
        audit = validate_bound_inputs(load_protocol())
        self.assertEqual(audit.records, 2000)
        self.assertEqual(audit.folds, {0: 400, 1: 400, 2: 400, 3: 400, 4: 400})
        self.assertEqual(audit.official_candidates, 6000)

    def test_no_v8_full_oof_prestudy_and_exact_gate(self):
        self.assertTrue(self.raw["scientific_claims"]["no_v8_full_oof_prestudy_before_preregistration"])
        self.assertEqual(self.raw["inner_promotion_gate"]["gold_3_4_balanced_accuracy_min_improvement"], 0.01)
        self.assertNotIn("macro_score5_recall_min_improvement", self.raw["inner_promotion_gate"])

    def test_candidate_and_gate_drift_fail_closed(self):
        drift = copy.deepcopy(self.raw); drift["candidates"][0]["nudge"] = 0.16
        with self.assertRaises(OfficialBoundaryProtocolError):
            validate_protocol_mapping(drift)
        drift = copy.deepcopy(self.raw); drift["inner_promotion_gate"]["gold_3_4_balanced_accuracy_min_improvement"] = 0.009
        with self.assertRaises(OfficialBoundaryProtocolError):
            validate_protocol_mapping(drift)

    def test_v7_is_frozen_not_reopened(self):
        freeze = self.raw["authorization_and_freeze"]
        self.assertFalse(freeze["v7_reopening"])
        self.assertTrue(freeze["v7_inventory_and_artifacts_frozen"])
        self.assertTrue(freeze["v7_gates_unchanged"])


if __name__ == "__main__":
    unittest.main()
