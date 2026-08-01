import copy
import json
import unittest

from mal2026.iterative_official_balanced_boundary_protocol import (
    CONFIG_PATH,
    OfficialBalancedBoundaryProtocolError,
    load_protocol,
    validate_bound_inputs,
    validate_protocol_mapping,
)


class OfficialBalancedBoundaryProtocolTest(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_canonical_protocol_and_bound_inputs(self):
        audit = validate_bound_inputs(load_protocol())
        self.assertEqual(audit.records, 2000)
        self.assertEqual(audit.folds, {0: 400, 1: 400, 2: 400, 3: 400, 4: 400})
        self.assertEqual(audit.terra_candidates, 6000)
        self.assertEqual(audit.luna_candidates, 6000)
        self.assertEqual(audit.v10_aggregate_sha256, "6a15396f23f10fe75ff5fd6d88e2542bb5d2898ada04a6b4c79bb123e524df9d")

    def test_exact_gate_and_no_full_oof_prestudy(self):
        self.assertTrue(self.raw["scientific_claims"]["no_v11_full_oof_prestudy_before_preregistration"])
        gate = self.raw["inner_promotion_gate"]
        self.assertEqual(gate["gold_3_4_balanced_accuracy_min_improvement"], 0.01)
        self.assertNotIn("macro_score5_recall_min_improvement", gate)
        self.assertTrue(gate["score1_descriptive_only"])

    def test_candidate_gate_and_weighting_drift_fail_closed(self):
        drift = copy.deepcopy(self.raw)
        drift["candidates"][0]["confidence"] = 0.51
        with self.assertRaises(OfficialBalancedBoundaryProtocolError):
            validate_protocol_mapping(drift)
        drift = copy.deepcopy(self.raw)
        drift["common_boundary_classifier"]["class_weighting"] = "none"
        with self.assertRaises(OfficialBalancedBoundaryProtocolError):
            validate_protocol_mapping(drift)
        drift = copy.deepcopy(self.raw)
        drift["inner_promotion_gate"]["macro_rmse_min_improvement"] = 0.004
        with self.assertRaises(OfficialBalancedBoundaryProtocolError):
            validate_protocol_mapping(drift)

    def test_v10_frozen_and_method_is_preapproved_3v4_head(self):
        freeze = self.raw["authorization_and_freeze"]
        self.assertFalse(freeze["v10_reopening"])
        self.assertTrue(freeze["v10_inventory_and_artifacts_frozen"])
        self.assertTrue(freeze["v10_gates_unchanged"])
        self.assertIn("dedicated_3_vs_4_auxiliary_head", freeze["study_role"])


if __name__ == "__main__":
    unittest.main()
