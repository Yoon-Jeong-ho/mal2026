import copy
import json
import unittest

from mal2026.iterative_official_selective_flip_protocol import (
    CONFIG_PATH,
    OfficialSelectiveFlipProtocolError,
    load_protocol,
    validate_bound_inputs,
    validate_protocol_mapping,
)


class OfficialSelectiveFlipProtocolTest(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_canonical_protocol_and_bound_inputs(self):
        audit = validate_bound_inputs(load_protocol())
        self.assertEqual(audit.records, 2000)
        self.assertEqual(audit.folds, {0: 400, 1: 400, 2: 400, 3: 400, 4: 400})
        self.assertEqual(audit.official_candidates, 6000)
        self.assertEqual(
            audit.v8_aggregate_sha256,
            "97dc01637c16a3448eb6971b03d266b38f7c1aee339add7b474db65866ecae4e",
        )

    def test_no_v9_full_oof_prestudy_and_exact_original_gate(self):
        claims = self.raw["scientific_claims"]
        self.assertTrue(claims["no_v9_full_oof_prestudy_before_preregistration"])
        gate = self.raw["inner_promotion_gate"]
        self.assertEqual(gate["gold_3_4_balanced_accuracy_min_improvement"], 0.01)
        self.assertNotIn("macro_score5_recall_min_improvement", gate)
        self.assertTrue(gate["score1_descriptive_only"])

    def test_candidate_and_gate_drift_fail_closed(self):
        drift = copy.deepcopy(self.raw)
        drift["candidates"][0]["confidence"] = 0.59
        with self.assertRaises(OfficialSelectiveFlipProtocolError):
            validate_protocol_mapping(drift)
        drift = copy.deepcopy(self.raw)
        drift["candidates"][1]["window"] = 0.16
        with self.assertRaises(OfficialSelectiveFlipProtocolError):
            validate_protocol_mapping(drift)
        drift = copy.deepcopy(self.raw)
        drift["inner_promotion_gate"]["macro_rmse_min_improvement"] = 0.004
        with self.assertRaises(OfficialSelectiveFlipProtocolError):
            validate_protocol_mapping(drift)

    def test_v8_is_frozen_and_v9_is_materially_distinct(self):
        freeze = self.raw["authorization_and_freeze"]
        self.assertFalse(freeze["v8_reopening"])
        self.assertTrue(freeze["v8_inventory_and_artifacts_frozen"])
        self.assertTrue(freeze["v8_gates_unchanged"])
        self.assertEqual(
            freeze["materially_distinct_application_rule"],
            "high_confidence_near_boundary_flip_only",
        )


if __name__ == "__main__":
    unittest.main()
