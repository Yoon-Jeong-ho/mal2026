import copy
import json
import unittest

from mal2026.iterative_official_agent_stack_protocol import (
    CONFIG_PATH, OfficialAgentStackProtocolError, load_protocol, validate_bound_inputs, validate_protocol_mapping,
)


class OfficialAgentStackProtocolTest(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_canonical_protocol_and_bound_inputs(self):
        protocol = load_protocol()
        audit = validate_bound_inputs(protocol)
        self.assertEqual(audit.records, 2000)
        self.assertEqual(audit.folds, {0: 400, 1: 400, 2: 400, 3: 400, 4: 400})
        self.assertEqual(audit.official_candidates, 6000)
        self.assertEqual(audit.official_model, "gpt-5.6-terra")

    def test_extra_score5_gate_is_not_registered(self):
        self.assertNotIn("macro_score5_recall_min_improvement", self.raw["inner_promotion_gate"])
        self.assertNotIn("macro_score5_recall_min_improvement", self.raw["final_evaluation"])

    def test_candidate_or_gate_drift_fails(self):
        drift = copy.deepcopy(self.raw)
        drift["candidates"][0]["ridge_alpha"] = 11.0
        with self.assertRaises(OfficialAgentStackProtocolError):
            validate_protocol_mapping(drift)
        drift = copy.deepcopy(self.raw)
        drift["inner_promotion_gate"]["macro_rmse_min_improvement"] = 0.004
        with self.assertRaises(OfficialAgentStackProtocolError):
            validate_protocol_mapping(drift)

    def test_adaptive_claim_and_prior_freeze_are_explicit(self):
        self.assertTrue(self.raw["scientific_claims"]["adaptive_after_v1_through_v6_and_full_oof_prestudy_observed"])
        self.assertFalse(self.raw["scientific_claims"]["independent_confirmation_claim_allowed"])
        self.assertTrue(self.raw["authorization_and_freeze"]["v4_v5_v6_artifacts_permanently_frozen"])
        self.assertTrue(self.raw["authorization_and_freeze"]["v6_same_train_same_feature_source_stop_rule_respected"])


if __name__ == "__main__":
    unittest.main()
