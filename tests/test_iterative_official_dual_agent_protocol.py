import copy
import json
import unittest

from mal2026.iterative_official_dual_agent_protocol import (
    CONFIG_PATH,
    OfficialDualAgentProtocolError,
    load_protocol,
    validate_bound_inputs,
    validate_protocol_mapping,
)


class OfficialDualAgentProtocolTest(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def _synthetic_bound(self):
        raw = copy.deepcopy(self.raw)
        raw["post_generation_binding"]["state"] = "validated_checksums_bound_before_smoke"
        raw["lineage"]["luna_candidate_manifest_sha256"] = "a" * 64
        raw["lineage"]["luna_candidate_rows_sha256"] = "b" * 64
        return raw

    def test_pending_config_fails_closed_or_final_config_binds_real_inputs(self):
        if self.raw["post_generation_binding"]["state"] == "pending_validated_luna_download":
            with self.assertRaises(OfficialDualAgentProtocolError):
                load_protocol()
        else:
            audit = validate_bound_inputs(load_protocol())
            self.assertEqual(audit.records, 2000)
            self.assertEqual(audit.folds, {0: 400, 1: 400, 2: 400, 3: 400, 4: 400})
            self.assertEqual(audit.terra_candidates, 6000)
            self.assertEqual(audit.luna_candidates, 6000)

    def test_final_mapping_preserves_exact_inventory_and_gate(self):
        protocol = validate_protocol_mapping(self._synthetic_bound())
        self.assertEqual(protocol.raw["data_contract"]["feature_dimensions"], 96)
        self.assertEqual(len(protocol.raw["candidates"]), 3)
        gate = protocol.raw["inner_promotion_gate"]
        self.assertEqual(gate["gold_3_4_balanced_accuracy_min_improvement"], 0.01)
        self.assertNotIn("macro_score5_recall_min_improvement", gate)
        self.assertTrue(gate["score1_descriptive_only"])

    def test_candidate_gate_and_binding_drift_fail_closed(self):
        drift = self._synthetic_bound()
        drift["candidates"][0]["ridge_alpha"] = 9.0
        with self.assertRaises(OfficialDualAgentProtocolError):
            validate_protocol_mapping(drift)
        drift = self._synthetic_bound()
        drift["inner_promotion_gate"]["macro_rmse_min_improvement"] = 0.004
        with self.assertRaises(OfficialDualAgentProtocolError):
            validate_protocol_mapping(drift)
        drift = self._synthetic_bound()
        drift["post_generation_binding"]["only_fields_allowed_to_change_after_this_preregistration"].append("candidates")
        with self.assertRaises(OfficialDualAgentProtocolError):
            validate_protocol_mapping(drift)

    def test_v9_frozen_and_candidate_inventory_precedes_download(self):
        raw = self._synthetic_bound()
        validate_protocol_mapping(raw)
        self.assertFalse(raw["authorization_and_freeze"]["v9_reopening"])
        self.assertTrue(raw["authorization_and_freeze"]["v9_inventory_and_artifacts_frozen"])
        self.assertTrue(raw["scientific_claims"]["candidate_inventory_frozen_before_luna_batch_results_downloaded"])
        self.assertTrue(raw["scientific_claims"]["no_v10_full_oof_prestudy_before_preregistration"])


if __name__ == "__main__":
    unittest.main()
