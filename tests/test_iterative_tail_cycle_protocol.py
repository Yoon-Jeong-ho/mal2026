from __future__ import annotations

from copy import deepcopy
import unittest

from mal2026.iterative_tail_cycle_protocol import (
    CONFIG_PATH,
    RUN_ID,
    IterativeTailCycleProtocolError,
    cycle_specs,
    load_protocol,
    validate_bound_inputs,
    validate_model_inventory,
    validate_protocol_mapping,
)


class IterativeTailCycleProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(CONFIG_PATH)

    def test_exact_five_by_four_cycle_inventory_and_initialization(self):
        raw = self.protocol.raw
        specs = cycle_specs()
        self.assertEqual(RUN_ID, raw["run_id"])
        self.assertEqual(20, len(specs))
        self.assertEqual(list(range(1, 21)), [item["cycle"] for item in specs])
        self.assertEqual(
            ["soft_routed_residual"] * 4
            + ["pareto_routed_stack"] * 4
            + ["group_dro_ridge"] * 4
            + ["selective_hurdle"] * 4
            + ["final_ordinal_stack"] * 4,
            [item["family"] for item in specs],
        )
        self.assertEqual(20, len({item["variant_id"] for item in specs}))
        self.assertEqual(10.0, specs[0]["parameters"]["alpha"])
        self.assertEqual(0.25, specs[0]["parameters"]["temperature"])
        self.assertEqual(96, specs[15]["parameters"]["evidence_dims"])
        self.assertEqual(0.75, specs[19]["parameters"]["ordinal_weight"])
        self.assertTrue(all(item["initialization_seed"] == 2026080103 for item in specs))
        self.assertTrue(all(item["fresh_initialization"] for item in specs))
        self.assertTrue(raw["execution"]["all_20_cycles_required"])
        self.assertFalse(raw["execution"]["early_stop_allowed"])
        self.assertFalse(raw["execution"]["checkpoint_reuse"])

    def test_train_only_fold_challenger_and_historical_isolation(self):
        raw = self.protocol.raw
        data = raw["data_contract"]
        self.assertEqual(2000, data["records"])
        self.assertFalse(data["validation_loaded"])
        self.assertFalse(data["validation_selection"])
        self.assertFalse(data["average_target_used"])
        self.assertFalse(data["optional_api_enabled"])
        self.assertFalse(data["external_api_calls_allowed"])
        fold = raw["fold_protocol"]
        self.assertEqual("fresh_3_of_4_crossfit_over_other_four", fold["r16_teacher"])
        self.assertEqual("other_four_only", fold["r17_challenger"]["fit_scope"])
        self.assertEqual("other_four_only", fold["direct_evidence_ridge_challenger"]["fit_scope"])
        self.assertEqual(100.0, fold["direct_evidence_ridge_challenger"]["ridge_alpha"])
        self.assertNotIn("alpha_grid", fold["direct_evidence_ridge_challenger"])
        self.assertEqual(
            "post_v2_outer_evidence_direct_alpha_100_selected_in_3_of_5_outer_folds",
            fold["direct_evidence_ridge_challenger"]["adaptive_basis"],
        )
        self.assertFalse(fold["direct_evidence_ridge_challenger"]["v2_aggregate_used_as_model_feature"])
        self.assertFalse(fold["historical_v1_predictions_used_as_features"])
        self.assertFalse(fold["historical_v2_predictions_used_as_features"])
        self.assertTrue(fold["complete_all_fold_predictions_before_aggregate"])
        self.assertFalse(fold["selection_before_aggregate"])

    def test_gate_selection_claim_and_agent_boundaries(self):
        raw = self.protocol.raw
        gate = raw["promotion_gate"]
        self.assertEqual("AND", gate["operator"])
        self.assertEqual(0.005, gate["macro_rmse_min_improvement"])
        self.assertEqual(0.01, gate["equal_group_rmse_min_improvement"])
        self.assertTrue(gate["low_tail_must_improve"] and gate["high_tail_must_improve"])
        self.assertEqual(0.01, gate["gold_3_4_balanced_accuracy_min_improvement"])
        self.assertEqual(0.01, gate["max_axis_rmse_worsening"])
        self.assertEqual(0.005, gate["max_macro_spearman_fall"])
        self.assertTrue(gate["score1_descriptive_only"])
        selection = raw["selection"]
        self.assertEqual("eligible_minimum_macro_rmse", selection["rule"])
        self.assertEqual("lowest_cycle_number", selection["tie_break"])
        self.assertEqual("exact_r0_oof_baseline", selection["no_eligible_fallback"])
        claims = raw["scientific_claims"]
        self.assertTrue(claims["adaptive_after_v2_outer_observed"])
        self.assertFalse(claims["confirmatory_claim_allowed"])
        self.assertFalse(claims["generalization_claim_allowed"])
        self.assertFalse(claims["deployment_claim_allowed"])
        agent = raw["agent_evidence"]
        self.assertEqual("preregistration_evidence_only", agent["role"])
        self.assertFalse(agent["model_feature_allowed"] or agent["pseudo_target_allowed"] or agent["reward_or_weight_allowed"])

    def test_exact_validator_rejects_inventory_isolation_gate_or_claim_drift(self):
        mutations = (
            lambda raw: raw["cycles"].pop(),
            lambda raw: raw["cycles"][4].__setitem__("family", "soft_routed_residual"),
            lambda raw: raw["cycles"][0]["parameters"].__setitem__("temperature", 0.3),
            lambda raw: raw["execution"].__setitem__("early_stop_allowed", True),
            lambda raw: raw["fold_protocol"].__setitem__("historical_v2_predictions_used_as_features", True),
            lambda raw: raw["fold_protocol"]["direct_evidence_ridge_challenger"].__setitem__("ridge_alpha", 10.0),
            lambda raw: raw["promotion_gate"].__setitem__("operator", "OR"),
            lambda raw: raw["scientific_claims"].__setitem__("generalization_claim_allowed", True),
            lambda raw: raw["agent_evidence"].__setitem__("model_feature_allowed", True),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                raw = deepcopy(self.protocol.raw)
                mutate(raw)
                with self.assertRaises(IterativeTailCycleProtocolError):
                    validate_protocol_mapping(raw)

    def test_actual_inputs_are_exact_2000_and_five_by_400(self):
        audit = validate_bound_inputs(self.protocol)
        self.assertEqual(2000, audit.records)
        self.assertEqual({fold: 400 for fold in range(5)}, audit.fold_counts)
        self.assertEqual("8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db", audit.fold_assignment_fingerprint)
        self.assertEqual("823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04", audit.baseline_sha256)
        self.assertEqual("c2770d2add3b08a46614ad4c56fddc2d6b06ba5784c39e80c95c9f419e46d7db", audit.evidence_cache_sha256)
        self.assertEqual("bc64443aafc48c45f9a882cf861c11eba9b22d6d417b62d90c43a1fbf81af81f", audit.historical_v2_aggregate_sha256)
        self.assertEqual("historical_preregistration_evidence_only_forbidden_as_model_feature", audit.historical_v2_aggregate_role)
        self.assertEqual(audit.model_inventory_available, validate_model_inventory(require_available=False))


if __name__ == "__main__":
    unittest.main()
