from __future__ import annotations

from copy import deepcopy
import unittest

from mal2026.iterative_tail_router_protocol import (
    CONFIG_PATH,
    RUN_ID,
    IterativeTailRouterProtocolError,
    load_protocol,
    router_specs,
    validate_bound_inputs,
    validate_model_inventory,
    validate_protocol_mapping,
)


class IterativeTailRouterProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(CONFIG_PATH)

    def test_exact_five_by_four_router_inventory_and_model_binding(self):
        specs = router_specs()
        self.assertEqual("iterative-tail-router-v4-20260801-001", RUN_ID)
        self.assertEqual(list(range(1, 21)), [item["cycle"] for item in specs])
        self.assertEqual(
            ["low_protected_sigmoid_stack"] * 4
            + ["four_zone_hard_stack"] * 4
            + ["boundary_hurdle_overlay"] * 4
            + ["sigmoid_four_expert_route"] * 4
            + ["formal_gate_lattice_stack"] * 4,
            [item["family"] for item in specs],
        )
        self.assertEqual(20, len({item["variant_id"] for item in specs}))
        self.assertTrue(all(item["initialization_seed"] == 2026080104 for item in specs))
        self.assertTrue(all(item["fresh_initialization"] for item in specs))
        self.assertTrue(validate_model_inventory(require_available=True))

    def test_exact_nested_inner_selection_and_fresh_outer_refit(self):
        nested = self.protocol.raw["nested_protocol"]
        self.assertEqual([0, 1, 2, 3, 4], nested["outer_folds"])
        self.assertEqual(4, nested["inner_fold_count_per_outer"])
        self.assertEqual("S_is_the_three_folds_excluding_O_and_D", nested["inner_training_rule"])
        self.assertEqual("fresh_2_of_3_crossfit_within_S", nested["inner_components"]["r16_teacher"]["method"])
        self.assertEqual(["D", "O"], nested["inner_components"]["r16_teacher"]["forbidden_folds"])
        self.assertEqual(100.0, nested["inner_components"]["r17_challenger"]["ridge_alpha"])
        self.assertEqual(100.0, nested["inner_components"]["direct_evidence_ridge"]["ridge_alpha"])
        self.assertEqual("fresh_v3_hurdle-v1", nested["inner_components"]["hurdle_component"]["source"])
        self.assertEqual("fresh_v3_soft-v4", nested["inner_components"]["soft_component"]["source"])
        self.assertEqual("every_outer_train_row_exactly_once", nested["inner_bank"]["coverage"])
        self.assertTrue(nested["outer_refit"]["after_route_freeze"])
        self.assertEqual("fresh_3_of_4_crossfit_over_all_outer_train", nested["outer_refit"]["r16_teacher"])
        self.assertEqual("O_once", nested["outer_refit"]["predict_scope"])
        self.assertTrue(nested["outer_metrics_locked_until_prediction_complete"])
        self.assertFalse(nested["historical_v1_predictions_used_as_features"])
        self.assertFalse(nested["historical_v2_predictions_used_as_features"])
        self.assertFalse(nested["historical_v3_predictions_used_as_features"])

    def test_inner_and_final_gates_claim_and_stop_boundaries(self):
        inner = self.protocol.raw["inner_promotion_gate"]
        self.assertEqual("AND", inner["operator"])
        self.assertEqual(0.005, inner["macro_rmse_min_improvement"])
        self.assertEqual(0.01, inner["equal_group_rmse_min_improvement"])
        self.assertTrue(inner["low_tail_must_improve"] and inner["high_tail_must_improve"])
        self.assertEqual(0.01, inner["gold_3_4_balanced_accuracy_min_improvement"])
        self.assertEqual(0.01, inner["max_axis_rmse_worsening"])
        self.assertEqual(0.005, inner["max_macro_spearman_fall"])
        self.assertTrue(inner["score1_descriptive_only"] and inner["require_all_four_inner_folds"])
        final = self.protocol.raw["final_evaluation"]
        self.assertEqual("concatenate_five_outer_predictions_once", final["construction"])
        self.assertFalse(final["selection_after_concatenation"])
        self.assertEqual(0.01, final["macro_rmse_min_improvement"])
        self.assertEqual(10000, final["paired_bootstrap"]["replicates"])
        self.assertEqual(0.0, final["paired_bootstrap"]["required_upper_bound_lt"])
        claims = self.protocol.raw["scientific_claims"]
        self.assertTrue(claims["final_same_train_adaptive_nested_evidence"])
        self.assertFalse(claims["independent_confirmation_claim_allowed"])
        self.assertFalse(claims["generalization_claim_allowed"] or claims["deployment_claim_allowed"])
        self.assertEqual("freeze_all_same_train_model_search", self.protocol.raw["stop_rule"]["action"])
        self.assertEqual("exact_r0_oof_baseline", self.protocol.raw["stop_rule"]["retained_model"])
        agent = self.protocol.raw["agent_evidence"]
        self.assertFalse(agent["model_feature_allowed"] or agent["pseudo_target_allowed"] or agent["reward_or_weight_allowed"])

    def test_exact_validator_rejects_nested_gate_inventory_or_claim_drift(self):
        mutations = (
            lambda raw: raw["routers"].pop(),
            lambda raw: raw["routers"][4].__setitem__("family", "low_protected_sigmoid_stack"),
            lambda raw: raw["nested_protocol"].__setitem__("inner_training_rule", "use_outer_holdout"),
            lambda raw: raw["nested_protocol"]["inner_components"]["r17_challenger"].__setitem__("ridge_alpha", 10.0),
            lambda raw: raw["nested_protocol"].__setitem__("historical_v3_predictions_used_as_features", True),
            lambda raw: raw["inner_promotion_gate"].__setitem__("operator", "OR"),
            lambda raw: raw["final_evaluation"].__setitem__("selection_after_concatenation", True),
            lambda raw: raw["scientific_claims"].__setitem__("generalization_claim_allowed", True),
            lambda raw: raw["agent_evidence"].__setitem__("model_feature_allowed", True),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                raw = deepcopy(self.protocol.raw)
                mutate(raw)
                with self.assertRaises(IterativeTailRouterProtocolError):
                    validate_protocol_mapping(raw)

    def test_actual_inputs_are_exact_2000_and_five_by_400(self):
        audit = validate_bound_inputs(self.protocol)
        self.assertEqual(2000, audit.records)
        self.assertEqual({fold: 400 for fold in range(5)}, audit.fold_counts)
        self.assertEqual("8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db", audit.fold_assignment_fingerprint)
        self.assertEqual("823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04", audit.baseline_sha256)
        self.assertEqual("c2770d2add3b08a46614ad4c56fddc2d6b06ba5784c39e80c95c9f419e46d7db", audit.evidence_cache_sha256)
        self.assertEqual("bc64443aafc48c45f9a882cf861c11eba9b22d6d417b62d90c43a1fbf81af81f", audit.historical_v2_aggregate_sha256)
        self.assertEqual("bc44123673b512ae20212a94f7a3e98b4d227823cd1d74d2fddf8f4a07e3667f", audit.historical_v3_aggregate_sha256)
        self.assertTrue(audit.model_inventory_available)


if __name__ == "__main__":
    unittest.main()
