from __future__ import annotations

from copy import deepcopy
import unittest

from mal2026.iterative_tail_learner_protocol import (
    CONFIG_PATH,
    RUN_ID,
    IterativeTailLearnerProtocolError,
    candidate_specs,
    load_protocol,
    validate_bound_inputs,
    validate_model_inventory,
    validate_protocol_mapping,
)


class IterativeTailLearnerProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(CONFIG_PATH)

    def test_exact_five_by_four_candidate_inventory_and_model_binding(self):
        specs = candidate_specs()
        self.assertEqual("iterative-tail-learner-v5-20260802-001", RUN_ID)
        self.assertEqual(list(range(1, 21)), [item["cycle"] for item in specs])
        self.assertEqual(
            ["anchored_multitask_residual"] * 4
            + ["r0_anchored_distributional"] * 4
            + ["joint_tail_boundary_hurdle"] * 4
            + ["axis_coupled_lowrank_moe"] * 4
            + ["band_risk_pareto_residual"] * 4,
            [item["family"] for item in specs],
        )
        self.assertEqual(20, len({item["variant_id"] for item in specs}))
        self.assertTrue(all(item["initialization_seed"] == 2026080205 for item in specs))
        self.assertTrue(all(item["fresh_initialization"] for item in specs))
        self.assertTrue(validate_model_inventory(require_available=True))

    def test_exact_candidate_level_five_by_four_nesting(self):
        raw = self.protocol.raw
        nested = raw["nested_protocol"]
        self.assertEqual([0, 1, 2, 3, 4], nested["outer_folds"])
        self.assertEqual(4, nested["inner_fold_count_per_outer"])
        self.assertEqual("S_is_the_other_three_folds_excluding_O_and_D", nested["inner_training_rule"])
        self.assertEqual("each_candidate_fresh_fit_on_S_predict_D_once", nested["candidate_inner_fit"])
        self.assertEqual("concatenate_four_D_predictions_to_1600_rows", nested["candidate_oof_construction"])
        self.assertEqual("each_outer_train_row_exactly_once_per_candidate", nested["candidate_oof_coverage"])
        self.assertTrue(nested["selection_after_all_candidates_complete"])
        selection = nested["inner_selection"]
        self.assertEqual("eligible_minimum_macro_rmse", selection["rule"])
        self.assertEqual("lowest_cycle_number", selection["tie_break"])
        self.assertEqual("exact_r0_oof_baseline", selection["no_eligible_fallback"])
        outer = nested["outer_refit"]
        self.assertTrue(outer["selected_spec_frozen_before_refit"] and outer["selected_candidate_only"])
        self.assertEqual("fresh", outer["initialization"])
        self.assertEqual("O_once", outer["predict_scope"])
        self.assertFalse(any(nested[key] for key in (
            "historical_v1_predictions_used", "historical_v2_predictions_used",
            "historical_v3_predictions_used", "historical_v4_predictions_used",
            "historical_weights_used", "historical_pseudo_targets_used",
        )))

    def test_gates_claims_and_v4_supersession_are_exact(self):
        gate = self.protocol.raw["inner_promotion_gate"]
        self.assertEqual("AND", gate["operator"])
        self.assertEqual(0.005, gate["macro_rmse_min_improvement"])
        self.assertEqual(0.01, gate["equal_group_rmse_min_improvement"])
        self.assertTrue(gate["low_tail_must_improve"] and gate["high_tail_must_improve"])
        self.assertEqual(0.01, gate["gold_3_4_balanced_accuracy_min_improvement"])
        self.assertEqual(0.01, gate["max_axis_rmse_worsening"])
        self.assertEqual(0.005, gate["max_macro_spearman_fall"])
        final = self.protocol.raw["final_evaluation"]
        self.assertFalse(final["selection_after_concatenation"])
        self.assertEqual(0.01, final["macro_rmse_min_improvement"])
        self.assertEqual(10000, final["paired_bootstrap"]["replicates"])
        self.assertEqual("candidate_minus_baseline_macro_rmse", final["paired_bootstrap"]["quantity"])
        self.assertEqual(0.0, final["paired_bootstrap"]["required_upper_bound_lt"])
        supersession = self.protocol.raw["v4_stop_supersession"]
        self.assertEqual("2026-08-02", supersession["authorization_date"])
        self.assertEqual("V4 실패 후보 동결이지 프로젝트 전체 중단 아님; V5 신규20 계속", supersession["authorization_statement"])
        self.assertTrue(supersession["v4_inventory_permanently_frozen"])
        self.assertTrue(supersession["v4_learned_weights_permanently_frozen"])
        self.assertFalse(supersession["v4_posthoc_tuning_allowed"])
        claims = self.protocol.raw["scientific_claims"]
        self.assertTrue(claims["same_train_nested_descriptive_evidence_only"])
        self.assertFalse(claims["independent_confirmation_claim_allowed"])
        self.assertFalse(claims["generalization_claim_allowed"] or claims["deployment_claim_allowed"])

    def test_exact_validator_rejects_inventory_nesting_gate_or_supersession_drift(self):
        mutations = (
            lambda raw: raw["candidates"].pop(),
            lambda raw: raw["candidates"][4].__setitem__("family", "anchored_multitask_residual"),
            lambda raw: raw["nested_protocol"].__setitem__("candidate_inner_fit", "fit_on_D"),
            lambda raw: raw["nested_protocol"].__setitem__("historical_v4_predictions_used", True),
            lambda raw: raw["execution"].__setitem__("checkpoint_reuse", True),
            lambda raw: raw["inner_promotion_gate"].__setitem__("operator", "OR"),
            lambda raw: raw["final_evaluation"].__setitem__("selection_after_concatenation", True),
            lambda raw: raw["v4_stop_supersession"].__setitem__("v4_posthoc_tuning_allowed", True),
            lambda raw: raw["scientific_claims"].__setitem__("generalization_claim_allowed", True),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                raw = deepcopy(self.protocol.raw)
                mutate(raw)
                with self.assertRaises(IterativeTailLearnerProtocolError):
                    validate_protocol_mapping(raw)

    def test_actual_inputs_and_all_historical_evidence_bindings(self):
        audit = validate_bound_inputs(self.protocol)
        self.assertEqual(2000, audit.records)
        self.assertEqual({fold: 400 for fold in range(5)}, audit.fold_counts)
        self.assertEqual("8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db", audit.fold_assignment_fingerprint)
        self.assertEqual("d3e0e2f7871518bf9123e554ad19afc764a5e257a7a9a087a9cdd1e466e3d0f7", audit.historical_v1_promotion_sha256)
        self.assertEqual("bc64443aafc48c45f9a882cf861c11eba9b22d6d417b62d90c43a1fbf81af81f", audit.historical_v2_aggregate_sha256)
        self.assertEqual("bc44123673b512ae20212a94f7a3e98b4d227823cd1d74d2fddf8f4a07e3667f", audit.historical_v3_aggregate_sha256)
        self.assertEqual("5c34e53706b1e2bc90ea24a402c9f9efc444549ba2ce12f01e7ac481bd978279", audit.historical_v4_aggregate_sha256)
        self.assertEqual("adaptive_preregistration_evidence_only_forbidden_as_model_input", audit.historical_artifact_role)
        self.assertTrue(audit.model_inventory_available)


if __name__ == "__main__":
    unittest.main()
