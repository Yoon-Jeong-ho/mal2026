from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from mal2026.iterative_tail_directional_protocol import (
    CONFIG_PATH,
    RUN_ID,
    MODEL_MODULE,
    IterativeTailDirectionalProtocolError,
    candidate_specs,
    load_protocol,
    validate_bound_inputs,
    validate_model_inventory,
    validate_protocol_mapping,
)


class IterativeTailDirectionalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(CONFIG_PATH)

    def test_exact_three_candidate_inventory_and_model_binding(self):
        specs = candidate_specs()
        self.assertEqual("iterative-tail-directional-v6-20260802-001", RUN_ID)
        self.assertEqual([1, 2, 3], [item["cycle"] for item in specs])
        self.assertEqual(["crossfit_safe_directional_residual"] * 3, [item["family"] for item in specs])
        self.assertEqual(
            ["crossfit_safe_directional_residual-primary",
             "crossfit_safe_directional_residual-conservative",
             "crossfit_safe_directional_residual-linear-safety-control"],
            [item["variant_id"] for item in specs],
        )
        self.assertEqual([True, True, False], [item["parameters"]["nonlinear"] for item in specs])
        self.assertEqual([.01, .02, .01], [item["parameters"]["benefit_margin"] for item in specs])
        self.assertEqual([4.0, 4.5, 4.0], [item["parameters"]["identity_bias"] for item in specs])
        self.assertTrue(all(item["parameters"]["hidden"] == 64 for item in specs))
        self.assertTrue(all(item["parameters"]["cap_low"] == .40 for item in specs))
        self.assertTrue(all(item["parameters"]["cap_high"] == .80 for item in specs))
        self.assertTrue(all(item["parameters"]["cap_center"] == .08 for item in specs))
        self.assertTrue(all(item["initialization_seed"] == 2026080206 for item in specs))
        self.assertTrue(validate_model_inventory(require_available=True))

    def test_model_absence_is_compile_tolerant_but_preflight_strict(self):
        missing = ModuleNotFoundError(MODEL_MODULE)
        missing.name = MODEL_MODULE
        with patch("mal2026.iterative_tail_directional_protocol.importlib.import_module", side_effect=missing):
            self.assertFalse(validate_model_inventory(require_available=False))
            with self.assertRaises(IterativeTailDirectionalProtocolError):
                validate_model_inventory(require_available=True)

    def test_projection_training_and_nested_benefit_labels_are_sealed(self):
        raw = self.protocol.raw
        self.assertEqual({
            "input_dimensions": 4672, "output_dimensions": 64, "seed": 2026080206,
            "deterministic": True, "fit_to_data": False, "gold_used": False,
            "normalization": "fixed_seed_random_projection_only",
            "generator": "numpy.random.default_rng_PCG64",
            "matrix_distribution": "rademacher_pm1_over_sqrt_output_dimensions",
        }, raw["random_projection"])
        self.assertEqual({"learning_rate": .0003, "weight_decay": .001, "batch_size": 128,
                          "epochs": 30, "gradient_clip": 1.0}, raw["training"])
        nested = raw["nested_protocol"]
        self.assertEqual(4, nested["inner_fold_count_per_outer"])
        self.assertEqual("S_is_the_other_three_original_folds_excluding_O_and_D", nested["inner_training_rule"])
        self.assertEqual("each_candidate_fresh_fit_on_S_predict_D_once", nested["candidate_inner_fit"])
        internal = nested["internal_expert_crossfit"]
        self.assertEqual("S_only", internal["scope"])
        self.assertEqual("original_folds", internal["split_unit"])
        self.assertEqual("derive_strict_OOF_benefit_labels", internal["purpose"])
        self.assertFalse(internal["D_allowed"] or internal["O_allowed"])
        self.assertTrue(nested["selection_after_all_candidates_complete"])
        self.assertTrue(nested["outer_refit"]["selected_spec_frozen_before_refit"])
        self.assertEqual("O_once", nested["outer_refit"]["predict_scope"])
        self.assertFalse(any(nested[key] for key in (
            "historical_predictions_used", "historical_row_errors_used", "historical_weights_used",
            "historical_checkpoints_used", "historical_pseudo_targets_used",
        )))

    def test_eight_gate_final_gate_claims_and_terminal_freeze(self):
        inner = self.protocol.raw["inner_promotion_gate"]
        self.assertEqual("AND", inner["operator"])
        self.assertEqual(.005, inner["macro_rmse_min_improvement"])
        self.assertEqual(.01, inner["equal_group_rmse_min_improvement"])
        self.assertEqual(.01, inner["gold_3_4_balanced_accuracy_min_improvement"])
        self.assertEqual(.01, inner["macro_score5_recall_min_improvement"])
        self.assertTrue(inner["low_tail_must_improve"] and inner["high_tail_must_improve"])
        final = self.protocol.raw["final_evaluation"]
        self.assertFalse(final["selection_after_concatenation"])
        self.assertEqual(.01, final["macro_rmse_min_improvement"])
        self.assertEqual(.01, final["macro_score5_recall_min_improvement"])
        self.assertEqual(10000, final["paired_bootstrap"]["replicates"])
        self.assertEqual("candidate_minus_baseline_macro_rmse", final["paired_bootstrap"]["quantity"])
        self.assertEqual(0.0, final["paired_bootstrap"]["required_upper_bound_lt"])
        freeze = self.protocol.raw["authorization_and_freeze"]
        self.assertFalse(freeze["v5_reopening"] or freeze["v4_v5_posthoc_tuning_allowed"])
        self.assertTrue(freeze["v4_inventory_permanently_frozen"] and freeze["v5_inventory_permanently_frozen"])
        self.assertTrue(self.protocol.raw["failure_action"]["terminal_freeze_same_train_and_feature_sources"])
        claims = self.protocol.raw["scientific_claims"]
        self.assertTrue(claims["same_train_nested_descriptive_and_falsification_evidence_only"])
        self.assertFalse(claims["independent_confirmation_claim_allowed"] or claims["generalization_claim_allowed"] or claims["deployment_claim_allowed"])

    def test_exact_validator_rejects_inventory_isolation_gate_or_freeze_drift(self):
        mutations = (
            lambda raw: raw["candidates"].pop(),
            lambda raw: raw["candidates"][0]["parameters"].__setitem__("benefit_margin", .02),
            lambda raw: raw["random_projection"].__setitem__("gold_used", True),
            lambda raw: raw["nested_protocol"]["internal_expert_crossfit"].__setitem__("D_allowed", True),
            lambda raw: raw["nested_protocol"].__setitem__("historical_row_errors_used", True),
            lambda raw: raw["inner_promotion_gate"].__setitem__("macro_score5_recall_min_improvement", 0.0),
            lambda raw: raw["final_evaluation"].__setitem__("selection_after_concatenation", True),
            lambda raw: raw["authorization_and_freeze"].__setitem__("v5_reopening", True),
            lambda raw: raw["failure_action"].__setitem__("terminal_freeze_same_train_and_feature_sources", False),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                raw = deepcopy(self.protocol.raw); mutate(raw)
                with self.assertRaises(IterativeTailDirectionalProtocolError):
                    validate_protocol_mapping(raw)

    def test_actual_inputs_and_v1_through_v5_public_evidence_bindings(self):
        audit = validate_bound_inputs(self.protocol)
        self.assertEqual(2000, audit.records)
        self.assertEqual({fold: 400 for fold in range(5)}, audit.fold_counts)
        self.assertEqual("8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db", audit.fold_assignment_fingerprint)
        self.assertEqual(10, len(audit.historical_sha256))
        self.assertEqual("d3e0e2f7871518bf9123e554ad19afc764a5e257a7a9a087a9cdd1e466e3d0f7", audit.historical_sha256["v1_aggregate"])
        self.assertEqual("5c34e53706b1e2bc90ea24a402c9f9efc444549ba2ce12f01e7ac481bd978279", audit.historical_sha256["v4_aggregate"])
        self.assertEqual("eb7906883fbe91d93ab0928848c91ffa8448cd8fc278033caea3c0c06dd99705", audit.historical_sha256["v5_aggregate"])
        self.assertEqual("37b96297e552c80793fa78a9dc2557b5b287e931061f88b545ae1c98cafbc34b", audit.historical_sha256["v5_completion"])
        self.assertEqual("adaptive_preregistration_and_falsification_evidence_only_forbidden_as_model_input", audit.historical_artifact_role)
        self.assertTrue(audit.model_inventory_available)


if __name__ == "__main__":
    unittest.main()
