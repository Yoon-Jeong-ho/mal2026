from __future__ import annotations

from copy import deepcopy
import unittest

from mal2026.iterative_tail_remediation_protocol import (
    CONFIG_PATH,
    RUN_ID,
    IterativeTailRemediationError,
    load_protocol,
    outer_inner_folds,
    validate_bound_inputs,
    validate_protocol_mapping,
)
from mal2026.iterative_tail_remediation_models import KNOTS_5, PREDECLARED_GRIDS


class IterativeTailRemediationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(CONFIG_PATH)

    def test_exact_identity_isolation_gpu_and_initialization_contract(self):
        raw = self.protocol.raw
        self.assertEqual(RUN_ID, raw["run_id"])
        self.assertEqual("40a28c758020b356e1d86e9790e55ea08a2ea69c", raw["lineage"]["v1_execution_git_commit"])
        self.assertFalse(raw["data_contract"]["validation_loaded"])
        self.assertFalse(raw["data_contract"]["validation_selection"])
        self.assertFalse(raw["data_contract"]["optional_api_enabled"])
        self.assertFalse(raw["data_contract"]["external_api_calls_allowed"])
        self.assertFalse(raw["data_contract"]["average_target_used"])
        self.assertEqual([0, 1, 2, 3], raw["execution"]["authorized_gpus"])
        self.assertEqual(0, raw["execution"]["smoke_gpu"])
        self.assertTrue(raw["execution"]["fresh_initialization"])
        self.assertFalse(raw["execution"]["proxy_implementations_allowed"])

    def test_outer_inner_complements_and_holdout_sealing_are_exact(self):
        nested = self.protocol.raw["nested_selection"]
        for outer in range(5):
            inner = outer_inner_folds(self.protocol, outer)
            self.assertEqual(4, len(inner))
            self.assertEqual(set(range(5)) - {outer}, set(inner))
        self.assertFalse(nested["outer_fold_access_before_selection"])
        self.assertFalse(nested["outer_holdout_gold_or_features_access_before_final_predict"])
        self.assertFalse(nested["historical_r17_oof_use_in_selection_or_fitting"])
        self.assertFalse(nested["posthoc_selection_on_concatenated_oof"])
        self.assertIn(
            "for_each_inner_validation_regenerate_R16_teacher_by_2_of_3_cross_fit_excluding_inner_validation_and_outer",
            nested["per_outer_procedure"],
        )
        self.assertIn(
            "after_selection_freeze_regenerate_selected_outer_refit_R16_teacher_by_3_of_4_cross_fit_without_outer_access",
            nested["per_outer_procedure"],
        )
        self.assertIn("predict_outer_fold_once_then_unlock_outer_gold_for_metrics", nested["per_outer_procedure"])
        r16 = nested["r16_teacher_regeneration"]
        self.assertEqual("joint_huber_ordinal", r16["method_family"])
        self.assertEqual("consensus_disagreement", r16["feature_view"])
        inner_teacher = r16["inner_selection_teacher_2_of_3"]
        self.assertTrue(inner_teacher["inner_validation_excluded"])
        self.assertTrue(inner_teacher["outer_holdout_excluded"])
        self.assertFalse(inner_teacher["reuse_for_outer_refit"])
        self.assertIn("other_two", inner_teacher["heldout_rule"])
        outer_teacher = r16["outer_refit_teacher_3_of_4"]
        self.assertTrue(outer_teacher["generated_after_selection_freeze"])
        self.assertFalse(outer_teacher["reuse_from_inner_selection"])
        selection = nested["eligible_selection"]
        self.assertEqual("base_identity_for_every_candidate", selection["gate_reference"])
        self.assertFalse(selection["sequential_incumbent_tournament"])

    def test_candidate_gate_and_final_gate_registration(self):
        raw = self.protocol.raw
        self.assertEqual(
            [
                "base identity", "R17 raw", "conditional R17 delta gate grid",
                "weighted isotonic/piecewise", "low/high tail offsets+3/4 nudge",
                "evidence direct ridge alpha grid", "top-two nested ensemble",
            ],
            [candidate["name"] for candidate in raw["candidates"]],
        )
        rebuilt = raw["candidates"][1]["hyperparameters"]
        self.assertEqual(
            "split_specific_fresh_R16_cross_fit_teacher_excluding_current_selection_holdout",
            rebuilt["pseudo_target_source"],
        )
        self.assertEqual(10.0, rebuilt["ridge_alpha"])
        self.assertFalse(rebuilt["historical_r17_artifact_used"])
        gate = raw["candidates"][2]["hyperparameters"]
        self.assertEqual(list(PREDECLARED_GRIDS["gate_kind"]), gate["gate_kind_grid"])
        self.assertEqual(list(PREDECLARED_GRIDS["gate_threshold"]), gate["gate_threshold_grid"])
        self.assertEqual(list(PREDECLARED_GRIDS["gate_temperature"]), gate["gate_temperature_grid"])
        self.assertEqual(list(PREDECLARED_GRIDS["delta_weight"]), gate["delta_weight_grid"])
        self.assertEqual(list(PREDECLARED_GRIDS["low_identity_threshold"]), gate["low_identity_threshold_grid"])
        calibration = raw["candidates"][3]["hyperparameters"]
        self.assertEqual(list(PREDECLARED_GRIDS["calibration_source"]), calibration["calibration_source_grid"])
        self.assertEqual(KNOTS_5.tolist(), calibration["piecewise_fixed_knots"])
        self.assertEqual("exact_weighted_least_squares", calibration["piecewise_fit_objective"])
        self.assertEqual(64, calibration["piecewise_active_face_count"])
        tail = raw["candidates"][4]["hyperparameters"]
        self.assertEqual(list(PREDECLARED_GRIDS["tail_source"]), tail["tail_source_grid"])
        self.assertEqual(list(PREDECLARED_GRIDS["boundary_nudge"]), tail["boundary_nudge_grid"])
        self.assertEqual(2.5, tail["low_score_threshold"])
        self.assertEqual(4.5, tail["high_score_threshold"])
        self.assertEqual("triangular", tail["boundary_kernel"])
        blend = raw["candidates"][6]["hyperparameters"]
        self.assertEqual(list(PREDECLARED_GRIDS["blend_weight"]), blend["candidate_two_weight_grid"])
        inner = raw["inner_promotion_gate"]
        self.assertEqual("AND", inner["operator"])
        self.assertEqual(0.005, inner["macro_rmse_min_improvement"])
        self.assertEqual(0.01, inner["equal_group_rmse_min_improvement"])
        self.assertTrue(inner["low_tail_must_improve"] and inner["high_tail_must_improve"])
        self.assertEqual(0.01, inner["gold_3_4_balanced_accuracy_min_improvement"])
        self.assertEqual(0.01, inner["max_axis_rmse_worsening"])
        self.assertEqual(0.005, inner["max_macro_spearman_fall"])
        outer = raw["outer_final_gate"]
        self.assertEqual(0.01, outer["macro_rmse_min_improvement"])
        self.assertEqual(0.95, outer["paired_bootstrap_confidence"])
        self.assertTrue(outer["paired_bootstrap_candidate_minus_baseline_ci_upper_below_zero"])
        self.assertEqual("base_identity", outer["fallback"])

    def test_exact_validator_rejects_selection_leakage_or_proxy_drift(self):
        mutations = (
            lambda raw: raw["nested_selection"].__setitem__("outer_fold_access_before_selection", True),
            lambda raw: raw["nested_selection"].__setitem__("historical_r17_oof_use_in_selection_or_fitting", True),
            lambda raw: raw["execution"].__setitem__("proxy_implementations_allowed", True),
            lambda raw: raw["nested_selection"]["eligible_selection"].__setitem__("sequential_incumbent_tournament", True),
            lambda raw: raw["nested_selection"]["r16_teacher_regeneration"]["inner_selection_teacher_2_of_3"].__setitem__("inner_validation_excluded", False),
            lambda raw: raw["candidates"][1]["hyperparameters"].__setitem__("historical_r17_artifact_used", True),
            lambda raw: raw["inner_promotion_gate"].__setitem__("operator", "OR"),
            lambda raw: raw["outer_final_gate"].__setitem__("macro_rmse_min_improvement", 0.005),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                raw = deepcopy(self.protocol.raw)
                mutate(raw)
                with self.assertRaises(IterativeTailRemediationError):
                    validate_protocol_mapping(raw)

    def test_actual_v1_lineage_is_2000_rows_and_five_by_400(self):
        audit = validate_bound_inputs(self.protocol)
        self.assertEqual(2000, audit.records)
        self.assertEqual({fold: 400 for fold in range(5)}, audit.fold_counts)
        self.assertEqual("8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db", audit.fold_assignment_fingerprint)
        self.assertEqual("5130e77abdee8866219a86d53896ddb6d1c680ae6cf4f3384fdc8a428ef7141a", audit.historical_r17_sha256)
        self.assertIn("forbidden_as_v2_feature", audit.historical_r17_role)


if __name__ == "__main__":
    unittest.main()
