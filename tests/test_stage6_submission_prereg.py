"""Contract tests for the non-runnable Stage 6 submission preregistration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "configs" / "stage6_submission_prereg.v1.json"
AXES = ["content", "organization", "expression"]
H0_SOURCE_STATE_ROOT = ROOT / (
    "outputs/rlaif-qwen3-embedding-epoch-sweep-v1/"
    "rlaif-qwen3-embedding-epoch-sweep-v1-full-003/epoch_checkpoints"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def pending_leaves(value: object, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], str]:
    if isinstance(value, dict):
        result: dict[tuple[str, ...], str] = {}
        for key, child in value.items():
            result.update(pending_leaves(child, (*path, key)))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(pending_leaves(child, (*path, str(index))))
        return result
    return {path: value} if isinstance(value, str) and value.startswith("PENDING_") else {}


class Stage6SubmissionPreregTests(unittest.TestCase):
    def test_schema_decision_list_gate_and_no_selection_validation_or_average(self) -> None:
        manifest = load_manifest()
        self.assertEqual({
            "schema_version", "status", "run_id", "purpose", "data_contract", "h0_final_refit_map",
            "upstream_preregistration", "common_stage3_promotion_gate", "decision_list", "full_refit_freeze",
            "full_refit_seed_contract", "deployment_freeze", "validation_descriptive_audit", "failure_rule",
            "runtime_fill_policy",
        }, set(manifest))
        self.assertEqual("mal2026-stage6-submission-prereg-v1", manifest["schema_version"])
        self.assertEqual("preregistered_non_runnable_pending_upstream_artifacts", manifest["status"])

        data = manifest["data_contract"]
        self.assertEqual("canonical_train_only_nested_oof", data["selection_population"])
        self.assertEqual(5, data["outer_folds"])
        self.assertEqual(4, data["inner_folds"])
        self.assertEqual(AXES, data["axes"])
        self.assertTrue(data["axis_models_and_predictions_independent"])
        self.assertTrue(data["average_target_forbidden"])
        self.assertTrue(data["average_feature_forbidden"])
        self.assertTrue(data["average_selection_metric_forbidden"])
        self.assertFalse(data["validation_used_for_selection"])
        self.assertTrue(data["validation_path_absent_from_manifest"])

        gate = manifest["common_stage3_promotion_gate"]
        self.assertEqual({
            "reference", "candidate", "operator", "macro_rmse_min_improvement", "maximum_axis_rmse_worsening",
            "maximum_gold_3_4_balanced_accuracy_drop", "maximum_macro_spearman_drop", "low_tail_1_2_noninferior",
            "high_tail_5_noninferior", "require_nonzero_low_1_2_and_high_5_support_every_axis",
            "require_all_five_outer_folds", "require_finite_metrics", "paired_bootstrap",
        }, set(gate))
        self.assertEqual("H0_exact_R0_continuous_train_OOF", gate["reference"])
        self.assertEqual("stage3_coral-natural_only", gate["candidate"])
        self.assertEqual("AND", gate["operator"])
        self.assertEqual(0.005, gate["macro_rmse_min_improvement"])
        self.assertEqual(0.01, gate["maximum_axis_rmse_worsening"])
        self.assertEqual(0.01, gate["maximum_gold_3_4_balanced_accuracy_drop"])
        self.assertEqual(0.005, gate["maximum_macro_spearman_drop"])
        self.assertTrue(gate["low_tail_1_2_noninferior"])
        self.assertTrue(gate["high_tail_5_noninferior"])
        self.assertTrue(gate["require_nonzero_low_1_2_and_high_5_support_every_axis"])
        self.assertTrue(gate["require_all_five_outer_folds"])
        self.assertTrue(gate["require_finite_metrics"])
        self.assertEqual({
            "resamples": 10000,
            "unit": "source_id_with_all_three_axes_clustered",
            "improvement_quantity": "H0_macro_RMSE_minus_candidate_macro_RMSE",
            "required_lower_bound_strictly_gt": 0.0,
        }, gate["paired_bootstrap"])

        decisions = manifest["decision_list"]
        self.assertEqual(["H0", "H1", "H2"], [item["slot"] for item in decisions])
        self.assertEqual(
            "always_emit_exact_historical_deployable_full_train_R0_epoch1_4_prediction_ensemble",
            decisions[0]["rule"],
        )
        self.assertIn("common_stage3_promotion_gate_passes", decisions[1]["rule"])
        self.assertIn("stage4_own_fixed_promotion_gate_passes", decisions[1]["rule"])
        self.assertIn("existing_stage5_fixed_gate_passes", decisions[2]["rule"])
        self.assertEqual(
            "0.8_H0_exact_R0_plus_0.2_Stage3_coral-natural_axiswise_continuous_predictions",
            decisions[2]["fixed_member_contract"],
        )
        self.assertEqual("identity_only", decisions[2]["calibration"])
        self.assertFalse(decisions[2]["learned_calibration_or_weight_selection"])
        self.assertTrue(decisions[2]["stage4_NPCR_is_not_an_H2_member"])

        audit = manifest["validation_descriptive_audit"]
        self.assertEqual({"H0", "new_H1_H2_only"}, set(audit))
        self.assertEqual({
            "status": "grandfathered_historical_deployable_context_only",
            "sole_evaluation": "the existing historical H0 metric recorded with the immutable bundle",
            "reevaluation_forbidden": True,
            "may_not_affect_H1_H2_gate_decisions_or_submission_order": True,
        }, audit["H0"])
        new_slots = audit["new_H1_H2_only"]
        self.assertIn("before any validation load", new_slots["precondition"])
        self.assertEqual("one aggregate-only evaluation per frozen present new slot", new_slots["maximum_evaluations"])
        self.assertTrue(new_slots["row_level_validation_output_forbidden"])
        self.assertEqual(
            ["selection", "ranking", "slot_reordering", "candidate_drop", "retraining", "retuning",
             "new_submission_candidate", "second_evaluation"],
            new_slots["forbidden_actions"],
        )

    def test_full_refit_seed_contract_and_h2_reuses_the_one_stage3_artifact(self) -> None:
        manifest = load_manifest()
        seeds = manifest["full_refit_seed_contract"]
        self.assertEqual({"stage3_coral_natural", "stage4_npcr"}, set(seeds))

        stage3 = seeds["stage3_coral_natural"]
        self.assertEqual({"base_seed", "formula", "phase1_seed_by_axis", "crt_seed_by_axis", "refit"}, set(stage3))
        self.assertEqual(2026080302, stage3["base_seed"])
        self.assertEqual(
            "int.from_bytes(sha256(f'{base}\\0full-refit\\0coral-natural\\0{axis}\\0{phase}'.encode()).digest()[:4], 'big') % (2**31 - 1)",
            stage3["formula"],
        )
        for phase, field in (("phase1", "phase1_seed_by_axis"), ("crt", "crt_seed_by_axis")):
            expected = {
                axis: int.from_bytes(
                    hashlib.sha256(f"{stage3['base_seed']}\0full-refit\0coral-natural\0{axis}\0{phase}".encode()).digest()[:4],
                    "big",
                ) % (2**31 - 1)
                for axis in AXES
            }
            self.assertEqual(expected, stage3[field])
        self.assertEqual(
            "one full-train refit only; the resulting artifact is the sole non-H0 member permitted for H2",
            stage3["refit"],
        )

        stage4 = seeds["stage4_npcr"]
        self.assertEqual({"base_seed", "formula", "axis_index_contract", "candidate_seed_by_axis", "refit"}, set(stage4))
        self.assertEqual(2026080304, stage4["base_seed"])
        self.assertEqual(
            "int(sha256(f'{base}\\0full-refit\\0{candidate}\\0{axis_index}\\0outer-refit'.encode()).hexdigest()[:15], 16) % (2**31 - 1)",
            stage4["formula"],
        )
        self.assertEqual({"content": 0, "organization": 1, "expression": 2}, stage4["axis_index_contract"])
        self.assertEqual({"adjacent-allskip", "adjacent-skip2"}, set(stage4["candidate_seed_by_axis"]))
        for candidate, per_axis in stage4["candidate_seed_by_axis"].items():
            expected = {
                axis: int(
                    hashlib.sha256(
                        f"{stage4['base_seed']}\0full-refit\0{candidate}\0{index}\0outer-refit".encode()
                    ).hexdigest()[:15],
                    16,
                ) % (2**31 - 1)
                for axis, index in stage4["axis_index_contract"].items()
            }
            self.assertEqual(expected, per_axis)
        self.assertEqual("one full-train refit of the fixed modal candidate only", stage4["refit"])

        freeze = manifest["full_refit_freeze"]
        self.assertEqual(
            "reuse the sole hash-bound full-train Stage3 coral-natural refit created when either the Stage3 H1 or Stage5 H2 gate passes; do not run another coral refit; apply only the fixed identity 0.8/0.2 axiswise blend",
            freeze["H2_stage5"],
        )
        self.assertIn("exactly one full-train coral-natural refit", freeze["H1_stage3"])
        self.assertIn("sole non-H0 member permitted for H2", stage3["refit"])

    def test_deployment_freeze_binds_rationale_embedding_runtime_and_fallback(self) -> None:
        manifest = load_manifest()
        deployment = manifest["deployment_freeze"]
        self.assertEqual({"h1_h2_final_rationale", "h1_stage3_score_input", "h1_stage4_feature_and_fallback_contract"}, set(deployment))

        final = deployment["h1_h2_final_rationale"]
        final_adapter = ROOT / final["adapter_path"] / "adapter_model.safetensors"
        self.assertEqual({"adapter_path", "adapter_sha256", "component", "max_tokens", "seed", "temperature", "top_p"}, set(final))
        self.assertTrue(final_adapter.is_file())
        self.assertEqual(final["adapter_sha256"], sha256_file(final_adapter))
        self.assertEqual("887abf9d1bf07693251a17b7a0fb655fe8203fa6945e9c178a38bdc538ded826", final["adapter_sha256"])
        self.assertEqual(512, final["max_tokens"])
        self.assertEqual(42, final["seed"])
        self.assertEqual(0.0, final["temperature"])
        self.assertEqual(1.0, final["top_p"])
        self.assertIn("own emitted integer content/organization/expression scores", final["component"])
        self.assertEqual("official canonical writing prompt plus essay only", deployment["h1_stage3_score_input"])

        stage4 = deployment["h1_stage4_feature_and_fallback_contract"]
        self.assertEqual({
            "anchor_library", "blind_rationale_generator", "canonical_prompt_text_mapping", "query_level_fallback",
            "score_encoder", "score_input",
        }, set(stage4))
        self.assertIn("all 2,000 canonical train rows", stage4["anchor_library"])
        self.assertIn("no validation or hidden rows", stage4["anchor_library"])
        self.assertEqual({
            "construction": "persist the exact bijective canonical prompt-text to prompt_num map from all 2,000 canonical train rows",
            "expected_unique_prompt_num_count": 9,
            "expected_unique_prompt_text_count": 9,
            "inference": "submission supplies prompt text; infer prompt_num only by exact prompt-text match",
            "require_bijection": True,
        }, stage4["canonical_prompt_text_mapping"])
        self.assertIn("H0 continuous axis prediction unchanged", stage4["query_level_fallback"])

        blind = stage4["blind_rationale_generator"]
        blind_adapter = ROOT / blind["adapter_path"] / "adapter_model.safetensors"
        self.assertEqual({"adapter_id", "adapter_path", "adapter_sha256", "max_tokens", "seed", "temperature", "top_p"}, set(blind))
        self.assertTrue(blind_adapter.is_file())
        self.assertEqual(blind["adapter_sha256"], sha256_file(blind_adapter))
        self.assertEqual("39c68bb5c98da25eaa466434ba1c6d4a47bedcec580c991f00335627382a3a73", blind["adapter_sha256"])
        self.assertEqual((512, 42, 0.0, 1.0), (blind["max_tokens"], blind["seed"], blind["temperature"], blind["top_p"]))

        encoder = stage4["score_encoder"]
        self.assertEqual({
            "model_id": "Qwen/Qwen3-Embedding-8B", "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
            "pooling": "last_nonpad_then_float32_L2", "public_base_only": True,
        }, encoder)
        score_input = stage4["score_input"]
        self.assertEqual({
            "contract": "writing_prompt plus student_essay plus canonical JSON evaluation_rationales, with ensure_ascii=false and compact separators",
            "max_length": 2048,
            "renderer": "deployment/src/mal2026_submission/production_r0.py:_score_input",
        }, score_input)

    def test_h0_runtime_bundle_member_and_source_state_hashes(self) -> None:
        manifest = load_manifest()
        h0 = manifest["h0_final_refit_map"]
        self.assertEqual("H0", h0["slot"])
        self.assertEqual("historical_r0_prediction_ensemble_dpo_v1", h0["candidate_id"])
        self.assertEqual("always_present_protected_deployable_full_train_artifact", h0["status"])
        self.assertEqual([1, 2, 3, 4], h0["epoch_prediction_ensemble"]["member_epochs"])
        self.assertEqual("uniform_axiswise_continuous_mean_then_clip_1_5_then_half_up",
                         h0["epoch_prediction_ensemble"]["ensemble"])

        runtime_manifest = ROOT / h0["runtime_manifest_path"]
        bundle_completion = ROOT / h0["bundle_completion_path"]
        staged_runtime_manifest = bundle_completion.parent / "manifest.json"
        self.assertTrue(runtime_manifest.is_file())
        self.assertTrue(bundle_completion.is_file())
        self.assertTrue(staged_runtime_manifest.is_file())
        self.assertEqual(h0["runtime_manifest_sha256"], sha256_file(runtime_manifest))
        self.assertEqual(h0["runtime_manifest_sha256"], sha256_file(staged_runtime_manifest))
        self.assertEqual(h0["bundle_completion_sha256"], sha256_file(bundle_completion))

        runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
        self.assertEqual("mal2026-submission-runtime-v1", runtime["schema_version"])
        self.assertEqual(
            [f"score/adapters/epoch_{epoch:02d}" for epoch in range(1, 5)],
            runtime["score"]["adapter_paths"],
        )
        self.assertEqual(
            [f"score/heads/epoch_{epoch:02d}.safetensors" for epoch in range(1, 5)],
            runtime["score"]["head_paths"],
        )
        self.assertEqual("uniform_continuous_mean_then_clip_1_5_then_half_up", runtime["score"]["ensemble"])

        completion = json.loads(bundle_completion.read_text(encoding="utf-8"))
        self.assertEqual("completed", completion["status"])
        self.assertEqual(h0["candidate_id"], completion["candidate"])
        members = h0["epoch_prediction_ensemble"]["members"]
        self.assertEqual([1, 2, 3, 4], [member["epoch"] for member in members])
        exports = {item["epoch"]: item for item in completion["score_export"] if "epoch" in item}
        self.assertEqual({1, 2, 3, 4}, set(exports))
        self.assertEqual({f"epoch_{epoch:02d}" for epoch in range(1, 5)}, set(completion["checkpoint_sha256"]))
        for member in members:
            epoch = member["epoch"]
            adapter = ROOT / member["adapter_path"]
            head = ROOT / member["head_path"]
            source_state = H0_SOURCE_STATE_ROOT / f"epoch-{epoch:02d}" / "trainable_model.safetensors"
            self.assertTrue(adapter.is_file())
            self.assertTrue(head.is_file())
            self.assertTrue(source_state.is_file())
            self.assertEqual(member["adapter_sha256"], sha256_file(adapter))
            self.assertEqual(member["head_sha256"], sha256_file(head))
            self.assertEqual(member["source_state_sha256"], sha256_file(source_state))
            self.assertEqual(member["adapter_sha256"], exports[epoch]["adapter_model_sha256"])
            self.assertEqual(member["head_sha256"], exports[epoch]["head_sha256"])
            self.assertEqual(member["source_state_sha256"], exports[epoch]["source_state_sha256"])
            self.assertEqual(member["source_state_sha256"], completion["checkpoint_sha256"][f"epoch_{epoch:02d}"])

    def test_stage3_stage4_stage5_hash_lineage_is_exact(self) -> None:
        manifest = load_manifest()
        upstream = manifest["upstream_preregistration"]
        self.assertEqual({"stage3", "stage4", "stage5"}, set(upstream))
        expected = {
            "stage3": ("configs/kure_ordinal_oof.v1.json", "5108e37c490482fbbf45c043f2ad4d0154048d432b4a4a163b3d102f8c6d756e"),
            "stage4": ("configs/prompt_reference_npcr.v1.json", "d9b715b24ed4c2144f1fe2a45c79494a953f098e3798e7f2f17a7a5634c63722"),
            "stage5": ("configs/conservative_oof_combiner.prereg.v1.json", "19a202f375e8e2d47be8d7f21ea015422f85a5d8b45eaccc090b6221cba58b35"),
        }
        for stage, (relative_path, digest) in expected.items():
            key = "preregistration_path" if stage == "stage5" else "config_path"
            hash_key = "preregistration_sha256" if stage == "stage5" else "config_file_sha256"
            self.assertEqual(relative_path, upstream[stage][key])
            self.assertEqual(digest, upstream[stage][hash_key])
            self.assertEqual(digest, sha256_file(ROOT / relative_path))

        self.assertEqual("coral-natural", upstream["stage3"]["allowed_h1_method_id"])
        self.assertEqual("descriptive_only_not_eligible_for_H1_H2_or_full_refit", upstream["stage3"]["rps_role"])
        self.assertEqual("checksum_bound_audit_source_not_a_combination_member",
                         json.loads((ROOT / "configs/conservative_oof_combiner.prereg.v1.json").read_text(encoding="utf-8"))["stage4_lineage"]["role"])

    def test_pending_fields_are_runtime_metadata_allowlisted(self) -> None:
        manifest = load_manifest()
        self.assertEqual({
            ("upstream_preregistration", "stage3", "aggregate_sha256"): "PENDING_AFTER_STAGE3_COMPLETION",
            ("upstream_preregistration", "stage4", "aggregate_sha256"): "PENDING_AFTER_STAGE4_COMPLETION",
            ("upstream_preregistration", "stage5", "aggregate_path"): "PENDING_AFTER_STAGE5_RUNTIME_CONFIGURATION",
            ("upstream_preregistration", "stage5", "aggregate_sha256"): "PENDING_AFTER_STAGE5_COMPLETION",
        }, pending_leaves(manifest))
        policy = manifest["runtime_fill_policy"]
        self.assertIn("only", policy.lower())
        self.assertIn("pending aggregate hashes", policy.lower())
        self.assertIn("completed outer-artifact hashes", policy.lower())
        self.assertIn("candidate full-refit artifact hashes", policy.lower())
        self.assertIn("execution metadata", policy.lower())
        self.assertIn("must not alter any decision-list", policy.lower())


if __name__ == "__main__":
    unittest.main()
