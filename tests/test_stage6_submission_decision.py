from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mal2026.stage6_submission_decision import (
    DecisionConfig, Stage6SubmissionDecisionError, _load_preregistration, run,
)


STAGE3_CONFIG_SHA = "5108e37c490482fbbf45c043f2ad4d0154048d432b4a4a163b3d102f8c6d756e"
STAGE3_REPORT_SHA = "28e6ba7465b91ba1fcc306f97f10a8e213d4a6e0069ba499fd92199d827a15aa"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def fold_hashes(fold: int) -> tuple[str, str]:
    return f"{fold + 1:x}" * 64, f"{fold + 6:x}" * 64


def stage3_upstream() -> dict:
    return {
        "schema_version": "mal2026-kure-ordinal-oof-aggregate-v1", "status": "completed",
        "run_id": "kure-ordinal-oof-v1-20260803-001", "config_sha256": STAGE3_REPORT_SHA,
        "records": 2000, "folds": 5,
        "methods": [{
            "method": "coral-natural",
            "fold_bindings": [
                {"outer_fold": fold, "public_sha256": fold_hashes(fold)[0],
                 "restricted_prediction_sha256": fold_hashes(fold)[1]}
                for fold in range(5)
            ],
        }],
    }


def stage3_runtime(output: Path, upstream: Path, upstream_sha: str) -> dict:
    return {
        "schema_version": "mal2026-stage3-coral-promotion-v1", "run_id": "stage3-unit",
        "output_path": str(output), "stage3_aggregate_path": str(upstream),
        "stage3_aggregate_sha256": upstream_sha, "stage3_config_file_sha256": STAGE3_CONFIG_SHA,
        "stage3_report_config_sha256": STAGE3_REPORT_SHA,
        "outer_bindings": [
            {"outer_fold": fold, "public_sha256": fold_hashes(fold)[0],
             "coral_restricted_sha256": fold_hashes(fold)[1]}
            for fold in range(5)
        ],
    }


def stage3_promotion(eligible: bool, upstream_sha: str, runtime_sha: str) -> dict:
    return {
        "schema_version": "mal2026-stage3-coral-promotion-v1", "status": "completed",
        "method": "coral-natural", "rps_eligible": False, "eligible": eligible,
        "promotion_gate": {"eligible": eligible},
        "stage6_preregistration_sha256": "7616e038dd0dcb8a10a15c09780ca178ff43700c132fa941ba4e050e2a8176e1",
        "stage6_preregistration_commit": "32b0a43eda5612284d5bd718c5afbce2be182eff",
        "stage3_config_file_sha256": STAGE3_CONFIG_SHA,
        "stage3_report_config_sha256": STAGE3_REPORT_SHA,
        "stage3_aggregate_sha256": upstream_sha, "config_sha256": runtime_sha,
        "outer_bindings": [
            {"outer_fold": fold, "public_sha256": fold_hashes(fold)[0],
             "coral_restricted_sha256": fold_hashes(fold)[1]}
            for fold in range(5)
        ],
        "run_id": "stage3-unit", "validation_rows_loaded": False, "average_target_used": False,
    }


def stage4(eligible: bool) -> dict:
    return {
        "schema_version": "mal2026-prompt-reference-npcr-v1", "status": "completed", "mode": "full",
        "run_id": "prompt-reference-npcr-v1-20260803-001",
        "config_sha256": "d9b715b24ed4c2144f1fe2a45c79494a953f098e3798e7f2f17a7a5634c63722",
        "records": 2000, "folds": 5, "promotion_gate_vs_exact_r0": {"eligible": eligible},
        "global_recommendation": "npcr" if eligible else "exact_r0_identity",
        "fold_bindings": [
            {"outer_fold": fold, "selected_candidate": "adjacent-skip2",
             "public_sha256": f"{fold + 11:x}" * 64,
             "restricted_predictions_sha256": f"{fold + 16:x}" * 64}
            for fold in range(5)
        ],
        "validation_rows_loaded": False, "average_target_used": False,
    }


def source_binding(source_id: str, kind: str, aggregate_path: Path, aggregate_sha: str,
                   upstream: dict) -> tuple[dict, dict]:
    if kind == "stage3_kure":
        folds = upstream["methods"][0]["fold_bindings"]
        restricted_key = "restricted_prediction_sha256"
        run_id, config_sha, method_id = upstream["run_id"], upstream["config_sha256"], "coral-natural"
    else:
        folds = upstream["fold_bindings"]
        restricted_key = "restricted_predictions_sha256"
        run_id, config_sha, method_id = upstream["run_id"], upstream["config_sha256"], "adjacent-skip2"
    normalized = [
        {"outer_fold": item["outer_fold"], "public_sha256": item["public_sha256"],
         "restricted_sha256": item[restricted_key]}
        for item in folds
    ]
    runtime = {
        "id": source_id, "kind": kind, "aggregate_path": str(aggregate_path),
        "aggregate_sha256": aggregate_sha, "upstream_run_id": run_id,
        "upstream_config_sha256": config_sha, "upstream_method_id": method_id,
        "fold_files": normalized,
    }
    report = {key: runtime[key] for key in (
        "id", "kind", "aggregate_sha256", "upstream_run_id", "upstream_config_sha256", "upstream_method_id")}
    report["folds"] = normalized
    if kind == "stage3_kure":
        report.update({"provenance": "standard_5fold_oof",
                       "upstream_method_inventory": ["coral-natural", "rps-natural"]})
    return runtime, report


def stage5_runtime(root: Path, runtime_sources: list[dict]) -> dict:
    return {
        "schema_version": "mal2026-conservative-oof-combiner-v1", "run_id": "stage5-unit",
        "output_root": str(root), "combination_mode": "preregistered_fixed_standard_oof",
        "fixed_partner_source_id": "stage3-coral",
        "fixed_partner_method_id": "coral-natural", "fixed_partner_weight": 0.2,
        "calibration_status": "unavailable_requires_genuinely_outer_nested_base_predictions",
        "sources": runtime_sources,
    }


def stage5(eligible: bool, runtime_sha: str, report_sources: list[dict]) -> dict:
    return {
        "schema_version": "mal2026-conservative-oof-combiner-v1", "status": "completed", "mode": "full_oof",
        "run_id": "stage5-unit", "config_sha256": runtime_sha,
        "combination_mode": "preregistered_fixed_standard_oof",
        "fixed_candidate": {"r0_weight": 0.8, "partner_weight": 0.2,
                            "partner_source_id": "stage3-coral", "partner_method_id": "coral-natural",
                            "calibration": "identity"},
        "calibration_status": "unavailable_requires_genuinely_outer_nested_base_predictions",
        "promotion_gate": {"eligible": eligible},
        "protected_output": "candidate" if eligible else "exact_r0_identity",
        "preregistration_sha256": "19a202f375e8e2d47be8d7f21ea015422f85a5d8b45eaccc090b6221cba58b35",
        "preregistration_commit": "722a46ba51399066e402dc7f9f3a67ea40e19ee0",
        "source_bindings": report_sources,
        "validation_rows_loaded": False, "average_target_used": False,
    }


class Fixture:
    def __init__(self, root: Path, *, s3: bool = False, s4: bool = False, s5: bool = False) -> None:
        self.root = root
        self.stage3_upstream = root / "stage3-upstream" / "aggregate.json"
        upstream3 = stage3_upstream()
        write_json(self.stage3_upstream, upstream3)

        self.stage3 = root / "stage3-unit" / "aggregate.json"
        self.stage3_runtime = root / "stage3-unit" / "runtime.json"
        write_json(self.stage3_runtime, stage3_runtime(self.stage3, self.stage3_upstream,
                                                       sha(self.stage3_upstream)))
        write_json(self.stage3, stage3_promotion(s3, sha(self.stage3_upstream), sha(self.stage3_runtime)))

        self.stage4 = root / "stage4.json"
        upstream4 = stage4(s4)
        write_json(self.stage4, upstream4)

        runtime3, report3 = source_binding("stage3-coral", "stage3_kure", self.stage3_upstream,
                                           sha(self.stage3_upstream), upstream3)
        runtime4, report4 = source_binding("stage4-npcr", "stage4_npcr", self.stage4,
                                           sha(self.stage4), upstream4)
        self.stage5 = root / "stage5-unit" / "aggregate.json"
        self.stage5_runtime = root / "stage5-unit" / "runtime.json"
        write_json(self.stage5_runtime, stage5_runtime(root, [runtime3, runtime4]))
        write_json(self.stage5, stage5(s5, sha(self.stage5_runtime), [report3, report4]))
        self.output = root / "decision.json"

    def mapping(self, deploy: list[dict] | None = None) -> dict:
        return {
            "schema_version": "mal2026-stage6-submission-decision-v1", "run_id": "stage6-unit",
            "stage6_preregistration_path": "configs/stage6_submission_prereg.v1.json",
            "stage6_preregistration_sha256": "7616e038dd0dcb8a10a15c09780ca178ff43700c132fa941ba4e050e2a8176e1",
            "stage6_preregistration_commit": "32b0a43eda5612284d5bd718c5afbce2be182eff",
            "h0_runtime_manifest": {"path": "deployment/runtime_manifest.r0.template.json",
                                    "sha256": "a44760a93d7f84d31aff2a6531cd5895044db64d394a373ffd154273c25b641d"},
            "h0_bundle_completion": {"path": "deployment/runtime_bundle_r0/bundle_complete.json",
                                     "sha256": "58c029d014b80ac9530a1e6e2535a235bb8e90f45b1713246b0919c11045e80f"},
            "h0_blind_adapter": {"path": "deployment/runtime_bundle_r0/rationale/adapters/rank2_ax4_random1/adapter_model.safetensors",
                                 "sha256": "39c68bb5c98da25eaa466434ba1c6d4a47bedcec580c991f00335627382a3a73"},
            "h0_final_adapter": {"path": "deployment/runtime_bundle_r0/rationale/adapters/final_dpo/adapter_model.safetensors",
                                 "sha256": "887abf9d1bf07693251a17b7a0fb655fe8203fa6945e9c178a38bdc538ded826"},
            "stage3_promotion_runtime_config": {"path": str(self.stage3_runtime), "sha256": sha(self.stage3_runtime)},
            "stage3_promotion_aggregate": {"path": str(self.stage3), "sha256": sha(self.stage3)},
            "stage3_upstream_aggregate": {"path": str(self.stage3_upstream), "sha256": sha(self.stage3_upstream)},
            "stage4_aggregate": {"path": str(self.stage4), "sha256": sha(self.stage4)},
            "stage5_runtime_config": {"path": str(self.stage5_runtime), "sha256": sha(self.stage5_runtime)},
            "stage5_aggregate": {"path": str(self.stage5), "sha256": sha(self.stage5)},
            "deploy_artifacts": deploy or [], "output_path": str(self.output),
        }

    def bound_preregistration(self, config: DecisionConfig) -> dict:
        value = json.loads(json.dumps(_load_preregistration(config)))
        value["upstream_preregistration"]["stage3"]["aggregate_path"] = str(self.stage3_upstream)
        value["upstream_preregistration"]["stage4"]["aggregate_path"] = str(self.stage4)
        return value


def materialize(fixture: Fixture, config: DecisionConfig) -> dict:
    with patch("mal2026.stage6_submission_decision._load_preregistration",
               side_effect=lambda value: fixture.bound_preregistration(value)):
        return dict(run(config))


class Stage6SubmissionDecisionTests(unittest.TestCase):
    def test_all_failed_gates_materialize_h0_only_with_grandfathered_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            result = materialize(fixture, DecisionConfig.from_mapping(fixture.mapping()))
            self.assertEqual("completed", result["status"])
            self.assertEqual(["H0"], [item["slot"] for item in result["submission_slots"]])
            self.assertEqual([], result["pending_deploy_artifacts"])
            self.assertEqual("grandfathered_not_reexecuted", result["h0_historical_evidence"]["status"])

    def test_input_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            config = DecisionConfig.from_mapping(fixture.mapping())
            fixture.stage3.write_text(fixture.stage3.read_text() + "\n")
            with self.assertRaisesRegex(Stage6SubmissionDecisionError, "Stage3 promotion aggregate checksum"):
                materialize(fixture, config)

    def test_rehashed_stage5_fixed_contract_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            value = json.loads(fixture.stage5.read_text())
            value["fixed_candidate"]["partner_weight"] = 0.3
            write_json(fixture.stage5, value)
            config = DecisionConfig.from_mapping(fixture.mapping())
            with self.assertRaisesRegex(Stage6SubmissionDecisionError, "Stage5 promotion evidence"):
                materialize(fixture, config)

    def test_passed_gate_without_trusted_full_refit_is_action_required_and_keeps_h0(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), s3=True, s5=True)
            result = materialize(fixture, DecisionConfig.from_mapping(fixture.mapping()))
            self.assertEqual("action_required", result["status"])
            self.assertEqual(["H0"], [item["slot"] for item in result["submission_slots"]])
            self.assertEqual(["stage3_coral_full_refit"], result["pending_deploy_artifacts"])

    def test_handcrafted_full_refit_metadata_still_requires_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), s3=True, s5=True)
            artifact = Path(temporary) / "coral.safetensors"
            artifact.write_bytes(b"synthetic-coral-artifact")
            completion = Path(temporary) / "coral-complete.json"
            seeds = json.loads(Path("configs/stage6_submission_prereg.v1.json").read_text())["full_refit_seed_contract"]["stage3_coral_natural"]
            write_json(completion, {
                "schema_version": "mal2026-stage6-deploy-completion-v1", "status": "completed",
                "artifact_id": "stage3_coral_full_refit", "artifact_sha256": sha(artifact),
                "upstream_run_id": "kure-ordinal-oof-v1-20260803-001",
                "upstream_config_file_sha256": STAGE3_CONFIG_SHA,
                "upstream_report_config_sha256": STAGE3_REPORT_SHA,
                "upstream_aggregate_sha256": sha(fixture.stage3_upstream),
                "phase1_seed_by_axis": seeds["phase1_seed_by_axis"],
                "crt_seed_by_axis": seeds["crt_seed_by_axis"], "sole_coral_refit": True,
            })
            deploy = [{"artifact_id": "stage3_coral_full_refit",
                       "artifact": {"path": str(artifact), "sha256": sha(artifact)},
                       "completion": {"path": str(completion), "sha256": sha(completion)}}]
            result = materialize(fixture, DecisionConfig.from_mapping(fixture.mapping(deploy)))
            self.assertEqual("action_required", result["status"])
            self.assertEqual(["H0"], [item["slot"] for item in result["submission_slots"]])
            self.assertEqual(["stage3_coral_full_refit"], result["pending_deploy_artifacts"])

    def test_three_field_forged_completion_is_action_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary), s3=True)
            artifact = Path(temporary) / "coral.safetensors"; artifact.write_bytes(b"artifact")
            completion = Path(temporary) / "forged.json"
            write_json(completion, {"schema_version": "mal2026-stage6-deploy-completion-v1",
                                    "status": "completed", "artifact_id": "stage3_coral_full_refit"})
            deploy = [{"artifact_id": "stage3_coral_full_refit",
                       "artifact": {"path": str(artifact), "sha256": sha(artifact)},
                       "completion": {"path": str(completion), "sha256": sha(completion)}}]
            result = materialize(fixture, DecisionConfig.from_mapping(fixture.mapping(deploy)))
            self.assertEqual("action_required", result["status"])
            self.assertEqual(["stage3_coral_full_refit"], result["pending_deploy_artifacts"])
            self.assertEqual(["H0"], [item["slot"] for item in result["submission_slots"]])

    def test_fresh_output_and_aggregate_only_privacy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            config = DecisionConfig.from_mapping(fixture.mapping())
            result = materialize(fixture, config)
            public = json.loads(fixture.output.read_text())
            self.assertEqual(result, public)
            serialized = fixture.output.read_text().lower()
            for forbidden in ("essay", "source_id", "row_prediction", "average_target", "validation_rows"):
                self.assertNotIn(forbidden, serialized)
            with self.assertRaisesRegex(Stage6SubmissionDecisionError, "overwrite"):
                materialize(fixture, config)


if __name__ == "__main__":
    unittest.main()
