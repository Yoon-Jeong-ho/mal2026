"""Fail-closed materializer for the preregistered Stage6 submission slots."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SCHEMA_VERSION = "mal2026-stage6-submission-decision-v1"
PREREG_PATH = "configs/stage6_submission_prereg.v1.json"
PREREG_SHA256 = "7616e038dd0dcb8a10a15c09780ca178ff43700c132fa941ba4e050e2a8176e1"
PREREG_COMMIT = "32b0a43eda5612284d5bd718c5afbce2be182eff"


class Stage6SubmissionDecisionError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Stage6SubmissionDecisionError(message)


def file_sha256(path: str | Path) -> str:
    location = Path(path)
    need(location.is_file() and not location.is_symlink(), "artifact must be an ordinary file")
    digest = sha256()
    with location.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contains_forbidden_config_data(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_forbidden_config_data(key) or _contains_forbidden_config_data(child)
                   for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_config_data(child) for child in value)
    return isinstance(value, str) and any(token in value.lower() for token in
                                          ("validation", "average", "essay", "source_id", "row_prediction"))


@dataclass(frozen=True)
class ArtifactBinding:
    path: str
    sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ArtifactBinding":
        need(isinstance(raw, Mapping) and set(raw) == {"path", "sha256"}, "artifact binding fields differ")
        value = cls(str(raw["path"]), str(raw["sha256"]))
        need(bool(value.path) and len(value.sha256) == 64, "artifact binding differs")
        return value

    def verify(self, label: str) -> None:
        need(file_sha256(self.path) == self.sha256, f"{label} checksum differs")


@dataclass(frozen=True)
class DeployArtifact:
    artifact_id: str
    artifact: ArtifactBinding
    completion: ArtifactBinding

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DeployArtifact":
        need(isinstance(raw, Mapping) and set(raw) == {"artifact_id", "artifact", "completion"},
             "deploy artifact fields differ")
        value = cls(str(raw["artifact_id"]), ArtifactBinding.from_mapping(raw["artifact"]),
                    ArtifactBinding.from_mapping(raw["completion"]))
        need(value.artifact_id in {"stage3_coral_full_refit", "stage4_npcr_full_refit"},
             "deploy artifact identity differs")
        return value


@dataclass(frozen=True)
class DecisionConfig:
    schema_version: str
    run_id: str
    stage6_preregistration_path: str
    stage6_preregistration_sha256: str
    stage6_preregistration_commit: str
    h0_runtime_manifest: ArtifactBinding
    h0_bundle_completion: ArtifactBinding
    h0_blind_adapter: ArtifactBinding
    h0_final_adapter: ArtifactBinding
    stage3_promotion_runtime_config: ArtifactBinding
    stage3_promotion_aggregate: ArtifactBinding
    stage3_upstream_aggregate: ArtifactBinding
    stage4_aggregate: ArtifactBinding
    stage5_runtime_config: ArtifactBinding
    stage5_aggregate: ArtifactBinding
    deploy_artifacts: tuple[DeployArtifact, ...]
    output_path: str
    config_sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, digest: str | None = None) -> "DecisionConfig":
        need(isinstance(raw, Mapping) and not _contains_forbidden_config_data(raw),
             "runtime config contains forbidden data")
        normalized = dict(raw)
        for key in ("h0_runtime_manifest", "h0_bundle_completion", "h0_blind_adapter", "h0_final_adapter",
                    "stage3_promotion_runtime_config", "stage3_promotion_aggregate", "stage3_upstream_aggregate",
                    "stage4_aggregate", "stage5_runtime_config", "stage5_aggregate"):
            need(isinstance(normalized.get(key), Mapping), f"{key} must be an artifact binding")
            normalized[key] = ArtifactBinding.from_mapping(normalized[key])
        need(isinstance(normalized.get("deploy_artifacts"), list), "deploy_artifacts must be a list")
        normalized["deploy_artifacts"] = tuple(DeployArtifact.from_mapping(item) for item in normalized["deploy_artifacts"])
        normalized["config_sha256"] = digest or sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        need(set(normalized) == set(cls.__dataclass_fields__), "decision config fields differ")
        value = cls(**normalized)
        value.validate()
        return value

    @classmethod
    def from_json(cls, path: str | Path) -> "DecisionConfig":
        location = Path(path)
        try:
            raw = json.loads(location.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Stage6SubmissionDecisionError("decision config is unreadable") from exc
        return cls.from_mapping(raw, digest=file_sha256(location))

    def validate(self) -> None:
        need(self.schema_version == SCHEMA_VERSION and bool(self.run_id), "decision identity differs")
        need(self.stage6_preregistration_path == PREREG_PATH
             and self.stage6_preregistration_sha256 == PREREG_SHA256
             and self.stage6_preregistration_commit == PREREG_COMMIT,
             "Stage6 preregistration binding differs")
        need(len(self.config_sha256) == 64, "config checksum differs")
        identities = [item.artifact_id for item in self.deploy_artifacts]
        need(len(identities) == len(set(identities)), "deploy artifacts must be unique")


def _load_preregistration(config: DecisionConfig) -> Mapping[str, Any]:
    need(file_sha256(config.stage6_preregistration_path) == PREREG_SHA256,
         "Stage6 preregistration checksum differs")
    prereg = json.loads(Path(config.stage6_preregistration_path).read_text(encoding="utf-8"))
    need(prereg.get("schema_version") == "mal2026-stage6-submission-prereg-v1"
         and prereg.get("status") == "preregistered_non_runnable_pending_upstream_artifacts"
         and [item.get("slot") for item in prereg.get("decision_list", ())] == ["H0", "H1", "H2"],
         "Stage6 preregistration decision contract differs")
    audit = prereg.get("validation_descriptive_audit", {}).get("H0", {})
    need(audit.get("status") == "grandfathered_historical_deployable_context_only"
         and audit.get("reevaluation_forbidden") is True,
         "H0 historical evidence contract differs")
    h0 = prereg.get("h0_final_refit_map", {})
    final = prereg.get("deployment_freeze", {}).get("h1_h2_final_rationale", {})
    blind = prereg.get("deployment_freeze", {}).get("h1_stage4_feature_and_fallback_contract", {}).get(
        "blind_rationale_generator", {})
    need((config.h0_runtime_manifest.path, config.h0_runtime_manifest.sha256)
         == (h0.get("runtime_manifest_path"), h0.get("runtime_manifest_sha256"))
         and (config.h0_bundle_completion.path, config.h0_bundle_completion.sha256)
         == (h0.get("bundle_completion_path"), h0.get("bundle_completion_sha256"))
         and (config.h0_blind_adapter.path, config.h0_blind_adapter.sha256)
         == (f"{blind.get('adapter_path')}/adapter_model.safetensors", blind.get("adapter_sha256"))
         and (config.h0_final_adapter.path, config.h0_final_adapter.sha256)
         == (f"{final.get('adapter_path')}/adapter_model.safetensors", final.get("adapter_sha256")),
         "H0 preregistered artifact binding differs")
    return prereg


def _load_json(binding: ArtifactBinding, label: str) -> Mapping[str, Any]:
    binding.verify(label)
    try:
        value = json.loads(Path(binding.path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Stage6SubmissionDecisionError(f"{label} is invalid JSON") from exc
    need(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _verify_h0(config: DecisionConfig) -> None:
    for label, binding in (("H0 runtime manifest", config.h0_runtime_manifest),
                           ("H0 blind adapter", config.h0_blind_adapter),
                           ("H0 final adapter", config.h0_final_adapter)):
        binding.verify(label)
    completion = _load_json(config.h0_bundle_completion, "H0 bundle completion")
    need(completion.get("status") == "completed"
         and completion.get("candidate") == "historical_r0_prediction_ensemble_dpo_v1"
         and completion.get("blind_adapter_sha256") == config.h0_blind_adapter.sha256
         and completion.get("final_adapter_sha256") == config.h0_final_adapter.sha256,
         "H0 completion/adapter evidence differs")


def _verify_stage3(config: DecisionConfig, prereg: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime = _load_json(config.stage3_promotion_runtime_config, "Stage3 promotion runtime config")
    upstream = _load_json(config.stage3_upstream_aggregate, "Stage3 upstream aggregate")
    value = _load_json(config.stage3_promotion_aggregate, "Stage3 promotion aggregate")
    gate = value.get("promotion_gate")
    expected_upstream_path = prereg["upstream_preregistration"]["stage3"]["aggregate_path"]
    need(runtime.get("schema_version") == "mal2026-stage3-coral-promotion-v1"
         and runtime.get("run_id") == value.get("run_id")
         and runtime.get("output_path") == config.stage3_promotion_aggregate.path
         and runtime.get("stage3_aggregate_path") == expected_upstream_path == config.stage3_upstream_aggregate.path
         and runtime.get("stage3_aggregate_sha256") == config.stage3_upstream_aggregate.sha256
         and runtime.get("stage3_config_file_sha256")
             == "5108e37c490482fbbf45c043f2ad4d0154048d432b4a4a163b3d102f8c6d756e"
         and runtime.get("stage3_report_config_sha256")
             == "28e6ba7465b91ba1fcc306f97f10a8e213d4a6e0069ba499fd92199d827a15aa"
         and isinstance(runtime.get("outer_bindings"), list) and len(runtime["outer_bindings"]) == 5
         and value.get("schema_version") == "mal2026-stage3-coral-promotion-v1"
         and value.get("status") == "completed" and value.get("method") == "coral-natural"
         and value.get("rps_eligible") is False and isinstance(value.get("eligible"), bool)
         and isinstance(gate, Mapping) and gate.get("eligible") is value["eligible"]
         and value.get("stage6_preregistration_sha256") == PREREG_SHA256
         and value.get("stage6_preregistration_commit") == PREREG_COMMIT
         and value.get("stage3_config_file_sha256")
             == "5108e37c490482fbbf45c043f2ad4d0154048d432b4a4a163b3d102f8c6d756e"
         and value.get("stage3_report_config_sha256")
             == "28e6ba7465b91ba1fcc306f97f10a8e213d4a6e0069ba499fd92199d827a15aa"
         and isinstance(value.get("stage3_aggregate_sha256"), str)
         and len(value["stage3_aggregate_sha256"]) == 64
         and value.get("stage3_aggregate_sha256") == config.stage3_upstream_aggregate.sha256
         and isinstance(value.get("config_sha256"), str) and len(value["config_sha256"]) == 64
         and value.get("config_sha256") == config.stage3_promotion_runtime_config.sha256
         and isinstance(value.get("outer_bindings"), list) and len(value["outer_bindings"]) == 5
         and Path(config.stage3_promotion_aggregate.path).name == "aggregate.json"
         and Path(config.stage3_promotion_aggregate.path).parent.name == value.get("run_id")
         and value.get("validation_rows_loaded") is False and value.get("average_target_used") is False,
         "Stage3 promotion evidence differs")
    need(upstream.get("schema_version") == "mal2026-kure-ordinal-oof-aggregate-v1"
         and upstream.get("status") == "completed" and upstream.get("run_id") == "kure-ordinal-oof-v1-20260803-001"
         and upstream.get("config_sha256") == runtime["stage3_report_config_sha256"]
         and upstream.get("records") == 2000 and upstream.get("folds") == 5,
         "Stage3 upstream aggregate lineage differs")
    coral = next((item for item in upstream.get("methods", ()) if item.get("method") == "coral-natural"), None)
    need(isinstance(coral, Mapping) and isinstance(coral.get("fold_bindings"), list)
         and len(coral["fold_bindings"]) == 5, "Stage3 upstream coral bindings differ")
    for runtime_fold, report_fold, upstream_fold in zip(runtime["outer_bindings"], value["outer_bindings"],
                                                        coral["fold_bindings"], strict=True):
        need(runtime_fold.get("outer_fold") == report_fold.get("outer_fold") == upstream_fold.get("outer_fold")
             and runtime_fold.get("public_sha256") == report_fold.get("public_sha256") == upstream_fold.get("public_sha256")
             and runtime_fold.get("coral_restricted_sha256") == report_fold.get("coral_restricted_sha256")
                 == upstream_fold.get("restricted_prediction_sha256"),
             "Stage3 outer hash trust chain differs")
    return {"eligible": bool(value["eligible"]), "stage3_aggregate_sha256": value["stage3_aggregate_sha256"],
            "producer_config_sha256": value["config_sha256"], "aggregate": upstream}


def _verify_stage4(config: DecisionConfig, prereg: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _load_json(config.stage4_aggregate, "Stage4 aggregate")
    gate = value.get("promotion_gate_vs_exact_r0")
    expected = prereg["upstream_preregistration"]["stage4"]
    need(config.stage4_aggregate.path == expected["aggregate_path"]
         and value.get("schema_version") == "mal2026-prompt-reference-npcr-v1"
         and value.get("status") == "completed" and value.get("mode") == "full"
         and value.get("run_id") == expected["run_id"] and value.get("config_sha256") == expected["config_file_sha256"]
         and value.get("records") == 2000 and value.get("folds") == 5
         and isinstance(gate, Mapping) and isinstance(gate.get("eligible"), bool)
         and value.get("global_recommendation") == ("npcr" if gate["eligible"] else "exact_r0_identity")
         and value.get("validation_rows_loaded") is False and value.get("average_target_used") is False,
         "Stage4 promotion evidence differs")
    selected = [item.get("selected_candidate") for item in value.get("fold_bindings", ())]
    need(len(selected) == 5 and all(item in {"adjacent-allskip", "adjacent-skip2"} for item in selected),
         "Stage4 selected-candidate bindings differ")
    counts = {item: selected.count(item) for item in set(selected)}
    modal = sorted(counts, key=lambda item: (-counts[item], item))[0]
    return {"eligible": bool(gate["eligible"]), "modal_candidate_id": modal, "aggregate": value}


def _verify_stage5(config: DecisionConfig, stage3: Mapping[str, Any], stage4: Mapping[str, Any]) -> bool:
    runtime = _load_json(config.stage5_runtime_config, "Stage5 runtime config")
    value = _load_json(config.stage5_aggregate, "Stage5 aggregate")
    gate = value.get("promotion_gate")
    fixed = value.get("fixed_candidate")
    sources = value.get("source_bindings")
    need(runtime.get("schema_version") == "mal2026-conservative-oof-combiner-v1"
         and runtime.get("run_id") == value.get("run_id")
         and str(Path(runtime.get("output_root", "")) / runtime.get("run_id", "") / "aggregate.json")
             == config.stage5_aggregate.path
         and runtime.get("combination_mode") == "preregistered_fixed_standard_oof"
         and runtime.get("fixed_partner_source_id") == "stage3-coral"
         and runtime.get("fixed_partner_method_id") == "coral-natural"
         and runtime.get("fixed_partner_weight") == 0.2
         and runtime.get("calibration_status") == "unavailable_requires_genuinely_outer_nested_base_predictions"
         and value.get("schema_version") == "mal2026-conservative-oof-combiner-v1"
         and value.get("status") == "completed" and value.get("mode") == "full_oof"
         and value.get("combination_mode") == "preregistered_fixed_standard_oof"
         and fixed == {"r0_weight": 0.8, "partner_weight": 0.2,
                       "partner_source_id": "stage3-coral", "partner_method_id": "coral-natural",
                       "calibration": "identity"}
         and value.get("calibration_status") == "unavailable_requires_genuinely_outer_nested_base_predictions"
         and isinstance(gate, Mapping) and isinstance(gate.get("eligible"), bool)
         and value.get("protected_output") == ("candidate" if gate["eligible"] else "exact_r0_identity")
         and value.get("preregistration_sha256") == "19a202f375e8e2d47be8d7f21ea015422f85a5d8b45eaccc090b6221cba58b35"
         and value.get("preregistration_commit") == "722a46ba51399066e402dc7f9f3a67ea40e19ee0"
         and isinstance(value.get("config_sha256"), str) and len(value["config_sha256"]) == 64
         and value.get("config_sha256") == config.stage5_runtime_config.sha256
         and Path(config.stage5_aggregate.path).name == "aggregate.json"
         and Path(config.stage5_aggregate.path).parent.name == value.get("run_id")
         and isinstance(sources, list)
         and value.get("validation_rows_loaded") is False and value.get("average_target_used") is False,
         "Stage5 promotion evidence differs")
    runtime_sources = runtime.get("sources")
    need(isinstance(runtime_sources, list) and all(isinstance(item, Mapping) for item in runtime_sources)
         and {item.get("kind") for item in runtime_sources}
         == {"stage3_kure", "stage4_npcr"}, "Stage5 runtime source inventory differs")
    report_by_id = {item.get("id"): item for item in sources}
    for source in runtime_sources:
        report = report_by_id.get(source.get("id"))
        need(isinstance(report, Mapping) and report.get("kind") == source.get("kind")
             and report.get("upstream_run_id") == source.get("upstream_run_id")
             and report.get("upstream_config_sha256") == source.get("upstream_config_sha256")
             and report.get("upstream_method_id") == source.get("upstream_method_id")
             and report.get("aggregate_sha256") == source.get("aggregate_sha256")
             and report.get("folds") == [{"outer_fold": item.get("outer_fold"),
                                          "public_sha256": item.get("public_sha256"),
                                          "restricted_sha256": item.get("restricted_sha256")}
                                         for item in source.get("fold_files", ())],
             "Stage5 runtime/report source binding differs")
        expected_binding = config.stage3_upstream_aggregate if source["kind"] == "stage3_kure" else config.stage4_aggregate
        expected_aggregate = stage3["aggregate"] if source["kind"] == "stage3_kure" else stage4["aggregate"]
        need((source.get("aggregate_path"), source.get("aggregate_sha256"))
             == (expected_binding.path, expected_binding.sha256), "Stage5 source aggregate binding differs")
        method_key = "method" if source["kind"] == "stage3_kure" else None
        method = (next((item for item in expected_aggregate.get("methods", ())
                        if item.get(method_key) == source.get("upstream_method_id")), None)
                  if method_key else expected_aggregate)
        upstream_folds = method.get("fold_bindings", ()) if isinstance(method, Mapping) else ()
        need(len(upstream_folds) == len(source.get("fold_files", ())) == 5, "Stage5 source fold inventory differs")
        restricted_key = "restricted_prediction_sha256" if source["kind"] == "stage3_kure" else "restricted_predictions_sha256"
        need(all(left.get("outer_fold") == right.get("outer_fold")
                 and left.get("public_sha256") == right.get("public_sha256")
                 and left.get("restricted_sha256") == right.get(restricted_key)
                 for left, right in zip(source["fold_files"], upstream_folds, strict=True)),
             "Stage5 source fold hash binding differs")
    partner = next((item for item in sources if item.get("id") == fixed["partner_source_id"]), None)
    need(isinstance(partner, Mapping) and partner.get("kind") == "stage3_kure"
         and partner.get("provenance") == "standard_5fold_oof"
         and partner.get("upstream_run_id") == "kure-ordinal-oof-v1-20260803-001"
         and partner.get("upstream_config_sha256") == "28e6ba7465b91ba1fcc306f97f10a8e213d4a6e0069ba499fd92199d827a15aa"
         and partner.get("upstream_method_id") == "coral-natural"
         and "coral-natural" in partner.get("upstream_method_inventory", ())
         and partner.get("aggregate_sha256") == stage3["stage3_aggregate_sha256"]
         and isinstance(partner.get("folds"), list) and len(partner["folds"]) == 5,
         "Stage5 preregistered Stage3 source binding differs")
    return bool(gate["eligible"])


def _write_fresh(path: Path, payload: Mapping[str, Any]) -> None:
    need(not path.exists(), "refusing to overwrite Stage6 decision")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def run(config: DecisionConfig | str | Path) -> Mapping[str, Any]:
    value = DecisionConfig.from_json(config) if isinstance(config, (str, Path)) else config
    value.validate()
    prereg = _load_preregistration(value)
    _verify_h0(value)
    stage3 = _verify_stage3(value, prereg)
    stage4 = _verify_stage4(value, prereg)
    stage5 = _verify_stage5(value, stage3, stage4)
    h1 = "stage3_coral_full_refit" if stage3["eligible"] else "stage4_npcr_full_refit" if stage4["eligible"] else None
    h2 = "stage3_coral_full_refit" if stage5 else None
    required = {item for item in (h1, h2) if item is not None}
    pending = sorted(required)

    slots = [{"slot": "H0", "candidate": "historical_r0_prediction_ensemble_dpo_v1",
              "completion_sha256": value.h0_bundle_completion.sha256}]
    status = "action_required" if pending else "completed"
    result = {
        "schema_version": SCHEMA_VERSION, "status": status, "run_id": value.run_id,
        "submission_slots": slots, "pending_deploy_artifacts": pending,
        "gate_evidence": {"stage3_coral": stage3["eligible"],
                          "stage4_npcr": stage4["eligible"],
                          "stage5_fixed_blend": stage5},
        "h0_historical_evidence": {"status": "grandfathered_not_reexecuted",
                                   "bundle_completion_sha256": value.h0_bundle_completion.sha256},
        "stage6_preregistration_sha256": value.stage6_preregistration_sha256,
        "stage6_preregistration_commit": value.stage6_preregistration_commit,
        "input_aggregate_sha256": {"stage3": value.stage3_promotion_aggregate.sha256,
                                   "stage4": value.stage4_aggregate.sha256,
                                   "stage5": value.stage5_aggregate.sha256},
        "config_sha256": value.config_sha256,
        "privacy": "aggregate_only_no_individual_content",
    }
    _write_fresh(Path(value.output_path), result)
    return result


__all__ = ["ArtifactBinding", "DecisionConfig", "DeployArtifact", "SCHEMA_VERSION",
           "Stage6SubmissionDecisionError", "run"]
