"""Fail-closed preregistration bindings for the V12 rationale-semantic study."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .iterative_official_rationale_embedding_data import (
    EMBEDDING_DIM, FEATURE_DIM, MODEL_ID, MODEL_REVISION, PROJECTION_DIM, PROJECTION_SEED,
    RUN_ID as FEATURE_RUN_ID, SCHEMA_VERSION as FEATURE_SCHEMA,
    load_feature_artifact, matrix_sha256, rademacher_projection,
)


CONFIG_PATH = Path("configs/iterative_official_rationale_semantic.v12.json")
RUN_ID = "iterative-official-rationale-semantic-v12-20260802-001"
AWAITING = "awaiting_generated_feature_artifact"
BOUND = "bound_generated_feature_artifact"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_FIXED_LINEAGE = {
    "canonical_train_path": "eval/train.jsonl",
    "canonical_train_sha256": "b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737",
    "baseline_oof_path": "data/processed/restricted/r0_exact_oof_v1/r0-exact-oof-20260731-002/merged/oof_predictions.jsonl",
    "baseline_oof_sha256": "823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04",
    "fold_assignment_fingerprint": "8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db",
    "terra_candidate_manifest_path": "data/processed/restricted/official_openai_candidates_v1/official-openai-candidates-v1-train3-20260727-001/manifest.json",
    "terra_candidate_manifest_sha256": "960ae42ac19e79bd8cff747844ee58a5c724abe30d46d6f3865cb92e22b9de53",
    "terra_candidate_rows_path": "data/processed/restricted/official_openai_candidates_v1/official-openai-candidates-v1-train3-20260727-001/candidates.train.jsonl",
    "terra_candidate_rows_sha256": "a1791c418c79c0b76399ddb993e862f34209c2da95b0c13f7cda87f403a24e4c",
    "terra_model": "gpt-5.6-terra",
    "luna_candidate_manifest_path": "data/processed/restricted/official_openai_candidates_v1/official-openai-candidates-luna-v1-train3-20260802-001/manifest.json",
    "luna_candidate_manifest_sha256": "dbdb6265bd808c6d2e08cb3c05507fd015c2a561e29da495ad2961223ee04c47",
    "luna_candidate_rows_path": "data/processed/restricted/official_openai_candidates_v1/official-openai-candidates-luna-v1-train3-20260802-001/candidates.train.jsonl",
    "luna_candidate_rows_sha256": "1397e870cffdbb66a58d7e2732fadb4e7911bc97af0e7cde17ccf506b90486ac",
    "luna_model": "gpt-5.6-luna",
    "official_system_prompt_sha256": "ea0454665da8e13ffb606c1b0fc7f8323bd62dbad65a9a363e463df888bbe5e9",
    "v11_config_path": "configs/iterative_official_balanced_boundary.v11.json",
    "v11_config_sha256": "8556079fc62612634606b4cec3404c884ec75613d85bfe445f414762e68722e0",
    "v11_aggregate_path": "outputs/iterative-official-balanced-boundary-v11/iterative-official-balanced-boundary-v11-20260802-001/aggregate.json",
    "v11_aggregate_sha256": "0a8cd65cfe6e688641abd1ed72fd9bbcc1ba5b5c6ecb0273eda1d513f94e2af2",
    "v11_completion_path": "outputs/iterative-official-balanced-boundary-v11/iterative-official-balanced-boundary-v11-20260802-001/completion.json",
    "v11_completion_sha256": "a7e75d8829955233c943933da9edf5c4b3e60f08eaf882254b30fd9ada21faf1",
    "v11_result_role": "balanced_boundary_nested_development_improved_core_metrics_but_failed_final_macro_and_3_4_ba_gates",
    "qwen_model_id": MODEL_ID,
    "qwen_model_revision": MODEL_REVISION,
    "qwen_model_path": "outputs/model-cache/Qwen--Qwen3-Embedding-8B-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
    "qwen_model_config_path": "outputs/model-cache/Qwen--Qwen3-Embedding-8B-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/config.json",
    "qwen_model_config_sha256": "dcccf7c7890c8debc4af8dace8d6acd8dd50bd56d50d4d8b60529381ba3de3d7",
    "generated_feature_manifest_path": "data/processed/restricted/iterative_official_rationale_embeddings_v12/iterative-official-rationale-embeddings-v12-20260802-001/merged/manifest.json",
    "generated_feature_rows_path": "data/processed/restricted/iterative_official_rationale_embeddings_v12/iterative-official-rationale-embeddings-v12-20260802-001/merged/rows.jsonl",
    "generated_feature_public_manifest_path": "outputs/iterative-official-rationale-embeddings-v12/iterative-official-rationale-embeddings-v12-20260802-001/manifest.json",
}


class OfficialRationaleSemanticProtocolError(ValueError):
    """Raised when V12 configuration or a bound input differs."""


@dataclass(frozen=True)
class OfficialRationaleSemanticProtocol:
    path: Path
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class OfficialRationaleSemanticInputAudit:
    records: int
    folds: Mapping[int, int]
    fold_fingerprint: str
    semantic_dimensions: int
    feature_rows_sha256: str
    terra_candidates: int
    luna_candidates: int
    v11_aggregate_sha256: str


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialRationaleSemanticProtocolError(message)


def _hex(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound(root: Path, relative: object, expected: object, label: str) -> Path:
    _need(isinstance(relative, str) and bool(relative) and _hex(expected), f"V12 {label} binding is unresolved")
    path = root / relative
    _need(path.is_file() and not path.is_symlink() and _sha256(path) == expected, f"V12 bound input differs: {label}")
    return path


def _exact_gates(raw: Mapping[str, Any]) -> None:
    inner = {
        "operator": "AND", "macro_rmse_min_improvement": 0.005,
        "equal_group_rmse_min_improvement": 0.01, "low_tail_must_improve": True,
        "high_tail_must_improve": True, "gold_3_4_balanced_accuracy_min_improvement": 0.01,
        "max_axis_rmse_worsening": 0.01, "max_macro_spearman_fall": 0.005,
        "score1_descriptive_only": True, "require_all_four_inner_folds": True,
        "require_finite_metrics": True,
    }
    final = {
        "construction": "concatenate_five_outer_predictions_once",
        "selection_after_concatenation": False, "reference": "exact_r0_oof_baseline",
        "operator": "AND", "macro_rmse_min_improvement": 0.01,
        "paired_bootstrap": {"replicates": 10000, "quantity": "candidate_minus_baseline_macro_rmse", "confidence_interval": 0.95, "required_upper_bound_lt": 0.0},
        "low_tail_must_improve": True, "high_tail_must_improve": True,
        "gold_3_4_balanced_accuracy_min_improvement": 0.01,
        "max_axis_rmse_worsening": 0.01, "max_macro_spearman_fall": 0.005,
        "score1_descriptive_only": True, "require_all_five_outer_folds": True,
        "require_finite_metrics": True,
    }
    _need(raw.get("inner_promotion_gate") == inner, "V12 original seven inner gates differ")
    _need(raw.get("final_evaluation") == final, "V12 original final gates differ")


def validate_protocol_mapping(raw: Mapping[str, Any], *, path: Path = CONFIG_PATH) -> OfficialRationaleSemanticProtocol:
    _need(set(raw) == {"schema_version", "run_id", "binding_state", "lineage", "data_contract", "semantic_feature_contract", "execution", "candidates", "nested_protocol", "inner_promotion_gate", "final_evaluation", "failure_action", "success_action", "scientific_claims", "binding_transition_contract"}, "V12 top-level protocol fields differ")
    _need(raw.get("schema_version") == "mal2026-iterative-official-rationale-semantic-v12" and raw.get("run_id") == RUN_ID, "V12 identity differs")
    state = raw.get("binding_state")
    _need(state in {AWAITING, BOUND}, "V12 feature binding state differs")
    lineage = raw.get("lineage")
    _need(isinstance(lineage, Mapping), "V12 lineage missing")
    _need(set(lineage) == set(_FIXED_LINEAGE) | {"generated_feature_manifest_sha256", "generated_feature_rows_sha256", "generated_feature_public_manifest_sha256"}, "V12 lineage fields differ")
    _need(all(lineage.get(key) == value for key, value in _FIXED_LINEAGE.items()), "V12 fixed lineage differs")
    fixed_hashes = (
        "canonical_train_sha256", "baseline_oof_sha256", "terra_candidate_manifest_sha256",
        "terra_candidate_rows_sha256", "luna_candidate_manifest_sha256", "luna_candidate_rows_sha256",
        "official_system_prompt_sha256", "v11_config_sha256", "v11_aggregate_sha256",
        "v11_completion_sha256", "qwen_model_config_sha256", "fold_assignment_fingerprint",
    )
    _need(all(_hex(lineage.get(key)) for key in fixed_hashes), "V12 fixed lineage is unbound")
    feature_hashes = (lineage.get("generated_feature_manifest_sha256"), lineage.get("generated_feature_rows_sha256"), lineage.get("generated_feature_public_manifest_sha256"))
    _need(all(value is None for value in feature_hashes) if state == AWAITING else all(_hex(value) for value in feature_hashes), "V12 generated feature bindings differ from binding state")

    expected_data = {
        "records": 2000, "fold_count": 5, "records_per_fold": 400, "split_role": "train",
        "terra_candidate_records": 6000, "luna_candidate_records": 6000,
        "candidates_per_model_per_essay": 3, "rationale_text_only": True,
        "candidate_scores_used_in_semantic_features": False,
        "human_or_reference_score_read_or_prompted_to_candidate_models": False,
        "validation_loaded": False, "validation_selection": False,
        "average_target_used": False, "external_api_calls_in_v12": 0,
    }
    _need(raw.get("data_contract") == expected_data, "V12 train-only score-blind data contract differs")

    feature = raw.get("semantic_feature_contract")
    expected_feature = {
        "embedding_artifact_run_id": FEATURE_RUN_ID,
        "encoder_output_dimensions": EMBEDDING_DIM,
        "projection_dimensions": PROJECTION_DIM,
        "projection_method": "fixed_rademacher_matrix_scaled_by_inverse_sqrt_32",
        "projection_seed": PROJECTION_SEED,
        "axis_order": ["content", "organization", "expression"],
        "per_axis_contract": {
            "terra_centroid": "l2_normalized_mean_of_three_candidate_rationale_embeddings",
            "luna_centroid": "l2_normalized_mean_of_three_candidate_rationale_embeddings",
            "projected_pooled_centroid_mean_dimensions": 32,
            "projected_signed_terra_minus_luna_centroid_dimensions": 32,
            "scalar_order": ["terra_within_pair_cosine_mean", "luna_within_pair_cosine_mean", "terra_luna_centroid_cosine"],
            "scalar_dimensions": 3, "dimensions": 67,
        },
        "semantic_dimensions": FEATURE_DIM, "structured_v10_dimensions": 96,
        "fusion_dimensions": 297, "pooling": "last_nonpad_float32_l2",
        "input": "one_axis_rationale_text_only_no_candidate_score_no_essay_no_prompt",
        "target_blind_generation": True,
    }
    _need(feature == expected_feature, "V12 rationale-semantic feature contract differs")

    expected_execution = {
        "authorized_gpus": [0, 1, 2, 3], "smoke_gpu": 0,
        "smoke_required_before_full_run": True, "feature_generation_shards": 4,
        "fresh_residual_and_optional_head_each_candidate_and_fold": True,
        "checkpoint_reuse": False, "all_3_candidates_required": True,
        "early_stop_allowed": False, "dtype": "torch.float64",
    }
    _need(raw.get("execution") == expected_execution, "V12 fresh GPU/smoke execution contract differs")

    expected_candidates = [
        {"cycle": 1, "variant_id": "rationale-semantic201-ridge-a10-cap050", "feature_kind": "semantic201", "feature_dimensions": 201, "head_kind": "identity", "ridge_alpha": 10.0, "max_correction": 0.5, "fresh_initialization": True},
        {"cycle": 2, "variant_id": "rationale-fusion297-ridge-a10-cap050", "feature_kind": "fusion297", "feature_dimensions": 297, "head_kind": "identity", "ridge_alpha": 10.0, "max_correction": 0.5, "fresh_initialization": True},
        {"cycle": 3, "variant_id": "rationale-fusion297-balanced-3v4-l2-001-c055-w020", "feature_kind": "fusion297", "feature_dimensions": 297, "head_kind": "balanced_adjacent_3v4", "ridge_alpha": 10.0, "max_correction": 0.5, "l2": 0.01, "confidence": 0.55, "window": 0.2, "epsilon": 0.001, "fresh_initialization": True},
    ]
    _need(raw.get("candidates") == expected_candidates, "V12 exact three-candidate inventory/order differs")
    expected_nested = {
        "outer_folds": [0, 1, 2, 3, 4], "outer_holdout_use": "predict_once_after_selected_spec_freeze",
        "outer_gold_locked_until_prediction_complete": True,
        "inner_validation_rule": "D_each_of_the_other_four_folds",
        "inner_training_rule": "S_is_the_other_three_original_folds_excluding_O_and_D",
        "inner_fold_count_per_outer": 4,
        "candidate_oof_coverage": "each_outer_train_row_exactly_once_per_candidate",
        "selection_after_all_candidates_complete": True,
        "inner_selection": {"reference": "exact_r0_oof_baseline_on_outer_train", "gate": "fixed_original_seven_gate", "rule": "eligible_minimum_macro_rmse", "tie_break": "lowest_cycle_number", "no_eligible_fallback": "exact_r0_oof_baseline", "require_all_four_inner_folds": True, "require_finite_metrics": True},
        "outer_refit": {"selected_spec_frozen_before_refit": True, "selected_candidate_only": True, "fit_scope": "all_four_outer_train_folds", "initialization": "fresh_closed_form_residual_and_optional_fresh_zero_class_balanced_logistic", "checkpoint_reuse": False, "predict_scope": "O_once"},
        "posthoc_selection_on_concatenated_outer_predictions": False,
    }
    _need(raw.get("nested_protocol") == expected_nested, "V12 sealed 5x4 nesting differs")
    _exact_gates(raw)
    _need(raw.get("failure_action") == {"when": "final_gate_fails", "retained_model": "exact_r0_oof_baseline", "terminal_stop": True, "v12_inventory_frozen": True}, "V12 terminal failure action differs")
    _need(raw.get("success_action") == {"when": "final_gate_passes", "retained_model": "nested_selected_rationale_semantic_stack", "independent_hidden_evaluation_still_required": True, "v12_inventory_frozen": True}, "V12 success action differs")
    expected_claims = {
        "adaptive_after_v11_observed": True, "descriptive_development_only": True,
        "no_v12_full_oof_prestudy_before_preregistration": True,
        "same_train_nested_development_evidence_only": True,
        "terminal_stop_if_final_gate_fails": True,
        "independent_confirmation_claim_allowed": False,
        "generalization_claim_allowed": False, "deployment_claim_allowed": False,
    }
    _need(raw.get("scientific_claims") == expected_claims, "V12 adaptive descriptive-only claim limits differ")
    _need(raw.get("binding_transition_contract") == {"from": AWAITING, "to": BOUND, "only_mutable_fields": ["binding_state", "lineage.generated_feature_manifest_sha256", "lineage.generated_feature_rows_sha256", "lineage.generated_feature_public_manifest_sha256"], "scientific_protocol_changes_allowed": False}, "V12 checksum-only binding transition differs")
    return OfficialRationaleSemanticProtocol(path, raw)


def load_protocol(path: str | Path = CONFIG_PATH) -> OfficialRationaleSemanticProtocol:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialRationaleSemanticProtocolError("V12 protocol is unreadable") from exc
    _need(isinstance(raw, Mapping), "V12 protocol must be an object")
    return validate_protocol_mapping(raw, path=config_path)


def _ids(path: Path) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            source_id = row.get("id") if isinstance(row, Mapping) else None
            _need(isinstance(source_id, str) and source_id and source_id not in seen, f"V12 canonical identity differs at {line_number}")
            seen.add(source_id); result.append(source_id)
    _need(len(result) == 2000, "V12 canonical population differs")
    return tuple(result)


def _baseline_folds(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            _need(isinstance(row, Mapping), f"V12 baseline row differs at {line_number}")
            source_id, fold = row.get("source_id"), row.get("fold")
            _need(isinstance(source_id, str) and source_id and source_id not in result and type(fold) is int and 0 <= fold < 5, f"V12 baseline identity/fold differs at {line_number}")
            result[source_id] = fold
    _need(Counter(result.values()) == Counter({fold: 400 for fold in range(5)}), "V12 baseline folds differ")
    return result


def _generation_manifest(path: Path, rows_path: Path, *, model: str, rows_sha: str, prompt_sha: str) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(raw, Mapping) and raw.get("schema_version") == "mal2026-official-openai-candidate-v1" and raw.get("status") == "validated" and raw.get("model") == model, "V12 generation manifest identity differs")
    _need(raw.get("split") == "train" and raw.get("train_rows") == 2000 and raw.get("candidates_per_essay") == 3 and raw.get("requests") == 6000 and raw.get("accepted") == 6000, "V12 generation population differs")
    _need(raw.get("human_or_reference_score_read_or_prompted") is False and raw.get("official_system_prompt_sha256") == prompt_sha and raw.get("candidates_sha256") == rows_sha and _sha256(rows_path) == rows_sha, "V12 generation prompt/score-blind row binding differs")


def validate_bound_inputs(protocol: OfficialRationaleSemanticProtocol, *, root: str | Path = ".") -> OfficialRationaleSemanticInputAudit:
    _need(protocol.raw.get("binding_state") == BOUND, "V12 generated feature artifact is awaiting checksum-only binding commit")
    root_path, lineage = Path(root), protocol.raw["lineage"]
    keys = (
        ("canonical_train_path", "canonical_train_sha256"), ("baseline_oof_path", "baseline_oof_sha256"),
        ("terra_candidate_manifest_path", "terra_candidate_manifest_sha256"), ("terra_candidate_rows_path", "terra_candidate_rows_sha256"),
        ("luna_candidate_manifest_path", "luna_candidate_manifest_sha256"), ("luna_candidate_rows_path", "luna_candidate_rows_sha256"),
        ("v11_config_path", "v11_config_sha256"), ("v11_aggregate_path", "v11_aggregate_sha256"),
        ("v11_completion_path", "v11_completion_sha256"), ("qwen_model_config_path", "qwen_model_config_sha256"),
        ("generated_feature_manifest_path", "generated_feature_manifest_sha256"), ("generated_feature_rows_path", "generated_feature_rows_sha256"),
        ("generated_feature_public_manifest_path", "generated_feature_public_manifest_sha256"),
    )
    paths = {path_key: _bound(root_path, lineage.get(path_key), lineage.get(sha_key), path_key) for path_key, sha_key in keys}
    model_root = root_path / str(lineage["qwen_model_path"])
    _need(model_root.is_dir() and not model_root.is_symlink() and paths["qwen_model_config_path"].parent.resolve() == model_root.resolve(), "V12 Qwen snapshot path differs")
    model_config = json.loads(paths["qwen_model_config_path"].read_text(encoding="utf-8"))
    _need(model_config.get("model_type") == "qwen3" and model_config.get("hidden_size") == EMBEDDING_DIM and model_config.get("architectures") == ["Qwen3ForCausalLM"], "V12 Qwen config differs")
    _generation_manifest(paths["terra_candidate_manifest_path"], paths["terra_candidate_rows_path"], model="gpt-5.6-terra", rows_sha=str(lineage["terra_candidate_rows_sha256"]), prompt_sha=str(lineage["official_system_prompt_sha256"]))
    _generation_manifest(paths["luna_candidate_manifest_path"], paths["luna_candidate_rows_path"], model="gpt-5.6-luna", rows_sha=str(lineage["luna_candidate_rows_sha256"]), prompt_sha=str(lineage["official_system_prompt_sha256"]))
    v11 = json.loads(paths["v11_aggregate_path"].read_text(encoding="utf-8"))
    completion = json.loads(paths["v11_completion_path"].read_text(encoding="utf-8"))
    _need(v11.get("schema_version") == "mal2026-iterative-official-balanced-boundary-aggregate-v11" and v11.get("status") == "completed" and v11.get("final_gate_pass") is False, "V12 V11 aggregate differs")
    _need(completion.get("schema_version") == "mal2026-iterative-official-balanced-boundary-completion-v11" and completion.get("status") == "completed_no_promotion_baseline_retained" and completion.get("final_gate_pass") is False, "V12 V11 completion differs")

    canonical_ids = _ids(paths["canonical_train_path"])
    folds = _baseline_folds(paths["baseline_oof_path"])
    _need(set(folds) == set(canonical_ids), "V12 canonical/baseline identity population differs")
    fingerprint = sha256("\n".join(f"{source_id}:{fold}" for source_id, fold in sorted(folds.items())).encode()).hexdigest()
    _need(fingerprint == lineage["fold_assignment_fingerprint"], "V12 fold fingerprint differs")

    public_manifest = paths["generated_feature_public_manifest_path"]
    _need(_sha256(public_manifest) == lineage["generated_feature_public_manifest_sha256"], "V12 public feature manifest binding differs")
    _need(public_manifest.read_bytes() == paths["generated_feature_manifest_path"].read_bytes(), "V12 public/restricted feature manifests differ")
    try:
        manifest, feature_rows = load_feature_artifact(
            paths["generated_feature_manifest_path"], paths["generated_feature_rows_path"],
            expected_source_ids=canonical_ids,
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise OfficialRationaleSemanticProtocolError("V12 generated feature artifact differs") from exc
    _need(manifest.get("feature_rows_sha256") == lineage["generated_feature_rows_sha256"], "V12 generated feature row binding differs")
    _need(manifest.get("model_config_sha256") == lineage["qwen_model_config_sha256"] and manifest.get("projection_matrix_sha256") == matrix_sha256(rademacher_projection()), "V12 model/projection artifact binding differs")
    expected_render = {"kind": "participant_axis_rationale_text_alone_v1", "source_order": ["terra", "luna"], "candidate_order": [1, 2, 3], "axis_order": ["content", "organization", "expression"], "essay_included": False, "prompt_included": False, "participant_score_included": False, "gold_included": False}
    _need(manifest.get("render_contract") == expected_render and manifest.get("candidate_score_in_embedding_text") is False and manifest.get("essay_in_embedding_text") is False and manifest.get("prompt_in_embedding_text") is False and manifest.get("validation_loaded") is False, "V12 feature score-blind/validation contract differs")
    render_hash = sha256(json.dumps(expected_render, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    bindings = manifest.get("candidate_bindings")
    _need(isinstance(bindings, Mapping) and bindings.get("canonical_train_sha256") == lineage["canonical_train_sha256"] and bindings.get("terra_manifest_sha256") == lineage["terra_candidate_manifest_sha256"] and bindings.get("luna_manifest_sha256") == lineage["luna_candidate_manifest_sha256"] and bindings.get("official_system_prompt_sha256") == lineage["official_system_prompt_sha256"] and bindings.get("render_contract_sha256") == render_hash and manifest.get("render_contract_sha256") == render_hash, "V12 feature generation lineage differs")
    _need(len(feature_rows) == 2000 and set(row.source_id for row in feature_rows) == set(canonical_ids), "V12 generated feature population differs")
    return OfficialRationaleSemanticInputAudit(2000, dict(sorted(Counter(folds.values()).items())), fingerprint, 201, str(lineage["generated_feature_rows_sha256"]), 6000, 6000, str(lineage["v11_aggregate_sha256"]))


__all__ = [
    "AWAITING", "BOUND", "CONFIG_PATH", "FEATURE_RUN_ID", "FEATURE_SCHEMA", "RUN_ID",
    "OfficialRationaleSemanticInputAudit", "OfficialRationaleSemanticProtocol",
    "OfficialRationaleSemanticProtocolError", "load_protocol", "validate_bound_inputs",
    "validate_protocol_mapping",
]
