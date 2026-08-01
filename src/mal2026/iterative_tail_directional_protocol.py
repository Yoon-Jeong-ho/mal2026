"""Exact train-only nested-selection contract for iterative tail directional v6.

This protocol uses adaptive same-train nested evidence. It is neither an
independent confirmation nor evidence of generalization or deployment fitness.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
import importlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mal2026.r0_ordinal_residual import AXES, load_embedding_artifact

SCHEMA_VERSION = "mal2026-iterative-tail-directional-v6"
RUN_ID = "iterative-tail-directional-v6-20260802-001"
CONFIG_PATH = Path("configs/iterative_tail_directional.v6.json")
MODEL_MODULE = "mal2026.iterative_tail_directional_models"
EXPECTED_PROTOCOL = {'schema_version': 'mal2026-iterative-tail-directional-v6', 'run_id': 'iterative-tail-directional-v6-20260802-001', 'lineage': {'canonical_train_path': 'eval/train.jsonl', 'canonical_train_sha256': 'b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737', 'baseline_oof_path': 'data/processed/restricted/r0_exact_oof_v1/r0-exact-oof-20260731-002/merged/oof_predictions.jsonl', 'baseline_oof_sha256': '823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04', 'embedding_manifest_path': 'data/processed/restricted/r0_ordinal_residual_embeddings_v1/r0-public-frozen-embedding-20260731-001/train/merged/manifest.json', 'embedding_rows_path': 'data/processed/restricted/r0_ordinal_residual_embeddings_v1/r0-public-frozen-embedding-20260731-001/train/merged/rows.jsonl', 'embedding_rows_sha256': '949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16', 'fold_assignment_fingerprint': '8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db', 'score_blind_feature_cache_path': 'data/processed/restricted/iterative_tail_refinement_v1/iterative-tail-refinement-v1-20260801-001/score_blind_features.npz', 'score_blind_feature_cache_sha256': 'c2770d2add3b08a46614ad4c56fddc2d6b06ba5784c39e80c95c9f419e46d7db', 'score_blind_feature_manifest_path': 'data/processed/restricted/iterative_tail_refinement_v1/iterative-tail-refinement-v1-20260801-001/score_blind_features.manifest.json', 'score_blind_feature_manifest_sha256': 'ae3e44270fdeb4d6d217fc82ebd7097ee4cd7031343221b855a6ac7207cf38b0', 'v1_aggregate_path': 'outputs/iterative-tail-refinement-v1/iterative-tail-refinement-v1-20260801-001/promotion_summary.json', 'v1_aggregate_sha256': 'd3e0e2f7871518bf9123e554ad19afc764a5e257a7a9a087a9cdd1e466e3d0f7', 'v1_completion_path': 'outputs/iterative-tail-refinement-v1/iterative-tail-refinement-v1-20260801-001/completion.json', 'v1_completion_sha256': '432c24066c4fef3c3b6c09c638104378d85bd2717dc44fbeb5c3c28d4b2d7262', 'v2_aggregate_path': 'outputs/iterative-tail-remediation-v2/iterative-tail-remediation-v2-20260801-001/aggregate.json', 'v2_aggregate_sha256': 'bc64443aafc48c45f9a882cf861c11eba9b22d6d417b62d90c43a1fbf81af81f', 'v2_completion_path': 'outputs/iterative-tail-remediation-v2/iterative-tail-remediation-v2-20260801-001/completion.json', 'v2_completion_sha256': '101474406ef9c8400859c52cc6ffcc4c4d2d2dc237626cd2f04e17bf8fa1e3ea', 'v3_aggregate_path': 'outputs/iterative-tail-cycle-v3/iterative-tail-cycle-v3-20260801-001/aggregate.json', 'v3_aggregate_sha256': 'bc44123673b512ae20212a94f7a3e98b4d227823cd1d74d2fddf8f4a07e3667f', 'v3_completion_path': 'outputs/iterative-tail-cycle-v3/iterative-tail-cycle-v3-20260801-001/completion.json', 'v3_completion_sha256': '238c6074de6abf35aed02c1d95e76b8b1789edf88f78377d56eb101b506cd9d3', 'v4_aggregate_path': 'outputs/iterative-tail-router-v4/iterative-tail-router-v4-20260801-001/aggregate.json', 'v4_aggregate_sha256': '5c34e53706b1e2bc90ea24a402c9f9efc444549ba2ce12f01e7ac481bd978279', 'v4_completion_path': 'outputs/iterative-tail-router-v4/iterative-tail-router-v4-20260801-001/completion.json', 'v4_completion_sha256': '2e52992e9dd1dc80ff15f1179ff2f59228cd733eeb4f86ef8f4f50ebcf54c720', 'v5_aggregate_path': 'outputs/iterative-tail-learner-v5/iterative-tail-learner-v5-20260802-001/aggregate.json', 'v5_aggregate_sha256': 'eb7906883fbe91d93ab0928848c91ffa8448cd8fc278033caea3c0c06dd99705', 'v5_completion_path': 'outputs/iterative-tail-learner-v5/iterative-tail-learner-v5-20260802-001/completion.json', 'v5_completion_sha256': '37b96297e552c80793fa78a9dc2557b5b287e931061f88b545ae1c98cafbc34b', 'historical_artifact_role': 'adaptive_preregistration_and_falsification_evidence_only_forbidden_as_model_input'}, 'data_contract': {'records': 2000, 'fold_count': 5, 'records_per_fold': 400, 'split_role': 'train', 'baseline_origin': 'exact_oof', 'feature_input': {'frozen_embedding_dimensions': 4096, 'score_blind_evidence_hash_dimensions': 576, 'concatenated_dimensions': 4672}, 'historical_row_predictions_allowed': False, 'historical_row_errors_allowed': False, 'historical_learned_weights_allowed': False, 'historical_checkpoints_allowed': False, 'historical_pseudo_targets_allowed': False, 'validation_loaded': False, 'validation_selection': False, 'average_target_used': False, 'optional_api_enabled': False, 'external_api_calls_allowed': False}, 'random_projection': {'input_dimensions': 4672, 'output_dimensions': 64, 'seed': 2026080206, 'deterministic': True, 'fit_to_data': False, 'gold_used': False, 'normalization': 'fixed_seed_random_projection_only', 'generator': 'numpy.random.default_rng_PCG64', 'matrix_distribution': 'rademacher_pm1_over_sqrt_output_dimensions'}, 'execution': {'authorized_gpus': [0, 1, 2, 3], 'smoke_gpu': 0, 'smoke_required_before_full_run': True, 'initialization_seed': 2026080206, 'same_initialization_across_candidates': True, 'fresh_initialization_each_candidate_expert_and_fold': True, 'checkpoint_reuse': False, 'all_3_candidates_required': True, 'early_stop_allowed': False}, 'authorization_and_freeze': {'study_role': 'materially_distinct_v6_under_users_ongoing_iterative_goal', 'v5_reopening': False, 'v4_inventory_permanently_frozen': True, 'v4_learned_weights_permanently_frozen': True, 'v5_inventory_permanently_frozen': True, 'v5_learned_weights_permanently_frozen': True, 'v4_v5_posthoc_tuning_allowed': False, 'v4_v5_artifact_modification_allowed': False}, 'nested_protocol': {'outer_folds': [0, 1, 2, 3, 4], 'outer_holdout_symbol': 'O', 'outer_holdout_use': 'predict_once_after_selected_spec_freeze', 'outer_gold_locked_until_prediction_complete': True, 'inner_validation_rule': 'D_each_of_the_other_four_folds', 'inner_training_rule': 'S_is_the_other_three_original_folds_excluding_O_and_D', 'inner_fold_count_per_outer': 4, 'candidate_inner_fit': 'each_candidate_fresh_fit_on_S_predict_D_once', 'internal_expert_crossfit': {'scope': 'S_only', 'split_unit': 'original_folds', 'procedure': 'for_each_original_fold_in_S_fit_expert_on_other_two_predict_held_fold', 'purpose': 'derive_strict_OOF_benefit_labels', 'D_allowed': False, 'O_allowed': False}, 'candidate_oof_construction': 'concatenate_four_D_predictions_to_1600_rows', 'candidate_oof_coverage': 'each_outer_train_row_exactly_once_per_candidate', 'selection_after_all_candidates_complete': True, 'inner_selection': {'reference': 'exact_r0_oof_baseline_on_outer_train', 'gate': 'fixed_eight_gate', 'rule': 'eligible_minimum_macro_rmse', 'tie_break': 'lowest_cycle_number', 'no_eligible_fallback': 'exact_r0_oof_baseline', 'require_all_four_inner_folds': True, 'require_finite_metrics': True}, 'outer_refit': {'selected_spec_frozen_before_refit': True, 'selected_candidate_only': True, 'fit_scope': 'all_four_outer_train_folds', 'internal_expert_crossfit': 'for_each_original_fold_fit_on_other_three_predict_held_fold_for_OOF_benefit_labels', 'initialization': 'fresh', 'checkpoint_reuse': False, 'predict_scope': 'O_once'}, 'historical_predictions_used': False, 'historical_row_errors_used': False, 'historical_weights_used': False, 'historical_checkpoints_used': False, 'historical_pseudo_targets_used': False}, 'candidates': [{'cycle': 1, 'family': 'crossfit_safe_directional_residual', 'variant_id': 'crossfit_safe_directional_residual-primary', 'parameters': {'nonlinear': True, 'hidden': 64, 'benefit_margin': 0.01, 'identity_bias': 4.0, 'cap_low': 0.4, 'cap_high': 0.8, 'cap_center': 0.08}, 'initialization_seed': 2026080206, 'fresh_initialization': True}, {'cycle': 2, 'family': 'crossfit_safe_directional_residual', 'variant_id': 'crossfit_safe_directional_residual-conservative', 'parameters': {'nonlinear': True, 'hidden': 64, 'benefit_margin': 0.02, 'identity_bias': 4.5, 'cap_low': 0.4, 'cap_high': 0.8, 'cap_center': 0.08}, 'initialization_seed': 2026080206, 'fresh_initialization': True}, {'cycle': 3, 'family': 'crossfit_safe_directional_residual', 'variant_id': 'crossfit_safe_directional_residual-linear-safety-control', 'parameters': {'nonlinear': False, 'hidden': 64, 'benefit_margin': 0.01, 'identity_bias': 4.0, 'cap_low': 0.4, 'cap_high': 0.8, 'cap_center': 0.08}, 'initialization_seed': 2026080206, 'fresh_initialization': True}], 'inner_promotion_gate': {'operator': 'AND', 'macro_rmse_min_improvement': 0.005, 'equal_group_rmse_min_improvement': 0.01, 'low_tail_must_improve': True, 'high_tail_must_improve': True, 'gold_3_4_balanced_accuracy_min_improvement': 0.01, 'max_axis_rmse_worsening': 0.01, 'max_macro_spearman_fall': 0.005, 'macro_score5_recall_min_improvement': 0.01, 'score1_descriptive_only': True, 'require_all_four_inner_folds': True, 'require_finite_metrics': True}, 'final_evaluation': {'construction': 'concatenate_five_outer_predictions_once', 'selection_after_concatenation': False, 'reference': 'exact_r0_oof_baseline', 'operator': 'AND', 'macro_rmse_min_improvement': 0.01, 'paired_bootstrap': {'replicates': 10000, 'quantity': 'candidate_minus_baseline_macro_rmse', 'confidence_interval': 0.95, 'required_upper_bound_lt': 0.0}, 'low_tail_must_improve': True, 'high_tail_must_improve': True, 'gold_3_4_balanced_accuracy_min_improvement': 0.01, 'max_axis_rmse_worsening': 0.01, 'max_macro_spearman_fall': 0.005, 'macro_score5_recall_min_improvement': 0.01, 'score1_descriptive_only': True, 'require_all_five_outer_folds': True, 'require_finite_metrics': True}, 'failure_action': {'when': 'final_gate_fails', 'retained_model': 'exact_r0_oof_baseline', 'v6_inventory_frozen': True, 'terminal_freeze_same_train_and_feature_sources': True}, 'scientific_claims': {'adaptive_after_v1_through_v5_observed': True, 'same_train_nested_descriptive_and_falsification_evidence_only': True, 'independent_confirmation_claim_allowed': False, 'generalization_claim_allowed': False, 'deployment_claim_allowed': False}, 'agent_evidence': {'role': 'preregistration_evidence_only', 'model_feature_allowed': False, 'pseudo_target_allowed': False, 'reward_or_weight_allowed': False}, 'training': {'learning_rate': 0.0003, 'weight_decay': 0.001, 'batch_size': 128, 'epochs': 30, 'gradient_clip': 1.0}}
EXPECTED_CANDIDATE_SPECS = tuple(EXPECTED_PROTOCOL["candidates"])

class IterativeTailDirectionalProtocolError(ValueError):
    """Raised before registered v6 nesting, isolation, or claims can drift."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise IterativeTailDirectionalProtocolError(message)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    _need(not path.is_absolute() and ".." not in path.parts, "input path must be repository-relative")
    return root / path


def candidate_specs() -> tuple[Mapping[str, Any], ...]:
    """Return an isolated copy of the registered three-candidate inventory."""
    return tuple(deepcopy(spec) for spec in EXPECTED_CANDIDATE_SPECS)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _normalize_model_specs(raw: Sequence[Any]) -> tuple[Mapping[str, Any], ...]:
    normalized = []
    for item in raw:
        if is_dataclass(item):
            item = {field.name: getattr(item, field.name) for field in fields(item)}
        _need(isinstance(item, Mapping), "model candidate_specs entries must be mappings or dataclasses")
        normalized.append(_json_value(dict(item)))
    return tuple(normalized)


def validate_model_inventory(*, require_available: bool = False) -> bool:
    """Require exact equality with ``iterative_tail_directional_models.candidate_specs()``.

    Absence is tolerated for protocol-only compilation. Runner preflight must
    call this with ``require_available=True``.
    """
    try:
        module = importlib.import_module(MODEL_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != MODEL_MODULE:
            raise
        _need(not require_available, "v6 directional model module is required for execution")
        return False
    inventory_function = getattr(module, "candidate_specs", None)
    _need(callable(inventory_function), "v6 directional model module must expose candidate_specs()")
    _need({
        "learning_rate": getattr(module, "LEARNING_RATE", None),
        "weight_decay": getattr(module, "WEIGHT_DECAY", None),
        "batch_size": getattr(module, "BATCH_SIZE", None),
        "epochs": getattr(module, "EPOCHS", None),
        "gradient_clip": getattr(module, "GRAD_CLIP", None),
    } == EXPECTED_PROTOCOL["training"], "model common training constants differ from v6 registration")
    raw_inventory = tuple(inventory_function())
    actual_full = _normalize_model_specs(raw_inventory)
    _need(all(item.get("seed") == 2026080206 and item.get("device") == "cpu" for item in actual_full),
          "model candidate seed or default device differs from v6 registration")
    actual = tuple({key: item[key] for key in ("cycle", "family", "variant_id", "parameters")}
                   for item in actual_full)
    expected = tuple({
        "cycle": item["cycle"], "family": item["family"],
        "variant_id": item["variant_id"], "parameters": item["parameters"],
    } for item in EXPECTED_CANDIDATE_SPECS)
    _need(actual == expected, "model candidate_specs inventory differs from the exact v6 registration")
    return True


@dataclass(frozen=True)
class DirectionalProtocol:
    path: Path
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class DirectionalInputAudit:
    records: int
    fold_counts: Mapping[int, int]
    fold_assignment_fingerprint: str
    baseline_sha256: str
    evidence_cache_sha256: str
    historical_sha256: Mapping[str, str]
    historical_artifact_role: str
    model_inventory_available: bool


def validate_protocol_mapping(raw: Mapping[str, Any], *, path: str | Path = CONFIG_PATH) -> DirectionalProtocol:
    """Exact-lock lineage, nested selection, gates, stop rule, and claims."""
    _need(isinstance(raw, Mapping) and dict(raw) == EXPECTED_PROTOCOL, "v6 protocol differs from exact registration")
    _need(raw["schema_version"] == SCHEMA_VERSION and raw["run_id"] == RUN_ID, "v6 identity differs")
    candidates = raw["candidates"]
    _need(len(candidates) == 3 and [item["cycle"] for item in candidates] == [1, 2, 3], "v6 requires exact candidates 1..3")
    _need([item["family"] for item in candidates] == ["crossfit_safe_directional_residual"] * 3, "v6 family differs")
    _need([item["variant_id"] for item in candidates] == [
        "crossfit_safe_directional_residual-primary",
        "crossfit_safe_directional_residual-conservative",
        "crossfit_safe_directional_residual-linear-safety-control",
    ], "v6 candidate order differs")
    execution = raw["execution"]
    _need(execution["all_3_candidates_required"] and not execution["early_stop_allowed"], "all three candidates must run")
    _need(execution["same_initialization_across_candidates"] and execution["fresh_initialization_each_candidate_expert_and_fold"], "initialization policy differs")
    _need(not execution["checkpoint_reuse"], "checkpoint reuse is forbidden")
    projection = raw["random_projection"]
    _need(projection == {"input_dimensions": 4672, "output_dimensions": 64, "seed": 2026080206,
                         "deterministic": True, "fit_to_data": False, "gold_used": False,
                         "normalization": "fixed_seed_random_projection_only",
                         "generator": "numpy.random.default_rng_PCG64",
                         "matrix_distribution": "rademacher_pm1_over_sqrt_output_dimensions"},
          "random projection contract differs")
    _need(raw["training"] == {"learning_rate": 0.0003, "weight_decay": 0.001, "batch_size": 128,
                              "epochs": 30, "gradient_clip": 1.0}, "common training contract differs")
    nested = raw["nested_protocol"]
    _need(nested["inner_fold_count_per_outer"] == 4 and nested["outer_gold_locked_until_prediction_complete"], "outer gold must remain locked")
    _need(nested["inner_training_rule"] == "S_is_the_other_three_original_folds_excluding_O_and_D", "inner S/D/O isolation differs")
    _need(nested["candidate_inner_fit"] == "each_candidate_fresh_fit_on_S_predict_D_once", "candidate S-to-D fit differs")
    _need(nested["candidate_oof_coverage"] == "each_outer_train_row_exactly_once_per_candidate", "candidate OOF coverage differs")
    _need(nested["selection_after_all_candidates_complete"], "selection must wait for all candidates")
    internal = nested["internal_expert_crossfit"]
    _need(internal["scope"] == "S_only" and internal["split_unit"] == "original_folds", "internal expert scope differs")
    _need(not internal["D_allowed"] and not internal["O_allowed"] and internal["purpose"] == "derive_strict_OOF_benefit_labels", "benefit labels are not strictly OOF")
    outer = nested["outer_refit"]
    _need(outer["selected_spec_frozen_before_refit"] and outer["selected_candidate_only"] and outer["initialization"] == "fresh", "selected candidate freeze/refit differs")
    _need(not any(nested[key] for key in ("historical_predictions_used", "historical_row_errors_used", "historical_weights_used", "historical_checkpoints_used", "historical_pseudo_targets_used")), "historical row artifacts are forbidden")
    inner_gate = raw["inner_promotion_gate"]
    _need(inner_gate["operator"] == "AND" and inner_gate["macro_score5_recall_min_improvement"] == 0.01, "inner eight gates must be conjunctive")
    final = raw["final_evaluation"]
    _need(final["construction"] == "concatenate_five_outer_predictions_once" and not final["selection_after_concatenation"], "final concatenation cannot trigger selection")
    _need(final["macro_rmse_min_improvement"] == 0.01 and final["paired_bootstrap"]["replicates"] == 10000, "final macro/bootstrap gate differs")
    _need(final["macro_score5_recall_min_improvement"] == 0.01, "final score5 recall gate differs")
    _need(final["paired_bootstrap"]["quantity"] == "candidate_minus_baseline_macro_rmse" and final["paired_bootstrap"]["required_upper_bound_lt"] == 0.0, "paired-bootstrap direction differs")
    claims = raw["scientific_claims"]
    _need(claims["same_train_nested_descriptive_and_falsification_evidence_only"], "adaptive same-train status must be explicit")
    _need(not claims["independent_confirmation_claim_allowed"] and not claims["generalization_claim_allowed"] and not claims["deployment_claim_allowed"], "v6 cannot make independent, generalization, or deployment claims")
    _need(raw["failure_action"] == {"when": "final_gate_fails", "retained_model": "exact_r0_oof_baseline", "v6_inventory_frozen": True, "terminal_freeze_same_train_and_feature_sources": True}, "terminal failure action differs")
    freeze = raw["authorization_and_freeze"]
    _need(freeze["study_role"] == "materially_distinct_v6_under_users_ongoing_iterative_goal" and not freeze["v5_reopening"], "V6 authorization scope differs")
    _need(all(freeze[key] for key in ("v4_inventory_permanently_frozen", "v4_learned_weights_permanently_frozen", "v5_inventory_permanently_frozen", "v5_learned_weights_permanently_frozen")), "V4/V5 artifacts must remain frozen")
    _need(not freeze["v4_v5_posthoc_tuning_allowed"] and not freeze["v4_v5_artifact_modification_allowed"], "V4/V5 reopening is forbidden")
    agent = raw["agent_evidence"]
    _need(not agent["model_feature_allowed"] and not agent["pseudo_target_allowed"] and not agent["reward_or_weight_allowed"], "agent opinions are preregistration evidence only")
    validate_model_inventory(require_available=False)
    return DirectionalProtocol(Path(path), raw)


def load_protocol(path: str | Path = CONFIG_PATH) -> DirectionalProtocol:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IterativeTailDirectionalProtocolError(f"v6 protocol is unreadable: {config_path}") from exc
    return validate_protocol_mapping(raw, path=config_path)


def _baseline_assignments(path: Path) -> dict[str, int]:
    required = {"source_id", "fold", "continuous_prediction", "half_up_integer_prediction", "reference_score"}
    result: dict[str, int] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            _need(isinstance(row, Mapping) and set(row) == required, f"baseline schema differs at line {line_number}")
            source_id, fold = row["source_id"], row["fold"]
            _need(isinstance(source_id, str) and source_id and source_id not in result, "baseline source IDs differ")
            _need(type(fold) is int and 0 <= fold < 5, "baseline fold differs")
            scores = row["continuous_prediction"]
            _need(isinstance(scores, Mapping) and set(scores) == set(AXES), "baseline axes differ or average was included")
            _need(all(type(scores[axis]) in {int, float} and math.isfinite(float(scores[axis])) for axis in AXES), "baseline scores must be finite")
            result[source_id] = fold
    return result


def validate_bound_inputs(protocol: DirectionalProtocol, *, root: str | Path = ".") -> DirectionalInputAudit:
    """Validate fixed artifacts, aggregates, and the exact 2,000/5x400 folds."""
    root_path = Path(root)
    lineage = protocol.raw["lineage"]
    checks = (
        ("canonical_train_path", "canonical_train_sha256", "canonical train"),
        ("baseline_oof_path", "baseline_oof_sha256", "exact R0 OOF baseline"),
        ("embedding_rows_path", "embedding_rows_sha256", "frozen embedding rows"),
        ("score_blind_feature_cache_path", "score_blind_feature_cache_sha256", "score-blind feature cache"),
        ("score_blind_feature_manifest_path", "score_blind_feature_manifest_sha256", "score-blind feature manifest"),
        *((f"v{version}_{kind}_path", f"v{version}_{kind}_sha256", f"historical v{version} {kind}")
          for version in range(1, 6) for kind in ("aggregate", "completion")),
    )
    resolved: dict[str, Path] = {}
    for path_key, sha_key, label in checks:
        artifact = _bound_path(root_path, lineage[path_key])
        _need(artifact.is_file() and not artifact.is_symlink(), f"{label} path is missing or mutable")
        _need(_sha256(artifact) == lineage[sha_key], f"{label} checksum differs")
        resolved[path_key] = artifact
    aggregates = (
        ("v1_aggregate", "mal2026-iterative-tail-promotion-v1", None),
        ("v2_aggregate", "mal2026-iterative-tail-remediation-aggregate-v2", 2000),
        ("v3_aggregate", "mal2026-iterative-tail-cycle-aggregate-v3", 2000),
        ("v4_aggregate", "mal2026-iterative-tail-router-aggregate-v4", 2000),
        ("v5_aggregate", "mal2026-iterative-tail-learner-aggregate-v5", 2000),
    )
    for name, schema, record_count in aggregates:
        try:
            artifact = json.loads(resolved[f"{name}_path"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IterativeTailDirectionalProtocolError(f"historical {name} artifact is unreadable") from exc
        _need(artifact.get("schema_version") == schema and artifact.get("status") == "completed", f"historical {name} identity differs")
        if record_count is not None:
            _need(artifact.get("record_count") == record_count, f"historical {name} population differs")
        _need(artifact.get("validation_loaded") is False and artifact.get("average_target_used") is False, f"historical {name} isolation differs")
    completion_schemas = {
        1: "mal2026-iterative-tail-completion-v1", 2: "mal2026-iterative-tail-remediation-completion-v2",
        3: "mal2026-iterative-tail-cycle-completion-v3", 4: "mal2026-iterative-tail-router-completion-v4",
        5: "mal2026-iterative-tail-learner-completion-v5",
    }
    for version, schema in completion_schemas.items():
        name = f"v{version}_completion"
        try:
            artifact = json.loads(resolved[f"{name}_path"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IterativeTailDirectionalProtocolError(f"historical {name} is unreadable") from exc
        _need(artifact.get("schema_version") == schema and str(artifact.get("status", "")).startswith("completed"), f"historical {name} identity differs")
        _need(artifact.get("validation_loaded") is False and artifact.get("average_target_used") is False, f"historical {name} isolation differs")
    _need(lineage["historical_artifact_role"] == "adaptive_preregistration_and_falsification_evidence_only_forbidden_as_model_input", "historical artifact role differs")

    embedding_manifest_path = _bound_path(root_path, lineage["embedding_manifest_path"])
    _need(embedding_manifest_path.is_file() and not embedding_manifest_path.is_symlink(), "embedding manifest path is missing or mutable")
    try:
        feature_manifest = json.loads(resolved["score_blind_feature_manifest_path"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IterativeTailDirectionalProtocolError("score-blind feature manifest is unreadable") from exc
    _need(feature_manifest.get("records") == 2000 and feature_manifest.get("axes") == list(AXES), "evidence population or axes differ")
    _need(feature_manifest.get("hash_dim") == 96 and feature_manifest.get("cache_sha256") == lineage["score_blind_feature_cache_sha256"], "evidence hash binding differs")
    _need(feature_manifest.get("score_conditioning") is False and feature_manifest.get("validation_loaded") is False, "evidence is not score-blind train-only")

    manifest, embedding_rows = load_embedding_artifact(embedding_manifest_path, resolved["embedding_rows_path"])
    _need(manifest.split_role == "train" and manifest.base_prediction_origin == "oof", "embeddings must be train OOF")
    _need(not manifest.evaluation_only and not manifest.contains_average_target, "validation or average target is forbidden")
    baseline = _baseline_assignments(resolved["baseline_oof_path"])
    embedding = {row.source_id: row.oof_fold for row in embedding_rows}
    _need(len(baseline) == len(embedding) == 2000 and baseline == embedding, "baseline/embedding population or folds differ")
    counts = Counter(baseline.values())
    _need(counts == Counter({fold: 400 for fold in range(5)}), "v6 requires exactly 400 rows per fold")
    payload = "\n".join(f"{source_id}:{fold}" for source_id, fold in sorted(baseline.items()))
    fingerprint = sha256(payload.encode("utf-8")).hexdigest()
    _need(fingerprint == lineage["fold_assignment_fingerprint"], "fold assignment fingerprint differs")
    historical_sha256 = {
        f"v{version}_{kind}": lineage[f"v{version}_{kind}_sha256"]
        for version in range(1, 6) for kind in ("aggregate", "completion")
    }
    return DirectionalInputAudit(
        2000, dict(sorted(counts.items())), fingerprint, lineage["baseline_oof_sha256"],
        lineage["score_blind_feature_cache_sha256"], historical_sha256,
        lineage["historical_artifact_role"], validate_model_inventory(require_available=False),
    )


load_config = load_protocol
