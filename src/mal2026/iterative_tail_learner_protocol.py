"""Exact train-only nested-selection contract for iterative tail learner v5.

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

SCHEMA_VERSION = "mal2026-iterative-tail-learner-v5"
RUN_ID = "iterative-tail-learner-v5-20260802-001"
CONFIG_PATH = Path("configs/iterative_tail_learner.v5.json")
MODEL_MODULE = "mal2026.iterative_tail_learner_models"
EXPECTED_PROTOCOL = {'schema_version': 'mal2026-iterative-tail-learner-v5', 'run_id': 'iterative-tail-learner-v5-20260802-001', 'lineage': {'canonical_train_path': 'eval/train.jsonl', 'canonical_train_sha256': 'b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737', 'baseline_oof_path': 'data/processed/restricted/r0_exact_oof_v1/r0-exact-oof-20260731-002/merged/oof_predictions.jsonl', 'baseline_oof_sha256': '823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04', 'embedding_manifest_path': 'data/processed/restricted/r0_ordinal_residual_embeddings_v1/r0-public-frozen-embedding-20260731-001/train/merged/manifest.json', 'embedding_rows_path': 'data/processed/restricted/r0_ordinal_residual_embeddings_v1/r0-public-frozen-embedding-20260731-001/train/merged/rows.jsonl', 'embedding_rows_sha256': '949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16', 'fold_assignment_fingerprint': '8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db', 'score_blind_feature_cache_path': 'data/processed/restricted/iterative_tail_refinement_v1/iterative-tail-refinement-v1-20260801-001/score_blind_features.npz', 'score_blind_feature_cache_sha256': 'c2770d2add3b08a46614ad4c56fddc2d6b06ba5784c39e80c95c9f419e46d7db', 'score_blind_feature_manifest_path': 'data/processed/restricted/iterative_tail_refinement_v1/iterative-tail-refinement-v1-20260801-001/score_blind_features.manifest.json', 'score_blind_feature_manifest_sha256': 'ae3e44270fdeb4d6d217fc82ebd7097ee4cd7031343221b855a6ac7207cf38b0', 'v1_promotion_path': 'outputs/iterative-tail-refinement-v1/iterative-tail-refinement-v1-20260801-001/promotion_summary.json', 'v1_promotion_sha256': 'd3e0e2f7871518bf9123e554ad19afc764a5e257a7a9a087a9cdd1e466e3d0f7', 'v2_aggregate_path': 'outputs/iterative-tail-remediation-v2/iterative-tail-remediation-v2-20260801-001/aggregate.json', 'v2_aggregate_sha256': 'bc64443aafc48c45f9a882cf861c11eba9b22d6d417b62d90c43a1fbf81af81f', 'v3_aggregate_path': 'outputs/iterative-tail-cycle-v3/iterative-tail-cycle-v3-20260801-001/aggregate.json', 'v3_aggregate_sha256': 'bc44123673b512ae20212a94f7a3e98b4d227823cd1d74d2fddf8f4a07e3667f', 'v4_aggregate_path': 'outputs/iterative-tail-router-v4/iterative-tail-router-v4-20260801-001/aggregate.json', 'v4_aggregate_sha256': '5c34e53706b1e2bc90ea24a402c9f9efc444549ba2ce12f01e7ac481bd978279', 'historical_artifact_role': 'adaptive_preregistration_evidence_only_forbidden_as_model_input'}, 'data_contract': {'records': 2000, 'fold_count': 5, 'records_per_fold': 400, 'split_role': 'train', 'baseline_origin': 'exact_oof', 'allowed_model_inputs': ['exact_r0_oof_baseline', 'frozen_train_oof_embeddings', 'score_blind_evidence_hash'], 'historical_row_predictions_allowed': False, 'historical_learned_weights_allowed': False, 'historical_pseudo_targets_allowed': False, 'validation_loaded': False, 'validation_selection': False, 'average_target_used': False, 'optional_api_enabled': False, 'external_api_calls_allowed': False}, 'execution': {'authorized_gpus': [0, 1, 2, 3], 'smoke_gpu': 0, 'smoke_required_before_full_run': True, 'initialization_seed': 2026080205, 'same_initialization_across_candidates': True, 'fresh_initialization_each_candidate_and_fold': True, 'checkpoint_reuse': False, 'all_20_candidates_required': True, 'early_stop_allowed': False}, 'v4_stop_supersession': {'authorization_date': '2026-08-02', 'authorization_statement': 'V4 실패 후보 동결이지 프로젝트 전체 중단 아님; V5 신규20 계속', 'scope': 'separately_named_exactly_preregistered_v5_only', 'v4_inventory_permanently_frozen': True, 'v4_learned_weights_permanently_frozen': True, 'v4_posthoc_tuning_allowed': False, 'v4_artifact_modification_allowed': False, 'project_wide_stop_superseded': True}, 'nested_protocol': {'outer_folds': [0, 1, 2, 3, 4], 'outer_holdout_symbol': 'O', 'outer_holdout_use': 'predict_once_after_selected_spec_freeze', 'outer_gold_locked_until_prediction_complete': True, 'inner_validation_rule': 'D_each_of_the_other_four_folds', 'inner_training_rule': 'S_is_the_other_three_folds_excluding_O_and_D', 'inner_fold_count_per_outer': 4, 'candidate_inner_fit': 'each_candidate_fresh_fit_on_S_predict_D_once', 'candidate_oof_construction': 'concatenate_four_D_predictions_to_1600_rows', 'candidate_oof_coverage': 'each_outer_train_row_exactly_once_per_candidate', 'selection_after_all_candidates_complete': True, 'inner_selection': {'reference': 'exact_r0_oof_baseline_on_outer_train', 'gate': 'fixed_seven_gate', 'rule': 'eligible_minimum_macro_rmse', 'tie_break': 'lowest_cycle_number', 'no_eligible_fallback': 'exact_r0_oof_baseline', 'require_all_four_inner_folds': True, 'require_finite_metrics': True}, 'outer_refit': {'selected_spec_frozen_before_refit': True, 'selected_candidate_only': True, 'fit_scope': 'all_four_outer_train_folds', 'initialization': 'fresh', 'checkpoint_reuse': False, 'predict_scope': 'O_once'}, 'historical_v1_predictions_used': False, 'historical_v2_predictions_used': False, 'historical_v3_predictions_used': False, 'historical_v4_predictions_used': False, 'historical_weights_used': False, 'historical_pseudo_targets_used': False}, 'candidates': [{'cycle': 1, 'family': 'anchored_multitask_residual', 'variant_id': 'anchored_multitask_residual-v1', 'parameters': {'bottleneck': 128, 'hidden': 128, 'cap': 0.15, 'band': 0.25, 'ordinal': 0.1, 'boundary': 0.1}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 2, 'family': 'anchored_multitask_residual', 'variant_id': 'anchored_multitask_residual-v2', 'parameters': {'bottleneck': 128, 'hidden': 256, 'cap': 0.2, 'band': 0.5, 'ordinal': 0.15, 'boundary': 0.2}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 3, 'family': 'anchored_multitask_residual', 'variant_id': 'anchored_multitask_residual-v3', 'parameters': {'bottleneck': 256, 'hidden': 256, 'cap': 0.25, 'band': 0.75, 'ordinal': 0.2, 'boundary': 0.3}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 4, 'family': 'anchored_multitask_residual', 'variant_id': 'anchored_multitask_residual-v4', 'parameters': {'bottleneck': 256, 'hidden': 384, 'cap': 0.3, 'band': 1.0, 'ordinal': 0.25, 'boundary': 0.4}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 5, 'family': 'r0_anchored_distributional', 'variant_id': 'r0_anchored_distributional-v1', 'parameters': {'hidden': 128, 'max_mix': 0.15, 'class_weight': 0.25, 'margin': 0.1, 'temperature': 1.0}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 6, 'family': 'r0_anchored_distributional', 'variant_id': 'r0_anchored_distributional-v2', 'parameters': {'hidden': 192, 'max_mix': 0.2, 'class_weight': 0.5, 'margin': 0.15, 'temperature': 0.8}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 7, 'family': 'r0_anchored_distributional', 'variant_id': 'r0_anchored_distributional-v3', 'parameters': {'hidden': 256, 'max_mix': 0.25, 'class_weight': 0.75, 'margin': 0.2, 'temperature': 0.7}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 8, 'family': 'r0_anchored_distributional', 'variant_id': 'r0_anchored_distributional-v4', 'parameters': {'hidden': 256, 'max_mix': 0.3, 'class_weight': 1.0, 'margin': 0.3, 'temperature': 0.6}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 9, 'family': 'joint_tail_boundary_hurdle', 'variant_id': 'joint_tail_boundary_hurdle-v1', 'parameters': {'hidden': 128, 'expert': 64, 'cap': 0.15, 'gate': 0.25, 'boundary': 0.25}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 10, 'family': 'joint_tail_boundary_hurdle', 'variant_id': 'joint_tail_boundary_hurdle-v2', 'parameters': {'hidden': 192, 'expert': 64, 'cap': 0.2, 'gate': 0.5, 'boundary': 0.35}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 11, 'family': 'joint_tail_boundary_hurdle', 'variant_id': 'joint_tail_boundary_hurdle-v3', 'parameters': {'hidden': 256, 'expert': 96, 'cap': 0.25, 'gate': 0.75, 'boundary': 0.5}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 12, 'family': 'joint_tail_boundary_hurdle', 'variant_id': 'joint_tail_boundary_hurdle-v4', 'parameters': {'hidden': 256, 'expert': 128, 'cap': 0.3, 'gate': 1.0, 'boundary': 0.75}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 13, 'family': 'axis_coupled_lowrank_moe', 'variant_id': 'axis_coupled_lowrank_moe-v1', 'parameters': {'hidden': 128, 'cap': 0.15, 'identity_floor': 0.7, 'energy': 0.01, 'entropy': 0.001}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 14, 'family': 'axis_coupled_lowrank_moe', 'variant_id': 'axis_coupled_lowrank_moe-v2', 'parameters': {'hidden': 192, 'cap': 0.2, 'identity_floor': 0.6, 'energy': 0.005, 'entropy': 0.001}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 15, 'family': 'axis_coupled_lowrank_moe', 'variant_id': 'axis_coupled_lowrank_moe-v3', 'parameters': {'hidden': 256, 'cap': 0.25, 'identity_floor': 0.5, 'energy': 0.002, 'entropy': 0.0005}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 16, 'family': 'axis_coupled_lowrank_moe', 'variant_id': 'axis_coupled_lowrank_moe-v4', 'parameters': {'hidden': 256, 'cap': 0.3, 'identity_floor': 0.4, 'energy': 0.001, 'entropy': 0.0002}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 17, 'family': 'band_risk_pareto_residual', 'variant_id': 'band_risk_pareto_residual-v1', 'parameters': {'hidden': 128, 'cap': 0.15, 'risk': 0.1, 'temperature': 0.2, 'boundary': 0.1, 'ranking': 0.02}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 18, 'family': 'band_risk_pareto_residual', 'variant_id': 'band_risk_pareto_residual-v2', 'parameters': {'hidden': 192, 'cap': 0.2, 'risk': 0.2, 'temperature': 0.15, 'boundary': 0.2, 'ranking': 0.03}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 19, 'family': 'band_risk_pareto_residual', 'variant_id': 'band_risk_pareto_residual-v3', 'parameters': {'hidden': 256, 'cap': 0.25, 'risk': 0.3, 'temperature': 0.1, 'boundary': 0.3, 'ranking': 0.05}, 'initialization_seed': 2026080205, 'fresh_initialization': True}, {'cycle': 20, 'family': 'band_risk_pareto_residual', 'variant_id': 'band_risk_pareto_residual-v4', 'parameters': {'hidden': 256, 'cap': 0.3, 'risk': 0.4, 'temperature': 0.075, 'boundary': 0.4, 'ranking': 0.075}, 'initialization_seed': 2026080205, 'fresh_initialization': True}], 'inner_promotion_gate': {'operator': 'AND', 'macro_rmse_min_improvement': 0.005, 'equal_group_rmse_min_improvement': 0.01, 'low_tail_must_improve': True, 'high_tail_must_improve': True, 'gold_3_4_balanced_accuracy_min_improvement': 0.01, 'max_axis_rmse_worsening': 0.01, 'max_macro_spearman_fall': 0.005, 'score1_descriptive_only': True, 'require_all_four_inner_folds': True, 'require_finite_metrics': True}, 'final_evaluation': {'construction': 'concatenate_five_outer_predictions_once', 'selection_after_concatenation': False, 'reference': 'exact_r0_oof_baseline', 'operator': 'AND', 'macro_rmse_min_improvement': 0.01, 'paired_bootstrap': {'replicates': 10000, 'quantity': 'candidate_minus_baseline_macro_rmse', 'confidence_interval': 0.95, 'required_upper_bound_lt': 0.0}, 'low_tail_must_improve': True, 'high_tail_must_improve': True, 'gold_3_4_balanced_accuracy_min_improvement': 0.01, 'max_axis_rmse_worsening': 0.01, 'max_macro_spearman_fall': 0.005, 'score1_descriptive_only': True, 'require_all_five_outer_folds': True, 'require_finite_metrics': True}, 'failure_action': {'when': 'final_gate_fails', 'retained_model': 'exact_r0_oof_baseline', 'v5_inventory_frozen': True}, 'scientific_claims': {'adaptive_after_v1_v2_v3_v4_observed': True, 'same_train_nested_descriptive_evidence_only': True, 'independent_confirmation_claim_allowed': False, 'generalization_claim_allowed': False, 'deployment_claim_allowed': False}, 'agent_evidence': {'role': 'preregistration_evidence_only', 'model_feature_allowed': False, 'pseudo_target_allowed': False, 'reward_or_weight_allowed': False}}
EXPECTED_CANDIDATE_SPECS = tuple(EXPECTED_PROTOCOL["candidates"])

class IterativeTailLearnerProtocolError(ValueError):
    """Raised before registered v5 nesting, isolation, or claims can drift."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise IterativeTailLearnerProtocolError(message)


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
    """Return an isolated copy of the registered 20-candidate inventory."""
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
    """Require exact equality with ``iterative_tail_learner_models.candidate_specs()``.

    Absence is tolerated for protocol-only compilation. Runner preflight must
    call this with ``require_available=True``.
    """
    try:
        module = importlib.import_module(MODEL_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != MODEL_MODULE:
            raise
        _need(not require_available, "v5 learner model module is required for execution")
        return False
    inventory_function = getattr(module, "candidate_specs", None)
    _need(callable(inventory_function), "v5 learner model module must expose candidate_specs()")
    raw_inventory = tuple(inventory_function())
    actual_full = _normalize_model_specs(raw_inventory)
    _need(all(item.get("seed") == 2026080205 and item.get("device") == "cpu" for item in actual_full),
          "model candidate seed or default device differs from v5 registration")
    actual = tuple({key: item[key] for key in ("cycle", "family", "variant_id", "parameters")}
                   for item in actual_full)
    expected = tuple({
        "cycle": item["cycle"], "family": item["family"],
        "variant_id": item["variant_id"], "parameters": item["parameters"],
    } for item in EXPECTED_CANDIDATE_SPECS)
    _need(actual == expected, "model candidate_specs inventory differs from the exact v5 registration")
    return True


@dataclass(frozen=True)
class LearnerProtocol:
    path: Path
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class LearnerInputAudit:
    records: int
    fold_counts: Mapping[int, int]
    fold_assignment_fingerprint: str
    baseline_sha256: str
    evidence_cache_sha256: str
    historical_v1_promotion_sha256: str
    historical_v2_aggregate_sha256: str
    historical_v3_aggregate_sha256: str
    historical_v4_aggregate_sha256: str
    historical_artifact_role: str
    model_inventory_available: bool


def validate_protocol_mapping(raw: Mapping[str, Any], *, path: str | Path = CONFIG_PATH) -> LearnerProtocol:
    """Exact-lock lineage, nested selection, gates, stop rule, and claims."""
    _need(isinstance(raw, Mapping) and dict(raw) == EXPECTED_PROTOCOL, "v5 protocol differs from exact registration")
    _need(raw["schema_version"] == SCHEMA_VERSION and raw["run_id"] == RUN_ID, "v5 identity differs")
    candidates = raw["candidates"]
    _need(len(candidates) == 20 and [item["cycle"] for item in candidates] == list(range(1, 21)), "v5 requires exact candidates 1..20")
    families = (
        ["anchored_multitask_residual"] * 4 + ["r0_anchored_distributional"] * 4
        + ["joint_tail_boundary_hurdle"] * 4 + ["axis_coupled_lowrank_moe"] * 4
        + ["band_risk_pareto_residual"] * 4
    )
    _need([item["family"] for item in candidates] == families, "v5 family order or cardinality differs")
    execution = raw["execution"]
    _need(execution["all_20_candidates_required"] and not execution["early_stop_allowed"], "all 20 candidates must run")
    _need(execution["same_initialization_across_candidates"] and execution["fresh_initialization_each_candidate_and_fold"], "initialization policy differs")
    _need(not execution["checkpoint_reuse"], "checkpoint reuse is forbidden")
    nested = raw["nested_protocol"]
    _need(nested["inner_fold_count_per_outer"] == 4 and nested["outer_gold_locked_until_prediction_complete"], "outer gold must remain locked")
    _need(nested["inner_training_rule"] == "S_is_the_other_three_folds_excluding_O_and_D", "inner S/D/O isolation differs")
    _need(nested["candidate_inner_fit"] == "each_candidate_fresh_fit_on_S_predict_D_once", "candidate S-to-D fit differs")
    _need(nested["candidate_oof_coverage"] == "each_outer_train_row_exactly_once_per_candidate", "candidate OOF coverage differs")
    _need(nested["selection_after_all_candidates_complete"], "selection must wait for all candidates")
    outer = nested["outer_refit"]
    _need(outer["selected_spec_frozen_before_refit"] and outer["selected_candidate_only"] and outer["initialization"] == "fresh", "selected candidate freeze/refit differs")
    _need(not any(nested[key] for key in ("historical_v1_predictions_used", "historical_v2_predictions_used", "historical_v3_predictions_used", "historical_v4_predictions_used", "historical_weights_used", "historical_pseudo_targets_used")), "historical predictions, weights, or pseudo-targets are forbidden")
    _need(raw["inner_promotion_gate"]["operator"] == "AND", "inner seven gates must be conjunctive")
    final = raw["final_evaluation"]
    _need(final["construction"] == "concatenate_five_outer_predictions_once" and not final["selection_after_concatenation"], "final concatenation cannot trigger selection")
    _need(final["macro_rmse_min_improvement"] == 0.01 and final["paired_bootstrap"]["replicates"] == 10000, "final macro/bootstrap gate differs")
    _need(final["paired_bootstrap"]["quantity"] == "candidate_minus_baseline_macro_rmse" and final["paired_bootstrap"]["required_upper_bound_lt"] == 0.0, "paired-bootstrap direction differs")
    claims = raw["scientific_claims"]
    _need(claims["same_train_nested_descriptive_evidence_only"], "adaptive same-train status must be explicit")
    _need(not claims["independent_confirmation_claim_allowed"] and not claims["generalization_claim_allowed"] and not claims["deployment_claim_allowed"], "v5 cannot make independent, generalization, or deployment claims")
    _need(raw["failure_action"] == {"when": "final_gate_fails", "retained_model": "exact_r0_oof_baseline", "v5_inventory_frozen": True}, "failure action differs")
    supersession = raw["v4_stop_supersession"]
    _need(supersession["authorization_date"] == "2026-08-02" and supersession["scope"] == "separately_named_exactly_preregistered_v5_only", "V4 stop supersession authorization differs")
    _need(supersession["v4_inventory_permanently_frozen"] and supersession["v4_learned_weights_permanently_frozen"], "V4 inventory and weights must remain frozen")
    _need(not supersession["v4_posthoc_tuning_allowed"] and not supersession["v4_artifact_modification_allowed"], "V4 posthoc changes remain forbidden")
    agent = raw["agent_evidence"]
    _need(not agent["model_feature_allowed"] and not agent["pseudo_target_allowed"] and not agent["reward_or_weight_allowed"], "agent opinions are preregistration evidence only")
    validate_model_inventory(require_available=False)
    return LearnerProtocol(Path(path), raw)


def load_protocol(path: str | Path = CONFIG_PATH) -> LearnerProtocol:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IterativeTailLearnerProtocolError(f"v5 protocol is unreadable: {config_path}") from exc
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


def validate_bound_inputs(protocol: LearnerProtocol, *, root: str | Path = ".") -> LearnerInputAudit:
    """Validate fixed artifacts, aggregates, and the exact 2,000/5x400 folds."""
    root_path = Path(root)
    lineage = protocol.raw["lineage"]
    checks = (
        ("canonical_train_path", "canonical_train_sha256", "canonical train"),
        ("baseline_oof_path", "baseline_oof_sha256", "exact R0 OOF baseline"),
        ("embedding_rows_path", "embedding_rows_sha256", "frozen embedding rows"),
        ("score_blind_feature_cache_path", "score_blind_feature_cache_sha256", "score-blind feature cache"),
        ("score_blind_feature_manifest_path", "score_blind_feature_manifest_sha256", "score-blind feature manifest"),
        ("v1_promotion_path", "v1_promotion_sha256", "historical v1 promotion"),
        ("v2_aggregate_path", "v2_aggregate_sha256", "historical v2 aggregate"),
        ("v3_aggregate_path", "v3_aggregate_sha256", "historical v3 aggregate"),
        ("v4_aggregate_path", "v4_aggregate_sha256", "historical v4 aggregate"),
    )
    resolved: dict[str, Path] = {}
    for path_key, sha_key, label in checks:
        artifact = _bound_path(root_path, lineage[path_key])
        _need(artifact.is_file() and not artifact.is_symlink(), f"{label} path is missing or mutable")
        _need(_sha256(artifact) == lineage[sha_key], f"{label} checksum differs")
        resolved[path_key] = artifact
    historical = (
        ("v1_promotion", "mal2026-iterative-tail-promotion-v1", None),
        ("v2_aggregate", "mal2026-iterative-tail-remediation-aggregate-v2", 2000),
        ("v3_aggregate", "mal2026-iterative-tail-cycle-aggregate-v3", 2000),
        ("v4_aggregate", "mal2026-iterative-tail-router-aggregate-v4", 2000),
    )
    for name, schema, record_count in historical:
        try:
            artifact = json.loads(resolved[f"{name}_path"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IterativeTailLearnerProtocolError(f"historical {name} artifact is unreadable") from exc
        _need(artifact.get("schema_version") == schema and artifact.get("status") == "completed", f"historical {name} identity differs")
        if record_count is not None:
            _need(artifact.get("record_count") == record_count, f"historical {name} population differs")
        _need(artifact.get("validation_loaded") is False and artifact.get("average_target_used") is False, f"historical {name} isolation differs")
    _need(lineage["historical_artifact_role"] == "adaptive_preregistration_evidence_only_forbidden_as_model_input", "historical artifact role differs")

    embedding_manifest_path = _bound_path(root_path, lineage["embedding_manifest_path"])
    _need(embedding_manifest_path.is_file() and not embedding_manifest_path.is_symlink(), "embedding manifest path is missing or mutable")
    try:
        feature_manifest = json.loads(resolved["score_blind_feature_manifest_path"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IterativeTailLearnerProtocolError("score-blind feature manifest is unreadable") from exc
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
    _need(counts == Counter({fold: 400 for fold in range(5)}), "v5 requires exactly 400 rows per fold")
    payload = "\n".join(f"{source_id}:{fold}" for source_id, fold in sorted(baseline.items()))
    fingerprint = sha256(payload.encode("utf-8")).hexdigest()
    _need(fingerprint == lineage["fold_assignment_fingerprint"], "fold assignment fingerprint differs")
    return LearnerInputAudit(2000, dict(sorted(counts.items())), fingerprint, lineage["baseline_oof_sha256"],
                            lineage["score_blind_feature_cache_sha256"], lineage["v1_promotion_sha256"],
                            lineage["v2_aggregate_sha256"], lineage["v3_aggregate_sha256"],
                            lineage["v4_aggregate_sha256"], lineage["historical_artifact_role"],
                            validate_model_inventory(require_available=False))


load_config = load_protocol
