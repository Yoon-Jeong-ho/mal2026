"""Exact train-only nested-selection contract for iterative tail router v4.

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

SCHEMA_VERSION = "mal2026-iterative-tail-router-v4"
RUN_ID = "iterative-tail-router-v4-20260801-001"
CONFIG_PATH = Path("configs/iterative_tail_router.v4.json")
MODEL_MODULE = "mal2026.iterative_tail_router_models"
EXPECTED_PROTOCOL = {'schema_version': 'mal2026-iterative-tail-router-v4', 'run_id': 'iterative-tail-router-v4-20260801-001', 'lineage': {'canonical_train_path': 'eval/train.jsonl', 'canonical_train_sha256': 'b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737', 'baseline_oof_path': 'data/processed/restricted/r0_exact_oof_v1/r0-exact-oof-20260731-002/merged/oof_predictions.jsonl', 'baseline_oof_sha256': '823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04', 'embedding_manifest_path': 'data/processed/restricted/r0_ordinal_residual_embeddings_v1/r0-public-frozen-embedding-20260731-001/train/merged/manifest.json', 'embedding_rows_path': 'data/processed/restricted/r0_ordinal_residual_embeddings_v1/r0-public-frozen-embedding-20260731-001/train/merged/rows.jsonl', 'embedding_rows_sha256': '949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16', 'fold_assignment_fingerprint': '8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db', 'score_blind_feature_cache_path': 'data/processed/restricted/iterative_tail_refinement_v1/iterative-tail-refinement-v1-20260801-001/score_blind_features.npz', 'score_blind_feature_cache_sha256': 'c2770d2add3b08a46614ad4c56fddc2d6b06ba5784c39e80c95c9f419e46d7db', 'score_blind_feature_manifest_path': 'data/processed/restricted/iterative_tail_refinement_v1/iterative-tail-refinement-v1-20260801-001/score_blind_features.manifest.json', 'score_blind_feature_manifest_sha256': 'ae3e44270fdeb4d6d217fc82ebd7097ee4cd7031343221b855a6ac7207cf38b0', 'v2_aggregate_path': 'outputs/iterative-tail-remediation-v2/iterative-tail-remediation-v2-20260801-001/aggregate.json', 'v2_aggregate_sha256': 'bc64443aafc48c45f9a882cf861c11eba9b22d6d417b62d90c43a1fbf81af81f', 'v3_aggregate_path': 'outputs/iterative-tail-cycle-v3/iterative-tail-cycle-v3-20260801-001/aggregate.json', 'v3_aggregate_sha256': 'bc44123673b512ae20212a94f7a3e98b4d227823cd1d74d2fddf8f4a07e3667f', 'historical_aggregate_role': 'preregistration_evidence_only_forbidden_as_model_feature'}, 'data_contract': {'records': 2000, 'fold_count': 5, 'records_per_fold': 400, 'split_role': 'train', 'baseline_origin': 'exact_oof', 'evidence_view': 'evidence_hash_score_blind', 'validation_loaded': False, 'validation_selection': False, 'average_target_used': False, 'optional_api_enabled': False, 'external_api_calls_allowed': False}, 'execution': {'authorized_gpus': [0, 1, 2, 3], 'smoke_gpu': 0, 'initialization_seed': 2026080104, 'same_initialization_across_routes': True, 'fresh_initialization_each_component_and_fold': True, 'checkpoint_reuse': False, 'all_20_routes_required': True, 'early_stop_allowed': False}, 'nested_protocol': {'outer_folds': [0, 1, 2, 3, 4], 'outer_holdout_use': 'prediction_once_only_after_route_freeze', 'outer_metrics_locked_until_prediction_complete': True, 'inner_validation_rule': 'for_outer_O_each_D_in_other_four_folds', 'inner_training_rule': 'S_is_the_three_folds_excluding_O_and_D', 'inner_fold_count_per_outer': 4, 'inner_components': {'r16_teacher': {'fit_scope': 'S_only', 'method': 'fresh_2_of_3_crossfit_within_S', 'forbidden_folds': ['D', 'O']}, 'r17_challenger': {'source': 'fresh_inner_r16_teacher', 'ridge_alpha': 100.0, 'fit_scope': 'S_only', 'predict_scope': 'D_only'}, 'direct_evidence_ridge': {'target': 'raw_axis_gold', 'ridge_alpha': 100.0, 'fit_scope': 'S_only', 'predict_scope': 'D_only'}, 'hurdle_component': {'source': 'fresh_v3_hurdle-v1', 'fit_scope': 'S_only', 'predict_scope': 'D_only'}, 'soft_component': {'source': 'fresh_v3_soft-v4', 'fit_scope': 'S_only', 'predict_scope': 'D_only'}}, 'inner_bank': {'components': ['r17_challenger', 'direct_evidence_ridge', 'hurdle-v1', 'soft-v4'], 'construction': 'concatenate_four_inner_OOF_component_banks', 'coverage': 'every_outer_train_row_exactly_once', 'route_fit_scope': 'outer_train_only'}, 'inner_selection': {'reference': 'exact_r0_oof_baseline_on_outer_train', 'gate': 'fixed_seven_gate', 'rule': 'eligible_minimum_macro_rmse', 'tie_break': 'lowest_route_number', 'no_eligible_fallback': 'exact_r0_oof_baseline', 'require_all_four_inner_folds': True, 'require_finite_metrics': True}, 'outer_refit': {'after_route_freeze': True, 'r16_teacher': 'fresh_3_of_4_crossfit_over_all_outer_train', 'r17_challenger': 'fresh_alpha_100_refit_on_all_outer_train', 'direct_evidence_ridge': 'fresh_alpha_100_refit_on_all_outer_train', 'hurdle_component': 'fresh_hurdle-v1_refit_on_all_outer_train', 'soft_component': 'fresh_soft-v4_refit_on_all_outer_train', 'apply': 'frozen_selected_route', 'predict_scope': 'O_once'}, 'historical_v1_predictions_used_as_features': False, 'historical_v2_predictions_used_as_features': False, 'historical_v3_predictions_used_as_features': False}, 'routers': [{'cycle': 1, 'family': 'low_protected_sigmoid_stack', 'variant_id': 'low_protected_sigmoid_stack-v1', 'parameters': {'temperature': 0.1, 'max_nonidentity': 0.15, 'step': 0.05}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 2, 'family': 'low_protected_sigmoid_stack', 'variant_id': 'low_protected_sigmoid_stack-v2', 'parameters': {'temperature': 0.2, 'max_nonidentity': 0.2, 'step': 0.05}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 3, 'family': 'low_protected_sigmoid_stack', 'variant_id': 'low_protected_sigmoid_stack-v3', 'parameters': {'temperature': 0.35, 'max_nonidentity': 0.25, 'step': 0.05}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 4, 'family': 'low_protected_sigmoid_stack', 'variant_id': 'low_protected_sigmoid_stack-v4', 'parameters': {'temperature': 0.5, 'max_nonidentity': 0.3, 'step': 0.05}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 5, 'family': 'four_zone_hard_stack', 'variant_id': 'four_zone_hard_stack-v1', 'parameters': {'identity_floors': [0.9, 0.7, 0.7, 0.6], 'step': 0.1, 'passes': 2}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 6, 'family': 'four_zone_hard_stack', 'variant_id': 'four_zone_hard_stack-v2', 'parameters': {'identity_floors': [0.85, 0.6, 0.6, 0.5], 'step': 0.1, 'passes': 2}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 7, 'family': 'four_zone_hard_stack', 'variant_id': 'four_zone_hard_stack-v3', 'parameters': {'identity_floors': [0.8, 0.5, 0.5, 0.4], 'step': 0.1, 'passes': 2}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 8, 'family': 'four_zone_hard_stack', 'variant_id': 'four_zone_hard_stack-v4', 'parameters': {'identity_floors': [0.75, 0.4, 0.4, 0.3], 'step': 0.1, 'passes': 2}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 9, 'family': 'boundary_hurdle_overlay', 'variant_id': 'boundary_hurdle_overlay-v1', 'parameters': {'window': 0.05, 'max_nonidentity': 0.15, 'step': 0.05}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 10, 'family': 'boundary_hurdle_overlay', 'variant_id': 'boundary_hurdle_overlay-v2', 'parameters': {'window': 0.1, 'max_nonidentity': 0.2, 'step': 0.05}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 11, 'family': 'boundary_hurdle_overlay', 'variant_id': 'boundary_hurdle_overlay-v3', 'parameters': {'window': 0.15, 'max_nonidentity': 0.25, 'step': 0.05}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 12, 'family': 'boundary_hurdle_overlay', 'variant_id': 'boundary_hurdle_overlay-v4', 'parameters': {'window': 0.2, 'max_nonidentity': 0.3, 'step': 0.05}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 13, 'family': 'sigmoid_four_expert_route', 'variant_id': 'sigmoid_four_expert_route-v1', 'parameters': {'temperature': 0.1, 'identity_floor': 0.8, 'pareto_margin': 0.0, 'step': 0.05, 'passes': 2}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 14, 'family': 'sigmoid_four_expert_route', 'variant_id': 'sigmoid_four_expert_route-v2', 'parameters': {'temperature': 0.2, 'identity_floor': 0.7, 'pareto_margin': 0.0, 'step': 0.05, 'passes': 2}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 15, 'family': 'sigmoid_four_expert_route', 'variant_id': 'sigmoid_four_expert_route-v3', 'parameters': {'temperature': 0.35, 'identity_floor': 0.6, 'pareto_margin': 0.0001, 'step': 0.05, 'passes': 2}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 16, 'family': 'sigmoid_four_expert_route', 'variant_id': 'sigmoid_four_expert_route-v4', 'parameters': {'temperature': 0.5, 'identity_floor': 0.5, 'pareto_margin': 0.00025, 'step': 0.05, 'passes': 2}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 17, 'family': 'formal_gate_lattice_stack', 'variant_id': 'formal_gate_lattice_stack-v1', 'parameters': {'temperature': 0.15, 'max_nonidentity': 0.2, 'correction_cap': 0.1, 'step': 0.1, 'passes': 2}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 18, 'family': 'formal_gate_lattice_stack', 'variant_id': 'formal_gate_lattice_stack-v2', 'parameters': {'temperature': 0.25, 'max_nonidentity': 0.3, 'correction_cap': 0.15, 'step': 0.1, 'passes': 2}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 19, 'family': 'formal_gate_lattice_stack', 'variant_id': 'formal_gate_lattice_stack-v3', 'parameters': {'temperature': 0.35, 'max_nonidentity': 0.25, 'correction_cap': 0.15, 'step': 0.05, 'passes': 2}, 'initialization_seed': 2026080104, 'fresh_initialization': True}, {'cycle': 20, 'family': 'formal_gate_lattice_stack', 'variant_id': 'formal_gate_lattice_stack-v4', 'parameters': {'temperature': 0.5, 'max_nonidentity': 0.35, 'correction_cap': 0.2, 'step': 0.05, 'passes': 2}, 'initialization_seed': 2026080104, 'fresh_initialization': True}], 'inner_promotion_gate': {'operator': 'AND', 'macro_rmse_min_improvement': 0.005, 'equal_group_rmse_min_improvement': 0.01, 'low_tail_must_improve': True, 'high_tail_must_improve': True, 'gold_3_4_balanced_accuracy_min_improvement': 0.01, 'max_axis_rmse_worsening': 0.01, 'max_macro_spearman_fall': 0.005, 'score1_descriptive_only': True, 'require_all_four_inner_folds': True, 'require_finite_metrics': True}, 'final_evaluation': {'construction': 'concatenate_five_outer_predictions_once', 'selection_after_concatenation': False, 'reference': 'exact_r0_oof_baseline', 'operator': 'AND', 'macro_rmse_min_improvement': 0.01, 'paired_bootstrap': {'replicates': 10000, 'quantity': 'candidate_minus_baseline_macro_rmse', 'confidence_interval': 0.95, 'required_upper_bound_lt': 0.0}, 'low_tail_must_improve': True, 'high_tail_must_improve': True, 'gold_3_4_balanced_accuracy_min_improvement': 0.01, 'max_axis_rmse_worsening': 0.01, 'max_macro_spearman_fall': 0.005, 'score1_descriptive_only': True, 'require_all_five_outer_folds': True, 'require_finite_metrics': True}, 'scientific_claims': {'adaptive_after_v2_and_v3_observed': True, 'final_same_train_adaptive_nested_evidence': True, 'independent_confirmation_claim_allowed': False, 'generalization_claim_allowed': False, 'deployment_claim_allowed': False}, 'stop_rule': {'when': 'final_gate_fails', 'action': 'freeze_all_same_train_model_search', 'retained_model': 'exact_r0_oof_baseline'}, 'agent_evidence': {'role': 'preregistration_evidence_only', 'allowed_fields': ['agent_role', 'timestamp', 'hypothesis', 'predicted_direction', 'falsification_condition', 'accepted_or_rejected_rationale', 'config_sha256'], 'model_feature_allowed': False, 'pseudo_target_allowed': False, 'reward_or_weight_allowed': False}}
EXPECTED_ROUTER_SPECS = tuple(EXPECTED_PROTOCOL["routers"])


class IterativeTailRouterProtocolError(ValueError):
    """Raised before registered v4 nesting, isolation, or claims can drift."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise IterativeTailRouterProtocolError(message)


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


def router_specs() -> tuple[Mapping[str, Any], ...]:
    """Return an isolated copy of the registered 20-router inventory."""
    return tuple(deepcopy(spec) for spec in EXPECTED_ROUTER_SPECS)


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
        _need(isinstance(item, Mapping), "model router_specs entries must be mappings or dataclasses")
        normalized.append(_json_value(dict(item)))
    return tuple(normalized)


def validate_model_inventory(*, require_available: bool = False) -> bool:
    """Require exact equality with ``iterative_tail_router_models.router_specs()``.

    Absence is tolerated for protocol-only compilation. Runner preflight must
    call this with ``require_available=True``.
    """
    try:
        module = importlib.import_module(MODEL_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != MODEL_MODULE:
            raise
        _need(not require_available, "v4 router model module is required for execution")
        return False
    inventory_function = getattr(module, "router_specs", None)
    _need(callable(inventory_function), "v4 router model module must expose router_specs()")
    actual = _normalize_model_specs(tuple(inventory_function()))
    expected = tuple({
        "cycle": item["cycle"], "family": item["family"],
        "variant_id": item["variant_id"], "parameters": item["parameters"],
    } for item in EXPECTED_ROUTER_SPECS)
    _need(actual == expected, "model router_specs inventory differs from the exact v4 registration")
    return True


@dataclass(frozen=True)
class RouterProtocol:
    path: Path
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class RouterInputAudit:
    records: int
    fold_counts: Mapping[int, int]
    fold_assignment_fingerprint: str
    baseline_sha256: str
    evidence_cache_sha256: str
    historical_v2_aggregate_sha256: str
    historical_v3_aggregate_sha256: str
    historical_aggregate_role: str
    model_inventory_available: bool


def validate_protocol_mapping(raw: Mapping[str, Any], *, path: str | Path = CONFIG_PATH) -> RouterProtocol:
    """Exact-lock lineage, nested selection, gates, stop rule, and claims."""
    _need(isinstance(raw, Mapping) and dict(raw) == EXPECTED_PROTOCOL, "v4 protocol differs from exact registration")
    _need(raw["schema_version"] == SCHEMA_VERSION and raw["run_id"] == RUN_ID, "v4 identity differs")
    routers = raw["routers"]
    _need(len(routers) == 20 and [item["cycle"] for item in routers] == list(range(1, 21)), "v4 requires exact routes 1..20")
    families = (
        ["low_protected_sigmoid_stack"] * 4 + ["four_zone_hard_stack"] * 4
        + ["boundary_hurdle_overlay"] * 4 + ["sigmoid_four_expert_route"] * 4
        + ["formal_gate_lattice_stack"] * 4
    )
    _need([item["family"] for item in routers] == families, "v4 family order or cardinality differs")
    execution = raw["execution"]
    _need(execution["all_20_routes_required"] and not execution["early_stop_allowed"], "all 20 routes must run")
    _need(execution["same_initialization_across_routes"] and execution["fresh_initialization_each_component_and_fold"], "initialization policy differs")
    _need(not execution["checkpoint_reuse"], "checkpoint reuse is forbidden")
    nested = raw["nested_protocol"]
    _need(nested["inner_fold_count_per_outer"] == 4 and nested["outer_metrics_locked_until_prediction_complete"], "outer metrics must remain locked through four inner folds")
    _need(nested["inner_training_rule"] == "S_is_the_three_folds_excluding_O_and_D", "inner S/O/D isolation differs")
    teacher = nested["inner_components"]["r16_teacher"]
    _need(teacher["method"] == "fresh_2_of_3_crossfit_within_S" and teacher["forbidden_folds"] == ["D", "O"], "inner R16 teacher isolation differs")
    _need(nested["inner_components"]["r17_challenger"]["ridge_alpha"] == 100.0, "inner R17 alpha must be 100")
    _need(nested["inner_components"]["direct_evidence_ridge"]["ridge_alpha"] == 100.0, "inner direct alpha must be 100")
    _need(nested["inner_bank"]["coverage"] == "every_outer_train_row_exactly_once", "inner component banks must be complete OOF")
    _need(not any(nested[key] for key in ("historical_v1_predictions_used_as_features", "historical_v2_predictions_used_as_features", "historical_v3_predictions_used_as_features")), "historical row predictions are forbidden features")
    _need(raw["inner_promotion_gate"]["operator"] == "AND", "inner seven gates must be conjunctive")
    final = raw["final_evaluation"]
    _need(final["construction"] == "concatenate_five_outer_predictions_once" and not final["selection_after_concatenation"], "final concatenation cannot trigger selection")
    _need(final["macro_rmse_min_improvement"] == 0.01 and final["paired_bootstrap"]["replicates"] == 10000, "final macro/bootstrap gate differs")
    _need(final["paired_bootstrap"]["quantity"] == "candidate_minus_baseline_macro_rmse" and final["paired_bootstrap"]["required_upper_bound_lt"] == 0.0, "paired-bootstrap direction differs")
    claims = raw["scientific_claims"]
    _need(claims["final_same_train_adaptive_nested_evidence"], "adaptive same-train status must be explicit")
    _need(not claims["independent_confirmation_claim_allowed"] and not claims["generalization_claim_allowed"] and not claims["deployment_claim_allowed"], "v4 cannot make independent, generalization, or deployment claims")
    _need(raw["stop_rule"] == {"when": "final_gate_fails", "action": "freeze_all_same_train_model_search", "retained_model": "exact_r0_oof_baseline"}, "failure stop rule differs")
    agent = raw["agent_evidence"]
    _need(not agent["model_feature_allowed"] and not agent["pseudo_target_allowed"] and not agent["reward_or_weight_allowed"], "agent opinions are preregistration evidence only")
    validate_model_inventory(require_available=False)
    return RouterProtocol(Path(path), raw)


def load_protocol(path: str | Path = CONFIG_PATH) -> RouterProtocol:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IterativeTailRouterProtocolError(f"v4 protocol is unreadable: {config_path}") from exc
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


def validate_bound_inputs(protocol: RouterProtocol, *, root: str | Path = ".") -> RouterInputAudit:
    """Validate fixed artifacts, aggregates, and the exact 2,000/5x400 folds."""
    root_path = Path(root)
    lineage = protocol.raw["lineage"]
    checks = (
        ("canonical_train_path", "canonical_train_sha256", "canonical train"),
        ("baseline_oof_path", "baseline_oof_sha256", "exact R0 OOF baseline"),
        ("embedding_rows_path", "embedding_rows_sha256", "frozen embedding rows"),
        ("score_blind_feature_cache_path", "score_blind_feature_cache_sha256", "score-blind feature cache"),
        ("score_blind_feature_manifest_path", "score_blind_feature_manifest_sha256", "score-blind feature manifest"),
        ("v2_aggregate_path", "v2_aggregate_sha256", "historical v2 aggregate"),
        ("v3_aggregate_path", "v3_aggregate_sha256", "historical v3 aggregate"),
    )
    resolved: dict[str, Path] = {}
    for path_key, sha_key, label in checks:
        artifact = _bound_path(root_path, lineage[path_key])
        _need(artifact.is_file() and not artifact.is_symlink(), f"{label} path is missing or mutable")
        _need(_sha256(artifact) == lineage[sha_key], f"{label} checksum differs")
        resolved[path_key] = artifact
    for version, schema in (("v2", "mal2026-iterative-tail-remediation-aggregate-v2"), ("v3", "mal2026-iterative-tail-cycle-aggregate-v3")):
        try:
            aggregate = json.loads(resolved[f"{version}_aggregate_path"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IterativeTailRouterProtocolError(f"historical {version} aggregate is unreadable") from exc
        _need(aggregate.get("schema_version") == schema, f"historical {version} aggregate schema differs")
        _need(aggregate.get("status") == "completed" and aggregate.get("record_count") == 2000, f"historical {version} completion differs")
        _need(aggregate.get("validation_loaded") is False and aggregate.get("average_target_used") is False, f"historical {version} isolation differs")
    _need(lineage["historical_aggregate_role"] == "preregistration_evidence_only_forbidden_as_model_feature", "historical aggregate role differs")

    embedding_manifest_path = _bound_path(root_path, lineage["embedding_manifest_path"])
    _need(embedding_manifest_path.is_file() and not embedding_manifest_path.is_symlink(), "embedding manifest path is missing or mutable")
    try:
        feature_manifest = json.loads(resolved["score_blind_feature_manifest_path"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IterativeTailRouterProtocolError("score-blind feature manifest is unreadable") from exc
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
    _need(counts == Counter({fold: 400 for fold in range(5)}), "v4 requires exactly 400 rows per fold")
    payload = "\n".join(f"{source_id}:{fold}" for source_id, fold in sorted(baseline.items()))
    fingerprint = sha256(payload.encode("utf-8")).hexdigest()
    _need(fingerprint == lineage["fold_assignment_fingerprint"], "fold assignment fingerprint differs")
    return RouterInputAudit(2000, dict(sorted(counts.items())), fingerprint, lineage["baseline_oof_sha256"],
                            lineage["score_blind_feature_cache_sha256"], lineage["v2_aggregate_sha256"],
                            lineage["v3_aggregate_sha256"], lineage["historical_aggregate_role"],
                            validate_model_inventory(require_available=False))


load_config = load_protocol
