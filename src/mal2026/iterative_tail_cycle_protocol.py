"""Preregistered 20-cycle train-only discovery contract for v3.

V3 was designed after observing the v2 outer result. Its OOF aggregates are
adaptive descriptive discovery only; a winner can only be frozen for a future
untouched evaluation.
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

SCHEMA_VERSION = "mal2026-iterative-tail-cycle-v3"
RUN_ID = "iterative-tail-cycle-v3-20260801-001"
CONFIG_PATH = Path("configs/iterative_tail_cycle.v3.json")
MODEL_MODULE = "mal2026.iterative_tail_cycle_models"
EXPECTED_PROTOCOL = {'schema_version': 'mal2026-iterative-tail-cycle-v3', 'run_id': 'iterative-tail-cycle-v3-20260801-001', 'lineage': {'canonical_train_path': 'eval/train.jsonl', 'canonical_train_sha256': 'b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737', 'baseline_oof_path': 'data/processed/restricted/r0_exact_oof_v1/r0-exact-oof-20260731-002/merged/oof_predictions.jsonl', 'baseline_oof_sha256': '823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04', 'embedding_manifest_path': 'data/processed/restricted/r0_ordinal_residual_embeddings_v1/r0-public-frozen-embedding-20260731-001/train/merged/manifest.json', 'embedding_rows_path': 'data/processed/restricted/r0_ordinal_residual_embeddings_v1/r0-public-frozen-embedding-20260731-001/train/merged/rows.jsonl', 'embedding_rows_sha256': '949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16', 'fold_assignment_fingerprint': '8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db', 'score_blind_feature_cache_path': 'data/processed/restricted/iterative_tail_refinement_v1/iterative-tail-refinement-v1-20260801-001/score_blind_features.npz', 'score_blind_feature_cache_sha256': 'c2770d2add3b08a46614ad4c56fddc2d6b06ba5784c39e80c95c9f419e46d7db', 'score_blind_feature_manifest_path': 'data/processed/restricted/iterative_tail_refinement_v1/iterative-tail-refinement-v1-20260801-001/score_blind_features.manifest.json', 'score_blind_feature_manifest_sha256': 'ae3e44270fdeb4d6d217fc82ebd7097ee4cd7031343221b855a6ac7207cf38b0', 'v2_aggregate_path': 'outputs/iterative-tail-remediation-v2/iterative-tail-remediation-v2-20260801-001/aggregate.json', 'v2_aggregate_sha256': 'bc64443aafc48c45f9a882cf861c11eba9b22d6d417b62d90c43a1fbf81af81f', 'v2_aggregate_role': 'historical_preregistration_evidence_only_forbidden_as_model_feature'}, 'data_contract': {'records': 2000, 'fold_count': 5, 'records_per_fold': 400, 'split_role': 'train', 'baseline_origin': 'exact_oof', 'evidence_view': 'evidence_hash_score_blind', 'validation_loaded': False, 'validation_selection': False, 'average_target_used': False, 'optional_api_enabled': False, 'external_api_calls_allowed': False}, 'execution': {'authorized_gpus': [0, 1, 2, 3], 'smoke_gpu': 0, 'initialization_seed': 2026080103, 'same_initialization_across_cycles': True, 'fresh_initialization_each_cycle_and_fold': True, 'checkpoint_reuse': False, 'all_20_cycles_required': True, 'early_stop_allowed': False}, 'fold_protocol': {'heldout_folds': [0, 1, 2, 3, 4], 'for_each_heldout': 'use_other_four_folds_only', 'r16_teacher': 'fresh_3_of_4_crossfit_over_other_four', 'r17_challenger': {'source': 'fresh_r16_teacher', 'ridge_alpha': 10.0, 'fit_scope': 'other_four_only'}, 'direct_evidence_ridge_challenger': {'target': 'raw_axis_gold', 'fit_scope': 'other_four_only', 'ridge_alpha': 100.0, 'adaptive_basis': 'post_v2_outer_evidence_direct_alpha_100_selected_in_3_of_5_outer_folds', 'v2_aggregate_used_as_model_feature': False}, 'cycle_fit_predict': 'fresh_fit_on_other_four_predict_heldout_once', 'historical_v1_predictions_used_as_features': False, 'historical_v2_predictions_used_as_features': False, 'complete_all_fold_predictions_before_aggregate': True, 'selection_before_aggregate': False}, 'cycles': [{'cycle': 1, 'family': 'soft_routed_residual', 'variant_id': 'soft_routed_residual-v1', 'parameters': {'alpha': 10.0, 'cap': 0.1, 'temperature': 0.25, 'anchor': 1.0}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 2, 'family': 'soft_routed_residual', 'variant_id': 'soft_routed_residual-v2', 'parameters': {'alpha': 30.0, 'cap': 0.15, 'temperature': 0.35, 'anchor': 2.0}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 3, 'family': 'soft_routed_residual', 'variant_id': 'soft_routed_residual-v3', 'parameters': {'alpha': 100.0, 'cap': 0.2, 'temperature': 0.5, 'anchor': 4.0}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 4, 'family': 'soft_routed_residual', 'variant_id': 'soft_routed_residual-v4', 'parameters': {'alpha': 300.0, 'cap': 0.25, 'temperature': 0.75, 'anchor': 8.0}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 5, 'family': 'pareto_routed_stack', 'variant_id': 'pareto_routed_stack-v1', 'parameters': {'r17_low': 0.0, 'r17_high': 0.25, 'direct_low': 0.1, 'direct_high': 0.1, 'temperature': 0.25}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 6, 'family': 'pareto_routed_stack', 'variant_id': 'pareto_routed_stack-v2', 'parameters': {'r17_low': 0.0, 'r17_high': 0.4, 'direct_low': 0.15, 'direct_high': 0.1, 'temperature': 0.35}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 7, 'family': 'pareto_routed_stack', 'variant_id': 'pareto_routed_stack-v3', 'parameters': {'r17_low': 0.05, 'r17_high': 0.3, 'direct_low': 0.15, 'direct_high': 0.15, 'temperature': 0.5}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 8, 'family': 'pareto_routed_stack', 'variant_id': 'pareto_routed_stack-v4', 'parameters': {'r17_low': 0.0, 'r17_high': 0.5, 'direct_low': 0.25, 'direct_high': 0.0, 'temperature': 0.75}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 9, 'family': 'group_dro_ridge', 'variant_id': 'group_dro_ridge-v1', 'parameters': {'alpha': 30.0, 'cap': 0.1, 'eta': 0.25, 'iterations': 2, 'group_weight_cap': 3.0}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 10, 'family': 'group_dro_ridge', 'variant_id': 'group_dro_ridge-v2', 'parameters': {'alpha': 100.0, 'cap': 0.15, 'eta': 0.2, 'iterations': 3, 'group_weight_cap': 5.0}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 11, 'family': 'group_dro_ridge', 'variant_id': 'group_dro_ridge-v3', 'parameters': {'alpha': 300.0, 'cap': 0.2, 'eta': 0.15, 'iterations': 4, 'group_weight_cap': 7.0}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 12, 'family': 'group_dro_ridge', 'variant_id': 'group_dro_ridge-v4', 'parameters': {'alpha': 1000.0, 'cap': 0.25, 'eta': 0.1, 'iterations': 5, 'group_weight_cap': 10.0}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 13, 'family': 'selective_hurdle', 'variant_id': 'selective_hurdle-v1', 'parameters': {'evidence_dims': 16, 'confidence': 0.6, 'cap': 0.1, 'logistic_l2': 0.1, 'steps': 40, 'learning_rate': 0.1}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 14, 'family': 'selective_hurdle', 'variant_id': 'selective_hurdle-v2', 'parameters': {'evidence_dims': 32, 'confidence': 0.7, 'cap': 0.15, 'logistic_l2': 0.1, 'steps': 50, 'learning_rate': 0.08}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 15, 'family': 'selective_hurdle', 'variant_id': 'selective_hurdle-v3', 'parameters': {'evidence_dims': 64, 'confidence': 0.8, 'cap': 0.2, 'logistic_l2': 0.3, 'steps': 60, 'learning_rate': 0.06}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 16, 'family': 'selective_hurdle', 'variant_id': 'selective_hurdle-v4', 'parameters': {'evidence_dims': 96, 'confidence': 0.9, 'cap': 0.25, 'logistic_l2': 1.0, 'steps': 70, 'learning_rate': 0.05}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 17, 'family': 'final_ordinal_stack', 'variant_id': 'final_ordinal_stack-v1', 'parameters': {'ordinal_weight': 0.1, 'ordinal_mix': 0.1, 'cap': 0.1, 'epochs': 40, 'learning_rate': 0.03}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 18, 'family': 'final_ordinal_stack', 'variant_id': 'final_ordinal_stack-v2', 'parameters': {'ordinal_weight': 0.25, 'ordinal_mix': 0.2, 'cap': 0.15, 'epochs': 50, 'learning_rate': 0.02}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 19, 'family': 'final_ordinal_stack', 'variant_id': 'final_ordinal_stack-v3', 'parameters': {'ordinal_weight': 0.5, 'ordinal_mix': 0.3, 'cap': 0.2, 'epochs': 60, 'learning_rate': 0.015}, 'initialization_seed': 2026080103, 'fresh_initialization': True}, {'cycle': 20, 'family': 'final_ordinal_stack', 'variant_id': 'final_ordinal_stack-v4', 'parameters': {'ordinal_weight': 0.75, 'ordinal_mix': 0.4, 'cap': 0.25, 'epochs': 70, 'learning_rate': 0.01}, 'initialization_seed': 2026080103, 'fresh_initialization': True}], 'promotion_gate': {'operator': 'AND', 'macro_rmse_min_improvement': 0.005, 'equal_group_rmse_min_improvement': 0.01, 'low_tail_must_improve': True, 'high_tail_must_improve': True, 'gold_3_4_balanced_accuracy_min_improvement': 0.01, 'max_axis_rmse_worsening': 0.01, 'max_macro_spearman_fall': 0.005, 'score1_descriptive_only': True, 'require_all_five_folds': True, 'require_finite_metrics': True}, 'selection': {'when': 'after_all_20_complete_five_fold_oof_aggregates', 'eligibility_reference': 'fixed_exact_r0_oof_baseline', 'rule': 'eligible_minimum_macro_rmse', 'tie_break': 'lowest_cycle_number', 'no_eligible_fallback': 'exact_r0_oof_baseline', 'strict_winner_role': 'frozen_candidate_for_future_untouched_evaluation_only'}, 'scientific_claims': {'adaptive_after_v2_outer_observed': True, 'train_only_descriptive_discovery': True, 'confirmatory_claim_allowed': False, 'generalization_claim_allowed': False, 'deployment_claim_allowed': False, 'future_untouched_evaluation_required': True}, 'agent_evidence': {'role': 'preregistration_evidence_only', 'allowed_fields': ['agent_role', 'timestamp', 'hypothesis', 'predicted_direction', 'falsification_condition', 'accepted_or_rejected_rationale', 'config_sha256'], 'model_feature_allowed': False, 'pseudo_target_allowed': False, 'reward_or_weight_allowed': False}}
EXPECTED_CYCLE_SPECS = tuple(EXPECTED_PROTOCOL["cycles"])


class IterativeTailCycleProtocolError(ValueError):
    """Raised before cycle drift, leakage, or claim inflation can enter v3."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise IterativeTailCycleProtocolError(message)


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


def cycle_specs() -> tuple[Mapping[str, Any], ...]:
    """Return an isolated copy of the exact 20-cycle model inventory."""
    return tuple(deepcopy(spec) for spec in EXPECTED_CYCLE_SPECS)


def _normalize_model_specs(raw: Sequence[Any]) -> tuple[Mapping[str, Any], ...]:
    normalized = []
    for item in raw:
        if is_dataclass(item):
            item = {field.name: getattr(item, field.name) for field in fields(item)}
        _need(isinstance(item, Mapping), "model cycle_specs entries must be mappings or dataclasses")
        value = dict(item)
        if isinstance(value.get("parameters"), Mapping):
            value["parameters"] = dict(value["parameters"])
        normalized.append(value)
    return tuple(normalized)


def validate_model_inventory(*, require_available: bool = False) -> bool:
    """Bind a model module's ``cycle_specs()`` to the exact config inventory.

    The model module is being implemented independently. Absence is permitted
    during protocol-only compilation, but once importable any mismatch is a
    hard failure. Set ``require_available`` at runner preflight.
    """
    try:
        module = importlib.import_module(MODEL_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != MODEL_MODULE:
            raise
        _need(not require_available, "v3 model module is required for execution")
        return False
    inventory_function = getattr(module, "cycle_specs", None)
    _need(callable(inventory_function), "v3 model module must expose cycle_specs()")
    actual = _normalize_model_specs(tuple(inventory_function()))
    expected = tuple({
        "cycle": item["cycle"], "family": item["family"],
        "variant_id": item["variant_id"], "parameters": item["parameters"],
    } for item in EXPECTED_CYCLE_SPECS)
    _need(actual == expected, "model cycle_specs inventory differs from the exact v3 registration")
    return True


@dataclass(frozen=True)
class CycleProtocol:
    path: Path
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class CycleInputAudit:
    records: int
    fold_counts: Mapping[int, int]
    fold_assignment_fingerprint: str
    baseline_sha256: str
    evidence_cache_sha256: str
    historical_v2_aggregate_sha256: str
    historical_v2_aggregate_role: str
    model_inventory_available: bool


def validate_protocol_mapping(raw: Mapping[str, Any], *, path: str | Path = CONFIG_PATH) -> CycleProtocol:
    """Exact-lock lineage, all 20 cycles, isolation, gates, and claim scope."""
    _need(isinstance(raw, Mapping) and dict(raw) == EXPECTED_PROTOCOL, "v3 protocol differs from exact registration")
    _need(raw["schema_version"] == SCHEMA_VERSION and raw["run_id"] == RUN_ID, "v3 identity differs")
    cycles = raw["cycles"]
    _need(len(cycles) == 20 and [item["cycle"] for item in cycles] == list(range(1, 21)), "v3 requires exact cycles 1..20")
    family_order = (
        ["soft_routed_residual"] * 4 + ["pareto_routed_stack"] * 4
        + ["group_dro_ridge"] * 4 + ["selective_hurdle"] * 4
        + ["final_ordinal_stack"] * 4
    )
    _need([item["family"] for item in cycles] == family_order, "v3 family order or cardinality differs")
    execution = raw["execution"]
    _need(execution["all_20_cycles_required"] and not execution["early_stop_allowed"], "all 20 cycles must run without early stopping")
    _need(execution["same_initialization_across_cycles"] and execution["fresh_initialization_each_cycle_and_fold"], "cycle initialization policy differs")
    _need(not execution["checkpoint_reuse"], "checkpoint reuse is forbidden")
    fold = raw["fold_protocol"]
    _need(not fold["historical_v1_predictions_used_as_features"] and not fold["historical_v2_predictions_used_as_features"], "historical predictions are forbidden features")
    direct = fold["direct_evidence_ridge_challenger"]
    _need(direct["ridge_alpha"] == 100.0 and "alpha_grid" not in direct, "direct evidence ridge must use fixed adaptive alpha 100")
    _need(not direct["v2_aggregate_used_as_model_feature"], "historical v2 aggregate cannot be a model feature")
    _need(fold["complete_all_fold_predictions_before_aggregate"] and not fold["selection_before_aggregate"], "selection must wait for all five-fold predictions")
    _need(raw["promotion_gate"]["operator"] == "AND", "all seven gates must be conjunctive")
    claims = raw["scientific_claims"]
    _need(claims["adaptive_after_v2_outer_observed"] and claims["train_only_descriptive_discovery"], "adaptive descriptive status must be explicit")
    _need(not claims["confirmatory_claim_allowed"] and not claims["generalization_claim_allowed"] and not claims["deployment_claim_allowed"], "v3 cannot make confirmatory claims")
    agent = raw["agent_evidence"]
    _need(not agent["model_feature_allowed"] and not agent["pseudo_target_allowed"] and not agent["reward_or_weight_allowed"], "agent opinions are preregistration evidence only")
    validate_model_inventory(require_available=False)
    return CycleProtocol(Path(path), raw)


def load_protocol(path: str | Path = CONFIG_PATH) -> CycleProtocol:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IterativeTailCycleProtocolError(f"v3 protocol is unreadable: {config_path}") from exc
    return validate_protocol_mapping(raw, path=config_path)


def _baseline_assignments(path: Path) -> dict[str, int]:
    required = {"source_id", "fold", "continuous_prediction", "half_up_integer_prediction", "reference_score"}
    result: dict[str, int] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            raw = json.loads(line)
            _need(isinstance(raw, Mapping) and set(raw) == required, f"baseline schema differs at line {line_number}")
            source_id, fold = raw["source_id"], raw["fold"]
            _need(isinstance(source_id, str) and source_id and source_id not in result, "baseline source IDs differ")
            _need(type(fold) is int and 0 <= fold < 5, "baseline fold differs")
            scores = raw["continuous_prediction"]
            _need(isinstance(scores, Mapping) and set(scores) == set(AXES), "baseline axes differ or average was included")
            _need(all(type(scores[axis]) in {int, float} and math.isfinite(float(scores[axis])) for axis in AXES), "baseline scores must be finite")
            result[source_id] = fold
    return result


def validate_bound_inputs(protocol: CycleProtocol, *, root: str | Path = ".") -> CycleInputAudit:
    """Validate canonical v3 input checksums and exact 2,000/5x400 folds."""
    root_path = Path(root)
    lineage = protocol.raw["lineage"]
    checks = (
        ("canonical_train_path", "canonical_train_sha256", "canonical train"),
        ("baseline_oof_path", "baseline_oof_sha256", "exact R0 OOF baseline"),
        ("embedding_rows_path", "embedding_rows_sha256", "frozen embedding rows"),
        ("score_blind_feature_cache_path", "score_blind_feature_cache_sha256", "score-blind feature cache"),
        ("score_blind_feature_manifest_path", "score_blind_feature_manifest_sha256", "score-blind feature manifest"),
        ("v2_aggregate_path", "v2_aggregate_sha256", "historical v2 aggregate"),
    )
    resolved: dict[str, Path] = {}
    for path_key, sha_key, label in checks:
        artifact = _bound_path(root_path, lineage[path_key])
        _need(artifact.is_file() and not artifact.is_symlink(), f"{label} path is missing or mutable")
        _need(_sha256(artifact) == lineage[sha_key], f"{label} checksum differs")
        resolved[path_key] = artifact
    try:
        v2_aggregate = json.loads(resolved["v2_aggregate_path"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IterativeTailCycleProtocolError("historical v2 aggregate is unreadable") from exc
    _need(v2_aggregate.get("schema_version") == "mal2026-iterative-tail-remediation-aggregate-v2", "historical v2 aggregate schema differs")
    _need(v2_aggregate.get("status") == "completed" and v2_aggregate.get("record_count") == 2000, "historical v2 aggregate completion differs")
    _need(v2_aggregate.get("validation_loaded") is False and v2_aggregate.get("average_target_used") is False, "historical v2 aggregate isolation differs")
    _need(lineage["v2_aggregate_role"] == "historical_preregistration_evidence_only_forbidden_as_model_feature", "historical v2 aggregate role differs")

    embedding_manifest_path = _bound_path(root_path, lineage["embedding_manifest_path"])
    _need(embedding_manifest_path.is_file() and not embedding_manifest_path.is_symlink(), "embedding manifest path is missing or mutable")
    try:
        feature_manifest = json.loads(resolved["score_blind_feature_manifest_path"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IterativeTailCycleProtocolError("score-blind feature manifest is unreadable") from exc
    _need(feature_manifest.get("records") == 2000 and feature_manifest.get("axes") == list(AXES), "evidence feature population or axes differ")
    _need(feature_manifest.get("hash_dim") == 96 and feature_manifest.get("cache_sha256") == lineage["score_blind_feature_cache_sha256"], "evidence_hash binding differs")
    _need(feature_manifest.get("score_conditioning") is False and feature_manifest.get("validation_loaded") is False, "evidence is not score-blind train-only")

    manifest, embedding_rows = load_embedding_artifact(embedding_manifest_path, resolved["embedding_rows_path"])
    _need(manifest.split_role == "train" and manifest.base_prediction_origin == "oof", "embeddings must be train OOF")
    _need(not manifest.evaluation_only and not manifest.contains_average_target, "validation or average target is forbidden")
    baseline = _baseline_assignments(resolved["baseline_oof_path"])
    embedding = {row.source_id: row.oof_fold for row in embedding_rows}
    _need(len(baseline) == len(embedding) == 2000 and baseline == embedding, "baseline/embedding population or folds differ")
    counts = Counter(baseline.values())
    _need(counts == Counter({fold: 400 for fold in range(5)}), "v3 requires exactly 400 rows per fold")
    payload = "\n".join(f"{source_id}:{fold}" for source_id, fold in sorted(baseline.items()))
    fingerprint = sha256(payload.encode("utf-8")).hexdigest()
    _need(fingerprint == lineage["fold_assignment_fingerprint"], "fold assignment fingerprint differs")
    return CycleInputAudit(
        records=2000,
        fold_counts=dict(sorted(counts.items())),
        fold_assignment_fingerprint=fingerprint,
        baseline_sha256=lineage["baseline_oof_sha256"],
        evidence_cache_sha256=lineage["score_blind_feature_cache_sha256"],
        historical_v2_aggregate_sha256=lineage["v2_aggregate_sha256"],
        historical_v2_aggregate_role=lineage["v2_aggregate_role"],
        model_inventory_available=validate_model_inventory(require_available=False),
    )


load_config = load_protocol
