"""Fail-closed bindings for the preregistered V9 selective-flip study."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .iterative_official_agent_stack_models import FEATURE_DIM
from .iterative_official_selective_flip_models import candidate_specs
from .iterative_tail_metrics import AXES


CONFIG_PATH = Path("configs/iterative_official_selective_flip.v9.json")
RUN_ID = "iterative-official-selective-flip-v9-20260802-001"


class OfficialSelectiveFlipProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class OfficialSelectiveFlipProtocol:
    path: Path
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class OfficialSelectiveFlipInputAudit:
    records: int
    folds: Mapping[int, int]
    fold_fingerprint: str
    official_candidates: int
    v8_aggregate_sha256: str


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialSelectiveFlipProtocolError(message)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound(root: Path, relative: str, expected: str) -> Path:
    path = root / relative
    _need(path.is_file() and not path.is_symlink() and _sha256(path) == expected, f"V9 bound input differs: {relative}")
    return path


def validate_protocol_mapping(raw: Mapping[str, Any], *, path: Path = CONFIG_PATH) -> OfficialSelectiveFlipProtocol:
    _need(raw.get("schema_version") == "mal2026-iterative-official-selective-flip-v9" and raw.get("run_id") == RUN_ID, "V9 identity differs")
    contract, execution = raw.get("data_contract"), raw.get("execution")
    _need(isinstance(contract, Mapping) and isinstance(execution, Mapping), "V9 contracts missing")
    _need(contract.get("records") == 2000 and contract.get("fold_count") == 5 and contract.get("records_per_fold") == 400, "V9 population differs")
    _need(contract.get("feature_dimensions") == FEATURE_DIM and contract.get("official_candidate_records") == 6000 and contract.get("official_candidates_per_essay") == 3, "V9 feature/candidate population differs")
    _need(contract.get("official_candidate_scores_are_model_predictions") is True and contract.get("human_or_reference_score_read_or_prompted_to_official_candidate_model") is False, "V9 candidate role differs")
    _need(contract.get("rationale_text_used_as_v9_feature") is False and contract.get("validation_loaded") is False and contract.get("validation_selection") is False and contract.get("average_target_used") is False and contract.get("external_api_calls_in_v9") == 0, "V9 isolation differs")
    _need(execution.get("authorized_gpus") == [0, 1, 2, 3] and execution.get("smoke_gpu") == 0 and execution.get("smoke_required_before_full_run") is True, "V9 GPU/smoke differs")
    _need(execution.get("fresh_residual_and_selective_heads_each_candidate_and_fold") is True and execution.get("checkpoint_reuse") is False and execution.get("all_3_candidates_required") is True and execution.get("early_stop_allowed") is False, "V9 fresh execution differs")
    common_residual, common_boundary = raw.get("common_residual"), raw.get("common_boundary_classifier")
    _need(isinstance(common_residual, Mapping) and common_residual == {"family": "official_terra_score_residual_ridge", "ridge_alpha": 10.0, "max_correction": 0.5, "fresh_per_candidate_and_fold": True}, "V9 common residual differs")
    _need(isinstance(common_boundary, Mapping) and common_boundary == {
        "feature_dimensions": FEATURE_DIM,
        "optimizer": "torch.optim.LBFGS_strong_wolfe",
        "max_iter": 80,
        "l2": 0.01,
        "fresh_zero_initialization": True,
        "application": "only_high_confidence_disagreement_within_fixed_3_5_window",
    }, "V9 common boundary differs")
    registered, specs = raw.get("candidates"), candidate_specs()
    _need(isinstance(registered, list) and len(registered) == len(specs) == 3, "V9 inventory differs")
    for entry, spec in zip(registered, specs, strict=True):
        _need(entry.get("cycle") == spec.cycle and entry.get("variant_id") == spec.variant_id and entry.get("head_kind") == spec.head_kind, "V9 candidate identity differs")
        _need(float(entry.get("confidence")) == spec.confidence and float(entry.get("window")) == spec.window and float(entry.get("epsilon")) == spec.epsilon and float(entry.get("l2")) == spec.l2 and entry.get("fresh_initialization") is True, "V9 candidate parameters differ")
    nested = raw.get("nested_protocol")
    _need(isinstance(nested, Mapping) and nested.get("outer_folds") == [0, 1, 2, 3, 4] and nested.get("outer_gold_locked_until_prediction_complete") is True and nested.get("selection_after_all_candidates_complete") is True and nested.get("posthoc_selection_on_concatenated_outer_predictions") is False, "V9 nesting differs")
    exact_inner = {
        "operator": "AND", "macro_rmse_min_improvement": 0.005,
        "equal_group_rmse_min_improvement": 0.01, "low_tail_must_improve": True,
        "high_tail_must_improve": True, "gold_3_4_balanced_accuracy_min_improvement": 0.01,
        "max_axis_rmse_worsening": 0.01, "max_macro_spearman_fall": 0.005,
        "score1_descriptive_only": True, "require_all_four_inner_folds": True,
        "require_finite_metrics": True,
    }
    _need(raw.get("inner_promotion_gate") == exact_inner, "V9 original seven gates differ")
    final = raw.get("final_evaluation")
    _need(isinstance(final, Mapping) and final.get("macro_rmse_min_improvement") == 0.01 and final.get("paired_bootstrap", {}).get("replicates") == 10000 and final.get("paired_bootstrap", {}).get("required_upper_bound_lt") == 0.0 and final.get("selection_after_concatenation") is False, "V9 final macro/bootstrap differs")
    _need(all(final.get(key) is True for key in ("low_tail_must_improve", "high_tail_must_improve", "score1_descriptive_only", "require_all_five_outer_folds", "require_finite_metrics")), "V9 final required flags differ")
    _need(final.get("gold_3_4_balanced_accuracy_min_improvement") == 0.01 and final.get("max_axis_rmse_worsening") == 0.01 and final.get("max_macro_spearman_fall") == 0.005, "V9 final safety gates differ")
    freeze = raw.get("authorization_and_freeze")
    _need(isinstance(freeze, Mapping) and freeze.get("v8_reopening") is False and freeze.get("v8_inventory_and_artifacts_frozen") is True and freeze.get("v8_gates_unchanged") is True and freeze.get("materially_distinct_application_rule") == "high_confidence_near_boundary_flip_only", "V8 freeze differs")
    claims = raw.get("scientific_claims")
    _need(isinstance(claims, Mapping) and claims.get("adaptive_after_v8_observed") is True and claims.get("no_v9_full_oof_prestudy_before_preregistration") is True and claims.get("same_train_nested_development_evidence_only") is True, "V9 adaptive/registration status differs")
    _need(claims.get("independent_confirmation_claim_allowed") is False and claims.get("generalization_claim_allowed") is False and claims.get("deployment_claim_allowed") is False, "V9 claim limits differ")
    return OfficialSelectiveFlipProtocol(path, raw)


def load_protocol(path: str | Path = CONFIG_PATH) -> OfficialSelectiveFlipProtocol:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialSelectiveFlipProtocolError("V9 protocol is unreadable") from exc
    _need(isinstance(raw, Mapping), "V9 protocol must be an object")
    return validate_protocol_mapping(raw, path=config_path)


def _baseline_folds(path: Path) -> dict[str, int]:
    required = {"source_id", "fold", "continuous_prediction", "half_up_integer_prediction", "reference_score"}
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            _need(isinstance(row, Mapping) and set(row) == required, f"V9 baseline schema differs at {line_number}")
            source_id, fold, scores = row.get("source_id"), row.get("fold"), row.get("continuous_prediction")
            _need(isinstance(source_id, str) and source_id and source_id not in result and type(fold) is int and 0 <= fold < 5, "V9 baseline identity/fold differs")
            _need(isinstance(scores, Mapping) and set(scores) == set(AXES) and all(type(scores[axis]) in {int, float} and math.isfinite(float(scores[axis])) for axis in AXES), "V9 baseline scores differ")
            result[source_id] = fold
    return result


def validate_bound_inputs(protocol: OfficialSelectiveFlipProtocol, *, root: str | Path = ".") -> OfficialSelectiveFlipInputAudit:
    root_path, lineage = Path(root), protocol.raw["lineage"]
    keys = (
        ("canonical_train_path", "canonical_train_sha256"), ("baseline_oof_path", "baseline_oof_sha256"),
        ("official_candidate_manifest_path", "official_candidate_manifest_sha256"),
        ("official_candidate_rows_path", "official_candidate_rows_sha256"),
        ("v8_aggregate_path", "v8_aggregate_sha256"), ("v8_completion_path", "v8_completion_sha256"),
    )
    resolved = {path_key: _bound(root_path, str(lineage[path_key]), str(lineage[sha_key])) for path_key, sha_key in keys}
    manifest = json.loads(resolved["official_candidate_manifest_path"].read_text(encoding="utf-8"))
    _need(manifest.get("status") == "validated" and manifest.get("model") == lineage.get("official_candidate_model") == "gpt-5.6-terra", "V9 official candidate manifest differs")
    _need(manifest.get("accepted") == 6000 and manifest.get("human_or_reference_score_read_or_prompted") is False and manifest.get("official_system_prompt_sha256") == lineage.get("official_system_prompt_sha256"), "V9 official candidate count/score-blind binding differs")
    v8_aggregate = json.loads(resolved["v8_aggregate_path"].read_text(encoding="utf-8"))
    v8_completion = json.loads(resolved["v8_completion_path"].read_text(encoding="utf-8"))
    _need(v8_aggregate.get("schema_version") == "mal2026-iterative-official-boundary-aggregate-v8" and v8_aggregate.get("status") == "completed" and v8_aggregate.get("final_gate_pass") is False, "V8 aggregate status differs")
    _need(v8_completion.get("status") == "completed_no_promotion_baseline_retained" and v8_completion.get("final_gate_pass") is False, "V8 completion status differs")
    _need(lineage.get("v8_result_role") == "smooth_boundary_nudges_improved_ba_inefficiently_and_failed_folds_2_3", "V8 evidence role differs")
    assignments = _baseline_folds(resolved["baseline_oof_path"])
    counts = Counter(assignments.values())
    _need(len(assignments) == 2000 and counts == Counter({fold: 400 for fold in range(5)}), "V9 baseline population/folds differ")
    payload = "\n".join(f"{source_id}:{fold}" for source_id, fold in sorted(assignments.items()))
    fingerprint = sha256(payload.encode("utf-8")).hexdigest()
    _need(fingerprint == lineage["fold_assignment_fingerprint"], "V9 fold fingerprint differs")
    return OfficialSelectiveFlipInputAudit(2000, dict(sorted(counts.items())), fingerprint, 6000, lineage["v8_aggregate_sha256"])


__all__ = [
    "CONFIG_PATH", "RUN_ID", "OfficialSelectiveFlipInputAudit", "OfficialSelectiveFlipProtocol",
    "OfficialSelectiveFlipProtocolError", "load_protocol", "validate_bound_inputs", "validate_protocol_mapping",
]
