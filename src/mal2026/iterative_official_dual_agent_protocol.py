"""Fail-closed bindings for the preregistered V10 Terra/Luna study."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .iterative_official_dual_agent_models import FEATURE_DIM, candidate_specs
from .iterative_tail_metrics import AXES


CONFIG_PATH = Path("configs/iterative_official_dual_agent.v10.json")
RUN_ID = "iterative-official-dual-agent-v10-20260802-001"
_HEX64 = re.compile(r"[0-9a-f]{64}")


class OfficialDualAgentProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class OfficialDualAgentProtocol:
    path: Path
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class OfficialDualAgentInputAudit:
    records: int
    folds: Mapping[int, int]
    fold_fingerprint: str
    terra_candidates: int
    luna_candidates: int
    luna_candidate_rows_sha256: str
    v9_aggregate_sha256: str


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialDualAgentProtocolError(message)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound(root: Path, relative: str, expected: str) -> Path:
    path = root / relative
    _need(
        isinstance(expected, str) and _HEX64.fullmatch(expected) is not None
        and path.is_file() and not path.is_symlink() and _sha256(path) == expected,
        f"V10 bound input differs: {relative}",
    )
    return path


def validate_protocol_mapping(raw: Mapping[str, Any], *, path: Path = CONFIG_PATH) -> OfficialDualAgentProtocol:
    _need(raw.get("schema_version") == "mal2026-iterative-official-dual-agent-v10" and raw.get("run_id") == RUN_ID, "V10 identity differs")
    lineage = raw.get("lineage")
    _need(isinstance(lineage, Mapping), "V10 lineage missing")
    binding = raw.get("post_generation_binding")
    _need(isinstance(binding, Mapping) and binding.get("state") == "validated_checksums_bound_before_smoke", "V10 Luna artifacts are not finally checksum-bound")
    _need(binding.get("only_fields_allowed_to_change_after_this_preregistration") == [
        "lineage.luna_candidate_manifest_sha256", "lineage.luna_candidate_rows_sha256", "post_generation_binding.state",
    ] and binding.get("scientific_inventory_change_allowed") is False, "V10 binding-only amendment contract differs")
    _need(all(isinstance(lineage.get(key), str) and _HEX64.fullmatch(str(lineage[key])) is not None for key in ("luna_candidate_manifest_sha256", "luna_candidate_rows_sha256")), "V10 Luna checksums are unbound")
    contract, execution = raw.get("data_contract"), raw.get("execution")
    _need(isinstance(contract, Mapping) and isinstance(execution, Mapping), "V10 contracts missing")
    _need(contract.get("records") == 2000 and contract.get("fold_count") == 5 and contract.get("records_per_fold") == 400, "V10 population differs")
    _need(contract.get("terra_candidate_records") == 6000 and contract.get("luna_candidate_records") == 6000 and contract.get("candidates_per_model_per_essay") == 3 and contract.get("total_model_candidates_per_essay") == 6, "V10 candidate population differs")
    _need(contract.get("feature_dimensions") == FEATURE_DIM and contract.get("candidate_scores_are_model_predictions") is True and contract.get("human_or_reference_score_read_or_prompted_to_candidate_models") is False, "V10 feature/source role differs")
    _need(contract.get("rationale_text_used_as_v10_feature") is False and contract.get("validation_loaded") is False and contract.get("validation_selection") is False and contract.get("average_target_used") is False and contract.get("external_api_calls_in_v10") == 0, "V10 isolation differs")
    _need(contract.get("upstream_luna_batch_requests") == 6000 and contract.get("upstream_luna_smoke_requests") == 1, "V10 upstream API accounting differs")
    feature = raw.get("feature_contract")
    _need(isinstance(feature, Mapping) and feature.get("terra_within_model_dimensions") == 39 and feature.get("luna_within_model_dimensions") == 39 and feature.get("cross_model_dimensions") == 18 and feature.get("target_blind_construction") is True, "V10 feature contract differs")
    _need(feature.get("cross_model_order") == [
        "terra_minus_luna_axis_mean", "absolute_terra_minus_luna_axis_mean",
        "pooled_six_candidate_axis_mean", "pooled_six_candidate_axis_std",
        "pooled_six_candidate_axis_min", "pooled_six_candidate_axis_max",
    ], "V10 cross-model feature order differs")
    _need(execution.get("authorized_gpus") == [0, 1, 2, 3] and execution.get("smoke_gpu") == 0 and execution.get("smoke_required_before_full_run") is True, "V10 GPU/smoke differs")
    _need(execution.get("fresh_residual_and_optional_heads_each_candidate_and_fold") is True and execution.get("checkpoint_reuse") is False and execution.get("all_3_candidates_required") is True and execution.get("early_stop_allowed") is False, "V10 fresh execution differs")
    _need(raw.get("common_residual") == {"family": "terra_luna_score_residual_ridge", "ridge_alpha": 10.0, "max_correction": 0.5, "fresh_per_candidate_and_fold": True}, "V10 common residual differs")
    _need(raw.get("common_boundary_classifier") == {
        "feature_dimensions": FEATURE_DIM, "optimizer": "torch.optim.LBFGS_strong_wolfe",
        "max_iter": 80, "l2": 0.01, "fresh_zero_initialization": True,
        "application": "only_high_confidence_disagreement_within_fixed_3_5_window",
    }, "V10 common boundary differs")
    registered, specs = raw.get("candidates"), candidate_specs()
    _need(isinstance(registered, list) and len(registered) == len(specs) == 3, "V10 inventory differs")
    for entry, spec in zip(registered, specs, strict=True):
        _need(entry.get("cycle") == spec.cycle and entry.get("variant_id") == spec.variant_id and entry.get("head_kind") == spec.head_kind, "V10 candidate identity differs")
        for key in ("ridge_alpha", "max_correction", "confidence", "window", "epsilon", "l2"):
            _need(float(entry.get(key)) == float(getattr(spec, key)), f"V10 candidate {key} differs")
        _need(entry.get("fresh_initialization") is True, "V10 candidate initialization differs")
    nested = raw.get("nested_protocol")
    _need(isinstance(nested, Mapping) and nested.get("outer_folds") == [0, 1, 2, 3, 4] and nested.get("outer_gold_locked_until_prediction_complete") is True and nested.get("selection_after_all_candidates_complete") is True and nested.get("posthoc_selection_on_concatenated_outer_predictions") is False, "V10 nesting differs")
    _need(nested.get("inner_fold_count_per_outer") == 4 and nested.get("inner_selection", {}).get("gate") == "fixed_original_seven_gate" and nested.get("inner_selection", {}).get("no_eligible_fallback") == "exact_r0_oof_baseline", "V10 inner selection differs")
    exact_inner = {
        "operator": "AND", "macro_rmse_min_improvement": 0.005,
        "equal_group_rmse_min_improvement": 0.01, "low_tail_must_improve": True,
        "high_tail_must_improve": True, "gold_3_4_balanced_accuracy_min_improvement": 0.01,
        "max_axis_rmse_worsening": 0.01, "max_macro_spearman_fall": 0.005,
        "score1_descriptive_only": True, "require_all_four_inner_folds": True,
        "require_finite_metrics": True,
    }
    _need(raw.get("inner_promotion_gate") == exact_inner, "V10 original seven gates differ")
    final = raw.get("final_evaluation")
    _need(isinstance(final, Mapping) and final.get("macro_rmse_min_improvement") == 0.01 and final.get("paired_bootstrap", {}).get("replicates") == 10000 and final.get("paired_bootstrap", {}).get("required_upper_bound_lt") == 0.0 and final.get("selection_after_concatenation") is False, "V10 final macro/bootstrap differs")
    _need(all(final.get(key) is True for key in ("low_tail_must_improve", "high_tail_must_improve", "score1_descriptive_only", "require_all_five_outer_folds", "require_finite_metrics")), "V10 final flags differ")
    _need(final.get("gold_3_4_balanced_accuracy_min_improvement") == 0.01 and final.get("max_axis_rmse_worsening") == 0.01 and final.get("max_macro_spearman_fall") == 0.005, "V10 final safety gates differ")
    freeze = raw.get("authorization_and_freeze")
    _need(isinstance(freeze, Mapping) and freeze.get("v9_reopening") is False and freeze.get("v9_inventory_and_artifacts_frozen") is True and freeze.get("v9_gates_unchanged") is True and freeze.get("materially_new_evidence") == "independent_score_blind_gpt_5_6_luna_outputs", "V10 V9 freeze differs")
    claims = raw.get("scientific_claims")
    _need(isinstance(claims, Mapping) and claims.get("adaptive_after_v9_observed") is True and claims.get("candidate_inventory_frozen_before_luna_batch_results_downloaded") is True and claims.get("no_v10_full_oof_prestudy_before_preregistration") is True and claims.get("same_train_nested_development_evidence_only") is True, "V10 registration status differs")
    _need(claims.get("independent_confirmation_claim_allowed") is False and claims.get("generalization_claim_allowed") is False and claims.get("deployment_claim_allowed") is False, "V10 claim limits differ")
    return OfficialDualAgentProtocol(path, raw)


def load_protocol(path: str | Path = CONFIG_PATH) -> OfficialDualAgentProtocol:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialDualAgentProtocolError("V10 protocol is unreadable") from exc
    _need(isinstance(raw, Mapping), "V10 protocol must be an object")
    return validate_protocol_mapping(raw, path=config_path)


def _baseline_folds(path: Path) -> dict[str, int]:
    required = {"source_id", "fold", "continuous_prediction", "half_up_integer_prediction", "reference_score"}
    result: dict[str, int] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            _need(isinstance(row, Mapping) and set(row) == required, f"V10 baseline schema differs at {line_number}")
            source_id, fold, scores = row.get("source_id"), row.get("fold"), row.get("continuous_prediction")
            _need(isinstance(source_id, str) and source_id and source_id not in result and type(fold) is int and 0 <= fold < 5, "V10 baseline identity/fold differs")
            _need(isinstance(scores, Mapping) and set(scores) == set(AXES) and all(type(scores[axis]) in {int, float} and math.isfinite(float(scores[axis])) for axis in AXES), "V10 baseline scores differ")
            result[source_id] = fold
    return result


def validate_bound_inputs(protocol: OfficialDualAgentProtocol, *, root: str | Path = ".") -> OfficialDualAgentInputAudit:
    root_path, lineage = Path(root), protocol.raw["lineage"]
    keys = (
        ("canonical_train_path", "canonical_train_sha256"),
        ("baseline_oof_path", "baseline_oof_sha256"),
        ("terra_candidate_manifest_path", "terra_candidate_manifest_sha256"),
        ("terra_candidate_rows_path", "terra_candidate_rows_sha256"),
        ("luna_generation_plan_path", "luna_generation_plan_sha256"),
        ("luna_candidate_manifest_path", "luna_candidate_manifest_sha256"),
        ("luna_candidate_rows_path", "luna_candidate_rows_sha256"),
        ("v9_aggregate_path", "v9_aggregate_sha256"),
        ("v9_completion_path", "v9_completion_sha256"),
    )
    resolved = {path_key: _bound(root_path, str(lineage[path_key]), str(lineage[sha_key])) for path_key, sha_key in keys}
    for prefix, model in (("terra", "gpt-5.6-terra"), ("luna", "gpt-5.6-luna")):
        manifest = json.loads(resolved[f"{prefix}_candidate_manifest_path"].read_text(encoding="utf-8"))
        _need(manifest.get("status") == "validated" and manifest.get("model") == model and manifest.get("accepted") == 6000 and manifest.get("requests") == 6000, f"V10 {prefix} manifest differs")
        _need(manifest.get("human_or_reference_score_read_or_prompted") is False and manifest.get("source_sha256") == lineage["canonical_train_sha256"] and manifest.get("official_system_prompt_sha256") == lineage["official_system_prompt_sha256"], f"V10 {prefix} score-blind prompt binding differs")
        _need(manifest.get("candidates_sha256") == lineage[f"{prefix}_candidate_rows_sha256"], f"V10 {prefix} row manifest binding differs")
    luna_manifest = json.loads(resolved["luna_candidate_manifest_path"].read_text(encoding="utf-8"))
    _need(luna_manifest.get("request_sha256") == lineage["luna_request_sha256"], "V10 Luna request binding differs")
    plan = json.loads(resolved["luna_generation_plan_path"].read_text(encoding="utf-8"))
    _need(plan.get("run_id") == luna_manifest.get("run_id") and plan.get("model") == "gpt-5.6-luna" and plan.get("generation", {}).get("requests") == 6000, "V10 Luna generation plan differs")
    v9_aggregate = json.loads(resolved["v9_aggregate_path"].read_text(encoding="utf-8"))
    v9_completion = json.loads(resolved["v9_completion_path"].read_text(encoding="utf-8"))
    _need(v9_aggregate.get("schema_version") == "mal2026-iterative-official-selective-flip-aggregate-v9" and v9_aggregate.get("status") == "completed" and v9_aggregate.get("final_gate_pass") is False, "V9 aggregate status differs")
    _need(v9_completion.get("status") == "completed_no_promotion_baseline_retained" and v9_completion.get("final_gate_pass") is False, "V9 completion status differs")
    _need(lineage.get("v9_result_role") == "selective_flips_improved_nested_development_but_macro_and_3_4_ba_final_gates_failed", "V9 result role differs")
    assignments = _baseline_folds(resolved["baseline_oof_path"])
    counts = Counter(assignments.values())
    _need(len(assignments) == 2000 and counts == Counter({fold: 400 for fold in range(5)}), "V10 baseline population/folds differ")
    payload = "\n".join(f"{source_id}:{fold}" for source_id, fold in sorted(assignments.items()))
    fingerprint = sha256(payload.encode()).hexdigest()
    _need(fingerprint == lineage["fold_assignment_fingerprint"], "V10 fold fingerprint differs")
    return OfficialDualAgentInputAudit(
        2000, dict(sorted(counts.items())), fingerprint, 6000, 6000,
        str(lineage["luna_candidate_rows_sha256"]), str(lineage["v9_aggregate_sha256"]),
    )


__all__ = [
    "CONFIG_PATH", "RUN_ID", "OfficialDualAgentInputAudit", "OfficialDualAgentProtocol",
    "OfficialDualAgentProtocolError", "load_protocol", "validate_bound_inputs", "validate_protocol_mapping",
]
