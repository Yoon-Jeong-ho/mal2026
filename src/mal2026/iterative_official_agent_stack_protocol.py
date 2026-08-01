"""Fail-closed bindings for the adaptive V7 official-agent stack."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .iterative_official_agent_stack_models import FEATURE_DIM, candidate_specs
from .iterative_tail_metrics import AXES


CONFIG_PATH = Path("configs/iterative_official_agent_stack.v7.json")
RUN_ID = "iterative-official-agent-stack-v7-20260802-001"


class OfficialAgentStackProtocolError(ValueError):
    """Raised when a preregistered V7 binding or rule differs."""


@dataclass(frozen=True)
class OfficialAgentStackProtocol:
    path: Path
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class OfficialAgentStackInputAudit:
    records: int
    folds: Mapping[int, int]
    fold_fingerprint: str
    official_candidates: int
    official_model: str
    prestudy_sha256: str


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialAgentStackProtocolError(message)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound(root: Path, relative: str) -> Path:
    path = root / relative
    _need(path.is_file() and not path.is_symlink(), f"bound input missing or mutable: {relative}")
    return path


def validate_protocol_mapping(raw: Mapping[str, Any], *, path: Path = CONFIG_PATH) -> OfficialAgentStackProtocol:
    _need(raw.get("schema_version") == "mal2026-iterative-official-agent-stack-v7", "V7 schema differs")
    _need(raw.get("run_id") == RUN_ID, "V7 run ID differs")
    contract, execution = raw.get("data_contract"), raw.get("execution")
    _need(isinstance(contract, Mapping) and isinstance(execution, Mapping), "V7 data/execution contract missing")
    _need(contract.get("records") == 2000 and contract.get("fold_count") == 5 and contract.get("records_per_fold") == 400, "V7 population differs")
    _need(contract.get("feature_dimensions") == FEATURE_DIM, "V7 feature dimensions differ")
    _need(contract.get("official_candidate_records") == 6000 and contract.get("official_candidates_per_essay") == 3, "V7 official candidate population differs")
    _need(contract.get("official_candidate_scores_are_model_predictions") is True, "V7 feature role differs")
    _need(contract.get("human_or_reference_score_read_or_prompted_to_official_candidate_model") is False, "V7 official candidates are not score-blind")
    _need(contract.get("rationale_text_used_as_v7_feature") is False, "V7 feature contract differs")
    _need(contract.get("validation_loaded") is False and contract.get("validation_selection") is False and contract.get("average_target_used") is False, "V7 split/target isolation differs")
    _need(contract.get("external_api_calls_in_v7") == 0, "V7 must reuse the bound completed API artifact")
    _need(execution.get("authorized_gpus") == [0, 1, 2, 3] and execution.get("smoke_gpu") == 0, "V7 GPU scope differs")
    _need(execution.get("smoke_required_before_full_run") is True and execution.get("fresh_closed_form_solve_each_candidate_and_fold") is True, "V7 fresh/smoke rule differs")
    _need(execution.get("checkpoint_reuse") is False and execution.get("all_3_candidates_required") is True and execution.get("early_stop_allowed") is False, "V7 execution rule differs")
    registered = raw.get("candidates")
    specs = candidate_specs()
    _need(isinstance(registered, list) and len(registered) == len(specs) == 3, "V7 candidate inventory differs")
    for entry, spec in zip(registered, specs, strict=True):
        _need(entry.get("cycle") == spec.cycle and entry.get("variant_id") == spec.variant_id, "V7 candidate identity differs")
        _need(entry.get("family") == "official_terra_score_residual_ridge", "V7 candidate family differs")
        _need(float(entry.get("ridge_alpha")) == spec.ridge_alpha and float(entry.get("max_correction")) == spec.max_correction, "V7 candidate parameters differ")
        _need(entry.get("fresh_initialization") is True, "V7 candidate must start fresh")
    nested = raw.get("nested_protocol")
    _need(isinstance(nested, Mapping) and nested.get("outer_folds") == [0, 1, 2, 3, 4], "V7 outer folds differ")
    _need(nested.get("outer_gold_locked_until_prediction_complete") is True and nested.get("selection_after_all_candidates_complete") is True, "V7 outer isolation differs")
    _need(nested.get("posthoc_selection_on_concatenated_outer_predictions") is False, "V7 posthoc selection is forbidden")
    inner, final = raw.get("inner_promotion_gate"), raw.get("final_evaluation")
    _need(isinstance(inner, Mapping) and isinstance(final, Mapping), "V7 gates missing")
    exact_inner = {
        "operator": "AND", "macro_rmse_min_improvement": 0.005,
        "equal_group_rmse_min_improvement": 0.01, "low_tail_must_improve": True,
        "high_tail_must_improve": True, "gold_3_4_balanced_accuracy_min_improvement": 0.01,
        "max_axis_rmse_worsening": 0.01, "max_macro_spearman_fall": 0.005,
        "score1_descriptive_only": True, "require_all_four_inner_folds": True,
        "require_finite_metrics": True,
    }
    _need(dict(inner) == exact_inner, "V7 original seven-gate conjunction differs")
    _need(final.get("macro_rmse_min_improvement") == 0.01 and final.get("paired_bootstrap", {}).get("replicates") == 10000, "V7 final macro/bootstrap differs")
    _need(final.get("paired_bootstrap", {}).get("required_upper_bound_lt") == 0.0 and final.get("selection_after_concatenation") is False, "V7 final selection/CI direction differs")
    _need(all(final.get(key) is True for key in ("low_tail_must_improve", "high_tail_must_improve", "score1_descriptive_only", "require_all_five_outer_folds", "require_finite_metrics")), "V7 final required flags differ")
    _need(final.get("gold_3_4_balanced_accuracy_min_improvement") == 0.01 and final.get("max_axis_rmse_worsening") == 0.01 and final.get("max_macro_spearman_fall") == 0.005, "V7 final safety gates differ")
    freeze = raw.get("authorization_and_freeze")
    _need(isinstance(freeze, Mapping) and freeze.get("v4_v5_v6_reopening") is False and freeze.get("v4_v5_v6_artifacts_permanently_frozen") is True, "V4--V6 freeze differs")
    _need(freeze.get("v6_same_train_same_feature_source_stop_rule_respected") is True and freeze.get("new_feature_source") == "three_score_blind_gpt_5_6_terra_official_participant_outputs", "V7 new-feature authorization differs")
    claims = raw.get("scientific_claims")
    _need(isinstance(claims, Mapping) and claims.get("adaptive_after_v1_through_v6_and_full_oof_prestudy_observed") is True, "V7 adaptive status missing")
    _need(claims.get("same_train_nested_development_evidence_only") is True and claims.get("independent_confirmation_claim_allowed") is False and claims.get("generalization_claim_allowed") is False and claims.get("deployment_claim_allowed") is False, "V7 claim limits differ")
    return OfficialAgentStackProtocol(path, raw)


def load_protocol(path: str | Path = CONFIG_PATH) -> OfficialAgentStackProtocol:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialAgentStackProtocolError("V7 protocol is unreadable") from exc
    _need(isinstance(raw, Mapping), "V7 protocol must be an object")
    return validate_protocol_mapping(raw, path=config_path)


def _baseline_folds(path: Path) -> dict[str, int]:
    required = {"source_id", "fold", "continuous_prediction", "half_up_integer_prediction", "reference_score"}
    result: dict[str, int] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            _need(isinstance(row, Mapping) and set(row) == required, f"baseline schema differs at line {line_number}")
            source_id, fold = row.get("source_id"), row.get("fold")
            scores = row.get("continuous_prediction")
            _need(isinstance(source_id, str) and source_id and source_id not in result, "baseline IDs differ")
            _need(type(fold) is int and 0 <= fold < 5, "baseline fold differs")
            _need(isinstance(scores, Mapping) and set(scores) == set(AXES), "baseline axes differ")
            _need(all(type(scores[axis]) in {int, float} and math.isfinite(float(scores[axis])) for axis in AXES), "baseline scores differ")
            result[source_id] = fold
    return result


def validate_bound_inputs(protocol: OfficialAgentStackProtocol, *, root: str | Path = ".") -> OfficialAgentStackInputAudit:
    root_path, lineage = Path(root), protocol.raw["lineage"]
    pairs = (
        ("canonical_train_path", "canonical_train_sha256"),
        ("baseline_oof_path", "baseline_oof_sha256"),
        ("official_candidate_manifest_path", "official_candidate_manifest_sha256"),
        ("official_candidate_rows_path", "official_candidate_rows_sha256"),
        ("v6_aggregate_path", "v6_aggregate_sha256"),
        ("v6_completion_path", "v6_completion_sha256"),
        ("prestudy_path", "prestudy_sha256"),
    )
    resolved: dict[str, Path] = {}
    for path_key, sha_key in pairs:
        artifact = _bound(root_path, str(lineage[path_key]))
        _need(_sha256(artifact) == lineage[sha_key], f"V7 binding differs: {path_key}")
        resolved[path_key] = artifact
    manifest = json.loads(resolved["official_candidate_manifest_path"].read_text(encoding="utf-8"))
    _need(manifest.get("schema_version") == "mal2026-official-openai-candidate-v1" and manifest.get("status") == "validated", "official candidate manifest identity differs")
    _need(manifest.get("model") == lineage["official_candidate_model"] == "gpt-5.6-terra", "official candidate model differs")
    _need(manifest.get("accepted") == manifest.get("requests") == 6000 and manifest.get("train_rows") == 2000 and manifest.get("candidates_per_essay") == 3, "official candidate counts differ")
    _need(manifest.get("candidates_sha256") == lineage["official_candidate_rows_sha256"], "official candidate row binding differs")
    _need(manifest.get("human_or_reference_score_read_or_prompted") is False and manifest.get("official_system_prompt_sha256") == lineage["official_system_prompt_sha256"], "official candidates are not score-blind or prompt differs")
    v6 = json.loads(resolved["v6_completion_path"].read_text(encoding="utf-8"))
    _need(v6.get("status") == "completed_no_promotion_terminal_same_train_feature_freeze" and v6.get("terminal_freeze_same_train_and_feature_sources") is True, "V6 terminal state differs")
    prestudy = json.loads(resolved["prestudy_path"].read_text(encoding="utf-8"))
    _need(prestudy.get("schema_version") == "mal2026-iterative-official-agent-stack-prestudy-v7" and prestudy.get("status") == "completed", "V7 prestudy identity differs")
    _need(prestudy.get("adaptive_before_v7_preregistration") is True and prestudy.get("validation_loaded") is False and prestudy.get("average_target_used") is False, "V7 prestudy isolation/status differs")
    _need(lineage.get("prestudy_role") == "adaptive_candidate_design_evidence_only_not_outer_confirmation", "V7 prestudy role differs")
    assignments = _baseline_folds(resolved["baseline_oof_path"])
    counts = Counter(assignments.values())
    _need(len(assignments) == 2000 and counts == Counter({fold: 400 for fold in range(5)}), "V7 baseline population/folds differ")
    payload = "\n".join(f"{source_id}:{fold}" for source_id, fold in sorted(assignments.items()))
    fingerprint = sha256(payload.encode("utf-8")).hexdigest()
    _need(fingerprint == lineage["fold_assignment_fingerprint"], "V7 fold fingerprint differs")
    return OfficialAgentStackInputAudit(2000, dict(sorted(counts.items())), fingerprint, 6000, manifest["model"], lineage["prestudy_sha256"])


__all__ = [
    "CONFIG_PATH", "RUN_ID", "OfficialAgentStackInputAudit", "OfficialAgentStackProtocol",
    "OfficialAgentStackProtocolError", "load_protocol", "validate_bound_inputs", "validate_protocol_mapping",
]
