"""Exact leakage-safe contract for the v2 iterative-tail remediation.

The historical v1 R17 OOF artifact is validated only as lineage evidence. It
is forbidden as a v2 feature because other v1 folds may include outer gold.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

from mal2026.r0_ordinal_residual import AXES, load_embedding_artifact
from mal2026.iterative_tail_remediation_models import KNOTS_5, PREDECLARED_GRIDS

SCHEMA_VERSION = "mal2026-iterative-tail-remediation-v2"
RUN_ID = "iterative-tail-remediation-v2-20260801-001"
CONFIG_PATH = Path("configs/iterative_tail_remediation.v2.json")
EXPECTED_PROTOCOL = {'schema_version': 'mal2026-iterative-tail-remediation-v2', 'run_id': 'iterative-tail-remediation-v2-20260801-001', 'lineage': {'v1_run_id': 'iterative-tail-refinement-v1-20260801-001', 'v1_execution_git_commit': '40a28c758020b356e1d86e9790e55ea08a2ea69c', 'v1_config_path': 'configs/iterative_tail_refinement.v1.json', 'v1_config_sha256': '25508fc5fb510251cf94dc1b6a7cd798b3ff102efceaef4cbc0ac5a47943c7a5', 'canonical_train_path': 'eval/train.jsonl', 'canonical_train_sha256': 'b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737', 'baseline_oof_path': 'data/processed/restricted/r0_exact_oof_v1/r0-exact-oof-20260731-002/merged/oof_predictions.jsonl', 'baseline_oof_sha256': '823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04', 'embedding_manifest_path': 'data/processed/restricted/r0_ordinal_residual_embeddings_v1/r0-public-frozen-embedding-20260731-001/train/merged/manifest.json', 'embedding_rows_path': 'data/processed/restricted/r0_ordinal_residual_embeddings_v1/r0-public-frozen-embedding-20260731-001/train/merged/rows.jsonl', 'embedding_rows_sha256': '949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16', 'fold_assignment_fingerprint': '8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db', 'historical_r17_selected_prediction_path': 'data/processed/restricted/iterative_tail_refinement_v1/iterative-tail-refinement-v1-20260801-001/round-17/selected_oof_predictions.jsonl', 'historical_r17_selected_prediction_sha256': '5130e77abdee8866219a86d53896ddb6d1c680ae6cf4f3384fdc8a428ef7141a', 'historical_r17_role': 'reproduction_comparison_only_forbidden_as_v2_feature_or_candidate_prediction', 'score_blind_feature_cache_path': 'data/processed/restricted/iterative_tail_refinement_v1/iterative-tail-refinement-v1-20260801-001/score_blind_features.npz', 'score_blind_feature_cache_sha256': 'c2770d2add3b08a46614ad4c56fddc2d6b06ba5784c39e80c95c9f419e46d7db', 'score_blind_feature_manifest_path': 'data/processed/restricted/iterative_tail_refinement_v1/iterative-tail-refinement-v1-20260801-001/score_blind_features.manifest.json', 'score_blind_feature_manifest_sha256': 'ae3e44270fdeb4d6d217fc82ebd7097ee4cd7031343221b855a6ac7207cf38b0'}, 'data_contract': {'records': 2000, 'fold_count': 5, 'records_per_fold': 400, 'split_role': 'train', 'base_prediction_origin': 'oof', 'average_target_used': False, 'validation_loaded': False, 'validation_selection': False, 'optional_api_enabled': False, 'external_api_calls_allowed': False}, 'execution': {'authorized_gpus': [0, 1, 2, 3], 'smoke_gpu': 0, 'initialization_policy': 'identical_frozen_start_per_outer_fold_and_candidate', 'initialization_seed': 2026080102, 'fresh_initialization': True, 'checkpoint_reuse_between_candidates': False, 'proxy_implementations_allowed': False}, 'nested_selection': {'outer_folds': [0, 1, 2, 3, 4], 'inner_folds_for_outer': {'0': [1, 2, 3, 4], '1': [0, 2, 3, 4], '2': [0, 1, 3, 4], '3': [0, 1, 2, 4], '4': [0, 1, 2, 3]}, 'selection_data': 'inner_remaining_four_folds_only', 'outer_fold_access_before_selection': False, 'per_outer_procedure': ['exclude_outer_fold_gold_features_and_historical_R17_predictions', 'for_each_inner_validation_regenerate_R16_teacher_by_2_of_3_cross_fit_excluding_inner_validation_and_outer', 'fit_and_score_all_preregistered_candidates_on_inner_only', 'apply_inner_seven_gate_conjunction_against_base_identity', 'select_global_inner_eligible_candidate_by_macro_rmse_with_registered_order_tiebreak', 'fallback_to_base_identity_when_none_eligible', 'after_selection_freeze_regenerate_selected_outer_refit_R16_teacher_by_3_of_4_cross_fit_without_outer_access', 'predict_outer_fold_once_then_unlock_outer_gold_for_metrics'], 'concatenate_outer_predictions_once': True, 'posthoc_selection_on_concatenated_oof': False, 'outer_holdout_gold_or_features_access_before_final_predict': False, 'historical_r17_oof_use_in_selection_or_fitting': False, 'r17_rebuild': 'inner_selection_uses_split_specific_2_of_3_R16_cross_fit_teacher_then_outer_refit_uses_fresh_3_of_4_teacher_after_selection_freeze', 'r16_teacher_regeneration': {'method_family': 'joint_huber_ordinal', 'feature_view': 'consensus_disagreement', 'embedding_view': 'frozen_r0_embedding', 'target': 'raw_axis_gold', 'hidden_dim': 128, 'epochs': 100, 'learning_rate': 0.001, 'huber_delta': 1.0, 'ordinal_loss_weight': 0.5, 'max_correction': 0.75, 'inner_selection_teacher_2_of_3': {'for_each_inner_validation_fold': True, 'teacher_universe': 'other_three_inner_folds_only', 'heldout_rule': 'for_each_of_three_fit_folds_train_on_other_two_then_predict_heldout_once', 'inner_validation_excluded': True, 'outer_holdout_excluded': True, 'reuse_for_outer_refit': False}, 'outer_refit_teacher_3_of_4': {'generated_after_selection_freeze': True, 'teacher_universe': 'all_four_outer_train_folds', 'heldout_rule': 'for_each_of_four_outer_train_folds_train_on_other_three_then_predict_heldout_once', 'outer_holdout_excluded': True, 'reuse_from_inner_selection': False}}, 'eligible_selection': {'gate_reference': 'base_identity_for_every_candidate', 'selection': 'global_minimum_macro_rmse_among_eligible_candidates', 'tie_break': 'registered_candidate_order_then_subvariant_key', 'sequential_incumbent_tournament': False}}, 'candidates': [{'order': 1, 'name': 'base identity', 'method_family': 'identity', 'hyperparameters': {'source': 'baseline_oof'}}, {'order': 2, 'name': 'R17 raw', 'method_family': 'nested_rebuilt_r17_evidence_ridge', 'hyperparameters': {'pseudo_target_source': 'split_specific_fresh_R16_cross_fit_teacher_excluding_current_selection_holdout', 'fit_scope': 'current_outer_train_only', 'outer_prediction_count': 1, 'historical_r17_artifact_used': False, 'embedding_view': 'evidence_hash', 'ridge_alpha': 10.0, 'fit_intercept': True, 'target': 'fresh_R16_inner_OOF_pseudo_target'}}, {'order': 3, 'name': 'conditional R17 delta gate grid', 'method_family': 'gated_delta', 'hyperparameters': {'gate_kind_grid': ['hard', 'sigmoid'], 'gate_threshold_grid': [2.5, 3.0, 3.5, 4.0], 'gate_temperature_grid': [0.1, 0.25, 0.5, 1.0], 'delta_weight_grid': [0.0, 0.25, 0.5, 0.75, 1.0], 'low_identity_threshold_grid': [None, 2.0, 2.5], 'delta_source': 'nested_rebuilt_R17_minus_baseline'}}, {'order': 4, 'name': 'weighted isotonic/piecewise', 'method_family': 'weighted_isotonic_and_piecewise_5knot', 'hyperparameters': {'family_grid': ['weighted_isotonic', 'piecewise_5knot'], 'calibration_source_grid': ['base', 'challenger'], 'equal_gold_band_weights_grid': [False, True], 'piecewise_fixed_knots': [1.0, 2.0, 3.0, 4.0, 5.0], 'bounds': [1.0, 5.0], 'piecewise_fit_objective': 'exact_weighted_least_squares', 'piecewise_constraints': 'bounded_nondecreasing_knot_ordinates', 'piecewise_solver': 'exhaustive_contiguous_active_faces', 'piecewise_active_face_count': 64}}, {'order': 5, 'name': 'low/high tail offsets+3/4 nudge', 'method_family': 'tail_boundary', 'hyperparameters': {'tail_source_grid': ['base', 'challenger'], 'low_offset_grid': [-0.2, -0.1, 0.0, 0.1, 0.2], 'high_offset_grid': [-0.2, -0.1, 0.0, 0.1, 0.2], 'boundary_nudge_grid': [0.0, 0.05, 0.1, 0.2], 'boundary_nudge_direction': 'away_from_3.5', 'bounds': [1.0, 5.0], 'gold_band_access_at_predict': False, 'low_score_threshold': 2.5, 'high_score_threshold': 4.5, 'boundary_center': 3.5, 'boundary_kernel': 'triangular', 'boundary_radius': 0.5}}, {'order': 6, 'name': 'evidence direct ridge alpha grid', 'method_family': 'direct_evidence_ridge', 'hyperparameters': {'alpha_grid': [0.01, 0.1, 1.0, 10.0, 100.0], 'fit_intercept': True, 'target': 'raw_axis_gold', 'proxy_or_distillation': False, 'fit_scope': 'current_outer_train_only', 'embedding_view': 'evidence_hash'}}, {'order': 7, 'name': 'top-two nested ensemble', 'method_family': 'convex_blend_top_two_inner_eligible', 'hyperparameters': {'member_count': 2, 'member_source': 'top_two_inner_gate_eligible_structurally_distinct_candidates', 'candidate_two_weight_grid': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], 'fallback': 'best_single_inner_eligible_or_base_identity'}}], 'inner_promotion_gate': {'operator': 'AND', 'macro_rmse_min_improvement': 0.005, 'equal_group_rmse_min_improvement': 0.01, 'low_tail_must_improve': True, 'high_tail_must_improve': True, 'gold_3_4_balanced_accuracy_min_improvement': 0.01, 'max_axis_rmse_worsening': 0.01, 'max_macro_spearman_fall': 0.005, 'score1_descriptive_only': True, 'require_all_four_inner_folds': True, 'require_finite_metrics': True}, 'outer_final_gate': {'comparison': 'nested_outer_candidate_minus_baseline', 'macro_rmse_min_improvement': 0.01, 'paired_bootstrap_confidence': 0.95, 'paired_bootstrap_candidate_minus_baseline_ci_upper_below_zero': True, 'bootstrap_resamples': 10000, 'bootstrap_seed': 2026080102, 'fallback': 'base_identity', 'freeze_after_gate': True}}


class IterativeTailRemediationError(ValueError):
    """Raised before lineage drift or outer-fold leakage can enter v2."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise IterativeTailRemediationError(message)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    _need(not path.is_absolute() and ".." not in path.parts, "lineage path must be repository-relative")
    return root / path


@dataclass(frozen=True)
class RemediationProtocol:
    path: Path
    raw: Mapping[str, Any]

    @property
    def lineage(self) -> Mapping[str, Any]:
        return self.raw["lineage"]


@dataclass(frozen=True)
class BoundInputAudit:
    records: int
    fold_counts: Mapping[int, int]
    fold_assignment_fingerprint: str
    baseline_sha256: str
    historical_r17_sha256: str
    historical_r17_role: str


def validate_protocol_mapping(raw: Mapping[str, Any], *, path: str | Path = CONFIG_PATH) -> RemediationProtocol:
    """Exact-lock every lineage, candidate, gate, and nested split rule."""
    _need(isinstance(raw, Mapping), "remediation protocol must be an object")
    _need(dict(raw) == EXPECTED_PROTOCOL, "remediation protocol differs from the exact v2 registration")
    _need(raw["schema_version"] == SCHEMA_VERSION and raw["run_id"] == RUN_ID, "v2 identity differs")
    nested = raw["nested_selection"]
    for outer in range(5):
        inner = nested["inner_folds_for_outer"][str(outer)]
        _need(len(inner) == 4 and set(inner) == set(range(5)) - {outer}, "outer/inner fold complement differs")
    _need(not nested["outer_fold_access_before_selection"], "outer fold access before selection is forbidden")
    _need(not nested["outer_holdout_gold_or_features_access_before_final_predict"], "outer holdout must remain sealed through refit")
    _need(not nested["historical_r17_oof_use_in_selection_or_fitting"], "historical R17 OOF cannot be a v2 feature")
    _need(not nested["posthoc_selection_on_concatenated_oof"], "posthoc selection on concatenated OOF is forbidden")
    teacher = nested["r16_teacher_regeneration"]
    inner_teacher = teacher["inner_selection_teacher_2_of_3"]
    outer_teacher = teacher["outer_refit_teacher_3_of_4"]
    _need(
        inner_teacher["inner_validation_excluded"]
        and inner_teacher["outer_holdout_excluded"]
        and not inner_teacher["reuse_for_outer_refit"],
        "inner selection teacher must exclude both holdouts and cannot be reused for outer refit",
    )
    _need(
        outer_teacher["generated_after_selection_freeze"]
        and outer_teacher["outer_holdout_excluded"]
        and not outer_teacher["reuse_from_inner_selection"],
        "outer refit teacher must be regenerated only after selection freeze",
    )
    selection = nested["eligible_selection"]
    _need(
        selection["gate_reference"] == "base_identity_for_every_candidate"
        and selection["selection"] == "global_minimum_macro_rmse_among_eligible_candidates"
        and not selection["sequential_incumbent_tournament"],
        "inner candidate selection must be baseline-relative global eligible minimum",
    )
    _need(raw["inner_promotion_gate"]["operator"] == "AND", "all seven inner gates must be conjunctive")
    _need(not raw["execution"]["proxy_implementations_allowed"], "proxy implementation deviations are forbidden")
    candidates = raw["candidates"]
    gate = candidates[2]["hyperparameters"]
    _need(gate["gate_kind_grid"] == list(PREDECLARED_GRIDS["gate_kind"]), "gate kind implementation grid differs")
    _need(gate["gate_threshold_grid"] == list(PREDECLARED_GRIDS["gate_threshold"]), "gate threshold implementation grid differs")
    _need(gate["gate_temperature_grid"] == list(PREDECLARED_GRIDS["gate_temperature"]), "gate temperature implementation grid differs")
    _need(gate["delta_weight_grid"] == list(PREDECLARED_GRIDS["delta_weight"]), "delta weight implementation grid differs")
    _need(gate["low_identity_threshold_grid"] == list(PREDECLARED_GRIDS["low_identity_threshold"]), "low identity implementation grid differs")
    calibration = candidates[3]["hyperparameters"]
    _need(calibration["calibration_source_grid"] == list(PREDECLARED_GRIDS["calibration_source"]), "calibration source grid differs")
    _need(calibration["piecewise_fixed_knots"] == KNOTS_5.tolist(), "piecewise implementation knots differ")
    _need(
        calibration["piecewise_fit_objective"] == "exact_weighted_least_squares"
        and calibration["piecewise_constraints"] == "bounded_nondecreasing_knot_ordinates"
        and calibration["piecewise_solver"] == "exhaustive_contiguous_active_faces"
        and calibration["piecewise_active_face_count"] == 64,
        "piecewise exact solver registration differs",
    )
    tail = candidates[4]["hyperparameters"]
    _need(tail["tail_source_grid"] == list(PREDECLARED_GRIDS["tail_source"]), "tail source grid differs")
    _need(tail["low_offset_grid"] == list(PREDECLARED_GRIDS["low_offset"]), "low offset grid differs")
    _need(tail["high_offset_grid"] == list(PREDECLARED_GRIDS["high_offset"]), "high offset grid differs")
    _need(tail["boundary_nudge_grid"] == list(PREDECLARED_GRIDS["boundary_nudge"]), "boundary nudge grid differs")
    _need(
        tail["low_score_threshold"] == 2.5
        and tail["high_score_threshold"] == 4.5
        and tail["boundary_center"] == 3.5
        and tail["boundary_kernel"] == "triangular"
        and tail["boundary_radius"] == 0.5,
        "tail-boundary fixed inference structure differs",
    )
    _need(candidates[6]["hyperparameters"]["candidate_two_weight_grid"] == list(PREDECLARED_GRIDS["blend_weight"]), "convex blend grid differs")
    return RemediationProtocol(Path(path), raw)


def load_protocol(path: str | Path = CONFIG_PATH) -> RemediationProtocol:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IterativeTailRemediationError(f"v2 protocol is unreadable: {config_path}") from exc
    return validate_protocol_mapping(raw, path=config_path)


def outer_inner_folds(protocol: RemediationProtocol, outer_fold: int) -> tuple[int, ...]:
    """Return the predeclared four-fold inner complement for one outer fold."""
    _need(type(outer_fold) is int and 0 <= outer_fold < 5, "outer fold must be 0..4")
    return tuple(protocol.raw["nested_selection"]["inner_folds_for_outer"][str(outer_fold)])


def _prediction_rows(path: Path, *, historical_r17: bool) -> tuple[dict[str, int], int]:
    assignments: dict[str, int] = {}
    required = {"fold", "prediction", "source_id"} if historical_r17 else {
        "source_id", "fold", "continuous_prediction", "half_up_integer_prediction", "reference_score",
    }
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            raw = json.loads(line)
            _need(isinstance(raw, Mapping) and set(raw) == required, f"prediction schema differs at line {line_number}")
            source_id, fold = raw["source_id"], raw["fold"]
            _need(isinstance(source_id, str) and source_id and source_id not in assignments, "prediction source IDs differ")
            _need(type(fold) is int and 0 <= fold < 5, "prediction fold differs")
            scores = raw["prediction"] if historical_r17 else raw["continuous_prediction"]
            if historical_r17:
                _need(isinstance(scores, list) and len(scores) == len(AXES), "historical R17 must contain exactly three ordered axes")
                values = scores
            else:
                _need(isinstance(scores, Mapping) and set(scores) == set(AXES), "prediction axes differ or average was included")
                values = [scores[axis] for axis in AXES]
            _need(all(type(value) in {int, float} and math.isfinite(float(value)) for value in values), "prediction values must be finite")
            assignments[source_id] = fold
    return assignments, len(assignments)


def validate_bound_inputs(protocol: RemediationProtocol, *, root: str | Path = ".") -> BoundInputAudit:
    """Validate actual v1 lineage and exact 2,000-row, 5x400 alignment.

    Historical R17 values are parsed only to prove population/fold lineage; the
    returned audit contains no row-level predictions and the protocol forbids
    their use during v2 selection or fitting.
    """
    root_path = Path(root)
    lineage = protocol.lineage
    checks = (
        ("v1_config_path", "v1_config_sha256", "v1 config"),
        ("canonical_train_path", "canonical_train_sha256", "canonical train"),
        ("baseline_oof_path", "baseline_oof_sha256", "baseline OOF"),
        ("embedding_rows_path", "embedding_rows_sha256", "embedding rows"),
        ("score_blind_feature_cache_path", "score_blind_feature_cache_sha256", "score-blind feature cache"),
        ("score_blind_feature_manifest_path", "score_blind_feature_manifest_sha256", "score-blind feature manifest"),
        ("historical_r17_selected_prediction_path", "historical_r17_selected_prediction_sha256", "historical R17 OOF"),
    )
    resolved: dict[str, Path] = {}
    for path_key, sha_key, label in checks:
        artifact = _bound_path(root_path, lineage[path_key])
        _need(artifact.is_file() and not artifact.is_symlink(), f"{label} path is missing or mutable")
        _need(_sha256(artifact) == lineage[sha_key], f"{label} checksum differs")
        resolved[path_key] = artifact
    manifest_path = _bound_path(root_path, lineage["embedding_manifest_path"])
    _need(manifest_path.is_file() and not manifest_path.is_symlink(), "embedding manifest path is missing or mutable")
    commit = lineage["v1_execution_git_commit"]
    git_check = subprocess.run(
        ["git", "-C", str(root_path), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _need(git_check.returncode == 0, "v1 execution commit is unavailable")

    try:
        feature_manifest = json.loads(resolved["score_blind_feature_manifest_path"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IterativeTailRemediationError("score-blind feature manifest is unreadable") from exc
    _need(feature_manifest.get("records") == 2000, "score-blind feature population differs")
    _need(feature_manifest.get("axes") == list(AXES), "score-blind feature axes differ")
    _need(feature_manifest.get("cache_sha256") == lineage["score_blind_feature_cache_sha256"], "score-blind cache binding differs")
    _need(feature_manifest.get("score_conditioning") is False and feature_manifest.get("validation_loaded") is False, "score-blind isolation differs")

    manifest, embedding_rows = load_embedding_artifact(manifest_path, resolved["embedding_rows_path"])
    _need(manifest.split_role == "train" and manifest.base_prediction_origin == "oof", "embedding input is not train OOF")
    _need(not manifest.evaluation_only and not manifest.contains_average_target, "validation or average target is forbidden")
    baseline, baseline_count = _prediction_rows(resolved["baseline_oof_path"], historical_r17=False)
    historical, historical_count = _prediction_rows(resolved["historical_r17_selected_prediction_path"], historical_r17=True)
    embedding = {row.source_id: row.oof_fold for row in embedding_rows}
    _need(baseline_count == historical_count == len(embedding_rows) == 2000, "v2 lineage population must be exactly 2,000")
    _need(baseline == historical == embedding, "baseline, historical R17, and embedding fold assignments differ")
    counts = Counter(baseline.values())
    _need(counts == Counter({fold: 400 for fold in range(5)}), "v2 lineage must contain exactly 400 rows per fold")
    payload = "\n".join(f"{source_id}:{fold}" for source_id, fold in sorted(baseline.items()))
    fingerprint = sha256(payload.encode("utf-8")).hexdigest()
    _need(fingerprint == lineage["fold_assignment_fingerprint"], "fold assignment fingerprint differs")
    return BoundInputAudit(
        records=2000,
        fold_counts=dict(sorted(counts.items())),
        fold_assignment_fingerprint=fingerprint,
        baseline_sha256=lineage["baseline_oof_sha256"],
        historical_r17_sha256=lineage["historical_r17_selected_prediction_sha256"],
        historical_r17_role=lineage["historical_r17_role"],
    )


load_config = load_protocol
