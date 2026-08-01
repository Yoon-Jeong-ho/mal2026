"""Sealed 5x4 nested runner for the V12 Terra/Luna rationale-semantic study."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .api_rationale_data import load_writing_rows
from .iterative_official_dual_agent_data import load_dual_candidates
from .iterative_official_dual_agent_models import (
    FEATURE_DIM as STRUCTURED_FEATURE_DIM,
    build_dual_agent_features,
)
from .iterative_official_rationale_embedding_data import (
    FEATURE_DIM as SEMANTIC_FEATURE_DIM,
    load_feature_artifact,
)
from .iterative_official_rationale_semantic_models import (
    RationaleSemanticSpec,
    candidate_specs,
    fit_predict_rationale_semantic,
)
from .iterative_official_rationale_semantic_protocol import RUN_ID, load_protocol, validate_bound_inputs
from .iterative_official_rationale_semantic_selection import final_gate, fold_direction_diagnostics, select_candidate
from .iterative_tail_metrics import compute_iterative_tail_metrics
from .iterative_tail_remediation_runner import _bootstrap_macro_rmse
from .iterative_tail_runner import ExperimentData, load_experiment_data


PUBLIC_ROOT = Path("outputs/iterative-official-rationale-semantic-v12") / RUN_ID
RESTRICTED_ROOT = Path("data/processed/restricted/iterative_official_rationale_semantic_v12") / RUN_ID
SEED = 2026080212


class OfficialRationaleSemanticRunError(RuntimeError):
    """Raised when V12 nesting, coverage, selection, or privacy differs."""


@dataclass(frozen=True)
class OfficialRationaleSemanticData:
    experiment: ExperimentData
    semantic_features: np.ndarray
    structured_features: np.ndarray
    feature_audit: Mapping[str, Any]
    provenance: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_safe(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _indices(data: ExperimentData, folds: Sequence[int]) -> np.ndarray:
    return np.flatnonzero(np.isin(data.folds, np.asarray(tuple(folds), dtype=int)))


def load_official_rationale_semantic_data() -> OfficialRationaleSemanticData:
    protocol = load_protocol()
    validate_bound_inputs(protocol)
    experiment = load_experiment_data()
    # This linkage map is deliberately constructed through the score-free
    # canonical loader. Gold scores never enter candidate verification.
    writings = load_writing_rows("train", include_scores=False)
    essay_sha256_by_source = {row.identifier: sha256(row.essay.encode("utf-8")).hexdigest() for row in writings}
    if len(essay_sha256_by_source) != 2000 or set(essay_sha256_by_source) != set(experiment.source_ids):
        raise OfficialRationaleSemanticRunError("canonical score-free essay SHA population differs")
    lineage = protocol.raw["lineage"]
    candidates, provenance = load_dual_candidates(
        lineage["terra_candidate_manifest_path"], lineage["terra_candidate_rows_path"],
        lineage["luna_candidate_manifest_path"], lineage["luna_candidate_rows_path"],
        essay_sha256_by_source=essay_sha256_by_source,
    )
    terra = [candidate for candidate in candidates if candidate.agent_source == "terra"]
    luna = [candidate for candidate in candidates if candidate.agent_source == "luna"]
    structured, structured_audit = build_dual_agent_features(
        experiment.base, experiment.source_ids, terra, luna,
    )
    semantic_manifest, semantic_rows = load_feature_artifact(
        lineage["generated_feature_manifest_path"],
        lineage["generated_feature_rows_path"],
        expected_source_ids=experiment.source_ids,
    )
    semantic = np.asarray([row.features for row in semantic_rows], dtype=np.float64)
    if (
        structured_audit.get("records") != 2000
        or structured_audit.get("dimensions") != STRUCTURED_FEATURE_DIM
        or structured_audit.get("human_or_reference_score_read_or_prompted") is not False
        or semantic.shape != (2000, SEMANTIC_FEATURE_DIM)
        or not np.isfinite(semantic).all()
        or semantic_manifest.get("candidate_score_in_embedding_text") is not False
        or semantic_manifest.get("validation_loaded") is not False
        or provenance.get("candidate_count") != 12000 or provenance.get("source_count") != 2
        or provenance.get("row_content_in_provenance") is not False
        or len(terra) != 6000 or len(luna) != 6000
    ):
        raise OfficialRationaleSemanticRunError("official rationale-semantic feature provenance differs")
    feature_audit = {
        "semantic": {
            "records": semantic_manifest["records"],
            "dimensions": semantic_manifest["feature_dim"],
            "feature_matrix_sha256": semantic_manifest["feature_matrix_sha256"],
            "feature_rows_sha256": semantic_manifest["feature_rows_sha256"],
            "model_id": semantic_manifest["model_id"],
            "model_revision": semantic_manifest["model_revision"],
            "projection_matrix_sha256": semantic_manifest["projection_matrix_sha256"],
            "candidate_score_in_embedding_text": False,
            "validation_loaded": False,
        },
        "structured": structured_audit,
        "semantic_dimensions": SEMANTIC_FEATURE_DIM,
        "structured_dimensions": STRUCTURED_FEATURE_DIM,
        "fusion_dimensions": SEMANTIC_FEATURE_DIM + STRUCTURED_FEATURE_DIM,
        "average_target_used": False,
    }
    return OfficialRationaleSemanticData(
        experiment, semantic, structured, feature_audit, provenance,
    )


def _candidate_oof(
    bundle: OfficialRationaleSemanticData,
    spec: RationaleSemanticSpec,
    outer_fold: int,
    *,
    device: str,
) -> tuple[np.ndarray, list[Mapping[str, Any]]]:
    data = bundle.experiment
    universe = tuple(fold for fold in range(5) if fold != outer_fold)
    prediction = np.full_like(data.targets, np.nan, dtype=np.float64)
    audit: list[Mapping[str, Any]] = []
    for inner_validation in universe:
        train_folds = tuple(fold for fold in universe if fold != inner_validation)
        train, predict = _indices(data, train_folds), _indices(data, (inner_validation,))
        if outer_fold in data.folds[train] or outer_fold in data.folds[predict] or inner_validation in data.folds[train]:
            raise OfficialRationaleSemanticRunError("V12 inner/outer fold sealing differs")
        result = fit_predict_rationale_semantic(
            spec,
            bundle.semantic_features[train], bundle.structured_features[train],
            data.base[train], data.targets[train],
            bundle.semantic_features[predict], bundle.structured_features[predict],
            data.base[predict], device=device,
        )
        prediction[predict] = result.predictions
        audit.append({
            "outer_fold": outer_fold,
            "inner_validation_fold": inner_validation,
            "train_folds": list(train_folds),
            "forbidden_folds": [outer_fold, inner_validation],
            "train_records": len(train),
            "prediction_records": len(predict),
            "fit_predict": result.audit,
        })
    outer_train = _indices(data, universe)
    outer = _indices(data, (outer_fold,))
    expected_outer = len(data.folds) // 5
    if (
        len(data.folds) % 5 or len(outer_train) != 4 * expected_outer or len(outer) != expected_outer
        or not np.isfinite(prediction[outer_train]).all() or np.isfinite(prediction[outer]).any()
    ):
        raise OfficialRationaleSemanticRunError("V12 candidate OOF coverage differs")
    return prediction, audit


def _outer_refit(
    bundle: OfficialRationaleSemanticData,
    spec: RationaleSemanticSpec,
    outer_fold: int,
    *,
    device: str,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    data = bundle.experiment
    train_folds = tuple(fold for fold in range(5) if fold != outer_fold)
    train, predict = _indices(data, train_folds), _indices(data, (outer_fold,))
    result = fit_predict_rationale_semantic(
        spec,
        bundle.semantic_features[train], bundle.structured_features[train],
        data.base[train], data.targets[train],
        bundle.semantic_features[predict], bundle.structured_features[predict],
        data.base[predict], device=device,
    )
    return result.predictions, {
        "selected_spec_frozen_before_refit": True,
        "outer_fold": outer_fold,
        "train_folds": list(train_folds),
        "outer_gold_used_before_prediction": False,
        "fit_predict": result.audit,
    }


def run_outer_fold(outer_fold: int, *, device: str = "cuda:0") -> Mapping[str, Any]:
    if outer_fold not in range(5):
        raise OfficialRationaleSemanticRunError("outer fold must be 0..4")
    protocol = load_protocol()
    validate_bound_inputs(protocol)
    bundle = load_official_rationale_semantic_data()
    data = bundle.experiment
    specs = candidate_specs()
    universe = tuple(fold for fold in range(5) if fold != outer_fold)
    outer_train, outer = _indices(data, universe), _indices(data, (outer_fold,))
    baseline_inner = compute_iterative_tail_metrics(data.targets[outer_train], data.base[outer_train])
    records, metrics_by_id = [], {}
    progress_path = PUBLIC_ROOT / f"outer-{outer_fold}" / "progress.json"
    for position, spec in enumerate(specs, start=1):
        prediction, audit = _candidate_oof(bundle, spec, outer_fold, device=device)
        metrics = compute_iterative_tail_metrics(data.targets[outer_train], prediction[outer_train])
        metrics_by_id[spec.variant_id] = metrics
        records.append({
            "cycle": spec.cycle,
            "variant_id": spec.variant_id,
            "head_kind": spec.head_kind,
            "ridge_alpha": spec.ridge_alpha,
            "max_correction": spec.max_correction,
            "confidence": spec.confidence,
            "window": spec.window,
            "epsilon": spec.epsilon,
            "l2": spec.l2,
            "inner_metrics": metrics,
            "inner_fold_audit": audit,
        })
        _write_json(progress_path, {
            "schema_version": "mal2026-iterative-official-rationale-semantic-progress-v12",
            "status": "running", "outer_fold": outer_fold,
            "completed_candidates": position, "candidate_count": 3,
            "completed_inner_predictions": position * 4,
            "percent": 100.0 * position / 3.0,
            "last_completed_candidate": spec.variant_id,
        })
    selection = select_candidate(specs, metrics_by_id, baseline_inner, protocol.raw)
    by_id = {record["variant_id"]: record for record in records}
    for decision in selection["decisions"]:
        by_id[decision["variant_id"]]["baseline_relative_decision"] = decision
    selected_id = selection["selected_id"]
    if selected_id == "baseline":
        outer_prediction = data.base[outer].astype(np.float64)
        outer_refit_audit = None
        selected_cycle = None
    else:
        selected_spec = next(spec for spec in specs if spec.variant_id == selected_id)
        outer_prediction, outer_refit_audit = _outer_refit(bundle, selected_spec, outer_fold, device=device)
        selected_cycle = selected_spec.cycle
    # Outer gold is unlocked only after the selected/fallback prediction exists.
    baseline_outer_metrics = compute_iterative_tail_metrics(data.targets[outer], data.base[outer])
    selected_outer_metrics = compute_iterative_tail_metrics(data.targets[outer], outer_prediction)
    restricted_path = RESTRICTED_ROOT / f"outer-{outer_fold}" / "predictions.jsonl"
    _write_jsonl(restricted_path, [
        {
            "source_id": data.source_ids[index], "outer_fold": outer_fold,
            "baseline": [float(value) for value in data.base[index]],
            "selected": [float(value) for value in prediction],
        }
        for index, prediction in zip(outer, outer_prediction, strict=True)
    ])
    result = {
        "schema_version": "mal2026-iterative-official-rationale-semantic-outer-v12",
        "status": "completed",
        "outer_fold": outer_fold,
        "outer_train_records": len(outer_train),
        "outer_holdout_records": len(outer),
        "inner_fold_count": 4,
        "candidate_count": 3,
        "candidate_records": records,
        "selection": selection,
        "selected_candidate": selected_id if selected_id != "baseline" else "exact-r0-oof-baseline",
        "selected_cycle": selected_cycle,
        "fell_back_to_baseline": selected_id == "baseline",
        "outer_refit_audit": outer_refit_audit,
        "baseline_metrics": baseline_outer_metrics,
        "selected_metrics": selected_outer_metrics,
        "feature_audit": bundle.feature_audit,
        "official_candidate_provenance": bundle.provenance,
        "restricted_prediction_sha256": _sha256(restricted_path),
        "outer_gold_used_before_selection_freeze_or_prediction": False,
        "posthoc_selection_on_outer_gold": False,
        "validation_loaded": False,
        "average_target_used": False,
        "external_api_calls": 0,
    }
    _write_json(PUBLIC_ROOT / f"outer-{outer_fold}" / "result.json", result)
    _write_json(progress_path, {
        "schema_version": "mal2026-iterative-official-rationale-semantic-progress-v12",
        "status": "completed", "outer_fold": outer_fold,
        "completed_candidates": 3, "candidate_count": 3,
        "completed_inner_predictions": 12, "percent": 100.0,
        "last_completed_candidate": specs[-1].variant_id,
    })
    return result


def _read_outer_predictions(data: ExperimentData) -> tuple[np.ndarray, np.ndarray, list[Mapping[str, Any]]]:
    baseline, selected = np.full_like(data.targets, np.nan, dtype=np.float64), np.full_like(data.targets, np.nan, dtype=np.float64)
    index = {source_id: row for row, source_id in enumerate(data.source_ids)}
    seen: set[str] = set()
    audits = []
    for outer_fold in range(5):
        result_path = PUBLIC_ROOT / f"outer-{outer_fold}" / "result.json"
        prediction_path = RESTRICTED_ROOT / f"outer-{outer_fold}" / "predictions.jsonl"
        if not result_path.is_file() or not prediction_path.is_file():
            raise OfficialRationaleSemanticRunError("V12 outer artifact missing")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("schema_version") != "mal2026-iterative-official-rationale-semantic-outer-v12"
            or result.get("status") != "completed" or result.get("candidate_count") != 3
            or result.get("restricted_prediction_sha256") != _sha256(prediction_path)
        ):
            raise OfficialRationaleSemanticRunError("V12 outer artifact binding differs")
        count = 0
        with prediction_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                source_id = row.get("source_id")
                if source_id not in index or source_id in seen or row.get("outer_fold") != outer_fold:
                    raise OfficialRationaleSemanticRunError("V12 outer row identity differs")
                position = index[source_id]
                if int(data.folds[position]) != outer_fold:
                    raise OfficialRationaleSemanticRunError("V12 outer row fold differs")
                base_vector = np.asarray(row.get("baseline"), dtype=np.float64)
                selected_vector = np.asarray(row.get("selected"), dtype=np.float64)
                if base_vector.shape != (3,) or selected_vector.shape != (3,) or not np.isfinite(base_vector).all() or not np.isfinite(selected_vector).all():
                    raise OfficialRationaleSemanticRunError("V12 outer prediction vector differs")
                baseline[position], selected[position] = base_vector, selected_vector
                seen.add(source_id); count += 1
        if count != 400:
            raise OfficialRationaleSemanticRunError("V12 outer row count differs")
        audits.append({
            "outer_fold": outer_fold,
            "selected_candidate": result["selected_candidate"],
            "selected_cycle": result["selected_cycle"],
            "fell_back_to_baseline": result["fell_back_to_baseline"],
            "result_sha256": _sha256(result_path),
            "prediction_sha256": _sha256(prediction_path),
        })
    if len(seen) != 2000 or not np.isfinite(baseline).all() or not np.isfinite(selected).all():
        raise OfficialRationaleSemanticRunError("V12 concatenated outer coverage differs")
    return baseline, selected, audits


def aggregate_outer_results() -> Mapping[str, Any]:
    protocol = load_protocol()
    validate_bound_inputs(protocol)
    bundle = load_official_rationale_semantic_data()
    data = bundle.experiment
    baseline, nested_selected, audits = _read_outer_predictions(data)
    if not np.allclose(baseline, data.base.astype(np.float64), rtol=0.0, atol=1e-7):
        raise OfficialRationaleSemanticRunError("V12 nested baseline differs from exact R0")
    baseline_metrics = compute_iterative_tail_metrics(data.targets, baseline)
    selected_metrics = compute_iterative_tail_metrics(data.targets, nested_selected)
    bootstrap = _bootstrap_macro_rmse(data.targets, baseline, nested_selected, resamples=10_000, seed=SEED)
    bootstrap = {
        "schema_version": "mal2026-iterative-official-rationale-semantic-bootstrap-v12",
        **bootstrap,
    }
    decision = final_gate(protocol.raw, baseline_metrics, selected_metrics, bootstrap)
    final_pass = bool(decision.get("pass"))
    final_prediction = nested_selected if final_pass else baseline
    per_fold_baseline, per_fold_selected = [], []
    for fold in range(5):
        mask = data.folds == fold
        per_fold_baseline.append(compute_iterative_tail_metrics(data.targets[mask], baseline[mask]))
        per_fold_selected.append(compute_iterative_tail_metrics(data.targets[mask], nested_selected[mask]))
    final_path = RESTRICTED_ROOT / "final_predictions.jsonl"
    _write_jsonl(final_path, [
        {
            "source_id": source_id, "fold": int(data.folds[row]),
            "prediction": [float(value) for value in final_prediction[row]],
            "role": "nested_selected_rationale_semantic_stack" if final_pass else "exact_r0_oof_baseline_fallback",
        }
        for row, source_id in enumerate(data.source_ids)
    ])
    payload = {
        "schema_version": "mal2026-iterative-official-rationale-semantic-aggregate-v12",
        "status": "completed",
        "record_count": 2000,
        "outer_fold_count": 5,
        "inner_fold_count_per_outer": 4,
        "candidate_count_per_outer": 3,
        "outer_audits": audits,
        "baseline_metrics": baseline_metrics,
        "nested_selected_metrics": selected_metrics,
        "paired_bootstrap": bootstrap,
        "final_decision": decision,
        "final_gate_pass": final_pass,
        "final_selection": "nested-selected-rationale-semantic-stack" if final_pass else "exact-r0-oof-baseline-fallback",
        "final_prediction_sha256": _sha256(final_path),
        "fold_direction_diagnostics": fold_direction_diagnostics(per_fold_baseline, per_fold_selected),
        "feature_audit": bundle.feature_audit,
        "official_candidate_provenance": bundle.provenance,
        "adaptive_after_v11_observed": True,
        "v12_full_oof_prestudy_before_preregistration": False,
        "independent_confirmation_or_generalization_claim": False,
        "posthoc_selection_on_concatenated_outer_predictions": False,
        "validation_loaded": False,
        "average_target_used": False,
        "external_api_calls": 0,
    }
    aggregate_path = PUBLIC_ROOT / "aggregate.json"
    _write_json(aggregate_path, payload)
    completion = {
        "schema_version": "mal2026-iterative-official-rationale-semantic-completion-v12",
        "status": "completed_final_gate_pass_development_only" if final_pass else "completed_no_promotion_baseline_retained",
        "aggregate_sha256": _sha256(aggregate_path),
        "final_prediction_sha256": _sha256(final_path),
        "final_gate_pass": final_pass,
        "final_selection": payload["final_selection"],
        "gpu_scope": [0, 1, 2, 3],
        "external_api_calls": 0,
        "validation_loaded": False,
        "average_target_used": False,
        "independent_hidden_evaluation_still_required": True,
    }
    _write_json(PUBLIC_ROOT / "completion.json", completion)
    return payload


def gpu0_smoke(*, device: str = "cuda:0") -> Mapping[str, Any]:
    protocol = load_protocol()
    validate_bound_inputs(protocol)
    bundle = load_official_rationale_semantic_data()
    data = bundle.experiment
    train = np.concatenate([_indices(data, (fold,))[:32] for fold in (1, 2, 3)])
    predict = _indices(data, (4,))[:32]
    records = []
    for spec in candidate_specs():
        result = fit_predict_rationale_semantic(
            spec,
            bundle.semantic_features[train], bundle.structured_features[train],
            data.base[train], data.targets[train],
            bundle.semantic_features[predict], bundle.structured_features[predict],
            data.base[predict], device=device,
        )
        if result.predictions.shape != (32, 3) or not np.isfinite(result.predictions).all():
            raise OfficialRationaleSemanticRunError("V12 smoke prediction differs")
        residual_hash = str(result.audit["residual"]["coefficient_sha256"])
        head_hashes = [
            value
            for head in result.audit["heads"].values()
            for value in head["axis_coefficient_sha256"]
        ]
        if (len(residual_hash) != 64 or (spec.head_kind != "identity" and not head_hashes)
                or any(len(str(value)) != 64 for value in head_hashes)):
            raise OfficialRationaleSemanticRunError("V12 smoke coefficient hash differs")
        records.append({"variant_id": spec.variant_id, "audit": result.audit})
    payload = {
        "schema_version": "mal2026-iterative-official-rationale-semantic-smoke-v12",
        "status": "completed", "gpu": 0,
        "train_records": len(train), "prediction_records": len(predict),
        "train_original_folds": [1, 2, 3], "prediction_original_fold": 4,
        "candidates": records,
        "feature_audit": bundle.feature_audit,
        "official_candidate_provenance": bundle.provenance,
        "validation_loaded": False, "average_target_used": False, "external_api_calls": 0,
    }
    _write_json(PUBLIC_ROOT / "smoke.json", payload)
    return payload


__all__ = [
    "PUBLIC_ROOT", "RESTRICTED_ROOT", "SEED", "OfficialRationaleSemanticData", "OfficialRationaleSemanticRunError",
    "aggregate_outer_results", "gpu0_smoke", "load_official_rationale_semantic_data", "run_outer_fold",
]
