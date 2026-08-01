"""Leakage-sealed nested execution for the fixed V6 directional learners."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mal2026.iterative_tail_directional_models import (
    DirectionalResult,
    DirectionalSpec,
    apply,
    candidate_specs,
    fit,
)
from mal2026.iterative_tail_directional_protocol import (
    RUN_ID,
    DirectionalProtocol,
    load_protocol,
    validate_bound_inputs,
    validate_model_inventory,
)
from mal2026.iterative_tail_directional_selection import (
    final_gate,
    fold_diagnostics,
    select_candidate,
)
from mal2026.iterative_tail_metrics import compute_iterative_tail_metrics
from mal2026.iterative_tail_remediation_runner import _bootstrap_macro_rmse, _indices
from mal2026.iterative_tail_runner import ExperimentData, load_experiment_data


PUBLIC_ROOT = Path("outputs/iterative-tail-directional-v6") / RUN_ID
RESTRICTED_ROOT = Path("data/processed/restricted/iterative_tail_directional_v6") / RUN_ID
SEED = 2026080206


class IterativeTailDirectionalRunError(RuntimeError):
    """Raised when V6 inventory, nesting, projection, coverage, or privacy differs."""


@dataclass(frozen=True)
class CandidateOOF:
    predictions: np.ndarray
    audit: Sequence[Mapping[str, Any]]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _outer_train_folds(outer_fold: int) -> tuple[int, ...]:
    if outer_fold not in range(5):
        raise IterativeTailDirectionalRunError("outer fold must be 0..4")
    return tuple(fold for fold in range(5) if fold != outer_fold)


def _projection_matrix(protocol: DirectionalProtocol) -> np.ndarray:
    registered = protocol.raw["random_projection"]
    if (
        registered["input_dimensions"] != 4672
        or registered["output_dimensions"] != 64
        or registered["seed"] != SEED
        or registered["generator"] != "numpy.random.default_rng_PCG64"
        or registered["matrix_distribution"] != "rademacher_pm1_over_sqrt_output_dimensions"
        or registered["fit_to_data"]
        or registered["gold_used"]
    ):
        raise IterativeTailDirectionalRunError("random projection registration differs")
    rng = np.random.default_rng(SEED)
    matrix = rng.integers(0, 2, size=(4672, 64), dtype=np.int8).astype(np.float32)
    matrix = (matrix * 2.0 - 1.0) / np.sqrt(np.float32(64.0))
    if matrix.shape != (4672, 64) or not np.isfinite(matrix).all():
        raise IterativeTailDirectionalRunError("random projection matrix differs")
    return matrix


def projection_audit(protocol: DirectionalProtocol | None = None) -> Mapping[str, Any]:
    protocol = protocol or load_protocol()
    matrix = _projection_matrix(protocol)
    return {
        "input_dimensions": 4672,
        "output_dimensions": 64,
        "seed": SEED,
        "generator": "numpy.random.default_rng_PCG64",
        "matrix_distribution": "rademacher_pm1_over_sqrt_output_dimensions",
        "matrix_sha256": sha256(matrix.tobytes(order="C")).hexdigest(),
        "fit_to_data": False,
        "gold_used": False,
    }


def _directional_features(data: ExperimentData, protocol: DirectionalProtocol) -> np.ndarray:
    evidence = data.evidence.view("evidence_hash")
    if (
        evidence is None
        or evidence.ndim != 2
        or len(evidence) != len(data.embeddings)
        or data.embeddings.shape[1] != 4096
        or evidence.shape[1] != 576
    ):
        raise IterativeTailDirectionalRunError("frozen embedding or score-blind evidence dimensions differ")
    raw = np.concatenate((data.embeddings.astype(np.float32), evidence.astype(np.float32)), axis=1)
    if raw.shape[1] != 4672 or not np.isfinite(raw).all():
        raise IterativeTailDirectionalRunError("directional raw feature matrix differs")
    projected = (raw @ _projection_matrix(protocol)).astype(np.float32)
    if projected.shape != (len(raw), 64) or not np.isfinite(projected).all():
        raise IterativeTailDirectionalRunError("directional projected feature matrix differs")
    return projected


def _candidate_inner_oof(
    data: ExperimentData,
    features: np.ndarray,
    outer_fold: int,
    spec: DirectionalSpec,
) -> CandidateOOF:
    """Fresh-fit one candidate S->D four times with internal expert OOF labels."""
    outer_folds = _outer_train_folds(outer_fold)
    outer_train = _indices(data, outer_folds)
    prediction = np.full_like(data.base, np.nan, dtype=np.float32)
    seen = np.zeros(len(data.targets), dtype=np.int8)
    audits: list[Mapping[str, Any]] = []
    for inner_validation in outer_folds:
        train_folds = tuple(fold for fold in outer_folds if fold != inner_validation)
        train = _indices(data, train_folds)
        predict = _indices(data, (inner_validation,))
        if len(train) != 1200 or len(predict) != 400:
            raise IterativeTailDirectionalRunError("inner S/D counts differ from 1200/400")
        if np.any(np.isin(data.folds[train], (outer_fold, inner_validation))):
            raise IterativeTailDirectionalRunError("D or O entered candidate fitting")
        fitted = fit(spec, features[train], data.base[train], data.targets[train], data.folds[train])
        result: DirectionalResult = apply(fitted, features[predict], data.base[predict])
        if result.predictions.shape != (400, 3) or not np.isfinite(result.predictions).all():
            raise IterativeTailDirectionalRunError("inner candidate prediction coverage differs")
        prediction[predict] = result.predictions
        seen[predict] += 1
        audits.append({
            "outer_fold": outer_fold,
            "inner_validation_fold": inner_validation,
            "train_folds": list(train_folds),
            "forbidden_folds": [inner_validation, outer_fold],
            "train_count": len(train),
            "predict_count": len(predict),
            "fit": _json_safe(fitted.audit),
            "predict": _json_safe(result.audit),
        })
    outer_holdout = _indices(data, (outer_fold,))
    if not np.all(seen[outer_train] == 1) or np.any(seen[outer_holdout]):
        raise IterativeTailDirectionalRunError("candidate OOF is not exactly-once and outer-sealed")
    if not np.isfinite(prediction[outer_train]).all() or np.isfinite(prediction[outer_holdout]).any():
        raise IterativeTailDirectionalRunError("candidate OOF population differs")
    return CandidateOOF(prediction[outer_train], tuple(audits))


def _fit_and_select_candidates(
    protocol: DirectionalProtocol,
    data: ExperimentData,
    features: np.ndarray,
    outer_fold: int,
    *,
    device: str,
    progress_callback: Callable[[int, DirectionalSpec], None] | None = None,
) -> tuple[DirectionalSpec | None, list[Mapping[str, Any]]]:
    outer_train = _indices(data, _outer_train_folds(outer_fold))
    baseline_metrics = compute_iterative_tail_metrics(data.targets[outer_train], data.base[outer_train])
    specs = candidate_specs(device=device)
    metrics_by_id: dict[str, Mapping[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for spec in specs:
        oof = _candidate_inner_oof(data, features, outer_fold, spec)
        metrics = compute_iterative_tail_metrics(data.targets[outer_train], oof.predictions)
        metrics_by_id[spec.variant_id] = metrics
        records.append({
            "cycle": spec.cycle,
            "variant_id": spec.variant_id,
            "family": spec.family,
            "registered_parameters": dict(spec.parameters),
            "metrics": metrics,
            "inner_fold_diagnostics": oof.audit,
        })
        if progress_callback is not None:
            progress_callback(len(records), spec)
    if len(records) != 3 or set(metrics_by_id) != {spec.variant_id for spec in specs}:
        raise IterativeTailDirectionalRunError("all three candidates must finish before selection")
    selection = select_candidate(specs, metrics_by_id, baseline_metrics, protocol.raw["inner_promotion_gate"])
    if not selection["inventory_valid"] or len(selection["decisions"]) != 3:
        raise IterativeTailDirectionalRunError("directional candidate selection barrier failed")
    decisions = {item["variant_id"]: item for item in selection["decisions"]}
    for record in records:
        record["baseline_relative_decision"] = decisions[record["variant_id"]]
    if selection["fell_back_to_baseline"]:
        return None, records
    selected_id = selection["selected_id"]
    return next(spec for spec in specs if spec.variant_id == selected_id), records


def _fresh_outer_prediction(
    data: ExperimentData,
    features: np.ndarray,
    outer_fold: int,
    selected_spec: DirectionalSpec,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    train = _indices(data, _outer_train_folds(outer_fold))
    predict = _indices(data, (outer_fold,))
    fitted = fit(selected_spec, features[train], data.base[train], data.targets[train], data.folds[train])
    result = apply(fitted, features[predict], data.base[predict])
    if result.predictions.shape != (400, 3) or not np.isfinite(result.predictions).all():
        raise IterativeTailDirectionalRunError("fresh outer prediction coverage differs")
    return result.predictions, {
        "selection_frozen_before_refit": True,
        "selected_candidate_only": True,
        "train_folds": list(_outer_train_folds(outer_fold)),
        "predict_fold": outer_fold,
        "train_count": len(train),
        "predict_count": len(predict),
        "fit": _json_safe(fitted.audit),
        "predict": _json_safe(result.audit),
    }


def run_outer_fold(
    outer_fold: int,
    *,
    device: str,
    protocol: DirectionalProtocol | None = None,
) -> Mapping[str, Any]:
    protocol = protocol or load_protocol()
    validate_bound_inputs(protocol)
    validate_model_inventory(require_available=True)
    if outer_fold not in protocol.raw["nested_protocol"]["outer_folds"]:
        raise IterativeTailDirectionalRunError("outer fold is not registered")
    data = load_experiment_data()
    features = _directional_features(data, protocol)
    outer_indices = _indices(data, (outer_fold,))
    progress_path = PUBLIC_ROOT / f"outer-{outer_fold}" / "progress.json"

    def report_progress(completed: int, spec: DirectionalSpec) -> None:
        _write_json(progress_path, {
            "schema_version": "mal2026-iterative-tail-directional-progress-v6",
            "status": "running",
            "outer_fold": outer_fold,
            "completed_candidates": completed,
            "candidate_count": 3,
            "completed_inner_predictions": completed * 4,
            "inner_prediction_count": 12,
            "percent": 100.0 * completed / 3.0,
            "last_completed_cycle": spec.cycle,
            "last_completed_candidate": spec.variant_id,
            "validation_loaded": False,
            "average_target_used": False,
        })

    _write_json(progress_path, {
        "schema_version": "mal2026-iterative-tail-directional-progress-v6",
        "status": "running",
        "outer_fold": outer_fold,
        "completed_candidates": 0,
        "candidate_count": 3,
        "completed_inner_predictions": 0,
        "inner_prediction_count": 12,
        "percent": 0.0,
        "last_completed_cycle": None,
        "last_completed_candidate": None,
        "validation_loaded": False,
        "average_target_used": False,
    })
    selected_spec, candidate_records = _fit_and_select_candidates(
        protocol, data, features, outer_fold, device=device, progress_callback=report_progress,
    )
    if selected_spec is None:
        prediction = data.base[outer_indices].copy()
        outer_refit_audit: Mapping[str, Any] = {"status": "baseline_fallback", "selection_frozen": True}
        selected_name, selected_cycle = "exact-r0-oof-baseline", None
    else:
        prediction, outer_refit_audit = _fresh_outer_prediction(data, features, outer_fold, selected_spec)
        selected_name, selected_cycle = selected_spec.variant_id, selected_spec.cycle
    # Outer gold is first scored only after the selected prediction has frozen.
    baseline_metrics = compute_iterative_tail_metrics(data.targets[outer_indices], data.base[outer_indices])
    selected_metrics = compute_iterative_tail_metrics(data.targets[outer_indices], prediction)
    restricted_path = RESTRICTED_ROOT / f"outer-{outer_fold}" / "predictions.jsonl"
    _write_jsonl(restricted_path, [{
        "source_id": data.source_ids[index],
        "outer_fold": outer_fold,
        "baseline_prediction": data.base[index],
        "selected_prediction": row,
    } for index, row in zip(outer_indices, prediction, strict=True)])
    payload = {
        "schema_version": "mal2026-iterative-tail-directional-outer-v6",
        "status": "completed",
        "outer_fold": outer_fold,
        "outer_train_count": 1600,
        "outer_holdout_count": 400,
        "inner_fold_count": 4,
        "candidate_count": 3,
        "selected_candidate": selected_name,
        "selected_cycle": selected_cycle,
        "fell_back_to_baseline": selected_spec is None,
        "candidate_records": candidate_records,
        "outer_refit_audit": outer_refit_audit,
        "baseline_metrics": baseline_metrics,
        "selected_metrics": selected_metrics,
        "projection": projection_audit(protocol),
        "restricted_prediction_sha256": _sha256(restricted_path),
        "outer_gold_used_before_selection_freeze_or_prediction": False,
        "historical_predictions_errors_weights_checkpoints_or_pseudo_targets_used": False,
        "validation_loaded": False,
        "average_target_used": False,
        "external_api_calls": 0,
    }
    _write_json(PUBLIC_ROOT / f"outer-{outer_fold}" / "result.json", payload)
    _write_json(progress_path, {
        "schema_version": "mal2026-iterative-tail-directional-progress-v6",
        "status": "completed",
        "outer_fold": outer_fold,
        "completed_candidates": 3,
        "candidate_count": 3,
        "completed_inner_predictions": 12,
        "inner_prediction_count": 12,
        "percent": 100.0,
        "last_completed_cycle": 3,
        "last_completed_candidate": candidate_records[-1]["variant_id"],
        "selected_candidate": selected_name,
        "validation_loaded": False,
        "average_target_used": False,
    })
    return payload


def _read_outer_predictions(data: ExperimentData) -> tuple[np.ndarray, np.ndarray, list[Mapping[str, Any]]]:
    baseline = np.full_like(data.base, np.nan, dtype=np.float64)
    selected = np.full_like(data.base, np.nan, dtype=np.float64)
    seen: set[str] = set()
    audits: list[Mapping[str, Any]] = []
    id_to_index = {source_id: index for index, source_id in enumerate(data.source_ids)}
    required = {"source_id", "outer_fold", "baseline_prediction", "selected_prediction"}
    for outer_fold in range(5):
        result_path = PUBLIC_ROOT / f"outer-{outer_fold}" / "result.json"
        prediction_path = RESTRICTED_ROOT / f"outer-{outer_fold}" / "predictions.jsonl"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "completed"
            or result.get("candidate_count") != 3
            or result.get("restricted_prediction_sha256") != _sha256(prediction_path)
        ):
            raise IterativeTailDirectionalRunError("outer result/prediction binding differs")
        with prediction_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if set(row) != required:
                    raise IterativeTailDirectionalRunError("restricted outer schema differs")
                source_id = row["source_id"]
                if source_id in seen or source_id not in id_to_index or row["outer_fold"] != outer_fold:
                    raise IterativeTailDirectionalRunError("restricted outer population differs")
                index = id_to_index[source_id]
                if int(data.folds[index]) != outer_fold:
                    raise IterativeTailDirectionalRunError("restricted outer fold assignment differs")
                baseline[index] = row["baseline_prediction"]
                selected[index] = row["selected_prediction"]
                seen.add(source_id)
        audits.append({
            "outer_fold": outer_fold,
            "selected_candidate": result["selected_candidate"],
            "selected_cycle": result["selected_cycle"],
            "fell_back_to_baseline": result["fell_back_to_baseline"],
            "result_sha256": _sha256(result_path),
            "prediction_sha256": _sha256(prediction_path),
        })
    if len(seen) != 2000 or not np.isfinite(baseline).all() or not np.isfinite(selected).all():
        raise IterativeTailDirectionalRunError("five-fold outer prediction coverage differs")
    return baseline, selected, audits


def aggregate_outer_results(protocol: DirectionalProtocol | None = None) -> Mapping[str, Any]:
    """Concatenate five outer predictions exactly once without posthoc selection."""
    protocol = protocol or load_protocol()
    validate_bound_inputs(protocol)
    data = load_experiment_data()
    baseline, selected, audits = _read_outer_predictions(data)
    if not np.allclose(baseline, data.base.astype(np.float64), rtol=0.0, atol=1e-7):
        raise IterativeTailDirectionalRunError("nested baseline differs from exact R0 OOF")
    baseline_metrics = compute_iterative_tail_metrics(data.targets, baseline)
    selected_metrics = compute_iterative_tail_metrics(data.targets, selected)
    per_fold_baseline, per_fold_selected = [], []
    for outer_fold in range(5):
        index = np.flatnonzero(data.folds == outer_fold)
        per_fold_baseline.append(compute_iterative_tail_metrics(data.targets[index], baseline[index]))
        per_fold_selected.append(compute_iterative_tail_metrics(data.targets[index], selected[index]))
    directions = fold_diagnostics(per_fold_baseline, per_fold_selected)
    final_config = protocol.raw["final_evaluation"]
    bootstrap = _bootstrap_macro_rmse(
        data.targets,
        baseline,
        selected,
        resamples=int(final_config["paired_bootstrap"]["replicates"]),
        seed=SEED,
    )
    decision = final_gate(final_config, baseline_metrics, selected_metrics, bootstrap)
    final_pass = bool(decision["pass"])
    terminal_freeze = not final_pass and bool(protocol.raw["failure_action"]["terminal_freeze_same_train_and_feature_sources"])
    payload = {
        "schema_version": "mal2026-iterative-tail-directional-aggregate-v6",
        "status": "completed",
        "record_count": 2000,
        "outer_fold_count": 5,
        "inner_fold_count_per_outer": 4,
        "candidate_count_per_outer": 3,
        "outer_audits": audits,
        "baseline_metrics": baseline_metrics,
        "nested_selected_metrics": selected_metrics,
        "fold_direction_diagnostics": directions,
        "paired_bootstrap": bootstrap,
        "final_decision": decision,
        "final_gate_pass": final_pass,
        "final_selection": "nested-v6-directional" if final_pass else "exact-r0-oof-baseline-fallback",
        "v6_inventory_frozen": True,
        "terminal_freeze_same_train_and_feature_sources": terminal_freeze,
        "adaptive_same_train_descriptive_and_falsification_evidence": True,
        "independent_confirmation_or_generalization_claim": False,
        "posthoc_selection_on_concatenated_outer_predictions": False,
        "historical_predictions_errors_weights_checkpoints_or_pseudo_targets_used": False,
        "validation_loaded": False,
        "average_target_used": False,
        "external_api_calls": 0,
    }
    aggregate_path = PUBLIC_ROOT / "aggregate.json"
    _write_json(aggregate_path, payload)
    _write_json(PUBLIC_ROOT / "completion.json", {
        "schema_version": "mal2026-iterative-tail-directional-completion-v6",
        "status": "completed_final_gate_pass" if final_pass else "completed_no_promotion_terminal_same_train_feature_freeze",
        "aggregate_sha256": _sha256(aggregate_path),
        "final_gate_pass": final_pass,
        "final_selection": payload["final_selection"],
        "v6_inventory_frozen": True,
        "terminal_freeze_same_train_and_feature_sources": terminal_freeze,
        "gpu_scope": [0, 1, 2, 3],
        "external_api_calls": 0,
        "validation_loaded": False,
        "average_target_used": False,
    })
    return payload


def gpu0_smoke(*, device: str = "cuda:0") -> Mapping[str, Any]:
    """Small real three-candidate integration gate on physical GPU0."""
    protocol = load_protocol()
    validate_bound_inputs(protocol)
    validate_model_inventory(require_available=True)
    data = load_experiment_data()
    features = _directional_features(data, protocol)
    train = np.concatenate([_indices(data, (fold,))[:32] for fold in (0, 1, 2)])
    predict = _indices(data, (3,))[:32]
    candidates = []
    for spec in candidate_specs(device=device):
        fitted = fit(spec, features[train], data.base[train], data.targets[train], data.folds[train])
        result = apply(fitted, features[predict], data.base[predict])
        if result.predictions.shape != (32, 3) or not np.isfinite(result.predictions).all():
            raise IterativeTailDirectionalRunError("directional candidate smoke coverage differs")
        candidates.append({
            "cycle": spec.cycle,
            "variant_id": spec.variant_id,
            "initial_state_hash": fitted.initial_state_hash,
            "final_state_hash": fitted.final_state_hash,
            "mean_identity_weight": result.audit["mean_identity_weight"],
        })
    payload = {
        "schema_version": "mal2026-iterative-tail-directional-smoke-v6",
        "status": "completed",
        "gpu": 0,
        "train_count": len(train),
        "train_fold_count": 3,
        "predict_count": len(predict),
        "projection": projection_audit(protocol),
        "candidates": candidates,
        "validation_loaded": False,
        "average_target_used": False,
    }
    _write_json(PUBLIC_ROOT / "smoke.json", payload)
    return payload


__all__ = [
    "PUBLIC_ROOT",
    "RESTRICTED_ROOT",
    "IterativeTailDirectionalRunError",
    "aggregate_outer_results",
    "gpu0_smoke",
    "projection_audit",
    "run_outer_fold",
]
