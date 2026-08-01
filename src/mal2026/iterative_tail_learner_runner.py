"""Leakage-sealed candidate-level nested execution for V5 learners."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mal2026.iterative_tail_learner_models import (
    LearnerResult,
    LearnerSpec,
    apply,
    candidate_specs,
    fit,
)
from mal2026.iterative_tail_learner_protocol import (
    RUN_ID,
    LearnerProtocol,
    load_protocol,
    validate_bound_inputs,
    validate_model_inventory,
)
from mal2026.iterative_tail_metrics import compute_iterative_tail_metrics, metric_improvements
from mal2026.iterative_tail_learner_selection import fold_direction_diagnostics
from mal2026.iterative_tail_remediation_runner import _bootstrap_macro_rmse, _indices
from mal2026.iterative_tail_runner import ExperimentData, load_experiment_data


PUBLIC_ROOT = Path("outputs/iterative-tail-learner-v5") / RUN_ID
RESTRICTED_ROOT = Path("data/processed/restricted/iterative_tail_learner_v5") / RUN_ID
SEED = 2026080205


class IterativeTailLearnerRunError(RuntimeError):
    """Raised when V5 inventory, nesting, coverage, or privacy differs."""


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
        raise IterativeTailLearnerRunError("outer fold must be 0..4")
    return tuple(fold for fold in range(5) if fold != outer_fold)


def _learner_features(data: ExperimentData) -> np.ndarray:
    evidence = data.evidence.view("evidence_hash")
    if (evidence is None or evidence.ndim != 2 or len(evidence) != len(data.embeddings)
            or evidence.shape[1] != 576):
        raise IterativeTailLearnerRunError("score-blind evidence_hash mean/std view differs")
    features = np.concatenate((data.embeddings.astype(np.float32), evidence.astype(np.float32)), axis=1)
    if not np.isfinite(features).all():
        raise IterativeTailLearnerRunError("learner features must be finite")
    return features


def gate_decision(
    config: Mapping[str, Any], baseline_metrics: Mapping[str, Any], candidate_metrics: Mapping[str, Any],
    *, final: bool = False, bootstrap: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Apply exact config gates with every delta oriented positive-is-better."""
    delta = metric_improvements(baseline_metrics, candidate_metrics)
    values = (delta["rmse"], delta["equal_group_rmse"], delta["low_tail_rmse"],
              delta["high_tail_rmse"], delta["gold_3_4_balanced_accuracy"], delta["spearman"],
              *delta["axis_rmse"].values())
    finite = all(value is not None and math.isfinite(float(value)) for value in values)
    gates: dict[str, bool] = {
        "macro_rmse_improvement": finite and float(delta["rmse"]) >= float(config["macro_rmse_min_improvement"]),
        "low_tail_improves": finite and (not config["low_tail_must_improve"] or float(delta["low_tail_rmse"]) > 0),
        "high_tail_improves": finite and (not config["high_tail_must_improve"] or float(delta["high_tail_rmse"]) > 0),
        "gold_3_4_balanced_accuracy_improvement": finite and float(delta["gold_3_4_balanced_accuracy"]) >= float(config["gold_3_4_balanced_accuracy_min_improvement"]),
        "axis_rmse_worsening_bound": finite and all(float(value) >= -float(config["max_axis_rmse_worsening"]) for value in delta["axis_rmse"].values()),
        "macro_spearman_fall_bound": finite and float(delta["spearman"]) >= -float(config["max_macro_spearman_fall"]),
    }
    if final:
        if bootstrap is None:
            raise IterativeTailLearnerRunError("final gate requires paired bootstrap evidence")
        gates["candidate_minus_baseline_rmse_ci_upper_below_bound"] = (
            float(bootstrap["candidate_minus_baseline_ci"]["upper"])
            < float(config["paired_bootstrap"]["required_upper_bound_lt"])
        )
    else:
        gates["equal_group_rmse_improvement"] = finite and float(delta["equal_group_rmse"]) >= float(config["equal_group_rmse_min_improvement"])
    if config.get("require_finite_metrics") and not finite:
        gates = {name: False for name in gates}
    return {
        "pass" if final else "eligible": config["operator"] == "AND" and all(gates.values()),
        "gates": gates, "improvements": delta, "finite_metrics": finite,
        "score1_used_for_promotion": False,
    }


def _candidate_inner_oof(
    data: ExperimentData, features: np.ndarray, outer_fold: int, spec: LearnerSpec,
) -> CandidateOOF:
    """Fresh-fit one candidate S->D four times without exposing outer O."""
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
            raise IterativeTailLearnerRunError("inner S/D counts differ from 1200/400")
        if np.any(np.isin(data.folds[train], (outer_fold, inner_validation))):
            raise IterativeTailLearnerRunError("D or O entered candidate fitting")
        fitted = fit(spec, features[train], data.base[train], data.targets[train])
        result: LearnerResult = apply(fitted, features[predict], data.base[predict])
        if result.predictions.shape != (400, 3) or not np.isfinite(result.predictions).all():
            raise IterativeTailLearnerRunError("inner candidate prediction coverage differs")
        prediction[predict] = result.predictions
        seen[predict] += 1
        audits.append({
            "outer_fold": outer_fold, "inner_validation_fold": inner_validation,
            "train_folds": list(train_folds), "forbidden_folds": [inner_validation, outer_fold],
            "train_count": len(train), "predict_count": len(predict),
            "fit": _json_safe(fitted.audit), "predict": _json_safe(result.audit),
        })
    outer_holdout = _indices(data, (outer_fold,))
    if not np.all(seen[outer_train] == 1) or np.any(seen[outer_holdout]):
        raise IterativeTailLearnerRunError("candidate OOF is not exactly-once and outer-sealed")
    if not np.isfinite(prediction[outer_train]).all() or np.isfinite(prediction[outer_holdout]).any():
        raise IterativeTailLearnerRunError("candidate OOF population differs")
    return CandidateOOF(prediction[outer_train], tuple(audits))


def _fit_and_select_candidates(
    protocol: LearnerProtocol, data: ExperimentData, features: np.ndarray, outer_fold: int,
    *, device: str, progress_callback: Callable[[int, LearnerSpec], None] | None = None,
) -> tuple[LearnerSpec | None, list[Mapping[str, Any]]]:
    outer_train = _indices(data, _outer_train_folds(outer_fold))
    baseline_metrics = compute_iterative_tail_metrics(data.targets[outer_train], data.base[outer_train])
    eligible: list[tuple[float, int, LearnerSpec]] = []
    records: list[Mapping[str, Any]] = []
    for spec in candidate_specs(device=device):
        oof = _candidate_inner_oof(data, features, outer_fold, spec)
        metrics = compute_iterative_tail_metrics(data.targets[outer_train], oof.predictions)
        decision = gate_decision(protocol.raw["inner_promotion_gate"], baseline_metrics, metrics)
        records.append({
            "cycle": spec.cycle, "variant_id": spec.variant_id, "family": spec.family,
            "registered_parameters": dict(spec.parameters), "metrics": metrics,
            "baseline_relative_decision": decision, "inner_fold_diagnostics": oof.audit,
        })
        if progress_callback is not None:
            progress_callback(len(records), spec)
        if decision["eligible"]:
            eligible.append((float(metrics["macro"]["rmse"]), spec.cycle, spec))
    if len(records) != 20:
        raise IterativeTailLearnerRunError("all twenty candidates must finish before selection")
    if not eligible:
        return None, records
    return min(eligible, key=lambda item: (item[0], item[1]))[2], records


def _fresh_outer_prediction(
    data: ExperimentData, features: np.ndarray, outer_fold: int, selected_spec: LearnerSpec,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    train = _indices(data, _outer_train_folds(outer_fold))
    predict = _indices(data, (outer_fold,))
    fitted = fit(selected_spec, features[train], data.base[train], data.targets[train])
    result = apply(fitted, features[predict], data.base[predict])
    if result.predictions.shape != (400, 3) or not np.isfinite(result.predictions).all():
        raise IterativeTailLearnerRunError("fresh outer prediction coverage differs")
    return result.predictions, {
        "selection_frozen_before_refit": True, "selected_candidate_only": True,
        "train_folds": list(_outer_train_folds(outer_fold)), "predict_fold": outer_fold,
        "train_count": len(train), "predict_count": len(predict),
        "fit": _json_safe(fitted.audit), "predict": _json_safe(result.audit),
    }


def run_outer_fold(outer_fold: int, *, device: str, protocol: LearnerProtocol | None = None) -> Mapping[str, Any]:
    protocol = protocol or load_protocol()
    validate_bound_inputs(protocol)
    validate_model_inventory(require_available=True)
    if outer_fold not in protocol.raw["nested_protocol"]["outer_folds"]:
        raise IterativeTailLearnerRunError("outer fold is not registered")
    data = load_experiment_data()
    features = _learner_features(data)
    outer_indices = _indices(data, (outer_fold,))
    progress_path = PUBLIC_ROOT / f"outer-{outer_fold}" / "progress.json"

    def report_progress(completed: int, spec: LearnerSpec) -> None:
        _write_json(progress_path, {
            "schema_version": "mal2026-iterative-tail-learner-progress-v5",
            "status": "running", "outer_fold": outer_fold,
            "completed_candidates": completed, "candidate_count": 20,
            "completed_inner_fits": completed * 4, "inner_fit_count": 80,
            "percent": 100.0 * completed / 20.0,
            "last_completed_cycle": spec.cycle,
            "last_completed_candidate": spec.variant_id,
            "validation_loaded": False, "average_target_used": False,
        })

    _write_json(progress_path, {
        "schema_version": "mal2026-iterative-tail-learner-progress-v5",
        "status": "running", "outer_fold": outer_fold,
        "completed_candidates": 0, "candidate_count": 20,
        "completed_inner_fits": 0, "inner_fit_count": 80, "percent": 0.0,
        "last_completed_cycle": None, "last_completed_candidate": None,
        "validation_loaded": False, "average_target_used": False,
    })
    selected_spec, candidate_records = _fit_and_select_candidates(
        protocol, data, features, outer_fold, device=device, progress_callback=report_progress,
    )
    # Candidate identity is immutable after the complete twenty-candidate selection above.
    if selected_spec is None:
        prediction = data.base[outer_indices].copy()
        outer_refit_audit: Mapping[str, Any] = {"status": "baseline_fallback", "selection_frozen": True}
        selected_name, selected_cycle = "exact-r0-oof-baseline", None
    else:
        prediction, outer_refit_audit = _fresh_outer_prediction(data, features, outer_fold, selected_spec)
        selected_name, selected_cycle = selected_spec.variant_id, selected_spec.cycle
    baseline_metrics = compute_iterative_tail_metrics(data.targets[outer_indices], data.base[outer_indices])
    selected_metrics = compute_iterative_tail_metrics(data.targets[outer_indices], prediction)
    restricted_path = RESTRICTED_ROOT / f"outer-{outer_fold}" / "predictions.jsonl"
    _write_jsonl(restricted_path, [{
        "source_id": data.source_ids[index], "outer_fold": outer_fold,
        "baseline_prediction": data.base[index], "selected_prediction": row,
    } for index, row in zip(outer_indices, prediction, strict=True)])
    payload = {
        "schema_version": "mal2026-iterative-tail-learner-outer-v5", "status": "completed",
        "outer_fold": outer_fold, "outer_train_count": 1600, "outer_holdout_count": 400,
        "inner_fold_count": 4, "candidate_count": 20, "selected_candidate": selected_name,
        "selected_cycle": selected_cycle, "fell_back_to_baseline": selected_spec is None,
        "candidate_records": candidate_records, "outer_refit_audit": outer_refit_audit,
        "baseline_metrics": baseline_metrics, "selected_metrics": selected_metrics,
        "restricted_prediction_sha256": _sha256(restricted_path),
        "outer_gold_used_before_selection_freeze_or_prediction": False,
        "historical_predictions_weights_or_pseudo_targets_used": False,
        "validation_loaded": False, "average_target_used": False, "external_api_calls": 0,
    }
    _write_json(PUBLIC_ROOT / f"outer-{outer_fold}" / "result.json", payload)
    _write_json(progress_path, {
        "schema_version": "mal2026-iterative-tail-learner-progress-v5",
        "status": "completed", "outer_fold": outer_fold,
        "completed_candidates": 20, "candidate_count": 20,
        "completed_inner_fits": 80, "inner_fit_count": 80, "percent": 100.0,
        "last_completed_cycle": 20,
        "last_completed_candidate": candidate_records[-1].get("variant_id", "completed-inventory"),
        "selected_candidate": selected_name,
        "validation_loaded": False, "average_target_used": False,
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
        if (result.get("status") != "completed" or result.get("candidate_count") != 20
                or result.get("restricted_prediction_sha256") != _sha256(prediction_path)):
            raise IterativeTailLearnerRunError("outer result/prediction binding differs")
        with prediction_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if set(row) != required:
                    raise IterativeTailLearnerRunError("restricted outer schema differs")
                source_id = row["source_id"]
                if source_id in seen or source_id not in id_to_index or row["outer_fold"] != outer_fold:
                    raise IterativeTailLearnerRunError("restricted outer population differs")
                index = id_to_index[source_id]
                if int(data.folds[index]) != outer_fold:
                    raise IterativeTailLearnerRunError("restricted outer fold assignment differs")
                baseline[index], selected[index] = row["baseline_prediction"], row["selected_prediction"]
                seen.add(source_id)
        audits.append({
            "outer_fold": outer_fold, "selected_candidate": result["selected_candidate"],
            "selected_cycle": result["selected_cycle"], "fell_back_to_baseline": result["fell_back_to_baseline"],
            "result_sha256": _sha256(result_path), "prediction_sha256": _sha256(prediction_path),
        })
    if len(seen) != 2000 or not np.isfinite(baseline).all() or not np.isfinite(selected).all():
        raise IterativeTailLearnerRunError("five-fold outer prediction coverage differs")
    return baseline, selected, audits


def aggregate_outer_results(protocol: LearnerProtocol | None = None) -> Mapping[str, Any]:
    """Concatenate five outer predictions exactly once without posthoc selection."""
    protocol = protocol or load_protocol()
    validate_bound_inputs(protocol)
    data = load_experiment_data()
    baseline, selected, audits = _read_outer_predictions(data)
    if not np.allclose(baseline, data.base.astype(np.float64), rtol=0, atol=1e-7):
        raise IterativeTailLearnerRunError("nested baseline differs from exact R0 OOF")
    baseline_metrics = compute_iterative_tail_metrics(data.targets, baseline)
    selected_metrics = compute_iterative_tail_metrics(data.targets, selected)
    per_fold_baseline = []
    per_fold_selected = []
    for outer_fold in range(5):
        fold_indices = np.flatnonzero(data.folds == outer_fold)
        per_fold_baseline.append(compute_iterative_tail_metrics(data.targets[fold_indices], baseline[fold_indices]))
        per_fold_selected.append(compute_iterative_tail_metrics(data.targets[fold_indices], selected[fold_indices]))
    fold_directions = fold_direction_diagnostics(per_fold_baseline, per_fold_selected)
    final_config = protocol.raw["final_evaluation"]
    bootstrap = _bootstrap_macro_rmse(
        data.targets, baseline, selected,
        resamples=int(final_config["paired_bootstrap"]["replicates"]), seed=SEED,
    )
    decision = gate_decision(final_config, baseline_metrics, selected_metrics, final=True, bootstrap=bootstrap)
    final_pass = bool(decision["pass"])
    payload = {
        "schema_version": "mal2026-iterative-tail-learner-aggregate-v5", "status": "completed",
        "record_count": 2000, "outer_fold_count": 5, "inner_fold_count_per_outer": 4,
        "candidate_count_per_outer": 20, "outer_audits": audits,
        "baseline_metrics": baseline_metrics, "nested_selected_metrics": selected_metrics,
        "fold_direction_diagnostics": fold_directions,
        "paired_bootstrap": bootstrap, "final_decision": decision, "final_gate_pass": final_pass,
        "final_selection": "nested-v5-learner" if final_pass else "exact-r0-oof-baseline-fallback",
        "v5_inventory_frozen": True, "adaptive_same_train_descriptive_evidence": True,
        "independent_confirmation_or_generalization_claim": False,
        "posthoc_selection_on_concatenated_outer_predictions": False,
        "historical_predictions_weights_or_pseudo_targets_used": False,
        "validation_loaded": False, "average_target_used": False, "external_api_calls": 0,
    }
    aggregate_path = PUBLIC_ROOT / "aggregate.json"
    _write_json(aggregate_path, payload)
    _write_json(PUBLIC_ROOT / "completion.json", {
        "schema_version": "mal2026-iterative-tail-learner-completion-v5",
        "status": "completed_final_gate_pass" if final_pass else "completed_no_promotion_v5_inventory_frozen",
        "aggregate_sha256": _sha256(aggregate_path), "final_gate_pass": final_pass,
        "final_selection": payload["final_selection"], "v5_inventory_frozen": True,
        "gpu_scope": [0, 1, 2, 3], "external_api_calls": 0,
        "validation_loaded": False, "average_target_used": False,
    })
    return payload


def gpu0_smoke(*, device: str = "cuda:0") -> Mapping[str, Any]:
    """Smallest real V5 feature/learner integration gate on physical GPU0."""
    protocol = load_protocol()
    validate_bound_inputs(protocol)
    validate_model_inventory(require_available=True)
    data = load_experiment_data()
    features = _learner_features(data)
    train = _indices(data, (1,))[:128]
    predict = _indices(data, (2,))[:32]
    families = []
    for registered in candidate_specs(device=device)[::4]:
        fitted = fit(registered, features[train], data.base[train], data.targets[train])
        result = apply(fitted, features[predict], data.base[predict])
        if result.predictions.shape != (32, 3) or not np.isfinite(result.predictions).all():
            raise IterativeTailLearnerRunError("learner family smoke coverage differs")
        families.append({
            "cycle": registered.cycle, "family": registered.family,
            "initial_state_hash": fitted.initial_state_hash, "final_state_hash": fitted.final_state_hash,
        })
    payload = {
        "schema_version": "mal2026-iterative-tail-learner-smoke-v5", "status": "completed",
        "gpu": 0, "train_count": len(train), "predict_count": len(predict),
        "feature_dimensions": features.shape[1], "family_smokes": families,
        "validation_loaded": False, "average_target_used": False,
    }
    _write_json(PUBLIC_ROOT / "smoke.json", payload)
    return payload


__all__ = [
    "PUBLIC_ROOT", "RESTRICTED_ROOT", "IterativeTailLearnerRunError",
    "aggregate_outer_results", "gate_decision", "gpu0_smoke", "run_outer_fold",
]
