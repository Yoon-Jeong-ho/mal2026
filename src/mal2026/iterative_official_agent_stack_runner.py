"""Sealed 5x4 nested runner for the V7 official Terra score stack."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .iterative_official_agent_stack_models import (
    AgentStackSpec,
    build_agent_score_features,
    candidate_specs,
    fit_predict_agent_stack,
)
from .iterative_official_agent_stack_protocol import RUN_ID, load_protocol, validate_bound_inputs
from .iterative_official_agent_stack_selection import final_gate, fold_direction_diagnostics, select_candidate
from .iterative_tail_metrics import compute_iterative_tail_metrics
from .iterative_tail_remediation_runner import _bootstrap_macro_rmse
from .iterative_tail_runner import ExperimentData, load_experiment_data
from .official_rationale_data import candidate_provenance, load_candidates


PUBLIC_ROOT = Path("outputs/iterative-official-agent-stack-v7") / RUN_ID
RESTRICTED_ROOT = Path("data/processed/restricted/iterative_official_agent_stack_v7") / RUN_ID
SEED = 2026080207


class OfficialAgentStackRunError(RuntimeError):
    """Raised when V7 nesting, coverage, selection, or privacy differs."""


@dataclass(frozen=True)
class AgentStackData:
    experiment: ExperimentData
    features: np.ndarray
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


def load_agent_stack_data() -> AgentStackData:
    protocol = load_protocol()
    validate_bound_inputs(protocol)
    experiment = load_experiment_data()
    candidates = load_candidates()
    features, audit = build_agent_score_features(experiment.base, experiment.source_ids, candidates)
    provenance = candidate_provenance()
    if audit["records"] != 2000 or provenance["candidates"] != 6000 or provenance["human_or_reference_score_read_or_prompted"] is not False:
        raise OfficialAgentStackRunError("official feature provenance differs")
    return AgentStackData(experiment, features, audit, provenance)


def _candidate_oof(
    bundle: AgentStackData,
    spec: AgentStackSpec,
    outer_fold: int,
    *,
    device: str,
) -> tuple[np.ndarray, list[Mapping[str, Any]]]:
    data, features = bundle.experiment, bundle.features
    universe = tuple(fold for fold in range(5) if fold != outer_fold)
    prediction = np.full_like(data.targets, np.nan, dtype=np.float64)
    audit: list[Mapping[str, Any]] = []
    for inner_validation in universe:
        train_folds = tuple(fold for fold in universe if fold != inner_validation)
        train, predict = _indices(data, train_folds), _indices(data, (inner_validation,))
        if outer_fold in data.folds[train] or outer_fold in data.folds[predict] or inner_validation in data.folds[train]:
            raise OfficialAgentStackRunError("V7 inner/outer fold sealing differs")
        result = fit_predict_agent_stack(
            spec, features[train], data.base[train], data.targets[train],
            features[predict], data.base[predict], device=device,
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
        raise OfficialAgentStackRunError("V7 candidate OOF coverage differs")
    return prediction, audit


def _outer_refit(
    bundle: AgentStackData,
    spec: AgentStackSpec,
    outer_fold: int,
    *,
    device: str,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    data, features = bundle.experiment, bundle.features
    train_folds = tuple(fold for fold in range(5) if fold != outer_fold)
    train, predict = _indices(data, train_folds), _indices(data, (outer_fold,))
    result = fit_predict_agent_stack(
        spec, features[train], data.base[train], data.targets[train],
        features[predict], data.base[predict], device=device,
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
        raise OfficialAgentStackRunError("outer fold must be 0..4")
    protocol = load_protocol()
    validate_bound_inputs(protocol)
    bundle = load_agent_stack_data()
    data = bundle.experiment
    specs = candidate_specs()
    universe = tuple(fold for fold in range(5) if fold != outer_fold)
    outer_train, outer = _indices(data, universe), _indices(data, (outer_fold,))
    baseline_inner = compute_iterative_tail_metrics(data.targets[outer_train], data.base[outer_train])
    records, matrices, metrics_by_id = [], {}, {}
    progress_path = PUBLIC_ROOT / f"outer-{outer_fold}" / "progress.json"
    for position, spec in enumerate(specs, start=1):
        prediction, audit = _candidate_oof(bundle, spec, outer_fold, device=device)
        metrics = compute_iterative_tail_metrics(data.targets[outer_train], prediction[outer_train])
        matrices[spec.variant_id], metrics_by_id[spec.variant_id] = prediction, metrics
        records.append({
            "cycle": spec.cycle,
            "variant_id": spec.variant_id,
            "ridge_alpha": spec.ridge_alpha,
            "max_correction": spec.max_correction,
            "inner_metrics": metrics,
            "inner_fold_audit": audit,
        })
        _write_json(progress_path, {
            "schema_version": "mal2026-iterative-official-agent-stack-progress-v7",
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
        "schema_version": "mal2026-iterative-official-agent-stack-outer-v7",
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
        "schema_version": "mal2026-iterative-official-agent-stack-progress-v7",
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
            raise OfficialAgentStackRunError("V7 outer artifact missing")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed" or result.get("candidate_count") != 3 or result.get("restricted_prediction_sha256") != _sha256(prediction_path):
            raise OfficialAgentStackRunError("V7 outer artifact binding differs")
        count = 0
        with prediction_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                source_id = row.get("source_id")
                if source_id not in index or source_id in seen or row.get("outer_fold") != outer_fold:
                    raise OfficialAgentStackRunError("V7 outer row identity differs")
                position = index[source_id]
                if int(data.folds[position]) != outer_fold:
                    raise OfficialAgentStackRunError("V7 outer row fold differs")
                base_vector = np.asarray(row.get("baseline"), dtype=np.float64)
                selected_vector = np.asarray(row.get("selected"), dtype=np.float64)
                if base_vector.shape != (3,) or selected_vector.shape != (3,) or not np.isfinite(base_vector).all() or not np.isfinite(selected_vector).all():
                    raise OfficialAgentStackRunError("V7 outer prediction vector differs")
                baseline[position], selected[position] = base_vector, selected_vector
                seen.add(source_id); count += 1
        if count != 400:
            raise OfficialAgentStackRunError("V7 outer row count differs")
        audits.append({
            "outer_fold": outer_fold,
            "selected_candidate": result["selected_candidate"],
            "selected_cycle": result["selected_cycle"],
            "fell_back_to_baseline": result["fell_back_to_baseline"],
            "result_sha256": _sha256(result_path),
            "prediction_sha256": _sha256(prediction_path),
        })
    if len(seen) != 2000 or not np.isfinite(baseline).all() or not np.isfinite(selected).all():
        raise OfficialAgentStackRunError("V7 concatenated outer coverage differs")
    return baseline, selected, audits


def aggregate_outer_results() -> Mapping[str, Any]:
    protocol = load_protocol()
    validate_bound_inputs(protocol)
    bundle = load_agent_stack_data()
    data = bundle.experiment
    baseline, nested_selected, audits = _read_outer_predictions(data)
    if not np.allclose(baseline, data.base.astype(np.float64), rtol=0.0, atol=1e-7):
        raise OfficialAgentStackRunError("V7 nested baseline differs from exact R0")
    baseline_metrics = compute_iterative_tail_metrics(data.targets, baseline)
    selected_metrics = compute_iterative_tail_metrics(data.targets, nested_selected)
    bootstrap = _bootstrap_macro_rmse(data.targets, baseline, nested_selected, resamples=10_000, seed=SEED)
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
            "role": "nested_selected_official_terra_stack" if final_pass else "exact_r0_oof_baseline_fallback",
        }
        for row, source_id in enumerate(data.source_ids)
    ])
    payload = {
        "schema_version": "mal2026-iterative-official-agent-stack-aggregate-v7",
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
        "final_selection": "nested-selected-official-terra-stack" if final_pass else "exact-r0-oof-baseline-fallback",
        "final_prediction_sha256": _sha256(final_path),
        "fold_direction_diagnostics": fold_direction_diagnostics(per_fold_baseline, per_fold_selected),
        "feature_audit": bundle.feature_audit,
        "official_candidate_provenance": bundle.provenance,
        "adaptive_after_full_oof_prestudy": True,
        "prestudy_role": "candidate_design_evidence_only_not_outer_confirmation",
        "independent_confirmation_or_generalization_claim": False,
        "posthoc_selection_on_concatenated_outer_predictions": False,
        "validation_loaded": False,
        "average_target_used": False,
        "external_api_calls": 0,
    }
    aggregate_path = PUBLIC_ROOT / "aggregate.json"
    _write_json(aggregate_path, payload)
    completion = {
        "schema_version": "mal2026-iterative-official-agent-stack-completion-v7",
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
    bundle = load_agent_stack_data()
    data = bundle.experiment
    train = np.concatenate([_indices(data, (fold,))[:32] for fold in (1, 2, 3)])
    predict = _indices(data, (4,))[:32]
    records = []
    hashes: set[str] = set()
    for spec in candidate_specs():
        result = fit_predict_agent_stack(
            spec, bundle.features[train], data.base[train], data.targets[train],
            bundle.features[predict], data.base[predict], device=device,
        )
        if result.predictions.shape != (32, 3) or not np.isfinite(result.predictions).all():
            raise OfficialAgentStackRunError("V7 smoke prediction differs")
        coefficient_hash = str(result.audit["coefficient_sha256"])
        if coefficient_hash in hashes:
            raise OfficialAgentStackRunError("V7 smoke candidate coefficients unexpectedly identical")
        hashes.add(coefficient_hash)
        records.append({"variant_id": spec.variant_id, "audit": result.audit})
    payload = {
        "schema_version": "mal2026-iterative-official-agent-stack-smoke-v7",
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
    "PUBLIC_ROOT", "RESTRICTED_ROOT", "SEED", "AgentStackData", "OfficialAgentStackRunError",
    "aggregate_outer_results", "gpu0_smoke", "load_agent_stack_data", "run_outer_fold",
]
