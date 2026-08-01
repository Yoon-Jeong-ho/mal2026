"""Execution and aggregate-only reporting for the fixed v3 20-cycle study."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mal2026.iterative_tail_cycle_models import CycleSpec, cycle_specs, fit_predict
from mal2026.iterative_tail_cycle_protocol import (
    RUN_ID,
    CycleProtocol,
    load_protocol,
    validate_bound_inputs,
    validate_model_inventory,
)
from mal2026.iterative_tail_metrics import compute_iterative_tail_metrics, metric_improvements
from mal2026.iterative_tail_models import CandidateSpec, fit_predict as fit_frozen_candidate
from mal2026.iterative_tail_remediation_runner import (
    _bootstrap_macro_rmse,
    _indices,
    _ridge_pair,
    regenerate_r16_teacher_oof,
)
from mal2026.iterative_tail_runner import ExperimentData, load_experiment_data


PUBLIC_ROOT = Path("outputs/iterative-tail-cycle-v3") / RUN_ID
RESTRICTED_ROOT = Path("data/processed/restricted/iterative_tail_cycle_v3") / RUN_ID
DIRECT_ALPHA = 100.0
SEED = 2026080103


class IterativeTailCycleRunError(RuntimeError):
    """Raised when fold isolation, inventory, or output coverage differs."""


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _protocol_direct_alpha(protocol: CycleProtocol) -> float:
    value = protocol.raw["fold_protocol"]["direct_evidence_ridge_challenger"].get("ridge_alpha")
    if value is None or float(value) != DIRECT_ALPHA:
        raise IterativeTailCycleRunError("direct challenger alpha differs from runner registration")
    return float(value)


def run_fold(
    heldout_fold: int,
    *,
    device: str,
    protocol: CycleProtocol | None = None,
) -> Mapping[str, Any]:
    """Generate all 20 heldout predictions without opening heldout metrics."""
    protocol = protocol or load_protocol()
    validate_bound_inputs(protocol)
    validate_model_inventory(require_available=True)
    if heldout_fold not in protocol.raw["fold_protocol"]["heldout_folds"]:
        raise IterativeTailCycleRunError("heldout fold is not registered")
    data = load_experiment_data()
    train = _indices(data, tuple(fold for fold in range(5) if fold != heldout_fold))
    predict = _indices(data, (heldout_fold,))
    if len(train) != 1600 or len(predict) != 400 or np.any(data.folds[train] == heldout_fold):
        raise IterativeTailCycleRunError("heldout/train partition differs")

    teacher, teacher_audit = regenerate_r16_teacher_oof(data, heldout_fold, device=device)
    r17 = _ridge_pair(data, train, predict, teacher, alpha=10.0, device=device)
    direct = _ridge_pair(
        data, train, predict, data.targets,
        alpha=_protocol_direct_alpha(protocol), device=device,
    )
    evidence = data.evidence.view("evidence_hash")
    assert evidence is not None

    predictions: dict[str, np.ndarray] = {}
    audits = []
    for spec in cycle_specs():
        result = fit_predict(
            spec,
            data.targets[train], data.base[train], r17.train, direct.train, evidence[train],
            data.base[predict], r17.predict, direct.predict, evidence[predict],
        )
        if result.predictions.shape != (400, 3) or not np.isfinite(result.predictions).all():
            raise IterativeTailCycleRunError(f"cycle {spec.cycle} prediction coverage differs")
        predictions[spec.variant_id] = result.predictions
        audits.append(dict(result.audit))

    if len(predictions) != 20:
        raise IterativeTailCycleRunError("all 20 cycle predictions are required")
    restricted_path = RESTRICTED_ROOT / f"fold-{heldout_fold}" / "predictions.jsonl"
    _write_jsonl(restricted_path, [
        {
            "source_id": data.source_ids[index],
            "fold": heldout_fold,
            "baseline_prediction": [float(value) for value in data.base[index]],
            "cycle_predictions": {
                spec.variant_id: [float(value) for value in predictions[spec.variant_id][row]]
                for spec in cycle_specs()
            },
        }
        for row, index in enumerate(predict)
    ])
    payload = {
        "schema_version": "mal2026-iterative-tail-cycle-fold-v3",
        "status": "completed",
        "heldout_fold": heldout_fold,
        "train_count": len(train),
        "heldout_count": len(predict),
        "cycle_count": len(predictions),
        "cycle_order": [spec.variant_id for spec in cycle_specs()],
        "cycle_audits": audits,
        "r16_teacher_audit": teacher_audit,
        "r17_ridge_alpha": 10.0,
        "direct_ridge_alpha": DIRECT_ALPHA,
        "restricted_prediction_sha256": _sha256(restricted_path),
        "heldout_gold_used_for_model_fit_or_selection": False,
        "heldout_metrics_computed_before_all_folds_complete": False,
        "historical_v1_or_v2_predictions_used_as_features": False,
        "validation_loaded": False,
        "average_target_used": False,
    }
    _write_json(PUBLIC_ROOT / f"fold-{heldout_fold}" / "result.json", payload)
    return payload


def _read_predictions(data: ExperimentData) -> tuple[np.ndarray, dict[str, np.ndarray], list[Mapping[str, Any]]]:
    specs = cycle_specs()
    baseline = np.full_like(data.base, np.nan, dtype=np.float64)
    predictions = {spec.variant_id: np.full_like(data.base, np.nan, dtype=np.float64) for spec in specs}
    id_to_index = {source_id: index for index, source_id in enumerate(data.source_ids)}
    seen: set[str] = set()
    audits = []
    required_row_keys = {"source_id", "fold", "baseline_prediction", "cycle_predictions"}
    required_variants = {spec.variant_id for spec in specs}
    for fold in range(5):
        result_path = PUBLIC_ROOT / f"fold-{fold}" / "result.json"
        prediction_path = RESTRICTED_ROOT / f"fold-{fold}" / "predictions.jsonl"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "completed"
            or result.get("cycle_count") != 20
            or result.get("restricted_prediction_sha256") != _sha256(prediction_path)
        ):
            raise IterativeTailCycleRunError("fold result binding differs")
        with prediction_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if set(row) != required_row_keys or set(row["cycle_predictions"]) != required_variants:
                    raise IterativeTailCycleRunError("restricted prediction schema differs")
                source_id = row["source_id"]
                if source_id in seen or source_id not in id_to_index or row["fold"] != fold:
                    raise IterativeTailCycleRunError("restricted prediction population differs")
                index = id_to_index[source_id]
                if int(data.folds[index]) != fold:
                    raise IterativeTailCycleRunError("restricted prediction fold differs")
                baseline[index] = row["baseline_prediction"]
                for variant, values in row["cycle_predictions"].items():
                    predictions[variant][index] = values
                seen.add(source_id)
        audits.append({
            "fold": fold,
            "result_sha256": _sha256(result_path),
            "prediction_sha256": _sha256(prediction_path),
            "cycle_count": result["cycle_count"],
        })
    if len(seen) != 2000 or not np.isfinite(baseline).all():
        raise IterativeTailCycleRunError("five-fold population coverage differs")
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise IterativeTailCycleRunError("one or more cycle OOF matrices are incomplete")
    return baseline, predictions, audits


def _promotion_decision(
    protocol: CycleProtocol,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Mapping[str, Any]:
    config = protocol.raw["promotion_gate"]
    delta = metric_improvements(baseline, candidate)
    finite = all(
        value is not None and math.isfinite(float(value))
        for value in (
            delta["rmse"], delta["equal_group_rmse"], delta["low_tail_rmse"],
            delta["high_tail_rmse"], delta["gold_3_4_balanced_accuracy"], delta["spearman"],
            *delta["axis_rmse"].values(),
        )
    )
    gates = {
        "macro_rmse_improvement": finite and delta["rmse"] >= config["macro_rmse_min_improvement"],
        "equal_group_rmse_improvement": finite and delta["equal_group_rmse"] >= config["equal_group_rmse_min_improvement"],
        "low_tail_improves": finite and delta["low_tail_rmse"] > 0.0,
        "high_tail_improves": finite and delta["high_tail_rmse"] > 0.0,
        "gold_3_4_balanced_accuracy_improvement": finite and delta["gold_3_4_balanced_accuracy"] >= config["gold_3_4_balanced_accuracy_min_improvement"],
        "axis_rmse_worsening_bound": finite and all(
            value >= -config["max_axis_rmse_worsening"] for value in delta["axis_rmse"].values()
        ),
        "macro_spearman_fall_bound": finite and delta["spearman"] >= -config["max_macro_spearman_fall"],
    }
    return {
        "eligible": finite and config["operator"] == "AND" and all(gates.values()),
        "gates": gates,
        "improvements": delta,
        "finite_metrics": finite,
        "score1_used_for_promotion": False,
    }


def _macro_band(metrics: Mapping[str, Any], band: int, key: str) -> float | None:
    values = [metrics["axes"][axis]["bands"][str(band)][key] for axis in ("content", "organization", "expression")]
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def aggregate_results(protocol: CycleProtocol | None = None) -> Mapping[str, Any]:
    """Unlock train OOF metrics only after all 100 fold-cycle outputs exist."""
    protocol = protocol or load_protocol()
    validate_bound_inputs(protocol)
    data = load_experiment_data()
    baseline_prediction, predictions, fold_audits = _read_predictions(data)
    if not np.allclose(baseline_prediction, data.base.astype(np.float64), atol=1e-7, rtol=0.0):
        raise IterativeTailCycleRunError("v3 baseline differs from exact R0 OOF")
    baseline_metrics = compute_iterative_tail_metrics(data.targets, baseline_prediction)
    records = []
    eligible = []
    for spec in cycle_specs():
        metrics = compute_iterative_tail_metrics(data.targets, predictions[spec.variant_id])
        decision = _promotion_decision(protocol, baseline_metrics, metrics)
        band_diagnostics = {}
        for band in (1, 2, 3, 4, 5):
            band_diagnostics[str(band)] = {
                key: {
                    "baseline": _macro_band(baseline_metrics, band, key),
                    "candidate": _macro_band(metrics, band, key),
                }
                for key in ("rmse", "recall", "one_off")
            }
        record = {
            "cycle": spec.cycle,
            "variant_id": spec.variant_id,
            "family": spec.family,
            "parameters": dict(spec.parameters),
            "metrics": metrics,
            "decision": decision,
            "band_diagnostics": band_diagnostics,
        }
        records.append(record)
        if decision["eligible"]:
            eligible.append((float(metrics["macro"]["rmse"]), spec.cycle, spec.variant_id))
    best_exploratory = min(records, key=lambda record: (float(record["metrics"]["macro"]["rmse"]), record["cycle"]))
    selected_variant = min(eligible)[2] if eligible else "exact-r0-oof-baseline"
    selected_prediction = baseline_prediction if not eligible else predictions[selected_variant]
    selected_metrics = baseline_metrics if not eligible else next(
        record["metrics"] for record in records if record["variant_id"] == selected_variant
    )
    bootstrap_selected = _bootstrap_macro_rmse(data.targets, baseline_prediction, selected_prediction)
    bootstrap_best = _bootstrap_macro_rmse(
        data.targets, baseline_prediction, predictions[best_exploratory["variant_id"]],
    )
    payload = {
        "schema_version": "mal2026-iterative-tail-cycle-aggregate-v3",
        "status": "completed",
        "record_count": 2000,
        "fold_count": 5,
        "cycle_count": len(records),
        "fold_cycle_prediction_count": 100,
        "fold_audits": fold_audits,
        "baseline_metrics": baseline_metrics,
        "cycles": records,
        "strict_eligible_cycles": [cycle for _, cycle, _ in sorted(eligible)],
        "strict_discovery_gate_pass": bool(eligible),
        "strict_selection": selected_variant,
        "strict_selected_metrics": selected_metrics,
        "strict_selected_bootstrap": bootstrap_selected,
        "best_exploratory_variant": best_exploratory["variant_id"],
        "best_exploratory_cycle": best_exploratory["cycle"],
        "best_exploratory_metrics": best_exploratory["metrics"],
        "best_exploratory_decision": best_exploratory["decision"],
        "best_exploratory_postselection_bootstrap": bootstrap_best,
        "adaptive_after_v2_outer_observed": True,
        "train_only_descriptive_discovery": True,
        "confirmatory_or_generalization_claim": False,
        "future_untouched_evaluation_required": True,
        "validation_loaded": False,
        "average_target_used": False,
        "external_api_calls": 0,
    }
    aggregate_path = PUBLIC_ROOT / "aggregate.json"
    _write_json(aggregate_path, payload)
    _write_json(PUBLIC_ROOT / "completion.json", {
        "schema_version": "mal2026-iterative-tail-cycle-completion-v3",
        "status": "completed_strict_candidate_frozen" if eligible else "completed_no_strict_candidate_baseline_retained",
        "aggregate_sha256": _sha256(aggregate_path),
        "cycle_count": 20,
        "fold_cycle_prediction_count": 100,
        "strict_discovery_gate_pass": bool(eligible),
        "strict_selection": selected_variant,
        "gpu_scope": [0, 1, 2, 3],
        "validation_loaded": False,
        "average_target_used": False,
        "external_api_calls": 0,
    })
    return payload


def gpu0_smoke(*, device: str = "cuda:0") -> Mapping[str, Any]:
    protocol = load_protocol()
    validate_bound_inputs(protocol)
    validate_model_inventory(require_available=True)
    data = load_experiment_data()
    consensus = data.evidence.view("consensus_disagreement")
    evidence = data.evidence.view("evidence_hash")
    assert consensus is not None and evidence is not None
    train = _indices(data, (1,))[:128]
    predict = _indices(data, (2,))[:32]
    neural = fit_frozen_candidate(
        CandidateSpec(
            family="joint_huber_ordinal", seed=SEED, device=device,
            hidden_dim=128, epochs=2, learning_rate=1e-3,
            huber_delta=1.0, ordinal_weight=0.5,
        ),
        data.embeddings[train], data.base[train], data.targets[train],
        data.embeddings[predict], data.base[predict],
        train_extra_features=consensus[train], predict_extra_features=consensus[predict],
    )
    train_r17 = np.clip(data.base[train] + 0.05, 1.0, 5.0)
    predict_r17 = np.clip(data.base[predict] + 0.05, 1.0, 5.0)
    train_direct = np.clip(data.base[train] - 0.02, 1.0, 5.0)
    predict_direct = np.clip(data.base[predict] - 0.02, 1.0, 5.0)
    family_smokes = []
    for spec in (cycle_specs()[0], cycle_specs()[4], cycle_specs()[8], cycle_specs()[12], cycle_specs()[16]):
        result = fit_predict(
            spec, data.targets[train], data.base[train], train_r17, train_direct, evidence[train],
            data.base[predict], predict_r17, predict_direct, evidence[predict],
        )
        if result.predictions.shape != (32, 3) or not np.isfinite(result.predictions).all():
            raise IterativeTailCycleRunError("v3 family smoke output differs")
        family_smokes.append({"cycle": spec.cycle, "family": spec.family})
    payload = {
        "schema_version": "mal2026-iterative-tail-cycle-smoke-v3",
        "status": "completed",
        "gpu": 0,
        "train_count": len(train),
        "predict_count": len(predict),
        "neural_initial_state_hashes": list(neural.initial_state_hashes),
        "neural_final_state_hashes": list(neural.final_state_hashes),
        "family_smokes": family_smokes,
        "validation_loaded": False,
        "average_target_used": False,
    }
    _write_json(PUBLIC_ROOT / "smoke.json", payload)
    return payload


__all__ = [
    "DIRECT_ALPHA", "PUBLIC_ROOT", "RESTRICTED_ROOT", "IterativeTailCycleRunError",
    "aggregate_results", "gpu0_smoke", "run_fold",
]
