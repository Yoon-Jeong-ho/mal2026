"""Leakage-sealed nested execution for the final v4 score router study.

For each outer fold, four fresh inner component banks are built without either
the inner validation fold or the outer holdout.  All twenty registered routers
are then fit and compared only on the concatenated inner OOF bank.  The chosen
route is frozen before fresh outer components are trained and the outer fold is
predicted once.  Row-level outputs stay in the restricted data tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mal2026.iterative_tail_cycle_models import cycle_specs, fit_predict as fit_cycle_component
from mal2026.iterative_tail_metrics import compute_iterative_tail_metrics, metric_improvements
from mal2026.iterative_tail_models import CandidateSpec, fit_predict as fit_frozen_candidate
from mal2026.iterative_tail_remediation_runner import (
    _bootstrap_macro_rmse,
    _indices,
    _ridge_pair,
    regenerate_inner_selection_teacher,
    regenerate_r16_teacher_oof,
)
from mal2026.iterative_tail_router_models import (
    RouterResult,
    RouterSpec,
    apply_route,
    fit_route,
    router_specs,
)
from mal2026.iterative_tail_router_protocol import (
    RUN_ID,
    RouterProtocol,
    load_protocol,
    validate_bound_inputs,
    validate_model_inventory,
)
from mal2026.iterative_tail_runner import ExperimentData, load_experiment_data


PUBLIC_ROOT = Path("outputs/iterative-tail-router-v4") / RUN_ID
RESTRICTED_ROOT = Path("data/processed/restricted/iterative_tail_router_v4") / RUN_ID
SEED = 2026080104
RIDGE_ALPHA = 100.0
SOFT_SPEC = cycle_specs()[3]
HURDLE_SPEC = cycle_specs()[12]


class IterativeTailRouterRunError(RuntimeError):
    """Raised when v4 inventory, nesting, coverage, or output binding differs."""


@dataclass(frozen=True)
class ComponentBank:
    r17: np.ndarray
    direct: np.ndarray
    hurdle: np.ndarray
    soft: np.ndarray
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
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
        raise IterativeTailRouterRunError("outer fold must be 0..4")
    return tuple(fold for fold in range(5) if fold != outer_fold)


def _component_prediction(
    spec: Any,
    data: ExperimentData,
    train: np.ndarray,
    predict: np.ndarray,
    r17_train: np.ndarray,
    r17_predict: np.ndarray,
    direct_train: np.ndarray,
    direct_predict: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    evidence = data.evidence.view("evidence_hash")
    if evidence is None:
        raise IterativeTailRouterRunError("score-blind evidence_hash view is unavailable")
    result = fit_cycle_component(
        spec,
        data.targets[train], data.base[train], r17_train, direct_train, evidence[train],
        data.base[predict], r17_predict, direct_predict, evidence[predict],
    )
    if result.predictions.shape != (len(predict), 3) or not np.isfinite(result.predictions).all():
        raise IterativeTailRouterRunError(f"{spec.variant_id} component coverage differs")
    return result.predictions, result.audit


def _build_inner_component_bank(
    data: ExperimentData,
    outer_fold: int,
    *,
    device: str,
) -> ComponentBank:
    """Build four fresh component OOF matrices over the sealed outer train."""
    outer_folds = _outer_train_folds(outer_fold)
    outer_train = _indices(data, outer_folds)
    matrices = {
        name: np.full_like(data.base, np.nan, dtype=np.float32)
        for name in ("r17", "direct", "hurdle", "soft")
    }
    seen = np.zeros(len(data.targets), dtype=np.int8)
    audits: list[Mapping[str, Any]] = []
    for inner_validation in outer_folds:
        train_folds = tuple(fold for fold in outer_folds if fold != inner_validation)
        train = _indices(data, train_folds)
        predict = _indices(data, (inner_validation,))
        if len(train) != 1200 or len(predict) != 400:
            raise IterativeTailRouterRunError("inner S/D counts differ from 1200/400")
        if np.any(np.isin(data.folds[train], (outer_fold, inner_validation))):
            raise IterativeTailRouterRunError("outer or inner holdout entered component fitting")

        teacher, teacher_audit = regenerate_inner_selection_teacher(
            data, outer_fold, inner_validation, device=device,
        )
        r17 = _ridge_pair(data, train, predict, teacher, alpha=RIDGE_ALPHA, device=device)
        direct = _ridge_pair(data, train, predict, data.targets, alpha=RIDGE_ALPHA, device=device)
        soft_prediction, soft_audit = _component_prediction(
            SOFT_SPEC, data, train, predict,
            r17.train, r17.predict, direct.train, direct.predict,
        )
        hurdle_prediction, hurdle_audit = _component_prediction(
            HURDLE_SPEC, data, train, predict,
            r17.train, r17.predict, direct.train, direct.predict,
        )
        matrices["r17"][predict] = r17.predict
        matrices["direct"][predict] = direct.predict
        matrices["soft"][predict] = soft_prediction
        matrices["hurdle"][predict] = hurdle_prediction
        seen[predict] += 1
        audits.append({
            "outer_fold": outer_fold,
            "inner_validation_fold": inner_validation,
            "component_train_folds": list(train_folds),
            "forbidden_folds": [inner_validation, outer_fold],
            "train_count": len(train),
            "predict_count": len(predict),
            "r16_teacher_regeneration": teacher_audit,
            "r17_ridge_alpha": RIDGE_ALPHA,
            "direct_ridge_alpha": RIDGE_ALPHA,
            "soft_component": _json_safe(soft_audit),
            "hurdle_component": _json_safe(hurdle_audit),
        })
    if not np.all(seen[outer_train] == 1) or np.any(seen[_indices(data, (outer_fold,))]):
        raise IterativeTailRouterRunError("inner OOF component assignment is not exactly-once and sealed")
    if any(not np.isfinite(matrix[outer_train]).all() for matrix in matrices.values()):
        raise IterativeTailRouterRunError("inner OOF component bank is incomplete")
    return ComponentBank(
        matrices["r17"][outer_train], matrices["direct"][outer_train],
        matrices["hurdle"][outer_train], matrices["soft"][outer_train], tuple(audits),
    )


def gate_decision(
    config: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    final: bool = False,
    bootstrap: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Apply config-bound gates; every delta is oriented positive-is-better."""
    delta = metric_improvements(baseline_metrics, candidate_metrics)
    values = (
        delta["rmse"], delta["equal_group_rmse"], delta["low_tail_rmse"],
        delta["high_tail_rmse"], delta["gold_3_4_balanced_accuracy"], delta["spearman"],
        *delta["axis_rmse"].values(),
    )
    finite = all(value is not None and math.isfinite(float(value)) for value in values)
    gates: dict[str, bool] = {
        "macro_rmse_improvement": finite and float(delta["rmse"]) >= float(config["macro_rmse_min_improvement"]),
        "low_tail_improves": finite and (not config["low_tail_must_improve"] or float(delta["low_tail_rmse"]) > 0.0),
        "high_tail_improves": finite and (not config["high_tail_must_improve"] or float(delta["high_tail_rmse"]) > 0.0),
        "gold_3_4_balanced_accuracy_improvement": finite and float(delta["gold_3_4_balanced_accuracy"]) >= float(config["gold_3_4_balanced_accuracy_min_improvement"]),
        "axis_rmse_worsening_bound": finite and all(
            float(value) >= -float(config["max_axis_rmse_worsening"])
            for value in delta["axis_rmse"].values()
        ),
        "macro_spearman_fall_bound": finite and float(delta["spearman"]) >= -float(config["max_macro_spearman_fall"]),
    }
    if not final:
        gates["equal_group_rmse_improvement"] = finite and float(delta["equal_group_rmse"]) >= float(config["equal_group_rmse_min_improvement"])
    else:
        if bootstrap is None:
            raise IterativeTailRouterRunError("final gate requires paired bootstrap evidence")
        required_upper = float(config["paired_bootstrap"]["required_upper_bound_lt"])
        gates["candidate_minus_baseline_rmse_ci_upper_below_bound"] = (
            float(bootstrap["candidate_minus_baseline_ci"]["upper"]) < required_upper
        )
    if config.get("require_finite_metrics") and not finite:
        gates = {name: False for name in gates}
    eligible = config["operator"] == "AND" and all(gates.values())
    return {
        "eligible" if not final else "pass": eligible,
        "gates": gates,
        "improvements": delta,
        "finite_metrics": finite,
        "score1_used_for_promotion": False,
    }


def _fit_and_select_routes(
    protocol: RouterProtocol,
    data: ExperimentData,
    outer_train: np.ndarray,
    bank: ComponentBank,
) -> tuple[RouterSpec | None, Mapping[str, Any] | None, list[Mapping[str, Any]]]:
    baseline_metrics = compute_iterative_tail_metrics(data.targets[outer_train], data.base[outer_train])
    eligible: list[tuple[float, int, RouterSpec, Mapping[str, Any]]] = []
    records: list[Mapping[str, Any]] = []
    for spec in router_specs():
        result: RouterResult = fit_route(
            spec, data.targets[outer_train], data.base[outer_train],
            bank.r17, bank.direct, bank.hurdle, bank.soft,
        )
        metrics = compute_iterative_tail_metrics(data.targets[outer_train], result.train_predictions)
        decision = gate_decision(protocol.raw["inner_promotion_gate"], baseline_metrics, metrics)
        parameters = _json_safe(result.selected_parameters)
        record = {
            "cycle": spec.cycle,
            "variant_id": spec.variant_id,
            "family": spec.family,
            "registered_parameters": dict(spec.parameters),
            "selected_parameters": parameters,
            "router_fit_audit": _json_safe(result.audit),
            "metrics": metrics,
            "baseline_relative_decision": decision,
        }
        records.append(record)
        if decision["eligible"]:
            eligible.append((float(metrics["macro"]["rmse"]), spec.cycle, spec, parameters))
    if len(records) != 20:
        raise IterativeTailRouterRunError("all twenty routes must finish before selection")
    if not eligible:
        return None, None, records
    _, _, selected_spec, selected_parameters = min(eligible, key=lambda item: (item[0], item[1]))
    return selected_spec, selected_parameters, records


def _outer_component_bank(
    data: ExperimentData,
    outer_fold: int,
    *,
    device: str,
) -> tuple[ComponentBank, Sequence[Mapping[str, Any]]]:
    """Freshly refit all components after route selection has frozen."""
    train = _indices(data, _outer_train_folds(outer_fold))
    predict = _indices(data, (outer_fold,))
    teacher, teacher_audit = regenerate_r16_teacher_oof(data, outer_fold, device=device)
    r17 = _ridge_pair(data, train, predict, teacher, alpha=RIDGE_ALPHA, device=device)
    direct = _ridge_pair(data, train, predict, data.targets, alpha=RIDGE_ALPHA, device=device)
    soft, soft_audit = _component_prediction(
        SOFT_SPEC, data, train, predict,
        r17.train, r17.predict, direct.train, direct.predict,
    )
    hurdle, hurdle_audit = _component_prediction(
        HURDLE_SPEC, data, train, predict,
        r17.train, r17.predict, direct.train, direct.predict,
    )
    audit = ({
        "outer_fold": outer_fold,
        "train_folds": list(_outer_train_folds(outer_fold)),
        "predict_fold": outer_fold,
        "train_count": len(train),
        "predict_count": len(predict),
        "selection_was_frozen_before_refit": True,
        "r16_teacher_regeneration": teacher_audit,
        "r17_ridge_alpha": RIDGE_ALPHA,
        "direct_ridge_alpha": RIDGE_ALPHA,
        "soft_component": _json_safe(soft_audit),
        "hurdle_component": _json_safe(hurdle_audit),
    },)
    return ComponentBank(r17.predict, direct.predict, hurdle, soft, audit), audit


def run_outer_fold(
    outer_fold: int,
    *,
    device: str,
    protocol: RouterProtocol | None = None,
) -> Mapping[str, Any]:
    """Run one sealed outer fold and persist public aggregate/restricted rows."""
    protocol = protocol or load_protocol()
    validate_bound_inputs(protocol)
    validate_model_inventory(require_available=True)
    if outer_fold not in protocol.raw["nested_protocol"]["outer_folds"]:
        raise IterativeTailRouterRunError("outer fold is not registered")
    if float(protocol.raw["nested_protocol"]["inner_components"]["r17_challenger"]["ridge_alpha"]) != RIDGE_ALPHA:
        raise IterativeTailRouterRunError("R17 ridge alpha differs from runner registration")
    data = load_experiment_data()
    outer_train = _indices(data, _outer_train_folds(outer_fold))
    outer_indices = _indices(data, (outer_fold,))

    inner_bank = _build_inner_component_bank(data, outer_fold, device=device)
    selected_spec, selected_parameters, route_records = _fit_and_select_routes(
        protocol, data, outer_train, inner_bank,
    )
    # The route and its fitted aggregate weights are immutable beyond this line.
    if selected_spec is None:
        prediction = data.base[outer_indices].copy()
        outer_refit_audit: Sequence[Mapping[str, Any]] = ({
            "status": "not_required_by_frozen_baseline_fallback",
            "selection_was_frozen": True,
        },)
        selected_name = "exact-r0-oof-baseline"
    else:
        outer_bank, outer_refit_audit = _outer_component_bank(data, outer_fold, device=device)
        prediction = apply_route(
            selected_spec, selected_parameters or {}, data.base[outer_indices],
            outer_bank.r17, outer_bank.direct, outer_bank.hurdle, outer_bank.soft,
        )
        selected_name = selected_spec.variant_id
    if prediction.shape != (400, 3) or not np.isfinite(prediction).all():
        raise IterativeTailRouterRunError("outer prediction coverage differs")

    # Outer gold is opened only after the prediction matrix exists and is frozen.
    baseline_metrics = compute_iterative_tail_metrics(data.targets[outer_indices], data.base[outer_indices])
    selected_metrics = compute_iterative_tail_metrics(data.targets[outer_indices], prediction)
    restricted_path = RESTRICTED_ROOT / f"outer-{outer_fold}" / "predictions.jsonl"
    _write_jsonl(restricted_path, [
        {
            "source_id": data.source_ids[index],
            "outer_fold": outer_fold,
            "baseline_prediction": data.base[index],
            "selected_prediction": row,
        }
        for index, row in zip(outer_indices, prediction, strict=True)
    ])
    payload = {
        "schema_version": "mal2026-iterative-tail-router-outer-v4",
        "status": "completed",
        "outer_fold": outer_fold,
        "outer_train_count": 1600,
        "outer_holdout_count": 400,
        "inner_fold_count": 4,
        "route_count": 20,
        "selected_route": selected_name,
        "selected_cycle": None if selected_spec is None else selected_spec.cycle,
        "selected_parameters": selected_parameters,
        "fell_back_to_baseline": selected_spec is None,
        "route_records": route_records,
        "inner_component_audit": inner_bank.audit,
        "outer_refit_audit": outer_refit_audit,
        "baseline_metrics": baseline_metrics,
        "selected_metrics": selected_metrics,
        "restricted_prediction_sha256": _sha256(restricted_path),
        "outer_gold_used_before_route_freeze_or_prediction": False,
        "historical_row_predictions_used_as_features": False,
        "validation_loaded": False,
        "average_target_used": False,
        "external_api_calls": 0,
    }
    _write_json(PUBLIC_ROOT / f"outer-{outer_fold}" / "result.json", payload)
    return payload


def _read_outer_predictions(data: ExperimentData) -> tuple[np.ndarray, np.ndarray, list[Mapping[str, Any]]]:
    baseline = np.full_like(data.base, np.nan, dtype=np.float64)
    selected = np.full_like(data.base, np.nan, dtype=np.float64)
    seen: set[str] = set()
    audits = []
    id_to_index = {source_id: index for index, source_id in enumerate(data.source_ids)}
    required_keys = {"source_id", "outer_fold", "baseline_prediction", "selected_prediction"}
    for outer_fold in range(5):
        result_path = PUBLIC_ROOT / f"outer-{outer_fold}" / "result.json"
        prediction_path = RESTRICTED_ROOT / f"outer-{outer_fold}" / "predictions.jsonl"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "completed" or result.get("route_count") != 20
            or result.get("restricted_prediction_sha256") != _sha256(prediction_path)
        ):
            raise IterativeTailRouterRunError("outer result/prediction binding differs")
        with prediction_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if set(row) != required_keys:
                    raise IterativeTailRouterRunError("restricted outer schema differs")
                source_id = row["source_id"]
                if source_id in seen or source_id not in id_to_index or row["outer_fold"] != outer_fold:
                    raise IterativeTailRouterRunError("restricted outer population differs")
                index = id_to_index[source_id]
                if int(data.folds[index]) != outer_fold:
                    raise IterativeTailRouterRunError("restricted outer fold assignment differs")
                baseline[index] = row["baseline_prediction"]
                selected[index] = row["selected_prediction"]
                seen.add(source_id)
        audits.append({
            "outer_fold": outer_fold,
            "selected_route": result["selected_route"],
            "selected_cycle": result["selected_cycle"],
            "fell_back_to_baseline": result["fell_back_to_baseline"],
            "result_sha256": _sha256(result_path),
            "prediction_sha256": _sha256(prediction_path),
        })
    if len(seen) != 2000 or not np.isfinite(baseline).all() or not np.isfinite(selected).all():
        raise IterativeTailRouterRunError("five-fold outer prediction coverage differs")
    return baseline, selected, audits


def aggregate_outer_results(protocol: RouterProtocol | None = None) -> Mapping[str, Any]:
    """Concatenate all five outer predictions once and apply the final gate."""
    protocol = protocol or load_protocol()
    validate_bound_inputs(protocol)
    data = load_experiment_data()
    baseline, selected, audits = _read_outer_predictions(data)
    if not np.allclose(baseline, data.base.astype(np.float64), rtol=0.0, atol=1e-7):
        raise IterativeTailRouterRunError("nested baseline differs from exact R0 OOF")
    baseline_metrics = compute_iterative_tail_metrics(data.targets, baseline)
    selected_metrics = compute_iterative_tail_metrics(data.targets, selected)
    final_config = protocol.raw["final_evaluation"]
    bootstrap = _bootstrap_macro_rmse(
        data.targets, baseline, selected,
        resamples=int(final_config["paired_bootstrap"]["replicates"]), seed=SEED,
    )
    decision = gate_decision(
        final_config, baseline_metrics, selected_metrics, final=True, bootstrap=bootstrap,
    )
    final_pass = bool(decision["pass"])
    payload = {
        "schema_version": "mal2026-iterative-tail-router-aggregate-v4",
        "status": "completed",
        "record_count": 2000,
        "outer_fold_count": 5,
        "inner_fold_count_per_outer": 4,
        "route_count_per_outer": 20,
        "outer_audits": audits,
        "baseline_metrics": baseline_metrics,
        "nested_selected_metrics": selected_metrics,
        "paired_bootstrap": bootstrap,
        "final_decision": decision,
        "final_gate_pass": final_pass,
        "final_selection": "nested-v4-router" if final_pass else "exact-r0-oof-baseline-fallback",
        "same_train_search_frozen": not final_pass,
        "adaptive_same_train_descriptive_evidence": True,
        "independent_confirmation_or_generalization_claim": False,
        "posthoc_selection_on_concatenated_outer_predictions": False,
        "historical_row_predictions_used_as_features": False,
        "validation_loaded": False,
        "average_target_used": False,
        "external_api_calls": 0,
    }
    aggregate_path = PUBLIC_ROOT / "aggregate.json"
    _write_json(aggregate_path, payload)
    _write_json(PUBLIC_ROOT / "completion.json", {
        "schema_version": "mal2026-iterative-tail-router-completion-v4",
        "status": "completed_final_gate_pass" if final_pass else "completed_no_promotion_same_train_search_frozen",
        "aggregate_sha256": _sha256(aggregate_path),
        "final_gate_pass": final_pass,
        "final_selection": payload["final_selection"],
        "same_train_search_frozen": not final_pass,
        "gpu_scope": [0, 1, 2, 3],
        "external_api_calls": 0,
        "validation_loaded": False,
        "average_target_used": False,
    })
    return payload


def gpu0_smoke(*, device: str = "cuda:0") -> Mapping[str, Any]:
    """Smallest real neural/component/router integration gate on physical GPU0."""
    protocol = load_protocol()
    validate_bound_inputs(protocol)
    validate_model_inventory(require_available=True)
    data = load_experiment_data()
    consensus = data.evidence.view("consensus_disagreement")
    evidence = data.evidence.view("evidence_hash")
    if consensus is None or evidence is None:
        raise IterativeTailRouterRunError("smoke evidence views are unavailable")
    train = _indices(data, (1,))[:128]
    predict = _indices(data, (2,))[:32]
    neural = fit_frozen_candidate(
        CandidateSpec(
            family="joint_huber_ordinal", seed=SEED, device=device,
            hidden_dim=128, epochs=2, learning_rate=1e-3,
            dropout=0.1, huber_delta=1.0, ordinal_weight=0.5,
        ),
        data.embeddings[train], data.base[train], data.targets[train],
        data.embeddings[predict], data.base[predict],
        train_extra_features=consensus[train], predict_extra_features=consensus[predict],
    )
    r17_train = np.clip(data.base[train] + .05, 1, 5)
    r17_predict = np.clip(data.base[predict] + .05, 1, 5)
    direct_train = np.clip(data.base[train] - .02, 1, 5)
    direct_predict = np.clip(data.base[predict] - .02, 1, 5)
    soft_train, _ = _component_prediction(
        SOFT_SPEC, data, train, train,
        r17_train, r17_train, direct_train, direct_train,
    )
    hurdle_train, _ = _component_prediction(
        HURDLE_SPEC, data, train, train,
        r17_train, r17_train, direct_train, direct_train,
    )
    soft_predict, _ = _component_prediction(
        SOFT_SPEC, data, train, predict,
        r17_train, r17_predict, direct_train, direct_predict,
    )
    hurdle_predict, _ = _component_prediction(
        HURDLE_SPEC, data, train, predict,
        r17_train, r17_predict, direct_train, direct_predict,
    )
    family_smokes = []
    for spec in (router_specs()[0], router_specs()[4], router_specs()[8], router_specs()[12], router_specs()[16]):
        result = fit_route(
            spec, data.targets[train], data.base[train], r17_train, direct_train,
            hurdle_train, soft_train,
        )
        prediction = apply_route(
            spec, result.selected_parameters, data.base[predict], r17_predict,
            direct_predict, hurdle_predict, soft_predict,
        )
        if prediction.shape != (32, 3) or not np.isfinite(prediction).all():
            raise IterativeTailRouterRunError("router family smoke coverage differs")
        family_smokes.append({"cycle": spec.cycle, "family": spec.family})
    payload = {
        "schema_version": "mal2026-iterative-tail-router-smoke-v4",
        "status": "completed",
        "gpu": 0,
        "train_count": len(train),
        "predict_count": len(predict),
        "neural_initial_state_hashes": list(neural.initial_state_hashes),
        "neural_final_state_hashes": list(neural.final_state_hashes),
        "soft_component": SOFT_SPEC.variant_id,
        "hurdle_component": HURDLE_SPEC.variant_id,
        "family_smokes": family_smokes,
        "validation_loaded": False,
        "average_target_used": False,
    }
    _write_json(PUBLIC_ROOT / "smoke.json", payload)
    return payload


__all__ = [
    "PUBLIC_ROOT", "RESTRICTED_ROOT", "IterativeTailRouterRunError",
    "aggregate_outer_results", "gate_decision", "gpu0_smoke", "run_outer_fold",
]
