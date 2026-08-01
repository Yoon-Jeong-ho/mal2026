"""Strict nested outer runner for v2 iterative-tail remediation.

Every outer worker excludes its held-out fold before regenerating R16
pseudo-targets, rebuilding the evidence-ridge challenger, selecting remediation
families on four inner OOF folds, and predicting the outer fold once.  Row-level
predictions stay under the restricted root; public results are aggregate-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mal2026.iterative_tail_metrics import compute_iterative_tail_metrics, metric_improvements
from mal2026.iterative_tail_models import CandidateSpec, fit_predict as fit_frozen_candidate
from mal2026.iterative_tail_remediation_models import (
    PREDECLARED_GRIDS,
    RemediationSpec,
    fit_predict as fit_remediation,
)
from mal2026.iterative_tail_remediation_protocol import (
    RUN_ID,
    RemediationProtocol,
    load_protocol,
    outer_inner_folds,
    validate_bound_inputs,
)
from mal2026.iterative_tail_runner import ExperimentData, load_experiment_data


PUBLIC_ROOT = Path("outputs/iterative-tail-remediation-v2") / RUN_ID
RESTRICTED_ROOT = Path("data/processed/restricted/iterative_tail_remediation_v2") / RUN_ID
SEED = 2026080102
DIRECT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


class RemediationRunError(RuntimeError):
    """Raised when nested isolation, output coverage, or lineage differs."""


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _indices(data: ExperimentData, folds: Sequence[int]) -> np.ndarray:
    return np.flatnonzero(np.isin(data.folds, np.asarray(folds, dtype=int)))


@dataclass(frozen=True)
class CandidateOOF:
    key: str
    family: str
    predictions: np.ndarray
    metrics: Mapping[str, Any]
    subvariant: Mapping[str, Any]


@dataclass(frozen=True)
class ChallengerPair:
    train: np.ndarray
    predict: np.ndarray


def _r16_spec(device: str) -> CandidateSpec:
    return CandidateSpec(
        family="joint_huber_ordinal",
        seed=SEED,
        device=device,
        hidden_dim=128,
        epochs=100,
        learning_rate=1e-3,
        dropout=0.1,
        huber_delta=1.0,
        ordinal_weight=0.5,
        max_correction=0.75,
    )


def _fit_r16_crossfit_teacher(
    data: ExperimentData,
    universe_folds: Sequence[int],
    *,
    device: str,
    forbidden_folds: Sequence[int],
    purpose: str,
) -> tuple[np.ndarray, list[Mapping[str, Any]]]:
    """Cross-fit an R16 teacher inside exactly the declared fold universe."""
    universe = tuple(int(fold) for fold in universe_folds)
    forbidden = tuple(int(fold) for fold in forbidden_folds)
    if len(universe) < 3 or set(universe) & set(forbidden):
        raise RemediationRunError("R16 teacher universe/forbidden folds differ")
    teacher = np.full_like(data.targets, np.nan, dtype=np.float32)
    consensus = data.evidence.view("consensus_disagreement")
    assert consensus is not None
    audit = []
    for heldout in universe:
        train_folds = tuple(fold for fold in universe if fold != heldout)
        train = _indices(data, train_folds)
        predict = _indices(data, (heldout,))
        if np.any(np.isin(data.folds[train], forbidden)) or np.any(np.isin(data.folds[predict], forbidden)):
            raise RemediationRunError("forbidden fold reached R16 teacher regeneration")
        result = fit_frozen_candidate(
            _r16_spec(device),
            data.embeddings[train], data.base[train], data.targets[train],
            data.embeddings[predict], data.base[predict],
            train_extra_features=consensus[train], predict_extra_features=consensus[predict],
        )
        teacher[predict] = result.predictions
        audit.append({
            "purpose": purpose,
            "heldout_fold": heldout,
            "train_folds": list(train_folds),
            "train_count": len(train),
            "predict_count": len(predict),
            "initial_state_hashes": list(result.initial_state_hashes),
            "final_state_hashes": list(result.final_state_hashes),
        })
    universe_indices = _indices(data, universe)
    forbidden_indices = _indices(data, forbidden)
    if not np.isfinite(teacher[universe_indices]).all() or np.isfinite(teacher[forbidden_indices]).any():
        raise RemediationRunError("R16 teacher coverage or holdout sealing differs")
    return teacher, audit


def regenerate_r16_teacher_oof(
    data: ExperimentData, outer_fold: int, *, device: str,
) -> tuple[np.ndarray, list[Mapping[str, Any]]]:
    """Regenerate the 3-of-4 teacher used only after selection freezes."""
    if outer_fold not in range(5):
        raise RemediationRunError("outer fold must be 0..4")
    inner_folds = outer_inner_folds(load_protocol(), outer_fold)
    return _fit_r16_crossfit_teacher(
        data, inner_folds, device=device, forbidden_folds=(outer_fold,),
        purpose="outer_refit_teacher_3_of_4",
    )


def regenerate_inner_selection_teacher(
    data: ExperimentData,
    outer_fold: int,
    inner_validation_fold: int,
    *,
    device: str,
) -> tuple[np.ndarray, list[Mapping[str, Any]]]:
    """Regenerate a 2-of-3 teacher excluding both selection holdouts."""
    inner_folds = outer_inner_folds(load_protocol(), outer_fold)
    if inner_validation_fold not in inner_folds:
        raise RemediationRunError("inner validation fold is outside outer-train universe")
    fit_folds = tuple(fold for fold in inner_folds if fold != inner_validation_fold)
    return _fit_r16_crossfit_teacher(
        data, fit_folds, device=device,
        forbidden_folds=(outer_fold, inner_validation_fold),
        purpose="inner_selection_teacher_2_of_3",
    )


def _ridge_pair(
    data: ExperimentData,
    train: np.ndarray,
    predict: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    device: str,
) -> ChallengerPair:
    evidence_hash = data.evidence.view("evidence_hash")
    assert evidence_hash is not None
    combined = np.concatenate((train, predict))
    result = fit_frozen_candidate(
        CandidateSpec(
            family="ridge_residual", seed=SEED, device=device,
            ridge_alpha=alpha, epochs=1,
        ),
        evidence_hash[train], data.base[train], targets[train],
        evidence_hash[combined], data.base[combined],
    )
    return ChallengerPair(
        train=result.predictions[: len(train)],
        predict=result.predictions[len(train) :],
    )


def _candidate_metrics(data: ExperimentData, outer_train: np.ndarray, prediction: np.ndarray) -> Mapping[str, Any]:
    if prediction.shape != (len(outer_train), 3) or not np.isfinite(prediction).all():
        raise RemediationRunError("inner OOF candidate coverage differs")
    return compute_iterative_tail_metrics(data.targets[outer_train], prediction)


def _remediation_subvariants() -> tuple[tuple[str, RemediationSpec], ...]:
    return (
        ("gated-delta", RemediationSpec("gated_delta", False)),
        ("weighted-isotonic-unweighted", RemediationSpec("weighted_isotonic", False)),
        ("weighted-isotonic-equal-band", RemediationSpec("weighted_isotonic", True)),
        ("piecewise-5knot-unweighted", RemediationSpec("piecewise_5knot", False)),
        ("piecewise-5knot-equal-band", RemediationSpec("piecewise_5knot", True)),
        ("tail-boundary", RemediationSpec("tail_boundary", False)),
    )


def _parameter_digest(value: Any) -> Mapping[str, Any]:
    """Return an aggregate-safe description of fitted split parameters.

    Isotonic x/y knots are derived from individual training predictions.  They
    are required in memory for inference but must never be copied to the public
    result tree.  A count plus deterministic digest is enough to audit refits.
    """
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return {"count": len(value), "sha256": sha256(encoded.encode("utf-8")).hexdigest()}
    return {"sha256": sha256(encoded.encode("utf-8")).hexdigest()}


def _public_subvariant(subvariant: Mapping[str, Any]) -> Mapping[str, Any]:
    """Strip row-derived calibration knots from public candidate metadata."""
    public: dict[str, Any] = {}
    for key, value in subvariant.items():
        if key == "split_parameters":
            public["split_parameter_fits"] = len(value)
            public["split_parameters_digest"] = _parameter_digest(value)["sha256"]
        elif key in {"selected_parameters", "member_refits"}:
            public[f"{key}_summary"] = _parameter_digest(value)
        else:
            public[key] = value
    return public


def _best_by_rmse(records: Sequence[CandidateOOF]) -> CandidateOOF:
    if not records:
        raise RemediationRunError("candidate family has no subvariants")
    return min(records, key=lambda record: (float(record.metrics["macro"]["rmse"]), record.key))


def _build_inner_candidates(
    data: ExperimentData,
    outer_fold: int,
    *,
    device: str,
) -> tuple[CandidateOOF, tuple[CandidateOOF, ...], Mapping[str, Any]]:
    inner_folds = outer_inner_folds(load_protocol(), outer_fold)
    outer_train = _indices(data, inner_folds)
    position = {int(index): offset for offset, index in enumerate(outer_train)}
    baseline_prediction = data.base[outer_train].astype(np.float64)
    baseline = CandidateOOF(
        key="base-identity", family="identity", predictions=baseline_prediction,
        metrics=_candidate_metrics(data, outer_train, baseline_prediction), subvariant={},
    )
    oof: dict[str, np.ndarray] = {"r17-raw": np.empty((len(outer_train), 3), dtype=np.float64)}
    subvariant_specs = _remediation_subvariants()
    for name, _ in subvariant_specs:
        oof[name] = np.empty((len(outer_train), 3), dtype=np.float64)
    for alpha in DIRECT_ALPHAS:
        oof[f"direct-ridge-alpha-{alpha:g}"] = np.empty((len(outer_train), 3), dtype=np.float64)

    split_parameters: dict[str, list[Mapping[str, Any]]] = {name: [] for name, _ in subvariant_specs}
    teacher_audits: list[Mapping[str, Any]] = []
    for inner_validation_fold in inner_folds:
        fit_folds = tuple(fold for fold in inner_folds if fold != inner_validation_fold)
        fit_indices = _indices(data, fit_folds)
        validation_indices = _indices(data, (inner_validation_fold,))
        if np.any(data.folds[fit_indices] == outer_fold) or np.any(data.folds[validation_indices] == outer_fold):
            raise RemediationRunError("outer fold reached inner candidate fitting")
        rows = np.asarray([position[int(index)] for index in validation_indices], dtype=int)
        split_teacher, split_teacher_audit = regenerate_inner_selection_teacher(
            data, outer_fold, inner_validation_fold, device=device,
        )
        if np.isfinite(split_teacher[validation_indices]).any():
            raise RemediationRunError("inner validation reached split-specific R16 teacher")
        teacher_audits.append({
            "inner_validation_fold": inner_validation_fold,
            "fit_folds": list(fit_folds),
            "crossfit_models": split_teacher_audit,
        })
        r17 = _ridge_pair(
            data, fit_indices, validation_indices, split_teacher,
            alpha=10.0, device=device,
        )
        oof["r17-raw"][rows] = r17.predict
        for name, spec in subvariant_specs:
            result = fit_remediation(
                spec,
                data.targets[fit_indices], data.base[fit_indices], r17.train,
                data.base[validation_indices], r17.predict,
            )
            oof[name][rows] = result.predictions
            split_parameters[name].append({
                "inner_validation_fold": inner_validation_fold,
                "train_objective": result.train_objective,
                "selected_parameters": list(result.selected_parameters),
            })
        for alpha in DIRECT_ALPHAS:
            direct = _ridge_pair(
                data, fit_indices, validation_indices, data.targets,
                alpha=alpha, device=device,
            )
            oof[f"direct-ridge-alpha-{alpha:g}"][rows] = direct.predict

    raw = CandidateOOF(
        key="r17-raw", family="nested-rebuilt-r17",
        predictions=oof["r17-raw"], metrics=_candidate_metrics(data, outer_train, oof["r17-raw"]),
        subvariant={"alpha": 10.0},
    )
    families: list[CandidateOOF] = [raw]
    gated = CandidateOOF(
        key="gated-delta", family="conditional-r17-delta",
        predictions=oof["gated-delta"], metrics=_candidate_metrics(data, outer_train, oof["gated-delta"]),
        subvariant={"spec": "gated_delta", "split_parameters": split_parameters["gated-delta"]},
    )
    families.append(gated)
    calibration_records = []
    for name in (
        "weighted-isotonic-unweighted", "weighted-isotonic-equal-band",
        "piecewise-5knot-unweighted", "piecewise-5knot-equal-band",
    ):
        calibration_records.append(CandidateOOF(
            key=name, family="weighted-isotonic-piecewise",
            predictions=oof[name], metrics=_candidate_metrics(data, outer_train, oof[name]),
            subvariant={"spec": name, "split_parameters": split_parameters[name]},
        ))
    families.append(_best_by_rmse(calibration_records))
    for name, family in (("tail-boundary", "tail-boundary"),):
        families.append(CandidateOOF(
            key=name, family=family, predictions=oof[name],
            metrics=_candidate_metrics(data, outer_train, oof[name]),
            subvariant={"spec": name, "split_parameters": split_parameters[name]},
        ))
    direct_records = []
    for alpha in DIRECT_ALPHAS:
        name = f"direct-ridge-alpha-{alpha:g}"
        direct_records.append(CandidateOOF(
            key=name, family="direct-evidence-ridge", predictions=oof[name],
            metrics=_candidate_metrics(data, outer_train, oof[name]),
            subvariant={"alpha": alpha},
        ))
    families.append(_best_by_rmse(direct_records))
    audit = {
        "inner_folds": list(inner_folds),
        "outer_train_count": len(outer_train),
        "calibration_subvariants": [
            {"key": record.key, "macro_rmse": record.metrics["macro"]["rmse"]}
            for record in calibration_records
        ],
        "direct_ridge_subvariants": [
            {"key": record.key, "macro_rmse": record.metrics["macro"]["rmse"]}
            for record in direct_records
        ],
        "inner_selection_teacher_regeneration": teacher_audits,
        "inner_validation_gold_used_for_teacher_or_r17_fit": False,
    }
    return baseline, tuple(families), audit


def _ensemble_candidate(
    data: ExperimentData,
    outer_train: np.ndarray,
    baseline: CandidateOOF,
    candidates: Sequence[CandidateOOF],
    protocol: RemediationProtocol,
) -> tuple[CandidateOOF | None, Mapping[str, Any]]:
    eligible = []
    for candidate in candidates:
        decision = _gate_decision(protocol, baseline.metrics, candidate.metrics)
        if decision["promote"]:
            eligible.append(candidate)
    eligible.sort(key=lambda record: (float(record.metrics["macro"]["rmse"]), record.family, record.key))
    structurally_distinct = []
    seen_families = set()
    for record in eligible:
        if record.family not in seen_families:
            structurally_distinct.append(record)
            seen_families.add(record.family)
        if len(structurally_distinct) == 2:
            break
    if len(structurally_distinct) < 2:
        return None, {
            "eligible_candidates": [record.key for record in eligible],
            "members": [record.key for record in structurally_distinct],
            "fallback": "fewer-than-two-eligible-structurally-distinct-candidates",
        }
    left, right = structurally_distinct
    best = None
    for right_weight in PREDECLARED_GRIDS["blend_weight"]:
        prediction = (1.0 - right_weight) * left.predictions + right_weight * right.predictions
        metrics = _candidate_metrics(data, outer_train, prediction)
        key = (float(metrics["macro"]["rmse"]), float(right_weight))
        if best is None or key < best[0]:
            best = (key, prediction, metrics, float(right_weight))
    assert best is not None
    candidate = CandidateOOF(
        key="top-two-ensemble", family="top-two-ensemble",
        predictions=best[1], metrics=best[2],
        subvariant={"members": [left.key, right.key], "right_weight": best[3]},
    )
    return candidate, {
        "eligible_candidates": [record.key for record in eligible],
        "members": [left.key, right.key], "right_weight": best[3], "fallback": None,
    }


def _gate_decision(
    protocol: RemediationProtocol,
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Apply the seven config-bound gates against the fixed baseline."""
    config = protocol.raw["inner_promotion_gate"]
    delta = metric_improvements(baseline_metrics, candidate_metrics)
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
        "low_tail_improves": finite and (not config["low_tail_must_improve"] or delta["low_tail_rmse"] > 0.0),
        "high_tail_improves": finite and (not config["high_tail_must_improve"] or delta["high_tail_rmse"] > 0.0),
        "gold_3_4_balanced_accuracy_improvement": finite and delta["gold_3_4_balanced_accuracy"] >= config["gold_3_4_balanced_accuracy_min_improvement"],
        "axis_rmse_worsening_bound": finite and all(
            value >= -config["max_axis_rmse_worsening"] for value in delta["axis_rmse"].values()
        ),
        "macro_spearman_fall_bound": finite and delta["spearman"] >= -config["max_macro_spearman_fall"],
    }
    if config["require_finite_metrics"] and not finite:
        gates = {name: False for name in gates}
    return {
        "promote": config["operator"] == "AND" and all(gates.values()),
        "gates": gates,
        "improvements": delta,
        "finite_metrics": finite,
        "score1_used_for_promotion": False,
        "comparison_reference": "base-identity",
    }


def _select_candidate(
    protocol: RemediationProtocol,
    baseline: CandidateOOF,
    candidates: Sequence[CandidateOOF],
    ensemble: CandidateOOF | None,
) -> tuple[CandidateOOF, list[Mapping[str, Any]]]:
    decisions = []
    sequence = [*candidates, *(() if ensemble is None else (ensemble,))]
    eligible: list[tuple[float, int, CandidateOOF]] = []
    for registered_order, candidate in enumerate(sequence, start=2):
        decision = _gate_decision(protocol, baseline.metrics, candidate.metrics)
        decisions.append({
            "candidate": candidate.key,
            "family": candidate.family,
            "registered_order": registered_order,
            **decision,
        })
        if decision["promote"]:
            eligible.append((float(candidate.metrics["macro"]["rmse"]), registered_order, candidate))
    selected = min(eligible, key=lambda item: (item[0], item[1], item[2].key))[2] if eligible else baseline
    decisions = [{**record, "selected": record["candidate"] == selected.key} for record in decisions]
    return selected, decisions


def _outer_refit_prediction(
    data: ExperimentData,
    outer_fold: int,
    teacher: np.ndarray | None,
    selected: CandidateOOF,
    candidates: Sequence[CandidateOOF],
    *,
    device: str,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    inner_folds = outer_inner_folds(load_protocol(), outer_fold)
    train = _indices(data, inner_folds)
    predict = _indices(data, (outer_fold,))
    if selected.key == "base-identity":
        return data.base[predict].copy(), {"family": "identity"}
    by_key = {candidate.key: candidate for candidate in candidates}
    r17: ChallengerPair | None = None

    def require_r17() -> ChallengerPair:
        nonlocal r17
        if teacher is None:
            raise RemediationRunError("selected candidate requires missing outer-refit R16 teacher")
        if r17 is None:
            r17 = _ridge_pair(data, train, predict, teacher, alpha=10.0, device=device)
        return r17

    def refit_one(record: CandidateOOF) -> tuple[np.ndarray, Mapping[str, Any]]:
        if record.key == "r17-raw":
            challenger = require_r17()
            return challenger.predict, {"family": record.family, "alpha": 10.0}
        if record.family == "direct-evidence-ridge":
            alpha = float(record.subvariant["alpha"])
            direct = _ridge_pair(data, train, predict, data.targets, alpha=alpha, device=device)
            return direct.predict, {"family": record.family, "alpha": alpha}
        spec_name = str(record.subvariant.get("spec", record.key))
        spec_map = dict(_remediation_subvariants())
        if spec_name not in spec_map:
            raise RemediationRunError(f"cannot refit selected candidate: {record.key}")
        challenger = require_r17()
        result = fit_remediation(
            spec_map[spec_name],
            data.targets[train], data.base[train], challenger.train,
            data.base[predict], challenger.predict,
        )
        return result.predictions, {
            "family": record.family,
            "spec": spec_name,
            "train_objective": result.train_objective,
            "selected_parameters_summary": _parameter_digest(result.selected_parameters),
        }

    if selected.family != "top-two-ensemble":
        return refit_one(selected)
    member_keys = list(selected.subvariant["members"])
    if len(member_keys) != 2 or any(key not in by_key for key in member_keys):
        raise RemediationRunError("ensemble member binding differs")
    left, left_audit = refit_one(by_key[member_keys[0]])
    right, right_audit = refit_one(by_key[member_keys[1]])
    right_weight = float(selected.subvariant["right_weight"])
    prediction = np.clip((1.0 - right_weight) * left + right_weight * right, 1.0, 5.0)
    return prediction, {
        "family": "top-two-ensemble", "members": member_keys,
        "right_weight": right_weight, "member_refits": [left_audit, right_audit],
    }


def _selected_requires_r17(selected: CandidateOOF, candidates: Sequence[CandidateOOF]) -> bool:
    if selected.key == "base-identity" or selected.family == "direct-evidence-ridge":
        return False
    if selected.family != "top-two-ensemble":
        return True
    by_key = {candidate.key: candidate for candidate in candidates}
    return any(by_key[key].family != "direct-evidence-ridge" for key in selected.subvariant["members"])


def run_outer_fold(
    outer_fold: int,
    *,
    device: str,
    protocol: RemediationProtocol | None = None,
) -> Mapping[str, Any]:
    """Run one fully sealed outer fold and persist aggregate/restricted outputs."""
    protocol = protocol or load_protocol()
    validate_bound_inputs(protocol)
    data = load_experiment_data()
    if outer_fold not in protocol.raw["nested_selection"]["outer_folds"]:
        raise RemediationRunError("outer fold is not registered")
    baseline, candidates, inner_audit = _build_inner_candidates(
        data, outer_fold, device=device,
    )
    outer_train = _indices(data, outer_inner_folds(protocol, outer_fold))
    ensemble, ensemble_audit = _ensemble_candidate(data, outer_train, baseline, candidates, protocol)
    selected, decisions = _select_candidate(protocol, baseline, candidates, ensemble)

    # Selection is frozen before the outer-refit teacher is ever generated.
    if _selected_requires_r17(selected, candidates):
        teacher, teacher_audit = regenerate_r16_teacher_oof(data, outer_fold, device=device)
    else:
        teacher = None
        teacher_audit = [{
            "purpose": "outer_refit_teacher_3_of_4",
            "status": "not_required_by_frozen_selection",
        }]

    # Selection is now frozen. Only after this point is the outer prediction
    # produced; outer gold is consumed later solely for aggregate metrics.
    prediction, refit_audit = _outer_refit_prediction(
        data, outer_fold, teacher, selected, candidates, device=device,
    )
    outer_indices = _indices(data, (outer_fold,))
    if prediction.shape != (400, 3) or not np.isfinite(prediction).all():
        raise RemediationRunError("outer prediction shape differs")
    metrics = compute_iterative_tail_metrics(data.targets[outer_indices], prediction)
    baseline_metrics = compute_iterative_tail_metrics(data.targets[outer_indices], data.base[outer_indices])
    restricted_path = RESTRICTED_ROOT / f"outer-{outer_fold}" / "predictions.jsonl"
    _write_jsonl(restricted_path, [
        {
            "source_id": data.source_ids[index],
            "outer_fold": outer_fold,
            "baseline_prediction": [float(value) for value in data.base[index]],
            "selected_prediction": [float(value) for value in row],
        }
        for index, row in zip(outer_indices, prediction, strict=True)
    ])
    payload = {
        "schema_version": "mal2026-iterative-tail-remediation-outer-v2",
        "status": "completed",
        "outer_fold": outer_fold,
        "outer_train_count": 1600,
        "outer_validation_count": 400,
        "inner_fold_count": 4,
        "selected_candidate": selected.key,
        "fell_back_to_baseline": selected.key == baseline.key,
        "candidate_order": [candidate.key for candidate in candidates],
        "candidate_summaries": [
            {
                "key": candidate.key,
                "family": candidate.family,
                "macro_rmse": candidate.metrics["macro"]["rmse"],
                "equal_group_rmse": candidate.metrics["macro"]["equal_group_rmse"],
                "low_tail_rmse": candidate.metrics["macro"]["low_tail_rmse"],
                "high_tail_rmse": candidate.metrics["macro"]["high_tail_rmse"],
                "gold_3_4_balanced_accuracy": candidate.metrics["macro"]["gold_3_4_balanced_accuracy"],
                "subvariant": _public_subvariant(candidate.subvariant),
            }
            for candidate in candidates
        ],
        "baseline_relative_candidate_decisions": decisions,
        "sequential_incumbent_tournament": False,
        "ensemble_audit": ensemble_audit,
        "inner_audit": inner_audit,
        "outer_refit_teacher_regeneration": teacher_audit,
        "outer_refit": _public_subvariant(refit_audit),
        "baseline_metrics": baseline_metrics,
        "selected_metrics": metrics,
        "restricted_prediction_sha256": _sha256(restricted_path),
        "outer_gold_used_before_selection_or_predict": False,
        "historical_r17_used_for_selection_or_fitting": False,
        "validation_loaded": False,
        "average_target_used": False,
    }
    _write_json(PUBLIC_ROOT / f"outer-{outer_fold}" / "result.json", payload)
    return payload


def _read_outer_predictions(data: ExperimentData) -> tuple[np.ndarray, np.ndarray, list[Mapping[str, Any]]]:
    baseline = np.full_like(data.base, np.nan, dtype=np.float64)
    selected = np.full_like(data.base, np.nan, dtype=np.float64)
    audits = []
    seen = set()
    id_to_index = {source_id: index for index, source_id in enumerate(data.source_ids)}
    for outer_fold in range(5):
        result_path = PUBLIC_ROOT / f"outer-{outer_fold}" / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        prediction_path = RESTRICTED_ROOT / f"outer-{outer_fold}" / "predictions.jsonl"
        if result.get("status") != "completed" or result.get("restricted_prediction_sha256") != _sha256(prediction_path):
            raise RemediationRunError("outer result binding differs")
        with prediction_path.open(encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                if set(raw) != {"source_id", "outer_fold", "baseline_prediction", "selected_prediction"}:
                    raise RemediationRunError("outer restricted row schema differs")
                source_id = raw["source_id"]
                if source_id in seen or source_id not in id_to_index or raw["outer_fold"] != outer_fold:
                    raise RemediationRunError("outer restricted population differs")
                index = id_to_index[source_id]
                if int(data.folds[index]) != outer_fold:
                    raise RemediationRunError("outer prediction fold differs")
                baseline[index] = raw["baseline_prediction"]
                selected[index] = raw["selected_prediction"]
                seen.add(source_id)
        audits.append({
            "outer_fold": outer_fold,
            "selected_candidate": result["selected_candidate"],
            "fell_back_to_baseline": result["fell_back_to_baseline"],
            "result_sha256": _sha256(result_path),
            "prediction_sha256": _sha256(prediction_path),
        })
    if len(seen) != 2000 or not np.isfinite(baseline).all() or not np.isfinite(selected).all():
        raise RemediationRunError("nested outer coverage differs")
    return baseline, selected, audits


def _bootstrap_macro_rmse(
    truth: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = SEED,
) -> Mapping[str, Any]:
    base_squared = np.square(truth.astype(np.float64) - baseline.astype(np.float64))
    candidate_squared = np.square(truth.astype(np.float64) - candidate.astype(np.float64))
    rng = np.random.default_rng(seed)
    improvement = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 200):
        stop = min(resamples, start + 200)
        sample = rng.integers(0, len(truth), size=(stop - start, len(truth)))
        base_rmse = np.sqrt(base_squared[sample].mean(axis=1)).mean(axis=1)
        candidate_rmse = np.sqrt(candidate_squared[sample].mean(axis=1)).mean(axis=1)
        improvement[start:stop] = base_rmse - candidate_rmse
    lower, upper = (float(value) for value in np.quantile(improvement, (0.025, 0.975)))
    estimate = float(
        compute_iterative_tail_metrics(truth, baseline)["macro"]["rmse"]
        - compute_iterative_tail_metrics(truth, candidate)["macro"]["rmse"]
    )
    return {
        "resamples": resamples,
        "seed": seed,
        "confidence": 0.95,
        "improvement_direction": "positive_means_candidate_better",
        "improvement_ci": {"estimate": estimate, "lower": lower, "upper": upper},
        "candidate_minus_baseline_ci": {"estimate": -estimate, "lower": -upper, "upper": -lower},
    }


def aggregate_outer_results(protocol: RemediationProtocol | None = None) -> Mapping[str, Any]:
    protocol = protocol or load_protocol()
    validate_bound_inputs(protocol)
    data = load_experiment_data()
    baseline, selected, audits = _read_outer_predictions(data)
    # Prove the nested baseline is the exact fixed R0 OOF matrix.
    if not np.allclose(baseline, data.base.astype(np.float64), rtol=0.0, atol=1e-7):
        raise RemediationRunError("nested baseline differs from exact R0 OOF")
    baseline_metrics = compute_iterative_tail_metrics(data.targets, baseline)
    selected_metrics = compute_iterative_tail_metrics(data.targets, selected)
    bootstrap = _bootstrap_macro_rmse(data.targets, baseline, selected)
    improvement = float(baseline_metrics["macro"]["rmse"] - selected_metrics["macro"]["rmse"])
    final_gates = {
        "macro_rmse_improvement_at_least_0_01": improvement >= 0.01,
        "candidate_minus_baseline_rmse_ci_upper_below_zero": bootstrap["candidate_minus_baseline_ci"]["upper"] < 0.0,
    }
    final_pass = all(final_gates.values())
    payload = {
        "schema_version": "mal2026-iterative-tail-remediation-aggregate-v2",
        "status": "completed",
        "record_count": 2000,
        "outer_fold_count": 5,
        "inner_fold_count": 4,
        "outer_audits": audits,
        "baseline_metrics": baseline_metrics,
        "selected_metrics": selected_metrics,
        "macro_rmse_improvement": improvement,
        "paired_bootstrap": bootstrap,
        "final_gates": final_gates,
        "final_gate_pass": final_pass,
        "final_selection": "nested-remediation-procedure" if final_pass else "exact-r0-oof-baseline-fallback",
        "historical_r17_used_for_selection_or_fitting": False,
        "posthoc_selection_on_concatenated_outer_predictions": False,
        "validation_loaded": False,
        "average_target_used": False,
    }
    aggregate_path = PUBLIC_ROOT / "aggregate.json"
    _write_json(aggregate_path, payload)
    completion = {
        "schema_version": "mal2026-iterative-tail-remediation-completion-v2",
        "status": "completed_final_gate_pass" if final_pass else "completed_no_promotion_baseline_retained",
        "aggregate_sha256": _sha256(aggregate_path),
        "final_gate_pass": final_pass,
        "final_selection": payload["final_selection"],
        "gpu_scope": [0, 1, 2, 3],
        "external_api_calls": 0,
        "validation_loaded": False,
        "average_target_used": False,
    }
    _write_json(PUBLIC_ROOT / "completion.json", completion)
    return payload


def gpu0_smoke(*, device: str = "cuda:0") -> Mapping[str, Any]:
    """Run the smallest real R16/remediation integration smoke on GPU0."""
    protocol = load_protocol()
    validate_bound_inputs(protocol)
    data = load_experiment_data()
    consensus = data.evidence.view("consensus_disagreement")
    evidence_hash = data.evidence.view("evidence_hash")
    assert consensus is not None and evidence_hash is not None
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
    teacher = data.targets.copy()
    ridge = _ridge_pair(data, train, predict, teacher, alpha=10.0, device=device)
    remediation = fit_remediation(
        RemediationSpec("gated_delta"),
        data.targets[train], data.base[train], ridge.train,
        data.base[predict], ridge.predict,
    )
    if neural.predictions.shape != (32, 3) or remediation.predictions.shape != (32, 3):
        raise RemediationRunError("GPU0 smoke output shape differs")
    payload = {
        "schema_version": "mal2026-iterative-tail-remediation-smoke-v2",
        "status": "completed",
        "gpu": 0,
        "train_count": 128,
        "predict_count": 32,
        "neural_initial_state_hashes": list(neural.initial_state_hashes),
        "neural_final_state_hashes": list(neural.final_state_hashes),
        "remediation_family": remediation.family,
        "validation_loaded": False,
        "average_target_used": False,
    }
    _write_json(PUBLIC_ROOT / "smoke.json", payload)
    return payload


__all__ = [
    "PUBLIC_ROOT", "RESTRICTED_ROOT", "RemediationRunError", "aggregate_outer_results",
    "gpu0_smoke", "regenerate_r16_teacher_oof", "run_outer_fold",
]
