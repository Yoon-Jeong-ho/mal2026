"""Twenty fixed cold-start candidates for tail-score remediation cycles.

All candidates are fit from scratch on train-only three-axis targets.  The
module performs no file I/O and returns only in-memory predictions plus an
aggregate-safe audit without learned row-level values or coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


SEED = 2026080103
FAMILIES = (
    "soft_routed_residual",
    "pareto_routed_stack",
    "group_dro_ridge",
    "selective_hurdle",
    "final_ordinal_stack",
)


@dataclass(frozen=True)
class CycleSpec:
    cycle: int
    family: str
    variant_id: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class CycleResult:
    predictions: np.ndarray
    audit: Mapping[str, Any]


def _spec(cycle: int, family: str, variant: int, **parameters: Any) -> CycleSpec:
    return CycleSpec(cycle, family, f"{family}-v{variant}", MappingProxyType(parameters))


def cycle_specs() -> tuple[CycleSpec, ...]:
    """Return the exact fixed inventory: five families, four variants each."""
    return (
        _spec(1, "soft_routed_residual", 1, alpha=10.0, cap=0.10, temperature=0.25, anchor=1.0),
        _spec(2, "soft_routed_residual", 2, alpha=30.0, cap=0.15, temperature=0.35, anchor=2.0),
        _spec(3, "soft_routed_residual", 3, alpha=100.0, cap=0.20, temperature=0.50, anchor=4.0),
        _spec(4, "soft_routed_residual", 4, alpha=300.0, cap=0.25, temperature=0.75, anchor=8.0),
        _spec(5, "pareto_routed_stack", 1, r17_low=0.00, r17_high=0.25, direct_low=0.10, direct_high=0.10, temperature=0.25),
        _spec(6, "pareto_routed_stack", 2, r17_low=0.00, r17_high=0.40, direct_low=0.15, direct_high=0.10, temperature=0.35),
        _spec(7, "pareto_routed_stack", 3, r17_low=0.05, r17_high=0.30, direct_low=0.15, direct_high=0.15, temperature=0.50),
        _spec(8, "pareto_routed_stack", 4, r17_low=0.00, r17_high=0.50, direct_low=0.25, direct_high=0.00, temperature=0.75),
        _spec(9, "group_dro_ridge", 1, alpha=30.0, cap=0.10, eta=0.25, iterations=2, group_weight_cap=3.0),
        _spec(10, "group_dro_ridge", 2, alpha=100.0, cap=0.15, eta=0.20, iterations=3, group_weight_cap=5.0),
        _spec(11, "group_dro_ridge", 3, alpha=300.0, cap=0.20, eta=0.15, iterations=4, group_weight_cap=7.0),
        _spec(12, "group_dro_ridge", 4, alpha=1000.0, cap=0.25, eta=0.10, iterations=5, group_weight_cap=10.0),
        _spec(13, "selective_hurdle", 1, evidence_dims=16, confidence=0.60, cap=0.10, logistic_l2=0.10, steps=40, learning_rate=0.10),
        _spec(14, "selective_hurdle", 2, evidence_dims=32, confidence=0.70, cap=0.15, logistic_l2=0.10, steps=50, learning_rate=0.08),
        _spec(15, "selective_hurdle", 3, evidence_dims=64, confidence=0.80, cap=0.20, logistic_l2=0.30, steps=60, learning_rate=0.06),
        _spec(16, "selective_hurdle", 4, evidence_dims=96, confidence=0.90, cap=0.25, logistic_l2=1.00, steps=70, learning_rate=0.05),
        _spec(17, "final_ordinal_stack", 1, ordinal_weight=0.10, ordinal_mix=0.10, cap=0.10, epochs=40, learning_rate=0.03),
        _spec(18, "final_ordinal_stack", 2, ordinal_weight=0.25, ordinal_mix=0.20, cap=0.15, epochs=50, learning_rate=0.02),
        _spec(19, "final_ordinal_stack", 3, ordinal_weight=0.50, ordinal_mix=0.30, cap=0.20, epochs=60, learning_rate=0.015),
        _spec(20, "final_ordinal_stack", 4, ordinal_weight=0.75, ordinal_mix=0.40, cap=0.25, epochs=70, learning_rate=0.01),
    )


_REGISTERED = {spec.variant_id: spec for spec in cycle_specs()}


def _matrix(value: Any, name: str, *, bounded: bool = True) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have exactly three axis columns; average is forbidden")
    if len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty and finite")
    if bounded and (np.any(result < 1.0) or np.any(result > 5.0)):
        raise ValueError(f"{name} must be within [1, 5]")
    return result


def _evidence(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or len(result) == 0 or result.shape[1] == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite feature matrix")
    return result


def _validate_spec(spec: CycleSpec) -> None:
    registered = _REGISTERED.get(spec.variant_id)
    if registered is None or spec.cycle != registered.cycle or spec.family != registered.family:
        raise ValueError("cycle spec is not in the fixed inventory")
    if dict(spec.parameters) != dict(registered.parameters):
        raise ValueError("cycle parameters differ from the fixed inventory")


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _bands(gold: np.ndarray) -> np.ndarray:
    return np.floor(gold + 0.5).astype(np.int64).clip(1, 5)


def _band_mse(gold: np.ndarray, prediction: np.ndarray) -> tuple[tuple[float | None, ...], ...]:
    band = _bands(gold)
    values = []
    for axis in range(3):
        axis_values = []
        for label in range(1, 6):
            mask = band[:, axis] == label
            axis_values.append(float(np.mean(np.square(gold[mask, axis] - prediction[mask, axis]))) if np.any(mask) else None)
        values.append(tuple(axis_values))
    return tuple(values)


def _macro_rmse(gold: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean(np.square(gold - prediction), axis=0))))


def _weighted_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    w = np.ones(len(train_x), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.mean()
    total = w.sum()
    x_mean = np.sum(train_x * w[:, None], axis=0) / total
    y_mean = float(np.sum(train_y * w) / total)
    xc, yc = train_x - x_mean, train_y - y_mean
    root = np.sqrt(w)
    xw, yw = xc * root[:, None], yc * root
    if xw.shape[1] <= xw.shape[0]:
        system = xw.T @ xw + alpha * np.eye(xw.shape[1])
        coefficient = np.linalg.solve(system, xw.T @ yw)
    else:
        system = xw @ xw.T + alpha * np.eye(xw.shape[0])
        coefficient = xw.T @ np.linalg.solve(system, yw)
    return (test_x - x_mean) @ coefficient + y_mean


def _route_basis(score: np.ndarray, temperature: float) -> np.ndarray:
    low = _sigmoid((2.5 - score) / temperature)
    above_low = _sigmoid((score - 2.5) / temperature)
    below_mid = _sigmoid((3.5 - score) / temperature)
    above_mid = _sigmoid((score - 3.5) / temperature)
    below_high = _sigmoid((4.5 - score) / temperature)
    high = _sigmoid((score - 4.5) / temperature)
    return np.column_stack((low, above_low * below_mid, above_mid * below_high, high))


def _soft_routed(
    parameters: Mapping[str, Any], gold: np.ndarray, base: np.ndarray, r17: np.ndarray,
    direct: np.ndarray, evidence: np.ndarray, test_base: np.ndarray, test_r17: np.ndarray,
    test_direct: np.ndarray, test_evidence: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    evidence, test_evidence = _standardize(evidence, test_evidence)
    prediction = np.empty_like(test_base)
    train_prediction = np.empty_like(base)
    cap, temp = float(parameters["cap"]), float(parameters["temperature"])
    for axis in range(3):
        route = _route_basis(base[:, axis], temp)
        test_route = _route_basis(test_base[:, axis], temp)
        delta = np.column_stack((r17[:, axis] - base[:, axis], direct[:, axis] - base[:, axis]))
        test_delta = np.column_stack((test_r17[:, axis] - test_base[:, axis], test_direct[:, axis] - test_base[:, axis]))
        x = np.concatenate((evidence, route, route * delta[:, :1], route * delta[:, 1:2]), axis=1)
        tx = np.concatenate((test_evidence, test_route, test_route * test_delta[:, :1], test_route * test_delta[:, 1:2]), axis=1)
        combined = np.concatenate((x, tx), axis=0)
        raw = _weighted_ridge(x, gold[:, axis] - base[:, axis], combined, alpha=float(parameters["alpha"]))
        correction = np.clip(raw / (1.0 + float(parameters["anchor"])), -cap, cap)
        train_prediction[:, axis] = np.clip(base[:, axis] + correction[: len(base)], 1.0, 5.0)
        prediction[:, axis] = np.clip(test_base[:, axis] + correction[len(base) :], 1.0, 5.0)
    return prediction, {"train_macro_rmse": _macro_rmse(gold, train_prediction), "correction_cap": cap}


def _pareto_stack(
    parameters: Mapping[str, Any], gold: np.ndarray, base: np.ndarray, r17: np.ndarray,
    direct: np.ndarray, test_base: np.ndarray, test_r17: np.ndarray, test_direct: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    temp = float(parameters["temperature"])

    def blend(b: np.ndarray, r: np.ndarray, d: np.ndarray) -> np.ndarray:
        high = _sigmoid((b - 3.5) / temp)
        wr = float(parameters["r17_low"]) * (1.0 - high) + float(parameters["r17_high"]) * high
        wd = float(parameters["direct_low"]) * (1.0 - high) + float(parameters["direct_high"]) * high
        if np.any(wr < 0) or np.any(wd < 0) or np.any(wr + wd > 1.0 + 1e-12):
            raise ValueError("pareto blend weights must be nonnegative and sum to at most one")
        return np.clip((1.0 - wr - wd) * b + wr * r + wd * d, 1.0, 5.0)

    candidate = blend(base, r17, direct)
    baseline_loss, candidate_loss = _band_mse(gold, base), _band_mse(gold, candidate)
    degradation = []
    feasible = True
    for axis in range(3):
        for label in range(5):
            if baseline_loss[axis][label] is not None:
                delta = float(candidate_loss[axis][label]) - float(baseline_loss[axis][label])
                degradation.append(delta)
                feasible = feasible and delta <= 1e-12
    output = blend(test_base, test_r17, test_direct) if feasible else test_base.copy()
    return output, {"pareto_feasible": feasible, "max_band_mse_degradation": max(degradation, default=0.0),
                    "fallback_identity": not feasible}


def _group_dro(
    parameters: Mapping[str, Any], gold: np.ndarray, base: np.ndarray, evidence: np.ndarray,
    test_base: np.ndarray, test_evidence: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    evidence, test_evidence = _standardize(evidence, test_evidence)
    x = np.concatenate((evidence, base), axis=1)
    tx = np.concatenate((test_evidence, test_base), axis=1)
    prediction, train_prediction = np.empty_like(test_base), np.empty_like(base)
    band = _bands(gold)
    final_group_weights = []
    for axis in range(3):
        group_weight = np.ones(5, dtype=np.float64)
        for _ in range(int(parameters["iterations"])):
            row_weight = group_weight[band[:, axis] - 1]
            combined = np.concatenate((x, tx), axis=0)
            correction = _weighted_ridge(
                x, gold[:, axis] - base[:, axis], combined,
                alpha=float(parameters["alpha"]), weights=row_weight,
            )
            train_candidate = np.clip(base[:, axis] + np.clip(correction[: len(base)], -float(parameters["cap"]), float(parameters["cap"])), 1.0, 5.0)
            losses = np.zeros(5, dtype=np.float64)
            for label in range(1, 6):
                mask = band[:, axis] == label
                losses[label - 1] = np.mean(np.square(gold[mask, axis] - train_candidate[mask])) if np.any(mask) else 0.0
            scale = max(float(losses.mean()), 1e-8)
            group_weight *= np.exp(np.clip(float(parameters["eta"]) * losses / scale, 0.0, 10.0))
            group_weight = np.minimum(group_weight / group_weight.mean(), float(parameters["group_weight_cap"]))
            group_weight /= group_weight.mean()
        final_row_weight = group_weight[band[:, axis] - 1]
        coefficient_prediction = _weighted_ridge(
            x, gold[:, axis] - base[:, axis], np.concatenate((x, tx), axis=0),
            alpha=float(parameters["alpha"]), weights=final_row_weight,
        )
        cap = float(parameters["cap"])
        train_prediction[:, axis] = np.clip(base[:, axis] + np.clip(coefficient_prediction[: len(base)], -cap, cap), 1.0, 5.0)
        prediction[:, axis] = np.clip(test_base[:, axis] + np.clip(coefficient_prediction[len(base) :], -cap, cap), 1.0, 5.0)
        final_group_weights.append(tuple(float(value) for value in group_weight))
    return prediction, {"train_macro_rmse": _macro_rmse(gold, train_prediction),
                        "final_group_weights": tuple(final_group_weights)}


def _logistic_fit(x: np.ndarray, target: np.ndarray, *, steps: int, learning_rate: float, l2: float) -> np.ndarray:
    y = target.astype(np.float64)
    positives, negatives = max(1, int(y.sum())), max(1, int(len(y) - y.sum()))
    weight = np.where(y == 1, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
    coefficient = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(steps):
        probability = _sigmoid(x @ coefficient)
        gradient = x.T @ (weight * (probability - y)) / len(y) + l2 * coefficient
        coefficient -= learning_rate * gradient
    return coefficient


def _hurdle_apply(
    x: np.ndarray, base: np.ndarray, coefficients: tuple[np.ndarray, ...], experts: tuple[float, ...],
    *, confidence: float, cap: float,
) -> np.ndarray:
    probabilities = np.column_stack([_sigmoid(x @ coefficient) for coefficient in coefficients])
    output = base.copy()
    for row in range(len(base)):
        for axis in range(3):
            offset = axis * 4
            options = [
                (probabilities[row, offset], experts[offset]),
                (probabilities[row, offset + 1], experts[offset + 1]),
                (probabilities[row, offset + 3], experts[offset + 3]),
            ]
            mid_probability = probabilities[row, offset + 2]
            mid_confidence = max(mid_probability, 1.0 - mid_probability)
            if 2.5 <= base[row, axis] <= 4.5:
                mid_target = 4.0 if mid_probability >= 0.5 else 3.0
                options.append((mid_confidence, float(np.clip(mid_target - base[row, axis], -cap, cap))))
            best_confidence, correction = max(options, key=lambda item: item[0])
            if best_confidence >= confidence:
                output[row, axis] = np.clip(base[row, axis] + np.clip(correction, -cap, cap), 1.0, 5.0)
    return output


def _selective_hurdle(
    parameters: Mapping[str, Any], gold: np.ndarray, base: np.ndarray, r17: np.ndarray,
    direct: np.ndarray, evidence: np.ndarray, test_base: np.ndarray, test_r17: np.ndarray,
    test_direct: np.ndarray, test_evidence: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    dims = min(int(parameters["evidence_dims"]), evidence.shape[1])
    raw = np.concatenate((base, r17, direct, r17 - base, direct - base, evidence[:, :dims]), axis=1)
    test_raw = np.concatenate((test_base, test_r17, test_direct, test_r17 - test_base,
                               test_direct - test_base, test_evidence[:, :dims]), axis=1)
    raw, test_raw = _standardize(raw, test_raw)
    x = np.column_stack((np.ones(len(raw)), raw))
    tx = np.column_stack((np.ones(len(test_raw)), test_raw))
    band = _bands(gold)
    coefficients = []
    experts = []
    head_losses = []
    for axis in range(3):
        targets = (
            band[:, axis] == 1,
            band[:, axis] == 2,
            band[:, axis] == 4,
            band[:, axis] == 5,
        )
        masks = (band[:, axis] == 1, band[:, axis] == 2, np.isin(band[:, axis], (3, 4)), band[:, axis] == 5)
        for head, (target, mask) in enumerate(zip(targets, masks, strict=True)):
            fit_mask = mask if head == 2 else np.ones(len(gold), dtype=bool)
            coefficient = _logistic_fit(
                x[fit_mask], target[fit_mask], steps=int(parameters["steps"]),
                learning_rate=float(parameters["learning_rate"]), l2=float(parameters["logistic_l2"]),
            )
            coefficients.append(coefficient)
            probability = _sigmoid(x[fit_mask] @ coefficient)
            target_float = target[fit_mask].astype(float)
            head_losses.append(float(-np.mean(target_float * np.log(probability + 1e-9)
                                               + (1.0 - target_float) * np.log(1.0 - probability + 1e-9))))
            if head == 2:
                experts.append(0.0)
            else:
                relevant = band[:, axis] == (head + 1 if head < 2 else 5)
                experts.append(float(np.mean(gold[relevant, axis] - base[relevant, axis])) if np.any(relevant) else 0.0)
    coefficient_tuple, expert_tuple = tuple(coefficients), tuple(experts)
    output = _hurdle_apply(tx, test_base, coefficient_tuple, expert_tuple,
                           confidence=float(parameters["confidence"]), cap=float(parameters["cap"]))
    train_prediction = _hurdle_apply(x, base, coefficient_tuple, expert_tuple,
                                      confidence=float(parameters["confidence"]), cap=float(parameters["cap"]))
    return output, {"train_macro_rmse": _macro_rmse(gold, train_prediction),
                    "mean_head_log_loss": float(np.mean(head_losses)), "evidence_dims_used": dims}


def _state_hash(tensors: tuple[Any, ...]) -> str:
    digest = sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _final_ordinal(
    parameters: Mapping[str, Any], gold: np.ndarray, base: np.ndarray, r17: np.ndarray,
    direct: np.ndarray, test_base: np.ndarray, test_r17: np.ndarray, test_direct: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("final ordinal cycles require torch") from exc
    torch.manual_seed(SEED)
    raw = np.concatenate((base, r17, direct, r17 - base, direct - base), axis=1)
    test_raw = np.concatenate((test_base, test_r17, test_direct, test_r17 - test_base, test_direct - test_base), axis=1)
    raw, test_raw = _standardize(raw, test_raw)
    x = torch.tensor(np.column_stack((np.ones(len(raw)), raw)), dtype=torch.float32)
    tx = torch.tensor(np.column_stack((np.ones(len(test_raw)), test_raw)), dtype=torch.float32)
    y = torch.tensor(gold, dtype=torch.float32)
    b = torch.tensor(base, dtype=torch.float32)
    tb = torch.tensor(test_base, dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED)
    ordinal_weight = torch.nn.Parameter(torch.randn(3, x.shape[1], generator=generator) * 0.01)
    residual_weight = torch.nn.Parameter(torch.randn(3, x.shape[1], generator=generator) * 0.01)
    raw_steps = torch.nn.Parameter(torch.zeros(3, 4))
    tensors = (ordinal_weight, residual_weight, raw_steps)
    initial_hash = _state_hash(tensors)
    optimizer = torch.optim.Adam(tensors, lr=float(parameters["learning_rate"]))
    threshold = torch.arange(1, 5).view(1, 1, 4)
    ordinal_target = (torch.floor(y + 0.5).long().clamp(1, 5).unsqueeze(-1) > threshold).float()
    cap = float(parameters["cap"])
    for _ in range(int(parameters["epochs"])):
        latent = x @ ordinal_weight.T
        cuts = torch.cumsum(F.softplus(raw_steps) + 1e-3, dim=-1)
        cuts = cuts - cuts.mean(dim=-1, keepdim=True)
        logits = latent.unsqueeze(-1) - cuts.unsqueeze(0)
        expected = 1.0 + torch.sigmoid(logits).sum(-1)
        residual = cap * torch.tanh((x @ residual_weight.T) / cap)
        final = (b + float(parameters["ordinal_mix"]) * (expected - b) + residual).clamp(1.0, 5.0)
        loss = F.smooth_l1_loss(final, y, beta=0.5) + float(parameters["ordinal_weight"]) * F.binary_cross_entropy_with_logits(logits, ordinal_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        latent = tx @ ordinal_weight.T
        cuts = torch.cumsum(F.softplus(raw_steps) + 1e-3, dim=-1)
        cuts = cuts - cuts.mean(dim=-1, keepdim=True)
        expected = 1.0 + torch.sigmoid(latent.unsqueeze(-1) - cuts.unsqueeze(0)).sum(-1)
        residual = cap * torch.tanh((tx @ residual_weight.T) / cap)
        output = (tb + float(parameters["ordinal_mix"]) * (expected - tb) + residual).clamp(1.0, 5.0)
    return output.numpy(), {"initial_state_hash": initial_hash, "final_state_hash": _state_hash(tensors),
                            "train_final_loss": float(loss.detach().item()), "seed": SEED}


def fit_predict(
    spec: CycleSpec,
    train_gold: Any,
    train_base: Any,
    train_r17: Any,
    train_direct: Any,
    train_evidence: Any,
    test_base: Any,
    test_r17: Any,
    test_direct: Any,
    test_evidence: Any,
) -> CycleResult:
    """Fit one fixed cycle from scratch and predict the held-out matrix."""
    _validate_spec(spec)
    gold = _matrix(train_gold, "train_gold")
    base = _matrix(train_base, "train_base")
    r17 = _matrix(train_r17, "train_r17")
    direct = _matrix(train_direct, "train_direct")
    evidence = _evidence(train_evidence, "train_evidence")
    test_base_matrix = _matrix(test_base, "test_base")
    test_r17_matrix = _matrix(test_r17, "test_r17")
    test_direct_matrix = _matrix(test_direct, "test_direct")
    test_evidence_matrix = _evidence(test_evidence, "test_evidence")
    if not (gold.shape == base.shape == r17.shape == direct.shape) or len(evidence) != len(gold):
        raise ValueError("training matrices must have identical row counts")
    if not (test_base_matrix.shape == test_r17_matrix.shape == test_direct_matrix.shape) or len(test_evidence_matrix) != len(test_base_matrix):
        raise ValueError("test matrices must have identical row counts")
    if evidence.shape[1] != test_evidence_matrix.shape[1]:
        raise ValueError("train/test evidence dimensions differ")

    parameters = spec.parameters
    if spec.family == "soft_routed_residual":
        prediction, detail = _soft_routed(parameters, gold, base, r17, direct, evidence,
                                           test_base_matrix, test_r17_matrix, test_direct_matrix, test_evidence_matrix)
    elif spec.family == "pareto_routed_stack":
        prediction, detail = _pareto_stack(parameters, gold, base, r17, direct,
                                            test_base_matrix, test_r17_matrix, test_direct_matrix)
    elif spec.family == "group_dro_ridge":
        prediction, detail = _group_dro(parameters, gold, base, evidence, test_base_matrix, test_evidence_matrix)
    elif spec.family == "selective_hurdle":
        prediction, detail = _selective_hurdle(parameters, gold, base, r17, direct, evidence,
                                                test_base_matrix, test_r17_matrix, test_direct_matrix, test_evidence_matrix)
    else:
        prediction, detail = _final_ordinal(parameters, gold, base, r17, direct,
                                             test_base_matrix, test_r17_matrix, test_direct_matrix)
    output = np.asarray(np.clip(prediction, 1.0, 5.0), dtype=np.float32)
    audit = {
        "cycle": spec.cycle,
        "family": spec.family,
        "variant_id": spec.variant_id,
        "parameters": dict(spec.parameters),
        "train_records": len(gold),
        "test_records": len(output),
        "fresh_initialization": True,
        "checkpoint_reused": False,
        "average_target_used": False,
        "score1_role": "descriptive_only",
        **detail,
    }
    return CycleResult(output, MappingProxyType(audit))


__all__ = ["CycleResult", "CycleSpec", "FAMILIES", "SEED", "cycle_specs", "fit_predict"]
