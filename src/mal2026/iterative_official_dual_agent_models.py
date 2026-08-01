"""Fresh residual and boundary models over independent Terra/Luna scores."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .iterative_official_agent_stack_models import build_agent_score_features


# Two complete 39-dimensional within-model views plus six three-axis
# cross-model summaries: signed/absolute mean delta and pooled mean/std/min/max.
FEATURE_DIM = 96
HeadKind = Literal["identity", "adjacent_3v4", "dual_average"]


@dataclass(frozen=True)
class DualAgentSpec:
    cycle: int
    variant_id: str
    head_kind: HeadKind
    ridge_alpha: float = 10.0
    max_correction: float = 0.5
    confidence: float = 0.60
    window: float = 0.20
    epsilon: float = 0.001
    l2: float = 0.01

    def __post_init__(self) -> None:
        if self.cycle not in {1, 2, 3} or self.head_kind not in {"identity", "adjacent_3v4", "dual_average"}:
            raise ValueError("dual-agent identity differs")
        if self.ridge_alpha <= 0 or self.max_correction <= 0 or not 0.5 < self.confidence < 1.0:
            raise ValueError("dual-agent residual parameters differ")
        if not 0 < self.window <= 0.5 or not 0 < self.epsilon < 0.01 or self.l2 <= 0:
            raise ValueError("dual-agent boundary parameters differ")


@dataclass(frozen=True)
class DualAgentResult:
    predictions: np.ndarray
    audit: Mapping[str, Any]


def candidate_specs() -> tuple[DualAgentSpec, ...]:
    return (
        DualAgentSpec(1, "terra-luna-ridge-a10-cap050", "identity"),
        DualAgentSpec(2, "terra-luna-adjacent-flip-c060-w020", "adjacent_3v4"),
        DualAgentSpec(3, "terra-luna-dual-flip-c060-w020", "dual_average"),
    )


def build_dual_agent_features(
    base_scores: Any,
    source_ids: Sequence[str],
    terra_candidates: Sequence[Any],
    luna_candidates: Sequence[Any],
) -> tuple[np.ndarray, Mapping[str, Any]]:
    terra, terra_audit = build_agent_score_features(base_scores, source_ids, terra_candidates)
    luna, luna_audit = build_agent_score_features(base_scores, source_ids, luna_candidates)
    n = len(source_ids)
    terra_scores = terra[:, :9].reshape(n, 3, 3)
    luna_scores = luna[:, :9].reshape(n, 3, 3)
    terra_mean, luna_mean = terra_scores.mean(1), luna_scores.mean(1)
    pooled = np.concatenate((terra_scores, luna_scores), axis=1)
    cross = np.concatenate(
        (
            terra_mean - luna_mean,
            np.abs(terra_mean - luna_mean),
            pooled.mean(1),
            pooled.std(1),
            pooled.min(1),
            pooled.max(1),
        ),
        axis=1,
    )
    features = np.concatenate((terra, luna, cross), axis=1).astype(np.float64, copy=False)
    if features.shape != (n, FEATURE_DIM) or not np.isfinite(features).all():
        raise ValueError("dual-agent feature matrix differs")
    digest = sha256(np.asarray(features, dtype="<f8").tobytes(order="C")).hexdigest()
    return features, {
        "records": n,
        "dimensions": FEATURE_DIM,
        "feature_order": [
            "terra_within_model_39",
            "luna_within_model_39",
            "terra_minus_luna_axis_mean",
            "absolute_terra_minus_luna_axis_mean",
            "pooled_six_candidate_axis_mean",
            "pooled_six_candidate_axis_std",
            "pooled_six_candidate_axis_min",
            "pooled_six_candidate_axis_max",
        ],
        "feature_matrix_sha256": digest,
        "terra_feature_matrix_sha256": terra_audit["feature_matrix_sha256"],
        "luna_feature_matrix_sha256": luna_audit["feature_matrix_sha256"],
        "human_or_reference_score_read_or_prompted": False,
        "average_target_used": False,
    }


def _tensor_hash(*values: np.ndarray) -> str:
    digest = sha256()
    for value in values:
        digest.update(np.asarray(value, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _check_matrices(
    train_features: Any,
    train_base: Any,
    train_targets: Any,
    predict_features: Any,
    predict_base: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(train_features, dtype=np.float64)
    base = np.asarray(train_base, dtype=np.float64)
    targets = np.asarray(train_targets, dtype=np.float64)
    z = np.asarray(predict_features, dtype=np.float64)
    pbase = np.asarray(predict_base, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM or z.ndim != 2 or z.shape[1] != FEATURE_DIM:
        raise ValueError("dual-agent features must have 96 dimensions")
    if base.shape != targets.shape or base.shape != (len(x), 3) or pbase.shape != (len(z), 3):
        raise ValueError("dual-agent base/target matrices differ")
    if min(len(x), len(z)) < 1 or not all(np.isfinite(value).all() for value in (x, base, targets, z, pbase)):
        raise ValueError("dual-agent matrices must be finite and nonempty")
    return x, base, targets, z, pbase


def _fit_residual(
    spec: DualAgentSpec,
    x: np.ndarray,
    base: np.ndarray,
    targets: np.ndarray,
    z: np.ndarray,
    pbase: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    import torch

    torch_device = torch.device(device)
    if torch_device.type not in {"cpu", "cuda"} or (torch_device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("dual-agent residual device is unavailable")
    tx = torch.as_tensor(x, dtype=torch.float64, device=torch_device)
    tz = torch.as_tensor(z, dtype=torch.float64, device=torch_device)
    residual = torch.as_tensor(targets - base, dtype=torch.float64, device=torch_device)
    mean = tx.mean(0)
    std = tx.std(0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    tx, tz = (tx - mean) / std, (tz - mean) / std
    intercept = residual.mean(0)
    system = tx.T @ tx + spec.ridge_alpha * torch.eye(FEATURE_DIM, dtype=torch.float64, device=torch_device)
    weights = torch.linalg.solve(system, tx.T @ (residual - intercept))
    correction = (tz @ weights + intercept).clamp(-spec.max_correction, spec.max_correction)
    output = (torch.as_tensor(pbase, dtype=torch.float64, device=torch_device) + correction).clamp(1.0, 5.0)
    prediction = output.detach().cpu().numpy()
    return prediction, {
        "ridge_alpha": spec.ridge_alpha,
        "max_correction": spec.max_correction,
        "train_records": len(x),
        "prediction_records": len(z),
        "feature_dimensions": FEATURE_DIM,
        "coefficient_sha256": _tensor_hash(intercept.detach().cpu().numpy(), weights.detach().cpu().numpy()),
        "mean_abs_correction": float(np.mean(np.abs(prediction - pbase))),
        "max_abs_correction": float(np.max(np.abs(prediction - pbase))),
        "device": str(torch_device),
        "dtype": "torch.float64",
        "fresh_closed_form_solve": True,
        "checkpoint_reused": False,
    }


def _half_up_classes(targets: np.ndarray) -> np.ndarray:
    return np.floor(targets + 0.5).astype(np.int64).clip(1, 5)


def _fit_head(
    x: np.ndarray,
    targets: np.ndarray,
    z: np.ndarray,
    *,
    kind: Literal["adjacent_3v4", "threshold_ge4"],
    l2: float,
    device: str,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    import torch
    import torch.nn.functional as functional

    torch_device = torch.device(device)
    tx_all = torch.as_tensor(x, dtype=torch.float64, device=torch_device)
    tz_all = torch.as_tensor(z, dtype=torch.float64, device=torch_device)
    mean, std = tx_all.mean(0), tx_all.std(0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    tx_all, tz_all = (tx_all - mean) / std, (tz_all - mean) / std
    classes = _half_up_classes(targets)
    probabilities = np.empty((len(z), 3), dtype=np.float64)
    hashes, counts, losses = [], [], []
    for axis in range(3):
        if kind == "adjacent_3v4":
            mask = np.isin(classes[:, axis], (3, 4))
            labels = (classes[mask, axis] == 4).astype(np.float64)
        else:
            mask = np.ones(len(x), dtype=bool)
            labels = (classes[:, axis] >= 4).astype(np.float64)
        if mask.sum() < 2 or len(np.unique(labels)) != 2:
            raise ValueError("dual-agent boundary head needs both classes")
        tx = tx_all[torch.as_tensor(mask, device=torch_device)]
        ty = torch.as_tensor(labels, dtype=torch.float64, device=torch_device)
        weight = torch.zeros(FEATURE_DIM, dtype=torch.float64, device=torch_device, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, device=torch_device, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [weight, bias], lr=1.0, max_iter=80, tolerance_grad=1e-10,
            tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = functional.binary_cross_entropy_with_logits(tx @ weight + bias, ty) + l2 * weight.square().mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            loss = functional.binary_cross_entropy_with_logits(tx @ weight + bias, ty) + l2 * weight.square().mean()
            probabilities[:, axis] = torch.sigmoid(tz_all @ weight + bias).cpu().numpy()
            weight_cpu = weight.detach().cpu().numpy()
            bias_cpu = np.asarray([float(bias.detach().cpu())])
        hashes.append(_tensor_hash(weight_cpu, bias_cpu))
        counts.append({"records": int(mask.sum()), "negative": int((labels == 0).sum()), "positive": int((labels == 1).sum())})
        losses.append(float(loss.cpu()))
    return probabilities, {
        "kind": kind,
        "l2": l2,
        "axis_coefficient_sha256": hashes,
        "axis_label_counts": counts,
        "axis_final_loss": losses,
        "fresh_zero_initialization": True,
        "optimizer": "torch.optim.LBFGS_strong_wolfe",
        "max_iter": 80,
        "checkpoint_reused": False,
        "device": str(torch_device),
        "dtype": "torch.float64",
    }


def fit_predict_dual_agent(
    spec: DualAgentSpec,
    train_features: Any,
    train_base: Any,
    train_targets: Any,
    predict_features: Any,
    predict_base: Any,
    *,
    device: str,
) -> DualAgentResult:
    x, base, targets, z, pbase = _check_matrices(
        train_features, train_base, train_targets, predict_features, predict_base
    )
    primary, residual_audit = _fit_residual(spec, x, base, targets, z, pbase, device=device)
    heads: dict[str, Any] = {}
    probability = None
    if spec.head_kind != "identity":
        adjacent, audit = _fit_head(x, targets, z, kind="adjacent_3v4", l2=spec.l2, device=device)
        heads["adjacent_3v4"] = audit
        probability = adjacent
    if spec.head_kind == "dual_average":
        threshold, audit = _fit_head(x, targets, z, kind="threshold_ge4", l2=spec.l2, device=device)
        heads["threshold_ge4"] = audit
        probability = 0.5 * (probability + threshold)
    prediction = primary.copy()
    upward = np.zeros_like(primary, dtype=bool)
    downward = np.zeros_like(primary, dtype=bool)
    if probability is not None:
        upward = (primary < 3.5) & (primary >= 3.5 - spec.window) & (probability >= spec.confidence)
        downward = (primary >= 3.5) & (primary <= 3.5 + spec.window) & (probability <= 1.0 - spec.confidence)
        prediction[upward] = 3.5 + spec.epsilon
        prediction[downward] = 3.5 - spec.epsilon
    correction = prediction - primary
    return DualAgentResult(np.clip(prediction, 1.0, 5.0), {
        "cycle": spec.cycle,
        "variant_id": spec.variant_id,
        "head_kind": spec.head_kind,
        "ridge_alpha": spec.ridge_alpha,
        "max_correction": spec.max_correction,
        "confidence": spec.confidence,
        "window": spec.window,
        "epsilon": spec.epsilon,
        "l2": spec.l2,
        "residual": residual_audit,
        "heads": heads,
        "upward_flip_cells": int(upward.sum()),
        "downward_flip_cells": int(downward.sum()),
        "total_flip_cells": int((upward | downward).sum()),
        "prediction_cells": int(prediction.size),
        "flip_rate": float((upward | downward).mean()),
        "mean_abs_flip_correction": float(np.mean(np.abs(correction))),
        "max_abs_flip_correction": float(np.max(np.abs(correction))),
        "fresh_initialization": True,
        "checkpoint_reused": False,
        "average_target_used": False,
    })


__all__ = [
    "FEATURE_DIM",
    "DualAgentResult",
    "DualAgentSpec",
    "build_dual_agent_features",
    "candidate_specs",
    "fit_predict_dual_agent",
]
