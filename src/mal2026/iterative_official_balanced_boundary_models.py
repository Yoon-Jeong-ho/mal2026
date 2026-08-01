"""Class-balanced adjacent 3/4 heads over the frozen Terra/Luna residual."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

import numpy as np

from .iterative_official_dual_agent_models import FEATURE_DIM, _check_matrices, _fit_residual


@dataclass(frozen=True)
class BalancedBoundarySpec:
    cycle: int
    variant_id: str
    confidence: float
    window: float
    l2: float
    ridge_alpha: float = 10.0
    max_correction: float = 0.5
    epsilon: float = 0.001
    head_kind: str = "class_balanced_adjacent_3v4"

    def __post_init__(self) -> None:
        if self.cycle not in {1, 2, 3} or self.head_kind != "class_balanced_adjacent_3v4":
            raise ValueError("balanced-boundary identity differs")
        if not 0.5 <= self.confidence < 1.0 or not 0 < self.window <= 0.5:
            raise ValueError("balanced-boundary application differs")
        if self.l2 <= 0 or self.ridge_alpha <= 0 or self.max_correction <= 0 or not 0 < self.epsilon < 0.01:
            raise ValueError("balanced-boundary parameters differ")


@dataclass(frozen=True)
class BalancedBoundaryResult:
    predictions: np.ndarray
    audit: Mapping[str, Any]


def candidate_specs() -> tuple[BalancedBoundarySpec, ...]:
    return (
        BalancedBoundarySpec(1, "terra-luna-balanced-adjacent-l2-001-c050-w025", 0.50, 0.25, 0.01),
        BalancedBoundarySpec(2, "terra-luna-balanced-adjacent-l2-010-c050-w025", 0.50, 0.25, 0.10),
        BalancedBoundarySpec(3, "terra-luna-balanced-adjacent-l2-001-c055-w020", 0.55, 0.20, 0.01),
    )


def _tensor_hash(*values: np.ndarray) -> str:
    digest = sha256()
    for value in values:
        digest.update(np.asarray(value, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _half_up_classes(targets: np.ndarray) -> np.ndarray:
    return np.floor(targets + 0.5).astype(np.int64).clip(1, 5)


def _fit_balanced_adjacent_head(
    x: np.ndarray,
    targets: np.ndarray,
    z: np.ndarray,
    *,
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
    hashes, counts, losses, weight_audits = [], [], [], []
    for axis in range(3):
        mask = np.isin(classes[:, axis], (3, 4))
        labels = (classes[mask, axis] == 4).astype(np.float64)
        negative, positive = int((labels == 0).sum()), int((labels == 1).sum())
        if min(negative, positive) < 1:
            raise ValueError("balanced adjacent head needs both 3 and 4")
        sample_weights = np.where(labels == 0, 0.5 / negative, 0.5 / positive)
        # Normalize to mean one for stable LBFGS/l2 scale while retaining equal
        # total weight for the two classes.
        sample_weights *= len(sample_weights)
        tx = tx_all[torch.as_tensor(mask, device=torch_device)]
        ty = torch.as_tensor(labels, dtype=torch.float64, device=torch_device)
        tw = torch.as_tensor(sample_weights, dtype=torch.float64, device=torch_device)
        weight = torch.zeros(FEATURE_DIM, dtype=torch.float64, device=torch_device, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, device=torch_device, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [weight, bias], lr=1.0, max_iter=80, tolerance_grad=1e-10,
            tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe",
        )

        def objective():
            point = functional.binary_cross_entropy_with_logits(tx @ weight + bias, ty, reduction="none")
            return (point * tw).mean() + l2 * weight.square().mean()

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            loss = objective()
            probabilities[:, axis] = torch.sigmoid(tz_all @ weight + bias).cpu().numpy()
            weight_cpu = weight.detach().cpu().numpy()
            bias_cpu = np.asarray([float(bias.detach().cpu())])
        hashes.append(_tensor_hash(weight_cpu, bias_cpu))
        counts.append({"records": int(mask.sum()), "negative_gold_3": negative, "positive_gold_4": positive})
        weight_audits.append({
            "negative_per_record": float(0.5 * len(labels) / negative),
            "positive_per_record": float(0.5 * len(labels) / positive),
            "negative_total": float(sample_weights[labels == 0].sum()),
            "positive_total": float(sample_weights[labels == 1].sum()),
        })
        losses.append(float(loss.cpu()))
    return probabilities, {
        "kind": "class_balanced_adjacent_3v4",
        "l2": l2,
        "axis_coefficient_sha256": hashes,
        "axis_label_counts": counts,
        "axis_class_weight_audit": weight_audits,
        "axis_final_loss": losses,
        "equal_total_class_weight": True,
        "fresh_zero_initialization": True,
        "optimizer": "torch.optim.LBFGS_strong_wolfe",
        "max_iter": 80,
        "checkpoint_reused": False,
        "device": str(torch_device),
        "dtype": "torch.float64",
    }


def fit_predict_balanced_boundary(
    spec: BalancedBoundarySpec,
    train_features: Any,
    train_base: Any,
    train_targets: Any,
    predict_features: Any,
    predict_base: Any,
    *,
    device: str,
) -> BalancedBoundaryResult:
    x, base, targets, z, pbase = _check_matrices(
        train_features, train_base, train_targets, predict_features, predict_base
    )
    primary, residual_audit = _fit_residual(spec, x, base, targets, z, pbase, device=device)
    probability, head_audit = _fit_balanced_adjacent_head(x, targets, z, l2=spec.l2, device=device)
    upward = (primary < 3.5) & (primary >= 3.5 - spec.window) & (probability >= spec.confidence)
    downward = (primary >= 3.5) & (primary <= 3.5 + spec.window) & (probability <= 1.0 - spec.confidence)
    prediction = primary.copy()
    prediction[upward] = 3.5 + spec.epsilon
    prediction[downward] = 3.5 - spec.epsilon
    correction = prediction - primary
    return BalancedBoundaryResult(np.clip(prediction, 1.0, 5.0), {
        "cycle": spec.cycle,
        "variant_id": spec.variant_id,
        "head_kind": spec.head_kind,
        "confidence": spec.confidence,
        "window": spec.window,
        "epsilon": spec.epsilon,
        "l2": spec.l2,
        "ridge_alpha": spec.ridge_alpha,
        "max_correction": spec.max_correction,
        "residual": residual_audit,
        "heads": {"class_balanced_adjacent_3v4": head_audit},
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
    "BalancedBoundaryResult", "BalancedBoundarySpec", "candidate_specs",
    "fit_predict_balanced_boundary",
]
