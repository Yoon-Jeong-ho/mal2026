"""Fresh official-agent residual stacks with explicit 3/4 boundary heads."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Mapping

import numpy as np

from .iterative_official_agent_stack_models import (
    FEATURE_DIM,
    AgentStackSpec,
    fit_predict_agent_stack,
)


HeadKind = Literal["adjacent_3v4", "threshold_ge4", "dual_average"]


@dataclass(frozen=True)
class OfficialBoundarySpec:
    cycle: int
    variant_id: str
    head_kind: HeadKind
    nudge: float
    temperature: float
    radius: float = 0.75
    l2: float = 0.01

    def __post_init__(self) -> None:
        if self.cycle not in {1, 2, 3} or self.head_kind not in {"adjacent_3v4", "threshold_ge4", "dual_average"}:
            raise ValueError("official boundary spec identity differs")
        if min(self.nudge, self.temperature, self.radius, self.l2) <= 0:
            raise ValueError("official boundary parameters must be positive")


@dataclass(frozen=True)
class OfficialBoundaryResult:
    predictions: np.ndarray
    audit: Mapping[str, Any]


def candidate_specs() -> tuple[OfficialBoundarySpec, ...]:
    return (
        OfficialBoundarySpec(1, "official-terra-adjacent-3v4-nudge015", "adjacent_3v4", 0.15, 0.20),
        OfficialBoundarySpec(2, "official-terra-threshold-ge4-nudge015", "threshold_ge4", 0.15, 0.20),
        OfficialBoundarySpec(3, "official-terra-dual-boundary-nudge020", "dual_average", 0.20, 0.15),
    )


def _half_up_classes(targets: np.ndarray) -> np.ndarray:
    return np.floor(targets + 0.5).astype(np.int64).clip(1, 5)


def _tensor_hash(*values: np.ndarray) -> str:
    digest = sha256()
    for value in values:
        digest.update(np.asarray(value, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _fit_logistic_heads(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    predict_features: np.ndarray,
    *,
    kind: Literal["adjacent_3v4", "threshold_ge4"],
    l2: float,
    device: str,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    import torch
    import torch.nn.functional as F

    torch_device = torch.device(device)
    if torch_device.type not in {"cpu", "cuda"} or (torch_device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("official boundary device is unavailable")
    x = np.asarray(train_features, dtype=np.float64)
    z = np.asarray(predict_features, dtype=np.float64)
    targets = np.asarray(train_targets, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM or z.ndim != 2 or z.shape[1] != FEATURE_DIM or targets.shape != (len(x), 3):
        raise ValueError("official boundary matrices differ")
    if not all(np.isfinite(value).all() for value in (x, z, targets)):
        raise ValueError("official boundary matrices must be finite")
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-8] = 1.0
    x = (x - mean) / std
    z = (z - mean) / std
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
            raise ValueError("official boundary head needs both classes")
        tx = torch.as_tensor(x[mask], dtype=torch.float64, device=torch_device)
        ty = torch.as_tensor(labels, dtype=torch.float64, device=torch_device)
        tz = torch.as_tensor(z, dtype=torch.float64, device=torch_device)
        weight = torch.zeros(FEATURE_DIM, dtype=torch.float64, device=torch_device, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, device=torch_device, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [weight, bias], lr=1.0, max_iter=80, tolerance_grad=1e-10,
            tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            logits = tx @ weight + bias
            loss = F.binary_cross_entropy_with_logits(logits, ty) + l2 * weight.square().mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            final_logits = tx @ weight + bias
            final_loss = float((F.binary_cross_entropy_with_logits(final_logits, ty) + l2 * weight.square().mean()).cpu())
            probability = torch.sigmoid(tz @ weight + bias).cpu().numpy()
            weight_cpu = weight.detach().cpu().numpy()
            bias_cpu = np.asarray([float(bias.detach().cpu())], dtype=np.float64)
        probabilities[:, axis] = probability
        hashes.append(_tensor_hash(weight_cpu, bias_cpu))
        counts.append({"records": int(mask.sum()), "negative": int((labels == 0).sum()), "positive": int((labels == 1).sum())})
        losses.append(final_loss)
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


def fit_predict_official_boundary(
    spec: OfficialBoundarySpec,
    train_features: Any,
    train_base: Any,
    train_targets: Any,
    predict_features: Any,
    predict_base: Any,
    *,
    device: str,
) -> OfficialBoundaryResult:
    x = np.asarray(train_features, dtype=np.float64)
    z = np.asarray(predict_features, dtype=np.float64)
    base = np.asarray(train_base, dtype=np.float64)
    targets = np.asarray(train_targets, dtype=np.float64)
    predict_base_array = np.asarray(predict_base, dtype=np.float64)
    residual = fit_predict_agent_stack(
        AgentStackSpec(1, "v8-fresh-primary-residual", 10.0, 0.5),
        x, base, targets, z, predict_base_array, device=device,
    )
    head_audit: dict[str, Any] = {}
    if spec.head_kind in {"adjacent_3v4", "dual_average"}:
        adjacent, audit = _fit_logistic_heads(x, targets, z, kind="adjacent_3v4", l2=spec.l2, device=device)
        head_audit["adjacent_3v4"] = audit
    else:
        adjacent = None
    if spec.head_kind in {"threshold_ge4", "dual_average"}:
        threshold, audit = _fit_logistic_heads(x, targets, z, kind="threshold_ge4", l2=spec.l2, device=device)
        head_audit["threshold_ge4"] = audit
    else:
        threshold = None
    if spec.head_kind == "adjacent_3v4":
        probability = adjacent
    elif spec.head_kind == "threshold_ge4":
        probability = threshold
    else:
        assert adjacent is not None and threshold is not None
        probability = 0.5 * (adjacent + threshold)
    assert probability is not None
    primary = np.asarray(residual.predictions, dtype=np.float64)
    proximity = np.clip(1.0 - np.abs(primary - 3.5) / spec.radius, 0.0, 1.0)
    direction = np.tanh((probability - 0.5) / spec.temperature)
    correction = spec.nudge * proximity * direction
    prediction = np.clip(primary + correction, 1.0, 5.0)
    return OfficialBoundaryResult(prediction, {
        "cycle": spec.cycle,
        "variant_id": spec.variant_id,
        "head_kind": spec.head_kind,
        "nudge": spec.nudge,
        "temperature": spec.temperature,
        "radius": spec.radius,
        "l2": spec.l2,
        "residual": residual.audit,
        "heads": head_audit,
        "mean_abs_boundary_correction": float(np.mean(np.abs(correction))),
        "max_abs_boundary_correction": float(np.max(np.abs(correction))),
        "prediction_records": len(z),
        "fresh_initialization": True,
        "checkpoint_reused": False,
        "average_target_used": False,
    })


__all__ = [
    "OfficialBoundaryResult", "OfficialBoundarySpec", "candidate_specs",
    "fit_predict_official_boundary",
]
