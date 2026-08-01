"""Fixed V12 residual learners over frozen rationale-semantic embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Mapping

import numpy as np


SEMANTIC_DIM = 201
STRUCTURED_DIM = 96
FUSION_DIM = SEMANTIC_DIM + STRUCTURED_DIM
FeatureKind = Literal["semantic201", "fusion297"]
HeadKind = Literal["identity", "balanced_adjacent_3v4"]


@dataclass(frozen=True)
class RationaleSemanticSpec:
    cycle: int
    variant_id: str
    feature_kind: FeatureKind
    head_kind: HeadKind
    ridge_alpha: float = 10.0
    max_correction: float = 0.5
    l2: float = 0.01
    confidence: float = 0.55
    window: float = 0.20
    epsilon: float = 0.001

    def __post_init__(self) -> None:
        expected = {
            1: ("semantic201", "identity"),
            2: ("fusion297", "identity"),
            3: ("fusion297", "balanced_adjacent_3v4"),
        }
        if expected.get(self.cycle) != (self.feature_kind, self.head_kind) or not self.variant_id:
            raise ValueError("V12 candidate identity differs")
        if (self.ridge_alpha, self.max_correction, self.l2, self.confidence, self.window, self.epsilon) != (10.0, 0.5, 0.01, 0.55, 0.20, 0.001):
            raise ValueError("V12 fixed parameters differ")


@dataclass(frozen=True)
class RationaleSemanticResult:
    predictions: np.ndarray
    audit: Mapping[str, Any]


def candidate_specs() -> tuple[RationaleSemanticSpec, ...]:
    return (
        RationaleSemanticSpec(1, "rationale-semantic201-ridge-a10-cap050", "semantic201", "identity"),
        RationaleSemanticSpec(2, "rationale-fusion297-ridge-a10-cap050", "fusion297", "identity"),
        RationaleSemanticSpec(3, "rationale-fusion297-balanced-3v4-l2-001-c055-w020", "fusion297", "balanced_adjacent_3v4"),
    )


def _hash(*arrays: np.ndarray) -> str:
    digest = sha256()
    for array in arrays:
        digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _matrix(value: Any, columns: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != columns or len(result) < 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty matrix with {columns} columns")
    return result


def _inputs(
    train_semantic: Any,
    train_structured: Any,
    train_base: Any,
    train_targets: Any,
    predict_semantic: Any,
    predict_structured: Any,
    predict_base: Any,
) -> tuple[np.ndarray, ...]:
    semantic = _matrix(train_semantic, SEMANTIC_DIM, "train semantic")
    structured = _matrix(train_structured, STRUCTURED_DIM, "train structured")
    psemantic = _matrix(predict_semantic, SEMANTIC_DIM, "predict semantic")
    pstructured = _matrix(predict_structured, STRUCTURED_DIM, "predict structured")
    base = _matrix(train_base, 3, "train base")
    targets = _matrix(train_targets, 3, "train targets")
    pbase = _matrix(predict_base, 3, "predict base")
    if len({len(semantic), len(structured), len(base), len(targets)}) != 1:
        raise ValueError("V12 train matrices differ")
    if len({len(psemantic), len(pstructured), len(pbase)}) != 1:
        raise ValueError("V12 predict matrices differ")
    return semantic, structured, base, targets, psemantic, pstructured, pbase


def _features(spec: RationaleSemanticSpec, semantic: np.ndarray, structured: np.ndarray) -> np.ndarray:
    return semantic if spec.feature_kind == "semantic201" else np.concatenate((structured, semantic), axis=1)


def _device(value: str):
    import torch

    device = torch.device(value)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("V12 assigned device is unavailable")
    return device


def _standardize(train: Any, predict: Any) -> tuple[Any, Any, Any, Any]:
    """Fit normalization on S only, then apply the frozen statistics to D/O."""
    import torch

    mean = train.mean(0)
    std = train.std(0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return (train - mean) / std, (predict - mean) / std, mean, std


def _fit_residual(
    spec: RationaleSemanticSpec,
    train_features: np.ndarray,
    train_base: np.ndarray,
    targets: np.ndarray,
    predict_features: np.ndarray,
    predict_base: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    import torch

    torch_device = _device(device)
    tx_raw = torch.as_tensor(train_features, dtype=torch.float64, device=torch_device)
    tz_raw = torch.as_tensor(predict_features, dtype=torch.float64, device=torch_device)
    tx, tz, mean, std = _standardize(tx_raw, tz_raw)
    residual = torch.as_tensor(targets - train_base, dtype=torch.float64, device=torch_device)
    intercept = residual.mean(0)
    dimension = train_features.shape[1]
    system = tx.T @ tx + spec.ridge_alpha * torch.eye(dimension, dtype=torch.float64, device=torch_device)
    weights = torch.linalg.solve(system, tx.T @ (residual - intercept))
    correction = (tz @ weights + intercept).clamp(-spec.max_correction, spec.max_correction)
    prediction = (
        torch.as_tensor(predict_base, dtype=torch.float64, device=torch_device) + correction
    ).clamp(1.0, 5.0).detach().cpu().numpy()
    return prediction, {
        "feature_kind": spec.feature_kind,
        "feature_dimensions": dimension,
        "ridge_alpha": spec.ridge_alpha,
        "max_correction": spec.max_correction,
        "train_records": len(train_features),
        "prediction_records": len(predict_features),
        "coefficient_sha256": _hash(intercept.detach().cpu().numpy(), weights.detach().cpu().numpy()),
        "fit_standardization_sha256": _hash(mean.detach().cpu().numpy(), std.detach().cpu().numpy()),
        "normalization_fit_scope": "fit_partition_only",
        "mean_abs_correction": float(np.mean(np.abs(prediction - predict_base))),
        "max_abs_correction": float(np.max(np.abs(prediction - predict_base))),
        "device": str(torch_device),
        "dtype": "torch.float64",
        "fresh_closed_form_solve": True,
        "checkpoint_reused": False,
    }


def _half_up_classes(targets: np.ndarray) -> np.ndarray:
    return np.floor(targets + 0.5).astype(np.int64).clip(1, 5)


def _fit_balanced_head(
    train_features: np.ndarray,
    targets: np.ndarray,
    predict_features: np.ndarray,
    *,
    l2: float,
    device: str,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    import torch
    import torch.nn.functional as functional

    torch_device = _device(device)
    tx_raw = torch.as_tensor(train_features, dtype=torch.float64, device=torch_device)
    tz_raw = torch.as_tensor(predict_features, dtype=torch.float64, device=torch_device)
    tx_all, tz_all, mean, std = _standardize(tx_raw, tz_raw)
    classes = _half_up_classes(targets)
    dimension = train_features.shape[1]
    probabilities = np.empty((len(predict_features), 3), dtype=np.float64)
    hashes, counts, losses, weight_audits = [], [], [], []
    for axis in range(3):
        mask = np.isin(classes[:, axis], (3, 4))
        labels = (classes[mask, axis] == 4).astype(np.float64)
        negative, positive = int((labels == 0).sum()), int((labels == 1).sum())
        if min(negative, positive) < 1:
            raise ValueError("V12 balanced head needs both gold 3 and gold 4")
        sample_weights = np.where(labels == 0, 0.5 / negative, 0.5 / positive) * len(labels)
        tx = tx_all[torch.as_tensor(mask, device=torch_device)]
        ty = torch.as_tensor(labels, dtype=torch.float64, device=torch_device)
        tw = torch.as_tensor(sample_weights, dtype=torch.float64, device=torch_device)
        weight = torch.zeros(dimension, dtype=torch.float64, device=torch_device, requires_grad=True)
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
        hashes.append(_hash(weight_cpu, bias_cpu))
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
        "feature_dimensions": dimension,
        "axis_coefficient_sha256": hashes,
        "axis_label_counts": counts,
        "axis_class_weight_audit": weight_audits,
        "axis_final_loss": losses,
        "fit_standardization_sha256": _hash(mean.detach().cpu().numpy(), std.detach().cpu().numpy()),
        "normalization_fit_scope": "fit_partition_only",
        "equal_total_class_weight": True,
        "fresh_zero_initialization": True,
        "optimizer": "torch.optim.LBFGS_strong_wolfe",
        "max_iter": 80,
        "checkpoint_reused": False,
        "device": str(torch_device),
        "dtype": "torch.float64",
    }


def fit_predict_rationale_semantic(
    spec: RationaleSemanticSpec,
    train_semantic: Any,
    train_structured96: Any,
    train_base: Any,
    train_targets: Any,
    predict_semantic: Any,
    predict_structured96: Any,
    predict_base: Any,
    *,
    device: str,
) -> RationaleSemanticResult:
    """Fresh-fit one fixed candidate and return bounded three-axis scores."""
    semantic, structured, base, targets, psemantic, pstructured, pbase = _inputs(
        train_semantic, train_structured96, train_base, train_targets,
        predict_semantic, predict_structured96, predict_base,
    )
    x, z = _features(spec, semantic, structured), _features(spec, psemantic, pstructured)
    primary, residual_audit = _fit_residual(spec, x, base, targets, z, pbase, device=device)
    prediction = primary.copy()
    heads: dict[str, Any] = {}
    upward = np.zeros_like(primary, dtype=bool)
    downward = np.zeros_like(primary, dtype=bool)
    if spec.head_kind == "balanced_adjacent_3v4":
        probability, head_audit = _fit_balanced_head(x, targets, z, l2=spec.l2, device=device)
        heads["class_balanced_adjacent_3v4"] = head_audit
        upward = (primary < 3.5) & (primary >= 3.5 - spec.window) & (probability >= spec.confidence)
        downward = (primary >= 3.5) & (primary <= 3.5 + spec.window) & (probability <= 1.0 - spec.confidence)
        prediction[upward] = 3.5 + spec.epsilon
        prediction[downward] = 3.5 - spec.epsilon
    prediction = np.clip(prediction, 1.0, 5.0)
    return RationaleSemanticResult(prediction, {
        "cycle": spec.cycle,
        "variant_id": spec.variant_id,
        "feature_kind": spec.feature_kind,
        "head_kind": spec.head_kind,
        "ridge_alpha": spec.ridge_alpha,
        "max_correction": spec.max_correction,
        "l2": spec.l2,
        "confidence": spec.confidence,
        "window": spec.window,
        "epsilon": spec.epsilon,
        "residual": residual_audit,
        "heads": heads,
        "upward_flip_cells": int(upward.sum()),
        "downward_flip_cells": int(downward.sum()),
        "total_flip_cells": int((upward | downward).sum()),
        "prediction_cells": int(prediction.size),
        "fresh_initialization": True,
        "checkpoint_reused": False,
        "average_target_used": False,
    })


__all__ = [
    "FUSION_DIM", "SEMANTIC_DIM", "STRUCTURED_DIM", "RationaleSemanticResult",
    "RationaleSemanticSpec", "candidate_specs", "fit_predict_rationale_semantic",
]
