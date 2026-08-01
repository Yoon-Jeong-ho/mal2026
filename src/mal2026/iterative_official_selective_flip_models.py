"""High-confidence near-boundary flips over the frozen official-agent residual."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

from .iterative_official_agent_stack_models import AgentStackSpec, fit_predict_agent_stack
from .iterative_official_boundary_models import _fit_logistic_heads


HeadKind = Literal["adjacent_3v4", "dual_average"]


@dataclass(frozen=True)
class SelectiveFlipSpec:
    cycle: int
    variant_id: str
    head_kind: HeadKind
    confidence: float
    window: float
    epsilon: float = 0.001
    l2: float = 0.01

    def __post_init__(self) -> None:
        if self.cycle not in {1, 2, 3} or self.head_kind not in {"adjacent_3v4", "dual_average"}:
            raise ValueError("selective-flip identity differs")
        if not 0.5 < self.confidence < 1.0 or not 0 < self.window <= 0.5 or not 0 < self.epsilon < 0.01 or self.l2 <= 0:
            raise ValueError("selective-flip parameters differ")


@dataclass(frozen=True)
class SelectiveFlipResult:
    predictions: np.ndarray
    audit: Mapping[str, Any]


def candidate_specs() -> tuple[SelectiveFlipSpec, ...]:
    return (
        SelectiveFlipSpec(1, "official-terra-adjacent-flip-c060-w020", "adjacent_3v4", 0.60, 0.20),
        SelectiveFlipSpec(2, "official-terra-adjacent-flip-c065-w015", "adjacent_3v4", 0.65, 0.15),
        SelectiveFlipSpec(3, "official-terra-dual-flip-c060-w020", "dual_average", 0.60, 0.20),
    )


def fit_predict_selective_flip(
    spec: SelectiveFlipSpec,
    train_features: Any,
    train_base: Any,
    train_targets: Any,
    predict_features: Any,
    predict_base: Any,
    *,
    device: str,
) -> SelectiveFlipResult:
    x = np.asarray(train_features, dtype=np.float64)
    z = np.asarray(predict_features, dtype=np.float64)
    targets = np.asarray(train_targets, dtype=np.float64)
    residual = fit_predict_agent_stack(
        AgentStackSpec(1, "v9-fresh-primary-residual", 10.0, 0.5),
        x, train_base, targets, z, predict_base, device=device,
    )
    adjacent, adjacent_audit = _fit_logistic_heads(
        x, targets, z, kind="adjacent_3v4", l2=spec.l2, device=device,
    )
    heads: dict[str, Any] = {"adjacent_3v4": adjacent_audit}
    if spec.head_kind == "dual_average":
        threshold, threshold_audit = _fit_logistic_heads(
            x, targets, z, kind="threshold_ge4", l2=spec.l2, device=device,
        )
        probability = 0.5 * (adjacent + threshold)
        heads["threshold_ge4"] = threshold_audit
    else:
        probability = adjacent
    primary = np.asarray(residual.predictions, dtype=np.float64)
    upward = (primary < 3.5) & (primary >= 3.5 - spec.window) & (probability >= spec.confidence)
    downward = (primary >= 3.5) & (primary <= 3.5 + spec.window) & (probability <= 1.0 - spec.confidence)
    prediction = primary.copy()
    prediction[upward] = 3.5 + spec.epsilon
    prediction[downward] = 3.5 - spec.epsilon
    correction = prediction - primary
    return SelectiveFlipResult(np.clip(prediction, 1.0, 5.0), {
        "cycle": spec.cycle,
        "variant_id": spec.variant_id,
        "head_kind": spec.head_kind,
        "confidence": spec.confidence,
        "window": spec.window,
        "epsilon": spec.epsilon,
        "l2": spec.l2,
        "residual": residual.audit,
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
    "SelectiveFlipResult", "SelectiveFlipSpec", "candidate_specs",
    "fit_predict_selective_flip",
]
