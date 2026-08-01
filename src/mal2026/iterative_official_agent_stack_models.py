"""Fresh closed-form residual stacks over score-blind official Terra outputs."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np


AXES = ("content", "organization", "expression")
FEATURE_DIM = 39


@dataclass(frozen=True)
class AgentStackSpec:
    cycle: int
    variant_id: str
    ridge_alpha: float
    max_correction: float

    def __post_init__(self) -> None:
        if self.cycle not in {1, 2, 3}:
            raise ValueError("agent-stack cycle must be 1..3")
        if not self.variant_id or self.ridge_alpha <= 0 or self.max_correction <= 0:
            raise ValueError("agent-stack spec is invalid")


@dataclass(frozen=True)
class AgentStackFitResult:
    predictions: np.ndarray
    audit: Mapping[str, Any]


def candidate_specs() -> tuple[AgentStackSpec, ...]:
    return (
        AgentStackSpec(1, "official-terra-ridge-a10-cap050-primary", 10.0, 0.5),
        AgentStackSpec(2, "official-terra-ridge-a010-cap030-diverse", 0.1, 0.3),
        AgentStackSpec(3, "official-terra-ridge-a10-cap010-conservative", 10.0, 0.1),
    )


def build_agent_score_features(
    base_scores: Any,
    source_ids: Sequence[str],
    official_candidates: Sequence[Any],
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Align three score-blind API outputs and construct the fixed 39 features."""
    base = np.asarray(base_scores, dtype=np.float64)
    identifiers = tuple(source_ids)
    if base.shape != (len(identifiers), 3) or len(identifiers) == 0 or len(set(identifiers)) != len(identifiers):
        raise ValueError("base scores/source IDs must be aligned N x 3 unique rows")
    if not np.isfinite(base).all() or np.any((base < 1.0) | (base > 5.0)):
        raise ValueError("base scores must be finite in [1,5]")
    index = {source_id: row for row, source_id in enumerate(identifiers)}
    values = np.empty((len(identifiers), 3, 3), dtype=np.float64)
    coverage = np.zeros((len(identifiers), 3), dtype=np.int8)
    for candidate in official_candidates:
        source_id = getattr(candidate, "source_id", None)
        number = getattr(candidate, "candidate_number", None)
        scores = getattr(candidate, "scores", None)
        if source_id not in index or type(number) is not int or number not in {1, 2, 3}:
            raise ValueError("official candidate identity differs")
        row, column = index[source_id], number - 1
        if coverage[row, column] or not isinstance(scores, Mapping) or set(scores) != set(AXES):
            raise ValueError("official candidate coverage or axes differ")
        vector = np.asarray([scores[axis] for axis in AXES], dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all() or np.any((vector < 1) | (vector > 5)):
            raise ValueError("official candidate score differs")
        coverage[row, column] = 1
        values[row, column] = vector
    if len(official_candidates) != 3 * len(identifiers) or not np.all(coverage == 1):
        raise ValueError("official candidates must cover each row exactly three times")
    mean, std = values.mean(1), values.std(1)
    minimum, maximum = values.min(1), values.max(1)
    pairwise_equal = np.concatenate(
        [(values[:, left] == values[:, right]).astype(np.float64)
         for left, right in ((0, 1), (0, 2), (1, 2))], axis=1,
    )
    repeated_base = np.repeat(base[:, None, :], 3, axis=1)
    features = np.concatenate(
        (values.reshape(len(values), -1), mean, std, minimum, maximum,
         pairwise_equal, (values - repeated_base).reshape(len(values), -1)), axis=1,
    )
    if features.shape != (len(identifiers), FEATURE_DIM) or not np.isfinite(features).all():
        raise ValueError("official agent feature matrix differs")
    feature_hash = sha256(np.asarray(features, dtype="<f8").tobytes(order="C")).hexdigest()
    return features, {
        "records": len(identifiers),
        "dimensions": FEATURE_DIM,
        "feature_order": [
            "candidate_scores_3x3", "axis_mean", "axis_std", "axis_min", "axis_max",
            "pairwise_equal_3x3", "candidate_minus_r0_3x3",
        ],
        "feature_matrix_sha256": feature_hash,
        "human_or_reference_score_read_or_prompted": False,
        "average_target_used": False,
    }


def _coefficient_hash(intercept: np.ndarray, weights: np.ndarray) -> str:
    digest = sha256()
    digest.update(np.asarray(intercept, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(weights, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def fit_predict_agent_stack(
    spec: AgentStackSpec,
    train_features: Any,
    train_base: Any,
    train_targets: Any,
    predict_features: Any,
    predict_base: Any,
    *,
    device: str,
) -> AgentStackFitResult:
    """Solve one fresh standardized ridge system and predict once.

    Torch float64 is used on the explicitly assigned experiment worker. The
    production runner requires CUDA; CPU remains available only for small
    deterministic unit tests. There is no checkpoint, warm start, or state
    shared across folds or candidates.
    """
    import torch

    x = np.asarray(train_features, dtype=np.float64)
    z = np.asarray(predict_features, dtype=np.float64)
    base = np.asarray(train_base, dtype=np.float64)
    y = np.asarray(train_targets, dtype=np.float64)
    predict_base_array = np.asarray(predict_base, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM or z.ndim != 2 or z.shape[1] != FEATURE_DIM:
        raise ValueError("agent-stack features must have fixed 39 dimensions")
    if base.shape != y.shape or base.shape != (len(x), 3) or predict_base_array.shape != (len(z), 3):
        raise ValueError("agent-stack base/target shapes differ")
    if min(len(x), len(z)) < 1 or not all(np.isfinite(value).all() for value in (x, z, base, y, predict_base_array)):
        raise ValueError("agent-stack matrices must be nonempty and finite")
    torch_device = torch.device(device)
    if torch_device.type not in {"cpu", "cuda"} or (torch_device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("agent-stack device is unavailable")
    tx = torch.as_tensor(x, dtype=torch.float64, device=torch_device)
    tz = torch.as_tensor(z, dtype=torch.float64, device=torch_device)
    residual = torch.as_tensor(y - base, dtype=torch.float64, device=torch_device)
    feature_mean = tx.mean(0)
    feature_std = tx.std(0, unbiased=False)
    feature_std = torch.where(feature_std < 1e-8, torch.ones_like(feature_std), feature_std)
    tx = (tx - feature_mean) / feature_std
    tz = (tz - feature_mean) / feature_std
    intercept = residual.mean(0)
    centered = residual - intercept
    system = tx.T @ tx + spec.ridge_alpha * torch.eye(FEATURE_DIM, dtype=torch.float64, device=torch_device)
    weights = torch.linalg.solve(system, tx.T @ centered)
    correction = tz @ weights + intercept
    correction = correction.clamp(-spec.max_correction, spec.max_correction)
    prediction = torch.as_tensor(predict_base_array, dtype=torch.float64, device=torch_device) + correction
    prediction = prediction.clamp(1.0, 5.0)
    intercept_cpu = intercept.detach().cpu().numpy()
    weights_cpu = weights.detach().cpu().numpy()
    output = prediction.detach().cpu().numpy().astype(np.float64, copy=False)
    return AgentStackFitResult(output, {
        "cycle": spec.cycle,
        "variant_id": spec.variant_id,
        "ridge_alpha": spec.ridge_alpha,
        "max_correction": spec.max_correction,
        "train_records": len(x),
        "prediction_records": len(z),
        "feature_dimensions": FEATURE_DIM,
        "device": str(torch_device),
        "dtype": "torch.float64",
        "fresh_closed_form_solve": True,
        "checkpoint_reused": False,
        "coefficient_sha256": _coefficient_hash(intercept_cpu, weights_cpu),
        "mean_abs_correction": float(np.mean(np.abs(output - predict_base_array))),
        "max_abs_correction": float(np.max(np.abs(output - predict_base_array))),
        "average_target_used": False,
    })


__all__ = [
    "AXES", "FEATURE_DIM", "AgentStackFitResult", "AgentStackSpec",
    "build_agent_score_features", "candidate_specs", "fit_predict_agent_stack",
]
