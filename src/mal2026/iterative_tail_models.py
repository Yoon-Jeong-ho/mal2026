"""Small frozen-feature candidates for iterative tail-score correction.

The functions in this module operate only on already-computed embeddings,
three continuous base scores, and optional structured features.  They do not
contain or update an encoder.  Every iterative round is fit from a fresh
initialization and predictions are always clipped to the official ``[1, 5]``
range.  In particular, this API accepts three axis targets and has no essay
``average`` target.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any, Literal

import numpy as np


Family = Literal[
    "ridge_residual",
    "mlp_residual",
    "coral",
    "joint_huber_ordinal",
    "uncertainty_gated",
    "tail_effective_number",
    "equal_band_replay",
    "auxiliary_3v4",
    "adjacent_contrastive",
]
TailWeightingMode = Literal["both", "low", "high"]

FAMILIES: tuple[str, ...] = (
    "ridge_residual",
    "mlp_residual",
    "coral",
    "joint_huber_ordinal",
    "uncertainty_gated",
    "tail_effective_number",
    "equal_band_replay",
    "auxiliary_3v4",
    "adjacent_contrastive",
)


@dataclass(frozen=True)
class CandidateSpec:
    """Configuration for one independently reproducible fold candidate."""

    family: Family
    rounds: int = 1
    seed: int = 2026
    device: str = "cpu"
    hidden_dim: int = 32
    dropout: float = 0.1
    epochs: int = 20
    learning_rate: float = 3e-3
    ridge_alpha: float = 1.0
    huber_delta: float = 0.5
    ordinal_weight: float = 0.5
    auxiliary_weight: float = 0.25
    contrastive_weight: float = 0.05
    contrastive_temperature: float = 0.2
    effective_number_beta: float = 0.99
    tail_weighting_mode: TailWeightingMode = "both"
    tail_weighting_strength: float = 1.0
    uncertainty_coverage: float = 1.0
    max_correction: float = 0.75

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown candidate family: {self.family!r}")
        if self.rounds < 1 or self.epochs < 1 or self.hidden_dim < 1:
            raise ValueError("rounds, epochs, and hidden_dim must be positive")
        if self.learning_rate <= 0 or self.ridge_alpha <= 0 or self.huber_delta <= 0:
            raise ValueError("learning_rate, ridge_alpha, and huber_delta must be positive")
        if not 0 <= self.effective_number_beta < 1:
            raise ValueError("effective_number_beta must be in [0, 1)")
        if self.tail_weighting_mode not in {"both", "low", "high"}:
            raise ValueError("tail_weighting_mode must be both, low, or high")
        if self.tail_weighting_strength < 0:
            raise ValueError("tail_weighting_strength must be nonnegative")
        if self.max_correction <= 0 or self.contrastive_temperature <= 0:
            raise ValueError("max_correction and contrastive_temperature must be positive")
        if min(self.ordinal_weight, self.auxiliary_weight, self.contrastive_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 < self.uncertainty_coverage <= 1:
            raise ValueError("uncertainty_coverage must be in (0, 1]")


@dataclass(frozen=True)
class FitPredictResult:
    """Fold predictions and auditable initialization evidence."""

    predictions: np.ndarray
    initial_state_hashes: tuple[str, ...]
    final_state_hashes: tuple[str, ...]
    family: str
    seed: int


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training extra is required
        raise RuntimeError("iterative tail candidates require torch") from exc
    return torch


def _as_matrix(value: Any, name: str, columns: int | None = None) -> np.ndarray:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 2 or (columns is not None and array.shape[1] != columns):
        suffix = f" with {columns} columns" if columns is not None else ""
        raise ValueError(f"{name} must be a rank-two matrix{suffix}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _inputs(
    embeddings: Any,
    base_scores: Any,
    extra_features: Any | None,
    *,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    embedding = _as_matrix(embeddings, f"{prefix}_embeddings")
    base = _as_matrix(base_scores, f"{prefix}_base_scores", 3)
    if len(embedding) != len(base):
        raise ValueError(f"{prefix} embedding/base row mismatch")
    if not ((base >= 1.0) & (base <= 5.0)).all():
        raise ValueError(f"{prefix}_base_scores must be within [1, 5]")
    pieces = [embedding, base]
    if extra_features is not None:
        extra = _as_matrix(extra_features, f"{prefix}_extra_features")
        if len(extra) != len(base):
            raise ValueError(f"{prefix} extra-feature row mismatch")
        pieces.append(extra)
    return np.concatenate(pieces, axis=1), base


def _rounded_classes(targets: Any) -> Any:
    torch = _torch()
    return torch.floor(targets + 0.5).long().clamp_(1, 5)


def effective_number_weights(
    targets: Any,
    beta: float = 0.99,
    mode: TailWeightingMode = "both",
    strength: float = 1.0,
) -> Any:
    """Return normalized effective-number weights for both or one score tail.

    ``low`` selects integer bands 1--2 and ``high`` selects bands 4--5.
    Non-selected bands retain unit weight.  ``both`` preserves the original
    inverse-effective-number behavior across all five bands.
    """
    torch = _torch()
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    if mode not in {"both", "low", "high"}:
        raise ValueError("mode must be both, low, or high")
    if strength < 0:
        raise ValueError("strength must be nonnegative")
    y = _rounded_classes(targets)
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError("targets must have shape [rows, 3]")
    axis_weights = []
    for axis in range(3):
        counts = torch.bincount(y[:, axis], minlength=6).float()[1:]
        if beta == 0:
            by_class = torch.ones_like(counts)
        else:
            by_class = torch.where(counts > 0, (1.0 - beta) / (1.0 - beta ** counts), 0.0)
        selected = by_class[y[:, axis] - 1]
        if mode != "both":
            observed = by_class[counts > 0]
            rarity = selected / observed.mean().clamp_min(1e-12)
            tail = y[:, axis] <= 2 if mode == "low" else y[:, axis] >= 4
            selected = torch.where(tail, 1.0 + strength * rarity, torch.ones_like(selected))
        elif strength != 1.0:
            # Strength zero disables weighting; one is exactly backward compatible.
            selected = selected.mean().clamp_min(1e-12) * (selected / selected.mean().clamp_min(1e-12)).pow(strength)
        axis_weights.append(selected)
    result = torch.stack(axis_weights, dim=1).mean(dim=1)
    return result / result.mean().clamp_min(1e-12)


def equal_band_replay_weights(targets: Any) -> Any:
    """Return deterministic inverse-frequency weights across all axis bands."""
    torch = _torch()
    y = _rounded_classes(targets)
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError("targets must have shape [rows, 3]")
    weights = []
    for axis in range(3):
        counts = torch.bincount(y[:, axis], minlength=6).float()[1:].clamp_min(1.0)
        weights.append(counts.reciprocal()[y[:, axis] - 1])
    result = torch.stack(weights, dim=1).mean(dim=1)
    return result / result.mean().clamp_min(1e-12)


def adjacent_score_supervised_contrastive_loss(
    representations: Any,
    targets: Any,
    temperature: float = 0.2,
) -> Any:
    """Supervised contrastive loss whose positives match/neighbor every axis.

    This deliberately compares the three raw axis labels rather than deriving
    or consuming an essay-average label.
    """
    torch = _torch()
    import torch.nn.functional as F

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if representations.ndim != 2 or targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("representations and three-axis targets must be matrices")
    if representations.shape[0] != targets.shape[0]:
        raise ValueError("representation/target row mismatch")
    n = representations.shape[0]
    if n < 2:
        return representations.sum() * 0.0
    z = F.normalize(representations, dim=1)
    logits = z @ z.T / temperature
    diagonal = torch.eye(n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(diagonal, -torch.inf)
    labels = _rounded_classes(targets)
    positives = ((labels[:, None, :] - labels[None, :, :]).abs() <= 1).all(dim=-1) & ~diagonal
    valid = positives.any(dim=1)
    if not bool(valid.any().item()):
        return representations.sum() * 0.0
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_row = -(log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positives.sum(dim=1).clamp_min(1))
    return per_row[valid].mean()


def apply_auxiliary_3v4_correction(
    predictions: Any,
    auxiliary_logits: Any,
    max_correction: float = 0.75,
) -> Any:
    """Apply a bounded 3-vs-4 posterior correction only near those bands.

    The posterior target is ``3 + sigmoid(logit)``.  A triangular proximity
    gate is one at 3.5 and zero outside [2.5, 4.5], preventing this auxiliary
    binary task from changing clear low- or high-tail predictions.
    """
    torch = _torch()
    if max_correction <= 0:
        raise ValueError("max_correction must be positive")
    if predictions.shape != auxiliary_logits.shape or predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("predictions and auxiliary_logits must have shape [rows, 3]")
    posterior_target = 3.0 + torch.sigmoid(auxiliary_logits)
    proximity = (1.0 - (predictions - 3.5).abs()).clamp(0.0, 1.0)
    correction = (posterior_target - predictions).clamp(-max_correction, max_correction)
    return (predictions + proximity * correction).clamp(1.0, 5.0)


def _state_hash(model: Any) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _seed(seed: int, device: Any) -> None:
    torch = _torch()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _ridge_correction(
    x: np.ndarray,
    residual: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
    device: str = "cpu",
) -> np.ndarray:
    """Closed-form float64 ridge, using CUDA when explicitly requested.

    The dual system is used when features outnumber rows, which is the common
    4096-dimensional fold case.  CUDA unavailability or a CUDA solve failure
    retries the identical regularized system on CPU rather than changing the
    estimator.
    """
    torch = _torch()
    requested = torch.device(device)
    solve_device = requested if requested.type == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    try:
        x64 = torch.as_tensor(x, dtype=torch.float64, device=solve_device)
        y64 = torch.as_tensor(residual, dtype=torch.float64, device=solve_device)
        test64 = torch.as_tensor(x_test, dtype=torch.float64, device=solve_device)
        x_mean, y_mean = x64.mean(0), y64.mean(0)
        xc, yc = x64 - x_mean, y64 - y_mean
        if xc.shape[1] > xc.shape[0]:
            system = xc @ xc.T + alpha * torch.eye(len(xc), dtype=torch.float64, device=solve_device)
            prediction = (test64 - x_mean) @ xc.T @ torch.linalg.solve(system, yc) + y_mean
        else:
            system = xc.T @ xc + alpha * torch.eye(xc.shape[1], dtype=torch.float64, device=solve_device)
            prediction = (test64 - x_mean) @ torch.linalg.solve(system, xc.T @ yc) + y_mean
        return prediction.float().cpu().numpy()
    except RuntimeError:
        if solve_device.type != "cuda":
            raise
        return _ridge_correction(x, residual, x_test, alpha, device="cpu")


def _build_model(input_dim: int, hidden_dim: int, dropout: float) -> Any:
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class TailModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
            self.residual_head = nn.Linear(hidden_dim, 3)
            self.ordinal_latent = nn.Linear(hidden_dim, 3)
            self.raw_cutpoint_steps = nn.Parameter(torch.zeros(3, 4))
            self.auxiliary_3v4_head = nn.Linear(hidden_dim, 3)

        def forward(self, value: Any) -> tuple[Any, Any, Any, Any]:
            hidden = self.encoder(value)
            # Positive steps guarantee ordered cutpoints and cumulative logits.
            cuts = torch.cumsum(F.softplus(self.raw_cutpoint_steps) + 1e-3, dim=-1)
            cuts = cuts - cuts.mean(dim=-1, keepdim=True)
            ordinal_logits = self.ordinal_latent(hidden).unsqueeze(-1) - cuts.unsqueeze(0)
            return hidden, self.residual_head(hidden), ordinal_logits, self.auxiliary_3v4_head(hidden)

    return TailModel()


def _weighted_mean(loss: Any, weights: Any) -> Any:
    return (loss * weights[:, None]).sum() / (weights.sum() * loss.shape[1]).clamp_min(1e-12)


def _neural_round(
    spec: CandidateSpec,
    x: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    test_base: np.ndarray,
    round_seed: int,
    external_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, str, str]:
    torch = _torch()
    import torch.nn.functional as F

    device = torch.device(spec.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {spec.device}")
    _seed(round_seed, device)
    model = _build_model(x.shape[1], spec.hidden_dim, spec.dropout).to(device)
    initial_hash = _state_hash(model)

    train_x = torch.as_tensor(x, dtype=torch.float32, device=device)
    infer_x = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y, dtype=torch.float32, device=device)
    train_base = torch.as_tensor(base, dtype=torch.float32, device=device)
    infer_base = torch.as_tensor(test_base, dtype=torch.float32, device=device)
    mean, std = train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(1e-5)
    train_x, infer_x = (train_x - mean) / std, (infer_x - mean) / std
    thresholds = torch.arange(1, 5, device=device).view(1, 1, 4)
    ordinal_targets = (_rounded_classes(train_y).unsqueeze(-1) > thresholds).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    row_weights = torch.ones(len(train_x), device=device)
    if spec.family == "tail_effective_number":
        row_weights = effective_number_weights(
            train_y,
            spec.effective_number_beta,
            spec.tail_weighting_mode,
            spec.tail_weighting_strength,
        )
    elif spec.family == "equal_band_replay":
        row_weights = equal_band_replay_weights(train_y)
    if external_weights is not None:
        supplied = torch.as_tensor(external_weights, dtype=torch.float32, device=device)
        if supplied.ndim != 1 or len(supplied) != len(train_x):
            raise ValueError("train_sample_weights must have one value per training row")
        if not bool(torch.isfinite(supplied).all().item()) or bool((supplied <= 0).any().item()):
            raise ValueError("train_sample_weights must be finite and positive")
        row_weights = row_weights * (supplied / supplied.mean().clamp_min(1e-12))

    for _ in range(spec.epochs):
        model.train()
        hidden, residual, ordinal_logits, auxiliary_logits = model(train_x)
        regression_loss = F.smooth_l1_loss(
            residual, train_y - train_base, beta=spec.huber_delta, reduction="none"
        )
        ordinal_loss = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets, reduction="none").mean(-1)
        if spec.family == "mlp_residual":
            loss = _weighted_mean((residual - (train_y - train_base)).square(), row_weights)
        elif spec.family == "coral":
            loss = _weighted_mean(ordinal_loss, row_weights)
        else:
            loss = _weighted_mean(regression_loss, row_weights) + spec.ordinal_weight * _weighted_mean(ordinal_loss, row_weights)
        if spec.family == "auxiliary_3v4":
            classes = _rounded_classes(train_y)
            mask = (classes == 3) | (classes == 4)
            if bool(mask.any().item()):
                aux_targets = (classes == 4).float()
                loss = loss + spec.auxiliary_weight * F.binary_cross_entropy_with_logits(
                    auxiliary_logits[mask], aux_targets[mask]
                )
        if spec.family == "adjacent_contrastive":
            loss = loss + spec.contrastive_weight * adjacent_score_supervised_contrastive_loss(
                hidden, train_y, spec.contrastive_temperature
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, residual, ordinal_logits, auxiliary_logits = model(infer_x)
        ordinal_probabilities = torch.sigmoid(ordinal_logits)
        ordinal_prediction = 1.0 + ordinal_probabilities.sum(dim=-1)
        bounded = residual.clamp(-spec.max_correction, spec.max_correction)
        if spec.family == "coral":
            prediction = ordinal_prediction
        elif spec.family == "uncertainty_gated":
            # Normalized Bernoulli entropy across the four cumulative heads.
            eps = torch.finfo(ordinal_probabilities.dtype).eps
            p = ordinal_probabilities.clamp(eps, 1.0 - eps)
            entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log()) / math.log(2.0)
            confidence = (1.0 - entropy.mean(dim=-1)).clamp(0.0, 1.0)
            if spec.uncertainty_coverage < 1.0:
                cutoff = torch.quantile(confidence.flatten(), 1.0 - spec.uncertainty_coverage)
                gate = confidence * (confidence >= cutoff)
            else:
                gate = confidence
            prediction = infer_base + gate * bounded
        elif spec.family == "mlp_residual":
            prediction = infer_base + residual
        else:
            regression_prediction = infer_base + bounded
            prediction = 0.5 * (regression_prediction + ordinal_prediction)
        if spec.family == "auxiliary_3v4":
            prediction = apply_auxiliary_3v4_correction(prediction, auxiliary_logits, spec.max_correction)
    return prediction.clamp(1.0, 5.0).cpu().numpy(), initial_hash, _state_hash(model)


def fit_predict(
    spec: CandidateSpec,
    train_embeddings: Any,
    train_base_scores: Any,
    train_targets: Any,
    predict_embeddings: Any,
    predict_base_scores: Any,
    *,
    train_extra_features: Any | None = None,
    predict_extra_features: Any | None = None,
    train_sample_weights: Any | None = None,
) -> FitPredictResult:
    """Fit one fold and predict it without updating any input feature encoder.

    Iterative rounds append the current three predictions to the same frozen
    embedding/structured features, then fit a newly initialized correction.
    ``initial_state_hashes`` proves neural rounds did not reuse model state.
    """
    x_train_static, train_base = _inputs(
        train_embeddings, train_base_scores, train_extra_features, prefix="train"
    )
    x_predict_static, predict_base = _inputs(
        predict_embeddings, predict_base_scores, predict_extra_features, prefix="predict"
    )
    targets = _as_matrix(train_targets, "train_targets", 3)
    if len(targets) != len(train_base):
        raise ValueError("target/training row mismatch")
    if not ((targets >= 1.0) & (targets <= 5.0)).all():
        raise ValueError("train_targets must be within [1, 5]")
    if x_train_static.shape[1] != x_predict_static.shape[1]:
        raise ValueError("training and prediction feature dimensions differ")
    sample_weights: np.ndarray | None = None
    if train_sample_weights is not None:
        sample_weights = np.asarray(train_sample_weights, dtype=np.float32)
        if sample_weights.ndim != 1 or len(sample_weights) != len(train_base):
            raise ValueError("train_sample_weights must have one value per training row")
        if not np.isfinite(sample_weights).all() or np.any(sample_weights <= 0):
            raise ValueError("train_sample_weights must be finite and positive")
        if spec.family == "ridge_residual":
            raise ValueError("ridge_residual does not support train_sample_weights")

    current_train = train_base.copy()
    current_predict = predict_base.copy()
    initial_hashes: list[str] = []
    final_hashes: list[str] = []
    for round_index in range(spec.rounds):
        # Replace the original base-score columns with the latest correction;
        # embeddings and optional structured features remain frozen.
        train_features = x_train_static.copy()
        predict_features = x_predict_static.copy()
        embedding_columns = train_features.shape[1] - 3 - (0 if train_extra_features is None else _as_matrix(train_extra_features, "train_extra_features").shape[1])
        train_features[:, embedding_columns : embedding_columns + 3] = current_train
        predict_features[:, embedding_columns : embedding_columns + 3] = current_predict
        if spec.family == "ridge_residual":
            combined = np.concatenate((train_features, predict_features), axis=0)
            correction = _ridge_correction(
                train_features, targets - current_train, combined, spec.ridge_alpha, spec.device
            )
            train_correction = correction[: len(current_train)]
            predict_correction = correction[len(current_train) :]
            current_train = np.clip(current_train + train_correction, 1.0, 5.0)
            current_predict = np.clip(current_predict + predict_correction, 1.0, 5.0)
        else:
            both_features = np.concatenate((train_features, predict_features), axis=0)
            both_base = np.concatenate((current_train, current_predict), axis=0)
            both_predictions, initial_hash, final_hash = _neural_round(
                spec,
                train_features,
                current_train,
                targets,
                both_features,
                both_base,
                spec.seed + round_index,
                sample_weights,
            )
            current_train = both_predictions[: len(current_train)]
            current_predict = both_predictions[len(current_train) :]
            initial_hashes.append(initial_hash)
            final_hashes.append(final_hash)
    if len(set(initial_hashes)) != len(initial_hashes):
        raise RuntimeError("iterative neural rounds reused an initialization")
    return FitPredictResult(
        predictions=np.asarray(np.clip(current_predict, 1.0, 5.0), dtype=np.float32),
        initial_state_hashes=tuple(initial_hashes),
        final_state_hashes=tuple(final_hashes),
        family=spec.family,
        seed=spec.seed,
    )


__all__ = [
    "CandidateSpec",
    "FAMILIES",
    "FitPredictResult",
    "adjacent_score_supervised_contrastive_loss",
    "apply_auxiliary_3v4_correction",
    "effective_number_weights",
    "equal_band_replay_weights",
    "fit_predict",
]
