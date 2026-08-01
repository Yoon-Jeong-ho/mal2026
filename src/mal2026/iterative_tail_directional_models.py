"""Cross-fitted safe directional residual learners for iterative tail V6.

The learner consumes a predeclared 64-dimensional frozen feature projection
and exact R0 axis scores.  Internal cross-fitting uses the supplied original
fold IDs; no learned expert ever creates a benefit label for one of its own
training rows.  Held-out application accepts no gold values.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


RUN_SEED = 2026080206
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
FAMILY = "crossfit_safe_directional_residual"
CAP_LOW = 0.40
CAP_HIGH = 0.80
CAP_CENTER = 0.08


@dataclass(frozen=True)
class DirectionalSpec:
    cycle: int
    family: str
    variant_id: str
    parameters: Mapping[str, Any]
    seed: int = RUN_SEED
    device: str = "cpu"


@dataclass(frozen=True)
class FittedDirectional:
    spec: DirectionalSpec
    model: Any
    feature_mean: np.ndarray
    feature_std: np.ndarray
    initial_state_hash: str
    final_state_hash: str
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class DirectionalResult:
    predictions: np.ndarray
    audit: Mapping[str, Any]


def _spec(cycle: int, variant: str, *, nonlinear: bool, margin: float, identity_bias: float) -> DirectionalSpec:
    return DirectionalSpec(
        cycle, FAMILY, f"{FAMILY}-{variant}",
        MappingProxyType({
            "nonlinear": nonlinear, "hidden": 64, "benefit_margin": margin,
            "identity_bias": identity_bias, "cap_low": CAP_LOW,
            "cap_high": CAP_HIGH, "cap_center": CAP_CENTER,
        }),
    )


def candidate_specs(*, device: str = "cpu") -> tuple[DirectionalSpec, ...]:
    specs = (
        _spec(1, "primary", nonlinear=True, margin=.01, identity_bias=4.0),
        _spec(2, "conservative", nonlinear=True, margin=.02, identity_bias=4.5),
        _spec(3, "linear-safety-control", nonlinear=False, margin=.01, identity_bias=4.0),
    )
    if device == "cpu":
        return specs
    return tuple(DirectionalSpec(s.cycle, s.family, s.variant_id, s.parameters, s.seed, device) for s in specs)


_REGISTERED = {spec.variant_id: spec for spec in candidate_specs()}


def _validate_spec(spec: DirectionalSpec) -> None:
    registered = _REGISTERED.get(spec.variant_id)
    if (
        registered is None or spec.cycle != registered.cycle or spec.family != registered.family
        or dict(spec.parameters) != dict(registered.parameters) or spec.seed != RUN_SEED
    ):
        raise ValueError("directional spec differs from the exact registered inventory")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("directional learners require torch") from exc
    return torch


def _matrix(value: Any, name: str, columns: int, *, bounded: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != columns or len(result) == 0:
        raise ValueError(f"{name} must be a nonempty N x {columns} matrix; average target is forbidden")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    if bounded and (np.any(result < 1.0) or np.any(result > 5.0)):
        raise ValueError(f"{name} must be bounded [1, 5]")
    return result


def _folds(value: Any, rows: int) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or len(result) != rows or not np.issubdtype(result.dtype, np.integer):
        raise ValueError("train_fold_ids must be one integer ID per training row")
    if len(np.unique(result)) < 2:
        raise ValueError("internal crossfit requires at least two distinct original folds")
    return result.astype(np.int64, copy=False)


def _seed(seed: int, device: Any) -> None:
    torch = _torch()
    torch.manual_seed(seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _state_hash(model: Any) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _body(torch: Any, input_dim: int, nonlinear: bool) -> Any:
    nn = torch.nn
    if nonlinear:
        return nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.LayerNorm(64))
    return nn.Identity()


class _ExpertBase:  # namespace marker for type checkers; concrete class is built lazily
    pass


def _build_components(spec: DirectionalSpec, input_dim: int) -> tuple[Any, Any, Any]:
    torch = _torch()
    nn = torch.nn
    nonlinear = bool(spec.parameters["nonlinear"])
    output_dim = 64 if nonlinear else input_dim

    class Expert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = _body(torch, input_dim, nonlinear)
            self.head = nn.Linear(output_dim, 9)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            # Directional sigmoid heads start close to zero correction, while
            # the signed center head starts at exact zero.
            with torch.no_grad():
                bias = self.head.bias.view(3, 3)
                bias[:, 0] = -4.0
                bias[:, 1] = -4.0
        def forward(self, x: Any) -> Any:
            raw = self.head(self.body(x)).view(-1, 3, 3)
            low = -CAP_LOW * torch.sigmoid(raw[:, :, 0])
            high = CAP_HIGH * torch.sigmoid(raw[:, :, 1])
            center = CAP_CENTER * torch.tanh(raw[:, :, 2])
            return torch.stack((low, high, center), dim=-1)

    class Gate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = _body(torch, input_dim, nonlinear)
            self.head = nn.Linear(output_dim, 12)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            with torch.no_grad():
                self.head.bias.view(3, 4)[:, 0] = float(spec.parameters["identity_bias"])
        def forward(self, x: Any) -> Any:
            return self.head(self.body(x)).view(-1, 3, 4)

    class Combined(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expert = Expert()
            self.gate = Gate()
        def forward(self, x: Any, base: Any) -> tuple[Any, Any, Any]:
            corrections = self.expert(x)
            weights = torch.softmax(self.gate(x), dim=-1)
            prediction = (base + (weights[:, :, 1:] * corrections).sum(-1)).clamp(1.0, 5.0)
            return prediction, corrections, weights

    combined = Combined()
    return combined, Expert, Gate


def _classes(target: Any) -> Any:
    return _torch().floor(target + .5).long().clamp(1, 5)


def _batches(rows: int, seed: int) -> list[Any]:
    torch = _torch()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = []
    for _ in range(EPOCHS):
        order = torch.randperm(rows, generator=generator)
        result.extend(order[start : start + BATCH_SIZE] for start in range(0, rows, BATCH_SIZE))
    return result


def _fit_expert(spec: DirectionalSpec, expert: Any, x: Any, base: Any, target: Any, *, seed: int) -> float:
    torch = _torch()
    import torch.nn.functional as F
    optimizer = torch.optim.AdamW(expert.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    classes = _classes(target)
    final = float("nan")
    expert.train()
    for index in _batches(len(x), seed):
        index = index.to(x.device)
        corrections = expert(x[index])
        residual = target[index] - base[index]
        masks = (classes[index] <= 2, classes[index] == 5, (classes[index] == 3) | (classes[index] == 4))
        losses = []
        for expert_index, mask in enumerate(masks):
            if bool(mask.any().item()):
                desired = residual.clamp(
                    (-CAP_LOW, 0.0, -CAP_CENTER)[expert_index],
                    (0.0, CAP_HIGH, CAP_CENTER)[expert_index],
                )
                losses.append(F.smooth_l1_loss(corrections[:, :, expert_index][mask], desired[mask]))
        loss = sum(losses) / max(1, len(losses)) + 1e-3 * corrections.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expert.parameters(), GRAD_CLIP)
        optimizer.step()
        final = float(loss.detach().cpu().item())
    expert.eval()
    return final


def _regret(prediction: Any, base: Any, target: Any) -> Any:
    torch = _torch()
    classes = _classes(target)
    losses = []
    for mask in (classes <= 2, classes == 5, (classes == 3) | (classes == 4)):
        if bool(mask.any().item()):
            candidate = (prediction[mask] - target[mask]).square().mean()
            reference = (base[mask] - target[mask]).square().mean()
            losses.append(torch.relu(candidate - reference))
    return sum(losses) if losses else prediction.sum() * 0.0


def _fit_gate(spec: DirectionalSpec, gate: Any, x: Any, base: Any, target: Any,
              corrections: Any, labels: Any, *, seed: int) -> float:
    torch = _torch()
    import torch.nn.functional as F
    optimizer = torch.optim.AdamW(gate.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    final = float("nan")
    gate.train()
    for index in _batches(len(x), seed):
        index = index.to(x.device)
        logits = gate(x[index])
        weights = torch.softmax(logits, dim=-1)
        prediction = (base[index] + (weights[:, :, 1:] * corrections[index]).sum(-1)).clamp(1.0, 5.0)
        ce = F.cross_entropy(logits.reshape(-1, 4), labels[index].reshape(-1))
        global_mse = (prediction - target[index]).square().mean()
        regret = _regret(prediction, base[index], target[index])
        classes = _classes(target[index])
        boundary_mask = (classes == 3) | (classes == 4)
        boundary = prediction.sum() * 0.0
        if bool(boundary_mask.any().item()):
            boundary = F.binary_cross_entropy_with_logits(
                ((prediction - 3.5) / .10)[boundary_mask], (classes == 4).float()[boundary_mask]
            )
        loss = ce + global_mse + regret + .10 * boundary
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), GRAD_CLIP)
        optimizer.step()
        final = float(loss.detach().cpu().item())
    gate.eval()
    return final


def fit(spec: DirectionalSpec, train_features: Any, train_base: Any, train_targets: Any,
        train_fold_ids: Any) -> FittedDirectional:
    """Cross-fit benefit labels, train the safe gate, and fresh-refit experts."""
    _validate_spec(spec)
    features = _matrix(train_features, "train_features", 64)
    base_values = _matrix(train_base, "train_base", 3, bounded=True)
    targets = _matrix(train_targets, "train_targets", 3, bounded=True)
    if not (len(features) == len(base_values) == len(targets)):
        raise ValueError("training feature/base/target row mismatch")
    fold_ids = _folds(train_fold_ids, len(features))
    torch = _torch()
    device = torch.device(spec.device)
    _seed(spec.seed, device)
    mean = features.mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(features.std(0, dtype=np.float64).astype(np.float32), 1e-5)
    inputs = np.concatenate(((features - mean) / std, base_values), axis=1)
    tx = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    tb = torch.as_tensor(base_values, dtype=torch.float32, device=device)
    ty = torch.as_tensor(targets, dtype=torch.float32, device=device)

    model, Expert, _ = _build_components(spec, inputs.shape[1])
    model = model.to(device)
    initial_hash = _state_hash(model)
    oof = torch.empty((len(features), 3, 3), dtype=torch.float32, device=device)
    coverage = np.zeros(len(features), dtype=np.int64)
    crossfit_audit = []
    unique_folds = tuple(int(value) for value in np.unique(fold_ids))
    for position, heldout in enumerate(unique_folds):
        train_index = np.flatnonzero(fold_ids != heldout)
        heldout_index = np.flatnonzero(fold_ids == heldout)
        _seed(spec.seed + 100 + position, device)
        expert = Expert().to(device)
        _fit_expert(spec, expert, tx[train_index], tb[train_index], ty[train_index], seed=spec.seed + 100 + position)
        with torch.no_grad():
            oof[heldout_index] = expert(tx[heldout_index])
        coverage[heldout_index] += 1
        crossfit_audit.append(MappingProxyType({
            "heldout_fold": heldout, "train_folds": tuple(value for value in unique_folds if value != heldout),
            "train_records": len(train_index), "heldout_records": len(heldout_index),
        }))
    if not np.all(coverage == 1):
        raise RuntimeError("internal crossfit did not predict every row exactly once")

    with torch.no_grad():
        baseline_error = (ty - tb).square().unsqueeze(-1)
        expert_error = (ty.unsqueeze(-1) - (tb.unsqueeze(-1) + oof)).square()
        benefits = baseline_error - expert_error
        best_benefit, best_expert = benefits.max(dim=-1)
        labels = torch.where(best_benefit > float(spec.parameters["benefit_margin"]), best_expert + 1,
                             torch.zeros_like(best_expert))
    gate_loss = _fit_gate(spec, model.gate, tx, tb, ty, oof.detach(), labels, seed=spec.seed + 10)
    expert_loss = _fit_expert(spec, model.expert, tx, tb, ty, seed=spec.seed + 20)
    model.eval()
    final_hash = _state_hash(model)
    if final_hash == initial_hash:
        raise RuntimeError("directional learner parameters did not update")

    label_counts = torch.bincount(labels.flatten().cpu(), minlength=4).tolist()
    benefit_np = best_benefit.detach().cpu().numpy()
    audit = MappingProxyType({
        "cycle": spec.cycle, "family": spec.family, "variant_id": spec.variant_id,
        "seed": spec.seed, "device": str(device), "train_records": len(features),
        "internal_fold_count": len(unique_folds), "internal_crossfit": tuple(crossfit_audit),
        "crossfit_coverage_min": int(coverage.min()), "crossfit_coverage_max": int(coverage.max()),
        "label_counts": MappingProxyType(dict(zip(("identity", "low", "high", "center"), label_counts, strict=True))),
        "benefit_mean": float(benefit_np.mean()), "benefit_positive_count": int((benefit_np > 0).sum()),
        "benefit_margin_pass_count": int((benefit_np > float(spec.parameters["benefit_margin"])).sum()),
        "expert_final_loss": expert_loss, "gate_final_loss": gate_loss,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
        "fresh_initialization": True, "checkpoint_reused": False, "average_target_used": False,
    })
    return FittedDirectional(spec, model, mean, std, initial_hash, final_hash, audit)


def apply(fitted: FittedDirectional, predict_features: Any, predict_base: Any) -> DirectionalResult:
    """Apply the frozen gate/expert mixture; gold is not accepted."""
    _validate_spec(fitted.spec)
    features = _matrix(predict_features, "predict_features", 64)
    base_values = _matrix(predict_base, "predict_base", 3, bounded=True)
    if len(features) != len(base_values):
        raise ValueError("prediction feature/base row mismatch")
    torch = _torch()
    device = torch.device(fitted.spec.device)
    inputs = np.concatenate(((features - fitted.feature_mean) / fitted.feature_std, base_values), axis=1)
    fitted.model.eval()
    with torch.no_grad():
        prediction, corrections, weights = fitted.model(
            torch.as_tensor(inputs, dtype=torch.float32, device=device),
            torch.as_tensor(base_values, dtype=torch.float32, device=device),
        )
    result = prediction.cpu().numpy().astype(np.float32)
    mean_weights = weights.mean(dim=(0, 1)).cpu().tolist()
    mean_corrections = corrections.mean(dim=(0, 1)).cpu().tolist()
    audit = MappingProxyType({
        "cycle": fitted.spec.cycle, "family": fitted.spec.family, "variant_id": fitted.spec.variant_id,
        "prediction_records": len(result), "mean_identity_weight": float(weights[:, :, 0].mean().cpu().item()),
        "mean_gate_weights": MappingProxyType(dict(zip(
            ("identity", "low", "high", "center"), (float(value) for value in mean_weights), strict=True,
        ))),
        "mean_expert_corrections": MappingProxyType(dict(zip(
            ("low", "high", "center"), (float(value) for value in mean_corrections), strict=True,
        ))),
        "initial_state_hash": fitted.initial_state_hash, "final_state_hash": fitted.final_state_hash,
        "gold_consumed": False, "average_target_used": False,
    })
    return DirectionalResult(result, audit)


__all__ = [
    "BATCH_SIZE", "CAP_CENTER", "CAP_HIGH", "CAP_LOW", "DirectionalResult", "DirectionalSpec",
    "EPOCHS", "FAMILY", "FittedDirectional", "GRAD_CLIP", "LEARNING_RATE", "RUN_SEED",
    "WEIGHT_DECAY", "apply", "candidate_specs", "fit",
]
