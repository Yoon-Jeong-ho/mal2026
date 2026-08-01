"""Fresh GPU learners over frozen Qwen embeddings and exact R0 scores.

Every candidate is initialized from the fixed run seed, trains only against
three axis targets, and predicts by a bounded correction anchored at R0.  The
module deliberately exposes separate ``fit`` and ``apply`` operations so an
outer-fold learner can be frozen before its holdout is touched.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


RUN_SEED = 2026080205
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

FAMILIES = (
    "anchored_multitask_residual",
    "r0_anchored_distributional",
    "joint_tail_boundary_hurdle",
    "axis_coupled_lowrank_moe",
    "band_risk_pareto_residual",
)


@dataclass(frozen=True)
class LearnerSpec:
    cycle: int
    family: str
    variant_id: str
    parameters: Mapping[str, Any]
    seed: int = RUN_SEED
    device: str = "cpu"


@dataclass(frozen=True)
class FittedLearner:
    spec: LearnerSpec
    model: Any
    embedding_mean: np.ndarray
    embedding_std: np.ndarray
    initial_state_hash: str
    final_state_hash: str
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class LearnerResult:
    predictions: np.ndarray
    audit: Mapping[str, Any]


def _spec(cycle: int, family: str, variant: int, **parameters: Any) -> LearnerSpec:
    return LearnerSpec(cycle, family, f"{family}-v{variant}", MappingProxyType(parameters))


def candidate_specs(*, device: str = "cpu") -> tuple[LearnerSpec, ...]:
    """Return the exact registered 5-family by 4-variant inventory."""
    specs = (
        _spec(1, FAMILIES[0], 1, bottleneck=128, hidden=128, cap=.15, band=.25, ordinal=.10, boundary=.10),
        _spec(2, FAMILIES[0], 2, bottleneck=128, hidden=256, cap=.20, band=.50, ordinal=.15, boundary=.20),
        _spec(3, FAMILIES[0], 3, bottleneck=256, hidden=256, cap=.25, band=.75, ordinal=.20, boundary=.30),
        _spec(4, FAMILIES[0], 4, bottleneck=256, hidden=384, cap=.30, band=1.00, ordinal=.25, boundary=.40),
        _spec(5, FAMILIES[1], 1, hidden=128, max_mix=.15, class_weight=.25, margin=.10, temperature=1.0),
        _spec(6, FAMILIES[1], 2, hidden=192, max_mix=.20, class_weight=.50, margin=.15, temperature=.8),
        _spec(7, FAMILIES[1], 3, hidden=256, max_mix=.25, class_weight=.75, margin=.20, temperature=.7),
        _spec(8, FAMILIES[1], 4, hidden=256, max_mix=.30, class_weight=1.00, margin=.30, temperature=.6),
        _spec(9, FAMILIES[2], 1, hidden=128, expert=64, cap=.15, gate=.25, boundary=.25),
        _spec(10, FAMILIES[2], 2, hidden=192, expert=64, cap=.20, gate=.50, boundary=.35),
        _spec(11, FAMILIES[2], 3, hidden=256, expert=96, cap=.25, gate=.75, boundary=.50),
        _spec(12, FAMILIES[2], 4, hidden=256, expert=128, cap=.30, gate=1.00, boundary=.75),
        _spec(13, FAMILIES[3], 1, hidden=128, cap=.15, identity_floor=.70, energy=1e-2, entropy=1e-3),
        _spec(14, FAMILIES[3], 2, hidden=192, cap=.20, identity_floor=.60, energy=5e-3, entropy=1e-3),
        _spec(15, FAMILIES[3], 3, hidden=256, cap=.25, identity_floor=.50, energy=2e-3, entropy=5e-4),
        _spec(16, FAMILIES[3], 4, hidden=256, cap=.30, identity_floor=.40, energy=1e-3, entropy=2e-4),
        _spec(17, FAMILIES[4], 1, hidden=128, cap=.15, risk=.10, temperature=.20, boundary=.10, ranking=.02),
        _spec(18, FAMILIES[4], 2, hidden=192, cap=.20, risk=.20, temperature=.15, boundary=.20, ranking=.03),
        _spec(19, FAMILIES[4], 3, hidden=256, cap=.25, risk=.30, temperature=.10, boundary=.30, ranking=.05),
        _spec(20, FAMILIES[4], 4, hidden=256, cap=.30, risk=.40, temperature=.075, boundary=.40, ranking=.075),
    )
    if device == "cpu":
        return specs
    return tuple(LearnerSpec(s.cycle, s.family, s.variant_id, s.parameters, s.seed, device) for s in specs)


_REGISTERED = {spec.variant_id: spec for spec in candidate_specs()}


def _validate_spec(spec: LearnerSpec) -> None:
    registered = _REGISTERED.get(spec.variant_id)
    if (
        registered is None
        or spec.cycle != registered.cycle
        or spec.family != registered.family
        or dict(spec.parameters) != dict(registered.parameters)
        or spec.seed != RUN_SEED
    ):
        raise ValueError("learner spec differs from the exact registered inventory")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("iterative tail learners require torch") from exc
    return torch


def _matrix(value: Any, name: str, *, columns: int | None = None, bounded: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or len(result) == 0 or (columns is not None and result.shape[1] != columns):
        suffix = f" with exactly {columns} columns" if columns is not None else ""
        raise ValueError(f"{name} must be a nonempty rank-two matrix{suffix}; average target is forbidden")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    if bounded and (np.any(result < 1.0) or np.any(result > 5.0)):
        raise ValueError(f"{name} must be bounded [1, 5]")
    return result


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


def _stem(torch: Any, input_dim: int, bottleneck: int, hidden: int) -> Any:
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(input_dim, bottleneck), nn.GELU(), nn.LayerNorm(bottleneck),
        nn.Linear(bottleneck, hidden), nn.GELU(),
    )


def _build_model(spec: LearnerSpec, input_dim: int) -> Any:
    torch = _torch()
    nn = torch.nn
    p = spec.parameters

    if spec.family == FAMILIES[0]:
        class AnchoredMultitask(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = _stem(torch, input_dim, int(p["bottleneck"]), int(p["hidden"]))
                self.delta = nn.Linear(int(p["hidden"]), 3)
                self.ordinal = nn.Linear(int(p["hidden"]), 12)
                self.boundary = nn.Linear(int(p["hidden"]), 3)
            def forward(self, x: Any) -> Mapping[str, Any]:
                h = self.stem(x)
                return {"raw": self.delta(h), "ordinal": self.ordinal(h).view(-1, 3, 4),
                        "boundary": self.boundary(h)}
        return AnchoredMultitask()

    if spec.family == FAMILIES[1]:
        class AnchoredDistributional(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = _stem(torch, input_dim, int(p["hidden"]), int(p["hidden"]))
                self.logits = nn.Linear(int(p["hidden"]), 15)
                self.mix = nn.Linear(int(p["hidden"]), 3)
            def forward(self, x: Any) -> Mapping[str, Any]:
                h = self.stem(x)
                return {"logits": self.logits(h).view(-1, 3, 5), "mix": self.mix(h)}
        return AnchoredDistributional()

    if spec.family == FAMILIES[2]:
        class JointHurdle(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = _stem(torch, input_dim, int(p["hidden"]), int(p["hidden"]))
                self.gates = nn.Linear(int(p["hidden"]), 12)
                self.experts = nn.ModuleList(nn.Sequential(nn.Linear(int(p["hidden"]), int(p["expert"])),
                                                           nn.GELU(), nn.Linear(int(p["expert"]), 3)) for _ in range(4))
                self.boundary = nn.Linear(int(p["hidden"]), 3)
            def forward(self, x: Any) -> Mapping[str, Any]:
                h = self.stem(x)
                return {"gates": self.gates(h).view(-1, 3, 4),
                        "experts": torch.stack([expert(h) for expert in self.experts], dim=-1),
                        "boundary": self.boundary(h)}
        return JointHurdle()

    if spec.family == FAMILIES[3]:
        class AxisCoupledMoE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                hidden = int(p["hidden"])
                self.stem = _stem(torch, input_dim, hidden, hidden)
                self.gates = nn.Linear(hidden, 12)
                self.factors = nn.Linear(hidden, 8)
                self.axis_loadings = nn.Parameter(torch.empty(3, 2))
                self.axis_specific = nn.Linear(hidden, 9)
                nn.init.normal_(self.axis_loadings, std=.02)
            def forward(self, x: Any) -> Mapping[str, Any]:
                h = self.stem(x)
                factors = self.factors(h).view(-1, 4, 2)
                shared = torch.einsum("nkr,ar->nak", factors, self.axis_loadings)
                specific = self.axis_specific(h).view(-1, 3, 3)
                experts = torch.cat((torch.zeros_like(shared[:, :, :1]), shared[:, :, 1:] + specific), dim=-1)
                return {"gates": self.gates(h).view(-1, 3, 4), "experts": experts}
        return AxisCoupledMoE()

    class BandRiskPareto(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = _stem(torch, input_dim, int(p["hidden"]), int(p["hidden"]))
            self.delta = nn.Linear(int(p["hidden"]), 3)
            self.boundary = nn.Linear(int(p["hidden"]), 3)
        def forward(self, x: Any) -> Mapping[str, Any]:
            h = self.stem(x)
            return {"raw": self.delta(h), "boundary": self.boundary(h)}
    return BandRiskPareto()


def _classes(target: Any) -> Any:
    return _torch().floor(target + .5).long().clamp(1, 5)


def _boundary_loss(logits: Any, target: Any) -> Any:
    torch = _torch()
    import torch.nn.functional as F
    classes = _classes(target)
    mask = (classes == 3) | (classes == 4)
    if not bool(mask.any().item()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], (classes == 4).float()[mask])


def _band_risks(prediction: Any, target: Any) -> Any:
    torch = _torch()
    classes = _classes(target)
    risks = []
    for axis in range(3):
        for band in range(1, 6):
            mask = classes[:, axis] == band
            if bool(mask.any().item()):
                risks.append((prediction[mask, axis] - target[mask, axis]).square().mean())
    return torch.stack(risks)


def _prediction(spec: LearnerSpec, output: Mapping[str, Any], base: Any) -> Any:
    torch = _torch()
    p = spec.parameters
    if spec.family == FAMILIES[1]:
        logits = output["logits"] / float(p["temperature"])
        expected = (torch.softmax(logits, dim=-1) * torch.arange(1, 6, device=base.device)).sum(-1)
        mix = float(p["max_mix"]) * torch.sigmoid(output["mix"])
        return (base + mix * (expected - base)).clamp(1.0, 5.0)
    if spec.family == FAMILIES[2]:
        weights = torch.softmax(output["gates"], dim=-1)
        raw = (weights * output["experts"]).sum(-1)
    elif spec.family == FAMILIES[3]:
        soft = torch.softmax(output["gates"], dim=-1)
        floor = float(p["identity_floor"])
        weights = torch.cat((floor + (1.0 - floor) * soft[:, :, :1],
                             (1.0 - floor) * soft[:, :, 1:]), dim=-1)
        raw = (weights * output["experts"]).sum(-1)
    else:
        raw = output["raw"]
    cap = float(p["cap"])
    return (base + cap * torch.tanh(raw / cap)).clamp(1.0, 5.0)


def _loss(spec: LearnerSpec, output: Mapping[str, Any], prediction: Any, base: Any, target: Any) -> Any:
    torch = _torch()
    import torch.nn.functional as F
    p = spec.parameters
    global_mse = (prediction - target).square().mean()
    if spec.family == FAMILIES[0]:
        thresholds = torch.arange(1, 5, device=target.device).view(1, 1, 4)
        ordinal_target = (_classes(target).unsqueeze(-1) > thresholds).float()
        ordinal = F.binary_cross_entropy_with_logits(output["ordinal"], ordinal_target)
        return (global_mse + float(p["band"]) * _band_risks(prediction, target).mean()
                + float(p["ordinal"]) * ordinal + float(p["boundary"]) * _boundary_loss(output["boundary"], target))
    if spec.family == FAMILIES[1]:
        classes = _classes(target) - 1
        counts = torch.stack([torch.bincount(classes[:, axis], minlength=5) for axis in range(3)]).float()
        weights = counts.clamp_min(1).rsqrt()
        weights = weights / weights.mean(dim=1, keepdim=True)
        ce = sum(F.cross_entropy(output["logits"][:, axis], classes[:, axis], weight=weights[axis]) for axis in range(3)) / 3
        margin = _boundary_loss(output["logits"][:, :, 3] - output["logits"][:, :, 2], target)
        return global_mse + float(p["class_weight"]) * ce + float(p["margin"]) * margin
    if spec.family == FAMILIES[2]:
        classes = _classes(target)
        zones = torch.where(classes <= 2, 0, torch.where(classes == 3, 1, torch.where(classes == 4, 2, 3)))
        gate_loss = F.cross_entropy(output["gates"].reshape(-1, 4), zones.reshape(-1))
        return global_mse + float(p["gate"]) * gate_loss + float(p["boundary"]) * _boundary_loss(output["boundary"], target)
    if spec.family == FAMILIES[3]:
        probs = torch.softmax(output["gates"], dim=-1)
        energy = (prediction - base).square().mean()
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean()
        return global_mse + float(p["energy"]) * energy - float(p["entropy"]) * entropy
    risks = _band_risks(prediction, target)
    tau = float(p["temperature"])
    smooth_worst = tau * torch.logsumexp(risks / tau, dim=0)
    rank = prediction.sum() * 0.0
    difference = target[:, None, :] - target[None, :, :]
    mask = difference.abs() >= .5
    if bool(mask.any().item()):
        predicted_difference = prediction[:, None, :] - prediction[None, :, :]
        rank = F.relu(.1 - difference.sign() * predicted_difference)[mask].mean()
    return (global_mse + float(p["risk"]) * smooth_worst
            + float(p["boundary"]) * _boundary_loss(output["boundary"], target) + float(p["ranking"]) * rank)


def fit(spec: LearnerSpec, train_embeddings: Any, train_base_scores: Any, train_targets: Any) -> FittedLearner:
    """Fresh-fit one registered learner; no checkpoint or prediction rows are consumed."""
    _validate_spec(spec)
    embeddings = _matrix(train_embeddings, "train_embeddings")
    base = _matrix(train_base_scores, "train_base_scores", columns=3, bounded=True)
    targets = _matrix(train_targets, "train_targets", columns=3, bounded=True)
    if not (len(embeddings) == len(base) == len(targets)):
        raise ValueError("training embedding/base/target row mismatch")
    torch = _torch()
    device = torch.device(spec.device)
    _seed(spec.seed, device)
    mean = embeddings.mean(0, dtype=np.float64).astype(np.float32)
    std = embeddings.std(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-5)
    normalized = (embeddings - mean) / std
    x = np.concatenate((normalized, base), axis=1)
    model = _build_model(spec, x.shape[1]).to(device)
    initial_hash = _state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_x = torch.as_tensor(x, dtype=torch.float32, device=device)
    train_base = torch.as_tensor(base, dtype=torch.float32, device=device)
    train_target = torch.as_tensor(targets, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    final_loss = float("nan")
    for _ in range(EPOCHS):
        order = torch.randperm(len(x), generator=generator)
        model.train()
        for start in range(0, len(x), BATCH_SIZE):
            index = order[start : start + BATCH_SIZE].to(device)
            output = model(train_x[index])
            prediction = _prediction(spec, output, train_base[index])
            loss = _loss(spec, output, prediction, train_base[index], train_target[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
    final_hash = _state_hash(model)
    if initial_hash == final_hash:
        raise RuntimeError("learner parameters did not update")
    model.eval()
    audit = MappingProxyType({
        "cycle": spec.cycle, "family": spec.family, "variant_id": spec.variant_id,
        "seed": spec.seed, "device": str(device), "train_records": len(x),
        "embedding_dimensions": embeddings.shape[1], "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
        "fresh_initialization": True, "checkpoint_reused": False, "average_target_used": False,
        "train_final_loss": final_loss,
    })
    return FittedLearner(spec, model, mean, std, initial_hash, final_hash, audit)


def apply(fitted: FittedLearner, predict_embeddings: Any, predict_base_scores: Any) -> LearnerResult:
    """Apply a frozen learner without accepting or consulting gold targets."""
    _validate_spec(fitted.spec)
    embeddings = _matrix(predict_embeddings, "predict_embeddings")
    base = _matrix(predict_base_scores, "predict_base_scores", columns=3, bounded=True)
    if len(embeddings) != len(base) or embeddings.shape[1] != len(fitted.embedding_mean):
        raise ValueError("prediction embedding/base shape differs from fitted learner")
    torch = _torch()
    device = torch.device(fitted.spec.device)
    x = np.concatenate(((embeddings - fitted.embedding_mean) / fitted.embedding_std, base), axis=1)
    fitted.model.eval()
    with torch.no_grad():
        tx = torch.as_tensor(x, dtype=torch.float32, device=device)
        tb = torch.as_tensor(base, dtype=torch.float32, device=device)
        prediction = _prediction(fitted.spec, fitted.model(tx), tb).cpu().numpy().astype(np.float32)
    audit = MappingProxyType({
        "cycle": fitted.spec.cycle, "family": fitted.spec.family, "variant_id": fitted.spec.variant_id,
        "prediction_records": len(prediction), "initial_state_hash": fitted.initial_state_hash,
        "final_state_hash": fitted.final_state_hash, "gold_consumed": False, "average_target_used": False,
    })
    return LearnerResult(prediction, audit)


def fit_predict(
    spec: LearnerSpec,
    train_embeddings: Any,
    train_base_scores: Any,
    train_targets: Any,
    predict_embeddings: Any,
    predict_base_scores: Any,
) -> LearnerResult:
    return apply(fit(spec, train_embeddings, train_base_scores, train_targets),
                 predict_embeddings, predict_base_scores)


__all__ = [
    "BATCH_SIZE", "EPOCHS", "FAMILIES", "FittedLearner", "GRAD_CLIP", "LEARNING_RATE",
    "LearnerResult", "LearnerSpec", "RUN_SEED", "WEIGHT_DECAY", "apply", "candidate_specs",
    "fit", "fit_predict",
]
