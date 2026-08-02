"""Train-only nested ordinal heads over the frozen exact-R0 feature artifact.

This module owns no text encoder.  It consumes the checksum-bound, score-blind
R0 embeddings and their exact OOF folds, fits every statistic inside the
current fit fold, and persists row-level predictions only below the configured
restricted output root.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .iterative_tail_metrics import (
    AXES, compute_iterative_tail_metrics, metric_improvements, paired_bootstrap_delta_ci,
)
from .r0_ordinal_residual import ResidualRow, load_embedding_artifact


FAMILIES = ("softmax_ce", "rps", "coral", "corn", "slace")
MODES = ("smoke", "outer_fold", "full")
EXPECTED_IDS = (
    "ce-natural", "rps-natural", "coral-natural", "corn-natural",
    "slace-a0.5", "slace-a1", "slace-a2", "ce-effective-b0.99",
    "ce-effective-b0.999", "ce-sqrt-sampler",
)


class OrdinalTailFixedFeatureError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise OrdinalTailFixedFeatureError(message)


def file_sha256(path: str | Path) -> str:
    value = Path(path)
    need(value.is_file() and not value.is_symlink(), "artifact must be an ordinary file")
    digest = sha256()
    with value.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CandidateSpec:
    identifier: str
    family: str
    prior_treatment: str
    alpha: float | None = None
    beta: float | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CandidateSpec":
        need(isinstance(raw, Mapping), "candidate must be an object")
        allowed = {"id", "family", "prior_treatment", "alpha", "beta"}
        need(set(raw) <= allowed and {"id", "family", "prior_treatment"} <= set(raw),
             "candidate fields differ")
        value = cls(str(raw["id"]), str(raw["family"]), str(raw["prior_treatment"]),
                    None if "alpha" not in raw else float(raw["alpha"]),
                    None if "beta" not in raw else float(raw["beta"]))
        value.validate()
        return value

    def validate(self) -> None:
        need(self.family in FAMILIES, "candidate family differs")
        if self.family == "slace":
            need(self.prior_treatment == "slace_internal" and self.alpha in {0.5, 1.0, 2.0}
                 and self.beta is None, "SLACE contract differs")
        elif self.prior_treatment == "effective_number":
            need(self.family == "softmax_ce" and self.beta in {0.99, 0.999}
                 and self.alpha is None, "effective-number contract differs")
        elif self.prior_treatment == "sqrt_sampler":
            need(self.family == "softmax_ce" and self.alpha is self.beta is None,
                 "sqrt sampler contract differs")
        else:
            need(self.prior_treatment == "natural" and self.alpha is self.beta is None,
                 "natural-prior candidate differs")


@dataclass(frozen=True)
class FixedFeatureConfig:
    schema_version: str
    run_id: str
    train_path: str
    train_sha256: str
    embedding_manifest_path: str
    embedding_rows_path: str
    embedding_rows_sha256: str
    r0_oof_prediction_path: str
    r0_oof_prediction_sha256: str
    output_root: str
    restricted_output_root: str
    seed: int
    outer_folds: int
    inner_folds: int
    candidates: tuple[CandidateSpec, ...]
    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    raw_rmse_auxiliary_weight: float
    promotion_gate: Mapping[str, Any]
    config_sha256: str
    primary_rmse_uses_natural_prior_correction: bool
    hybrid_objective_disclosure: str
    screen_selection: str
    phase2_recommendation: str

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        root: str | Path = ".",
        config_sha256: str | None = None,
    ) -> "FixedFeatureConfig":
        need(raw.get("schema_version") == "mal2026-ordinal-tail-program-v1", "program schema differs")
        need(raw.get("axes") == list(AXES) and raw.get("average_target_forbidden") is True,
             "three-axis/average contract differs")
        need(raw.get("validation_selection_forbidden") is True, "validation selection must be forbidden")
        block = raw.get("fixed_feature_candidates")
        need(isinstance(block, Mapping), "fixed-feature block missing")
        candidates = tuple(CandidateSpec.from_mapping(item) for item in block.get("candidates", ()))
        value = cls(
            schema_version=str(raw["schema_version"]), run_id=str(raw["run_id"]),
            train_path=str(Path(root) / str(raw["train_path"])), train_sha256=str(raw["train_sha256"]),
            embedding_manifest_path=str(Path(root) / str(raw["r0_embedding_manifest_path"])),
            embedding_rows_path=str(Path(root) / str(raw["r0_embedding_rows_path"])),
            embedding_rows_sha256=str(raw["r0_embedding_rows_sha256"]),
            r0_oof_prediction_path=str(Path(root) / str(raw["r0_oof_prediction_path"])),
            r0_oof_prediction_sha256=str(raw["r0_oof_prediction_sha256"]),
            output_root=str(Path(root) / str(raw["output_root"])),
            restricted_output_root=str(Path(root) / str(raw["restricted_output_root"])),
            seed=int(raw["seed"]), outer_folds=int(raw["outer_folds"]), inner_folds=int(raw["inner_folds"]),
            candidates=candidates, hidden_dim=int(block["hidden_dim"]), dropout=float(block["dropout"]),
            learning_rate=float(block["learning_rate"]), weight_decay=float(block["weight_decay"]),
            epochs=int(block["epochs"]), batch_size=int(block["batch_size"]),
            raw_rmse_auxiliary_weight=float(block["raw_rmse_auxiliary_weight"]),
            promotion_gate=dict(raw["promotion_gate"]),
            config_sha256=(config_sha256 or sha256(json.dumps(
                raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()),
            primary_rmse_uses_natural_prior_correction=bool(block["primary_rmse_uses_natural_prior_correction"]),
            hybrid_objective_disclosure=str(block["hybrid_objective_disclosure"]),
            screen_selection=str(block["screen_selection"]),
            phase2_recommendation=str(block["phase2_recommendation"]),
        )
        value.validate()
        return value

    @classmethod
    def from_json(cls, path: str | Path, *, root: str | Path = ".") -> "FixedFeatureConfig":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OrdinalTailFixedFeatureError("fixed-feature config is unreadable") from exc
        need(isinstance(raw, Mapping), "fixed-feature config must be an object")
        return cls.from_mapping(raw, root=root, config_sha256=file_sha256(path))

    def validate(self) -> None:
        need(self.outer_folds == 5 and self.inner_folds == 4, "nested fold contract differs")
        need(tuple(item.identifier for item in self.candidates) == EXPECTED_IDS,
             "fixed-feature inventory/order differs")
        need((self.hidden_dim, self.dropout, self.learning_rate, self.weight_decay,
              self.epochs, self.batch_size, self.raw_rmse_auxiliary_weight) ==
             (64, 0.1, 0.001, 0.0001, 24, 128, 0.25), "fixed-feature hyperparameters differ")
        need(len(self.embedding_rows_sha256) == len(self.r0_oof_prediction_sha256) == 64,
             "input hash differs")
        need(len(self.config_sha256) == 64, "config hash differs")
        need(Path(self.output_root).resolve() != Path(self.restricted_output_root).resolve(),
             "public and restricted outputs must differ")
        need(self.primary_rmse_uses_natural_prior_correction, "primary decoding contract differs")
        need(self.hybrid_objective_disclosure.startswith("every candidate uses ordinal loss plus 0.25 raw-score MSE"),
             "hybrid objective disclosure differs")
        need(self.screen_selection == "within each outer fold select lowest inner-OOF macro RMSE; retain exact R0 as protected output when promotion gate fails",
             "screen selection contract differs")
        need(self.phase2_recommendation.startswith("two distinct families"), "phase-2 recommendation contract differs")
        required_gate = {
            "minimum_macro_rmse_improvement", "maximum_axis_rmse_worsening",
            "low_tail_must_not_worsen", "high_tail_must_not_worsen",
            "maximum_gold_3_4_balanced_accuracy_drop", "maximum_spearman_drop",
            "paired_bootstrap_resamples", "paired_bootstrap_rmse_lower_bound_above_zero",
        }
        need(set(self.promotion_gate) == required_gate, "promotion gate fields differ")


def candidate_specs(config: FixedFeatureConfig) -> tuple[CandidateSpec, ...]:
    return config.candidates


def class_counts(labels: Any) -> Any:
    import torch
    value = torch.as_tensor(labels, dtype=torch.long)
    need(value.ndim == 1 and bool(((value >= 1) & (value <= 5)).all()), "labels must be 1..5")
    counts = torch.bincount(value, minlength=6)[1:].float()
    need(bool((counts > 0).all()), "every fold-fit class needs support")
    return counts


def natural_prior(labels: Any) -> Any:
    counts = class_counts(labels)
    return counts / counts.sum()


def effective_number_weights(labels: Any, beta: float) -> Any:
    import torch
    need(beta in {0.99, 0.999}, "effective-number beta differs")
    counts = class_counts(labels)
    weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), counts))
    return weights / (weights * (counts / counts.sum())).sum()


def sampling_prior(labels: Any, spec: CandidateSpec) -> Any:
    import torch
    counts = class_counts(labels)
    if spec.prior_treatment == "sqrt_sampler":
        mass = torch.sqrt(counts)
    elif spec.prior_treatment == "effective_number":
        mass = counts * effective_number_weights(labels, float(spec.beta))
    else:
        mass = counts
    return mass / mass.sum()


def natural_prior_correction(pmf: Any, fit_prior: Any, observed_prior: Any) -> Any:
    """Convert a posterior learned under ``observed_prior`` to fit natural prior."""
    import torch
    value = torch.as_tensor(pmf).float()
    natural = torch.as_tensor(fit_prior, device=value.device).float()
    observed = torch.as_tensor(observed_prior, device=value.device).float()
    need(value.ndim == 2 and value.shape[1] == 5, "PMF must be [N,5]")
    need(bool((value >= 0).all()) and bool(torch.isfinite(value).all()), "PMF is invalid")
    need(natural.shape == observed.shape == (5,) and bool((natural > 0).all()) and bool((observed > 0).all()),
         "prior support differs")
    adjusted = value * (natural / observed).unsqueeze(0)
    return adjusted / adjusted.sum(dim=1, keepdim=True).clamp_min(1e-12)


def rps_loss(pmf: Any, labels: Any) -> Any:
    import torch
    value = torch.as_tensor(pmf).float()
    target = torch.nn.functional.one_hot(torch.as_tensor(labels).long() - 1, 5).float()
    need(value.shape == target.shape and bool((value >= 0).all()), "RPS tensors differ")
    return ((value.cumsum(1)[:, :-1] - target.cumsum(1)[:, :-1]) ** 2).mean()


def coral_pmf(logits: Any) -> Any:
    import torch
    q = torch.sigmoid(torch.as_tensor(logits).float())
    need(q.ndim == 2 and q.shape[1] == 4, "CORAL logits must be [N,4]")
    # Defense in depth if a caller supplies unconstrained logits.
    q = torch.cummin(q, dim=1).values
    return torch.cat((1.0 - q[:, :1], q[:, :-1] - q[:, 1:], q[:, -1:]), dim=1)


def corn_targets(labels: Any) -> tuple[Any, Any]:
    import torch
    y = torch.as_tensor(labels).long()
    thresholds = torch.arange(1, 5, device=y.device).view(1, 4)
    mask = y.view(-1, 1) > (thresholds - 1)
    target = (y.view(-1, 1) > thresholds).float()
    return target, mask


def corn_pmf(logits: Any) -> Any:
    import torch
    conditional = torch.sigmoid(torch.as_tensor(logits).float())
    need(conditional.ndim == 2 and conditional.shape[1] == 4, "CORN logits must be [N,4]")
    q = torch.cumprod(conditional, dim=1)
    return torch.cat((1.0 - q[:, :1], q[:, :-1] - q[:, 1:], q[:, -1:]), dim=1)


def slace_components(prior: Any, alpha: float) -> tuple[Any, Any, Any]:
    """Build count-aware directed SLACE distances, soft labels, and masks.

    The fit-fold prior is sufficient because the published proximity divides
    count-path mass by total count; a common count scale cancels exactly.
    Returned tensors are indexed as ``[gold, candidate]`` and
    ``[gold, candidate, accumulated_class]``.
    """
    import torch
    need(alpha in {0.5, 1.0, 2.0}, "SLACE alpha differs")
    p = torch.as_tensor(prior).float()
    need(p.shape == (5,) and bool((p > 0).all()), "SLACE prior support differs")
    distance = torch.empty((5, 5), dtype=torch.float32, device=p.device)
    for gold in range(5):
        for candidate in range(5):
            step = 1 if gold >= candidate else -1
            path = list(range(candidate, gold + step, step))
            mass = p[candidate] / 2.0
            if len(path) > 1:
                mass = mass + p[path[1:]].sum()
            distance[gold, candidate] = -torch.log(mass.clamp_min(1e-12))
    phi = distance.max(dim=1, keepdim=True).values - distance
    soft = torch.softmax(-alpha * phi, dim=1)
    mask = (distance[:, None, :] >= distance[:, :, None]).float()
    return distance, soft, mask


def slace_loss(logits: Any, labels: Any, prior: Any, alpha: float) -> Any:
    import torch
    value = torch.as_tensor(logits).float()
    _, soft, mask = slace_components(torch.as_tensor(prior, device=value.device), alpha)
    target_index = torch.as_tensor(labels, device=value.device).long() - 1
    mass_weights = soft[target_index]
    accumulating = torch.bmm(mask[target_index], torch.softmax(value, dim=1).unsqueeze(2)).squeeze(2)
    loss = -(mass_weights * torch.log(accumulating.clamp_min(1e-9))).sum(1).mean()
    need(bool(torch.isfinite(loss)), "SLACE loss is non-finite")
    return loss


def nested_indices(rows: Sequence[ResidualRow], outer_fold: int) -> tuple[tuple[int, ...], dict[int, tuple[tuple[int, ...], tuple[int, ...]]]]:
    need(0 <= outer_fold < 5 and len(rows) == 2000, "outer fold/population differs")
    outer = tuple(index for index, row in enumerate(rows) if row.oof_fold == outer_fold)
    remaining = tuple(fold for fold in range(5) if fold != outer_fold)
    inner: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for dev_fold in remaining:
        dev = tuple(index for index, row in enumerate(rows) if row.oof_fold == dev_fold)
        train = tuple(index for index, row in enumerate(rows) if row.oof_fold not in {outer_fold, dev_fold})
        inner[dev_fold] = train, dev
    need(len(outer) == 400 and all(len(train) == 1200 and len(dev) == 400 for train, dev in inner.values()),
         "nested 5x4 sizes differ")
    need(set(outer).isdisjoint(index for pair in inner.values() for part in pair for index in part),
         "outer fold entered inner selection")
    return outer, inner


def _pmf_valid(pmf: Any) -> None:
    import torch
    need(pmf.ndim == 2 and pmf.shape[1] == 5 and bool(torch.isfinite(pmf).all())
         and bool((pmf >= -1e-7).all()) and bool(torch.allclose(pmf.sum(1), torch.ones(len(pmf), device=pmf.device), atol=1e-5)),
         "model PMF is invalid")


def build_axis_model(input_dim: int, spec: CandidateSpec, hidden_dim: int = 64, dropout: float = 0.1) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    need(input_dim > 0 and hidden_dim > 0, "model dimensions differ")

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
            if spec.family == "coral":
                self.score = nn.Linear(hidden_dim, 1)
                self.cut_base = nn.Parameter(torch.tensor(-1.0))
                self.cut_gaps = nn.Parameter(torch.zeros(3))
            else:
                self.head = nn.Linear(hidden_dim, 4 if spec.family == "corn" else 5)

        def forward(self, features: Any) -> Mapping[str, Any]:
            hidden = self.encoder(features.float())
            if spec.family == "coral":
                cuts = torch.cat((self.cut_base.view(1), self.cut_base + torch.cumsum(functional.softplus(self.cut_gaps), 0)))
                logits = self.score(hidden) - cuts.view(1, 4)
                pmf = coral_pmf(logits)
            else:
                logits = self.head(hidden)
                pmf = corn_pmf(logits) if spec.family == "corn" else torch.softmax(logits, dim=1)
            return {"logits": logits, "pmf": pmf}

    return Model()


def _features(rows: Sequence[ResidualRow], indices: Sequence[int], axis: int) -> Any:
    import torch
    # The R0 OOF score is comparison-only: an inner-fit row's R0 model may have
    # seen the current outer fold, so attaching it would breach outer isolation.
    return torch.tensor([rows[index].shared_embedding for index in indices], dtype=torch.float32)


def _labels(rows: Sequence[ResidualRow], indices: Sequence[int], axis: int) -> tuple[Any, Any]:
    import torch
    return (torch.tensor([rows[index].labels[axis] for index in indices], dtype=torch.long),
            torch.tensor([rows[index].raw_labels[axis] for index in indices], dtype=torch.float32))


def _loss(output: Mapping[str, Any], labels: Any, raw: Any, spec: CandidateSpec,
          prior: Any, config: FixedFeatureConfig, *, class_weights: Any | None = None) -> Any:
    import torch
    import torch.nn.functional as functional
    pmf, logits = output["pmf"], output["logits"]
    if spec.family == "rps":
        ordinal = rps_loss(pmf, labels)
    elif spec.family == "coral":
        target = (labels[:, None] > torch.arange(1, 5, device=labels.device)).float()
        ordinal = functional.binary_cross_entropy_with_logits(logits, target)
    elif spec.family == "corn":
        target, mask = corn_targets(labels)
        ordinal = functional.binary_cross_entropy_with_logits(logits[mask], target[mask])
    elif spec.family == "slace":
        ordinal = slace_loss(logits, labels, prior, float(spec.alpha))
    else:
        ordinal = functional.cross_entropy(logits, labels - 1, weight=class_weights)
    expected = (pmf * torch.arange(1, 6, device=pmf.device).float()).sum(1)
    return ordinal + config.raw_rmse_auxiliary_weight * functional.mse_loss(expected, raw)


def _train_predict(rows: Sequence[ResidualRow], fit: Sequence[int], predict: Sequence[int], axis: int,
                   spec: CandidateSpec, config: FixedFeatureConfig, *, seed: int, smoke: bool = False) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, (y, raw) = _features(rows, fit, axis), _labels(rows, fit, axis)
    fit_prior = natural_prior(y)
    spec_prior = sampling_prior(y, spec)
    class_weights = (effective_number_weights(y, float(spec.beta)).to(device)
                     if spec.prior_treatment == "effective_number" else None)
    dataset = TensorDataset(x, y, raw)
    generator = torch.Generator().manual_seed(seed)
    if spec.prior_treatment == "sqrt_sampler":
        counts = class_counts(y)
        sample_weights = torch.tensor([1.0 / math.sqrt(float(counts[int(label) - 1])) for label in y])
        sampler = WeightedRandomSampler(sample_weights, len(dataset), replacement=True, generator=generator)
        loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, generator=generator)
    else:
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    model = build_axis_model(x.shape[1], spec, config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    epochs = 1 if smoke else config.epochs
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y, batch_raw in loader:
            batch_x, batch_y, batch_raw = batch_x.to(device), batch_y.to(device), batch_raw.to(device)
            loss = _loss(model(batch_x), batch_y, batch_raw, spec, fit_prior.to(device), config,
                         class_weights=class_weights)
            need(bool(torch.isfinite(loss)), "fixed-feature loss is non-finite")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    model.eval()
    with torch.inference_mode():
        output = model(_features(rows, predict, axis).to(device))
        pmf = output["pmf"]
        _pmf_valid(pmf)
        # For the two imbalance candidates this is a preregistered decoding
        # heuristic under a hybrid ordinal + raw-score-MSE objective. It is
        # evaluated empirically by inner OOF, not claimed as exact calibration.
        corrected = (natural_prior_correction(pmf, fit_prior.to(device), spec_prior.to(device))
                     if spec.prior_treatment in {"effective_number", "sqrt_sampler"} else pmf)
        prediction = (corrected * torch.arange(1, 6, device=device).float()).sum(1)
    return prediction.float().cpu().numpy()


def _candidate_inner_oof(rows: Sequence[ResidualRow], outer_fold: int, spec: CandidateSpec,
                         config: FixedFeatureConfig, *, smoke: bool) -> tuple[np.ndarray, np.ndarray]:
    _, folds = nested_indices(rows, outer_fold)
    prediction = np.full((len(rows), 3), np.nan)
    truth = np.full((len(rows), 3), np.nan)
    for inner_position, (dev_fold, (fit, dev)) in enumerate(folds.items()):
        for axis in range(3):
            prediction[list(dev), axis] = _train_predict(
                rows, fit, dev, axis, spec, config,
                seed=config.seed + outer_fold * 10000 + inner_position * 1000 + axis * 100,
                smoke=smoke,
            )
            truth[list(dev), axis] = [rows[index].raw_labels[axis] for index in dev]
    keep = np.isfinite(prediction).all(1)
    need(int(keep.sum()) == 1600 and np.isfinite(truth[keep]).all(), "inner OOF coverage differs")
    return truth[keep], prediction[keep]


def _write_json_fresh(path: Path, value: Mapping[str, Any], *, private: bool = False) -> str:
    need(not path.exists(), f"refusing to overwrite {path}")
    if not private:
        _validate_public_payload(value)
    if private:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return file_sha256(path)


def _validate_public_payload(value: Any) -> None:
    forbidden = {"source_id", "group_id", "essay", "prompt", "shared_embedding",
                 "raw_gold", "candidate_prediction", "exact_r0_prediction"}
    if isinstance(value, Mapping):
        need(not (set(value) & forbidden), "restricted row content cannot enter public output")
        for nested in value.values():
            _validate_public_payload(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_public_payload(nested)


def _write_jsonl_fresh(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    need(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    return file_sha256(path)


def load_rows(config: FixedFeatureConfig) -> tuple[ResidualRow, ...]:
    need(file_sha256(config.train_path) == config.train_sha256, "train input binding differs")
    need(file_sha256(config.embedding_rows_path) == config.embedding_rows_sha256, "embedding rows binding differs")
    need(file_sha256(config.r0_oof_prediction_path) == config.r0_oof_prediction_sha256,
         "exact R0 prediction binding differs")
    manifest, rows = load_embedding_artifact(config.embedding_manifest_path, config.embedding_rows_path)
    need(manifest.split_role == "train" and manifest.base_prediction_origin == "oof"
         and not manifest.contains_average_target and len(rows) == 2000, "fixed-feature input contract differs")
    exact_r0: dict[str, Mapping[str, Any]] = {}
    with Path(config.r0_oof_prediction_path).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            source_id = item.get("source_id")
            need(isinstance(source_id, str) and source_id not in exact_r0, "exact R0 row identity differs")
            exact_r0[source_id] = item
    need(len(exact_r0) == len(rows), "exact R0 population differs")
    for row in rows:
        item = exact_r0.get(row.source_id)
        need(item is not None and item.get("fold") == row.oof_fold, "exact R0 fold linkage differs")
        prediction = item.get("continuous_prediction")
        need(isinstance(prediction, Mapping) and set(prediction) == set(AXES), "exact R0 axis schema differs")
        need(all(abs(float(prediction[axis]) - row.base_predictions[index]) <= 1e-9
                 for index, axis in enumerate(AXES)), "embedded R0 prediction differs from exact artifact")
    return rows


def run_outer_fold(config: FixedFeatureConfig, outer_fold: int, *, smoke: bool = False) -> dict[str, Any]:
    rows = load_rows(config)
    outer, _ = nested_indices(rows, outer_fold)
    screens = []
    for spec in config.candidates:
        truth, prediction = _candidate_inner_oof(rows, outer_fold, spec, config, smoke=smoke)
        metrics = compute_iterative_tail_metrics(truth, prediction)
        screens.append({"candidate_id": spec.identifier, "family": spec.family, "metrics": metrics})
    selected = min(screens, key=lambda item: (float(item["metrics"]["macro"]["rmse"]), item["candidate_id"]))
    selected_spec = next(item for item in config.candidates if item.identifier == selected["candidate_id"])
    fit = tuple(index for index in range(len(rows)) if index not in set(outer))
    predictions = np.column_stack([
        _train_predict(rows, fit, outer, axis, selected_spec, config,
                       seed=config.seed + outer_fold * 10000 + axis * 100, smoke=smoke)
        for axis in range(3)
    ])
    truth = np.asarray([rows[index].raw_labels for index in outer])
    base = np.asarray([rows[index].base_predictions for index in outer])
    restricted_rows = ({
        "source_id": rows[index].source_id, "outer_fold": outer_fold,
        "candidate_prediction": {axis: float(predictions[position, axis_index]) for axis_index, axis in enumerate(AXES)},
        "exact_r0_prediction": {axis: float(base[position, axis_index]) for axis_index, axis in enumerate(AXES)},
        "raw_gold": {axis: float(truth[position, axis_index]) for axis_index, axis in enumerate(AXES)},
    } for position, index in enumerate(outer))
    suffix = "smoke" if smoke else "full"
    restricted_path = Path(config.restricted_output_root) / "fixed_feature" / suffix / f"outer-{outer_fold:02d}.jsonl"
    rows_sha = _write_jsonl_fresh(restricted_path, restricted_rows)
    result = {
        "schema_version": "mal2026-ordinal-tail-fixed-feature-outer-v1", "status": "completed",
        "mode": "smoke" if smoke else "outer_fold", "run_id": config.run_id, "outer_fold": outer_fold,
        "records": len(outer), "selected_candidate": selected["candidate_id"], "candidate_count": len(screens),
        "candidate_inventory": list(EXPECTED_IDS), "config_sha256": config.config_sha256,
        "inner_screen": screens, "outer_metrics": compute_iterative_tail_metrics(truth, predictions),
        "exact_r0_outer_metrics": compute_iterative_tail_metrics(truth, base),
        "restricted_rows_path": str(restricted_path.resolve()), "restricted_rows_sha256": rows_sha,
        "embedding_rows_sha256": config.embedding_rows_sha256, "r0_oof_prediction_sha256": config.r0_oof_prediction_sha256,
        "validation_rows_loaded": False, "average_target_used": False,
        "privacy": "aggregate_only_public_row_predictions_restricted",
    }
    public_path = Path(config.output_root) / "fixed_feature" / suffix / f"outer-{outer_fold:02d}.json"
    _write_json_fresh(public_path, result)
    return result


def aggregate_full(config: FixedFeatureConfig) -> dict[str, Any]:
    rows = load_rows(config)
    by_id: dict[str, Mapping[str, Any]] = {}
    fold_bindings = []
    screen_by_candidate: dict[str, list[Mapping[str, Any]]] = {identifier: [] for identifier in EXPECTED_IDS}
    for fold in range(5):
        public_path = Path(config.output_root) / "fixed_feature/full" / f"outer-{fold:02d}.json"
        restricted_path = Path(config.restricted_output_root) / "fixed_feature/full" / f"outer-{fold:02d}.jsonl"
        public = json.loads(public_path.read_text(encoding="utf-8"))
        need(
            public.get("schema_version") == "mal2026-ordinal-tail-fixed-feature-outer-v1"
            and public.get("status") == "completed"
            and public.get("mode") == "outer_fold"
            and public.get("run_id") == config.run_id
            and public.get("outer_fold") == fold
            and public.get("records") == 400
            and public.get("candidate_count") == len(EXPECTED_IDS)
            and public.get("candidate_inventory") == list(EXPECTED_IDS)
            and public.get("config_sha256") == config.config_sha256
            and public.get("embedding_rows_sha256") == config.embedding_rows_sha256
            and public.get("r0_oof_prediction_sha256") == config.r0_oof_prediction_sha256
            and public.get("validation_rows_loaded") is False
            and public.get("average_target_used") is False
            and public.get("restricted_rows_sha256") == file_sha256(restricted_path),
            "outer result binding differs",
        )
        screen = public.get("inner_screen")
        need(isinstance(screen, list) and [item.get("candidate_id") for item in screen] == list(EXPECTED_IDS),
             "outer screen inventory differs")
        for item in screen:
            screen_by_candidate[item["candidate_id"]].append(item["metrics"])
        with restricted_path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line); source_id = item["source_id"]
                need(source_id not in by_id, "outer prediction duplicated")
                by_id[source_id] = item
        fold_bindings.append({"outer_fold": fold, "public_sha256": file_sha256(public_path),
                              "restricted_rows_sha256": file_sha256(restricted_path),
                              "selected_candidate": public["selected_candidate"]})
    need(len(by_id) == len(rows) == 2000, "full OOF coverage differs")
    ordered = [by_id[row.source_id] for row in rows]
    truth = [[item["raw_gold"][axis] for axis in AXES] for item in ordered]
    prediction = [[item["candidate_prediction"][axis] for axis in AXES] for item in ordered]
    base = [[item["exact_r0_prediction"][axis] for axis in AXES] for item in ordered]
    candidate_metrics = compute_iterative_tail_metrics(truth, prediction)
    exact_r0_metrics = compute_iterative_tail_metrics(truth, base)
    improvements = metric_improvements(exact_r0_metrics, candidate_metrics)
    gate = config.promotion_gate
    point_gates = {
        "macro_rmse_improvement": improvements["rmse"] >= float(gate["minimum_macro_rmse_improvement"]),
        "axis_rmse_noninferiority": all(value >= -float(gate["maximum_axis_rmse_worsening"])
                                         for value in improvements["axis_rmse"].values()),
        "low_tail_nonworsening": (not bool(gate["low_tail_must_not_worsen"])) or improvements["low_tail_rmse"] >= 0.0,
        "high_tail_nonworsening": (not bool(gate["high_tail_must_not_worsen"])) or improvements["high_tail_rmse"] >= 0.0,
        "gold_3_4_balanced_accuracy_noninferiority": improvements["gold_3_4_balanced_accuracy"] >= -float(gate["maximum_gold_3_4_balanced_accuracy_drop"]),
        "spearman_noninferiority": improvements["spearman"] >= -float(gate["maximum_spearman_drop"]),
    }
    bootstrap = paired_bootstrap_delta_ci(
        truth, base, prediction, document_ids=[row.group_id for row in rows],
        n_resamples=int(gate["paired_bootstrap_resamples"]), seed=config.seed,
    )
    bootstrap_gate = ((not bool(gate["paired_bootstrap_rmse_lower_bound_above_zero"]))
                      or bootstrap["intervals"]["rmse"]["lower"] > 0.0)
    promote = all(point_gates.values()) and bootstrap_gate
    screen_summary = []
    for spec in config.candidates:
        metrics = screen_by_candidate[spec.identifier]
        need(len(metrics) == 5, "candidate screen coverage differs")
        macro_keys = ("rmse", "low_tail_rmse", "high_tail_rmse", "gold_3_4_balanced_accuracy", "spearman")
        screen_summary.append({
            "candidate_id": spec.identifier,
            "family": spec.family,
            "mean_outer_inner_oof_macro": {
                key: float(np.mean([float(item["macro"][key]) for item in metrics])) for key in macro_keys
            },
        })
    ranked = sorted(screen_summary, key=lambda item: (
        item["mean_outer_inner_oof_macro"]["rmse"], item["candidate_id"],
    ))
    recommended = []
    seen_families: set[str] = set()
    for item in ranked:
        if item["family"] not in seen_families:
            recommended.append(item)
            seen_families.add(item["family"])
        if len(recommended) == 2:
            break
    need(len(recommended) == 2, "two distinct phase-2 families are required")
    selected_manifest = {
        "schema_version": "mal2026-ordinal-tail-protected-output-v1",
        "run_id": config.run_id,
        "selected_model": "candidate" if promote else "exact_r0",
        "config_sha256": config.config_sha256,
        "exact_r0": {
            "path": str(Path(config.r0_oof_prediction_path).resolve()),
            "sha256": config.r0_oof_prediction_sha256,
        },
        "candidate_outer_rows": [
            {"outer_fold": item["outer_fold"], "sha256": item["restricted_rows_sha256"]}
            for item in fold_bindings
        ] if promote else [],
        "validation_selection": False,
        "average_target_used": False,
    }
    selected_manifest_path = Path(config.restricted_output_root) / "fixed_feature/selected_output_manifest.json"
    selected_manifest_sha = _write_json_fresh(selected_manifest_path, selected_manifest, private=True)
    result = {
        "schema_version": "mal2026-ordinal-tail-fixed-feature-aggregate-v1", "status": "completed",
        "mode": "full", "run_id": config.run_id, "records": 2000, "folds": 5,
        "candidate_metrics": candidate_metrics, "exact_r0_metrics": exact_r0_metrics,
        "exploratory_candidate_improvements": improvements, "point_gates": point_gates,
        "paired_bootstrap": bootstrap, "bootstrap_rmse_lower_above_zero": bootstrap_gate,
        "protected_output": "candidate" if promote else "exact_r0", "promote": promote,
        "protected_output_manifest_path": str(selected_manifest_path.resolve()),
        "protected_output_manifest_sha256": selected_manifest_sha,
        "candidate_screen_summary": ranked,
        "phase2_recommended_distinct_families": recommended,
        "objective_contract": config.hybrid_objective_disclosure,
        "fold_bindings": fold_bindings,
        "candidate_inventory": list(EXPECTED_IDS), "config_sha256": config.config_sha256,
        "embedding_rows_sha256": config.embedding_rows_sha256, "r0_oof_prediction_sha256": config.r0_oof_prediction_sha256,
        "validation_rows_loaded": False, "average_target_used": False,
        "privacy": "aggregate_only_no_ids_embeddings_or_row_predictions",
    }
    _write_json_fresh(Path(config.output_root) / "fixed_feature/aggregate.json", result)
    return result


def run(config: FixedFeatureConfig | str | Path, *, mode: str, outer_fold: int | None = None) -> dict[str, Any]:
    """Public entry point used by the stage orchestrator."""
    value = FixedFeatureConfig.from_json(config) if isinstance(config, (str, Path)) else config
    need(mode in MODES, "fixed-feature mode differs")
    if mode == "full":
        need(outer_fold is None, "full aggregation takes no outer fold")
        return aggregate_full(value)
    need(isinstance(outer_fold, int) and 0 <= outer_fold < 5, "outer fold must be 0..4")
    return run_outer_fold(value, outer_fold, smoke=mode == "smoke")


__all__ = [
    "CandidateSpec", "EXPECTED_IDS", "FixedFeatureConfig", "OrdinalTailFixedFeatureError",
    "aggregate_full", "build_axis_model", "candidate_specs", "coral_pmf", "corn_pmf",
    "corn_targets", "effective_number_weights", "natural_prior", "natural_prior_correction",
    "nested_indices", "rps_loss", "run", "run_outer_fold", "sampling_prior", "slace_components", "slace_loss",
]
