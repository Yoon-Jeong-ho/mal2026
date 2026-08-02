"""Leakage-safe ordinal residual core for the three R0 scoring axes.

The training target is *never* the essay-level ``average``.  Each axis target
is the raw continuous human gold rounded half-up to an integer in ``1..5``.
Training-time R0 predictions must be out-of-fold (OOF); validation is a held-
out, evaluation-only split and cannot be used for fitting or model selection.

The model uses an explicit five-way head.  A Gaussian-shaped prior derived
from the R0 continuous score is added to learned residual logits, so every
forward pass yields a full probability distribution over all five classes.
The returned ``loss`` and ``logits`` follow the Hugging Face ``Trainer`` model
contract; no custom optimizer or training loop is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

AXES = ("content", "organization", "expression")
CLASSES = (1, 2, 3, 4, 5)
GOLD_LABEL_POLICY = "round_half_up_raw_axis_score"
EMBEDDING_SCHEMA_VERSION = "r0_ordinal_residual_embedding_v1"


class R0OrdinalResidualContractError(ValueError):
    """Raised before data leakage or an ambiguous target can enter training."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise R0OrdinalResidualContractError(message)


def rounded_axis_label(raw_gold: Real) -> int:
    """Round one finite raw human axis score half-up to a class in ``1..5``."""
    _need(isinstance(raw_gold, Real) and not isinstance(raw_gold, bool), "axis gold must be numeric")
    value = float(raw_gold)
    _need(math.isfinite(value) and 1.0 <= value <= 5.0, "axis gold must be finite and within [1, 5]")
    return min(5, max(1, math.floor(value + 0.5)))


def axis_class_labels(raw_gold: Mapping[str, Real]) -> tuple[int, int, int]:
    """Create the three class labels while rejecting the forbidden average."""
    _need(isinstance(raw_gold, Mapping), "raw gold must be an axis mapping")
    _need("average" not in raw_gold, "average target is forbidden for ordinal residual training")
    _need(set(raw_gold) == set(AXES), "raw gold must contain exactly the three canonical axes")
    return tuple(rounded_axis_label(raw_gold[axis]) for axis in AXES)


@dataclass(frozen=True)
class BasePredictionContract:
    """Provenance gate applied before a split may be loaded.

    ``train`` requires OOF predictions.  ``validation`` requires predictions
    from a base fit that excluded validation and is permanently evaluation-
    only.  These constraints are intentionally metadata-level gates so a
    caller cannot silently substitute in-sample base predictions.
    """

    split_role: str
    base_prediction_origin: str
    base_model_fit_excludes_split: bool
    evaluation_only: bool
    gold_label_policy: str
    contains_average_target: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BasePredictionContract":
        _need(isinstance(raw, Mapping), "base prediction contract must be an object")
        _need(set(raw) == set(cls.__dataclass_fields__), "base prediction contract has unknown or missing fields")
        _need(all(isinstance(raw[name], bool) for name in ("base_model_fit_excludes_split", "evaluation_only", "contains_average_target")), "contract flags must be booleans")
        contract = cls(**raw)
        contract.validate()
        return contract

    @classmethod
    def from_json(cls, path: str | Path) -> "BasePredictionContract":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        _need(self.split_role in {"train", "validation"}, "split_role must be train or validation")
        _need(self.gold_label_policy == GOLD_LABEL_POLICY, "gold labels must use raw-axis half-up integer classes")
        _need(not self.contains_average_target, "average target is forbidden")
        _need(self.base_model_fit_excludes_split, "base model fit must exclude every row receiving its prediction")
        if self.split_role == "train":
            _need(self.base_prediction_origin == "oof", "training base predictions must be OOF")
            _need(not self.evaluation_only, "training split cannot be evaluation-only")
        else:
            _need(self.base_prediction_origin == "held_out", "validation predictions must be held-out")
            _need(self.evaluation_only, "validation is evaluation-only")


def validate_split_contracts(train: BasePredictionContract, validation: BasePredictionContract) -> None:
    """Validate both lifecycle contracts together before constructing data."""
    train.validate()
    validation.validate()
    _need(train.split_role == "train" and validation.split_role == "validation", "train and validation contracts were swapped")


def validate_base_predictions(values: Mapping[str, Real]) -> tuple[float, float, float]:
    """Return finite three-axis continuous base predictions in canonical order."""
    _need(isinstance(values, Mapping) and set(values) == set(AXES), "base predictions must contain exactly three axes")
    result = tuple(float(values[axis]) for axis in AXES)
    _need(all(math.isfinite(value) and 1.0 <= value <= 5.0 for value in result), "base predictions must be finite within [1, 5]")
    return result


def build_ordinal_residual_model(
    embedding_dim: int,
    *,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    prior_temperature: float = 0.75,
    raw_target_loss_weight: float = 0.25,
) -> Any:
    """Build a CPU/GPU-agnostic ``Trainer``-compatible three-axis model.

    Inputs are ``shared_embedding`` with shape ``[batch, embedding_dim]`` and
    ``base_predictions`` with shape ``[batch, 3]``.  Optional ``labels`` must
    be integer classes ``[batch, 3]`` in ``1..5``.  Axis heads are separate,
    while the embedding/base fusion is shared.
    """
    _need(isinstance(embedding_dim, int) and embedding_dim > 0, "embedding_dim must be positive")
    _need(isinstance(hidden_dim, int) and hidden_dim > 0, "hidden_dim must be positive")
    _need(isinstance(dropout, float) and 0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
    _need(isinstance(prior_temperature, Real) and float(prior_temperature) > 0, "prior_temperature must be positive")
    _need(isinstance(raw_target_loss_weight, Real) and float(raw_target_loss_weight) >= 0, "raw_target_loss_weight must be nonnegative")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("ordinal residual model requires torch") from exc

    class R0OrdinalResidualModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared_fusion = nn.Sequential(
                nn.Linear(embedding_dim + len(AXES), hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.axis_heads = nn.ModuleList(nn.Linear(hidden_dim, len(CLASSES)) for _ in AXES)
            self.prior_temperature = float(prior_temperature)
            self.raw_target_loss_weight = float(raw_target_loss_weight)
            self.register_buffer("class_values", torch.tensor(CLASSES, dtype=torch.float32), persistent=False)

        def forward(
            self,
            shared_embedding: Any,
            base_predictions: Any,
            labels: Any | None = None,
            raw_labels: Any | None = None,
            **_: Any,
        ) -> Mapping[str, Any]:
            _need(getattr(shared_embedding, "ndim", None) == 2 and shared_embedding.shape[1] == embedding_dim, "shared_embedding has invalid shape")
            _need(getattr(base_predictions, "ndim", None) == 2 and base_predictions.shape[1] == len(AXES), "base_predictions must have shape [batch, 3]")
            _need(shared_embedding.shape[0] == base_predictions.shape[0], "embedding/base batch mismatch")
            base = base_predictions.float()
            _need(bool(torch.isfinite(base).all().item()) and bool(((base >= 1) & (base <= 5)).all().item()), "base predictions must be finite within [1, 5]")
            fused = self.shared_fusion(torch.cat((shared_embedding.float(), (base - 3.0) / 2.0), dim=-1))
            residual_logits = torch.stack([head(fused) for head in self.axis_heads], dim=1)
            prior_logits = -0.5 * ((self.class_values.view(1, 1, -1) - base.unsqueeze(-1)) / self.prior_temperature) ** 2
            logits = residual_logits + prior_logits
            probabilities = torch.softmax(logits, dim=-1)
            expected_scores = (probabilities * self.class_values.view(1, 1, -1)).sum(dim=-1)
            result: dict[str, Any] = {
                "logits": logits,
                "probabilities": probabilities,
                "expected_scores": expected_scores,
            }
            if labels is not None:
                _need(tuple(labels.shape) == tuple(base.shape), "labels must have shape [batch, 3]")
                _need(labels.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8), "labels must be integer classes, not average targets")
                _need(bool(((labels >= 1) & (labels <= 5)).all().item()), "labels must be in 1..5")
                loss = F.cross_entropy(logits.reshape(-1, len(CLASSES)), (labels.long() - 1).reshape(-1))
                if raw_labels is not None:
                    _need(tuple(raw_labels.shape) == tuple(base.shape), "raw_labels must have shape [batch, 3]")
                    raw = raw_labels.float()
                    _need(bool(torch.isfinite(raw).all().item()) and bool(((raw >= 1) & (raw <= 5)).all().item()), "raw_labels must be finite within [1, 5]")
                    loss = loss + self.raw_target_loss_weight * F.mse_loss(expected_scores, raw)
                result["loss"] = loss
            return result

    return R0OrdinalResidualModel()


def blend_axis_posteriors(
    base_predictions: Sequence[Real],
    probabilities: Sequence[Sequence[Real]],
    *,
    strategy: str = "expected_risk",
    fixed_posterior_weight: float = 0.5,
) -> tuple[dict[str, Any], ...]:
    """Decode all five probabilities and blend base/posterior expectations.

    ``expected_risk`` uses inverse expected-squared-risk weighting;
    ``confidence`` uses the posterior maximum probability; and ``fixed`` uses
    ``fixed_posterior_weight``.  No hard one-way cascade is performed.
    """
    _need(len(base_predictions) == len(AXES) and len(probabilities) == len(AXES), "decode requires three axes")
    _need(strategy in {"expected_risk", "confidence", "fixed"}, "unsupported blend strategy")
    _need(0.0 <= fixed_posterior_weight <= 1.0, "fixed posterior weight must be in [0, 1]")
    decoded: list[dict[str, Any]] = []
    for axis, raw_base, raw_probabilities in zip(AXES, base_predictions, probabilities, strict=True):
        base = float(raw_base)
        probs = tuple(float(value) for value in raw_probabilities)
        _need(math.isfinite(base) and 1.0 <= base <= 5.0, "base prediction must be finite within [1, 5]")
        _need(len(probs) == len(CLASSES) and all(math.isfinite(value) and value >= 0 for value in probs), "each axis requires five nonnegative finite probabilities")
        total = sum(probs)
        _need(abs(total - 1.0) <= 1e-6, "class probabilities must sum to one")
        expected = sum(label * probability for label, probability in zip(CLASSES, probs, strict=True))
        posterior_risk = sum(probability * (label - expected) ** 2 for label, probability in zip(CLASSES, probs, strict=True))
        base_risk = sum(probability * (label - base) ** 2 for label, probability in zip(CLASSES, probs, strict=True))
        confidence = max(probs)
        if strategy == "expected_risk":
            denominator = base_risk + posterior_risk
            weight = base_risk / denominator if denominator > 0 else 0.5
        elif strategy == "confidence":
            weight = confidence
        else:
            weight = fixed_posterior_weight
        blended = (1.0 - weight) * base + weight * expected
        decoded.append({
            "axis": axis,
            "probabilities_1_to_5": probs,
            "base_continuous_score": base,
            "posterior_expected_score": expected,
            "posterior_confidence": confidence,
            "posterior_expected_squared_risk": posterior_risk,
            "base_expected_squared_risk": base_risk,
            "posterior_weight": weight,
            "blended_continuous_score": blended,
            "blended_integer_class": rounded_axis_label(blended),
        })
    return tuple(decoded)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _immutable_revision(value: object) -> bool:
    return isinstance(value, str) and len(value) in {40, 64} and all(char in "0123456789abcdef" for char in value.lower())


@dataclass(frozen=True)
class EmbeddingArtifactManifest:
    """Exact schema for frozen semantic embeddings and associated R0 scores."""

    schema_version: str
    split_role: str
    base_prediction_origin: str
    base_model_fit_excludes_split: bool
    evaluation_only: bool
    gold_label_policy: str
    contains_average_target: bool
    embedding_model_id: str
    embedding_model_revision: str
    embedding_source: str
    embedding_frozen: bool
    embedding_dim: int
    fold_count: int
    rows_sha256: str

    @classmethod
    def from_json(cls, path: str | Path) -> "EmbeddingArtifactManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        _need(isinstance(raw, Mapping) and set(raw) == set(cls.__dataclass_fields__), "embedding manifest has unknown or missing fields")
        manifest = cls(**raw)
        manifest.validate()
        return manifest

    def base_contract(self) -> BasePredictionContract:
        return BasePredictionContract(
            split_role=self.split_role,
            base_prediction_origin=self.base_prediction_origin,
            base_model_fit_excludes_split=self.base_model_fit_excludes_split,
            evaluation_only=self.evaluation_only,
            gold_label_policy=self.gold_label_policy,
            contains_average_target=self.contains_average_target,
        )

    def validate(self) -> None:
        self.base_contract().validate()
        _need(self.schema_version == EMBEDDING_SCHEMA_VERSION, "unsupported embedding artifact schema")
        _need(isinstance(self.embedding_model_id, str) and bool(self.embedding_model_id.strip()), "embedding model ID is required")
        _need(_immutable_revision(self.embedding_model_revision), "embedding model revision must be immutable")
        _need(self.embedding_source in {"public", "aihub_warm"}, "embedding source must be public or aihub_warm")
        _need(self.embedding_frozen, "semantic embeddings must be frozen")
        _need(isinstance(self.embedding_dim, int) and self.embedding_dim > 0, "embedding_dim must be positive")
        _need(self.fold_count == (5 if self.split_role == "train" else 0), "train requires exact 5-fold OOF; validation has no folds")
        _need(isinstance(self.rows_sha256, str) and len(self.rows_sha256) == 64 and all(c in "0123456789abcdef" for c in self.rows_sha256.lower()), "rows_sha256 is invalid")


@dataclass(frozen=True)
class ResidualRow:
    """One restricted row; identifiers are never copied into public output."""

    source_id: str
    group_id: str
    shared_embedding: tuple[float, ...]
    base_predictions: tuple[float, float, float]
    raw_labels: tuple[float, float, float]
    labels: tuple[int, int, int]
    oof_fold: int | None


def load_embedding_artifact(manifest_path: str | Path, rows_path: str | Path) -> tuple[EmbeddingArtifactManifest, tuple[ResidualRow, ...]]:
    """Load and fully validate a restricted frozen-embedding JSONL artifact.

    JSONL row schema is exactly ``source_id``, ``group_id``,
    ``shared_embedding``, ``base_continuous_prediction``,
    ``raw_continuous_gold``, and ``oof_fold``.  Train rows use folds ``0..4``;
    held-out validation rows use JSON ``null``.
    """
    manifest = EmbeddingArtifactManifest.from_json(manifest_path)
    path = Path(rows_path)
    _need(path.is_file() and _file_sha256(path) == manifest.rows_sha256, "embedding rows checksum mismatch")
    result: list[ResidualRow] = []
    seen: set[str] = set()
    group_folds: dict[str, int] = {}
    required = {"source_id", "group_id", "shared_embedding", "base_continuous_prediction", "raw_continuous_gold", "oof_fold"}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            raw = json.loads(line)
            _need(isinstance(raw, Mapping) and set(raw) == required, f"invalid embedding row schema at line {line_number}")
            source_id, group_id = raw["source_id"], raw["group_id"]
            _need(isinstance(source_id, str) and source_id and source_id not in seen, f"duplicate or invalid source_id at line {line_number}")
            _need(isinstance(group_id, str) and bool(group_id), f"invalid group_id at line {line_number}")
            seen.add(source_id)
            embedding_raw = raw["shared_embedding"]
            _need(isinstance(embedding_raw, list) and len(embedding_raw) == manifest.embedding_dim, f"embedding dimension mismatch at line {line_number}")
            embedding = tuple(float(value) for value in embedding_raw)
            _need(all(math.isfinite(value) for value in embedding), f"non-finite embedding at line {line_number}")
            base = validate_base_predictions(raw["base_continuous_prediction"])
            gold = raw["raw_continuous_gold"]
            labels = axis_class_labels(gold)
            raw_labels = tuple(float(gold[axis]) for axis in AXES)
            fold = raw["oof_fold"]
            if manifest.split_role == "train":
                _need(isinstance(fold, int) and not isinstance(fold, bool) and 0 <= fold < 5, f"train oof_fold must be 0..4 at line {line_number}")
                prior = group_folds.setdefault(group_id, fold)
                _need(prior == fold, "a group cannot span OOF folds")
            else:
                _need(fold is None, f"validation oof_fold must be null at line {line_number}")
            result.append(ResidualRow(source_id, group_id, embedding, base, raw_labels, labels, fold))
    _need(bool(result), "embedding artifact is empty")
    if manifest.split_role == "train":
        _need({row.oof_fold for row in result} == set(range(5)), "train artifact must contain all five OOF folds")
    return manifest, tuple(result)


def group_selection_split(rows: Sequence[ResidualRow], *, seed: int, dev_ratio: float) -> tuple[tuple[ResidualRow, ...], tuple[ResidualRow, ...]]:
    """Create a deterministic group-disjoint train/dev split from train only."""
    _need(isinstance(seed, int) and not isinstance(seed, bool), "seed must be an integer")
    _need(0.0 < dev_ratio < 0.5, "dev_ratio must be in (0, 0.5)")
    groups = sorted({row.group_id for row in rows}, key=lambda group: sha256(f"{seed}:{group}".encode()).hexdigest())
    _need(len(groups) >= 2, "group split requires at least two groups")
    dev_count = min(len(groups) - 1, max(1, math.ceil(len(groups) * dev_ratio)))
    dev_groups = set(groups[:dev_count])
    train = tuple(row for row in rows if row.group_id not in dev_groups)
    dev = tuple(row for row in rows if row.group_id in dev_groups)
    _need(bool(train) and bool(dev) and not ({row.group_id for row in train} & {row.group_id for row in dev}), "group split failed")
    return train, dev


def _qwk(labels: Sequence[int], predictions: Sequence[int]) -> float:
    _need(len(labels) == len(predictions) and bool(labels), "QWK requires paired labels")
    observed = [[0] * 5 for _ in range(5)]
    left, right = [0] * 5, [0] * 5
    for label, prediction in zip(labels, predictions, strict=True):
        observed[label - 1][prediction - 1] += 1
        left[label - 1] += 1
        right[prediction - 1] += 1
    numerator = sum(((i - j) ** 2 / 16) * observed[i][j] for i in range(5) for j in range(5))
    denominator = sum(((i - j) ** 2 / 16) * left[i] * right[j] / len(labels) for i in range(5) for j in range(5))
    return 1.0 - numerator / denominator if denominator else 0.0


def aggregate_predictions(
    base: Sequence[Sequence[Real]],
    raw_gold: Sequence[Sequence[Real]],
    probabilities: Sequence[Sequence[Sequence[Real]]],
    *,
    blend_weight: float,
    thresholds: Sequence[Real],
) -> dict[str, Any]:
    """Compute overall/axis residual metrics and the unchanged R0 baseline.

    NLL is mean ``-log p(gold_class)``; multiclass Brier is the mean sum of
    five squared probability errors. Confidence ECE uses ten fixed bins over
    max posterior probability and posterior-argmax correctness. Both posterior
    argmax and final blended/thresholded class-3/4 conditionals are retained.
    """
    _need(len(base) == len(raw_gold) == len(probabilities) and bool(base), "prediction arrays must align")
    _need(0 <= blend_weight <= 1 and len(thresholds) == 4, "invalid decoding parameters")
    cuts = tuple(float(value) for value in thresholds)
    _need(all(a < b for a, b in zip(cuts, cuts[1:])), "thresholds must be strictly increasing")
    buckets: dict[str, dict[str, Any]] = {
        name: {
            "count": 0, "residual_squared": 0.0, "residual_absolute": 0.0,
            "base_squared": 0.0, "base_absolute": 0.0,
            "truth_classes": [], "residual_classes": [], "base_classes": [],
            "posterior_classes": [], "probabilities": [],
        }
        for name in ("overall", *AXES)
    }
    for base_row, gold_row, probability_row in zip(base, raw_gold, probabilities, strict=True):
        decoded = blend_axis_posteriors(base_row, probability_row, strategy="fixed", fixed_posterior_weight=blend_weight)
        for axis_index, item in enumerate(decoded):
            axis = AXES[axis_index]
            gold = float(gold_row[axis_index])
            prediction = float(item["blended_continuous_score"])
            base_prediction = float(base_row[axis_index])
            truth_class = rounded_axis_label(gold)
            predicted_class = 1 + sum(prediction >= cut for cut in cuts)
            base_class = rounded_axis_label(base_prediction)
            for bucket_name in ("overall", axis):
                bucket = buckets[bucket_name]
                bucket["count"] += 1
                bucket["residual_squared"] += (prediction - gold) ** 2
                bucket["residual_absolute"] += abs(prediction - gold)
                bucket["base_squared"] += (base_prediction - gold) ** 2
                bucket["base_absolute"] += abs(base_prediction - gold)
                bucket["truth_classes"].append(truth_class)
                bucket["residual_classes"].append(predicted_class)
                bucket["base_classes"].append(base_class)
                axis_probabilities = tuple(float(value) for value in probability_row[axis_index])
                bucket["probabilities"].append(axis_probabilities)
                bucket["posterior_classes"].append(1 + max(range(5), key=lambda index: axis_probabilities[index]))

    result: dict[str, Any] = {"by_axis": {}}
    for name, bucket in buckets.items():
        count = bucket["count"]
        residual_rmse = math.sqrt(bucket["residual_squared"] / count)
        base_rmse = math.sqrt(bucket["base_squared"] / count)
        nll = 0.0
        brier = 0.0
        confidence_bins = [{"count": 0, "confidence_sum": 0.0, "correct_sum": 0} for _ in range(10)]
        for truth_class, posterior_class, probability_row in zip(
            bucket["truth_classes"], bucket["posterior_classes"], bucket["probabilities"], strict=True
        ):
            nll -= math.log(max(probability_row[truth_class - 1], 1e-12))
            brier += sum(
                (probability - (1.0 if class_index == truth_class else 0.0)) ** 2
                for class_index, probability in enumerate(probability_row, start=1)
            )
            confidence = max(probability_row)
            bin_index = min(9, int(confidence * 10))
            confidence_bins[bin_index]["count"] += 1
            confidence_bins[bin_index]["confidence_sum"] += confidence
            confidence_bins[bin_index]["correct_sum"] += posterior_class == truth_class
        ece = sum(
            item["count"] / count * abs(item["confidence_sum"] / item["count"] - item["correct_sum"] / item["count"])
            for item in confidence_bins if item["count"]
        )
        metrics = {
            "count": count,
            "residual_raw_rmse": residual_rmse,
            "residual_raw_mae": bucket["residual_absolute"] / count,
            "base_raw_rmse": base_rmse,
            "base_raw_mae": bucket["base_absolute"] / count,
            "residual_minus_base_rmse": residual_rmse - base_rmse,
            "residual_class_accuracy": sum(a == b for a, b in zip(bucket["truth_classes"], bucket["residual_classes"], strict=True)) / count,
            "base_class_accuracy": sum(a == b for a, b in zip(bucket["truth_classes"], bucket["base_classes"], strict=True)) / count,
            "residual_qwk": _qwk(bucket["truth_classes"], bucket["residual_classes"]),
            "base_qwk": _qwk(bucket["truth_classes"], bucket["base_classes"]),
            "posterior_nll": nll / count,
            "posterior_multiclass_brier": brier / count,
            "posterior_confidence_ece_10bin": ece,
        }
        for target_class in (3, 4):
            predicted_count = sum(value == target_class for value in bucket["residual_classes"])
            correct_count = sum(
                prediction == target_class and truth == target_class
                for truth, prediction in zip(bucket["truth_classes"], bucket["residual_classes"], strict=True)
            )
            metrics[f"predicted_{target_class}_count"] = predicted_count
            metrics[f"predicted_{target_class}_coverage"] = predicted_count / count
            metrics[f"predicted_{target_class}_conditional_accuracy"] = correct_count / predicted_count if predicted_count else 0.0
            posterior_count = sum(value == target_class for value in bucket["posterior_classes"])
            posterior_correct = sum(
                prediction == target_class and truth == target_class
                for truth, prediction in zip(bucket["truth_classes"], bucket["posterior_classes"], strict=True)
            )
            metrics[f"posterior_predicted_{target_class}_count"] = posterior_count
            metrics[f"posterior_predicted_{target_class}_coverage"] = posterior_count / count
            metrics[f"posterior_predicted_{target_class}_conditional_accuracy"] = posterior_correct / posterior_count if posterior_count else 0.0
        metrics["confidence_bins"] = {
            f"bin_{index:02d}": {
                "count": item["count"],
                "mean_confidence": item["confidence_sum"] / item["count"] if item["count"] else 0.0,
                "accuracy": item["correct_sum"] / item["count"] if item["count"] else 0.0,
            }
            for index, item in enumerate(confidence_bins)
        }
        if name == "overall":
            result["overall"] = metrics
        else:
            result["by_axis"][name] = metrics
    return result


@dataclass(frozen=True)
class ResidualRunConfig:
    """Trainer runner configuration; validation never participates in selection."""

    run_id: str
    train_manifest: str
    train_rows: str
    validation_manifest: str
    validation_rows: str
    output_dir: str
    public_aggregate_path: str
    seeds: tuple[int, ...] = (2026,)
    selection_split_seed: int = 1729
    dev_ratio: float = 0.2
    hidden_dim: int = 256
    dropout: float = 0.1
    prior_temperature: float = 0.75
    raw_target_loss_weight: float = 0.25
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_train_epochs: float = 10.0
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    use_cpu: bool = False
    blend_weight_candidates: tuple[float, ...] = (0.5, 0.75, 1.0)
    posterior_temperature_candidates: tuple[float, ...] = (1.0,)
    threshold_candidates: tuple[tuple[float, float, float, float], ...] = ((1.5, 2.5, 3.5, 4.5),)

    @classmethod
    def from_json(cls, path: str | Path) -> "ResidualRunConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        _need(isinstance(raw, Mapping), "run config must be an object")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        _need(not unknown, "run config has unknown fields")
        # Backward compatibility for configs written before temperature scaling.
        raw.setdefault("posterior_temperature_candidates", [1.0])
        raw.setdefault("selection_split_seed", 1729)
        _need(set(raw) == set(cls.__dataclass_fields__), "run config has missing fields")
        for key in ("seeds", "blend_weight_candidates", "posterior_temperature_candidates", "threshold_candidates"):
            raw[key] = tuple(tuple(item) if isinstance(item, list) else item for item in raw[key])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        _need(isinstance(self.run_id, str) and bool(self.run_id), "run_id is required")
        _need(bool(self.seeds) and len(set(self.seeds)) == len(self.seeds) and all(isinstance(seed, int) for seed in self.seeds), "seeds must be unique integers")
        _need(isinstance(self.selection_split_seed, int) and not isinstance(self.selection_split_seed, bool), "selection_split_seed must be an integer")
        _need(0 < self.dev_ratio < 0.5 and self.hidden_dim > 0 and 0 <= self.dropout < 1, "invalid model/split config")
        _need(self.prior_temperature > 0 and self.raw_target_loss_weight >= 0, "invalid loss config")
        _need(self.learning_rate > 0 and self.weight_decay >= 0 and self.num_train_epochs > 0, "invalid training config")
        _need(self.per_device_train_batch_size > 0 and self.per_device_eval_batch_size > 0, "invalid batch config")
        _need(isinstance(self.use_cpu, bool), "use_cpu must be a boolean")
        _need(bool(self.blend_weight_candidates) and all(0 <= value <= 1 for value in self.blend_weight_candidates), "invalid blend candidates")
        _need(bool(self.posterior_temperature_candidates) and all(value > 0 and math.isfinite(value) for value in self.posterior_temperature_candidates), "invalid posterior temperature candidates")
        _need(bool(self.threshold_candidates), "threshold candidates are required")
        for values in self.threshold_candidates:
            _need(len(values) == 4 and all(a < b for a, b in zip(values, values[1:])), "invalid threshold candidate")
        _need(Path(self.output_dir).resolve() != Path(self.public_aggregate_path).resolve(), "model output and public aggregate paths must differ")


def _trainer_dataset(rows: Sequence[ResidualRow]) -> Any:
    import torch

    class Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> Mapping[str, Any]:
            row = rows[index]
            return {
                "shared_embedding": torch.tensor(row.shared_embedding, dtype=torch.float32),
                "base_predictions": torch.tensor(row.base_predictions, dtype=torch.float32),
                "labels": torch.tensor(row.labels, dtype=torch.long),
                "raw_labels": torch.tensor(row.raw_labels, dtype=torch.float32),
            }

    return Dataset()


def _logits_from_prediction_output(output: Any) -> Any:
    import torch

    predictions = output.predictions
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    return torch.as_tensor(logits).float().cpu()


def _temperature_probabilities(logits: Any, temperature: Real) -> Any:
    import torch

    value = float(temperature)
    _need(math.isfinite(value) and value > 0, "posterior temperature must be positive and finite")
    return torch.softmax(torch.as_tensor(logits).float() / value, dim=-1).cpu().tolist()


def run_residual_experiment(config: ResidualRunConfig) -> dict[str, Any]:
    """Train on train-only group splits and evaluate validation exactly once."""
    config.validate()
    train_manifest, train_rows = load_embedding_artifact(config.train_manifest, config.train_rows)
    # Only validation metadata/checksum is inspected before model selection;
    # validation rows and gold labels remain unparsed until the final block.
    validation_manifest = EmbeddingArtifactManifest.from_json(config.validation_manifest)
    _need(_file_sha256(Path(config.validation_rows)) == validation_manifest.rows_sha256, "validation rows checksum mismatch")
    validate_split_contracts(train_manifest.base_contract(), validation_manifest.base_contract())
    _need(train_manifest.embedding_dim == validation_manifest.embedding_dim, "train/validation embedding dimensions differ")
    _need(train_manifest.embedding_model_id == validation_manifest.embedding_model_id and train_manifest.embedding_model_revision == validation_manifest.embedding_model_revision, "train/validation embeddings use different frozen models")
    try:
        from transformers import Trainer, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("residual runner requires the existing training environment") from exc

    output_root = Path(config.output_dir)
    _need(not output_root.exists(), "output_dir must be new")
    output_root.mkdir(parents=True)
    # The data split is fixed independently of model initialization seeds, so
    # every candidate is compared on exactly the same train/dev examples.
    selection_train, selection_dev = group_selection_split(
        train_rows, seed=config.selection_split_seed, dev_ratio=config.dev_ratio
    )
    candidates: list[tuple[float, int, float, float, tuple[float, ...], dict[str, Any]]] = []
    for seed in config.seeds:
        set_seed(seed)
        model = build_ordinal_residual_model(
            train_manifest.embedding_dim, hidden_dim=config.hidden_dim, dropout=float(config.dropout),
            prior_temperature=config.prior_temperature, raw_target_loss_weight=config.raw_target_loss_weight,
        )
        seed_output = output_root / f"selection-seed-{seed}"
        arguments = TrainingArguments(
            output_dir=str(seed_output), do_train=True, do_eval=True,
            eval_strategy="epoch", save_strategy="no", logging_strategy="epoch",
            learning_rate=config.learning_rate, weight_decay=config.weight_decay,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            load_best_model_at_end=False,
            seed=seed, data_seed=seed, report_to="none", remove_unused_columns=False,
            use_cpu=config.use_cpu,
        )
        trainer = Trainer(model=model, args=arguments, train_dataset=_trainer_dataset(selection_train), eval_dataset=_trainer_dataset(selection_dev))
        trainer.train()
        dev_output = trainer.predict(_trainer_dataset(selection_dev), metric_key_prefix="selection_dev")
        dev_logits = _logits_from_prediction_output(dev_output)
        base = [row.base_predictions for row in selection_dev]
        gold = [row.raw_labels for row in selection_dev]
        best_decode: tuple[float, float, float, tuple[float, ...], dict[str, Any]] | None = None
        for temperature in config.posterior_temperature_candidates:
            probabilities = _temperature_probabilities(dev_logits, temperature)
            for weight in config.blend_weight_candidates:
                for thresholds in config.threshold_candidates:
                    metrics = aggregate_predictions(base, gold, probabilities, blend_weight=weight, thresholds=thresholds)
                    overall = metrics["overall"]
                    rank = (overall["residual_raw_rmse"], -overall["residual_qwk"], overall["posterior_nll"])
                    if best_decode is None or rank < (
                        best_decode[0], -best_decode[4]["overall"]["residual_qwk"], best_decode[4]["overall"]["posterior_nll"]
                    ):
                        best_decode = (overall["residual_raw_rmse"], float(temperature), float(weight), tuple(float(v) for v in thresholds), metrics)
        assert best_decode is not None
        candidates.append((best_decode[0], seed, best_decode[1], best_decode[2], best_decode[3], best_decode[4]))
    _, selected_seed, selected_temperature, selected_weight, selected_thresholds, dev_metrics = min(
        candidates, key=lambda item: (item[0], -item[5]["overall"]["residual_qwk"], item[5]["overall"]["posterior_nll"], item[1])
    )

    # Fresh refit: the selected seed/config is trained for the same fixed epoch
    # budget on every train row. No selection checkpoint or validation data is
    # reused, and this Trainer has no evaluation dataset.
    set_seed(selected_seed)
    refit_model = build_ordinal_residual_model(
        train_manifest.embedding_dim, hidden_dim=config.hidden_dim, dropout=float(config.dropout),
        prior_temperature=config.prior_temperature, raw_target_loss_weight=config.raw_target_loss_weight,
    )
    refit_arguments = TrainingArguments(
        output_dir=str(output_root / "refit-full-train"), do_train=True, do_eval=False,
        eval_strategy="no", save_strategy="no", logging_strategy="epoch",
        learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        load_best_model_at_end=False, seed=selected_seed, data_seed=selected_seed,
        report_to="none", remove_unused_columns=False, use_cpu=config.use_cpu,
    )
    final_trainer = Trainer(model=refit_model, args=refit_arguments, train_dataset=_trainer_dataset(train_rows))
    final_trainer.train()

    # This is the sole access/evaluation of held-out validation in the runner.
    _, validation_rows = load_embedding_artifact(config.validation_manifest, config.validation_rows)
    _need(not ({row.source_id for row in train_rows} & {row.source_id for row in validation_rows}), "train/validation IDs overlap")
    validation_output = final_trainer.predict(_trainer_dataset(validation_rows), metric_key_prefix="validation_final_once")
    validation_probabilities = _temperature_probabilities(_logits_from_prediction_output(validation_output), selected_temperature)
    validation_metrics = aggregate_predictions(
        [row.base_predictions for row in validation_rows], [row.raw_labels for row in validation_rows],
        validation_probabilities, blend_weight=selected_weight, thresholds=selected_thresholds,
    )
    final_trainer.save_model(str(output_root / "selected_model"))
    aggregate = {
        "data": {"train_count": len(train_rows), "validation_count": len(validation_rows), "embedding_dim": train_manifest.embedding_dim},
        "selection": {
            "candidate_seed_count": len(config.seeds), "selected_seed": selected_seed,
            "split_seed": config.selection_split_seed,
            "train_count": len(selection_train), "dev_count": len(selection_dev),
            "selected_posterior_temperature": selected_temperature,
            "selected_blend_weight": selected_weight,
            "threshold_1": selected_thresholds[0], "threshold_2": selected_thresholds[1],
            "threshold_3": selected_thresholds[2], "threshold_4": selected_thresholds[3],
            "dev": dev_metrics,
        },
        "refit": {"full_train_count": len(train_rows), "epoch_budget": config.num_train_epochs, "fresh_model": 1},
        "validation_final_once": {"metrics": validation_metrics, "evaluation_call_count": 1},
    }
    write_public_aggregate(config.public_aggregate_path, aggregate)
    return aggregate


def write_public_aggregate(path: str | Path, aggregate: Mapping[str, Any]) -> Path:
    """Write numeric aggregate metrics only, rejecting row-level payloads."""
    _need(isinstance(aggregate, Mapping) and bool(aggregate), "public aggregate must be a nonempty mapping")

    def validate(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            _need(bool(value), f"empty aggregate mapping at {location}")
            for key, child in value.items():
                _need(isinstance(key, str) and key not in {"id", "source_id", "essay", "prompt", "prediction"}, f"row-level field forbidden at {location}")
                validate(child, f"{location}.{key}")
            return
        _need(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)), f"public aggregate leaves must be finite numbers at {location}")

    validate(aggregate, "aggregate")
    output = Path(path)
    _need(not output.exists(), "public aggregate output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output
