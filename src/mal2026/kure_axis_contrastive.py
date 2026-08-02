"""Axis-wise ordinal contrastive adaptation of the pinned KURE encoder.

The implementation is intentionally self-contained because the loss couples
examples inside a deterministic score-balanced batch and selection evaluates
fit-only prototypes after every epoch.  Standard ``Trainer`` random batching
cannot enforce that contract without replacing its dataloader.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

from .official_score_matrix import AXES, ScoreRow, deterministic_internal_split, file_sha256, load_score_rows
from .official_score_prompt import USER_SUPPLIED_EVALUATION, embedding_input, provenance as prompt_provenance


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "nlpai-lab/KURE-v1"
MODEL_REVISION = "d14c8a9423946e268a0c9952fecf3a7aabd73bd9"
MODEL_PATH = ROOT / "outputs/model-cache/nlpai-lab--KURE-v1-d14c8a9423946e268a0c9952fecf3a7aabd73bd9"
OUTPUT_ROOT = ROOT / "outputs/kure-axis-ordinal-contrastive-v1"
TRAIN_PATH = ROOT / "eval/train.jsonl"
VALIDATION_PATH = ROOT / "eval/validation.jsonl"
TRAIN_SHA256 = "b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737"
VALIDATION_SHA256 = "0805445029328848164cf15f34b90b88fb5f7896d7b73f24f1717b733a9a00a4"
MODEL_CONFIG_SHA256 = "852d42e020c7f989c2acaf30fc683b7f768e8c6d1ab17166e835442162bd825d"
AIHUB_COMPLETION_SHA256 = "c91704e5a5c5f54b086552731fe87febdaee8c42273f93a5492f1f8626b47959"
AIHUB_ARTIFACT_SHA256 = "ffdc985d56c655c03e8964927b127b24f0c5bb7fdde8d89e944941f5419cf25a"
ARMS = ("base", "aihub_full_backbone")
QUOTAS = (2, 4, 5, 5, 4)
METHODS = ("continuous_head", "prototype_soft", "hybrid", "center_0.5", "center_0.1", "cluster_k2")
LORA_TARGETS = ("query", "key", "value", "dense")


class KUREAxisContrastiveError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise KUREAxisContrastiveError(message)


@dataclass(frozen=True)
class AxisContrastiveConfig:
    schema_version: str
    run_id: str
    arm: str
    axis: str
    model_id: str
    model_revision: str
    model_path: str
    warmstart_completion_path: str
    warmstart_completion_sha256: str
    warmstart_artifact_path: str
    warmstart_artifact_sha256: str
    train_path: str
    train_sha256: str
    validation_path: str
    validation_sha256: str
    output_root: str
    seed: int
    max_length: int
    selection_epochs: tuple[int, ...]
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    temperature: float
    compact_margin: float
    soft_close_margin: float
    far_margin: float
    rank_margin: float
    contrastive_weight: float
    rank_weight: float
    coral_weight: float
    cluster_k: int
    prototype_temperature: float
    hybrid_weight: float

    @classmethod
    def from_json(cls, path: Path, *, require_dependencies: bool = True) -> "AxisContrastiveConfig":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KUREAxisContrastiveError("axis contrastive config is unreadable") from exc
        need(isinstance(raw, dict) and isinstance(raw.get("selection_epochs"), list), "axis contrastive config differs")
        raw["selection_epochs"] = tuple(raw["selection_epochs"])
        need(set(raw) == set(cls.__dataclass_fields__), "axis contrastive config fields differ")
        value = cls(**raw)
        value.validate(require_dependencies=require_dependencies)
        return value

    def validate(self, *, require_dependencies: bool = True) -> None:
        need(self.schema_version == "mal2026-kure-axis-ordinal-contrastive-v1", "schema differs")
        need(self.arm in ARMS and self.axis in AXES, "arm or axis differs")
        short = "aihub" if self.arm == "aihub_full_backbone" else "base"
        need(self.run_id == f"kure-axis-contrastive-{short}-{self.axis}-20260802-001", "run identity differs")
        need((self.model_id, self.model_revision) == (MODEL_ID, MODEL_REVISION), "KURE pin differs")
        need(Path(self.model_path).resolve() == MODEL_PATH.resolve(), "KURE path differs")
        need(Path(self.train_path).resolve() == TRAIN_PATH.resolve() and self.train_sha256 == TRAIN_SHA256, "train pin differs")
        need(Path(self.validation_path).resolve() == VALIDATION_PATH.resolve() and self.validation_sha256 == VALIDATION_SHA256, "validation pin differs")
        need(Path(self.output_root).resolve() == OUTPUT_ROOT.resolve(), "output root differs")
        need(self.selection_epochs == tuple(range(1, 7)) and self.max_length == 1536, "epoch or length contract differs")
        expected_seed = {"content": 2026080201, "organization": 2026080202, "expression": 2026080203}[self.axis]
        need(self.seed == expected_seed, "axis seed differs")
        need((self.learning_rate, self.weight_decay, self.warmup_ratio) == (5e-5, 0.01, 0.10), "optimizer contract differs")
        need((self.per_device_train_batch_size, self.per_device_eval_batch_size, self.gradient_accumulation_steps) == (20, 32, 2), "batch contract differs")
        need((self.lora_r, self.lora_alpha, self.lora_dropout) == (16, 32, 0.05), "LoRA contract differs")
        need((self.temperature, self.compact_margin, self.soft_close_margin, self.far_margin, self.rank_margin) == (0.10, 0.05, 0.10, 0.20, 0.10), "contrastive margin contract differs")
        need((self.contrastive_weight, self.rank_weight, self.coral_weight) == (0.20, 0.10, 0.30), "loss weights differ")
        need((self.cluster_k, self.prototype_temperature, self.hybrid_weight) == (2, 0.10, 0.50), "inference contract differs")
        if self.arm == "base":
            need(not any((self.warmstart_completion_path, self.warmstart_completion_sha256, self.warmstart_artifact_path, self.warmstart_artifact_sha256)), "base arm has a warm start")
        else:
            need(self.warmstart_completion_sha256 == AIHUB_COMPLETION_SHA256 and self.warmstart_artifact_sha256 == AIHUB_ARTIFACT_SHA256, "AI-Hub hashes differ")
        if require_dependencies:
            need(MODEL_PATH.is_dir() and not MODEL_PATH.is_symlink(), "KURE snapshot is unavailable")
            need(file_sha256(MODEL_PATH / "config.json") == MODEL_CONFIG_SHA256, "KURE config checksum differs")
            need(TRAIN_PATH.is_file() and file_sha256(TRAIN_PATH) == TRAIN_SHA256, "train file differs")
            # Hashing is allowed before the isolation gate; labels/rows are not loaded.
            need(VALIDATION_PATH.is_file() and file_sha256(VALIDATION_PATH) == VALIDATION_SHA256, "validation file differs")
            if self.arm == "aihub_full_backbone":
                completion, artifact = Path(self.warmstart_completion_path), Path(self.warmstart_artifact_path)
                need(completion.is_file() and not completion.is_symlink() and file_sha256(completion) == AIHUB_COMPLETION_SHA256, "AI-Hub completion differs")
                need(artifact.is_file() and not artifact.is_symlink() and file_sha256(artifact) == AIHUB_ARTIFACT_SHA256, "AI-Hub model differs")
        prompt_provenance(USER_SUPPLIED_EVALUATION)


def render_input(row: ScoreRow) -> str:
    return embedding_input(row.prompt, row.essay, USER_SUPPLIED_EVALUATION)


def axis_labels(rows: Sequence[ScoreRow], axis: str) -> list[int]:
    index = AXES.index(axis)
    return [int(row.labels[index]) for row in rows]


def token_length_audit(rows: Sequence[ScoreRow], tokenizer: Any, max_length: int) -> dict[str, int]:
    lengths: list[int] = []
    for start in range(0, len(rows), 128):
        texts = [render_input(row) for row in rows[start:start + 128]]
        lengths.extend(len(ids) for ids in tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"])
    need(bool(lengths) and max(lengths) <= max_length, "KURE MAL input would be truncated")
    ordered = sorted(lengths)
    return {
        "records": len(ordered), "maximum": ordered[-1],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "max_length": max_length, "truncated_records": 0,
    }


class EncodedRows:
    def __init__(self, rows: Sequence[ScoreRow], labels: Sequence[int], tokenizer: Any, max_length: int) -> None:
        self.labels = list(labels)
        self.identifiers = [row.identifier for row in rows]
        encoded: list[list[int]] = []
        for start in range(0, len(rows), 128):
            texts = [render_input(row) for row in rows[start:start + 128]]
            encoded.extend(tokenizer(texts, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"])
        need(len(encoded) == len(rows) == len(labels), "encoded row count differs")
        self.input_ids = encoded

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"input_ids": self.input_ids[index], "label": self.labels[index], "index": index}


def make_collator(tokenizer: Any, importance: Mapping[int, float] | None = None):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        padded = tokenizer.pad([{"input_ids": row["input_ids"]} for row in features], padding=True, return_tensors="pt")
        labels = torch.tensor([row["label"] for row in features], dtype=torch.float32)
        weights = torch.tensor([1.0 if importance is None else importance[int(row["label"])] for row in features], dtype=torch.float32)
        return {**padded, "labels": labels, "importance": weights, "indices": torch.tensor([row["index"] for row in features])}
    return collate


class ScoreBalancedBatchSampler:
    """Deterministic five-score batches with no duplicate row inside a batch."""

    def __init__(self, labels: Sequence[int], seed: int, *, batch_size: int = 20, smoke_batches: int | None = None) -> None:
        need(batch_size == sum(QUOTAS), "balanced sampler batch size differs")
        self.seed = seed
        self.epoch = 0
        self.smoke_batches = smoke_batches
        self.buckets = {score: [i for i, value in enumerate(labels) if value == score] for score in range(1, 6)}
        need(all(len(self.buckets[score]) >= QUOTAS[score - 1] for score in range(1, 6)), "a score bucket cannot fill one duplicate-free batch")
        self.batch_count = smoke_batches if smoke_batches is not None else math.ceil(len(labels) / batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self):
        rng = random.Random(self.seed + 1009 * self.epoch)
        buckets = {score: values[:] for score, values in self.buckets.items()}
        for values in buckets.values():
            rng.shuffle(values)
        cursor = {score: 0 for score in buckets}
        for _ in range(self.batch_count):
            batch: list[int] = []
            for score, quota in zip(range(1, 6), QUOTAS, strict=True):
                chosen: list[int] = []
                while len(chosen) < quota:
                    if cursor[score] >= len(buckets[score]):
                        rng.shuffle(buckets[score])
                        cursor[score] = 0
                    candidate = buckets[score][cursor[score]]
                    cursor[score] += 1
                    if candidate not in chosen:
                        chosen.append(candidate)
                batch.extend(chosen)
            rng.shuffle(batch)
            need(len(batch) == len(set(batch)) == sum(QUOTAS), "balanced batch contains a duplicate")
            yield batch


def importance_by_score(labels: Sequence[int]) -> dict[int, float]:
    counts = {score: labels.count(score) for score in range(1, 6)}
    n, batch = len(labels), sum(QUOTAS)
    raw = {score: (counts[score] / n) / (QUOTAS[score - 1] / batch) for score in range(1, 6)}
    mean_under_batch = sum((QUOTAS[score - 1] / batch) * raw[score] for score in range(1, 6))
    return {score: raw[score] / mean_under_batch for score in range(1, 6)}


def ordinal_pair_target(labels: Any) -> Any:
    import torch
    distance = (labels[:, None] - labels[None, :]).abs()
    return torch.cos(math.pi * distance / 4.0)


def ordinal_contrastive_loss(embeddings: Any, labels: Any, config: AxisContrastiveConfig) -> tuple[Any, dict[str, float], Any]:
    import torch
    import torch.nn.functional as functional
    need(embeddings.ndim == 2 and labels.ndim == 1 and len(embeddings) == len(labels), "contrastive tensor shape differs")
    similarity = embeddings.float() @ embeddings.float().T
    mask = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pair = functional.smooth_l1_loss(similarity[mask], ordinal_pair_target(labels)[mask])
    terms: list[Any] = []
    compact_terms: list[Any] = []
    for i in range(len(labels)):
        distance = (labels - labels[i]).abs()
        same = torch.where((distance == 0) & (torch.arange(len(labels), device=labels.device) != i))[0]
        adjacent = torch.where(distance == 1)[0]
        far = torch.where(distance >= 2)[0]
        if not (len(same) and len(adjacent) and len(far)):
            continue
        worst_positive = similarity[i, same].min()
        hardest_adjacent = similarity[i, adjacent].max()
        weakest_adjacent = similarity[i, adjacent].min()
        hardest_far = similarity[i, far].max()
        compact_terms.append(functional.relu((1.0 - config.compact_margin) - worst_positive))
        terms.append(functional.relu(config.soft_close_margin + hardest_adjacent - worst_positive))
        terms.append(functional.relu(config.far_margin + hardest_far - weakest_adjacent))
        # The label-gap-scaled term directly penalizes the worst ordinal inversion.
        far_gap = distance[far][similarity[i, far].argmax()].float().clamp_min(2.0)
        terms.append(functional.softplus((hardest_far - worst_positive + config.rank_margin * far_gap) / config.temperature) * config.temperature)
    zero = similarity.sum() * 0.0
    rank = torch.stack(terms).mean() if terms else zero
    compact = torch.stack(compact_terms).mean() if compact_terms else zero
    rank = rank + compact
    return pair, {"pair": float(pair.detach()), "rank": float(rank.detach())}, rank


def build_model(config: AxisContrastiveConfig) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from peft import LoraConfig, TaskType, get_peft_model
    from safetensors import safe_open
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        config.model_path, revision=config.model_revision, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        add_pooling_layer=False,
    )
    need(getattr(backbone.config, "hidden_size", None) == 1024, "KURE hidden size differs")
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    initialization: dict[str, Any] = {"arm": config.arm, "pooling": "cls_l2", "loaded_backbone_tensors": 0}
    if config.arm == "aihub_full_backbone":
        targets = dict(backbone.named_parameters())
        loaded: set[str] = set()
        with safe_open(config.warmstart_artifact_path, framework="pt", device="cpu") as handle:
            for artifact_name in handle.keys():
                if artifact_name.startswith("score_head."):
                    continue
                need(artifact_name.startswith("backbone."), f"unexpected AI-Hub tensor: {artifact_name}")
                name = artifact_name.removeprefix("backbone.")
                need(name in targets and name not in loaded, f"AI-Hub backbone tensor differs: {name}")
                tensor = handle.get_tensor(artifact_name)
                need(tuple(tensor.shape) == tuple(targets[name].shape), f"AI-Hub tensor shape differs: {name}")
                targets[name].data.copy_(tensor.to(dtype=targets[name].dtype))
                loaded.add(name)
        need(set(targets) == loaded, "AI-Hub warm start is missing backbone parameters")
        initialization.update({"loaded_backbone_tensors": len(loaded), "artifact_sha256": config.warmstart_artifact_sha256})
    leaves = {name.rsplit(".", 1)[-1] for name, _ in backbone.named_modules()}
    need(set(LORA_TARGETS) <= leaves, "KURE LoRA targets differ")
    backbone = get_peft_model(backbone, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, target_modules=list(LORA_TARGETS), bias="none",
    ))
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    class AxisModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.projection = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 256))
            self.score_head = nn.Linear(256, 1)
            self.coral_head = nn.Linear(256, 4)

        def forward(self, input_ids: Any, attention_mask: Any) -> Mapping[str, Any]:
            hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state[:, 0].float()
            pooled = functional.normalize(hidden, p=2, dim=-1)
            projected = functional.normalize(self.projection(pooled), p=2, dim=-1)
            score = 1.0 + 4.0 * torch.sigmoid(self.score_head(projected).squeeze(-1).float())
            return {"embedding": projected, "score": score, "coral_logits": self.coral_head(projected).float()}

    model = AxisModel()
    need(any("lora_" in name and parameter.requires_grad for name, parameter in model.named_parameters()), "fresh KURE LoRA is absent")
    need(not any(parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("backbone.") and "lora_" not in name), "base KURE parameters are trainable")
    initialization["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    initialization["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return model, initialization


def training_loss(model: Any, batch: Mapping[str, Any], config: AxisContrastiveConfig) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as functional
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    labels, weights = batch["labels"].float(), batch["importance"].float()
    regression_rows = functional.smooth_l1_loss(output["score"], labels, beta=0.25, reduction="none")
    regression = (regression_rows * weights).sum() / weights.sum()
    thresholds = torch.tensor([1.5, 2.5, 3.5, 4.5], device=labels.device)
    targets = (labels[:, None] > thresholds[None, :]).float()
    coral_rows = functional.binary_cross_entropy_with_logits(output["coral_logits"], targets, reduction="none").mean(dim=1)
    coral = (coral_rows * weights).sum() / weights.sum()
    pair, detail, rank = ordinal_contrastive_loss(output["embedding"], labels, config)
    loss = regression + config.coral_weight * coral + config.contrastive_weight * pair + config.rank_weight * rank
    values = {"loss": float(loss.detach()), "regression": float(regression.detach()), "coral": float(coral.detach()), **detail}
    need(all(math.isfinite(value) for value in values.values()), "training loss is non-finite")
    return loss, values


def _move(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}


def _seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dataloader(dataset: EncodedRows, tokenizer: Any, config: AxisContrastiveConfig, *, train: bool, epoch: int = 0, smoke_batches: int | None = None):
    from torch.utils.data import DataLoader
    if train:
        sampler = ScoreBalancedBatchSampler(dataset.labels, config.seed, batch_size=config.per_device_train_batch_size, smoke_batches=smoke_batches)
        sampler.set_epoch(epoch)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=make_collator(tokenizer, importance_by_score(dataset.labels)), num_workers=0, pin_memory=True), sampler
    return DataLoader(dataset, batch_size=config.per_device_eval_batch_size, shuffle=False, collate_fn=make_collator(tokenizer), num_workers=0, pin_memory=True), None


def _train_epochs(model: Any, dataset: EncodedRows, tokenizer: Any, config: AxisContrastiveConfig, epochs: int, *, evaluate: Any | None = None, smoke: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import get_linear_schedule_with_warmup
    device = torch.device("cuda")
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay, fused=True)
    base_loader, _ = _dataloader(dataset, tokenizer, config, train=True, smoke_batches=2 if smoke else None)
    updates_per_epoch = math.ceil(len(base_loader) / (1 if smoke else config.gradient_accumulation_steps))
    total_updates = max(1, epochs * updates_per_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * config.warmup_ratio), total_updates)
    events: list[dict[str, Any]] = []
    global_step = 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        loader, sampler = _dataloader(dataset, tokenizer, config, train=True, epoch=epoch, smoke_batches=2 if smoke else None)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        batches = 0
        for step, raw in enumerate(loader, 1):
            batch = _move(raw, device)
            loss, details = training_loss(model, batch, config)
            accumulation = 1 if smoke else config.gradient_accumulation_steps
            (loss / accumulation).backward()
            if step % accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                global_step += 1
            for key, value in details.items():
                sums[key] = sums.get(key, 0.0) + value
            batches += 1
        event: dict[str, Any] = {
            "epoch": epoch, "global_step": global_step, "batches": batches,
            "train_losses": {key: value / batches for key, value in sorted(sums.items())},
            "sampler": sampler_audit(dataset.labels, sampler),
        }
        if evaluate is not None:
            event["evaluation"] = evaluate(model, epoch)
        events.append(event)
    return events, {"global_step": global_step, "runtime_seconds": time.monotonic() - started}


def sampler_audit(labels: Sequence[int], sampler: ScoreBalancedBatchSampler | None) -> dict[str, Any]:
    need(sampler is not None, "sampler audit requires a sampler")
    counts = {str(score): 0 for score in range(1, 6)}
    duplicates = 0
    for batch in sampler:
        duplicates += len(batch) - len(set(batch))
        for index in batch:
            counts[str(labels[index])] += 1
    return {"batches": len(sampler), "sampled_by_score": counts, "within_batch_duplicates": duplicates, "quota": list(QUOTAS)}


def embeddings_and_head(model: Any, dataset: EncodedRows, tokenizer: Any, config: AxisContrastiveConfig) -> tuple[Any, Any, Any]:
    import torch
    loader, _ = _dataloader(dataset, tokenizer, config, train=False)
    model.eval()
    embeddings: list[Any] = []
    scores: list[Any] = []
    labels: list[Any] = []
    with torch.inference_mode():
        for raw in loader:
            batch = _move(raw, torch.device("cuda"))
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            embeddings.append(output["embedding"].float().cpu())
            scores.append(output["score"].float().cpu())
            labels.append(batch["labels"].float().cpu())
    return torch.cat(embeddings), torch.cat(scores), torch.cat(labels)


def _normalize(value: Any) -> Any:
    import torch.nn.functional as functional
    return functional.normalize(value.float(), p=2, dim=-1)


def label_centroids(embeddings: Any, labels: Any) -> Any:
    import torch
    centroids = []
    for score in range(1, 6):
        selected = embeddings[labels == score]
        need(len(selected) > 0, f"score {score} has no prototype support")
        centroids.append(_normalize(selected.mean(dim=0, keepdim=True))[0])
    return torch.stack(centroids)


def interpolated_centroids(centroids: Any, step: float) -> tuple[Any, Any]:
    import torch
    need(step in {0.1, 0.5}, "centering step differs")
    count = int(round(4.0 / step)) + 1
    values = torch.linspace(1.0, 5.0, count)
    rows = []
    for value in values.tolist():
        low = min(4, int(math.floor(value)))
        high = max(2, int(math.ceil(value)))
        if math.isclose(value, round(value)):
            rows.append(centroids[int(round(value)) - 1])
        else:
            fraction = value - low
            rows.append(_normalize(((1.0 - fraction) * centroids[low - 1] + fraction * centroids[high - 1]).unsqueeze(0))[0])
    return torch.stack(rows), values


def spherical_score_clusters(embeddings: Any, labels: Any, k: int, seed: int) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    need(k == 2, "cluster contract differs")
    all_centroids, all_scores, support = [], [], {}
    for score in range(1, 6):
        points = embeddings[labels == score].float()
        actual_k = min(k, len(points))
        need(actual_k >= 1, "cluster score has no support")
        first = int((seed + score * 104729) % len(points))
        chosen = [points[first]]
        while len(chosen) < actual_k:
            current = torch.stack(chosen)
            distance = 1.0 - (points @ current.T).max(dim=1).values
            chosen.append(points[int(distance.argmax())])
        centroids = _normalize(torch.stack(chosen))
        assignment = torch.zeros(len(points), dtype=torch.long)
        for _ in range(20):
            assignment = (points @ centroids.T).argmax(dim=1)
            updated = []
            for cluster in range(actual_k):
                members = points[assignment == cluster]
                if not len(members):
                    distance = 1.0 - (points @ centroids.T).max(dim=1).values
                    updated.append(points[int(distance.argmax())])
                else:
                    updated.append(members.mean(dim=0))
            new_centroids = _normalize(torch.stack(updated))
            if torch.allclose(new_centroids, centroids, atol=1e-6, rtol=0):
                centroids = new_centroids
                break
            centroids = new_centroids
        cluster_counts = [int((assignment == cluster).sum()) for cluster in range(actual_k)]
        support[str(score)] = cluster_counts
        all_centroids.extend(centroids)
        all_scores.extend([float(score)] * actual_k)
    return torch.stack(all_centroids), torch.tensor(all_scores), support


def inference_methods(train_embeddings: Any, train_labels: Any, eval_embeddings: Any, head_scores: Any, config: AxisContrastiveConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    centroids = label_centroids(train_embeddings, train_labels)
    similarities = eval_embeddings.float() @ centroids.T
    probability = torch.softmax(similarities / config.prototype_temperature, dim=1)
    score_values = torch.arange(1, 6, dtype=torch.float32)
    prototype = probability @ score_values
    methods = {
        "continuous_head": head_scores.float().clamp(1, 5),
        "prototype_soft": prototype.clamp(1, 5),
        "hybrid": ((1.0 - config.hybrid_weight) * head_scores.float() + config.hybrid_weight * prototype).clamp(1, 5),
    }
    for step in (0.5, 0.1):
        grid_centroids, grid_values = interpolated_centroids(centroids, step)
        methods[f"center_{step:.1f}"] = grid_values[(eval_embeddings.float() @ grid_centroids.T).argmax(dim=1)].clamp(1, 5)
    cluster_centroids, cluster_scores, support = spherical_score_clusters(train_embeddings, train_labels, config.cluster_k, config.seed)
    cluster_probability = torch.softmax((eval_embeddings.float() @ cluster_centroids.T) / config.prototype_temperature, dim=1)
    methods["cluster_k2"] = (cluster_probability @ cluster_scores).clamp(1, 5)
    adjacent = [float((centroids[i] * centroids[i + 1]).sum()) for i in range(4)]
    nonadjacent = [float((centroids[i] * centroids[j]).sum()) for i in range(5) for j in range(i + 2, 5)]
    diagnostics = {
        "prototype_support": {str(score): int((train_labels == score).sum()) for score in range(1, 6)},
        "adjacent_cosines": adjacent, "nonadjacent_cosines": nonadjacent,
        "mean_adjacent_cosine": sum(adjacent) / len(adjacent),
        "mean_nonadjacent_cosine": sum(nonadjacent) / len(nonadjacent),
        "cluster_support": support,
    }
    need(set(methods) == set(METHODS), "inference method inventory differs")
    return methods, diagnostics


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for position in range(cursor, end):
            result[order[position]] = rank
        cursor = end
    return result


def _spearman(gold: Sequence[float], prediction: Sequence[float]) -> float:
    left, right = _ranks(gold), _ranks(prediction)
    ml, mr = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((a - ml) * (b - mr) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum((a - ml) ** 2 for a in left) * sum((b - mr) ** 2 for b in right))
    return numerator / denominator if denominator else 0.0


def score_axis(gold_tensor: Any, prediction_tensor: Any) -> dict[str, Any]:
    gold = [float(value) for value in gold_tensor.tolist()]
    prediction = [min(5.0, max(1.0, float(value))) for value in prediction_tensor.tolist()]
    integer = [min(5, max(1, math.floor(value + 0.5))) for value in prediction]
    def rmse(indices: Sequence[int]) -> float:
        return math.sqrt(sum((gold[i] - prediction[i]) ** 2 for i in indices) / len(indices)) if indices else float("nan")
    all_indices = list(range(len(gold)))
    per_score: dict[str, Any] = {}
    for score in range(1, 6):
        indices = [i for i, value in enumerate(gold) if value == score]
        per_score[str(score)] = {
            "count": len(indices), "continuous_rmse": rmse(indices),
            "integer_recall": sum(integer[i] == score for i in indices) / len(indices) if indices else float("nan"),
        }
    gold34 = [i for i, value in enumerate(gold) if value in {3.0, 4.0}]
    recall3 = sum(integer[i] == 3 for i in gold34 if gold[i] == 3) / sum(gold[i] == 3 for i in gold34)
    recall4 = sum(integer[i] == 4 for i in gold34 if gold[i] == 4) / sum(gold[i] == 4 for i in gold34)
    values = {
        "records": len(gold), "continuous_rmse": rmse(all_indices), "continuous_spearman": _spearman(gold, prediction),
        "integer_rmse": math.sqrt(sum((gold[i] - integer[i]) ** 2 for i in all_indices) / len(gold)),
        "integer_accuracy": sum(gold[i] == integer[i] for i in all_indices) / len(gold),
        "one_off_accuracy": sum(abs(gold[i] - integer[i]) <= 1 for i in all_indices) / len(gold),
        "low_1_2_rmse": rmse([i for i, value in enumerate(gold) if value <= 2]),
        "score5_rmse": rmse([i for i, value in enumerate(gold) if value == 5]),
        "gold34_balanced_accuracy": (recall3 + recall4) / 2.0,
        "gold3_to4_rate": sum(integer[i] == 4 for i in gold34 if gold[i] == 3) / sum(gold[i] == 3 for i in gold34),
        "gold4_to3_rate": sum(integer[i] == 3 for i in gold34 if gold[i] == 4) / sum(gold[i] == 4 for i in gold34),
        "per_score": per_score,
    }
    finite_required = ("continuous_rmse", "continuous_spearman", "integer_rmse", "integer_accuracy", "one_off_accuracy", "low_1_2_rmse", "score5_rmse", "gold34_balanced_accuracy")
    need(all(math.isfinite(float(values[key])) for key in finite_required), "axis metric is non-finite")
    return values


def evaluate_representation(model: Any, fit_dataset: EncodedRows, eval_dataset: EncodedRows, tokenizer: Any, config: AxisContrastiveConfig) -> dict[str, Any]:
    fit_embeddings, _, fit_labels = embeddings_and_head(model, fit_dataset, tokenizer, config)
    eval_embeddings, head_scores, eval_labels = embeddings_and_head(model, eval_dataset, tokenizer, config)
    methods, diagnostics = inference_methods(fit_embeddings, fit_labels, eval_embeddings, head_scores, config)
    return {"methods": {name: score_axis(eval_labels, prediction) for name, prediction in methods.items()}, "representation": diagnostics}


def trainable_state(model: Any) -> dict[str, Any]:
    state = {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters() if parameter.requires_grad}
    need(any("lora_" in name for name in state) and any(name.startswith("projection.") for name in state), "trainable state is incomplete")
    need(not any("average" in name for name in state), "average state leaked")
    return state


def _initialize(config: AxisContrastiveConfig) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoTokenizer
    _seed_everything(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    need(tokenizer.pad_token_id is not None, "KURE tokenizer has no pad token")
    model, lineage = build_model(config)
    return tokenizer, model, lineage


def _make_smoke_split(rows: Sequence[ScoreRow], axis: str) -> tuple[list[ScoreRow], list[ScoreRow]]:
    labels = axis_labels(rows, axis)
    buckets = {score: [row for row, label in zip(rows, labels, strict=True) if label == score] for score in range(1, 6)}
    needed = {score: max(QUOTAS[score - 1], 4) for score in range(1, 6)}
    need(all(len(buckets[score]) >= needed[score] for score in buckets), "smoke cannot satisfy the score quotas")
    train = [row for score in range(1, 6) for row in buckets[score][:needed[score]]]
    dev = [row for score in range(1, 6) for row in buckets[score][needed[score]:needed[score] + 2]]
    # Expression score 1 has exactly four train rows, so use a disjoint dev
    # score-1 example only when available and otherwise a train-only smoke eval.
    if any(len(buckets[score]) < needed[score] + 2 for score in buckets):
        dev = train[:]
    return train, dev


def run(config: AxisContrastiveConfig, *, smoke: bool = False) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    config.validate(require_dependencies=True)
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one physical GPU must be exposed per axis job")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output = Path(config.output_root) / (f"smoke-{config.run_id}" if smoke else config.run_id)
    need(not output.exists(), f"refusing to overwrite axis contrastive output: {output}")
    output.mkdir(parents=True)
    all_train = load_score_rows(Path(config.train_path), config.train_sha256, 2000)
    split_train, split_dev, split_fingerprint = deterministic_internal_split(all_train, config.seed)
    if smoke:
        split_train, split_dev = _make_smoke_split(all_train, config.axis)
    tokenizer, model, initialization = _initialize(config)
    audit_rows = split_train + split_dev if smoke else all_train
    train_token_audit = token_length_audit(audit_rows, tokenizer, config.max_length)
    selection_train = EncodedRows(split_train, axis_labels(split_train, config.axis), tokenizer, config.max_length)
    selection_dev = EncodedRows(split_dev, axis_labels(split_dev, config.axis), tokenizer, config.max_length)

    def evaluate(current_model: Any, _: int) -> dict[str, Any]:
        return evaluate_representation(current_model, selection_train, selection_dev, tokenizer, config)

    events, selection_runtime = _train_epochs(model, selection_train, tokenizer, config, 1 if smoke else len(config.selection_epochs), evaluate=evaluate, smoke=smoke)
    selected = min(events, key=lambda row: (
        float(row["evaluation"]["methods"]["hybrid"]["continuous_rmse"]),
        -float(row["evaluation"]["methods"]["hybrid"]["continuous_spearman"]), int(row["epoch"]),
    ))
    if smoke:
        payload = {
            "schema_version": "mal2026-kure-axis-ordinal-contrastive-result-v1", "status": "completed",
            "mode": "gpu0_two_update_smoke", "run_id": config.run_id, "arm": config.arm, "axis": config.axis,
            "average_read": False, "average_target_used": False, "selection": {"events": events, "selected_epoch": int(selected["epoch"])},
            "initialization": initialization, "train_token_length_audit": train_token_audit,
            **prompt_provenance(USER_SUPPLIED_EVALUATION), "config": asdict(config),
        }
        (output / "smoke_complete.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    del model
    torch.cuda.empty_cache()
    tokenizer, refit_model, refit_initialization = _initialize(config)
    need(refit_initialization == initialization, "selection/refit initialization differs")
    refit_dataset = EncodedRows(all_train, axis_labels(all_train, config.axis), tokenizer, config.max_length)
    refit_events, refit_runtime = _train_epochs(refit_model, refit_dataset, tokenizer, config, int(selected["epoch"]), evaluate=None)
    state_path = output / "selected_refit_trainable.safetensors"
    save_file(trainable_state(refit_model), str(state_path))

    # First row-level validation load occurs only after epoch selection and the
    # all-train refit have completed.
    validation = load_score_rows(Path(config.validation_path), config.validation_sha256, 400)
    validation_token_audit = token_length_audit(validation, tokenizer, config.max_length)
    validation_dataset = EncodedRows(validation, axis_labels(validation, config.axis), tokenizer, config.max_length)
    validation_result = evaluate_representation(refit_model, refit_dataset, validation_dataset, tokenizer, config)
    result = {
        "schema_version": "mal2026-kure-axis-ordinal-contrastive-result-v1", "status": "completed", "mode": "full",
        "run_id": config.run_id, "arm": config.arm, "axis": config.axis,
        "model_id": config.model_id, "model_revision": config.model_revision,
        "average_read": False, "average_target_used": False,
        "selection": {
            "source": "train_internal_prompt_stratified_1600_400_only", "split_fingerprint": split_fingerprint,
            "events": events, "selected_epoch": int(selected["epoch"]), "selected_primary": selected["evaluation"]["methods"]["hybrid"],
            "rule": "lowest hybrid continuous RMSE, highest hybrid Spearman, earlier epoch", "runtime": selection_runtime,
        },
        "refit": {"records": 2000, "epochs": int(selected["epoch"]), "events": refit_events, "runtime": refit_runtime},
        "canonical_validation": {"records": 400, "use": "single_final_descriptive_evaluation_not_selection", **validation_result},
        "initialization": initialization, "train_token_length_audit": train_token_audit,
        "validation_token_length_audit": validation_token_audit,
        "state_path": str(state_path.resolve()), "state_sha256": file_sha256(state_path),
        **prompt_provenance(USER_SUPPLIED_EVALUATION), "config": asdict(config),
        "privacy": "aggregate_only_no_rows_text_ids_embeddings_or_predictions_persisted",
    }
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
