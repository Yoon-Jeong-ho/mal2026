#!/usr/bin/env python3
"""Train one rationale-aware Qwen3-Embedding-8B or KURE score arm.

Selection and refit use both the train-only teacher and score-blind student
rationale views.  Internal development and frozen validation use only the
score-blind student view, with source-disjoint partitioning across views.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_score_matrix import file_sha256  # noqa: E402
from mal2026.rationale_aware_encoder import (  # noqa: E402
    MODEL_SPECS,
    ContinuousScoreRow,
    deterministic_split,
    load_continuous_rows,
)
from mal2026.rationale_pipeline_prompts import (  # noqa: E402
    AXES,
    rationale_to_score_text,
    regression_evaluation_score,
    round_half_up_score,
    routing,
)


OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-score-encoder-v1"
RESTRICTED_PARENT = ROOT / "data/processed/restricted/rationale_pipeline_v1/score_encoder"
TOKEN_AUDIT_PARENT = ROOT / "outputs/rationale-pipeline-score-encoder-token-audit-v1"
OBJECTIVES = ("bounded_regression", "categorical_5class")
INITIALIZATIONS = ("base", "aihub")
BALANCE_MODES = ("none", "per_axis_exact_inverse_frequency_loss")
TRAINING_PROTOCOLS = ("select_then_refit", "fixed_full_train")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class Config:
    schema_version: str
    run_id: str
    model_key: str
    model_id: str
    model_revision: str
    model_path: str
    objective: str
    initialization: str
    aihub_completion_path: str | None
    aihub_completion_sha256: str | None
    aihub_artifact_path: str | None
    aihub_artifact_sha256: str | None
    train_path: str
    train_sha256: str
    validation_path: str
    validation_sha256: str
    rationale_handoff_path: str
    rationale_handoff_sha256: str
    rationale_ratio: str
    seed: int
    epochs: tuple[int, ...]
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    selective_gradient_checkpointing_stride: int | None
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    training_dtype: str
    classification_weighting: str
    gpu_scope: tuple[int, ...]
    user_authorization: str
    score_balance_mode: str = "none"
    training_protocol: str = "select_then_refit"
    fixed_epochs: int | None = None
    fixed_epoch_source: str | None = None

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Keep the already completed unbalanced configurations reproducible
        # while making the balance contract explicit for new arms.
        raw.setdefault("score_balance_mode", "none")
        raw.setdefault("training_protocol", "select_then_refit")
        raw.setdefault("fixed_epochs", None)
        raw.setdefault("fixed_epoch_source", None)
        for key in ("epochs", "gpu_scope"):
            need(isinstance(raw.get(key), list), f"{key} differs")
            raw[key] = tuple(raw[key])
        need(set(raw) == set(cls.__dataclass_fields__), "score encoder config fields differ")
        value = cls(**raw); value.validate(); return value

    def validate(self) -> None:
        need(self.schema_version == "mal2026-rationale-pipeline-score-encoder-v1", "score encoder schema differs")
        need(self.model_key in MODEL_SPECS and self.objective in OBJECTIVES and self.initialization in INITIALIZATIONS, "score encoder arm differs")
        need(self.rationale_ratio in {"1to1", "1to2", "1to3"}, "score encoder rationale ratio differs")
        need(self.score_balance_mode in BALANCE_MODES, "score encoder balance mode differs")
        need(self.training_protocol in TRAINING_PROTOCOLS, "score encoder training protocol differs")
        spec = MODEL_SPECS[self.model_key]
        need((self.model_id, self.model_revision) == (spec["model_id"], spec["model_revision"]), "score encoder model pin differs")
        need(self.gpu_scope == (0, 1, 2, 3) and bool(self.user_authorization.strip()), "score encoder GPU authorization differs")
        need(self.epochs == tuple(range(1, 9)) and self.seed == 2026080707, "score encoder epoch inventory differs")
        if self.training_protocol == "select_then_refit":
            need(self.fixed_epochs is None and self.fixed_epoch_source is None, "select-then-refit received a fixed epoch")
        else:
            need(isinstance(self.fixed_epochs, int) and 1 <= self.fixed_epochs <= 8, "fixed-full epoch differs")
            need(isinstance(self.fixed_epoch_source, str) and bool(self.fixed_epoch_source.strip()), "fixed-full epoch source differs")
        need((self.learning_rate, self.weight_decay, self.warmup_ratio) == (1e-4, 0.01, 0.05), "score encoder optimizer differs")
        need((self.lora_r, self.lora_alpha, self.lora_dropout) == (16, 32, 0.05), "score encoder LoRA differs")
        expected_dtype = "bfloat16" if self.model_key == "qwen3_embedding_8b" else "float32"
        need(self.training_dtype == expected_dtype, "score encoder dtype differs")
        expected_classification_weighting = (
            "inverse_sqrt_train_class_frequency_normalized_per_axis"
            if self.score_balance_mode == "none"
            else "per_example_per_axis_exact_inverse_frequency"
        )
        need(self.classification_weighting == expected_classification_weighting, "classification weighting contract differs")
        # Each independent single-GPU arm keeps the original effective batch
        # while using the available 80-GiB memory: Qwen 8*4=32 and
        # KURE 16*4=64. Full checkpointing is required by the Qwen length
        # envelope and retained uniformly after bounded recovery testing.
        expected_batch = (8, 8, 1) if self.model_key == "qwen3_embedding_8b" else (16, 32, 1)
        need((self.per_device_train_batch_size, self.per_device_eval_batch_size, self.gradient_accumulation_steps) == expected_batch, "score encoder batch differs")
        need(self.gradient_checkpointing is True, "score encoder checkpointing contract differs")
        need(self.selective_gradient_checkpointing_stride is None, "score encoder selective checkpointing contract differs")
        expected_length = 2560 if self.model_key == "qwen3_embedding_8b" else 2048
        need(self.max_length == expected_length, "score encoder max length differs")
        for raw_path, digest in ((self.train_path, self.train_sha256), (self.validation_path, self.validation_sha256), (self.rationale_handoff_path, self.rationale_handoff_sha256)):
            path = Path(raw_path); need(path.is_file() and file_sha256(path) == digest, "score encoder dependency differs")
        need(Path(self.model_path).is_dir() and (Path(self.model_path) / "config.json").is_file(), "score encoder model snapshot unavailable")
        if self.initialization == "base":
            need(all(value is None for value in (self.aihub_completion_path, self.aihub_completion_sha256, self.aihub_artifact_path, self.aihub_artifact_sha256)), "base arm received AI-Hub dependencies")
        else:
            need(all(isinstance(value, str) and value for value in (self.aihub_completion_path, self.aihub_completion_sha256, self.aihub_artifact_path, self.aihub_artifact_sha256)), "AI-Hub arm has unresolved dependencies")
            completion_path = Path(str(self.aihub_completion_path)); artifact = Path(str(self.aihub_artifact_path))
            need(completion_path.is_file() and file_sha256(completion_path) == self.aihub_completion_sha256, "AI-Hub completion differs")
            need(artifact.is_dir(), "AI-Hub artifact unavailable")
            completion = json.loads(completion_path.read_text(encoding="utf-8")); state = completion.get("state")
            need(completion.get("status") == "completed" and completion.get("training_method") == "full_parameter", "AI-Hub completion status differs")
            need(completion.get("score_fields") == list(AXES) and completion.get("average_target_used") is False, "AI-Hub score axes differ")
            need(isinstance(state, dict) and Path(state.get("artifact_path", "")).resolve() == artifact.resolve() and state.get("artifact_sha256") == self.aihub_artifact_sha256, "AI-Hub artifact lineage differs")


def rationale_multimap(path: Path, expected: set[str], expected_records: int, *, single: bool = False) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in read_jsonl(path):
        source_id = str(row["source_id"]); value = row["rationales"]
        need(source_id in expected and isinstance(value, dict) and set(value) == set(AXES), "score encoder rationale linkage differs")
        need(all(isinstance(value[axis], str) and value[axis].strip() for axis in AXES), "score encoder rationale text differs")
        result.setdefault(source_id, []).append({axis: value[axis].strip() for axis in AXES})
    need(set(result) == expected, "score encoder rationale coverage differs")
    need(sum(map(len, result.values())) == expected_records, "score encoder rationale record count differs")
    need(not single or all(len(values) == 1 for values in result.values()), "score encoder single rationale view differs")
    return result


def load_views(config: Config, train: Sequence[ContinuousScoreRow], validation: Sequence[ContinuousScoreRow]) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]], dict[str, Any]]:
    handoff_path = Path(config.rationale_handoff_path)
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    need(handoff.get("schema_version") == "mal2026-rationale-pipeline-encoder-ratio-handoff-v2" and handoff.get("status") == "completed" and handoff.get("arm") == config.rationale_ratio, "score encoder rationale handoff differs")
    need(handoff.get("selection_dev_view") == handoff.get("validation_view") == "student_score_blind_single_only", "score encoder student-only evaluation view differs")
    need(handoff.get("teacher_use") == "train_only_label_aware_augmentation_never_validation_or_selection_dev", "score encoder teacher-use contract differs")
    paths, digests = handoff["paths"], handoff["sha256"]
    for key in ("teacher_train_all", "student_train_ratio", "student_train_single", "student_validation_single"):
        path = Path(paths[key]); need(path.is_file() and path.resolve().is_relative_to((ROOT / "data/processed/restricted").resolve()) and file_sha256(path) == digests[key], "score encoder rationale view differs")
    train_ids = {row.identifier for row in train}; validation_ids = {row.identifier for row in validation}
    records = handoff["records"]
    return (
        rationale_multimap(Path(paths["teacher_train_all"]), train_ids, int(records["teacher_train_all"])),
        rationale_multimap(Path(paths["student_train_ratio"]), train_ids, int(records["student_train_ratio"])),
        rationale_multimap(Path(paths["student_train_single"]), train_ids, int(records["student_train_single"]), single=True),
        rationale_multimap(Path(paths["student_validation_single"]), validation_ids, int(records["student_validation_single"]), single=True),
        handoff,
    )


def render(row: ContinuousScoreRow, rationale: Mapping[str, str]) -> str:
    return rationale_to_score_text(row.prompt, row.essay, rationale)


def exact_balance_weights(
    labels: Sequence[Sequence[float | int]], objective: str
) -> tuple[list[list[float]], dict[str, Any]]:
    """Equalize aggregate loss mass for every axis x integer-score cell.

    Whole-record oversampling cannot in general make all three score-axis
    marginals uniform because each essay carries three coupled labels.  Loss
    weights are axis-specific, so this is exactly equivalent to a uniform
    marginal training distribution for each axis without silently changing
    the other two labels or duplicating the already scarce source essays.
    """
    need(objective in OBJECTIVES and bool(labels), "score balance labels unavailable")
    bands = [
        [round_half_up_score(value) if objective == "bounded_regression" else int(value) + 1 for value in row]
        for row in labels
    ]
    need(all(len(row) == len(AXES) and all(1 <= value <= 5 for value in row) for row in bands), "score balance band differs")
    counts = [{score: 0 for score in range(1, 6)} for _ in AXES]
    for row in bands:
        for axis_index, value in enumerate(row): counts[axis_index][value] += 1
    need(all(all(count > 0 for count in axis.values()) for axis in counts), "score balance cell has no support")
    records = len(labels)
    weights = [
        [records / (5.0 * counts[axis_index][value]) for axis_index, value in enumerate(row)]
        for row in bands
    ]
    weighted_mass = [{score: 0.0 for score in range(1, 6)} for _ in AXES]
    for row, row_weights in zip(bands, weights, strict=True):
        for axis_index, (value, weight) in enumerate(zip(row, row_weights, strict=True)):
            weighted_mass[axis_index][value] += weight
    expected_mass = records / 5.0
    need(
        all(math.isclose(value, expected_mass, rel_tol=1e-10, abs_tol=1e-8) for axis in weighted_mass for value in axis.values()),
        "score balance weighted mass differs",
    )
    audit = {
        "mode": "per_axis_exact_inverse_frequency_loss",
        "records": records,
        "counts": {axis: {str(score): counts[index][score] for score in range(1, 6)} for index, axis in enumerate(AXES)},
        "weights": {
            axis: {str(score): records / (5.0 * counts[index][score]) for score in range(1, 6)}
            for index, axis in enumerate(AXES)
        },
        "weighted_mass": {
            axis: {str(score): weighted_mass[index][score] for score in range(1, 6)}
            for index, axis in enumerate(AXES)
        },
        "expected_mass_per_axis_score": expected_mass,
        "all_axis_score_cells_equal": True,
    }
    return weights, audit


def balanced_smoke_subset(rows: Sequence[ContinuousScoreRow]) -> list[ContinuousScoreRow]:
    """Small deterministic source subset covering every axis x score cell."""
    uncovered = {(axis_index, score) for axis_index in range(len(AXES)) for score in range(1, 6)}
    remaining = list(rows); chosen: list[ContinuousScoreRow] = []
    while uncovered:
        ranked = []
        for index, row in enumerate(remaining):
            cells = {(axis_index, round_half_up_score(value)) for axis_index, value in enumerate(row.labels)}
            ranked.append((len(cells & uncovered), -index, row, cells))
        gain, _, row, cells = max(ranked, key=lambda value: (value[0], value[1]))
        need(gain > 0, "balanced smoke cannot cover every axis-score cell")
        chosen.append(row); uncovered -= cells; remaining.remove(row)
    need(len(chosen) <= 15, "balanced smoke source coverage differs")
    return chosen


def dataset(
    rows: Sequence[ContinuousScoreRow],
    views: Sequence[Mapping[str, Sequence[Mapping[str, str]]]],
    tokenizer: Any,
    config: Config,
    *,
    training_balance: bool,
) -> Any:
    from datasets import Dataset
    texts: list[str] = []; labels: list[list[float] | list[int]] = []
    for row in rows:
        for view in views:
            for rationale in view[row.identifier]:
                texts.append(render(row, rationale))
                if config.objective == "bounded_regression":
                    labels.append(list(row.labels))
                else:
                    labels.append([round_half_up_score(value) - 1 for value in row.labels])
    columns: dict[str, Any] = {"text": texts, "labels": labels}
    if config.score_balance_mode != "none":
        if training_balance:
            columns["loss_weights"], _ = exact_balance_weights(labels, config.objective)
        else:
            # Selection-dev and canonical validation preserve their natural
            # distributions.  Unit weights make their loss well-defined while
            # all reported/selected metrics remain unweighted predictions.
            columns["loss_weights"] = [[1.0] * len(AXES) for _ in labels]
    result = Dataset.from_dict(columns)
    return result.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=config.max_length), batched=True, remove_columns=["text"])


def collator(tokenizer: Any, objective: str):
    def collect(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        labels = [feature.pop("labels") for feature in features]
        loss_weights = [feature.pop("loss_weights") for feature in features] if "loss_weights" in features[0] else None
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32 if objective == "bounded_regression" else torch.long)
        if loss_weights is not None: batch["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float32)
        return batch
    return collect


def dataset_balance_audit(data: Any, config: Config) -> dict[str, Any]:
    if config.score_balance_mode == "none": return {"mode": "none"}
    _, audit = exact_balance_weights(data["labels"], config.objective)
    need("loss_weights" in data.column_names, "balanced score encoder dataset lacks loss weights")
    return audit


def token_audit(rows: Sequence[ContinuousScoreRow], views: Sequence[Mapping[str, Sequence[Mapping[str, str]]]], tokenizer: Any, maximum: int) -> dict[str, Any]:
    lengths: list[int] = []
    texts = [render(row, rationale) for row in rows for view in views for rationale in view[row.identifier]]
    for start in range(0, len(texts), 64):
        encoded = tokenizer(texts[start:start + 64], truncation=False, add_special_tokens=True)
        lengths.extend(len(value) for value in encoded["input_ids"])
    need(lengths and max(lengths) <= maximum, "score encoder input would be truncated")
    ordered = sorted(lengths)
    return {"records": len(lengths), "maximum": ordered[-1], "p95": ordered[int(.95 * (len(ordered) - 1))], "p99": ordered[int(.99 * (len(ordered) - 1))], "limit": maximum, "truncated": 0}


def cached_token_audits(
    config: Config,
    handoff: Mapping[str, Any],
    train: Sequence[ContinuousScoreRow],
    validation: Sequence[ContinuousScoreRow],
    teacher_train: Mapping[str, Sequence[Mapping[str, str]]],
    student_train_ratio: Mapping[str, Sequence[Mapping[str, str]]],
    student_validation_single: Mapping[str, Sequence[Mapping[str, str]]],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    identity = {
        "rationale_handoff_sha256": config.rationale_handoff_sha256,
        "model_key": config.model_key,
        "model_revision": config.model_revision,
        "max_length": config.max_length,
        "score_prompt_sha256": routing()["rationale_to_score_encoder"]["source_file_sha256"],
    }
    key = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    TOKEN_AUDIT_PARENT.mkdir(parents=True, exist_ok=True)
    path = TOKEN_AUDIT_PARENT / f"{key}.json"; lock_path = TOKEN_AUDIT_PARENT / f"{key}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            need(value.get("status") == "completed" and value.get("identity") == identity, "score encoder cached token audit differs")
        else:
            train_audit = token_audit(train, (teacher_train, student_train_ratio), tokenizer, config.max_length)
            validation_audit = token_audit(validation, (student_validation_single,), tokenizer, config.max_length)
            need(train_audit["records"] == int(handoff["train_records_total"]) and validation_audit["records"] == 400, "score encoder token audit population differs")
            value = {"schema_version": "mal2026-rationale-pipeline-score-encoder-token-audit-v1", "status": "completed", "created_at": now(), "identity": identity, "train": train_audit, "validation": validation_audit, "average_used": False, "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_or_tokens"}
            atomic_json(path, value)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return value["train"], value["validation"], {"path": str(path.resolve()), "sha256": file_sha256(path), "key": key}


def load_aihub(model: Any, config: Config) -> dict[str, Any]:
    from safetensors import safe_open
    artifact = Path(str(config.aihub_artifact_path))
    targets = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    parameters = dict(model.named_parameters())
    required = {name for name in parameters if name.startswith("backbone.")}
    if config.objective == "bounded_regression":
        required |= {name for name in parameters if name.startswith("score_head.")}
    loaded: set[str] = set()
    for tensor_file in sorted(artifact.glob("*.safetensors")):
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name not in required:
                    continue
                need(name in targets and name not in loaded, f"AI-Hub score encoder tensor differs: {name}")
                tensor = handle.get_tensor(name)
                need(tuple(tensor.shape) == tuple(targets[name].shape), f"AI-Hub score encoder tensor shape differs: {name}")
                targets[name].data.copy_(tensor.to(dtype=targets[name].dtype)); loaded.add(name)
    need(required <= loaded, "AI-Hub score encoder artifact is incomplete")
    return {"mode": "aihub_full_backbone_matched_head" if config.objective == "bounded_regression" else "aihub_full_backbone_fresh_categorical_head", "loaded_tensors": len(loaded), "artifact_sha256": config.aihub_artifact_sha256}


def build_model(config: Config) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel
    spec = MODEL_SPECS[config.model_key]
    handoff = json.loads(Path(config.rationale_handoff_path).read_text(encoding="utf-8"))
    combined_counts = handoff.get("combined_axis_score_band_counts")
    need(isinstance(combined_counts, dict) and set(combined_counts) == set(AXES), "classification view-frequency counts unavailable")
    class_counts = [[int(combined_counts[axis][str(score)]) for score in range(1, 6)] for axis in AXES]
    need(all(all(count > 0 for count in axis) for axis in class_counts), "classification train class support differs")
    raw_weights = [[1.0 / math.sqrt(count) for count in axis] for axis in class_counts]
    class_weights = [[value / statistics.fmean(axis) for value in axis] for axis in raw_weights]
    kwargs: dict[str, Any] = {"revision": config.model_revision, "local_files_only": True, "trust_remote_code": False, "dtype": torch.bfloat16 if config.training_dtype == "bfloat16" else torch.float32, "low_cpu_mem_usage": True}
    if config.model_key == "kure_v1": kwargs["add_pooling_layer"] = False
    backbone = AutoModel.from_pretrained(config.model_path, **kwargs)
    if hasattr(backbone.config, "use_cache"): backbone.config.use_cache = False
    hidden = int(spec["hidden_size"]); need(getattr(backbone.config, "hidden_size", None) == hidden, "score encoder hidden size differs")
    selective_layers: list[int] = []
    if config.selective_gradient_checkpointing_stride is not None:
        from functools import partial
        from torch.utils.checkpoint import checkpoint
        layers = getattr(backbone, "layers", None)
        need(layers is not None and len(layers) == int(backbone.config.num_hidden_layers), "score encoder selective checkpointing layers unavailable")
        function = partial(checkpoint, use_reentrant=False)
        for index, layer in enumerate(layers):
            if index % config.selective_gradient_checkpointing_stride == 0:
                need(hasattr(layer, "gradient_checkpointing"), "score encoder layer lacks checkpointing support")
                layer.gradient_checkpointing = True; layer._gradient_checkpointing_func = function
                selective_layers.append(index)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__(); self.backbone = backbone
            self.score_head = nn.Linear(hidden, 3 if config.objective == "bounded_regression" else 15, dtype=next(backbone.parameters()).dtype)
            self.register_buffer("classification_weights", torch.tensor(class_weights, dtype=torch.float32), persistent=False)
        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Mapping[str, Any] | None = None) -> None:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        def gradient_checkpointing_disable(self) -> None: self.backbone.gradient_checkpointing_disable()
        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None, loss_weights: Any | None = None, **_: Any) -> Mapping[str, Any]:
            state = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
            if spec["pooling"] == "cls": pooled = state[:, 0]
            else:
                positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
                final = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
                need(bool((final >= 0).all().item()), "score encoder input has no token")
                pooled = state[torch.arange(state.shape[0], device=state.device), final]
            logits = self.score_head(functional.normalize(pooled, p=2, dim=-1).to(self.score_head.weight.dtype)).float()
            result: dict[str, Any] = {"logits": logits}
            if labels is not None:
                if config.objective == "bounded_regression":
                    predicted = 1.0 + 4.0 * torch.sigmoid(logits)
                    squared = functional.mse_loss(predicted, labels.float(), reduction="none")
                    if config.score_balance_mode == "none":
                        result["loss"] = squared.mean()
                    else:
                        need(loss_weights is not None and loss_weights.shape == squared.shape, "balanced regression weights differ")
                        result["loss"] = (squared * loss_weights.float()).mean()
                else:
                    shaped = logits.reshape(-1, 3, 5); target = labels.long()
                    if config.score_balance_mode == "none":
                        result["loss"] = sum(functional.cross_entropy(shaped[:, axis_index], target[:, axis_index], weight=self.classification_weights[axis_index]) for axis_index in range(3)) / 3.0
                    else:
                        need(loss_weights is not None and tuple(loss_weights.shape) == tuple(target.shape), "balanced classification weights differ")
                        result["loss"] = sum(
                            (functional.cross_entropy(shaped[:, axis_index], target[:, axis_index], reduction="none") * loss_weights[:, axis_index].float()).mean()
                            for axis_index in range(3)
                        ) / 3.0
            return result

    model = Model(); initialization = {"mode": "base_backbone_fresh_head", "loaded_tensors": 0}
    if config.initialization == "aihub": initialization = load_aihub(model, config)
    model.backbone = get_peft_model(model.backbone, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=list(spec["lora_targets"]), bias="none"))
    if hasattr(model.backbone, "enable_input_require_grads"): model.backbone.enable_input_require_grads()
    need(any("lora_" in name and parameter.requires_grad for name, parameter in model.named_parameters()), "score encoder LoRA absent")
    need(all(parameter.requires_grad for parameter in model.score_head.parameters()), "score encoder head frozen")
    return model, {**initialization, "classification_train_class_counts": class_counts, "classification_weights": class_weights, "classification_frequency_basis": "combined_openai_and_student_training_views", "selective_gradient_checkpointing_stride": config.selective_gradient_checkpointing_stride, "selective_gradient_checkpointing_layers": selective_layers}


def decode(logits: Any, objective: str) -> tuple[Any, Any]:
    import torch
    tensor = torch.as_tensor(logits).float()
    if objective == "bounded_regression":
        continuous = 1.0 + 4.0 * torch.sigmoid(tensor)
        # The public evaluator contract is Decimal ROUND_HALF_UP after
        # clipping, not banker rounding or a framework-specific cast.
        integer = torch.tensor(
            [[regression_evaluation_score(value) for value in row] for row in continuous.detach().cpu().tolist()],
            dtype=torch.long,
        )
    else:
        integer = tensor.reshape(-1, 3, 5).argmax(dim=-1).long() + 1
        continuous = integer.float()
    return continuous, integer


def ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index]); result = [0.0] * len(values); start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]: end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]: result[index] = rank
        start = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = ranks(left), ranks(right); mx, my = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def metrics(labels: Sequence[Sequence[float | int]], continuous: Sequence[Sequence[float]], integers: Sequence[Sequence[int]], objective: str) -> dict[str, Any]:
    gold = [[round_half_up_score(value) if objective == "bounded_regression" else int(value) + 1 for value in row] for row in labels]
    result: dict[str, Any] = {}
    squared: list[float] = []
    for index, axis in enumerate(AXES):
        truth = [float(row[index]) for row in gold]; pred = [float(row[index]) for row in integers]
        errors = [(a - b) ** 2 for a, b in zip(truth, pred, strict=True)]; squared.extend(errors)
        confusion = {str(value): {str(emitted): 0 for emitted in range(1, 6)} for value in range(1, 6)}
        for value, emitted in zip(truth, pred, strict=True): confusion[str(int(value))][str(int(emitted))] += 1
        result[axis] = {
            "integer_rmse": math.sqrt(statistics.fmean(errors)),
            "integer_spearman": spearman(truth, pred),
            "integer_accuracy": statistics.fmean(a == b for a, b in zip(truth, pred, strict=True)),
            "gold_support": {str(value): sum(confusion[str(value)].values()) for value in range(1, 6)},
            "per_gold_recall": {str(value): (confusion[str(value)][str(value)] / sum(confusion[str(value)].values()) if sum(confusion[str(value)].values()) else None) for value in range(1, 6)},
            "confusion_gold_by_prediction": confusion,
        }
    result["macro_integer_rmse"] = statistics.fmean(result[axis]["integer_rmse"] for axis in AXES)
    result["overall_integer_rmse"] = math.sqrt(statistics.fmean(squared))
    result["macro_integer_spearman"] = statistics.fmean(result[axis]["integer_spearman"] for axis in AXES)
    if objective == "bounded_regression":
        continuous_sq: list[float] = []
        for index, axis in enumerate(AXES):
            truth = [float(row[index]) for row in labels]; pred = [float(row[index]) for row in continuous]
            values = [(a - b) ** 2 for a, b in zip(truth, pred, strict=True)]; continuous_sq.extend(values)
            result[axis]["continuous_rmse_raw_decimal_gold"] = math.sqrt(statistics.fmean(values)); result[axis]["continuous_spearman_raw_decimal_gold"] = spearman(truth, pred)
        result["overall_continuous_rmse_raw_decimal_gold"] = math.sqrt(statistics.fmean(continuous_sq))
        result["macro_continuous_rmse_raw_decimal_gold"] = statistics.fmean(result[axis]["continuous_rmse_raw_decimal_gold"] for axis in AXES)
    return result


def prediction_metrics(trainer: Any, data: Any, objective: str) -> tuple[dict[str, Any], list[list[float]], list[list[int]], list[list[float | int]]]:
    prediction = trainer.predict(data); continuous, integer = decode(prediction.predictions, objective)
    values = metrics(prediction.label_ids.tolist(), continuous.tolist(), integer.tolist(), objective)
    need(math.isfinite(float(values["macro_integer_rmse"])), "score encoder metric non-finite")
    return values, continuous.tolist(), integer.tolist(), prediction.label_ids.tolist()


def trainable_state(model: Any) -> dict[str, Any]:
    state = {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters() if parameter.requires_grad}
    need(any(name.startswith("score_head.") for name in state) and any("lora_" in name for name in state), "score encoder trainable state differs")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    config = Config.load(args.config)
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
    physical_gpu = int(os.environ.get("MAL2026_PHYSICAL_GPU", "0"))
    setproctitle(f"mal2026:score-encoder:{config.rationale_ratio}:{config.model_key}:{config.objective}:{config.initialization}:gpu{physical_gpu}:rank{rank}"[:255])
    # Four independent single-GPU arms are launched concurrently.  This uses
    # the whole authorized scope without replicating one arm four times and
    # preserves its DDP4 effective batch via 4x accumulation below.
    need(world == 1 and physical_gpu in config.gpu_scope, "score encoder process topology differs")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu), "score encoder physical GPU binding differs")
    output = OUTPUT_PARENT / (("smoke-" if args.smoke else "") + config.run_id)
    restricted = RESTRICTED_PARENT / (("smoke-" if args.smoke else "") + config.run_id)
    if rank == 0:
        need(not output.exists() and not restricted.exists(), "score encoder output must be fresh")
        output.mkdir(parents=True); restricted.mkdir(parents=True, mode=0o700)
    else:
        deadline = time.monotonic() + 120
        while not output.is_dir() and time.monotonic() < deadline: time.sleep(.05)
        need(output.is_dir(), "score encoder output reservation timed out")

    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    set_seed(config.seed)
    all_train = load_continuous_rows(Path(config.train_path), config.train_sha256, 2000)
    validation = load_continuous_rows(Path(config.validation_path), config.validation_sha256, 400)
    teacher_train, student_train_ratio, student_train_single, student_validation_single, handoff = load_views(config, all_train, validation)
    select_train, select_dev, split_fingerprint = deterministic_split(all_train, config.seed)
    if args.smoke:
        select_train = balanced_smoke_subset(select_train) if config.score_balance_mode != "none" else select_train[:4]
        select_dev = select_dev[:4]

    def initialize() -> tuple[Any, Any, dict[str, Any]]:
        set_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
        if tokenizer.pad_token is None:
            need(tokenizer.eos_token is not None, "score encoder tokenizer has no pad token"); tokenizer.pad_token = tokenizer.eos_token
        model, lineage = build_model(config); return tokenizer, model, lineage

    tokenizer, model, initialization = initialize()
    # Audit full declared populations even for the one-update smoke.
    audit_train, audit_validation, audit_lineage = cached_token_audits(config, handoff, all_train, validation, teacher_train, student_train_ratio, student_validation_single, tokenizer)

    def compute(result: Any) -> dict[str, float]:
        continuous, integer = decode(result.predictions, config.objective)
        values = metrics(result.label_ids.tolist(), continuous.tolist(), integer.tolist(), config.objective)
        return {key: float(values[key]) for key in ("macro_integer_rmse", "overall_integer_rmse", "macro_integer_spearman")}

    target_effective_batch = 32 if config.model_key == "qwen3_embedding_8b" else 64
    need(target_effective_batch % config.per_device_train_batch_size == 0, "score encoder effective batch is not integral")
    effective_accumulation = target_effective_batch // config.per_device_train_batch_size

    # Fast protocol: use a predeclared epoch count and train on every source
    # exactly once.  It deliberately performs neither internal-dev epoch
    # selection nor a second refit.  Canonical validation is read once only
    # after the fixed full-data training has completed.
    if config.training_protocol == "fixed_full_train" and not args.smoke:
        fixed_data = dataset(all_train, (teacher_train, student_train_ratio), tokenizer, config, training_balance=True)
        fixed_balance_audit = dataset_balance_audit(fixed_data, config)
        fixed_events: list[dict[str, Any]] = []

        class FixedCapture(TrainerCallback):
            def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **_: Any) -> Any:
                for key, value in (logs or {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        need(math.isfinite(float(value)), f"non-finite fixed-full score encoder log: {key}")
                return control

            def on_epoch_end(self, args: Any, state: Any, control: Any, model: Any | None = None, **_: Any) -> Any:
                need(model is not None, "fixed-full epoch model unavailable")
                epoch = int(round(float(state.epoch or 0)))
                need(1 <= epoch <= int(config.fixed_epochs or 0), "fixed-full epoch callback differs")
                if state.is_world_process_zero:
                    path = output / "fixed" / f"epoch-{epoch:02d}.safetensors"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(trainable_state(model), str(path))
                    fixed_events.append({"epoch": epoch, "global_step": int(state.global_step), "state_path": str(path.resolve()), "state_sha256": file_sha256(path)})
                return control

        fixed_args = TrainingArguments(
            output_dir=str(output / "fixed/trainer"), do_train=True, do_eval=False, eval_strategy="no", save_strategy="no",
            num_train_epochs=float(config.fixed_epochs or 0), learning_rate=config.learning_rate, weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio, optim="adamw_torch_fused", per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=effective_accumulation, bf16=config.training_dtype == "bfloat16", tf32=True,
            gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
            report_to=[], remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=True,
            ddp_find_unused_parameters=False, logging_steps=10, seed=config.seed, data_seed=config.seed,
        )
        fixed_trainer = Trainer(
            model=model, args=fixed_args, train_dataset=fixed_data, data_collator=collator(tokenizer, config.objective),
            callbacks=[FixedCapture()],
        )
        torch.cuda.reset_peak_memory_stats()
        fixed_trained = fixed_trainer.train(); fixed_trainer.accelerator.wait_for_everyone()
        fixed_peak_memory_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
        need([event["epoch"] for event in fixed_events] == list(range(1, int(config.fixed_epochs or 0) + 1)), "fixed-full checkpoints differ")
        validation_data = dataset(validation, (student_validation_single,), tokenizer, config, training_balance=False)
        validation_metrics, continuous, integers, _ = prediction_metrics(fixed_trainer, validation_data, config.objective)
        state_path = output / "fixed_full_trainable.safetensors"
        prediction_path = restricted / "validation_predictions.jsonl"
        if fixed_trainer.is_world_process_zero():
            save_file(trainable_state(model), str(state_path))
            with prediction_path.open("x", encoding="utf-8") as handle:
                for row, raw, integer in zip(validation, continuous, integers, strict=True):
                    handle.write(json.dumps({"source_id": row.identifier, "continuous_prediction": {axis: float(raw[index]) for index, axis in enumerate(AXES)}, "emitted_integer_prediction": {axis: int(integer[index]) for index, axis in enumerate(AXES)}}, ensure_ascii=False, separators=(",", ":")) + "\n")
            os.chmod(prediction_path, 0o600)
            result = {
                "schema_version": "mal2026-rationale-pipeline-score-encoder-result-v4", "status": "completed", "completed_at": now(),
                "run_id": config.run_id, "rationale_ratio": config.rationale_ratio, "openai_to_student_ratio": handoff["openai_to_student_ratio"],
                "model_key": config.model_key, "model_id": config.model_id, "model_revision": config.model_revision,
                "objective": config.objective, "initialization": config.initialization, "authorized_gpu_scope": list(config.gpu_scope),
                "physical_gpu": physical_gpu, "world_size": world, "effective_batch_size": config.per_device_train_batch_size * effective_accumulation,
                "training_protocol": config.training_protocol, "fixed_epochs": config.fixed_epochs, "fixed_epoch_source": config.fixed_epoch_source,
                "score_fields": list(AXES), "average_read": False, "average_target_used": False,
                "score_balance_mode": config.score_balance_mode,
                "score_balance_contract": "every axis x rounded_integer_score cell has equal aggregate training-loss mass" if config.score_balance_mode != "none" else "natural_tail_enriched_view_frequency",
                "regression_train_target": "raw_per_axis_decimal" if config.objective == "bounded_regression" else None,
                "classification_train_target": "per_axis_Decimal_ROUND_HALF_UP_integer_1_to_5" if config.objective == "categorical_5class" else None,
                "evaluation_projection": "clip_[1,5]_then_Decimal_ROUND_HALF_UP" if config.objective == "bounded_regression" else "argmax_class_plus_1",
                "evaluation_gold": "per_axis_Decimal_ROUND_HALF_UP_integer_1_to_5",
                "selection": {"mode": "predeclared_fixed_epoch_no_selection", "source": config.fixed_epoch_source, "selected": {"epoch": config.fixed_epochs}, "events": []},
                "refit": {"mode": "not_applicable_single_full_data_pass"},
                "fixed_train": {
                    "records": int(handoff["train_records_total"]), "source_records": 2000,
                    "openai_records": int(handoff["records"]["teacher_train_all"]), "student_records": int(handoff["records"]["student_train_ratio"]),
                    "epochs": int(config.fixed_epochs or 0), "global_step": int(fixed_trainer.state.global_step), "checkpoints": fixed_events,
                    "balance_audit": fixed_balance_audit, "peak_memory_mib": fixed_peak_memory_mib,
                    "train_metrics": {key: float(value) for key, value in fixed_trained.metrics.items() if isinstance(value, (int, float))},
                },
                "canonical_validation": {"records": 400, "view": "student_score_blind_single_only", "use": "single_final_descriptive_evaluation_not_selection", "metrics": validation_metrics},
                "rationale_handoff_sha256": config.rationale_handoff_sha256, "teacher_use": handoff["teacher_use"],
                "token_audit": {"train_all_declared_views": audit_train, "validation_student_single_view": audit_validation, "cache": audit_lineage},
                "initialization_lineage": initialization, "state_path": str(state_path.resolve()), "state_sha256": file_sha256(state_path),
                "validation_predictions_path": str(prediction_path.resolve()), "validation_predictions_sha256": file_sha256(prediction_path),
                "config_sha256": file_sha256(args.config), "rationale_to_score_prompt_sha256": routing()["rationale_to_score_encoder"]["source_file_sha256"],
                "privacy": "aggregate_result_only_row_predictions_restricted_no_text_or_scores_in_result",
            }
            atomic_json(output / "result.json", result)
            print(json.dumps({"status": "completed", "run_id": config.run_id, "macro_integer_rmse": validation_metrics["macro_integer_rmse"], "macro_integer_spearman": validation_metrics["macro_integer_spearman"]}, sort_keys=True), flush=True)
        fixed_trainer.accelerator.wait_for_everyone()
        return

    train_data = dataset(select_train, (teacher_train, student_train_ratio), tokenizer, config, training_balance=True)
    dev_data = dataset(select_dev, (student_train_single,), tokenizer, config, training_balance=False)
    selection_balance_audit = dataset_balance_audit(train_data, config)
    events: list[dict[str, Any]] = []

    class Capture(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **_: Any) -> Any:
            for key, value in (logs or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool): need(math.isfinite(float(value)), f"non-finite score encoder log: {key}")
            return control
        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Mapping[str, Any] | None = None, model: Any | None = None, **_: Any) -> Any:
            need(metrics is not None and model is not None, "score encoder selection event differs")
            event = {"epoch": int(round(float(state.epoch or 0))), "global_step": int(state.global_step), "macro_integer_rmse": float(metrics["eval_macro_integer_rmse"]), "overall_integer_rmse": float(metrics["eval_overall_integer_rmse"]), "macro_integer_spearman": float(metrics["eval_macro_integer_spearman"])}
            if state.is_world_process_zero:
                path = output / "selection" / f"epoch-{event['epoch']:02d}.safetensors"; path.parent.mkdir(parents=True, exist_ok=True); save_file(trainable_state(model), str(path)); event.update({"state_path": str(path.resolve()), "state_sha256": file_sha256(path)})
            events.append(event); return control

    training_args = TrainingArguments(output_dir=str(output / "selection/trainer"), do_train=True, do_eval=True, eval_strategy="epoch", save_strategy="no", num_train_epochs=1 if args.smoke else len(config.epochs), max_steps=1 if args.smoke else -1, learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio, optim="adamw_torch_fused", per_device_train_batch_size=config.per_device_train_batch_size, per_device_eval_batch_size=config.per_device_eval_batch_size, gradient_accumulation_steps=1 if args.smoke else effective_accumulation, bf16=config.training_dtype == "bfloat16", tf32=True, gradient_checkpointing=config.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None, report_to=[], remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=True, ddp_find_unused_parameters=False, logging_steps=1 if args.smoke else 10, seed=config.seed, data_seed=config.seed)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=dev_data, data_collator=collator(tokenizer, config.objective), compute_metrics=compute, callbacks=[Capture()])
    torch.cuda.reset_peak_memory_stats()
    trained = trainer.train(); trainer.accelerator.wait_for_everyone()
    selection_peak_memory_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    shared: list[Any] = [events if trainer.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized(): torch.distributed.broadcast_object_list(shared, src=0)
    events = shared[0]; need(isinstance(events, list) and events, "score encoder selection events absent")
    selected = min(events, key=lambda row: (row["macro_integer_rmse"], -row["macro_integer_spearman"], row["overall_integer_rmse"], row["epoch"]))
    if args.smoke:
        if trainer.is_world_process_zero(): atomic_json(output / "smoke_complete.json", {"schema_version": "mal2026-rationale-pipeline-score-encoder-smoke-v1", "status": "completed", "run_id": config.run_id, "rationale_ratio": config.rationale_ratio, "model_key": config.model_key, "objective": config.objective, "initialization": config.initialization, "physical_gpu": physical_gpu, "events": events, "score_balance_mode": config.score_balance_mode, "training_protocol": config.training_protocol, "fixed_epochs": config.fixed_epochs, "fixed_epoch_source": config.fixed_epoch_source, "selection_balance_audit": selection_balance_audit, "selection_peak_memory_mib": selection_peak_memory_mib, "token_audit": {"train": audit_train, "validation": audit_validation, "cache": audit_lineage}, "average_used": False})
        print(json.dumps({"status": "completed", "mode": "smoke", "run_id": config.run_id}, sort_keys=True), flush=True); return

    del trainer, model, tokenizer, train_data, dev_data; torch.cuda.empty_cache()
    tokenizer, model, refit_initialization = initialize(); need(refit_initialization == initialization, "score encoder refit initialization differs")
    refit_data = dataset(all_train, (teacher_train, student_train_ratio), tokenizer, config, training_balance=True)
    refit_balance_audit = dataset_balance_audit(refit_data, config)
    refit_args = TrainingArguments(output_dir=str(output / "refit/trainer"), do_train=True, do_eval=False, eval_strategy="no", save_strategy="no", num_train_epochs=float(selected["epoch"]), learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio, optim="adamw_torch_fused", per_device_train_batch_size=config.per_device_train_batch_size, gradient_accumulation_steps=effective_accumulation, bf16=config.training_dtype == "bfloat16", tf32=True, gradient_checkpointing=config.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None, report_to=[], remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=True, ddp_find_unused_parameters=False, logging_steps=10, seed=config.seed, data_seed=config.seed)
    refitter = Trainer(model=model, args=refit_args, train_dataset=refit_data, data_collator=collator(tokenizer, config.objective), callbacks=[Capture()]); torch.cuda.reset_peak_memory_stats(); refit_trained = refitter.train(); refitter.accelerator.wait_for_everyone(); refit_peak_memory_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    validation_data = dataset(validation, (student_validation_single,), tokenizer, config, training_balance=False)
    validation_metrics, continuous, integers, _ = prediction_metrics(refitter, validation_data, config.objective)
    state_path = output / "selected_refit_trainable.safetensors"
    prediction_path = restricted / "validation_predictions.jsonl"
    if refitter.is_world_process_zero():
        save_file(trainable_state(model), str(state_path))
        with prediction_path.open("x", encoding="utf-8") as handle:
            for row, raw, integer in zip(validation, continuous, integers, strict=True):
                handle.write(json.dumps({"source_id": row.identifier, "continuous_prediction": {axis: float(raw[index]) for index, axis in enumerate(AXES)}, "emitted_integer_prediction": {axis: int(integer[index]) for index, axis in enumerate(AXES)}}, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.chmod(prediction_path, 0o600)
        result = {"schema_version": "mal2026-rationale-pipeline-score-encoder-result-v3", "status": "completed", "completed_at": now(), "run_id": config.run_id, "rationale_ratio": config.rationale_ratio, "openai_to_student_ratio": handoff["openai_to_student_ratio"], "model_key": config.model_key, "model_id": config.model_id, "model_revision": config.model_revision, "objective": config.objective, "initialization": config.initialization, "authorized_gpu_scope": list(config.gpu_scope), "physical_gpu": physical_gpu, "world_size": world, "effective_batch_size": config.per_device_train_batch_size * effective_accumulation, "training_protocol": config.training_protocol, "fixed_epochs": config.fixed_epochs, "fixed_epoch_source": config.fixed_epoch_source, "score_fields": list(AXES), "average_read": False, "average_target_used": False, "score_balance_mode": config.score_balance_mode, "score_balance_contract": "every axis x rounded_integer_score cell has equal aggregate training-loss mass" if config.score_balance_mode != "none" else "natural_tail_enriched_view_frequency", "regression_train_target": "raw_per_axis_decimal" if config.objective == "bounded_regression" else None, "classification_train_target": "per_axis_Decimal_ROUND_HALF_UP_integer_1_to_5" if config.objective == "categorical_5class" else None, "evaluation_projection": "clip_[1,5]_then_Decimal_ROUND_HALF_UP" if config.objective == "bounded_regression" else "argmax_class_plus_1", "evaluation_gold": "per_axis_Decimal_ROUND_HALF_UP_integer_1_to_5", "selection": {"source": "train_internal_source_disjoint_1600_400", "train_views": "all_openai_plus_ratio_student_for_selection_train_sources", "dev_view": "student_score_blind_single_only", "split_fingerprint": split_fingerprint, "events": events, "selected": selected, "balance_audit": selection_balance_audit, "peak_memory_mib": selection_peak_memory_mib, "rule": "lowest macro integer RMSE, then highest integer Spearman, then overall integer RMSE, then earlier epoch", "train_metrics": {key: float(value) for key, value in trained.metrics.items() if isinstance(value, (int, float))}}, "refit": {"records": int(handoff["train_records_total"]), "source_records": 2000, "openai_records": int(handoff["records"]["teacher_train_all"]), "student_records": int(handoff["records"]["student_train_ratio"]), "epochs": int(selected["epoch"]), "global_step": int(refitter.state.global_step), "balance_audit": refit_balance_audit, "peak_memory_mib": refit_peak_memory_mib, "train_metrics": {key: float(value) for key, value in refit_trained.metrics.items() if isinstance(value, (int, float))}}, "canonical_validation": {"records": 400, "view": "student_score_blind_single_only", "use": "single_final_descriptive_evaluation_not_selection", "metrics": validation_metrics}, "rationale_handoff_sha256": config.rationale_handoff_sha256, "teacher_use": handoff["teacher_use"], "token_audit": {"train_all_declared_views": audit_train, "validation_student_single_view": audit_validation, "cache": audit_lineage}, "initialization_lineage": initialization, "state_path": str(state_path.resolve()), "state_sha256": file_sha256(state_path), "validation_predictions_path": str(prediction_path.resolve()), "validation_predictions_sha256": file_sha256(prediction_path), "config_sha256": file_sha256(args.config), "rationale_to_score_prompt_sha256": routing()["rationale_to_score_encoder"]["source_file_sha256"], "privacy": "aggregate_result_only_row_predictions_restricted_no_text_or_scores_in_result"}
        atomic_json(output / "result.json", result)
        print(json.dumps({"status": "completed", "run_id": config.run_id, "macro_integer_rmse": validation_metrics["macro_integer_rmse"], "macro_integer_spearman": validation_metrics["macro_integer_spearman"]}, sort_keys=True), flush=True)
    refitter.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
