"""Qwen3-Embedding score regression over one fixed RLAIF rationale source.

Two prespecified initialization arms share the same three-axis data and
optimization contract:

* the immutable public Qwen3-Embedding-8B snapshot; and
* the same snapshot after the completed AI-Hub 48,016-row standard-encoder
  refit, with its first three score heads retained and its ``average`` head
  deliberately discarded.

Restricted essays and generated rationales remain in memory.  Only aggregate
metrics, provenance, and trainable LoRA/head states are persisted beneath
ignored output roots.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .api_rationale_data import EXPECTED_ESSAYS, SOURCE_SHA256, load_writing_rows
from .rlaif_top3_encoder import (
    AXES,
    _input_text,
    _labels,
    _load_generated_rationales,
    generation_dir,
    three_axis_metrics,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3-Embedding-8B"
MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
MODEL_PATH = ROOT / "outputs" / "model-cache" / f"Qwen--Qwen3-Embedding-8B-{MODEL_REVISION}"
RATIONALE_SOURCE = "rank2_ax4_random1"
WARMSTART_METADATA = ROOT / "outputs" / "standard-encoder-runs" / "matrix-4gpu-20260717-cont5-encoder-qwen3-refit" / "standard_encoder_training_complete.json"
WARMSTART_STATE = WARMSTART_METADATA.parent / "final_model" / "model.safetensors"
EXPECTED_WARMSTART_SHA256 = "756cfa5d627adb3d2bd9d22b1f9d9df1af801a59ee9cceba07fa5ea9957e6bef"
EXPECTED_PREPARED_MANIFEST = ROOT / "data" / "manifests" / "aihub_human_feedback_v1.json"
TRAIN_ROOT = ROOT / "outputs" / "rlaif-qwen3-embedding-score-regression-v1"
EVAL_ROOT = ROOT / "outputs" / "rlaif-qwen3-embedding-score-regression-evals-v1"
ARMS = ("qwen3_base", "qwen3_aihub_warmstart")
STANDARD_FIELDS = (*AXES, "average")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class Qwen3EmbeddingExperimentError(ValueError):
    """Raised when the fixed two-arm experiment contract differs."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Qwen3EmbeddingExperimentError(message)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Qwen3EmbeddingExperimentError(f"{label} is unreadable") from exc
    _need(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _need(not path.exists() and not temporary.exists(), f"refusing to replace {path}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def warmstart_provenance() -> dict[str, Any]:
    metadata = _read_json(WARMSTART_METADATA, "AI-Hub Qwen3 warm-start metadata")
    config = metadata.get("config")
    _need(metadata.get("status") == "completed" and isinstance(config, dict), "AI-Hub Qwen3 warm-start is incomplete")
    _need(config.get("backbone") == "qwen3_embedding" and config.get("model_id") == MODEL_ID, "AI-Hub warm-start backbone differs")
    _need(config.get("model_revision") == MODEL_REVISION and Path(config.get("model_path", "")).resolve() == MODEL_PATH.resolve(), "AI-Hub warm-start snapshot differs")
    _need(Path(config.get("prepared_manifest", "")).resolve() == EXPECTED_PREPARED_MANIFEST.resolve(), "AI-Hub warm-start manifest differs")
    _need(WARMSTART_STATE.is_file() and metadata.get("model_state_sha256") == EXPECTED_WARMSTART_SHA256, "AI-Hub warm-start state binding differs")
    manifest = _read_json(EXPECTED_PREPARED_MANIFEST, "AI-Hub aggregate manifest")
    _need(manifest.get("source", {}).get("source_records") == 48016 and manifest.get("eligibility", {}).get("eligible_records") == 48016, "AI-Hub warm-start record count differs")
    _need(manifest.get("score_contract", {}).get("fields") == list(STANDARD_FIELDS), "AI-Hub score order differs")
    return {
        "training_run_id": metadata.get("run_id"),
        "training_metadata_sha256": _sha(WARMSTART_METADATA),
        "model_state_sha256": EXPECTED_WARMSTART_SHA256,
        "prepared_manifest_sha256": _sha(EXPECTED_PREPARED_MANIFEST),
        "source_records": 48016,
        "loaded_score_fields": list(STANDARD_FIELDS),
        "continued_score_fields": list(AXES),
        "average_head_discarded_before_continuation": True,
    }


def training_dir(arm: str, phase: str) -> Path:
    return TRAIN_ROOT / f"rlaif-qwen3-embedding-v1-{arm}-{phase}-001"


def evaluation_dir(arm: str) -> Path:
    return EVAL_ROOT / f"rlaif-qwen3-embedding-eval-v1-{arm}-validation-001"


@dataclass(frozen=True)
class Qwen3EmbeddingTrainConfig:
    schema_version: str
    run_id: str
    arm: str
    phase: str
    source_key: str
    model_id: str
    model_revision: str
    model_path: str
    initialization: str
    warmstart_metadata_path: str | None
    output_dir: str
    score_fields: tuple[str, str, str]
    seed: int
    max_length: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    num_train_epochs: float
    max_steps: int
    train_record_limit: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    logging_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    training_dtype: str

    @classmethod
    def from_json(cls, path: Path) -> "Qwen3EmbeddingTrainConfig":
        raw = _read_json(path, "Qwen3 train config")
        _need(isinstance(raw.get("score_fields"), list), "Qwen3 score fields must be a list")
        raw["score_fields"] = tuple(raw["score_fields"])
        _need(set(raw) == set(cls.__dataclass_fields__), "Qwen3 train config has unknown or missing fields")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self, *, require_fresh_output: bool = True) -> None:
        _need(self.schema_version == "mal2026-rlaif-qwen3-embedding-train-v1", "Qwen3 train schema differs")
        _need(self.arm in ARMS and self.phase in {"gpu0_preflight", "full"}, "Qwen3 arm/phase differs")
        expected_initialization = "public_base" if self.arm == "qwen3_base" else "aihub_48016_warmstart"
        _need(self.initialization == expected_initialization, "Qwen3 initialization differs")
        expected_warm = None if self.arm == "qwen3_base" else str(WARMSTART_METADATA.resolve())
        _need(self.warmstart_metadata_path == expected_warm, "Qwen3 warm-start metadata path differs")
        if self.arm != "qwen3_base":
            warmstart_provenance()
        _need(self.source_key == RATIONALE_SOURCE and self.score_fields == AXES, "Qwen3 data source or three-axis targets differ")
        _need((self.model_id, self.model_revision, Path(self.model_path).resolve()) == (MODEL_ID, MODEL_REVISION, MODEL_PATH.resolve()), "Qwen3 snapshot differs")
        _need(MODEL_PATH.is_dir() and not MODEL_PATH.is_symlink(), "Qwen3 snapshot is unavailable")
        output = Path(self.output_dir)
        _need(output.is_absolute() and output.parent == TRAIN_ROOT.resolve(), "Qwen3 training output root differs")
        _need(output.name == self.run_id == f"rlaif-qwen3-embedding-v1-{self.arm}-{self.phase}-001", "Qwen3 run identity differs")
        _need((not output.exists()) if require_fresh_output else output.is_dir(), "Qwen3 training output freshness differs")
        _need((self.seed, self.max_length, self.learning_rate, self.weight_decay, self.warmup_ratio) == (2026072601, 2048, 1e-4, 0.01, 0.05), "Qwen3 optimization constants differ")
        _need((self.lora_r, self.lora_alpha, self.lora_dropout, self.training_dtype) == (16, 32, 0.05, "bfloat16"), "Qwen3 LoRA/numeric contract differs")
        if self.phase == "full":
            _need((self.num_train_epochs, self.max_steps, self.train_record_limit, self.per_device_train_batch_size, self.gradient_accumulation_steps) == (12.0, -1, 2000, 4, 4), "Qwen3 full schedule differs")
        else:
            _need((self.num_train_epochs, self.max_steps, self.train_record_limit, self.per_device_train_batch_size, self.gradient_accumulation_steps) == (1.0, 1, 4, 4, 1), "Qwen3 preflight schedule differs")


def training_config(arm: str, phase: str) -> dict[str, Any]:
    _need(arm in ARMS and phase in {"gpu0_preflight", "full"}, "unknown Qwen3 arm/phase")
    full = phase == "full"
    run_id = f"rlaif-qwen3-embedding-v1-{arm}-{phase}-001"
    return {
        "schema_version": "mal2026-rlaif-qwen3-embedding-train-v1",
        "run_id": run_id,
        "arm": arm,
        "phase": phase,
        "source_key": RATIONALE_SOURCE,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(MODEL_PATH.resolve()),
        "initialization": "public_base" if arm == "qwen3_base" else "aihub_48016_warmstart",
        "warmstart_metadata_path": None if arm == "qwen3_base" else str(WARMSTART_METADATA.resolve()),
        "output_dir": str(training_dir(arm, phase).resolve()),
        "score_fields": list(AXES),
        "seed": 2026072601,
        "max_length": 2048,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "num_train_epochs": 12.0 if full else 1.0,
        "max_steps": -1 if full else 1,
        "train_record_limit": 2000 if full else 4,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4 if full else 1,
        "logging_steps": 5 if full else 1,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "training_dtype": "bfloat16",
    }


def _examples(source: str, limit: int) -> list[dict[str, Any]]:
    generated = _load_generated_rationales(generation_dir(RATIONALE_SOURCE, source, "full"), RATIONALE_SOURCE, source, "full", EXPECTED_ESSAYS[source])
    rows = load_writing_rows(source, include_scores=True)[:limit]
    result = [{"source_id": row.identifier, "text": _input_text(row.prompt, row.essay, generated[row.identifier]), "labels": _labels(row)} for row in rows]
    _need(len(result) == limit, "Qwen3 example count differs")
    return result


def _tokenized(examples: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int, *, include_source: bool) -> Any:
    from datasets import Dataset

    payload: dict[str, Any] = {"text": [item["text"] for item in examples], "labels": [item["labels"] for item in examples]}
    if include_source:
        payload["source_id"] = [item["source_id"] for item in examples]
    dataset = Dataset.from_dict(payload)
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])


def _collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        labels = [feature.pop("labels") for feature in features]
        for feature in features:
            feature.pop("source_id", None)
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch

    return collate


def _fresh_regressor(config: Qwen3EmbeddingTrainConfig, fields: Sequence[str]) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel

    field_tuple = tuple(fields)
    _need(field_tuple in {AXES, STANDARD_FIELDS}, "Qwen3 regressor fields differ")
    base = AutoModel.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    base.config.use_cache = False
    leaves = {name.rsplit(".", maxsplit=1)[-1] for name, _ in base.named_modules()}
    _need(set(LORA_TARGETS) <= leaves, "Qwen3 snapshot lacks reviewed LoRA targets")
    backbone = get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=list(LORA_TARGETS), bias="none"))
    hidden = getattr(backbone.config, "hidden_size", None)
    _need(type(hidden) is int and hidden > 0, "Qwen3 embedding hidden size is unavailable")

    class Regressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.regression_head = nn.Linear(hidden, len(field_tuple))

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None, **_: Any) -> Mapping[str, Any]:
            hidden_state = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
            positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
            index = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
            _need(bool((index >= 0).all().item()), "Qwen3 examples require a nonpad token")
            embedding = hidden_state[torch.arange(hidden_state.shape[0], device=hidden_state.device), index]
            logits = self.regression_head(functional.normalize(embedding, p=2, dim=-1).float())
            result: dict[str, Any] = {"logits": logits}
            if labels is not None:
                _need(tuple(labels.shape[-1:]) == (len(field_tuple),), "Qwen3 label dimension differs")
                result["loss"] = functional.mse_loss(logits, labels.float(), reduction="mean")
            return result

    return Regressor()


def build_model(config: Qwen3EmbeddingTrainConfig) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn
    from safetensors import safe_open

    if config.arm == "qwen3_base":
        return _fresh_regressor(config, AXES), {"initialization": "public_base", "average_head_loaded": False}
    provenance = warmstart_provenance()
    model = _fresh_regressor(config, STANDARD_FIELDS)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    with safe_open(WARMSTART_STATE, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        _need(trainable_names <= available, "AI-Hub Qwen3 warm-start lacks a trainable tensor")
        warm_state = {name: handle.get_tensor(name) for name in sorted(trainable_names)}
    incompatible = model.load_state_dict(warm_state, strict=False)
    _need(not incompatible.unexpected_keys and not (trainable_names & set(incompatible.missing_keys)), "AI-Hub Qwen3 warm-start trainable state differs")
    previous = model.regression_head
    replacement = nn.Linear(previous.in_features, len(AXES), dtype=previous.weight.dtype, device=previous.weight.device)
    with torch.no_grad():
        replacement.weight.copy_(previous.weight[: len(AXES)])
        replacement.bias.copy_(previous.bias[: len(AXES)])
    model.regression_head = replacement
    return model, {"initialization": "aihub_48016_warmstart", "average_head_loaded": True, **provenance}


def _trainable_state(model: Any) -> dict[str, Any]:
    state = {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters() if parameter.requires_grad}
    _need(bool(state) and "regression_head.weight" in state and "regression_head.bias" in state, "Qwen3 trainable state is incomplete")
    _need(all("average" not in name for name in state), "Qwen3 state contains an average target")
    return state


def run_training(config: Qwen3EmbeddingTrainConfig) -> dict[str, Any]:
    config.validate()
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Qwen3 embedding training requires .venv-standard") from exc
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        _need(tokenizer.eos_token is not None, "Qwen3 tokenizer lacks pad/EOS")
        tokenizer.pad_token = tokenizer.eos_token
    examples = _examples("train", config.train_record_limit)
    dataset = _tokenized(examples, tokenizer, config.max_length, include_source=False)
    model, initialization = build_model(config)
    arguments = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_id,
        do_train=True,
        do_eval=False,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=True,
        tf32=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        optim="adamw_torch",
        logging_nan_inf_filter=False,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=dataset, data_collator=_collator(tokenizer))
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None
    failed = False
    if trainer.is_world_process_zero():
        try:
            metrics = {str(key): float(value) for key, value in trained.metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
            _need("train_loss" in metrics and all(math.isfinite(value) for value in metrics.values()), "Qwen3 Trainer metrics are non-finite")
            _need(all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters() if parameter.requires_grad), "Qwen3 trainable parameter is non-finite")
            output = Path(config.output_dir)
            _need(output.is_dir() and not (output / "training_complete.json").exists(), "Qwen3 Trainer output root differs")
            state_path = output / "trainable_model.safetensors"
            _need(not state_path.exists(), "Qwen3 trainable state already exists")
            state = _trainable_state(model)
            save_file(state, str(state_path))
            payload = {
                "status": "completed",
                "run_id": config.run_id,
                "arm": config.arm,
                "phase": config.phase,
                "source_key": config.source_key,
                "backbone": "qwen3_embedding",
                "model_id": config.model_id,
                "model_revision": config.model_revision,
                "initialization": initialization,
                "score_fields": list(AXES),
                "average_target_used": False,
                "train_records": len(examples),
                "global_step": int(trainer.state.global_step),
                "train_metrics": metrics,
                "trainable_tensor_count": len(state),
                "trainable_state_sha256": _sha(state_path),
                "input_provenance": {
                    "canonical_source_sha256": dict(SOURCE_SHA256),
                    "rationale_generation_sha256": _sha(generation_dir(RATIONALE_SOURCE, "train", "full") / "generated_rationales.jsonl"),
                },
                "config": asdict(config),
                "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
            }
            _atomic_json(output / "training_complete.json", payload)
        except Exception:
            failed = True
    state_payload: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state_payload, src=0)
    if state_payload[0]:
        raise Qwen3EmbeddingExperimentError("rank-zero Qwen3 persistence/health gate failed")
    _need(isinstance(state_payload[1], dict) and state_payload[1].get("status") == "completed", "Qwen3 completion was not published")
    trainer.accelerator.wait_for_everyone()
    return state_payload[1]


@dataclass(frozen=True)
class Qwen3EmbeddingEvalConfig:
    schema_version: str
    run_id: str
    arm: str
    training_metadata_path: str
    output_dir: str
    per_device_eval_batch_size: int

    @classmethod
    def from_json(cls, path: Path) -> "Qwen3EmbeddingEvalConfig":
        raw = _read_json(path, "Qwen3 evaluation config")
        _need(set(raw) == set(cls.__dataclass_fields__), "Qwen3 evaluation config has unknown or missing fields")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self) -> None:
        _need(self.schema_version == "mal2026-rlaif-qwen3-embedding-eval-v1" and self.arm in ARMS, "Qwen3 evaluation identity differs")
        metadata = training_dir(self.arm, "full") / "training_complete.json"
        _need(Path(self.training_metadata_path).resolve() == metadata.resolve(), "Qwen3 evaluation training lineage differs")
        output = Path(self.output_dir)
        _need(output.is_absolute() and output.parent == EVAL_ROOT.resolve() and not output.exists(), "Qwen3 evaluation output freshness differs")
        _need(output.name == self.run_id == f"rlaif-qwen3-embedding-eval-v1-{self.arm}-validation-001" and self.per_device_eval_batch_size == 8, "Qwen3 evaluation config differs")


def evaluation_config(arm: str) -> dict[str, Any]:
    _need(arm in ARMS, "unknown Qwen3 evaluation arm")
    run_id = f"rlaif-qwen3-embedding-eval-v1-{arm}-validation-001"
    return {
        "schema_version": "mal2026-rlaif-qwen3-embedding-eval-v1",
        "run_id": run_id,
        "arm": arm,
        "training_metadata_path": str((training_dir(arm, "full") / "training_complete.json").resolve()),
        "output_dir": str(evaluation_dir(arm).resolve()),
        "per_device_eval_batch_size": 8,
    }


def run_evaluation(config: Qwen3EmbeddingEvalConfig) -> dict[str, Any]:
    config.validate()
    metadata = _read_json(Path(config.training_metadata_path), "Qwen3 training completion")
    _need(metadata.get("status") == "completed" and metadata.get("arm") == config.arm and metadata.get("score_fields") == list(AXES), "Qwen3 training provenance differs")
    raw = metadata.get("config")
    _need(isinstance(raw, dict) and isinstance(raw.get("score_fields"), list), "Qwen3 saved train config differs")
    raw["score_fields"] = tuple(raw["score_fields"])
    train = Qwen3EmbeddingTrainConfig(**raw)
    train.validate(require_fresh_output=False)
    state_path = Path(train.output_dir) / "trainable_model.safetensors"
    _need(state_path.is_file() and _sha(state_path) == metadata.get("trainable_state_sha256"), "Qwen3 trainable state checksum differs")
    try:
        import numpy as np
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Qwen3 embedding evaluation requires .venv-standard") from exc
    tokenizer = AutoTokenizer.from_pretrained(train.model_path, revision=train.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = _examples("validation", EXPECTED_ESSAYS["validation"])
    dataset = _tokenized(examples, tokenizer, train.max_length, include_source=True)
    model, _ = build_model(train)
    state = load_file(str(state_path), device="cpu")
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    _need(set(state) == expected, "Qwen3 saved trainable tensor names differ")
    incompatible = model.load_state_dict(state, strict=False)
    _need(not incompatible.unexpected_keys and not (expected & set(incompatible.missing_keys)), "Qwen3 trainable state load differs")
    output = Path(config.output_dir)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(output), do_train=False, do_eval=False, per_device_eval_batch_size=config.per_device_eval_batch_size, bf16=True, tf32=True, report_to=[], remove_unused_columns=False),
        data_collator=_collator(tokenizer),
    )
    prediction = trainer.predict(dataset).predictions
    values = prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
    _need(len(values) == len(examples), "Qwen3 prediction count differs")
    truth = [[float(value) for value in item["labels"]] for item in examples]
    predicted = [[float(value) for value in vector] for vector in values]
    metrics = three_axis_metrics(truth, predicted)
    _need(all(math.isfinite(value) for axis in AXES for value in metrics[axis].values()), "Qwen3 evaluation metric is non-finite")
    payload = {
        "status": "completed",
        "run_id": config.run_id,
        "training_run_id": metadata.get("run_id"),
        "arm": config.arm,
        "source_key": RATIONALE_SOURCE,
        "backbone": "qwen3_embedding",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "initialization": metadata.get("initialization"),
        "score_fields": list(AXES),
        "average_target_used": False,
        "metrics": metrics,
        "validation": {"unique_essays": len(examples), "input_records": len(examples), "predictions_per_essay": 1, "rationale_sources_combined": 0},
        "trainable_state_sha256": _sha(state_path),
        "config": asdict(config),
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
    }
    trainer.accelerator.wait_for_everyone()
    failed = False
    if trainer.is_world_process_zero():
        try:
            _need(output.is_dir() and not (output / "aggregate_metrics.json").exists(), "Qwen3 evaluation output was reused")
            _atomic_json(output / "aggregate_metrics.json", payload)
        except Exception:
            failed = True
    state_payload: list[Any] = [failed, payload if trainer.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state_payload, src=0)
    if state_payload[0]:
        raise Qwen3EmbeddingExperimentError("rank-zero Qwen3 evaluation persistence failed")
    _need(isinstance(state_payload[1], dict) and state_payload[1].get("status") == "completed", "Qwen3 evaluation completion was not published")
    trainer.accelerator.wait_for_everyone()
    return state_payload[1]
