"""Epoch-checkpoint sweep for the winning Qwen3-Embedding warm start.

The experiment repeats the fixed 12-epoch rationale-conditioned training run,
but saves the reconstructable LoRA plus three-score head after every epoch and
then evaluates all twelve checkpoints on the same descriptive validation set.
No raw input, rationale, identifier, or prediction is persisted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from .api_rationale_data import SOURCE_SHA256
from .rlaif_qwen3_embedding import (
    AXES,
    MODEL_ID,
    MODEL_PATH,
    MODEL_REVISION,
    RATIONALE_SOURCE,
    WARMSTART_METADATA,
    _atomic_json,
    _collator,
    _examples,
    _sha,
    _tokenized,
    _trainable_state,
    build_model,
    warmstart_provenance,
)
from .rlaif_top3_encoder import generation_dir, three_axis_metrics


ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "outputs" / "rlaif-qwen3-embedding-epoch-sweep-v1"
EVAL_ROOT = ROOT / "outputs" / "rlaif-qwen3-embedding-epoch-sweep-evals-v1"
PHASES = ("gpu0_preflight", "full")
FULL_EPOCHS = 12
FULL_STEPS_PER_EPOCH = 32


class EpochSweepError(ValueError):
    """Raised when the fixed epoch-sweep contract differs."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise EpochSweepError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpochSweepError(f"{label} is unreadable") from exc
    _need(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def training_dir(phase: str) -> Path:
    return TRAIN_ROOT / f"rlaif-qwen3-embedding-epoch-sweep-v1-{phase}-003"


def evaluation_dir(phase: str) -> Path:
    return EVAL_ROOT / f"rlaif-qwen3-embedding-epoch-sweep-eval-v1-{phase}-003"


@dataclass(frozen=True)
class EpochSweepTrainConfig:
    schema_version: str
    run_id: str
    phase: str
    arm: str
    source_key: str
    model_id: str
    model_revision: str
    model_path: str
    initialization: str
    warmstart_metadata_path: str
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
    def from_json(cls, path: Path) -> "EpochSweepTrainConfig":
        raw = _read_json(path, "epoch-sweep train config")
        _need(isinstance(raw.get("score_fields"), list), "epoch-sweep score fields must be a list")
        raw["score_fields"] = tuple(raw["score_fields"])
        _need(set(raw) == set(cls.__dataclass_fields__), "epoch-sweep train config fields differ")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self, *, require_fresh_output: bool = True) -> None:
        _need(self.schema_version == "mal2026-rlaif-qwen3-embedding-epoch-sweep-train-v1", "epoch-sweep train schema differs")
        _need(self.phase in PHASES and self.arm == "qwen3_aihub_warmstart", "epoch-sweep phase/arm differs")
        _need(self.source_key == RATIONALE_SOURCE and self.score_fields == AXES, "epoch-sweep data/target contract differs")
        _need(self.initialization == "aihub_48016_warmstart" and Path(self.warmstart_metadata_path).resolve() == WARMSTART_METADATA.resolve(), "epoch-sweep initialization differs")
        _need((self.model_id, self.model_revision, Path(self.model_path).resolve()) == (MODEL_ID, MODEL_REVISION, MODEL_PATH.resolve()), "epoch-sweep model snapshot differs")
        warmstart_provenance()
        output = Path(self.output_dir)
        _need(output.is_absolute() and output.parent == TRAIN_ROOT.resolve(), "epoch-sweep train output root differs")
        expected = f"rlaif-qwen3-embedding-epoch-sweep-v1-{self.phase}-003"
        _need(output.name == self.run_id == expected, "epoch-sweep train run identity differs")
        _need((not output.exists()) if require_fresh_output else output.is_dir(), "epoch-sweep train output freshness differs")
        _need((self.seed, self.max_length, self.learning_rate, self.weight_decay, self.warmup_ratio) == (2026072601, 2048, 1e-4, 0.01, 0.05), "epoch-sweep optimizer constants differ")
        _need((self.lora_r, self.lora_alpha, self.lora_dropout, self.training_dtype) == (16, 32, 0.05, "bfloat16"), "epoch-sweep LoRA/numeric contract differs")
        if self.phase == "full":
            _need((self.num_train_epochs, self.max_steps, self.train_record_limit, self.per_device_train_batch_size, self.gradient_accumulation_steps) == (12.0, -1, 2000, 4, 4), "epoch-sweep full schedule differs")
        else:
            _need((self.num_train_epochs, self.max_steps, self.train_record_limit, self.per_device_train_batch_size, self.gradient_accumulation_steps) == (1.0, 1, 4, 4, 1), "epoch-sweep preflight schedule differs")


def training_config(phase: str) -> dict[str, Any]:
    _need(phase in PHASES, "unknown epoch-sweep train phase")
    full = phase == "full"
    run_id = f"rlaif-qwen3-embedding-epoch-sweep-v1-{phase}-003"
    return {
        "schema_version": "mal2026-rlaif-qwen3-embedding-epoch-sweep-train-v1",
        "run_id": run_id,
        "phase": phase,
        "arm": "qwen3_aihub_warmstart",
        "source_key": RATIONALE_SOURCE,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(MODEL_PATH.resolve()),
        "initialization": "aihub_48016_warmstart",
        "warmstart_metadata_path": str(WARMSTART_METADATA.resolve()),
        "output_dir": str(training_dir(phase).resolve()),
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


def expected_checkpoint_steps(phase: str) -> dict[int, int]:
    _need(phase in PHASES, "unknown epoch-sweep phase")
    if phase == "gpu0_preflight":
        return {1: 1}
    return {epoch: epoch * FULL_STEPS_PER_EPOCH for epoch in range(1, FULL_EPOCHS + 1)}


def checkpoint_dir(output: Path, epoch: int) -> Path:
    return output / "epoch_checkpoints" / f"epoch-{epoch:02d}"


def run_training(config: EpochSweepTrainConfig) -> dict[str, Any]:
    config.validate()
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("epoch sweep requires .venv-standard") from exc

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        _need(tokenizer.eos_token is not None, "epoch-sweep tokenizer lacks pad/EOS")
        tokenizer.pad_token = tokenizer.eos_token
    examples = _examples("train", config.train_record_limit)
    dataset = _tokenized(examples, tokenizer, config.max_length, include_source=False)
    model, initialization = build_model(config)
    expected = expected_checkpoint_steps(config.phase)
    output = Path(config.output_dir)

    class EpochCheckpointCallback(TrainerCallback):
        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            epoch = int(round(float(state.epoch or 0.0)))
            _need(epoch in expected and int(state.global_step) == expected[epoch], "epoch-sweep callback boundary differs")
            failed = False
            failure_message = None
            if state.is_world_process_zero:
                try:
                    root = checkpoint_dir(output, epoch)
                    _need(not root.exists(), "epoch-sweep checkpoint already exists")
                    root.mkdir(parents=True)
                    state_path = root / "trainable_model.safetensors"
                    trainable = _trainable_state(kwargs["model"])
                    save_file(trainable, str(state_path))
                    _atomic_json(root / "checkpoint_metadata.json", {
                        "status": "completed",
                        "run_id": config.run_id,
                        "epoch": epoch,
                        "global_step": int(state.global_step),
                        "score_fields": list(AXES),
                        "average_target_used": False,
                        "trainable_tensor_count": len(trainable),
                        "trainable_state_sha256": _sha(state_path),
                        "model_id": MODEL_ID,
                        "model_revision": MODEL_REVISION,
                        "initialization": initialization,
                        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
                    })
                except Exception as exc:
                    failed = True
                    failure_message = f"{type(exc).__name__}: {exc}"
            checkpoint_status: list[Any] = [failed, failure_message]
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(checkpoint_status, src=0)
            if checkpoint_status[0]:
                raise EpochSweepError(f"epoch {epoch} checkpoint persistence failed: {checkpoint_status[1]}")
            return control

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
    trainer = Trainer(model=model, args=arguments, train_dataset=dataset, data_collator=_collator(tokenizer), callbacks=[EpochCheckpointCallback()])
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None
    failed = False
    if trainer.is_world_process_zero():
        try:
            metrics = {str(key): float(value) for key, value in trained.metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
            _need("train_loss" in metrics and all(math.isfinite(value) for value in metrics.values()), "epoch-sweep Trainer metrics are non-finite")
            checkpoints = []
            for epoch, step in expected.items():
                root = checkpoint_dir(output, epoch)
                metadata_path = root / "checkpoint_metadata.json"
                metadata = _read_json(metadata_path, "epoch checkpoint metadata")
                state_path = root / "trainable_model.safetensors"
                _need(metadata.get("epoch") == epoch and metadata.get("global_step") == step, "epoch checkpoint identity differs")
                _need(state_path.is_file() and metadata.get("trainable_state_sha256") == _sha(state_path), "epoch checkpoint state differs")
                checkpoints.append({"epoch": epoch, "global_step": step, "metadata_path": str(metadata_path.resolve()), "metadata_sha256": _sha(metadata_path), "trainable_state_path": str(state_path.resolve()), "trainable_state_sha256": metadata["trainable_state_sha256"]})
            payload = {
                "status": "completed",
                "run_id": config.run_id,
                "phase": config.phase,
                "arm": config.arm,
                "source_key": config.source_key,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "initialization": initialization,
                "score_fields": list(AXES),
                "average_target_used": False,
                "train_records": len(examples),
                "global_step": int(trainer.state.global_step),
                "train_metrics": metrics,
                "checkpoints": checkpoints,
                "input_provenance": {"canonical_source_sha256": dict(SOURCE_SHA256), "rationale_generation_sha256": _sha(generation_dir(RATIONALE_SOURCE, "train", "full") / "generated_rationales.jsonl")},
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
        raise EpochSweepError("rank-zero epoch-sweep persistence/health gate failed")
    _need(isinstance(state_payload[1], dict) and state_payload[1].get("status") == "completed", "epoch-sweep completion was not published")
    trainer.accelerator.wait_for_everyone()
    return state_payload[1]


@dataclass(frozen=True)
class EpochSweepEvalConfig:
    schema_version: str
    run_id: str
    phase: str
    training_metadata_path: str
    output_dir: str
    validation_record_limit: int
    per_device_eval_batch_size: int

    @classmethod
    def from_json(cls, path: Path) -> "EpochSweepEvalConfig":
        raw = _read_json(path, "epoch-sweep evaluation config")
        _need(set(raw) == set(cls.__dataclass_fields__), "epoch-sweep evaluation config fields differ")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self) -> None:
        _need(self.schema_version == "mal2026-rlaif-qwen3-embedding-epoch-sweep-eval-v1" and self.phase in PHASES, "epoch-sweep evaluation identity differs")
        metadata = training_dir(self.phase) / "training_complete.json"
        _need(Path(self.training_metadata_path).resolve() == metadata.resolve(), "epoch-sweep evaluation lineage differs")
        output = Path(self.output_dir)
        _need(output.is_absolute() and output.parent == EVAL_ROOT.resolve() and not output.exists(), "epoch-sweep evaluation output freshness differs")
        expected = f"rlaif-qwen3-embedding-epoch-sweep-eval-v1-{self.phase}-003"
        _need(output.name == self.run_id == expected, "epoch-sweep evaluation run identity differs")
        expected_limit, expected_batch = (400, 8) if self.phase == "full" else (4, 4)
        _need((self.validation_record_limit, self.per_device_eval_batch_size) == (expected_limit, expected_batch), "epoch-sweep evaluation population/batch differs")


def evaluation_config(phase: str) -> dict[str, Any]:
    _need(phase in PHASES, "unknown epoch-sweep evaluation phase")
    full = phase == "full"
    run_id = f"rlaif-qwen3-embedding-epoch-sweep-eval-v1-{phase}-003"
    return {
        "schema_version": "mal2026-rlaif-qwen3-embedding-epoch-sweep-eval-v1",
        "run_id": run_id,
        "phase": phase,
        "training_metadata_path": str((training_dir(phase) / "training_complete.json").resolve()),
        "output_dir": str(evaluation_dir(phase).resolve()),
        "validation_record_limit": 400 if full else 4,
        "per_device_eval_batch_size": 8 if full else 4,
    }


def run_evaluation(config: EpochSweepEvalConfig) -> dict[str, Any]:
    config.validate()
    training = _read_json(Path(config.training_metadata_path), "epoch-sweep training completion")
    _need(training.get("status") == "completed" and training.get("score_fields") == list(AXES) and training.get("average_target_used") is False, "epoch-sweep training provenance differs")
    raw = training.get("config")
    _need(isinstance(raw, dict) and isinstance(raw.get("score_fields"), list), "epoch-sweep saved train config differs")
    raw["score_fields"] = tuple(raw["score_fields"])
    train = EpochSweepTrainConfig(**raw)
    train.validate(require_fresh_output=False)
    expected = expected_checkpoint_steps(config.phase)
    checkpoints = training.get("checkpoints")
    _need(isinstance(checkpoints, list) and [item.get("epoch") for item in checkpoints if isinstance(item, dict)] == list(expected), "epoch-sweep checkpoint sequence differs")
    try:
        import numpy as np
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("epoch-sweep evaluation requires .venv-standard") from exc
    tokenizer = AutoTokenizer.from_pretrained(train.model_path, revision=train.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = _examples("validation", config.validation_record_limit)
    dataset = _tokenized(examples, tokenizer, train.max_length, include_source=True)
    model, _ = build_model(train)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    output = Path(config.output_dir)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(output), do_train=False, do_eval=False, per_device_eval_batch_size=config.per_device_eval_batch_size, bf16=True, tf32=True, report_to=[], remove_unused_columns=False), data_collator=_collator(tokenizer))
    truth = [[float(value) for value in item["labels"]] for item in examples]
    rows = []
    for item in checkpoints:
        epoch = int(item["epoch"])
        state_path = Path(item["trainable_state_path"])
        _need(state_path.is_file() and _sha(state_path) == item["trainable_state_sha256"], "epoch-sweep evaluation checkpoint checksum differs")
        state = load_file(str(state_path), device="cpu")
        _need(set(state) == trainable_names, "epoch-sweep checkpoint tensor names differ")
        incompatible = model.load_state_dict(state, strict=False)
        _need(not incompatible.unexpected_keys and not (trainable_names & set(incompatible.missing_keys)), "epoch-sweep checkpoint load differs")
        prediction = trainer.predict(dataset).predictions
        values = prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
        _need(len(values) == len(examples), "epoch-sweep prediction count differs")
        predicted = [[float(value) for value in vector] for vector in values]
        metrics = three_axis_metrics(truth, predicted)
        _need(all(math.isfinite(value) for axis in AXES for value in metrics[axis].values()), "epoch-sweep metric is non-finite")
        rows.append({"epoch": epoch, "global_step": expected[epoch], "metrics": metrics, "trainable_state_sha256": item["trainable_state_sha256"]})
    best = min(rows, key=lambda row: (float(row["metrics"]["three_axis_macro_rmse"]), -float(row["metrics"]["three_axis_macro_spearman"]), int(row["epoch"])))
    payload = {
        "status": "completed",
        "run_id": config.run_id,
        "training_run_id": training.get("run_id"),
        "phase": config.phase,
        "arm": "qwen3_aihub_warmstart",
        "source_key": RATIONALE_SOURCE,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "score_fields": list(AXES),
        "average_target_used": False,
        "epoch_results": rows,
        "best_epoch_by_validation_macro_rmse_then_spearman": best,
        "validation": {"unique_essays": len(examples), "input_records": len(examples), "predictions_per_essay_per_checkpoint": 1, "checkpoint_evaluations": len(rows), "rationale_sources_combined": 0},
        "selection_caveat": "canonical validation was explicitly reused to diagnose epochs; the best epoch is validation-selected and is not an untouched generalization estimate",
        "config": asdict(config),
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
    }
    trainer.accelerator.wait_for_everyone()
    failed = False
    if trainer.is_world_process_zero():
        try:
            _need(output.is_dir() and not (output / "epoch_sweep_metrics.json").exists(), "epoch-sweep evaluation output was reused")
            _atomic_json(output / "epoch_sweep_metrics.json", payload)
        except Exception:
            failed = True
    state_payload: list[Any] = [failed, payload if trainer.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state_payload, src=0)
    if state_payload[0]:
        raise EpochSweepError("rank-zero epoch-sweep evaluation persistence failed")
    _need(isinstance(state_payload[1], dict) and state_payload[1].get("status") == "completed", "epoch-sweep evaluation completion was not published")
    trainer.accelerator.wait_for_everyone()
    return state_payload[1]
