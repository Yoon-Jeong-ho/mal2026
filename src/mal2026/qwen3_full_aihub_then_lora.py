"""Full-parameter AI-Hub Qwen3 training followed by rationale LoRA.

This is the separately predeclared F-AIHUB arm.  Full-parameter state is kept
under ignored outputs; public artifacts are aggregate-only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .api_rationale_data import EXPECTED_ESSAYS, SOURCE_SHA256, load_writing_rows
from .rlaif_qwen3_embedding import (
    AXES, LORA_TARGETS, MODEL_ID, MODEL_PATH, MODEL_REVISION, RATIONALE_SOURCE,
    _atomic_json, _sha,
)
from .rlaif_top3_encoder import _input_text, _labels, _load_generated_rationales, generation_dir, three_axis_metrics
from .standard_decoder_data import DEFAULT_MANIFEST, SCORE_FIELDS, load_prepared_split


ROOT = Path(__file__).resolve().parents[2]
FULL_ROOT = ROOT / "outputs" / "qwen3-full-aihub-v1"
RATIONALE_ROOT = ROOT / "outputs" / "qwen3-full-aihub-rationale-lora-v1"
EVAL_ROOT = ROOT / "outputs" / "qwen3-full-aihub-rationale-lora-evals-v1"
PROGRAM_ID = "20260726-009"
FULL_PHASES = ("fsdp_gate", "selection", "refit")
RATIONALE_PHASES = ("gpu0_preflight", "full")
FULL_FINAL_STATE = FULL_ROOT / "qwen3-full-aihub-v1-refit-009" / "final_model" / "model.safetensors"
FULL_REFIT_METADATA = FULL_ROOT / "qwen3-full-aihub-v1-refit-009" / "full_aihub_training_complete.json"
FULL_SELECTION_METADATA = FULL_ROOT / "qwen3-full-aihub-v1-selection-009" / "full_aihub_training_complete.json"
FULL_LR = 2e-5
FULL_EPOCH_CAP = 20.0
FULL_SELECTION_MAX_STEPS = 2200
FULL_GLOBAL_BATCH = 64
FULL_PER_DEVICE_BATCH = 4
FULL_GRAD_ACCUM = 4
RATIONALE_EPOCHS = 4


class FullAIHubError(ValueError):
    """Raised when the full-tune/continuation contract differs."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise FullAIHubError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullAIHubError(f"{label} is unreadable") from exc
    _need(isinstance(value, dict), f"{label} must be an object")
    return value


def _file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_dir(phase: str) -> Path:
    return FULL_ROOT / f"qwen3-full-aihub-v1-{phase}-009"


def rationale_dir(phase: str) -> Path:
    return RATIONALE_ROOT / f"qwen3-full-aihub-rationale-lora-v1-{phase}-009"


def rationale_eval_dir(phase: str) -> Path:
    return EVAL_ROOT / f"qwen3-full-aihub-rationale-lora-eval-v1-{phase}-009"


def rationale_checkpoint_dir(output: Path, epoch: int) -> Path:
    return output / "epoch_checkpoints" / f"epoch-{epoch:02d}"


@dataclass(frozen=True)
class FullTrainConfig:
    schema_version: str
    run_id: str
    phase: str
    output_dir: str
    model_id: str
    model_revision: str
    model_path: str
    prepared_manifest: str
    selection_metadata_path: str | None
    score_fields: tuple[str, str, str, str]
    seed: int
    max_length: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    num_train_epochs: float
    max_steps: int
    record_limit: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    eval_steps: int
    save_steps: int
    logging_steps: int
    early_stopping_patience: int
    training_dtype: str
    optimizer: str
    fsdp_version: int
    fsdp_transformer_layer: str
    fsdp_state_dict_type: str

    @classmethod
    def from_json(cls, path: Path) -> "FullTrainConfig":
        raw = _read_json(path, "full AI-Hub config")
        _need(isinstance(raw.get("score_fields"), list), "full score fields must be a list")
        raw["score_fields"] = tuple(raw["score_fields"])
        _need(set(raw) == set(cls.__dataclass_fields__), "full config fields differ")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self, *, require_fresh_output: bool = True) -> None:
        _need(self.schema_version == "mal2026-qwen3-full-aihub-train-v1" and self.phase in FULL_PHASES, "full training identity differs")
        _need((self.model_id, self.model_revision, Path(self.model_path).resolve()) == (MODEL_ID, MODEL_REVISION, MODEL_PATH.resolve()), "full model snapshot differs")
        _need(Path(self.prepared_manifest).resolve() == DEFAULT_MANIFEST.resolve() and self.score_fields == SCORE_FIELDS, "full data/score contract differs")
        output = Path(self.output_dir)
        expected = f"qwen3-full-aihub-v1-{self.phase}-009"
        _need(output.is_absolute() and output.parent == FULL_ROOT.resolve() and output.name == self.run_id == expected, "full output identity differs")
        _need((not output.exists()) if require_fresh_output else output.is_dir(), "full output freshness differs")
        _need((self.seed, self.max_length, self.learning_rate, self.weight_decay, self.warmup_ratio) == (2026072602, 2048, FULL_LR, 0.01, 0.05), "full optimization contract differs")
        _need((self.training_dtype, self.optimizer, self.fsdp_version, self.fsdp_transformer_layer) == ("bfloat16", "adamw_torch_fused", 2, "Qwen3DecoderLayer"), "full numeric/FSDP contract differs")
        if self.phase == "fsdp_gate":
            _need((self.num_train_epochs, self.max_steps, self.record_limit, self.per_device_train_batch_size, self.gradient_accumulation_steps, self.eval_steps, self.save_steps, self.fsdp_state_dict_type) == (1.0, 1, 4, 1, 1, 0, 0, "SHARDED_STATE_DICT"), "FSDP gate schedule differs")
            _need(self.selection_metadata_path is None, "FSDP gate has no selection metadata")
        elif self.phase == "selection":
            _need((self.num_train_epochs, self.max_steps, self.record_limit, self.per_device_train_batch_size, self.per_device_eval_batch_size, self.gradient_accumulation_steps, self.eval_steps, self.save_steps, self.early_stopping_patience, self.fsdp_state_dict_type) == (FULL_EPOCH_CAP, FULL_SELECTION_MAX_STEPS, -1, FULL_PER_DEVICE_BATCH, 8, FULL_GRAD_ACCUM, 100, 0, 3, "SHARDED_STATE_DICT"), "selection schedule differs")
            _need(self.selection_metadata_path is None, "selection has no prior metadata")
        else:
            _need((self.num_train_epochs, self.max_steps, self.record_limit, self.per_device_train_batch_size, self.gradient_accumulation_steps, self.eval_steps, self.save_steps, self.fsdp_state_dict_type) == (FULL_EPOCH_CAP, -1, -1, FULL_PER_DEVICE_BATCH, FULL_GRAD_ACCUM, 0, 0, "FULL_STATE_DICT"), "refit schedule differs")
            _need(self.selection_metadata_path is not None and Path(self.selection_metadata_path).resolve() == FULL_SELECTION_METADATA.resolve(), "refit selection lineage differs")


def full_config(phase: str) -> dict[str, Any]:
    _need(phase in FULL_PHASES, "unknown full phase")
    run_id = f"qwen3-full-aihub-v1-{phase}-009"
    gate = phase == "fsdp_gate"
    selection = phase == "selection"
    return {
        "schema_version": "mal2026-qwen3-full-aihub-train-v1", "run_id": run_id, "phase": phase,
        "output_dir": str(full_dir(phase).resolve()), "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_path": str(MODEL_PATH.resolve()), "prepared_manifest": str(DEFAULT_MANIFEST.resolve()),
        "selection_metadata_path": str(FULL_SELECTION_METADATA.resolve()) if phase == "refit" else None,
        "score_fields": list(SCORE_FIELDS), "seed": 2026072602, "max_length": 2048,
        "learning_rate": FULL_LR, "weight_decay": 0.01, "warmup_ratio": 0.05,
        "num_train_epochs": 1.0 if gate else FULL_EPOCH_CAP,
        "max_steps": 1 if gate else (FULL_SELECTION_MAX_STEPS if selection else -1),
        "record_limit": 4 if gate else -1,
        "per_device_train_batch_size": 1 if gate else FULL_PER_DEVICE_BATCH,
        "per_device_eval_batch_size": 1 if gate else 8,
        "gradient_accumulation_steps": 1 if gate else FULL_GRAD_ACCUM,
        "eval_steps": 100 if selection else 0, "save_steps": 0,
        "logging_steps": 1 if gate else 5, "early_stopping_patience": 3,
        "training_dtype": "bfloat16", "optimizer": "adamw_torch_fused", "fsdp_version": 2,
        "fsdp_transformer_layer": "Qwen3DecoderLayer",
        "fsdp_state_dict_type": "FULL_STATE_DICT" if phase == "refit" else "SHARDED_STATE_DICT",
    }


def _last_nonpad(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
    final = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    _need(bool((final >= 0).all().item()), "full encoder example is empty")
    return last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), final]


def build_full_regressor(model_path: str, revision: str, fields: Sequence[str] = SCORE_FIELDS) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from transformers import AutoModel

    chosen = tuple(fields)
    _need(chosen in {SCORE_FIELDS, AXES}, "full regressor fields differ")
    backbone = AutoModel.from_pretrained(model_path, revision=revision, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    backbone.config.use_cache = False
    hidden = getattr(backbone.config, "hidden_size", None)
    _need(isinstance(hidden, int) and hidden > 0, "Qwen hidden size is unavailable")

    class FullRegressor(nn.Module):
        _no_split_modules = ["Qwen3DecoderLayer"]

        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.regression_head = nn.Linear(hidden, len(chosen))

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None, **_: Any) -> Mapping[str, Any]:
            hidden_state = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
            embedding = _last_nonpad(hidden_state, attention_mask)
            # FSDP2's BF16 mixed policy also casts the non-wrapped head. Match
            # its parameter dtype for GEMM, then expose FP32 logits/loss.
            head_input = functional.normalize(embedding, p=2, dim=-1).to(self.regression_head.weight.dtype)
            logits = self.regression_head(head_input).float()
            result: dict[str, Any] = {"logits": logits}
            if labels is not None:
                _need(tuple(labels.shape[-1:]) == (len(chosen),), "full label dimension differs")
                result["loss"] = functional.mse_loss(logits, labels.float(), reduction="mean")
            return result

    return FullRegressor()


def _aihub_dataset(rows: Sequence[Any], tokenizer: Any, max_length: int) -> Any:
    import torch
    from torch.utils.data import Dataset

    class AIHubDataset(Dataset[Any]):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> Mapping[str, Any]:
            row = rows[index]
            text = f"[과제]\n{row.prompt}\n[학생 글]\n{row.essay}"
            encoded = tokenizer(text, truncation=True, max_length=max_length, padding=False, return_attention_mask=True)
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": torch.tensor([row.score[field] for field in SCORE_FIELDS], dtype=torch.float32)}

    return AIHubDataset()


def _collator(tokenizer: Any):
    def collate(features: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        import torch
        batch = tokenizer.pad([{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in features], padding=True, return_tensors="pt")
        batch["labels"] = torch.stack([row["labels"] for row in features])
        return batch
    return collate


def _selection_metrics(prediction: Any) -> dict[str, float]:
    import numpy as np
    values = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    predicted = np.clip(np.asarray(values, dtype=np.float64), 1.0, 5.0)
    truth = np.asarray(prediction.label_ids, dtype=np.float64)
    _need(predicted.shape == truth.shape and predicted.shape[1] == len(SCORE_FIELDS), "selection prediction shape differs")
    result = {f"{field}_mae": float(np.mean(np.abs(predicted[:, index] - truth[:, index]))) for index, field in enumerate(SCORE_FIELDS)}
    result["primary_macro_mae"] = sum(result[f"{field}_mae"] for field in SCORE_FIELDS) / len(SCORE_FIELDS)
    return result


def _selected_step(metadata_path: str) -> int:
    value = _read_json(Path(metadata_path), "full selection metadata")
    _need(value.get("status") == "completed" and value.get("phase") == "selection", "full selection metadata differs")
    step = value.get("selected_global_step")
    _need(isinstance(step, int) and 0 < step <= FULL_SELECTION_MAX_STEPS, "full selected step differs")
    return step


def _selected_metrics(state: Any, step: int) -> dict[str, float]:
    for event in reversed(state.log_history):
        if event.get("step") == step and "eval_primary_macro_mae" in event:
            result = {str(key): float(value) for key, value in event.items() if key.startswith("eval_") and isinstance(value, (int, float)) and not isinstance(value, bool)}
            _need(all(math.isfinite(value) for value in result.values()), "selected metrics are non-finite")
            return result
    raise FullAIHubError("selected metrics event is unavailable")


def run_full_training(config: FullTrainConfig) -> dict[str, Any]:
    config.validate()
    try:
        import torch
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("full AI-Hub training requires .venv-standard") from exc
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    split = "refit_train" if config.phase == "refit" else "selection_train"
    rows = load_prepared_split(split, Path(config.prepared_manifest))
    if config.record_limit > 0:
        rows = rows[:config.record_limit]
    dev = load_prepared_split("selection_dev", Path(config.prepared_manifest)) if config.phase == "selection" else None
    model = build_full_regressor(config.model_path, config.model_revision, SCORE_FIELDS)
    train_dataset = _aihub_dataset(rows, tokenizer, config.max_length)
    eval_dataset = _aihub_dataset(dev, tokenizer, config.max_length) if dev is not None else None
    resolved_steps = _selected_step(config.selection_metadata_path) if config.phase == "refit" and config.selection_metadata_path else config.max_steps
    fsdp_config = {
        "version": config.fsdp_version, "reshard_after_forward": True,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "transformer_layer_cls_to_wrap": [config.fsdp_transformer_layer],
        "activation_checkpointing": True, "state_dict_type": config.fsdp_state_dict_type,
        "cpu_ram_efficient_loading": False,
    }
    selection = config.phase == "selection"
    class SelectionTracker(TrainerCallback):
        """Track/stop selection without materializing repeated 16 GiB states."""

        def __init__(self) -> None:
            self.best_metric: float | None = None
            self.best_step: int | None = None
            self.patience = 0

        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Mapping[str, Any], **kwargs: Any) -> Any:
            value = metrics.get("eval_primary_macro_mae")
            _need(isinstance(value, (int, float)) and math.isfinite(float(value)), "selection callback metric differs")
            parsed = float(value)
            if self.best_metric is None or parsed < self.best_metric:
                self.best_metric, self.best_step, self.patience = parsed, int(state.global_step), 0
            else:
                self.patience += 1
                if self.patience >= config.early_stopping_patience:
                    control.should_training_stop = True
            return control

    tracker = SelectionTracker() if selection else None
    args = TrainingArguments(
        output_dir=config.output_dir, run_name=config.run_id, do_train=True, do_eval=selection,
        eval_strategy="steps" if selection else "no", eval_steps=config.eval_steps if selection else None,
        save_strategy="no", save_steps=None, load_best_model_at_end=False,
        logging_strategy="steps", logging_steps=config.logging_steps,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs, max_steps=resolved_steps,
        per_device_train_batch_size=config.per_device_train_batch_size, per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=True, tf32=True, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=True, max_grad_norm=1.0,
        optim=config.optimizer, logging_nan_inf_filter=False, seed=config.seed, data_seed=config.seed,
        fsdp=True, fsdp_config=fsdp_config,
    )
    callbacks = [tracker] if tracker is not None else []
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=_collator(tokenizer), compute_metrics=_selection_metrics if selection else None, callbacks=callbacks)
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    if config.phase == "refit":
        # FSDP full-state materialization is collective; every rank must enter.
        trainer.save_model(str(Path(config.output_dir) / "final_model"))
        trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None
    failed = False
    if trainer.is_world_process_zero():
        try:
            metrics = {str(key): float(value) for key, value in trained.metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
            _need("train_loss" in metrics and all(math.isfinite(value) for value in metrics.values()), "full training metrics are non-finite")
            if selection:
                selected = tracker.best_step if tracker is not None else None
                _need(isinstance(selected, int) and selected > 0, "selection best step is unavailable")
                selection_metrics = _selected_metrics(trainer.state, selected)
            else:
                selected = int(trainer.state.global_step)
                selection_metrics = {}
            if config.phase == "refit":
                _need(FULL_FINAL_STATE.is_file(), "refit full state was not saved")
            payload = {
                "status": "completed", "run_id": config.run_id, "phase": config.phase,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "score_fields": list(SCORE_FIELDS),
                "average_target_used": True, "train_records": len(rows), "selection_dev_records": len(dev) if dev is not None else 0,
                "selected_global_step": int(selected), "trainer_global_step": int(trainer.state.global_step),
                "train_metrics": metrics, "selection_metrics": selection_metrics,
                "model_state_path": str(FULL_FINAL_STATE.resolve()) if config.phase == "refit" else None,
                "model_state_sha256": _file_sha(FULL_FINAL_STATE) if config.phase == "refit" else None,
                "config": asdict(config),
                "privacy": "aggregate_only_no_rows_prompts_essays_feedback_ids_or_predictions_persisted",
            }
            _atomic_json(Path(config.output_dir) / "full_aihub_training_complete.json", payload)
        except Exception:
            failed = True
    message: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(message, src=0)
    if message[0]:
        raise FullAIHubError("rank-zero full-training finalization failed")
    _need(isinstance(message[1], dict), "full training completion was not published")
    trainer.accelerator.wait_for_everyone()
    return message[1]


def run_gpu0_construction_gate(output: Path) -> dict[str, Any]:
    _need(output.is_absolute() and output.parent == FULL_ROOT.resolve() and not output.exists(), "construction gate output differs")
    import torch
    from transformers import AutoTokenizer
    rows = load_prepared_split("selection_train", DEFAULT_MANIFEST)[:1]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = _aihub_dataset(rows, tokenizer, 2048)
    batch = _collator(tokenizer)([dataset[0]])
    model = build_full_regressor(str(MODEL_PATH), MODEL_REVISION, SCORE_FIELDS).to("cuda")
    with torch.no_grad():
        output_value = model(**{key: value.to("cuda") for key, value in batch.items()})
    logits = output_value["logits"]
    _need(tuple(logits.shape) == (1, 4) and bool(torch.isfinite(logits).all().item()) and math.isfinite(float(output_value["loss"].item())), "construction forward gate failed")
    payload = {"status": "completed", "run_id": "qwen3-full-aihub-v1-gpu0-construction-009", "gpu_scope": [0], "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "score_fields": list(SCORE_FIELDS), "forward_shape": [1, 4], "finite_loss": True, "privacy": "aggregate_only"}
    _atomic_json(output / "construction_gate.json", payload)
    return payload


@dataclass(frozen=True)
class FullRationaleConfig:
    schema_version: str
    run_id: str
    phase: str
    output_dir: str
    full_refit_metadata_path: str
    full_model_state_path: str
    score_fields: tuple[str, str, str]
    seed: int
    max_length: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    num_train_epochs: float
    max_steps: int
    essay_limit: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float

    @classmethod
    def from_json(cls, path: Path) -> "FullRationaleConfig":
        raw = _read_json(path, "full rationale config")
        _need(isinstance(raw.get("score_fields"), list), "rationale score fields must be a list")
        raw["score_fields"] = tuple(raw["score_fields"])
        _need(set(raw) == set(cls.__dataclass_fields__), "rationale config fields differ")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self, *, require_fresh_output: bool = True) -> None:
        _need(self.schema_version == "mal2026-qwen3-full-aihub-rationale-lora-v1" and self.phase in RATIONALE_PHASES, "rationale identity differs")
        output = Path(self.output_dir)
        expected = f"qwen3-full-aihub-rationale-lora-v1-{self.phase}-009"
        _need(output.is_absolute() and output.parent == RATIONALE_ROOT.resolve() and output.name == self.run_id == expected, "rationale output identity differs")
        _need((not output.exists()) if require_fresh_output else output.is_dir(), "rationale output freshness differs")
        _need(Path(self.full_refit_metadata_path).resolve() == FULL_REFIT_METADATA.resolve() and Path(self.full_model_state_path).resolve() == FULL_FINAL_STATE.resolve(), "rationale full-refit lineage differs")
        _need(self.score_fields == AXES and (self.seed, self.max_length, self.learning_rate, self.weight_decay, self.warmup_ratio) == (2026072601, 2048, 1e-4, 0.01, 0.05), "rationale optimization contract differs")
        _need((self.lora_r, self.lora_alpha, self.lora_dropout) == (16, 32, 0.05), "rationale LoRA contract differs")
        if self.phase == "full":
            _need((self.num_train_epochs, self.max_steps, self.essay_limit, self.per_device_batch_size, self.gradient_accumulation_steps) == (4.0, -1, 2000, 2, 8), "rationale full schedule differs")
        else:
            _need((self.num_train_epochs, self.max_steps, self.essay_limit, self.per_device_batch_size, self.gradient_accumulation_steps) == (1.0, 1, 4, 4, 1), "rationale preflight schedule differs")


def rationale_config(phase: str) -> dict[str, Any]:
    _need(phase in RATIONALE_PHASES, "unknown rationale phase")
    full = phase == "full"
    run_id = f"qwen3-full-aihub-rationale-lora-v1-{phase}-009"
    return {
        "schema_version": "mal2026-qwen3-full-aihub-rationale-lora-v1", "run_id": run_id, "phase": phase,
        "output_dir": str(rationale_dir(phase).resolve()), "full_refit_metadata_path": str(FULL_REFIT_METADATA.resolve()),
        "full_model_state_path": str(FULL_FINAL_STATE.resolve()), "score_fields": list(AXES),
        "seed": 2026072601, "max_length": 2048, "learning_rate": 1e-4, "weight_decay": 0.01, "warmup_ratio": 0.05,
        "num_train_epochs": 4.0 if full else 1.0, "max_steps": -1 if full else 1,
        "essay_limit": 2000 if full else 4, "per_device_batch_size": 2 if full else 4,
        "gradient_accumulation_steps": 8 if full else 1, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    }


def _validate_refit(config: FullRationaleConfig) -> dict[str, Any]:
    metadata = _read_json(Path(config.full_refit_metadata_path), "full refit metadata")
    state = Path(config.full_model_state_path)
    _need(metadata.get("status") == "completed" and metadata.get("phase") == "refit" and metadata.get("score_fields") == list(SCORE_FIELDS), "full refit metadata differs")
    _need(state.is_file() and metadata.get("model_state_sha256") == _file_sha(state), "full refit state checksum differs")
    return metadata


def _rationale_examples(split: str, limit: int) -> list[dict[str, Any]]:
    generated = _load_generated_rationales(generation_dir(RATIONALE_SOURCE, split, "full"), RATIONALE_SOURCE, split, "full", EXPECTED_ESSAYS[split])
    rows = load_writing_rows(split, include_scores=True)[:limit]
    result = [{"text": _input_text(row.prompt, row.essay, generated[row.identifier]), "labels": _labels(row)} for row in rows]
    _need(len(result) == limit, "full rationale example count differs")
    return result


def _rationale_dataset(items: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int) -> Any:
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": [item["text"] for item in items], "labels": [item["labels"] for item in items]})
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])


def _rationale_collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        labels = [feature.pop("labels") for feature in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch
    return collate


def build_full_warm_lora(config: FullRationaleConfig) -> Any:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from safetensors.torch import load_file
    model = build_full_regressor(str(MODEL_PATH), MODEL_REVISION, AXES)
    four_state = load_file(config.full_model_state_path, device="cpu")
    three_state = {}
    for name, tensor in four_state.items():
        three_state[name] = tensor[:3] if name in {"regression_head.weight", "regression_head.bias"} else tensor
    incompatible = model.load_state_dict(three_state, strict=True)
    _need(not incompatible.missing_keys and not incompatible.unexpected_keys, "full refit state load differs")
    leaves = {name.rsplit(".", 1)[-1] for name, _ in model.backbone.named_modules()}
    _need(set(LORA_TARGETS) <= leaves, "full refit backbone lacks LoRA targets")
    model.backbone = get_peft_model(model.backbone, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=list(LORA_TARGETS), bias="none"))
    _need(all(not parameter.requires_grad for name, parameter in model.backbone.named_parameters() if "lora_" not in name), "full backbone base parameters were not frozen")
    return model


def _trainable_state(model: Any) -> dict[str, Any]:
    state = {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters() if parameter.requires_grad}
    _need(bool(state) and "regression_head.weight" in state and all("average" not in name for name in state), "full rationale trainable state differs")
    return state


def rationale_expected_steps(phase: str) -> dict[int, int]:
    return {1: 1} if phase == "gpu0_preflight" else {epoch: epoch * 32 for epoch in range(1, RATIONALE_EPOCHS + 1)}


def run_rationale_training(config: FullRationaleConfig) -> dict[str, Any]:
    config.validate()
    refit = _validate_refit(config)
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("full rationale LoRA requires .venv-standard") from exc
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    items = _rationale_examples("train", config.essay_limit)
    dataset = _rationale_dataset(items, tokenizer, config.max_length)
    model = build_full_warm_lora(config)
    expected = rationale_expected_steps(config.phase)
    output = Path(config.output_dir)

    class EpochCheckpoint(TrainerCallback):
        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            epoch = int(round(float(state.epoch or 0.0)))
            _need(epoch in expected and int(state.global_step) == expected[epoch], "full rationale checkpoint boundary differs")
            failed = False
            if state.is_world_process_zero:
                try:
                    root = rationale_checkpoint_dir(output, epoch)
                    _need(not root.exists(), "full rationale checkpoint exists")
                    root.mkdir(parents=True)
                    state_path = root / "trainable_model.safetensors"
                    tensors = _trainable_state(kwargs["model"])
                    save_file(tensors, str(state_path))
                    _atomic_json(root / "checkpoint_metadata.json", {"status": "completed", "epoch": epoch, "global_step": int(state.global_step), "trainable_state_sha256": _sha(state_path), "score_fields": list(AXES), "average_target_used": False, "privacy": "aggregate_only"})
                except Exception:
                    failed = True
            message = [failed]
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(message, src=0)
            if message[0]:
                raise FullAIHubError("full rationale checkpoint persistence failed")
            return control

    args = TrainingArguments(output_dir=config.output_dir, run_name=config.run_id, do_train=True, do_eval=False, eval_strategy="no", save_strategy="no", logging_strategy="steps", logging_steps=1 if config.phase == "gpu0_preflight" else 5, learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio, num_train_epochs=config.num_train_epochs, max_steps=config.max_steps, per_device_train_batch_size=config.per_device_batch_size, gradient_accumulation_steps=config.gradient_accumulation_steps, bf16=True, tf32=True, report_to=[], remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=True, ddp_find_unused_parameters=False, max_grad_norm=1.0, optim="adamw_torch", logging_nan_inf_filter=False, seed=config.seed, data_seed=config.seed)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=_rationale_collator(tokenizer), callbacks=[EpochCheckpoint()])
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None
    failed = False
    if trainer.is_world_process_zero():
        try:
            metrics = {str(key): float(value) for key, value in trained.metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
            _need("train_loss" in metrics and all(math.isfinite(value) for value in metrics.values()), "full rationale metrics are non-finite")
            checkpoints = []
            for epoch, step in expected.items():
                path = rationale_checkpoint_dir(output, epoch) / "trainable_model.safetensors"
                meta = _read_json(path.parent / "checkpoint_metadata.json", "full rationale checkpoint")
                _need(path.is_file() and meta.get("global_step") == step and meta.get("trainable_state_sha256") == _sha(path), "full rationale checkpoint differs")
                checkpoints.append({"epoch": epoch, "global_step": step, "trainable_state_path": str(path.resolve()), "trainable_state_sha256": meta["trainable_state_sha256"]})
            payload = {"status": "completed", "run_id": config.run_id, "phase": config.phase, "initialization": "full_parameter_aihub_48016_then_lora", "full_refit_run_id": refit["run_id"], "full_refit_state_sha256": refit["model_state_sha256"], "score_fields": list(AXES), "average_target_used": False, "unique_train_essays": len(items), "global_step": int(trainer.state.global_step), "train_metrics": metrics, "checkpoints": checkpoints, "input_provenance": {"canonical_source_sha256": dict(SOURCE_SHA256), "rationale_generation_sha256": _sha(generation_dir(RATIONALE_SOURCE, "train", "full") / "generated_rationales.jsonl")}, "config": asdict(config), "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted"}
            _atomic_json(output / "training_complete.json", payload)
        except Exception:
            failed = True
    message: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(message, src=0)
    if message[0]:
        raise FullAIHubError("rank-zero full rationale finalization failed")
    _need(isinstance(message[1], dict), "full rationale completion was not published")
    trainer.accelerator.wait_for_everyone()
    return message[1]


def run_rationale_evaluation(config: FullRationaleConfig, output: Path, essay_limit: int, per_device_batch_size: int) -> dict[str, Any]:
    config.validate(require_fresh_output=False)
    _need(output.is_absolute() and output.parent == EVAL_ROOT.resolve() and not output.exists(), "full rationale evaluation output differs")
    training = _read_json(Path(config.output_dir) / "training_complete.json", "full rationale training")
    _need(training.get("status") == "completed" and training.get("score_fields") == list(AXES) and training.get("average_target_used") is False, "full rationale training provenance differs")
    _validate_refit(config)
    try:
        import numpy as np
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("full rationale evaluation requires .venv-standard") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    items = _rationale_examples("validation", essay_limit)
    dataset = _rationale_dataset(items, tokenizer, config.max_length)
    model = build_full_warm_lora(config)
    expected_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(output), do_train=False, do_eval=False, per_device_eval_batch_size=per_device_batch_size, bf16=True, tf32=True, report_to=[], remove_unused_columns=False), data_collator=_rationale_collator(tokenizer))
    truth = [[float(value) for value in item["labels"]] for item in items]
    rows = []
    for checkpoint in training["checkpoints"]:
        path = Path(checkpoint["trainable_state_path"])
        _need(path.is_file() and checkpoint["trainable_state_sha256"] == _sha(path), "full rationale evaluation checkpoint differs")
        state = load_file(str(path), device="cpu")
        _need(set(state) == expected_names, "full rationale trainable tensor names differ")
        incompatible = model.load_state_dict(state, strict=False)
        _need(not incompatible.unexpected_keys and not (expected_names & set(incompatible.missing_keys)), "full rationale checkpoint load differs")
        raw = trainer.predict(dataset).predictions
        values = raw.tolist() if isinstance(raw, np.ndarray) else raw
        metrics = three_axis_metrics(truth, [[float(value) for value in vector] for vector in values])
        rows.append({"epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"], "metrics": metrics, "trainable_state_sha256": checkpoint["trainable_state_sha256"]})
    best = min(rows, key=lambda row: (float(row["metrics"]["three_axis_macro_rmse"]), -float(row["metrics"]["three_axis_macro_spearman"]), int(row["epoch"])))
    payload = {"status": "completed", "run_id": f"qwen3-full-aihub-rationale-lora-eval-v1-{config.phase}-009", "training_run_id": training["run_id"], "phase": config.phase, "initialization": "full_parameter_aihub_48016_then_lora", "score_fields": list(AXES), "average_target_used": False, "epoch_results": rows, "best_epoch_by_validation_macro_rmse_then_spearman": best, "validation": {"unique_essays": essay_limit, "input_records": essay_limit, "predictions_per_essay_per_checkpoint": 1}, "selection_caveat": "validation was previously exposed; descriptive development evidence only", "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted"}
    trainer.accelerator.wait_for_everyone()
    failed = False
    if trainer.is_world_process_zero():
        try:
            output.mkdir(parents=True)
            _atomic_json(output / "epoch_metrics.json", payload)
        except Exception:
            failed = True
    message: list[Any] = [failed, payload if trainer.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(message, src=0)
    if message[0]:
        raise FullAIHubError("rank-zero full rationale evaluation persistence failed")
    _need(isinstance(message[1], dict), "full rationale evaluation was not published")
    trainer.accelerator.wait_for_everyone()
    return message[1]
