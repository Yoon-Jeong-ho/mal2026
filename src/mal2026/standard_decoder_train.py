"""TRL SFTTrainer entry point for the two Qwen decoder protocols.

All optimizer, distributed, checkpoint, and early-stopping behavior remains in
TRL/Transformers.  This module only validates the research contract and maps
restricted in-memory rows into conversational examples.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .standard_decoder_data import (
    DEFAULT_MANIFEST, ROOT, StandardDecoderContractError, load_prepared_split,
    messages_for_sft, score_mean,
)


@dataclass(frozen=True)
class StandardSFTConfig:
    run_id: str
    phase: str  # selection | refit
    mode: str  # direct | human_feedback
    model_path: str
    tokenizer_path: str
    model_revision: str
    tokenizer_revision: str
    prepared_manifest: str
    output_dir: str
    seed: int = 2026
    max_length: int = 4096
    learning_rate: float = 2e-5
    num_train_epochs: float = 12.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 5
    early_stopping_patience: int = 4
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    wandb_project: str = "mal2026-korean-writing-scoring"
    wandb_entity: str | None = None

    @classmethod
    def from_json(cls, path: Path) -> "StandardSFTConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise StandardDecoderContractError("standard SFT config has missing or unknown fields")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.phase not in {"selection", "refit"} or self.mode not in {"direct", "human_feedback"}:
            raise StandardDecoderContractError("invalid phase or mode")
        if Path(self.prepared_manifest).resolve() != DEFAULT_MANIFEST.resolve():
            raise StandardDecoderContractError("SFT must use canonical aggregate prepared manifest")
        if not self.run_id or Path(self.output_dir).resolve().parent != (ROOT / "outputs" / "standard-runs").resolve():
            raise StandardDecoderContractError("output_dir must be a direct child of ignored outputs/standard-runs")
        if Path(self.output_dir).exists():
            raise StandardDecoderContractError("refusing to overwrite an existing run directory")
        if self.max_length not in ({2048} if self.mode == "direct" else {4096}):
            raise StandardDecoderContractError("mode-specific max_length is frozen")
        if min(self.learning_rate, self.num_train_epochs) <= 0 or min(self.per_device_train_batch_size, self.gradient_accumulation_steps) <= 0:
            raise StandardDecoderContractError("training hyperparameters must be positive")
        if self.phase == "selection" and (self.eval_steps <= 0 or self.save_steps != self.eval_steps or self.early_stopping_patience <= 0):
            raise StandardDecoderContractError("selection requires matched positive eval/save steps and early stopping")


def _conversation_dataset(rows, mode: str):
    from datasets import Dataset
    # Dataset.from_list requires a concrete list; a generator can silently fail
    # before Trainer sees the restricted split.
    return Dataset.from_list([{"messages": messages_for_sft(row, mode)} for row in rows])


def run_sft(config: StandardSFTConfig) -> None:
    """Run maintained TRL SFTTrainer; output contains checkpoints/aggregate metadata only."""
    config.validate()
    train_split = "selection_train" if config.phase == "selection" else "refit_train"
    train_rows = load_prepared_split(train_split, Path(config.prepared_manifest))
    eval_rows = load_prepared_split("selection_dev", Path(config.prepared_manifest)) if config.phase == "selection" else None
    try:
        import torch
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("standard stack requires the project .venv-standard (TRL/Transformers/PEFT)") from exc

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, revision=config.tokenizer_revision, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path, revision=config.model_revision, local_files_only=True, torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args = SFTConfig(
        output_dir=config.output_dir, run_name=config.run_id, seed=config.seed,
        max_length=config.max_length, packing=False, assistant_only_loss=True,
        learning_rate=config.learning_rate, num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps, logging_strategy="steps",
        eval_strategy="steps" if eval_rows else "no", save_strategy="steps" if eval_rows else "epoch",
        eval_steps=config.eval_steps if eval_rows else None, save_steps=config.save_steps if eval_rows else None,
        load_best_model_at_end=bool(eval_rows), metric_for_best_model="eval_loss" if eval_rows else None,
        greater_is_better=False if eval_rows else None, save_total_limit=2 if eval_rows else 1,
        bf16=True, tf32=True, gradient_checkpointing=True, report_to=["wandb"],
        remove_unused_columns=False, dataset_num_proc=1,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)] if eval_rows else []
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=_conversation_dataset(train_rows, config.mode),
        eval_dataset=_conversation_dataset(eval_rows, config.mode) if eval_rows else None,
        processing_class=tokenizer, peft_config=peft_config, callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(str(Path(config.output_dir) / "adapter"))
    tokenizer.save_pretrained(str(Path(config.output_dir) / "adapter"))
    # Only aggregate/provenance metadata. Do not write row predictions, source text, or outputs.
    completion = {
        "status": "completed", "run_id": config.run_id, "phase": config.phase, "mode": config.mode,
        "global_step": int(trainer.state.global_step), "best_metric": trainer.state.best_metric,
        "model_revision": config.model_revision, "tokenizer_revision": config.tokenizer_revision,
        "train_records": len(train_rows), "eval_records": len(eval_rows or []),
        "fallback_mean": score_mean(train_rows), "config": asdict(config),
    }
    (Path(config.output_dir) / "standard_training_complete.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
