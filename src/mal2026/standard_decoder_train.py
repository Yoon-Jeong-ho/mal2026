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
    selection_summary_path: str | None = None
    selected_global_step: int | None = None
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
        if self.phase == "selection":
            if self.eval_steps <= 0 or self.save_steps != self.eval_steps or self.early_stopping_patience <= 0:
                raise StandardDecoderContractError("selection requires matched positive eval/save steps and early stopping")
            if self.selection_summary_path is not None or self.selected_global_step is not None:
                raise StandardDecoderContractError("selection must not receive an external checkpoint selection")
        else:
            if not isinstance(self.selected_global_step, int) or self.selected_global_step <= 0:
                raise StandardDecoderContractError("refit requires a positive vLLM-selected global step")
            if not self.selection_summary_path:
                raise StandardDecoderContractError("refit requires the immutable vLLM checkpoint selection summary")
            _verify_refit_selection(self)


def _verify_refit_selection(config: StandardSFTConfig) -> dict[str, Any]:
    """Bind refit step count to the aggregate-only source-dev vLLM selector."""
    path = Path(config.selection_summary_path).resolve()
    standard_runs = (ROOT / "outputs" / "standard-runs").resolve()
    if not path.is_file() or not path.is_relative_to(standard_runs) or path.name != "selected_checkpoint.json":
        raise StandardDecoderContractError("refit selection summary must be a standard-run selected_checkpoint.json")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StandardDecoderContractError("refit selection summary is unreadable") from exc
    if not isinstance(summary, dict) or summary.get("status") != "completed":
        raise StandardDecoderContractError("refit selection summary is incomplete")
    expected = {"phase": "selection", "mode": config.mode, "model_revision": config.model_revision, "selected_global_step": config.selected_global_step}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise StandardDecoderContractError("refit selection summary does not match model/mode/selected step")
    return summary


def _prompt_completion_dataset(rows, mode: str):
    """TRL conversational prompt-completion form, independent of template masks.

    Qwen2.5's chat template does not implement the Jinja `{% generation %}`
    block required by `assistant_only_loss`; TRL's maintained prompt/completion
    preprocessing instead derives labels from the explicit completion boundary.
    """
    from datasets import Dataset
    examples = []
    for row in rows:
        messages = messages_for_sft(row, mode)
        examples.append({"prompt": messages[:-1], "completion": [messages[-1]]})
    return Dataset.from_list(examples)


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
        max_length=config.max_length, packing=False, completion_only_loss=True, assistant_only_loss=False,
        learning_rate=config.learning_rate, num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps, logging_strategy="steps",
        eval_strategy="steps" if eval_rows else "no", save_strategy="steps" if eval_rows else "epoch",
        eval_steps=config.eval_steps if eval_rows else None, save_steps=config.save_steps if eval_rows else None,
        # Trainer's lifecycle monitor is eval_loss only. It may stop loss-
        # converged selection runs, but it never selects the research winner.
        # Retain every candidate checkpoint; vLLM source-dev macro MAE selects
        # a checkpoint afterwards and refit receives that exact update count.
        load_best_model_at_end=bool(eval_rows), metric_for_best_model="eval_loss" if eval_rows else None,
        greater_is_better=False if eval_rows else None, save_total_limit=None if eval_rows else 1,
        max_steps=config.selected_global_step if config.phase == "refit" else -1,
        bf16=True, tf32=True, gradient_checkpointing=True, report_to=["wandb"],
        remove_unused_columns=False, dataset_num_proc=1,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)] if eval_rows else []
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=_prompt_completion_dataset(train_rows, config.mode),
        eval_dataset=_prompt_completion_dataset(eval_rows, config.mode) if eval_rows else None,
        processing_class=tokenizer, peft_config=peft_config, callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(str(Path(config.output_dir) / "adapter"))
    tokenizer.save_pretrained(str(Path(config.output_dir) / "adapter"))
    # Only aggregate/provenance metadata. Do not write row predictions, source text, or outputs.
    checkpoint_steps = sorted(
        int(path.name.removeprefix("checkpoint-"))
        for path in Path(config.output_dir).glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    )
    completion = {
        "status": "completed", "run_id": config.run_id, "phase": config.phase, "mode": config.mode,
        "global_step": int(trainer.state.global_step), "best_metric": trainer.state.best_metric,
        "model_revision": config.model_revision, "tokenizer_revision": config.tokenizer_revision,
        "train_records": len(train_rows), "eval_records": len(eval_rows or []),
        "fallback_mean": score_mean(train_rows), "selection_candidate_steps": checkpoint_steps if config.phase == "selection" else [],
        "selection_lifecycle": "Trainer eval_loss early-stopping/checkpointing; external vLLM source-dev macro-MAE selects refit step" if config.phase == "selection" else "refit uses selected_global_step from aggregate vLLM summary",
        "selected_global_step": config.selected_global_step, "selection_summary_path": config.selection_summary_path,
        "config": asdict(config),
    }
    (Path(config.output_dir) / "standard_training_complete.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
