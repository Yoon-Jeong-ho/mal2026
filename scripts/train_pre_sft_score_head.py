#!/usr/bin/env python3
"""Train one pre-SFT encoder score head with the maintained HF Trainer.

This deliberately has no decoder/SFT/DPO/GRPO code path.  Each run predicts
exactly one of content, organization, or expression.  The ensemble evaluator
is solely responsible for calculating the predicted average outside a model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mal2026.standard_decoder_data import DEFAULT_MANIFEST, ROOT, SCORE_FIELDS, StandardDecoderContractError, load_prepared_split
from mal2026.standard_encoder_data import build_encoder_dataset, encoder_collator
from mal2026.standard_encoder_model import EncoderModelSpec, build_encoder_regressor, build_encoder_tokenizer

RUN_ROOT = ROOT / "outputs" / "standard-encoder-runs"
HEAD_FIELDS = ("content", "organization", "expression")


class PreSFTScoreHeadError(StandardDecoderContractError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise PreSFTScoreHeadError(message)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@dataclass(frozen=True)
class Config:
    run_id: str
    target_field: str
    backbone: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    model_path: str
    prepared_manifest: str
    output_dir: str
    max_length: int = 2048
    seed: int = 2026
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    num_train_epochs: float = 20.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 5
    early_stopping_patience: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    wandb_project: str = "mal2026-korean-writing-scoring"
    wandb_entity: str | None = None
    utilization_only: bool = False
    utilization_label: str = ""

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        # The two utilization fields were added after the validated score-head
        # configs.  Preserve byte-compatible old configs while requiring an
        # explicit label for the non-scientific utilization mode.
        allowed_missing = {"utilization_only", "utilization_label"}
        need(isinstance(raw, dict) and set(raw).issubset(set(cls.__dataclass_fields__)) and set(cls.__dataclass_fields__).difference(raw).issubset(allowed_missing), "pre-SFT score-head config has missing or unknown fields")
        if isinstance(raw.get("lora_target_modules"), list):
            raw["lora_target_modules"] = tuple(raw["lora_target_modules"])
        config = cls(**raw)
        config.validate()
        return config

    def model_spec(self) -> EncoderModelSpec:
        return EncoderModelSpec.from_mapping({
            "backbone": self.backbone, "model_id": self.model_id,
            "revision": self.model_revision, "tokenizer_revision": self.tokenizer_revision,
            "model_path": self.model_path,
            "pooling": "last_nonpad" if self.backbone == "qwen3_embedding" else "remote_sentence_embedding",
            "normalize_embeddings": True, "lora_target_modules": list(self.lora_target_modules),
            "lora_r": self.lora_r, "lora_alpha": self.lora_alpha, "lora_dropout": self.lora_dropout,
            "nv_snapshot_dir": None, "nv_review": None,
        })

    def validate(self) -> None:
        need(self.target_field in HEAD_FIELDS, "target_field must be content, organization, or expression")
        need(Path(self.prepared_manifest).resolve() == DEFAULT_MANIFEST.resolve(), "must use canonical aggregate prepared manifest")
        output = Path(self.output_dir)
        need(output.is_absolute() and output.parent == RUN_ROOT.resolve() and not output.exists(), "output_dir must be a new direct child of standard encoder runs")
        need(self.max_length == 2048, "max_length is frozen at 2048")
        need(self.seed > 0 and self.learning_rate > 0 and self.weight_decay >= 0 and 0 <= self.warmup_ratio < 1, "invalid optimization values")
        need(self.num_train_epochs > 0 and self.per_device_train_batch_size > 0 and self.per_device_eval_batch_size > 0 and self.gradient_accumulation_steps > 0, "invalid batch values")
        need(self.eval_steps > 0 and self.save_steps == self.eval_steps and self.logging_steps > 0 and self.early_stopping_patience > 0, "selection/checkpoint cadence is invalid")
        if self.utilization_only:
            need(self.utilization_label == "utilization_only", "utilization mode requires utilization_only label")
            need(self.num_train_epochs <= 10000, "utilization mode hard caps epochs at 10000")
            need(self.save_steps >= 10000 and self.logging_steps >= 1000, "utilization mode requires sparse bounded checkpointing and logging")
        else:
            need(self.utilization_label == "", "utilization label is reserved for utilization-only runs")
        self.model_spec()


def metric_for(field: str):
    def metric(prediction: Any) -> dict[str, float]:
        values, labels = prediction.predictions, prediction.label_ids
        if isinstance(values, tuple):
            values = values[0]
        predicted = [min(5.0, max(1.0, float(row[0]))) for row in values]
        actual = [float(row[0]) for row in labels]
        mae = sum(abs(left - right) for left, right in zip(actual, predicted, strict=True)) / len(actual)
        return {"target_mae": mae}
    return metric


def finite(value: Any, label: str) -> float:
    need(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def run(config: Config) -> dict[str, Any]:
    config.validate()
    import os
    assigned_gpu = os.environ.get("MAL2026_RESERVED_PHYSICAL_GPU")
    need(assigned_gpu in {"1", "2", "3"} and os.environ.get("CUDA_VISIBLE_DEVICES") == assigned_gpu,
         "score-head training requires exactly its watchdog-assigned GPU 1, 2, or 3")
    try:
        import torch
        from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pre-SFT score heads require the project .venv-standard") from exc

    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_PROJECT"] = config.wandb_project
    os.environ["WANDB_RUN_NAME"] = config.run_id
    if config.wandb_entity:
        os.environ["WANDB_ENTITY"] = config.wandb_entity
    else:
        os.environ.pop("WANDB_ENTITY", None)

    set_seed(config.seed)
    spec = config.model_spec()
    tokenizer = build_encoder_tokenizer(spec)
    train_rows = load_prepared_split("selection_train", Path(config.prepared_manifest))
    train_dataset = build_encoder_dataset(train_rows, tokenizer, config.max_length, (config.target_field,))
    dev_dataset = None if config.utilization_only else build_encoder_dataset(load_prepared_split("selection_dev", Path(config.prepared_manifest)), tokenizer, config.max_length, (config.target_field,))
    model = build_encoder_regressor(spec, (config.target_field,))
    if config.utilization_only:
        # This branch is deliberately train-only: it never loads selection-dev,
        # evaluates, selects a checkpoint, or writes a usable metric payload.
        os.environ["WANDB_MODE"] = "disabled"
    args = TrainingArguments(
        output_dir=config.output_dir, do_train=True, do_eval=not config.utilization_only, eval_strategy="no" if config.utilization_only else "steps", save_strategy="steps",
        eval_steps=config.eval_steps, save_steps=config.save_steps, logging_steps=config.logging_steps,
        logging_strategy="steps", learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio, num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size, per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps, bf16=True, tf32=True, save_total_limit=1 if config.utilization_only else 2,
        load_best_model_at_end=not config.utilization_only, metric_for_best_model=None if config.utilization_only else "target_mae", greater_is_better=False,
        report_to=[] if config.utilization_only else ["wandb"], run_name=config.run_id, remove_unused_columns=False, dataloader_num_workers=0,
        dataloader_pin_memory=True, ddp_find_unused_parameters=False, seed=config.seed, data_seed=config.seed,
        save_only_model=False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=dev_dataset,
        data_collator=encoder_collator(tokenizer), compute_metrics=None if config.utilization_only else metric_for(config.target_field),
        callbacks=[] if config.utilization_only else [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    result = trainer.train()
    trainer.accelerator.wait_for_everyone()
    failed = False
    payload: dict[str, Any] | None = None
    if trainer.is_world_process_zero():
        try:
            state = trainer.state
            if config.utilization_only:
                need(isinstance(state.global_step, int) and state.global_step > 0, "utilization Trainer did not complete an update")
                status = {
                    "status": "completed", "run_id": config.run_id, "run_purpose": "utilization_only",
                    "target_field": config.target_field, "trainer_global_step": state.global_step,
                    "config": asdict(config),
                    "privacy": "aggregate_only_no_rows_prompts_essays_feedback_ids_predictions_or_model_outputs_persisted",
                    "metrics_policy": "no_validation_loaded_or_evaluated; no_metrics_are_scientific_evidence",
                }
                complete = Path(config.output_dir) / "utilization_only_status.json"
                need(not complete.exists(), "utilization status artifact already exists")
                complete.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                payload = status
                return payload
            need(isinstance(state.global_step, int) and state.global_step > 0, "Trainer did not complete an update")
            need(isinstance(state.best_global_step, int) and state.best_global_step > 0, "selection has no best checkpoint")
            best = finite(state.best_metric, "best metric")
            need(0.0 <= best <= 4.0, "best target MAE outside score range")
            final = Path(config.output_dir) / "final_model"
            trainer.save_model(str(final))
            model_state = final / "model.safetensors"
            need(model_state.is_file(), "Trainer did not save model.safetensors")
            payload = {
                "status": "completed", "run_id": config.run_id, "target_field": config.target_field,
                "phase": "selection_train_to_isolated_selection_dev", "selected_global_step": state.best_global_step,
                "trainer_global_step": state.global_step, "selection_best_target_mae": best,
                "train_metrics": {str(k): finite(v, f"train metric {k}") for k, v in result.metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
                "config": asdict(config), "model_state_sha256": digest(model_state),
                "prepared_manifest_sha256": digest(Path(config.prepared_manifest)),
                "average_policy": "not_modelled; predicted average is computed only by pre_sft_score_ensemble_eval.py",
                "privacy": "aggregate_only_no_rows_prompts_essays_feedback_ids_predictions_or_model_outputs_persisted",
            }
            complete = Path(config.output_dir) / "pre_sft_score_head_complete.json"
            need(not complete.exists(), "completion artifact already exists")
            complete.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            failed = True
    message: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(message, src=0)
    if message[0]:
        raise PreSFTScoreHeadError("rank-zero finalization failed; inspect job log")
    need(isinstance(message[1], dict) and message[1].get("status") == "completed", "missing completion payload")
    trainer.accelerator.wait_for_everyone()
    return message[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-config", action="store_true", help="validate paths and static contract without importing torch or touching a GPU")
    args = parser.parse_args()
    config = Config.from_json(args.config)
    if args.validate_config:
        print(json.dumps({"status": "validated", "run_id": config.run_id, "target_field": config.target_field, "gpu_free": True}, sort_keys=True))
        return
    print(json.dumps(run(config), sort_keys=True))


if __name__ == "__main__":
    main()
