#!/usr/bin/env python3
"""Continue a completed scorer on ``eval/train`` and score ``eval/validation``.

This is intentionally separate from the original frozen-final matrix.  It
never changes an existing adapter, checkpoint, matrix ledger, or evaluation
artifact; it writes a new aggregate-only run under the standard ignored output
roots.  The validation split is used exactly once after a predeclared update
budget, so it is a validation result, not a new held-out final result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from mal2026.metrics import compute_regression_metrics
from mal2026.standard_decoder_data import (
    ROOT, SCORE_FIELDS, RestrictedRow, StandardDecoderContractError,
    messages_for_generation, messages_for_sft, parse_decoder_scores, score_mean,
)
from mal2026.standard_encoder_data import build_encoder_dataset, encoder_collator
from mal2026.standard_encoder_model import EncoderModelSpec, build_encoder_regressor, build_encoder_tokenizer


TRAIN_PATH = ROOT / "eval" / "train.jsonl"
VALIDATION_PATH = ROOT / "eval" / "validation.jsonl"
DECODER_RUN_ROOT = ROOT / "outputs" / "standard-runs"
DECODER_EVAL_ROOT = ROOT / "outputs" / "standard-evals"
ENCODER_RUN_ROOT = ROOT / "outputs" / "standard-encoder-runs"
ENCODER_EVAL_ROOT = ROOT / "outputs" / "standard-encoder-evals"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise StandardDecoderContractError(message)


def _load_eval_split(path: Path, expected_sha256: str) -> list[RestrictedRow]:
    _need(path in {TRAIN_PATH, VALIDATION_PATH}, "only the fixed eval train/validation paths are allowed")
    _need(isinstance(expected_sha256, str) and len(expected_sha256) == 64, "eval split SHA-256 is invalid")
    _need(path.is_file() and _sha256(path) == expected_sha256, "eval split hash check failed")
    rows: list[RestrictedRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        _need(isinstance(raw, dict) and set(raw) == {"id", "document_id", "prompt_num", "prompt", "essay", "score"}, "eval row schema changed")
        score = raw["score"]
        _need(isinstance(score, dict) and tuple(score) == SCORE_FIELDS, "eval score schema changed")
        clean = {field: float(score[field]) for field in SCORE_FIELDS}
        _need(all(math.isfinite(value) and 1.0 <= value <= 5.0 for value in clean.values()), "score outside [1, 5]")
        _need(isinstance(raw["id"], str) and raw["id"].strip() and isinstance(raw["prompt"], str) and raw["prompt"].strip() and isinstance(raw["essay"], str) and raw["essay"].strip(), "eval text/id field is blank")
        rows.append(RestrictedRow(identifier=raw["id"], prompt=raw["prompt"], essay=raw["essay"], score=clean, feedback=None))
    _need(bool(rows) and len({row.identifier for row in rows}) == len(rows), "eval split must have unique nonempty IDs")
    return rows


def _macro(metrics: dict[str, Any]) -> dict[str, float | None]:
    per_target = metrics["per_target"]
    rmses = [float(per_target[field]["rmse"]) for field in SCORE_FIELDS]
    maes = [float(per_target[field]["mae"]) for field in SCORE_FIELDS]
    rhos = [per_target[field]["spearman_rho"] for field in SCORE_FIELDS]
    return {
        "primary_macro_rmse": sum(rmses) / len(rmses),
        "macro_mae": sum(maes) / len(maes),
        "macro_spearman_rho": None if any(value is None for value in rhos) else sum(float(value) for value in rhos) / len(rhos),
    }


@dataclass(frozen=True)
class Config:
    run_id: str
    kind: str  # decoder_direct | encoder_qwen3
    output_dir: str
    evaluation_output_dir: str
    train_sha256: str
    validation_sha256: str
    initial_path: str
    model_path: str
    model_revision: str
    tokenizer_revision: str
    max_steps: int
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    seed: int
    wandb_project: str

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        _need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "adaptation config has missing or unknown fields")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        _need(self.kind in {"decoder_direct", "encoder_qwen3"}, "unsupported adaptation kind")
        _need(bool(self.run_id) and self.max_steps > 0 and self.learning_rate > 0 and self.per_device_train_batch_size > 0 and self.per_device_eval_batch_size > 0 and self.gradient_accumulation_steps > 0, "invalid adaptation hyperparameters")
        _need(Path(self.model_path).is_dir() and Path(self.initial_path).exists(), "pinned initial model artifact is missing")
        expected_train = DECODER_RUN_ROOT if self.kind == "decoder_direct" else ENCODER_RUN_ROOT
        expected_eval = DECODER_EVAL_ROOT if self.kind == "decoder_direct" else ENCODER_EVAL_ROOT
        _need(Path(self.output_dir).is_absolute() and Path(self.output_dir).parent == expected_train, "training output must be a direct child of the proper ignored output root")
        _need(Path(self.evaluation_output_dir).is_absolute() and Path(self.evaluation_output_dir).parent == expected_eval, "evaluation output must be a direct child of the proper ignored output root")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _need(not path.exists(), "refusing to overwrite a completion artifact")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configure_wandb(cfg: Config) -> None:
    """Keep telemetry aggregate-only and route both adaptation runs together."""
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    os.environ["WANDB_RUN_NAME"] = cfg.run_id


def _train_decoder(cfg: Config, train_rows: list[RestrictedRow]) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, revision=cfg.tokenizer_revision, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.model_path, revision=cfg.model_revision, local_files_only=True, torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, cfg.initial_path, is_trainable=True)
    examples = []
    for row in train_rows:
        messages = messages_for_sft(row, "direct")
        examples.append({"prompt": messages[:-1], "completion": [messages[-1]]})
    set_seed(cfg.seed)
    _configure_wandb(cfg)
    output = Path(cfg.output_dir)
    args = SFTConfig(
        output_dir=str(output), run_name=cfg.run_id, seed=cfg.seed, max_length=2048,
        max_steps=cfg.max_steps, learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_strategy="steps", logging_steps=5, eval_strategy="no", save_strategy="no",
        packing=False, completion_only_loss=True, assistant_only_loss=False,
        bf16=True, tf32=True, gradient_checkpointing=True, report_to=["wandb"],
        ddp_find_unused_parameters=False, remove_unused_columns=False, dataset_num_proc=1,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(examples), processing_class=tokenizer)
    result = trainer.train()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        adapter = output / "adapter"
        trainer.save_model(str(adapter))
        tokenizer.save_pretrained(str(adapter))
        _write_json(output / "adaptation_complete.json", {
            "status": "completed", "run_id": cfg.run_id, "kind": cfg.kind, "global_step": int(trainer.state.global_step),
            "initial_path": str(Path(cfg.initial_path).resolve()), "initial_path_sha256": _sha256(Path(cfg.initial_path) / "adapter_model.safetensors"),
            "train_records": len(train_rows), "train_sha256": cfg.train_sha256, "validation_sha256": cfg.validation_sha256,
            "train_metrics": {key: float(value) for key, value in result.metrics.items() if isinstance(value, (int, float))},
            "config": asdict(cfg), "privacy": "aggregate_only_no_rows_prompts_essays_ids_predictions_or_model_outputs_persisted",
        })
    trainer.accelerator.wait_for_everyone()


def _encoder_spec_from_initial(metadata: Path) -> EncoderModelSpec:
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    _need(isinstance(payload, dict) and payload.get("status") == "completed", "initial encoder provenance is incomplete")
    raw = payload.get("config")
    _need(isinstance(raw, dict), "initial encoder config missing")
    return EncoderModelSpec.from_mapping({
        "backbone": raw["backbone"], "model_id": raw["model_id"], "revision": raw["model_revision"], "tokenizer_revision": raw["tokenizer_revision"], "model_path": raw["model_path"],
        "pooling": "last_nonpad", "normalize_embeddings": True, "lora_target_modules": raw["lora_target_modules"], "lora_r": raw["lora_r"], "lora_alpha": raw["lora_alpha"], "lora_dropout": raw["lora_dropout"],
        "nv_snapshot_dir": raw["nv_snapshot_dir"], "nv_review": raw["nv_review"],
    })


def _train_encoder(cfg: Config, train_rows: list[RestrictedRow], validation_rows: list[RestrictedRow]) -> None:
    from safetensors.torch import load_model
    from transformers import Trainer, TrainingArguments, set_seed

    metadata = Path(cfg.initial_path)
    spec = _encoder_spec_from_initial(metadata)
    _need(spec.backbone == "qwen3_embedding" and Path(spec.model_path).resolve() == Path(cfg.model_path).resolve(), "encoder initial provenance/model mismatch")
    state_path = metadata.parent / "final_model" / "model.safetensors"
    _need(state_path.is_file(), "initial encoder model state is missing")
    tokenizer = build_encoder_tokenizer(spec)
    model = build_encoder_regressor(spec)
    missing, unexpected = load_model(model, str(state_path), strict=False)
    _need(not missing and not unexpected, "initial encoder state does not match model architecture")
    set_seed(cfg.seed)
    _configure_wandb(cfg)
    output = Path(cfg.output_dir)
    args = TrainingArguments(
        output_dir=str(output), do_train=True, do_eval=False, eval_strategy="no", save_strategy="no",
        max_steps=cfg.max_steps, learning_rate=cfg.learning_rate, per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size, gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_strategy="steps", logging_steps=5, bf16=True, tf32=True, report_to=["wandb"], run_name=cfg.run_id,
        remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=True, ddp_find_unused_parameters=False, seed=cfg.seed, data_seed=cfg.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=build_encoder_dataset(train_rows, tokenizer, 2048), data_collator=encoder_collator(tokenizer))
    result = trainer.train()
    trainer.accelerator.wait_for_everyone()
    # All ranks remain present for the validation gather; only aggregate metrics survive it.
    predicted = trainer.predict(build_encoder_dataset(validation_rows, tokenizer, 2048), metric_key_prefix="validation")
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        final = output / "final_model"
        trainer.save_model(str(final))
        model_state = final / "model.safetensors"
        _need(model_state.is_file(), "encoder adaptation did not save a safe model state")
        targets = [row.score for row in validation_rows]
        values = predicted.predictions[0] if isinstance(predicted.predictions, tuple) else predicted.predictions
        predictions = [{field: min(5.0, max(1.0, float(value))) for field, value in zip(SCORE_FIELDS, row, strict=True)} for row in values]
        metrics = compute_regression_metrics(targets, predictions)
        metrics.update(_macro(metrics))
        _write_json(output / "adaptation_complete.json", {
            "status": "completed", "run_id": cfg.run_id, "kind": cfg.kind, "global_step": int(trainer.state.global_step),
            "initial_path": str(metadata.resolve()), "initial_model_state_sha256": _sha256(state_path), "adapted_model_state_sha256": _sha256(model_state),
            "train_records": len(train_rows), "validation_records": len(validation_rows), "train_sha256": cfg.train_sha256, "validation_sha256": cfg.validation_sha256,
            "train_metrics": {key: float(value) for key, value in result.metrics.items() if isinstance(value, (int, float))}, "validation_metrics": metrics,
            "config": asdict(cfg), "privacy": "aggregate_only_no_rows_prompts_essays_ids_predictions_or_model_outputs_persisted",
        })
        eval_out = Path(cfg.evaluation_output_dir)
        eval_out.mkdir(parents=True, exist_ok=False)
        _write_json(eval_out / "aggregate_metrics.json", {"status": "completed", "run_id": cfg.run_id, "kind": cfg.kind, "source": "eval_validation_after_eval_train_adaptation", "metrics": metrics, "config": asdict(cfg), "privacy": "aggregate_only_no_rows_prompts_essays_ids_predictions_or_model_outputs_persisted"})
    trainer.accelerator.wait_for_everyone()


def _evaluate_decoder(cfg: Config, train_rows: list[RestrictedRow], validation_rows: list[RestrictedRow]) -> None:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    adapter = Path(cfg.output_dir) / "adapter"
    completion = Path(cfg.output_dir) / "adaptation_complete.json"
    _need(adapter.is_dir() and completion.is_file(), "completed decoder adaptation adapter is missing")
    llm = LLM(model=cfg.model_path, revision=cfg.model_revision, dtype="bfloat16", trust_remote_code=False, enable_lora=True,
              tensor_parallel_size=4, max_model_len=2048, gpu_memory_utilization=0.90, enforce_eager=True, max_lora_rank=32)
    output = llm.chat([messages_for_generation(row, "direct") for row in validation_rows], SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256, skip_special_tokens=True), lora_request=LoRARequest("eval_split_adaptation", 1, str(adapter)), use_tqdm=True)
    fallback = score_mean(train_rows)
    predictions, valid = [], []
    for item in output:
        text = item.outputs[0].text if item.outputs else ""
        parsed = parse_decoder_scores(text, "direct")
        valid.append(parsed is not None)
        predictions.append(parsed if parsed is not None else fallback)
    metrics = compute_regression_metrics([row.score for row in validation_rows], predictions)
    metrics.update(_macro(metrics))
    metrics["decoder_parse_failure_rate"] = 1.0 - sum(valid) / len(valid)
    eval_out = Path(cfg.evaluation_output_dir)
    eval_out.mkdir(parents=True, exist_ok=False)
    _write_json(eval_out / "aggregate_metrics.json", {"status": "completed", "run_id": cfg.run_id, "kind": cfg.kind, "source": "eval_validation_after_eval_train_adaptation", "adapter_path": str(adapter.resolve()), "metrics": metrics, "config": asdict(cfg), "privacy": "aggregate_only_no_rows_prompts_essays_ids_predictions_or_model_outputs_persisted"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("train", "evaluate_decoder"))
    args = parser.parse_args()
    cfg = Config.from_json(args.config)
    train_rows = _load_eval_split(TRAIN_PATH, cfg.train_sha256)
    validation_rows = _load_eval_split(VALIDATION_PATH, cfg.validation_sha256)
    if args.stage == "train":
        _need(not Path(cfg.output_dir).exists() and not Path(cfg.evaluation_output_dir).exists(), "adaptation outputs already exist")
        if cfg.kind == "decoder_direct":
            _train_decoder(cfg, train_rows)
        else:
            _train_encoder(cfg, train_rows, validation_rows)
    else:
        _need(cfg.kind == "decoder_direct", "only the decoder has a separate vLLM evaluation stage")
        _evaluate_decoder(cfg, train_rows, validation_rows)


if __name__ == "__main__":
    main()
