#!/usr/bin/env python3
"""Aggregate-only diagnosis of the failed full-tune Qwen3 regression arm."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from mal2026.qwen3_full_aihub_then_lora import (
    AXES,
    MODEL_PATH,
    MODEL_REVISION,
    FullRationaleConfig,
    _rationale_collator,
    _rationale_dataset,
    _rationale_examples,
    build_full_warm_lora,
    rationale_checkpoint_dir,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "outputs/qwen3-full-aihub-v1/20260726-009/configs/rationale-full.json"
OUTPUT_ROOT = ROOT / "outputs/qwen3-full-diagnostics"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "min": float(value.min()),
        "max": float(value.max()),
        "l2_norm": float(torch.linalg.vector_norm(value)),
    }


def prediction_stats(truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, axis in enumerate(AXES):
        observed, predicted = truth[:, index], predictions[:, index]
        result[axis] = {
            "raw_mean": float(predicted.mean()),
            "raw_std": float(predicted.std()),
            "raw_min": float(predicted.min()),
            "raw_max": float(predicted.max()),
            "unique_raw_values": int(np.unique(predicted).size),
            "fraction_below_1": float(np.mean(predicted < 1.0)),
            "fraction_above_5": float(np.mean(predicted > 5.0)),
            "raw_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
            "clipped_rmse": float(np.sqrt(np.mean((np.clip(predicted, 1.0, 5.0) - observed) ** 2))),
            "truth_mean": float(observed.mean()),
            "truth_std": float(observed.std()),
        }
    result["three_axis_macro_raw_rmse"] = float(np.mean([result[axis]["raw_rmse"] for axis in AXES]))
    result["three_axis_macro_clipped_rmse"] = float(np.mean([result[axis]["clipped_rmse"] for axis in AXES]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    need(output.parent == OUTPUT_ROOT.resolve() and not output.exists(), "diagnostic output freshness differs")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, "diagnostic requires exactly one visible GPU")
    output.mkdir(parents=True)

    config = FullRationaleConfig.from_json(CONFIG, require_fresh_output=False)
    full_state = Path(config.full_model_state_path)
    epoch4 = rationale_checkpoint_dir(Path(config.output_dir), 4) / "trainable_model.safetensors"
    with safe_open(str(full_state), framework="pt", device="cpu") as handle:
        full_weight = handle.get_tensor("regression_head.weight")[:3].clone()
        full_bias = handle.get_tensor("regression_head.bias")[:3].clone()
    trained = load_file(str(epoch4), device="cpu")
    trained_weight = trained["regression_head.weight"]
    trained_bias = trained["regression_head.bias"]
    lora_norms = [float(torch.linalg.vector_norm(tensor.float())) for name, tensor in trained.items() if "lora_" in name]
    head = {
        "full_refit_weight": tensor_stats(full_weight),
        "full_refit_bias": tensor_stats(full_bias),
        "epoch4_weight": tensor_stats(trained_weight),
        "epoch4_bias": tensor_stats(trained_bias),
        "epoch4_minus_refit_weight": tensor_stats(trained_weight - full_weight.to(trained_weight.dtype)),
        "epoch4_minus_refit_bias": tensor_stats(trained_bias - full_bias.to(trained_bias.dtype)),
        "lora": {"tensor_count": len(lora_norms), "sum_l2_norm": float(sum(lora_norms)), "max_l2_norm": float(max(lora_norms))},
    }

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    items = _rationale_examples("validation", 400)
    dataset = _rationale_dataset(items, tokenizer, config.max_length)
    model = build_full_warm_lora(config)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(output / "trainer"), do_train=False, do_eval=False, per_device_eval_batch_size=8, bf16=True, tf32=True, report_to=[], remove_unused_columns=False),
        data_collator=_rationale_collator(tokenizer),
    )

    def predict() -> np.ndarray:
        raw = trainer.predict(dataset).predictions
        if isinstance(raw, tuple):
            raw = raw[0]
        values = np.asarray(raw, dtype=np.float64)
        need(values.shape == (400, 3) and bool(np.isfinite(values).all()), "diagnostic predictions differ")
        return values

    truth = np.asarray([item["labels"] for item in items], dtype=np.float64)
    initial = predict()
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    need(set(trained) == trainable_names, "diagnostic checkpoint tensor names differ")
    incompatible = model.load_state_dict(trained, strict=False)
    need(not incompatible.unexpected_keys and not (trainable_names & set(incompatible.missing_keys)), "diagnostic checkpoint load differs")
    epoch4_predictions = predict()
    report = {
        "schema_version": "mal2026-qwen3-full-collapse-diagnostic-v1",
        "status": "completed",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": f"CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-standard/bin/python scripts/{Path(__file__).name} --output {output.relative_to(ROOT)}",
        "gpu_scope": [0],
        "records": 400,
        "score_fields": list(AXES),
        "average_target_used": False,
        "full_state_sha256": file_sha(full_state),
        "epoch4_trainable_state_sha256": file_sha(epoch4),
        "head_parameter_stats": head,
        "prediction_stats": {
            "full_refit_before_rationale_lora": prediction_stats(truth, initial),
            "after_rationale_lora_epoch4": prediction_stats(truth, epoch4_predictions),
        },
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
    }
    report_path = output / "aggregate_prediction_diagnostics.json"
    need(not report_path.exists(), "diagnostic report exists")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
