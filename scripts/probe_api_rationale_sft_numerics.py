#!/usr/bin/env python3
"""One-batch numerical health probe for a declared rationale SFT setup.

This is a recovery diagnostic, not an additional training experiment.  It uses
two real train candidates in memory, never reads candidate scores, persists
only aggregate finiteness indicators, and removes the temporary Trainer root.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_tensor(tensor: Any) -> bool:
    import torch
    return bool(torch.isfinite(tensor).all().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-key", required=True, choices=("ax4_light", "phi4_mini", "midm2_base"))
    parser.add_argument("--report", required=True)
    parser.add_argument("--optimizer-steps", type=int, default=1)
    parser.add_argument("--trainable-fp32", action="store_true")
    parser.add_argument("--non-reentrant-checkpointing", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--base-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    need(1 <= args.optimizer_steps <= 8, "optimizer-step probe budget must be in 1..8")
    need(not (args.non_reentrant_checkpointing and args.disable_gradient_checkpointing), "non-reentrant mode requires gradient checkpointing")
    need(0.0 < args.learning_rate <= 2e-5, "learning rate must be positive and no greater than the failed setting")
    model_path, report_path = Path(args.model_path).resolve(), Path(args.report).resolve()
    need(model_path.is_dir(), "model path is unavailable")
    need(report_path.parent == (ROOT / "outputs" / "aggregate-reports").resolve(), "report must be aggregate-only under ignored outputs/aggregate-reports")
    need(not report_path.exists(), "refusing to overwrite a numerical-probe report")

    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer
    from mal2026.api_rationale_data import candidate_sft_examples
    from mal2026.api_rationale_sft import _lora_targets, _promote_trainable_lora_parameters

    temporary = ROOT / "outputs" / "api-rationale-sft-v1" / f"numerical-probe-{args.base_key}-temporary"
    need(not temporary.exists(), "numerical-probe temporary root already exists")
    payload: dict[str, Any] = {
        "schema_version": "mal2026-api-rationale-sft-numerical-probe-v1",
        "base_key": args.base_key,
        "input_records": 2 * 8 * args.optimizer_steps,
        "candidate_scores_read_or_prompted": False,
        "raw_text_persisted": False,
        "gradient_checkpointing": not args.disable_gradient_checkpointing,
        "non_reentrant_checkpointing": bool(args.non_reentrant_checkpointing),
        "dtype": args.base_dtype,
        "trainable_adapter_fp32": bool(args.trainable_fp32),
        "learning_rate": args.learning_rate,
    }
    try:
        set_seed(2026072108)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, use_fast=True)
        if tokenizer.pad_token is None:
            need(tokenizer.eos_token is not None, "tokenizer lacks pad and EOS tokens")
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False, dtype=args.base_dtype, low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        targets = _lora_targets(model)
        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(
                output_dir=str(temporary), max_length=3072, packing=False, completion_only_loss=True,
                assistant_only_loss=False, learning_rate=args.learning_rate, per_device_train_batch_size=2,
                gradient_accumulation_steps=8, save_strategy="no", logging_strategy="no",
                bf16=args.base_dtype == "bfloat16", tf32=True,
                gradient_checkpointing=not args.disable_gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False} if args.non_reentrant_checkpointing else None,
                report_to=[], remove_unused_columns=False, dataset_num_proc=1,
                ddp_find_unused_parameters=False, logging_nan_inf_filter=False,
            ),
            train_dataset=Dataset.from_list(candidate_sft_examples("bundle")[: 2 * 8 * args.optimizer_steps]), processing_class=tokenizer,
            peft_config=LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, lora_dropout=0.05, target_modules=targets, bias="none"),
        )
        if args.trainable_fp32:
            payload["adapter_precision"] = _promote_trainable_lora_parameters(trainer.model)
        trainer.model.train()
        trainer.create_optimizer()
        iterator = iter(trainer.get_train_dataloader())
        records: list[dict[str, Any]] = []
        for step in range(args.optimizer_steps):
            trainer.optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            cumulative_gradient_finite: list[bool] = []
            for _ in range(8):
                batch = trainer._prepare_inputs(next(iterator))
                loss = trainer.compute_loss(trainer.model, batch)
                losses.append(float(loss.detach().float().item()))
                trainer.accelerator.backward(loss / 8)
                current = [parameter.grad for parameter in trainer.model.parameters() if parameter.requires_grad and parameter.grad is not None]
                cumulative_gradient_finite.append(bool(current) and all(finite_tensor(gradient) for gradient in current))
            gradients = [parameter.grad for parameter in trainer.model.parameters() if parameter.requires_grad and parameter.grad is not None]
            norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), trainer.args.max_grad_norm)
            gradients_after_clip = [parameter.grad for parameter in trainer.model.parameters() if parameter.requires_grad and parameter.grad is not None]
            trainer.optimizer.step()
            trainable = [parameter for parameter in trainer.model.parameters() if parameter.requires_grad]
            trainable_max_abs = max(float(parameter.detach().abs().float().max().item()) for parameter in trainable)
            records.append({
                "step": step + 1,
                "microbatch_losses_finite": all(math.isfinite(loss) for loss in losses),
                "microbatch_loss_min": min(losses),
                "microbatch_loss_max": max(losses),
                "cumulative_gradient_finite_after_each_microbatch": cumulative_gradient_finite,
                "trainable_gradient_tensors": len(gradients),
                "gradients_finite_before_clip": bool(gradients) and all(finite_tensor(gradient) for gradient in gradients),
                "gradient_norm_before_clip": float(norm.detach().float().item()),
                "gradient_norm_finite_before_clip": math.isfinite(float(norm.detach().float().item())),
                "gradients_finite_after_clip": bool(gradients_after_clip) and all(finite_tensor(gradient) for gradient in gradients_after_clip),
                "trainable_parameters_finite_after_step": all(finite_tensor(parameter) for parameter in trainable),
                "trainable_parameter_max_abs_after_step": trainable_max_abs,
            })
        payload["optimizer_steps"] = records
        payload["lora_targets"] = targets
        health_fields = (
            "microbatch_losses_finite", "gradients_finite_before_clip", "gradient_norm_finite_before_clip",
            "gradients_finite_after_clip", "trainable_parameters_finite_after_step",
        )
        payload["status"] = "completed" if all(all(record[field] for field in health_fields) for record in records) else "failed_numerical_gate"
    except Exception as exc:
        payload["status"] = "failed_exception"
        payload["exception_type"] = type(exc).__name__
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    report_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
