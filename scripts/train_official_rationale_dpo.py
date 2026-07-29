#!/usr/bin/env python3
"""Offline TRL DPO continuation for one official rationale SFT adapter."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import sha256_file  # noqa: E402
from mal2026.official_rationale_rl import (  # noqa: E402
    MODEL_ID,
    MODEL_PATH,
    MODEL_REVISION,
    RLSettings,
    TASKS,
    finite_metrics,
    legacy_ablation,
    load_preferences,
    output_fresh,
    validate_preference_report,
    validate_runtime_versions,
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def adapter_hash(model: Any, name: str) -> str:
    digest = sha256()
    count = 0
    marker = f".{name}."
    for parameter_name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if marker in parameter_name:
            digest.update(parameter_name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
            count += 1
    need(count > 0, f"adapter {name} has no parameters")
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def distributed_output_dir(path: Path) -> tuple[Path, int, int]:
    """Reserve one fresh output directory for a torchrun/Accelerate launch."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    need(world_size > 0 and 0 <= rank < world_size, "distributed rank metadata differs")
    resolved = path.resolve()
    need(resolved.is_relative_to(ROOT / "outputs"), "RL aggregate/model output root differs")
    if world_size == 1:
        output_fresh(resolved)
        resolved.mkdir(mode=0o700, parents=True)
        return resolved, rank, world_size
    if rank == 0:
        output_fresh(resolved)
        resolved.mkdir(mode=0o700, parents=True)
    else:
        deadline = time.monotonic() + 120.0
        while not resolved.is_dir() and time.monotonic() < deadline:
            time.sleep(0.05)
        need(resolved.is_dir(), "distributed output reservation timed out")
    return resolved, rank, world_size


def per_rank_gradient_accumulation(configured: int, world_size: int) -> int:
    """Keep the configured global effective batch invariant under DDP."""
    need(configured > 0 and world_size > 0, "gradient accumulation metadata differs")
    need(configured % world_size == 0, "configured gradient accumulation is not divisible by world size")
    return configured // world_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/official_rationale_dpo.v1.json")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--legacy-arm")
    parser.add_argument("--preferences", type=Path)
    parser.add_argument("--preference-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    settings = RLSettings.from_json(args.config)
    validate_runtime_versions()
    need(settings.algorithm == "dpo", "DPO settings required")
    if args.validate_only:
        print(json.dumps({"status": "validated", "trainer": settings.policy["trainer"], "integrated_vllm": settings.runtime["integrated_vllm"]}, sort_keys=True))
        return
    need((args.task is None) != (args.legacy_arm is None), "select exactly one official task or legacy arm")
    task = "bundle" if args.legacy_arm is not None else args.task
    need(task in TASKS and args.preferences is not None and args.preference_report is not None and args.output_dir is not None, "DPO arguments are incomplete")
    gate = settings.gate_evidence()
    output, rank, world_size = distributed_output_dir(args.output_dir)
    rows, preference_provenance = load_preferences(args.preferences, task)
    preference_report = validate_preference_report(args.preference_report, args.preferences, task, settings, gate)
    if args.train_limit is not None:
        need(0 < args.train_limit <= len(rows), "DPO train limit differs")
        rows = rows[:args.train_limit]

    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError("official DPO requires .venv-standard") from exc

    seed = int(settings.policy["seed"])
    set_seed(seed)
    legacy = legacy_ablation(settings, args.legacy_arm) if args.legacy_arm is not None else None
    model_id = str(legacy["model_id"]) if legacy else MODEL_ID
    model_revision = str(legacy["model_revision"]) if legacy else MODEL_REVISION
    model_path = Path(str(legacy["model_path"])) if legacy else MODEL_PATH
    warm_start = Path(str(legacy["adapter_path"])) if legacy else Path(settings.warm_starts[task])
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        need(tokenizer.eos_token is not None, "DPO tokenizer lacks PAD/EOS")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=model_revision, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, warm_start, adapter_name="default", is_trainable=True)
    model.set_adapter("default")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".default." in name)
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)
    need(any(parameter.requires_grad for parameter in model.parameters()), "DPO adapter is not trainable")
    training_args = DPOConfig(
        output_dir=str(output),
        run_name=output.name,
        seed=seed,
        max_length=int(settings.policy["max_length"]),
        truncation_mode="keep_end",
        loss_type=str(settings.policy["loss_type"]),
        beta=float(settings.policy["beta"]),
        learning_rate=float(settings.policy["learning_rate"]),
        num_train_epochs=float(settings.policy["num_train_epochs"]),
        max_steps=args.max_steps,
        per_device_train_batch_size=int(settings.policy["per_device_train_batch_size"]),
        gradient_accumulation_steps=per_rank_gradient_accumulation(
            int(settings.policy["gradient_accumulation_steps"]), world_size
        ),
        bf16=False,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=1 if args.max_steps == 1 else 10,
        save_strategy="no" if args.max_steps == 1 else "epoch",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        logging_nan_inf_filter=False,
        disable_dropout=True,
        ddp_find_unused_parameters=False,
    )
    trainer = DPOTrainer(model=model, args=training_args, train_dataset=Dataset.from_list(rows), processing_class=tokenizer)
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    need("ref" in unwrapped.peft_config, "DPO frozen warm-start reference adapter is absent")
    for name, parameter in unwrapped.named_parameters():
        if ".ref." in name:
            parameter.requires_grad_(False)
    reference_before = adapter_hash(unwrapped, "ref")
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        reference_after = adapter_hash(unwrapped, "ref")
        need(reference_before == reference_after, "DPO reference adapter changed")
        adapter = output / "adapter"
        unwrapped.save_pretrained(str(adapter), selected_adapters=["default"], safe_serialization=True)
        tokenizer.save_pretrained(str(adapter))
        need((adapter / "adapter_config.json").is_file(), "DPO adapter export failed")
        payload = {
            "schema_version": "mal2026-official-rationale-dpo-complete-v1",
            "status": "completed",
            "run_id": output.name,
            "task": task,
            "legacy_arm": args.legacy_arm,
            "classification": None if legacy is None else legacy["classification"],
            "contract_shift": None if legacy is None else legacy["contract_shift"],
            "model_id": model_id,
            "model_revision": model_revision,
            "model_path": str(model_path.resolve()),
            "global_step": int(trainer.state.global_step),
            "train_rows": len(rows),
            "distributed_world_size": world_size,
            "configured_global_gradient_accumulation_steps": int(settings.policy["gradient_accumulation_steps"]),
            "per_rank_gradient_accumulation_steps": per_rank_gradient_accumulation(
                int(settings.policy["gradient_accumulation_steps"]), world_size
            ),
            "preference_provenance": preference_provenance,
            "preference_report": preference_report,
            "warm_start_adapter": str(warm_start.resolve()),
            "warm_start_adapter_config_sha256": sha256_file(warm_start / "adapter_config.json"),
            "legacy_completion_sha256": None if legacy is None else legacy["completion_sha256"],
            "reference_adapter_sha256_before": reference_before,
            "reference_adapter_sha256_after": reference_after,
            "contrastive_gate_sha256": gate["directional"]["sha256"],
            "rl_safety_gate_sha256": gate["combined_safety"]["sha256"],
            "config_sha256": sha256_file(args.config),
            "judge_model_sha256": settings.judge["model_sha256"],
            "judge_prompt_sha256": settings.judge["prompt_sha256"],
            "trainer_metrics": finite_metrics([*trainer.state.log_history, trained.metrics]),
            "trl_trainer": "DPOTrainer",
            "offline_preferences": True,
            "split": "train",
            "validation_used_for_preferences_or_training": False,
            "human_or_reference_score_read_or_prompted": False,
            "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_or_predictions",
        }
        atomic_json(output / "training_complete.json", payload)
        print(json.dumps({"status": "completed", "run_id": output.name, "task": task, "legacy_arm": args.legacy_arm, "global_step": trainer.state.global_step}, sort_keys=True))


if __name__ == "__main__":
    main()
