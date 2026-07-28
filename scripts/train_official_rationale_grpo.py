#!/usr/bin/env python3
"""Official score-conditioned rationale GRPO with external vLLM rollouts."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import sha256_file  # noqa: E402
from mal2026.official_rationale_rl import (  # noqa: E402
    ExternalVLLMRollout,
    ExactQ4Reward,
    MODEL_ID,
    MODEL_PATH,
    MODEL_REVISION,
    RLSettings,
    TASKS,
    finite_metrics,
    legacy_ablation,
    mean_history,
    official_train_rows,
    output_fresh,
    validate_policy_attestation,
    validate_q4_attestation,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/official_rationale_grpo.v1.json")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--legacy-arm")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rollout-endpoint")
    parser.add_argument("--rollout-model")
    parser.add_argument("--rollout-attestation", type=Path)
    parser.add_argument("--judge-endpoint", action="append", default=[])
    parser.add_argument("--judge-attestation", type=Path)
    parser.add_argument("--train-limit", type=int, default=1920)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    settings = RLSettings.from_json(args.config)
    validate_runtime_versions()
    need(settings.algorithm == "grpo", "GRPO settings required")
    if args.validate_only:
        print(json.dumps({"status": "validated", "trainer": settings.policy["trainer"], "rollout_backend": settings.policy["rollout_backend"], "integrated_vllm": settings.runtime["integrated_vllm"]}, sort_keys=True))
        return
    need((args.task is None) != (args.legacy_arm is None), "select exactly one official GRPO task or legacy arm")
    task = "bundle" if args.legacy_arm is not None else args.task
    need(task in TASKS and args.output_dir is not None and args.rollout_endpoint and args.rollout_model and args.rollout_attestation and args.judge_endpoint and args.judge_attestation, "GRPO arguments are incomplete")
    gate = settings.gate_evidence()
    legacy = legacy_ablation(settings, args.legacy_arm) if args.legacy_arm is not None else None
    model_id = str(legacy["model_id"]) if legacy else MODEL_ID
    model_revision = str(legacy["model_revision"]) if legacy else MODEL_REVISION
    model_path = Path(str(legacy["model_path"])) if legacy else MODEL_PATH
    warm_start = Path(str(legacy["adapter_path"])) if legacy else Path(settings.warm_starts[task])
    validate_policy_attestation(
        args.rollout_attestation, args.rollout_endpoint, {task: args.rollout_model},
        expected_model_id=model_id, expected_model_revision=model_revision,
    )
    validate_q4_attestation(args.judge_attestation, args.judge_endpoint, settings.judge["prompt_sha256"])
    output = output_fresh(args.output_dir)
    rows, input_provenance = official_train_rows(task, args.train_limit)

    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("official GRPO requires .venv-standard") from exc

    output.mkdir(mode=0o700, parents=True)
    seed = int(settings.policy["seed"])
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        need(tokenizer.eos_token is not None, "GRPO tokenizer lacks PAD/EOS")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=model_revision, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, warm_start, adapter_name="default", is_trainable=True)
    model.set_adapter("default")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".default." in name)
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)
    need(any(parameter.requires_grad for parameter in model.parameters()), "GRPO adapter is not trainable")
    max_steps = int(settings.policy["max_steps"] if args.max_steps is None else args.max_steps)
    training_args = GRPOConfig(
        output_dir=str(output),
        run_name=output.name,
        seed=seed,
        max_steps=max_steps,
        num_train_epochs=float(settings.policy["num_train_epochs"]),
        learning_rate=float(settings.policy["learning_rate"]),
        per_device_train_batch_size=int(settings.policy["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(settings.policy["gradient_accumulation_steps"]),
        generation_batch_size=int(settings.policy["generation_batch_size"]),
        num_generations=int(settings.policy["num_generations"]),
        max_completion_length=int(settings.policy["max_completion_tokens"]),
        temperature=float(settings.policy["sampling_temperature"]),
        top_p=float(settings.policy["sampling_top_p"]),
        top_k=0,
        beta=float(settings.policy["beta"]),
        loss_type=str(settings.policy["loss_type"]),
        scale_rewards=str(settings.policy["scale_rewards"]),
        num_iterations=1,
        epsilon=0.2,
        use_vllm=False,
        bf16=False,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        logging_nan_inf_filter=False,
        disable_dropout=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        dataloader_drop_last=False,
        log_completions=False,
    )
    need(training_args.use_vllm is False, "integrated TRL vLLM must remain disabled")
    rollout = ExternalVLLMRollout(settings, task, args.rollout_endpoint, args.rollout_model, tokenizer, output / "rollout-sync")
    reward = ExactQ4Reward(settings, task, args.judge_endpoint, settings.judge["model_alias"])
    dataset_rows = [{
        "prompt": row["prompt"],
        "source_key": row["source_key"],
        "prompt_text": row["prompt_text"],
        "essay_text": row["essay_text"],
        "scores": row["scores"],
        "frozen_rationales": row["frozen_rationales"],
    } for row in rows]
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=training_args, train_dataset=Dataset.from_list(dataset_rows), processing_class=tokenizer, rollout_func=rollout)
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    need("ref" in unwrapped.peft_config, "GRPO frozen warm-start reference adapter is absent")
    for name, parameter in unwrapped.named_parameters():
        if ".ref." in name:
            parameter.requires_grad_(False)
    reference_before = adapter_hash(unwrapped, "ref")
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        reference_after = adapter_hash(unwrapped, "ref")
        need(reference_before == reference_after, "GRPO reference adapter changed")
        history = [item for item in trainer.state.log_history if isinstance(item, dict)]
        reward_summary = reward.aggregate()
        zero_variance = mean_history(history, "frac_reward_zero_std")
        hard_gates = {
            "parse_valid_rate": reward_summary["parse_valid_rate"] is not None and reward_summary["parse_valid_rate"] >= float(settings.reward["parse_valid_rate_min"]),
            "reward_variance": zero_variance is not None and zero_variance <= float(settings.reward["max_zero_variance_group_fraction"]),
            "reference_unchanged": reference_before == reference_after,
            "judge_call_accounting": reward_summary["judge_calls"] == reward_summary["parse_valid"],
        }
        if not all(hard_gates.values()):
            atomic_json(output / "training_failed_gate.json", {
                "schema_version": "mal2026-official-rationale-grpo-failed-v1",
                "status": "failed_gates",
                "producer_status": "failed_gates",
                "handoff_eligible": False,
                "run_id": output.name,
                "task": task,
                "legacy_arm": args.legacy_arm,
                "classification": None if legacy is None else legacy["classification"],
                "contract_shift": None if legacy is None else legacy["contract_shift"],
                "model_id": model_id,
                "model_revision": model_revision,
                "model_path": str(model_path.resolve()),
                "model_config_sha256": sha256_file(model_path / "config.json"),
                "warm_start_adapter": str(warm_start.resolve()),
                "warm_start_adapter_model_sha256": sha256_file(warm_start / "adapter_model.safetensors"),
                "legacy_completion_sha256": None if legacy is None else legacy["completion_sha256"],
                "hard_gates": hard_gates,
                "reward_summary": reward_summary,
                "mean_zero_variance_group_fraction": zero_variance,
                "contrastive_gate_sha256": gate["directional"]["sha256"],
                "rl_safety_gate_sha256": gate["combined_safety"]["sha256"],
                "judge_model_sha256": settings.judge["model_sha256"],
                "judge_prompt_sha256": settings.judge["prompt_sha256"],
                "split": "train",
                "validation_used": False,
            })
            raise RuntimeError("GRPO post-training hard gates failed")
        adapter = output / "adapter"
        unwrapped.save_pretrained(str(adapter), selected_adapters=["default"], safe_serialization=True)
        tokenizer.save_pretrained(str(adapter))
        need((adapter / "adapter_config.json").is_file(), "GRPO adapter export failed")
        payload = {
            "schema_version": "mal2026-official-rationale-grpo-complete-v1",
            "status": "completed",
            "producer_status": "completed",
            "handoff_eligible": True,
            "run_id": output.name,
            "task": task,
            "legacy_arm": args.legacy_arm,
            "classification": None if legacy is None else legacy["classification"],
            "contract_shift": None if legacy is None else legacy["contract_shift"],
            "model_id": model_id,
            "model_revision": model_revision,
            "model_path": str(model_path.resolve()),
            "model_config_sha256": sha256_file(model_path / "config.json"),
            "global_step": int(trainer.state.global_step),
            "train_rows": len(rows),
            "input_provenance": input_provenance,
            "warm_start_adapter": str(warm_start.resolve()),
            "warm_start_adapter_config_sha256": sha256_file(warm_start / "adapter_config.json"),
            "warm_start_adapter_model_sha256": sha256_file(warm_start / "adapter_model.safetensors"),
            "legacy_completion_path": None if legacy is None else str(Path(str(legacy["completion_path"])).resolve()),
            "legacy_completion_sha256": None if legacy is None else legacy["completion_sha256"],
            "output_adapter": str(adapter.resolve()),
            "output_adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
            "output_adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
            "reference_adapter_sha256_before": reference_before,
            "reference_adapter_sha256_after": reference_after,
            "contrastive_gate_sha256": gate["directional"]["sha256"],
            "rl_safety_gate_sha256": gate["combined_safety"]["sha256"],
            "config_sha256": sha256_file(args.config),
            "judge_model_sha256": settings.judge["model_sha256"],
            "judge_prompt_sha256": settings.judge["prompt_sha256"],
            "trainer_metrics": finite_metrics([*history, trained.metrics]),
            "reward_summary": reward_summary,
            "mean_zero_variance_group_fraction": zero_variance,
            "hard_gates": hard_gates,
            "rollout": rollout.aggregate(),
            "trl_trainer": "GRPOTrainer",
            "rollout_backend": "external_vllm_http_rollout_func",
            "integrated_vllm": False,
            "split": "train",
            "validation_used_for_reward_or_training": False,
            "human_or_reference_score_read_or_prompted": False,
            "training_contract": "public_spec_aligned_score_conditioned_rationale_only_descriptive_no_improvement_advice",
            "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_or_predictions",
        }
        atomic_json(output / "training_complete.json", payload)
        print(json.dumps({"status": "completed", "run_id": output.name, "task": args.task, "global_step": trainer.state.global_step}, sort_keys=True))


if __name__ == "__main__":
    main()
