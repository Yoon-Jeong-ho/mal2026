#!/usr/bin/env python3
"""Train one score-blind rationale policy with train-only exact-Q4 DPO pairs."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import sha256_file  # noqa: E402
from mal2026.api_rationale_sft import _promote_trainable_lora_parameters, _template_provenance  # noqa: E402
from mal2026.rationale_pipeline_prompts import routing  # noqa: E402


OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-dpo-v1"
Q4_MODEL_SHA = "b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def world() -> tuple[int, int]:
    size, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    need(size > 0 and 0 <= rank < size, "DPO distributed metadata differs")
    return size, rank


def reserve(path: Path) -> tuple[int, int]:
    size, rank = world()
    resolved = path.resolve()
    need(resolved.parent == OUTPUT_PARENT.resolve(), "DPO output parent differs")
    if rank == 0:
        need(not resolved.exists(), "DPO output must be fresh")
        resolved.mkdir(parents=True, mode=0o700)
    else:
        deadline = time.monotonic() + 120
        while not resolved.is_dir() and time.monotonic() < deadline:
            time.sleep(0.05)
        need(resolved.is_dir(), "DPO output reservation timed out")
    return size, rank


def adapter_hash(model: Any, adapter: str) -> str:
    digest = sha256(); count = 0; marker = f".{adapter}."
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if marker in name:
            digest.update(name.encode()); digest.update(parameter.detach().cpu().contiguous().numpy().tobytes()); count += 1
    need(count > 0, f"DPO adapter unavailable: {adapter}")
    return digest.hexdigest()


def finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = float(value); need(math.isfinite(parsed), f"non-finite DPO metric: {key}"); result[str(key)] = parsed
    need("train_loss" in result, "DPO train loss unavailable")
    return result


def read_preferences(path: Path, report_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    need(path.is_file() and path.resolve().is_relative_to((ROOT / "data/processed/restricted").resolve()), "DPO preferences must be restricted")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    need(report.get("schema_version") == "mal2026-rationale-pipeline-dpo-preferences-aggregate-v1" and report.get("status") == "completed", "DPO preference report differs")
    need(report.get("preferences_sha256") == sha256_file(path) and report.get("split") == "train" and report.get("validation_used") is False, "DPO preference provenance differs")
    need(report.get("judge_prompt_sha256") == routing()["rationale_reward_and_quality_judge"]["source_file_sha256"], "DPO judge prompt differs")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    need(len(rows) == int(report["preference_pairs"]), "DPO preference population differs")
    trainer_rows = []
    for row in rows:
        need(set(row) == {"prompt", "chosen", "rejected", "metadata"}, "DPO row fields differ")
        serialized_prompt = json.dumps(row["prompt"], ensure_ascii=False)
        need("predicted_score" not in serialized_prompt and "reference_scores_integer" not in serialized_prompt, "score leaked into DPO policy prompt")
        metadata = row["metadata"]
        need(int(metadata["chosen_judge_total"]) > int(metadata["rejected_judge_total"]), "DPO preference direction differs")
        trainer_rows.append({key: row[key] for key in ("prompt", "chosen", "rejected")})
    return trainer_rows, report


def token_audit(rows: list[dict[str, Any]], tokenizer: Any, limit: int) -> dict[str, Any]:
    maximum = 0
    for row in rows:
        for key in ("chosen", "rejected"):
            rendered = tokenizer.apply_chat_template(
                [*row["prompt"], *row[key]], tokenize=False, add_generation_prompt=False
            )
            length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            maximum = max(maximum, length)
            need(length <= limit, "DPO example would be truncated")
    return {"records": len(rows), "maximum": maximum, "limit": limit, "truncated": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--warm-start-adapter", type=Path, required=True)
    parser.add_argument("--warm-start-completion", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--preference-report", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--train-limit", type=int)
    args = parser.parse_args()
    setproctitle(f"mal2026:rationale-dpo:{args.candidate_key}:rank{os.environ.get('RANK', '0')}"[:255])
    output = OUTPUT_PARENT / args.run_id
    size, rank = reserve(output)
    need((size, os.environ.get("CUDA_VISIBLE_DEVICES")) == ((1, "0") if args.max_steps == 1 else (4, "0,1,2,3")), "DPO GPU scope differs")
    need((args.base_model / "config.json").is_file() and (args.warm_start_adapter / "adapter_model.safetensors").is_file(), "DPO model artifact unavailable")
    completion = json.loads(args.warm_start_completion.read_text(encoding="utf-8"))
    need(completion.get("status") == "completed" and completion.get("human_or_reference_score_read_or_prompted") is False, "DPO warm-start completion differs")
    rows, preference_report = read_preferences(args.preferences, args.preference_report)
    if args.train_limit is not None:
        need(0 < args.train_limit <= len(rows), "DPO train limit differs"); rows = rows[:args.train_limit]

    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("rationale DPO requires .venv-standard") from exc
    seed = 2026080704; set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        need(tokenizer.eos_token is not None, "DPO tokenizer lacks PAD/EOS"); tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    template = _template_provenance(tokenizer)
    audit = token_audit(rows, tokenizer, 3072)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=True, trust_remote_code=False, dtype=torch.float32, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, args.warm_start_adapter, adapter_name="default", is_trainable=True)
    model.set_adapter("default")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".default." in name)
    adapter_precision = _promote_trainable_lora_parameters(model)
    need(any(parameter.requires_grad for parameter in model.parameters()), "DPO has no trainable adapter")
    training_args = DPOConfig(
        output_dir=str(output), run_name=args.run_id, seed=seed, data_seed=seed,
        max_length=3072, truncation_mode="keep_end", loss_type="sigmoid", beta=0.1,
        learning_rate=1e-6, weight_decay=0.0, warmup_ratio=0.03, lr_scheduler_type="cosine",
        num_train_epochs=1.0, max_steps=args.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=1 if args.max_steps == 1 else 4,
        bf16=False, tf32=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps", logging_steps=1 if args.max_steps == 1 else 5,
        save_strategy="no" if args.max_steps == 1 else "epoch", save_total_limit=1,
        report_to=[], remove_unused_columns=False, logging_nan_inf_filter=False,
        disable_dropout=True, ddp_find_unused_parameters=False,
    )

    class Guard(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
            for key, value in (logs or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    need(math.isfinite(float(value)), f"non-finite DPO log {key} at step {state.global_step}")
            return control

    trainer = DPOTrainer(model=model, args=training_args, train_dataset=Dataset.from_list(rows), processing_class=tokenizer, callbacks=[Guard()])
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    need("ref" in unwrapped.peft_config, "DPO frozen warm-start reference adapter is absent")
    for name, parameter in unwrapped.named_parameters():
        if ".ref." in name:
            parameter.requires_grad_(False)
    reference_before = adapter_hash(unwrapped, "ref")
    trained = trainer.train(); trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None; failed = False
    if trainer.is_world_process_zero():
        try:
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            reference_after = adapter_hash(unwrapped, "ref")
            need(reference_after == reference_before, "DPO reference adapter changed")
            if args.max_steps != 1:
                adapter = output / "adapter"
                unwrapped.save_pretrained(str(adapter), selected_adapters=["default"], safe_serialization=True)
                tokenizer.save_pretrained(str(adapter))
                need((adapter / "adapter_model.safetensors").is_file(), "DPO adapter export failed")
            payload = {
                "schema_version": "mal2026-rationale-pipeline-dpo-complete-v1", "status": "completed",
                "run_id": args.run_id, "candidate_key": args.candidate_key, "split": "train",
                "global_step": int(trainer.state.global_step), "train_rows": len(rows), "world_size": size,
                "model_path": str(args.base_model.resolve()), "model_config_sha256": sha256_file(args.base_model / "config.json"),
                "warm_start_adapter": str(args.warm_start_adapter.resolve()),
                "warm_start_adapter_sha256": sha256_file(args.warm_start_adapter / "adapter_model.safetensors"),
                "warm_start_completion_sha256": sha256_file(args.warm_start_completion),
                "preferences_sha256": sha256_file(args.preferences), "preference_report_sha256": sha256_file(args.preference_report),
                "preference_pairs_full": preference_report["preference_pairs"],
                "judge_prompt_sha256": preference_report["judge_prompt_sha256"], "judge_model_sha256": Q4_MODEL_SHA,
                "trainer": "trl.DPOTrainer", "loss_type": "sigmoid", "beta": 0.1,
                "metrics": finite_metrics(trained.metrics), "token_audit": audit, "template": template,
                "adapter_precision": adapter_precision, "reference_adapter_sha256_before": reference_before,
                "reference_adapter_sha256_after": reference_after,
                "scores_in_policy_prompt": False, "validation_used": False, "average_used": False,
                "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights",
            }
            atomic_json(output / ("smoke_complete.json" if args.max_steps == 1 else "training_complete.json"), payload)
        except Exception:
            failed = True
    state: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state, src=0)
    if state[0] or not isinstance(state[1], dict):
        raise RuntimeError("DPO completion persistence failed")
    if rank == 0:
        print(json.dumps({"status": "completed", "run_id": args.run_id, "global_step": state[1]["global_step"], "train_rows": len(rows)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
