#!/usr/bin/env python3
"""Run a bounded score-blind GRPO pilot with exact-Q4 rewards."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping as MappingABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import math
from numbers import Integral
import os
from pathlib import Path
import shutil
import statistics
import sys
from threading import Lock
from typing import Any, Mapping
from urllib.request import Request, urlopen

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from mal2026.api_rationale_data import load_writing_rows, sha256_file  # noqa: E402
from mal2026.api_rationale_sft import _promote_trainable_lora_parameters, _template_provenance  # noqa: E402
from mal2026.official_rationale_rl import judge_total, q4_score  # noqa: E402
from mal2026.rationale_pipeline_prompts import AXES, judge_participant, rationale_messages, rationale_output, routing  # noqa: E402
from generate_rationale_pipeline_outputs_vllm import schema as rationale_schema  # noqa: E402


OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-grpo-v1"
JUDGE_PROMPT = ROOT / "llm_as_judge.txt"
JUDGE_PROMPT_SHA = "91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d"


def need(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(message)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp"); temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(path)


def adapter_hash(model: Any, name: str) -> str:
    digest = sha256(); count = 0; marker = f".{name}."
    for parameter_name, parameter in sorted(model.named_parameters()):
        if marker in parameter_name: digest.update(parameter_name.encode()); digest.update(parameter.detach().cpu().contiguous().numpy().tobytes()); count += 1
    need(count > 0, f"GRPO adapter absent: {name}"); return digest.hexdigest()


def http_json(endpoint: str, body: Mapping[str, Any]) -> dict[str, Any]:
    wire = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(); request = Request(endpoint.rstrip("/") + "/v1/chat/completions", data=wire, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=600) as response: value = json.loads(response.read().decode())
    need(isinstance(value, dict), "GRPO rollout response differs"); return value


class Rollout:
    def __init__(self, endpoint: str, alias: str, tokenizer: Any, sync_root: Path, seed: int):
        self.endpoint, self.alias, self.tokenizer, self.sync_root, self.seed = endpoint.rstrip("/"), alias, tokenizer, sync_root, seed
        self.last_step: int | None = None; self.active: Path | None = None; self.counts: Counter[str] = Counter()
    def sync(self, trainer: Any, step: int) -> None:
        accelerator = trainer.accelerator; accelerator.wait_for_everyone()
        need(accelerator.num_processes == 1 and accelerator.is_main_process, "GRPO policy topology differs")
        self.sync_root.mkdir(parents=True, exist_ok=True); snapshot = self.sync_root / f"step-{step:04d}"; need(not snapshot.exists(), "GRPO rollout snapshot exists")
        accelerator.unwrap_model(trainer.model).save_pretrained(str(snapshot), selected_adapters=["default"], safe_serialization=True)
        wire = json.dumps({"lora_name": self.alias, "lora_path": str(snapshot.resolve()), "load_inplace": True}, separators=(",", ":")).encode(); request = Request(self.endpoint + "/v1/load_lora_adapter", data=wire, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=180) as response: need(200 <= response.status < 300, "GRPO rollout adapter reload failed"); response.read()
        if self.active is not None and self.active.exists(): shutil.rmtree(self.active)
        self.active = snapshot; self.counts["syncs"] += 1; accelerator.wait_for_everyone()
    def __call__(self, prompts: list[Any], trainer: Any) -> dict[str, Any]:
        step = int(trainer.state.global_step)
        if step != self.last_step: self.sync(trainer, step); self.last_step = step
        group_size = 4; request_concurrency = 8; requests = []
        for index in range(0, len(prompts), group_size):
            group = prompts[index:index + group_size]; need(len(group) == group_size and all(item == group[0] for item in group), "GRPO prompt group differs")
            rendered = self.tokenizer.apply_chat_template(group[0], tokenize=True, add_generation_prompt=True)
            if isinstance(rendered, MappingABC): rendered = rendered.get("input_ids")
            if hasattr(rendered, "ids"): rendered = list(rendered.ids)
            elif hasattr(rendered, "tolist"): rendered = rendered.tolist()
            if rendered and isinstance(rendered[0], list): rendered = rendered[0]
            need(isinstance(rendered, list) and all(isinstance(token, Integral) for token in rendered), f"GRPO prompt tokens differ: container={type(rendered).__name__}, first={type(rendered[0]).__name__ if rendered else 'empty'}")
            rendered = [int(token) for token in rendered]
            digest = int.from_bytes(sha256(json.dumps(group[0], ensure_ascii=False, sort_keys=True).encode()).digest()[:8], "big")
            body = {"model": self.alias, "messages": group[0], "n": group_size, "temperature": .7, "top_p": .95, "seed": (digest ^ self.seed ^ step ^ index) % (2**31 - 1), "max_tokens": 2000, "response_format": {"type": "json_schema", "json_schema": {"name": "mal2026_score_blind_grpo_v1", "strict": True, "schema": rationale_schema()}}}
            requests.append((index, rendered, body))
        responses = {}
        with ThreadPoolExecutor(max_workers=min(request_concurrency, len(requests))) as pool:
            futures = {pool.submit(http_json, self.endpoint, body): index for index, _, body in requests}
            for future in as_completed(futures): responses[futures[future]] = future.result()
        prompt_ids = []; completion_ids = []
        for index, rendered, _ in requests:
            choices = responses[index].get("choices"); need(isinstance(choices, list) and len(choices) == group_size, "GRPO rollout count differs")
            for choice in choices:
                content = choice.get("message", {}).get("content"); need(isinstance(content, str) and choice.get("finish_reason") == "stop", "GRPO rollout completion differs")
                ids = self.tokenizer.encode(content, add_special_tokens=False); need(ids, "GRPO completion tokens absent")
                prompt_ids.append(rendered); completion_ids.append([*ids, self.tokenizer.eos_token_id]); self.counts[f"finish_{choice.get('finish_reason')}"] += 1
            self.counts["requests"] += 1; self.counts["completions"] += group_size
        self.counts["http_batches"] += 1
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": None}


class Reward:
    __name__ = "mal2026_exact_q4_score_blind_rationale_reward"
    def __init__(self, endpoint: str): self.endpoint = endpoint.rstrip("/"); self.system = JUDGE_PROMPT.read_text(encoding="utf-8"); self.counts: Counter[str] = Counter(); self.values: list[float] = []; self.lock = Lock()
    def __call__(self, completions: list[Any], prompt_text: list[str], essay_text: list[str], scores: list[Mapping[str, float]], **_: Any) -> list[float]:
        need(len(completions) == len(prompt_text) == len(essay_text) == len(scores), "GRPO reward columns differ"); result = [-1.0] * len(completions); tasks = []
        for index, (completion, prompt, essay, score) in enumerate(zip(completions, prompt_text, essay_text, scores, strict=True)):
            raw = completion[0].get("content") if isinstance(completion, list) and len(completion) == 1 and isinstance(completion[0], dict) else completion
            try: rationales = rationale_output(raw)
            except Exception: self.counts["parse_invalid"] += 1; continue
            tasks.append((index, prompt, essay, judge_participant(score, rationales)))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(q4_score, self.endpoint, "qwen36-35b-a3b-q4_k_m", prompt, essay, participant, system_prompt=self.system): index for index, prompt, essay, participant in tasks}
            for future in as_completed(futures):
                result[futures[future]] = judge_total(future.result()) / 12.0; self.counts["parse_valid"] += 1; self.counts["judge_calls"] += 1
        with self.lock: self.values.extend(result); self.counts["completions"] += len(result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--base-model", type=Path, required=True); parser.add_argument("--warm-start-adapter", type=Path, required=True); parser.add_argument("--warm-start-completion", type=Path, required=True); parser.add_argument("--variance-report", type=Path, required=True); parser.add_argument("--rollout-endpoint", required=True); parser.add_argument("--rollout-alias", required=True); parser.add_argument("--judge-endpoint", required=True); parser.add_argument("--max-steps", type=int, default=20); parser.add_argument("--train-limit", type=int, default=160); args = parser.parse_args()
    setproctitle("mal2026:rationale-grpo:score-blind:policy-gpu2")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "2" and args.max_steps == 20 and args.train_limit == 160, "GRPO pilot protocol differs")
    need(sha256_file(JUDGE_PROMPT) == JUDGE_PROMPT_SHA, "GRPO judge prompt differs")
    gate = json.loads(args.variance_report.read_text(encoding="utf-8")); need(gate.get("status") == "passed" and gate.get("zero_variance_group_fraction", 1) <= .8 and gate.get("validation_used") is False, "GRPO variance gate did not pass")
    completion = json.loads(args.warm_start_completion.read_text(encoding="utf-8")); need(completion.get("status") == "completed" and completion.get("human_or_reference_score_read_or_prompted") is False, "GRPO warm-start differs")
    output = OUTPUT_PARENT / args.run_id; need(not output.exists(), "GRPO output must be fresh"); output.mkdir(parents=True)
    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc: raise RuntimeError("GRPO requires .venv-standard") from exc
    seed = 2026080709; set_seed(seed); tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None: need(tokenizer.eos_token is not None, "GRPO tokenizer lacks PAD/EOS"); tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=True, trust_remote_code=False, dtype=torch.float32, low_cpu_mem_usage=True); model.config.use_cache = False
    model = PeftModel.from_pretrained(model, args.warm_start_adapter, adapter_name="default", is_trainable=True); model.set_adapter("default")
    for name, parameter in model.named_parameters(): parameter.requires_grad_(".default." in name)
    precision = _promote_trainable_lora_parameters(model)
    writings = sorted(load_writing_rows("train", include_scores=True), key=lambda row: sha256(f"{seed}:{row.identifier}".encode()).hexdigest())[:args.train_limit]
    rows = [{"prompt": rationale_messages(row.prompt, row.essay), "prompt_text": row.prompt, "essay_text": row.essay, "scores": dict(row.scores or {})} for row in writings]
    need(len(rows) == args.train_limit and all(set(row["scores"]) == set(AXES) for row in rows), "GRPO train population differs")
    settings = GRPOConfig(output_dir=str(output), run_name=args.run_id, seed=seed, max_steps=args.max_steps, num_train_epochs=1., learning_rate=1e-6, per_device_train_batch_size=8, gradient_accumulation_steps=1, generation_batch_size=32, num_generations=4, max_completion_length=2000, temperature=.7, top_p=.95, top_k=0, beta=.02, loss_type="dr_grpo", scale_rewards="none", num_iterations=1, epsilon=.2, use_vllm=False, bf16=False, tf32=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, logging_strategy="steps", logging_steps=1, save_strategy="no", report_to=[], remove_unused_columns=False, logging_nan_inf_filter=False, disable_dropout=True, dataloader_num_workers=0, log_completions=False)
    rollout = Rollout(args.rollout_endpoint, args.rollout_alias, tokenizer, output / "rollout-sync", seed); reward = Reward(args.judge_endpoint)
    class Guard(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **_: Any) -> Any:
            for key, value in (logs or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool): need(math.isfinite(float(value)), f"non-finite GRPO log: {key}")
            return control
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=settings, train_dataset=Dataset.from_list(rows), processing_class=tokenizer, rollout_func=rollout, callbacks=[Guard()])
    unwrapped = trainer.accelerator.unwrap_model(trainer.model); need("ref" in unwrapped.peft_config, "GRPO reference adapter absent")
    for name, parameter in unwrapped.named_parameters():
        if ".ref." in name: parameter.requires_grad_(False)
    before = adapter_hash(unwrapped, "ref"); trained = trainer.train(); unwrapped = trainer.accelerator.unwrap_model(trainer.model); after = adapter_hash(unwrapped, "ref"); need(before == after, "GRPO reference adapter changed")
    history = [row for row in trainer.state.log_history if isinstance(row, dict)]; zero_values = [float(row["frac_reward_zero_std"]) for row in history if isinstance(row.get("frac_reward_zero_std"), (int, float))]; zero_fraction = statistics.fmean(zero_values) if zero_values else None
    valid_rate = reward.counts["parse_valid"] / reward.counts["completions"] if reward.counts["completions"] else 0.; gates = {"parse_valid_rate_gte_0_98": valid_rate >= .98, "zero_variance_group_fraction_lte_0_8": zero_fraction is not None and zero_fraction <= .8, "reference_unchanged": before == after, "judge_call_accounting": reward.counts["judge_calls"] == reward.counts["parse_valid"]}
    metrics = {key: float(value) for key, value in trained.metrics.items() if isinstance(value, (int, float))}; need(all(math.isfinite(value) for value in metrics.values()), "GRPO metrics non-finite")
    if all(gates.values()):
        adapter = output / "adapter"; unwrapped.save_pretrained(str(adapter), selected_adapters=["default"], safe_serialization=True); tokenizer.save_pretrained(str(adapter)); status = "completed"
    else: adapter = None; status = "failed_gates"
    payload = {"schema_version": "mal2026-rationale-pipeline-grpo-complete-v1", "status": status, "run_id": args.run_id, "stage": "bounded_pilot", "global_step": int(trainer.state.global_step), "train_rows": len(rows), "split": "train", "scores_in_policy_prompt": False, "canonical_scores_attached_only_to_exact_judge": True, "validation_used": False, "average_used": False, "max_steps": args.max_steps, "num_generations": 4, "rollout_request_concurrency": 8, "sampling": {"temperature": .7, "top_p": .95}, "metrics": metrics, "reward": {"completions": reward.counts["completions"], "parse_valid": reward.counts["parse_valid"], "parse_valid_rate": valid_rate, "judge_calls": reward.counts["judge_calls"], "mean": statistics.fmean(reward.values) if reward.values else None, "std": statistics.pstdev(reward.values) if len(reward.values) > 1 else 0.}, "mean_zero_variance_group_fraction": zero_fraction, "hard_gates": gates, "rollout_counts": dict(rollout.counts), "variance_report_sha256": sha256_file(args.variance_report), "judge_prompt_sha256": JUDGE_PROMPT_SHA, "rationale_prompt_sha256": routing()["rationale_generation_training_evaluation"]["source_file_sha256"], "warm_start_adapter_sha256": sha256_file(args.warm_start_adapter / "adapter_model.safetensors"), "reference_adapter_sha256_before": before, "reference_adapter_sha256_after": after, "adapter_path": None if adapter is None else str(adapter.resolve()), "adapter_sha256": None if adapter is None else sha256_file(adapter / "adapter_model.safetensors"), "adapter_precision": precision, "template": _template_provenance(tokenizer), "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights"}
    atomic_json(output / ("training_complete.json" if status == "completed" else "training_failed_gate.json"), payload); print(json.dumps({"status": status, "global_step": payload["global_step"], "zero_variance": zero_fraction, "reward_mean": payload["reward"]["mean"]}, sort_keys=True), flush=True)
    if status != "completed": raise SystemExit(2)


if __name__ == "__main__": main()
