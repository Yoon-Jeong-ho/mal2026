#!/usr/bin/env python3
"""Train the final prompt+essay -> three score+rationale decoder baseline."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

from mal2026.api_rationale_data import load_writing_rows, sha256_file  # noqa: E402
from mal2026.api_rationale_sft import _lora_targets, _promote_trainable_lora_parameters, _template_provenance  # noqa: E402
from mal2026.rationale_pipeline_prompts import AXES, normalize_rationales, round_half_up_score  # noqa: E402


OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-joint-decoder-v1"
MODEL_ID = "skt/A.X-4.0-Light"
MODEL_REVISION = "ba21c20ea1b31ded1ec3e2fb432335077dc4be98"
EVALUATION_PROMPT = ROOT / "evaluation.txt"
EVALUATION_PROMPT_SHA = "1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145"


def need(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(message)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp"); temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(path)


@dataclass(frozen=True)
class Config:
    schema_version: str
    run_id: str
    model_path: str
    encoder_handoff_path: str
    encoder_handoff_sha256: str
    output_dir: str
    seed: int
    max_length: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    gpu_scope: tuple[int, ...]
    user_authorization: str

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8")); need(isinstance(raw.get("gpu_scope"), list), "joint decoder GPU scope differs"); raw["gpu_scope"] = tuple(raw["gpu_scope"]); need(set(raw) == set(cls.__dataclass_fields__), "joint decoder config fields differ"); value = cls(**raw); value.validate(); return value

    def validate(self) -> None:
        need(self.schema_version == "mal2026-rationale-pipeline-joint-decoder-v1", "joint decoder schema differs")
        need(self.gpu_scope == (0, 1, 2, 3) and bool(self.user_authorization.strip()), "joint decoder GPU authorization differs")
        need(Path(self.model_path).is_dir() and Path(self.output_dir).resolve() == (OUTPUT_PARENT / self.run_id).resolve(), "joint decoder model/output differs")
        handoff = Path(self.encoder_handoff_path); need(handoff.is_file() and sha256_file(handoff) == self.encoder_handoff_sha256, "joint decoder rationale handoff differs")
        need((self.seed, self.max_length) == (2026080708, 3072), "joint decoder seed/length differs")
        need((self.learning_rate, self.weight_decay, self.warmup_ratio, self.num_train_epochs) == (2e-5, .01, .03, 2.0), "joint decoder optimizer differs")
        need((self.per_device_train_batch_size, self.gradient_accumulation_steps) == (2, 8), "joint decoder batch differs")
        need((self.lora_r, self.lora_alpha, self.lora_dropout) == (32, 64, .05), "joint decoder LoRA differs")
        need(sha256_file(EVALUATION_PROMPT) == EVALUATION_PROMPT_SHA, "evaluation.txt prompt differs")


def messages(prompt: str, essay: str) -> list[dict[str, str]]:
    text = EVALUATION_PROMPT.read_text(encoding="utf-8")
    need(text.count("[시스템 프롬프트]") == text.count("[유저 프롬프트]") == 1, "evaluation.txt section markers differ")
    before, user = text.split("[유저 프롬프트]", 1); system = before.split("[시스템 프롬프트]", 1)[1].strip()
    need(user.count("{주제 지문}") == user.count("{논증적 글 본문}") == 1, "evaluation.txt placeholders differ")
    rendered = user.replace("{주제 지문}", prompt, 1).replace("{논증적 글 본문}", essay, 1).strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": rendered}]


def examples(config: Config, limit: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    handoff = json.loads(Path(config.encoder_handoff_path).read_text(encoding="utf-8"))
    need(handoff.get("schema_version") == "mal2026-rationale-pipeline-encoder-ratio-handoff-v2" and handoff.get("status") == "completed" and handoff.get("teacher_use") == "train_only_label_aware_augmentation_never_validation_or_selection_dev", "joint decoder handoff contract differs")
    teacher_path = Path(handoff["paths"]["teacher_train_single_best"]); need(sha256_file(teacher_path) == handoff["sha256"]["teacher_train_single_best"], "joint decoder teacher rationale differs")
    teacher = {str(row["source_id"]): normalize_rationales(row["rationales"]) for row in (json.loads(line) for line in teacher_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    writings = load_writing_rows("train", include_scores=True); need(len(teacher) == len(writings) == 2000, "joint decoder train population differs")
    if limit is not None: writings = writings[:limit]
    result = []
    for writing in writings:
        need(writing.scores is not None and writing.identifier in teacher, "joint decoder score/rationale linkage differs")
        target = {axis: {"score": round_half_up_score(writing.scores[axis]), "rationale": teacher[writing.identifier][axis]} for axis in AXES}
        result.append({"prompt": messages(writing.prompt, writing.essay), "completion": [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))}]})
    return result, {"records_full": 2000, "split": "train", "score_target": "canonical_axis_Decimal_ROUND_HALF_UP", "rationale_target": "train_only_exact_Q4_best_openai_teacher_single_per_source", "average_used": False, "handoff_sha256": config.encoder_handoff_sha256}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args(); config = Config.load(args.config)
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1")); setproctitle(f"mal2026:joint-decoder:{'smoke' if args.smoke else 'full'}:rank{rank}"[:255])
    need(world == (1 if args.smoke else 4) and os.environ.get("CUDA_VISIBLE_DEVICES") == ("0" if args.smoke else "0,1,2,3"), "joint decoder runtime GPU scope differs")
    output = Path(config.output_dir) if not args.smoke else OUTPUT_PARENT / f"smoke-{config.run_id}"
    if rank == 0: need(not output.exists(), "joint decoder output must be fresh"); output.mkdir(parents=True)
    else:
        deadline = time.monotonic() + 120
        while not output.is_dir() and time.monotonic() < deadline: time.sleep(.05)
        need(output.is_dir(), "joint decoder output reservation timed out")
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc: raise RuntimeError("joint decoder requires .venv-standard") from exc
    set_seed(config.seed); tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None: need(tokenizer.eos_token is not None, "joint decoder tokenizer lacks PAD/EOS"); tokenizer.pad_token = tokenizer.eos_token
    rows, provenance = examples(config, 1 if args.smoke else None)
    lengths = [len(tokenizer(tokenizer.apply_chat_template([*row["prompt"], *row["completion"]], tokenize=False, add_generation_prompt=False), add_special_tokens=False)["input_ids"]) for row in rows]
    need(lengths and max(lengths) <= config.max_length, "joint decoder example would be truncated")
    model = AutoModelForCausalLM.from_pretrained(config.model_path, local_files_only=True, trust_remote_code=False, dtype=torch.float32, low_cpu_mem_usage=True); model.config.use_cache = False
    targets = _lora_targets(model); peft = LoraConfig(task_type=TaskType.CAUSAL_LM, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=targets, bias="none")
    arguments = SFTConfig(output_dir=str(output), run_name=config.run_id, seed=config.seed, data_seed=config.seed, max_length=config.max_length, packing=False, completion_only_loss=True, assistant_only_loss=False, learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio, lr_scheduler_type="cosine", num_train_epochs=1 if args.smoke else config.num_train_epochs, max_steps=1 if args.smoke else -1, per_device_train_batch_size=config.per_device_train_batch_size, gradient_accumulation_steps=1 if args.smoke else config.gradient_accumulation_steps, logging_steps=1 if args.smoke else 10, save_strategy="no" if args.smoke else "epoch", save_total_limit=2, bf16=False, tf32=True, gradient_checkpointing=True, optim="adamw_torch_fused", report_to=[], logging_nan_inf_filter=False, remove_unused_columns=False, dataset_num_proc=1, ddp_find_unused_parameters=False)
    class Guard(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **_: Any) -> Any:
            for key, value in (logs or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool): need(math.isfinite(float(value)), f"non-finite joint decoder log: {key}")
            return control
    trainer = SFTTrainer(model=model, args=arguments, train_dataset=Dataset.from_list(rows), processing_class=tokenizer, peft_config=peft, callbacks=[Guard()]); precision = _promote_trainable_lora_parameters(trainer.model); trained = trainer.train(); trainer.accelerator.wait_for_everyone()
    adapter = output / "adapter"
    if not args.smoke: trainer.save_model(str(adapter)); trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        if not args.smoke: tokenizer.save_pretrained(str(adapter)); need((adapter / "adapter_model.safetensors").is_file(), "joint decoder adapter export failed")
        metrics = {key: float(value) for key, value in trained.metrics.items() if isinstance(value, (int, float))}; need("train_loss" in metrics and all(math.isfinite(value) for value in metrics.values()), "joint decoder metrics differ")
        payload = {"schema_version": "mal2026-rationale-pipeline-joint-decoder-complete-v1", "status": "completed", "mode": "smoke" if args.smoke else "full", "run_id": config.run_id, "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "world_size": world, "gpu_scope": [0] if args.smoke else list(config.gpu_scope), "global_step": int(trainer.state.global_step), "train_records": len(rows), "metrics": metrics, "token_audit": {"records": len(lengths), "maximum": max(lengths), "limit": config.max_length, "truncated": 0}, "input_provenance": provenance, "evaluation_prompt_sha256": EVALUATION_PROMPT_SHA, "score_target": "per_axis_Decimal_ROUND_HALF_UP_integer", "rationale_target": "train_only_exact_Q4_best_teacher", "average_used": False, "lora_targets": targets, "adapter_precision": precision, "template": _template_provenance(tokenizer), "config": asdict(config), "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights"}
        atomic_json(output / ("smoke_complete.json" if args.smoke else "training_complete.json"), payload); print(json.dumps({"status": "completed", "mode": payload["mode"], "global_step": payload["global_step"]}, sort_keys=True), flush=True)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__": main()
