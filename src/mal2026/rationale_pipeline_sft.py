"""Score-blind rationale SFT stages for the v3 dual-teacher pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .api_rationale_data import ROOT, load_writing_rows
from .api_rationale_sft import _lora_targets, _promote_trainable_lora_parameters, _template_provenance
from .official_aihub_rationale_data import EXPECTED_COUNTS, load_argumentative, projected_rationales
from .rationale_pipeline_prompts import rationale_messages, rationale_output, routing


MODEL_ID = "skt/A.X-4.0-Light"
MODEL_REVISION = "ba21c20ea1b31ded1ec3e2fb432335077dc4be98"
MODEL_PATH = ROOT / "outputs/model-cache/skt--A.X-4.0-Light-ba21c20ea1b31ded1ec3e2fb432335077dc4be98"
MERGED_ROOT = ROOT / "data/processed/restricted/rationale_v3_tail_sft_merged"
OUTPUT_ROOT = ROOT / "outputs/rationale-pipeline-sft-v1"
STAGES = {
    "mal_direct_lora_smoke",
    "mal_direct_lora_full",
    "aihub_score_blind_full_smoke",
    "aihub_score_blind_full",
    "mal_after_aihub_lora_smoke",
    "mal_after_aihub_lora_full",
}


class RationalePipelineSFTError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RationalePipelineSFTError(message)


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = float(value)
            need(math.isfinite(parsed), f"non-finite SFT metric {key}")
            result[str(key)] = parsed
    need("train_loss" in result, "training did not return train_loss")
    return result


def world_size() -> int:
    raw = os.environ.get("WORLD_SIZE", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RationalePipelineSFTError("WORLD_SIZE is invalid") from exc
    need(value > 0, "WORLD_SIZE is invalid")
    return value


@dataclass(frozen=True)
class RationalePipelineSFTConfig:
    schema_version: str
    run_id: str
    stage: str
    merged_run_id: str | None
    initialization_path: str
    output_dir: str
    seed: int
    max_length: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    logging_steps: int
    lora_r: int | None
    lora_alpha: int | None
    lora_dropout: float | None

    @classmethod
    def from_json(cls, path: Path) -> "RationalePipelineSFTConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "SFT config fields differ")
        value = cls(**raw)
        value.validate()
        return value

    @property
    def smoke(self) -> bool:
        return self.stage.endswith("_smoke")

    @property
    def full_parameter(self) -> bool:
        return self.stage.startswith("aihub_score_blind_full")

    @property
    def mal_stage(self) -> bool:
        return self.stage.startswith("mal_")

    def validate(self) -> None:
        need(self.schema_version == "mal2026-rationale-pipeline-sft-v1" and self.stage in STAGES, "SFT stage identity differs")
        output = Path(self.output_dir)
        need(output.is_absolute() and output.parent == OUTPUT_ROOT.resolve() and output.name == self.run_id, "SFT output identity differs")
        need(not output.exists(), "SFT output must be fresh")
        initialization = Path(self.initialization_path)
        need(initialization.is_dir() and not initialization.is_symlink(), "SFT initialization is unavailable")
        if self.stage.startswith("mal_direct") or self.stage.startswith("aihub_score_blind"):
            need(initialization.resolve() == MODEL_PATH.resolve(), "base initialization differs")
        else:
            need((initialization / "config.json").is_file() and initialization.name == "final_model", "AI-Hub initialization differs")
        if self.mal_stage:
            need(bool(self.merged_run_id) and (MERGED_ROOT / str(self.merged_run_id) / "manifest.json").is_file(), "merged MAL handoff is unavailable")
        else:
            need(self.merged_run_id is None, "AI-Hub full stage must not read MAL handoff")
        need((self.seed, self.max_length) == (2026080704, 3072), "SFT seed or token budget differs")
        expected_lr = 5e-6 if self.full_parameter else 2e-5
        expected_epochs = 1.0 if self.full_parameter else 2.0
        need((self.learning_rate, self.weight_decay, self.warmup_ratio, self.num_train_epochs) == (expected_lr, 0.01, 0.03, expected_epochs), "SFT optimization differs")
        expected_batch = (1, 1) if self.smoke else (1, 8) if self.full_parameter else (2, 8)
        need((self.per_device_train_batch_size, self.gradient_accumulation_steps) == expected_batch, "SFT batch differs")
        if self.full_parameter:
            need((self.lora_r, self.lora_alpha, self.lora_dropout) == (None, None, None), "full-parameter stage has LoRA settings")
        else:
            need((self.lora_r, self.lora_alpha, self.lora_dropout) == (32, 64, 0.05), "LoRA settings differ")
        need(self.logging_steps == (1 if self.smoke else 10), "SFT logging interval differs")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mal_examples(merged_run_id: str, limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = MERGED_ROOT / merged_run_id
    manifest_path = root / "manifest.json"
    target_path = root / "sft_targets.train.quality_filtered.jsonl"
    need(manifest_path.is_file() and target_path.is_file(), "merged MAL target is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("files", {}).get(target_path.name, {})
    need(manifest.get("status") == "completed" and metadata.get("sha256") == file_sha(target_path), "merged MAL target provenance differs")
    rows = jsonl(target_path)
    need(len(rows) == int(metadata.get("records", -1)), "merged MAL target population differs")
    if limit is not None:
        need(0 < limit <= len(rows), "MAL SFT limit differs")
        rows = rows[:limit]
    writings = {row.identifier: row for row in load_writing_rows("train", include_scores=False)}
    examples: list[dict[str, Any]] = []
    for row in rows:
        source_id = str(row["source_id"])
        need(source_id in writings, "MAL target source is unavailable")
        writing = writings[source_id]
        target = rationale_output(row["rationale"])
        examples.append({
            "prompt": rationale_messages(writing.prompt, writing.essay),
            "completion": [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))}],
        })
    return examples, {
        "kind": "merged_v3_tail_quality_filtered_train",
        "merged_run_id": merged_run_id,
        "manifest_sha256": file_sha(manifest_path),
        "target_sha256": file_sha(target_path),
        "full_records": int(metadata["records"]),
        "score_read_or_prompted": False,
    }


def aihub_examples(limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_argumentative("refit_train")
    need(len(rows) == EXPECTED_COUNTS["refit_train"], "AI-Hub population differs")
    if limit is not None:
        need(0 < limit <= len(rows), "AI-Hub SFT limit differs")
        rows = rows[:limit]
    examples: list[dict[str, Any]] = []
    for row in rows:
        target = rationale_output(projected_rationales(row))
        examples.append({
            "prompt": rationale_messages(row.prompt, row.essay),
            "completion": [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))}],
        })
    return examples, {
        "kind": "aihub_argumentative_human_analytic_feedback_score_blind_bundle",
        "split": "Training_only_refit_train",
        "full_records": EXPECTED_COUNTS["refit_train"],
        "feedback_fields": {
            "content": ["content_1", "content_2", "content_3"],
            "organization": ["organization_1", "organization_2"],
            "expression": ["expression_1", "expression_2"],
        },
        "score_read_or_prompted": False,
    }


def token_length_audit(examples: list[dict[str, Any]], tokenizer: Any, max_length: int) -> dict[str, Any]:
    lengths: list[int] = []
    for example in examples:
        rendered = tokenizer.apply_chat_template([*example["prompt"], *example["completion"]], tokenize=False, add_generation_prompt=False)
        lengths.append(len(tokenizer(rendered, add_special_tokens=False)["input_ids"]))
    need(bool(lengths) and max(lengths) <= max_length, "SFT example would be truncated")
    ordered = sorted(lengths)
    return {
        "records": len(lengths),
        "maximum": max(lengths),
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "limit": max_length,
        "truncated": 0,
    }


def run(config: RationalePipelineSFTConfig) -> dict[str, Any]:
    config.validate()
    routing()
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("rationale pipeline SFT requires .venv-standard") from exc

    expected_world = 1 if config.smoke else 4
    need(world_size() == expected_world, "SFT world size differs")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == ("0" if config.smoke else "0,1,2,3"), "SFT GPU scope differs")
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.initialization_path, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        need(tokenizer.eos_token is not None, "SFT tokenizer lacks pad/EOS")
        tokenizer.pad_token = tokenizer.eos_token
    template = _template_provenance(tokenizer)
    examples, input_provenance = (
        mal_examples(str(config.merged_run_id), 1 if config.smoke else None)
        if config.mal_stage else aihub_examples(1 if config.smoke else None)
    )
    length_audit = token_length_audit(examples, tokenizer, config.max_length)
    master_dtype = (
        torch.bfloat16 if config.full_parameter and config.smoke else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.initialization_path, local_files_only=True, trust_remote_code=False,
        dtype=master_dtype, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    dataset = Dataset.from_list(examples)
    peft_config = None
    lora_targets: list[str] = []
    if not config.full_parameter:
        lora_targets = _lora_targets(model)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=int(config.lora_r), lora_alpha=int(config.lora_alpha),
            lora_dropout=float(config.lora_dropout), target_modules=lora_targets, bias="none",
        )
    fsdp: bool | None = None
    fsdp_config: dict[str, Any] | None = None
    if config.full_parameter and not config.smoke:
        fsdp = True
        fsdp_config = {
            "version": 2,
            "reshard_after_forward": True,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],
            "activation_checkpointing": True,
            "state_dict_type": "FULL_STATE_DICT",
            "cpu_ram_efficient_loading": False,
        }
    arguments = SFTConfig(
        output_dir=config.output_dir, run_name=config.run_id, seed=config.seed, data_seed=config.seed,
        max_length=config.max_length, packing=False, completion_only_loss=True, assistant_only_loss=False,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine", num_train_epochs=1.0 if config.smoke else config.num_train_epochs,
        max_steps=1 if config.smoke else -1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps, logging_strategy="steps",
        save_strategy="no" if config.smoke or config.full_parameter else "epoch",
        save_total_limit=2,
        bf16=config.full_parameter and config.smoke,
        tf32=True,
        gradient_checkpointing=not (config.full_parameter and not config.smoke),
        optim="adamw_torch_fused", report_to=[], logging_nan_inf_filter=False,
        remove_unused_columns=False, dataset_num_proc=1,
        ddp_find_unused_parameters=False if not config.full_parameter else None,
        fsdp=fsdp, fsdp_config=fsdp_config,
    )

    class NonFiniteGuard(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
            for key, value in (logs or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    need(math.isfinite(float(value)), f"non-finite SFT log {key} at step {state.global_step}")
            return control

    trainer = SFTTrainer(
        model=model, args=arguments, train_dataset=dataset, processing_class=tokenizer,
        peft_config=peft_config, callbacks=[NonFiniteGuard()],
    )
    adapter_precision: Mapping[str, Any] | None = None
    if not config.full_parameter:
        adapter_precision = _promote_trainable_lora_parameters(trainer.model)
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()

    final_name = "final_model" if config.full_parameter else "adapter"
    final_path = Path(config.output_dir) / final_name
    if not config.smoke:
        trainer.save_model(str(final_path))
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(str(final_path))
        trainer.accelerator.wait_for_everyone()

    payload: dict[str, Any] | None = None
    failed = False
    if trainer.is_world_process_zero():
        try:
            metrics = finite_metrics(trained.metrics)
            if not config.smoke:
                need((final_path / "config.json").is_file() or (final_path / "adapter_config.json").is_file(), "SFT final artifact is unavailable")
            payload = {
                "schema_version": "mal2026-rationale-pipeline-sft-complete-v1",
                "status": "completed",
                "run_id": config.run_id,
                "stage": config.stage,
                "training_kind": "full_parameter" if config.full_parameter else "lora",
                "master_parameter_dtype": str(master_dtype),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "initialization_path": config.initialization_path,
                "world_size": world_size(),
                "global_step": int(trainer.state.global_step),
                "train_records": len(examples),
                "train_metrics": metrics,
                "token_length_audit": length_audit,
                "template": template,
                "input_provenance": input_provenance,
                "lora_targets": lora_targets,
                "adapter_precision": adapter_precision,
                "prompt_routing": routing(),
                "human_or_reference_score_read_or_prompted": False,
                "average_used": False,
                "config": asdict(config),
                "privacy": "aggregate_only_no_rows_prompts_essays_feedback_rationales_ids_scores_or_model_weights",
            }
            completion = Path(config.output_dir) / ("smoke_complete.json" if config.smoke else "training_complete.json")
            completion.parent.mkdir(parents=True, exist_ok=True)
            completion.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            failed = True
    state: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state, src=0)
    if state[0] or not isinstance(state[1], dict):
        raise RationalePipelineSFTError("SFT completion persistence failed")
    return state[1]
