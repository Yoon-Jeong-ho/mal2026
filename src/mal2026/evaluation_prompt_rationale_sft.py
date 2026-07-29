"""LoRA SFT for the two evaluation.txt-derived bundled rationale prompts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .api_rationale_data import ROOT, load_writing_rows
from .api_rationale_sft import _lora_targets, _promote_trainable_lora_parameters, _template_provenance
from .evaluation_prompt_matrix import (
    ALL_RATIONALE_KINDS,
    LEGACY_RATIONALE_KINDS,
    RATIONALE_KINDS,
    RATIONALE_SCORE_BLIND,
    RATIONALE_SCORE_BLIND_V1,
    RATIONALE_SCORE_CONDITIONED,
    RATIONALE_SCORE_CONDITIONED_V1,
    prompt_provenance,
    rationale_messages,
    rationale_output,
)
from .official_rationale_data import candidate_provenance, load_candidates


MODEL_ID = "skt/A.X-4.0-Light"
MODEL_REVISION = "ba21c20ea1b31ded1ec3e2fb432335077dc4be98"
MODEL_PATH = ROOT / "outputs/model-cache/skt--A.X-4.0-Light-ba21c20ea1b31ded1ec3e2fb432335077dc4be98"
OUTPUT_ROOT_V1 = ROOT / "outputs/evaluation-prompt-rationale-sft-v1"
OUTPUT_ROOT = ROOT / "outputs/evaluation-prompt-rationale-sft-v2"


class EvaluationPromptRationaleSFTError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationPromptRationaleSFTError(message)


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
            need(math.isfinite(parsed), f"non-finite training metric {key}")
            result[str(key)] = parsed
    need("train_loss" in result, "training did not return train_loss")
    return result


@dataclass(frozen=True)
class EvaluationPromptRationaleSFTConfig:
    schema_version: str
    run_id: str
    prompt_kind: str
    output_root: str
    seed: int
    max_length: int
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    logging_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float

    @classmethod
    def from_json(cls, path: Path) -> "EvaluationPromptRationaleSFTConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "rationale SFT config fields differ")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self) -> None:
        need(self.prompt_kind in ALL_RATIONALE_KINDS, "rationale SFT prompt kind differs")
        legacy = self.prompt_kind in LEGACY_RATIONALE_KINDS
        expected_version = "v1" if legacy else "v2"
        need(self.schema_version == f"mal2026-evaluation-prompt-rationale-sft-{expected_version}", "rationale SFT schema differs")
        suffix = "score-conditioned" if self.prompt_kind in (RATIONALE_SCORE_CONDITIONED_V1, RATIONALE_SCORE_CONDITIONED) else "score-blind"
        need(self.run_id == f"evaluation-prompt-rationale-sft-{expected_version}-ax4-{suffix}-20260729-001", "rationale SFT run ID differs")
        expected_root = OUTPUT_ROOT_V1 if legacy else OUTPUT_ROOT
        need(Path(self.output_root).resolve() == expected_root.resolve(), "rationale SFT output root differs")
        need(MODEL_PATH.is_dir() and not MODEL_PATH.is_symlink(), "rationale SFT model is unavailable")
        need((self.seed, self.max_length, self.learning_rate, self.num_train_epochs) == (2026072904, 3072, 2e-5, 2.0), "rationale SFT optimization differs")
        need((self.per_device_train_batch_size, self.gradient_accumulation_steps) == (2, 8), "rationale SFT batch differs")
        need((self.lora_r, self.lora_alpha, self.lora_dropout) == (32, 64, 0.05), "rationale SFT LoRA differs")


def sft_examples(prompt_kind: str, limit: int | None = None) -> list[dict[str, Any]]:
    need(prompt_kind in ALL_RATIONALE_KINDS, "rationale example prompt kind differs")
    writings = {row.identifier: row for row in load_writing_rows("train", include_scores=False)}
    candidates = load_candidates()
    if limit is not None:
        need(0 < limit <= len(candidates), "rationale example limit differs")
        candidates = candidates[:limit]
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        writing = writings[candidate.source_id]
        scores = candidate.scores if prompt_kind in (RATIONALE_SCORE_CONDITIONED_V1, RATIONALE_SCORE_CONDITIONED) else None
        result.append({
            "prompt": rationale_messages(writing.prompt, writing.essay, prompt_kind, scores),
            "completion": [{
                "role": "assistant",
                "content": json.dumps(rationale_output(candidate.rationales), ensure_ascii=False, separators=(",", ":")),
            }],
        })
    need(len(result) == (limit if limit is not None else 6000), "rationale example population differs")
    return result


def token_length_audit(examples: list[dict[str, Any]], tokenizer: Any, max_length: int) -> dict[str, Any]:
    lengths: list[int] = []
    for example in examples:
        conversation = [*example["prompt"], *example["completion"]]
        rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        lengths.append(len(tokenizer(rendered, add_special_tokens=False)["input_ids"]))
    need(bool(lengths) and max(lengths) <= max_length, "rationale SFT input would be truncated")
    ordered = sorted(lengths)
    return {
        "records": len(lengths),
        "maximum": max(lengths),
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "limit": max_length,
        "truncated": 0,
    }


def run(config: EvaluationPromptRationaleSFTConfig, *, smoke: bool = False) -> dict[str, Any]:
    config.validate()
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("evaluation prompt rationale SFT requires .venv-standard") from exc

    output = Path(config.output_root) / (f"smoke-{config.run_id}" if smoke else config.run_id)
    rank = int(__import__("os").environ.get("RANK", "0"))
    if rank == 0:
        need(not output.exists(), f"refusing to reuse rationale SFT output: {output}")
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, revision=MODEL_REVISION, local_files_only=True,
        trust_remote_code=False, use_fast=True,
    )
    if tokenizer.pad_token is None:
        need(tokenizer.eos_token is not None, "rationale SFT tokenizer lacks pad/EOS")
        tokenizer.pad_token = tokenizer.eos_token
    template = _template_provenance(tokenizer)
    examples = sft_examples(config.prompt_kind, 1 if smoke else None)
    length_audit = token_length_audit(examples, tokenizer, config.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, revision=MODEL_REVISION, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    targets = _lora_targets(model)
    peft = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, target_modules=targets, bias="none",
    )
    args = SFTConfig(
        output_dir=str(output), run_name=config.run_id, seed=config.seed,
        max_length=config.max_length, packing=False, completion_only_loss=True,
        assistant_only_loss=False, learning_rate=config.learning_rate,
        num_train_epochs=1.0 if smoke else config.num_train_epochs,
        max_steps=1 if smoke else -1,
        per_device_train_batch_size=1 if smoke else config.per_device_train_batch_size,
        gradient_accumulation_steps=1 if smoke else config.gradient_accumulation_steps,
        logging_steps=1 if smoke else config.logging_steps, logging_strategy="steps",
        save_strategy="no" if smoke else "epoch", save_total_limit=2,
        bf16=False, tf32=True, gradient_checkpointing=True, report_to=[],
        logging_nan_inf_filter=False, remove_unused_columns=False, dataset_num_proc=1,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=Dataset.from_list(examples),
        processing_class=tokenizer, peft_config=peft,
    )
    adapter_precision = _promote_trainable_lora_parameters(trainer.model)
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None
    failed = False
    if trainer.is_world_process_zero():
        try:
            metrics = finite_metrics(trained.metrics)
            adapter = output / "adapter"
            trainer.save_model(str(adapter))
            tokenizer.save_pretrained(str(adapter))
            need((adapter / "adapter_config.json").is_file(), "rationale SFT adapter was not saved")
            payload = {
                "schema_version": "mal2026-evaluation-prompt-rationale-sft-complete-v1",
                "status": "completed", "mode": "gpu0_smoke" if smoke else "full",
                "run_id": config.run_id, "prompt_kind": config.prompt_kind,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "global_step": int(trainer.state.global_step), "train_records": len(examples),
                "train_metrics": metrics, "token_length_audit": length_audit,
                "lora_targets": targets, "adapter_precision": adapter_precision,
                "template": template, "candidate_provenance": candidate_provenance(),
                "model_config_sha256": file_sha(MODEL_PATH / "config.json"),
                "human_or_reference_score_read_or_prompted": False,
                "score_conditioning": config.prompt_kind in (RATIONALE_SCORE_CONDITIONED_V1, RATIONALE_SCORE_CONDITIONED),
                "score_kind": "api_candidate_emitted_integer_prediction" if config.prompt_kind in (RATIONALE_SCORE_CONDITIONED_V1, RATIONALE_SCORE_CONDITIONED) else None,
                "config": asdict(config), **prompt_provenance(config.prompt_kind),
                "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_or_predictions",
            }
            (output / ("smoke_complete.json" if smoke else "training_complete.json")).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except Exception:
            failed = True
    state: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state, src=0)
    if state[0] or not isinstance(state[1], dict):
        raise EvaluationPromptRationaleSFTError("rationale SFT completion persistence failed")
    return state[1]
