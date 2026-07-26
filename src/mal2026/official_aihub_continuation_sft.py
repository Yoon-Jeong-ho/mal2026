"""LoRA continuation on official API rationales after AI-Hub full tuning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .api_rationale_data import AXES, ROOT
from .api_rationale_sft import _lora_targets, _promote_trainable_lora_parameters, _template_provenance
from .official_rationale_data import candidate_provenance, sft_examples
from .official_rationale_sft import MODEL_ID, MODEL_REVISION


FULL_ROOT = ROOT / "outputs/official-aihub-rationale-full-sft-v1"
OUTPUT_ROOT = ROOT / "outputs/official-aihub-then-api-rationale-lora-v1"


class OfficialAIHubContinuationError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialAIHubContinuationError(message)


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
            need(math.isfinite(parsed), f"non-finite continuation metric {key}")
            result[str(key)] = parsed
    need("train_loss" in result, "continuation did not return train_loss")
    return result


def full_run_id(structure: str) -> str:
    return f"official-aihub-rationale-full-sft-v1-ax4-{structure}-full-002"


def full_model_path(structure: str) -> Path:
    return FULL_ROOT / full_run_id(structure) / "final_model"


def full_completion_path(structure: str) -> Path:
    return FULL_ROOT / full_run_id(structure) / "training_complete.json"


@dataclass(frozen=True)
class OfficialAIHubContinuationConfig:
    schema_version: str
    run_id: str
    structure: str
    task: str
    phase: str
    physical_gpu: int
    output_dir: str
    seed: int
    max_length: int
    learning_rate: float
    num_train_epochs: float
    max_steps: int
    train_limit: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    logging_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float

    @classmethod
    def from_json(cls, path: Path) -> "OfficialAIHubContinuationConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "AI-Hub continuation config fields differ")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self) -> None:
        need(self.schema_version == "mal2026-official-aihub-then-api-rationale-lora-v1", "AI-Hub continuation schema differs")
        need(self.structure in {"bundle", "axis_triplet"} and self.phase in {"gpu0_smoke", "full"}, "AI-Hub continuation identity differs")
        allowed_tasks = {"bundle"} if self.structure == "bundle" else set(AXES)
        need(self.task in allowed_tasks, "AI-Hub continuation task differs from winner structure")
        need(self.physical_gpu in {0, 1, 2, 3} and (self.phase != "gpu0_smoke" or self.physical_gpu == 0), "AI-Hub continuation GPU scope differs")
        expected_id = f"official-aihub-then-api-rationale-lora-v1-ax4-{self.structure}-{self.task}-{self.phase}-001"
        output = Path(self.output_dir)
        need(self.run_id == expected_id and output.is_absolute() and output.parent == OUTPUT_ROOT.resolve() and output.name == expected_id and not output.exists(), "AI-Hub continuation output identity/freshness differs")
        need((self.seed, self.max_length, self.learning_rate, self.num_train_epochs) == (2026072701, 3072, 2e-5, 2.0), "AI-Hub continuation optimization differs")
        need((self.train_limit, self.max_steps) == ((1, 1) if self.phase == "gpu0_smoke" else (6000, -1)), "AI-Hub continuation population differs")
        need((self.per_device_train_batch_size, self.gradient_accumulation_steps) == (2, 8), "AI-Hub continuation global batch differs")
        need((self.logging_steps, self.lora_r, self.lora_alpha, self.lora_dropout) == ((1 if self.phase == "gpu0_smoke" else 10), 32, 64, 0.05), "AI-Hub continuation LoRA differs")
        completion_path = full_completion_path(self.structure)
        model_path = full_model_path(self.structure)
        need(completion_path.is_file() and model_path.is_dir(), "AI-Hub full-tuned base is unavailable")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        need(
            completion.get("status") == "completed"
            and completion.get("structure") == self.structure
            and completion.get("phase") == "full"
            and completion.get("training_kind") == "full_parameter",
            "AI-Hub full-tuned base provenance differs",
        )


def run(config: OfficialAIHubContinuationConfig) -> dict[str, Any]:
    config.validate()
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("AI-Hub continuation requires .venv-standard") from exc
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == str(config.physical_gpu), "AI-Hub continuation CUDA binding differs")
    set_seed(config.seed)
    base = full_model_path(config.structure)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        need(tokenizer.eos_token is not None, "AI-Hub continuation tokenizer lacks pad/EOS")
        tokenizer.pad_token = tokenizer.eos_token
    template = _template_provenance(tokenizer)
    # Match the no-AI-Hub SFT arm's numeric contract so the comparison changes
    # only the full-parameter pretraining lineage.
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, trust_remote_code=False, dtype=torch.float32, low_cpu_mem_usage=True)
    model.config.use_cache = False
    targets = _lora_targets(model)
    examples = sft_examples(config.task, config.train_limit)
    dataset = Dataset.from_list(examples)
    peft = LoraConfig(task_type=TaskType.CAUSAL_LM, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=targets, bias="none")
    arguments = SFTConfig(
        output_dir=config.output_dir,
        run_name=config.run_id,
        seed=config.seed,
        data_seed=config.seed,
        max_length=config.max_length,
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=False,
        tf32=True,
        gradient_checkpointing=True,
        report_to=[],
        logging_nan_inf_filter=False,
        remove_unused_columns=False,
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(model=model, args=arguments, train_dataset=dataset, processing_class=tokenizer, peft_config=peft)
    adapter_precision = _promote_trainable_lora_parameters(trainer.model)
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None
    failed = False
    if trainer.is_world_process_zero():
        try:
            metrics = finite_metrics(trained.metrics)
            adapter = Path(config.output_dir) / "adapter"
            trainer.save_model(str(adapter))
            tokenizer.save_pretrained(str(adapter))
            need((adapter / "adapter_config.json").is_file(), "AI-Hub continuation adapter was not saved")
            completion = full_completion_path(config.structure)
            weights = sorted(full_model_path(config.structure).glob("*.safetensors"))
            payload = {
                "schema_version": "mal2026-official-aihub-then-api-rationale-lora-complete-v1",
                "status": "completed",
                "run_id": config.run_id,
                "structure": config.structure,
                "task": config.task,
                "phase": config.phase,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "global_step": int(trainer.state.global_step),
                "train_records": len(examples),
                "train_metrics": metrics,
                "training_kind": "lora_after_aihub_full_parameter",
                "lora_targets": targets,
                "adapter_precision": adapter_precision,
                "template": template,
                "candidate_provenance": candidate_provenance(),
                "full_base_completion_sha256": file_sha(completion),
                "full_base_weight_sha256": {path.name: file_sha(path) for path in weights},
                "config": asdict(config),
                "privacy": "aggregate_only_no_rows_prompts_essays_feedback_rationales_ids_or_scores",
            }
            (Path(config.output_dir) / "training_complete.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            failed = True
    state: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state, src=0)
    if state[0] or not isinstance(state[1], dict):
        raise OfficialAIHubContinuationError("AI-Hub continuation completion persistence failed")
    return state[1]
