"""LoRA SFT for bundled or axis-specific official rationale generators."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .api_rationale_data import ROOT
from .api_rationale_sft import _lora_targets, _promote_trainable_lora_parameters, _template_provenance
from .official_rationale_data import TASKS, candidate_provenance, sft_examples


MODEL_ID = "skt/A.X-4.0-Light"
MODEL_REVISION = "ba21c20ea1b31ded1ec3e2fb432335077dc4be98"
MODEL_PATH = ROOT / "outputs/model-cache/skt--A.X-4.0-Light-ba21c20ea1b31ded1ec3e2fb432335077dc4be98"
OUTPUT_ROOT = ROOT / "outputs/official-rationale-sft-v1"


class OfficialRationaleSFTError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialRationaleSFTError(message)


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
class OfficialRationaleSFTConfig:
    schema_version: str
    run_id: str
    task: str
    phase: str
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
    def from_json(cls, path: Path) -> "OfficialRationaleSFTConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "official SFT config fields differ")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self) -> None:
        need(self.schema_version == "mal2026-official-rationale-sft-v1", "official SFT schema differs")
        need(self.task in TASKS and self.phase in {"gpu0_smoke", "full"}, "official SFT task/phase differs")
        output = Path(self.output_dir)
        suffix = "gpu0-smoke-001" if self.phase == "gpu0_smoke" else "full-001"
        expected = f"official-rationale-sft-v1-ax4-{self.task}-{suffix}"
        need(self.run_id == expected and output.is_absolute() and output.parent == OUTPUT_ROOT.resolve() and output.name == expected and not output.exists(), "official SFT output identity/freshness differs")
        need(MODEL_PATH.is_dir() and not MODEL_PATH.is_symlink(), "official SFT model snapshot is unavailable")
        need((self.seed, self.max_length, self.learning_rate, self.num_train_epochs) == (2026072701, 3072, 2e-5, 2.0), "official SFT optimization differs")
        expected_population = (1, 1) if self.phase == "gpu0_smoke" else (6000, -1)
        need((self.train_limit, self.max_steps) == expected_population, "official SFT population differs")
        need((self.per_device_train_batch_size, self.gradient_accumulation_steps) == (2, 8), "official SFT global batch differs")
        need((self.lora_r, self.lora_alpha, self.lora_dropout) == (32, 64, 0.05), "official SFT LoRA differs")


def run(config: OfficialRationaleSFTConfig) -> dict[str, Any]:
    config.validate()
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("official rationale SFT requires .venv-standard") from exc
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        need(tokenizer.eos_token is not None, "official SFT tokenizer lacks pad/EOS")
        tokenizer.pad_token = tokenizer.eos_token
    template = _template_provenance(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.config.use_cache = False
    targets = _lora_targets(model)
    examples = sft_examples(config.task, config.train_limit)
    dataset = Dataset.from_list(examples)
    peft = LoraConfig(task_type=TaskType.CAUSAL_LM, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=targets, bias="none")
    args = SFTConfig(
        output_dir=config.output_dir,
        run_name=config.run_id,
        seed=config.seed,
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
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer, peft_config=peft)
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
            need((adapter / "adapter_config.json").is_file(), "official SFT adapter was not saved")
            payload = {
                "schema_version": "mal2026-official-rationale-sft-complete-v1",
                "status": "completed",
                "run_id": config.run_id,
                "task": config.task,
                "phase": config.phase,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "global_step": int(trainer.state.global_step),
                "train_records": len(examples),
                "train_metrics": metrics,
                "lora_targets": targets,
                "adapter_precision": adapter_precision,
                "template": template,
                "candidate_provenance": candidate_provenance(),
                "model_config_sha256": file_sha(MODEL_PATH / "config.json"),
                "config": asdict(config),
                "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions",
            }
            (Path(config.output_dir) / "training_complete.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            failed = True
    state: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state, src=0)
    if state[0] or not isinstance(state[1], dict):
        raise OfficialRationaleSFTError("official SFT completion persistence failed")
    return state[1]
