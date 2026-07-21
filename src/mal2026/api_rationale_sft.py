"""Maintained TRL SFTTrainer for API-rationale-only decoder training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .api_rationale_data import (
    ROOT, APIRationaleContractError, aggregate_input_provenance, axes_for_task,
    candidate_sft_examples,
)


RUN_ROOT = ROOT / "outputs" / "api-rationale-sft-v1"
SUPPORTED_MODELS = {
    "ax4_light": ("skt/A.X-4.0-Light", "ba21c20ea1b31ded1ec3e2fb432335077dc4be98"),
    "phi4_mini": ("microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083"),
    "midm2_base": ("K-intelligence/Midm-2.0-Base-Instruct", "35479c5fc9a18a5db7cc6dbadcf1db68db7beab0"),
}


class APIRationaleSFTError(APIRationaleContractError):
    """Raised for a non-reproducible or incomplete rationale SFT run."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise APIRationaleSFTError(message)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class APIRationaleSFTConfig:
    schema_version: str
    run_id: str
    base_key: str
    model_id: str
    model_revision: str
    model_path: str
    task: str
    phase: str
    train_limit: int
    max_steps: int
    output_dir: str
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
    training_dtype: str
    trust_remote_code: bool

    @classmethod
    def from_json(cls, path: Path) -> "APIRationaleSFTConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        _need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "SFT config has unknown or missing fields")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self) -> None:
        _need(self.schema_version == "mal2026-api-rationale-sft-v1", "unexpected SFT config schema")
        _need(self.base_key in SUPPORTED_MODELS and SUPPORTED_MODELS[self.base_key] == (self.model_id, self.model_revision), "base model identity differs")
        axes_for_task(self.task)
        model_path, output = Path(self.model_path), Path(self.output_dir)
        _need(model_path.is_dir() and not model_path.is_symlink() and model_path.resolve().name.endswith(self.model_revision), "model snapshot must be a local immutable non-symlink directory")
        _need(output.is_absolute() and output.parent == RUN_ROOT.resolve() and not output.exists(), "output directory must be a fresh ignored direct child")
        _need(self.phase in {"gpu0_smoke", "numeric_recovery", "full"}, "SFT phase differs")
        expected = (
            f"api-rationale-sft-v1-{self.base_key}-{self.task}-001" if self.phase == "full" else
            f"api-rationale-sft-v1-{self.base_key}-{self.task}-gpu0_smoke-001" if self.phase == "gpu0_smoke" else
            f"api-rationale-sft-v1-{self.base_key}-{self.task}-numeric_recovery-001"
        )
        _need(self.run_id == expected, "run ID does not bind base/task/phase lineage")
        expected_population = (6000, -1) if self.phase == "full" else (1, 1) if self.phase == "gpu0_smoke" else (6000, 5)
        _need((self.train_limit, self.max_steps) == expected_population, "SFT phase population/update contract differs")
        _need(self.seed == 2026072108 and self.max_length == 3072, "seed or sequence budget differs from task card")
        _need(self.learning_rate == 2e-5 and self.num_train_epochs == 2.0, "decoder optimizer schedule differs from task card")
        _need(self.per_device_train_batch_size == 2 and self.gradient_accumulation_steps == 8, "decoder global batch contract differs")
        _need(self.logging_steps > 0 and self.lora_r == 32 and self.lora_alpha == 64 and 0.0 <= self.lora_dropout < 1.0, "invalid decoder SFT setting")
        _need(self.training_dtype == "float32", "numerical-recovery policy requires float32 frozen-base compute")
        _need(self.trust_remote_code is False, "unreviewed remote code is prohibited for decoder SFT")


def _template_provenance(tokenizer: Any) -> dict[str, Any]:
    """Verify native chat-template behavior without persisting rendered text."""
    messages = [
        {"role": "system", "content": "synthetic system sentinel"},
        {"role": "user", "content": "synthetic user sentinel"},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": "synthetic assistant sentinel"}], tokenize=False, add_generation_prompt=False)
    except Exception as exc:
        raise APIRationaleSFTError("tokenizer native chat template is unusable") from exc
    _need(isinstance(prompt, str) and isinstance(full, str) and prompt and full, "chat template rendered empty output")
    _need("synthetic system sentinel" in prompt and "synthetic user sentinel" in prompt and "synthetic assistant sentinel" in full, "chat template lost a required message boundary")
    return {
        "native_chat_template_used": True,
        "prompt_template_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
        "full_template_sha256": sha256(full.encode("utf-8")).hexdigest(),
        "assistant_generation_boundary_distinct": prompt != full,
        "rendered_text_persisted": False,
    }


def _lora_targets(model: Any) -> list[str]:
    """Select a reviewed complete attention/MLP leaf set for each architecture."""
    leaves = {name.rsplit(".", maxsplit=1)[-1] for name, _ in model.named_modules()}
    alternatives = [
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    ]
    for candidate in alternatives:
        if set(candidate) <= leaves:
            return candidate
    raise APIRationaleSFTError("decoder architecture lacks a reviewed LoRA target set")


def _promote_trainable_lora_parameters(model: Any) -> dict[str, int | str]:
    """Keep adapter weights/states in fp32 while the frozen base stays bf16.

    With the installed PyTorch/PEFT stack, bf16 LoRA optimizer states became
    non-finite after several AdamW updates in the actual A.X SFT integration.
    Keeping the small trainable adapters in fp32 is the maintained LoRA mixed-
    precision convention; it neither changes the frozen base precision nor the
    training objective.
    """
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    _need(bool(trainable) and all("lora_" in name for name, _ in trainable), "only LoRA adapter parameters may be trainable")
    for _, parameter in trainable:
        parameter.data = parameter.data.to(dtype=__import__("torch").float32)
    _need(all(str(parameter.dtype) == "torch.float32" for _, parameter in trainable), "LoRA parameter fp32 promotion failed")
    return {"trainable_adapter_dtype": "float32", "trainable_adapter_tensors": len(trainable)}


def _finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = float(value)
            _need(math.isfinite(parsed), f"non-finite trainer metric {key}")
            result[str(key)] = parsed
    _need("train_loss" in result, "Trainer did not return train_loss")
    return result


def run_api_rationale_sft(config: APIRationaleSFTConfig) -> dict[str, Any]:
    """Train one declared task through TRL SFTTrainer and save aggregate provenance."""
    config.validate()
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError("API rationale SFT requires the project .venv-standard") from exc
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        _need(tokenizer.eos_token is not None, "tokenizer lacks both pad and EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    template = _template_provenance(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    targets = _lora_targets(model)
    examples = candidate_sft_examples(config.task)
    _need(len(examples) == 6000, "full candidate population changed")
    examples = examples[:config.train_limit]
    dataset = Dataset.from_list(examples)
    peft = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, target_modules=targets, bias="none",
    )
    args = SFTConfig(
        output_dir=config.output_dir, run_name=config.run_id, seed=config.seed,
        max_length=config.max_length, packing=False, completion_only_loss=True, assistant_only_loss=False,
        learning_rate=config.learning_rate, num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps, logging_strategy="steps", save_strategy="epoch", save_total_limit=1, max_steps=config.max_steps,
        bf16=False, tf32=True, gradient_checkpointing=True, report_to=[], logging_nan_inf_filter=False,
        remove_unused_columns=False, dataset_num_proc=1, ddp_find_unused_parameters=False,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer, peft_config=peft)
    adapter_precision = _promote_trainable_lora_parameters(trainer.model)
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None
    failed = False
    if trainer.is_world_process_zero():
        try:
            metrics = _finite_metrics(trained.metrics)
            adapter = Path(config.output_dir) / "adapter"
            trainer.save_model(str(adapter)); tokenizer.save_pretrained(str(adapter))
            _need((adapter / "adapter_config.json").is_file(), "Trainer did not export adapter configuration")
            payload = {
                "status": "completed", "run_id": config.run_id, "base_key": config.base_key, "model_id": config.model_id,
                "model_revision": config.model_revision, "task": config.task, "phase": config.phase, "global_step": int(trainer.state.global_step),
                "train_records": len(examples), "train_metrics": metrics, "lora_targets": targets, "adapter_precision": adapter_precision,
                "model_config_sha256": _sha(Path(config.model_path) / "config.json"),
                "tokenizer_config_sha256": _sha(Path(config.model_path) / "tokenizer_config.json"),
                "template": template, "input_provenance": aggregate_input_provenance(), "config": asdict(config),
                "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_candidate_scores_persisted",
            }
            completion = Path(config.output_dir) / "training_complete.json"
            _need(not completion.exists(), "training completion already exists")
            completion.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            failed = True
    state: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state, src=0)
    if state[0]:
        raise APIRationaleSFTError("rank-zero SFT persistence/health gate failed")
    _need(isinstance(state[1], dict) and state[1].get("status") == "completed", "SFT completion artifact was not published")
    trainer.accelerator.wait_for_everyone()
    return state[1]
