"""Full-parameter A.X rationale SFT on closest-match AI-Hub feedback."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .api_rationale_data import ROOT
from .api_rationale_sft import _template_provenance
from .official_aihub_rationale_data import EXPECTED_COUNTS, provenance, structure_sft_examples
from .official_rationale_sft import MODEL_ID, MODEL_PATH, MODEL_REVISION


OUTPUT_ROOT = ROOT / "outputs/official-aihub-rationale-full-sft-v1"
STRUCTURES = ("bundle", "axis_triplet")


class OfficialAIHubSFTError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialAIHubSFTError(message)


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
            need(math.isfinite(parsed), f"non-finite AI-Hub training metric {key}")
            result[str(key)] = parsed
    need("train_loss" in result, "AI-Hub training did not return train_loss")
    return result


@dataclass(frozen=True)
class OfficialAIHubSFTConfig:
    schema_version: str
    run_id: str
    structure: str
    phase: str
    output_dir: str
    seed: int
    max_length: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    num_train_epochs: float
    max_steps: int
    train_limit: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    logging_steps: int
    compute_dtype: str

    @classmethod
    def from_json(cls, path: Path) -> "OfficialAIHubSFTConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "AI-Hub full SFT config fields differ")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self) -> None:
        need(self.schema_version == "mal2026-official-aihub-rationale-full-sft-v1", "AI-Hub full SFT schema differs")
        need(self.structure in STRUCTURES and self.phase in {"gpu0_smoke", "fsdp4_smoke", "fsdp4_numeric_smoke", "fsdp4_fp32_numeric_smoke", "full"}, "AI-Hub full SFT identity differs")
        expected_prefix = f"official-aihub-rationale-full-sft-v1-ax4-{self.structure}-{self.phase}-"
        output = Path(self.output_dir)
        allowed_suffixes = {"001"} if self.phase in {"fsdp4_smoke", "fsdp4_numeric_smoke", "fsdp4_fp32_numeric_smoke"} else {"001", "002"}
        need(
            self.run_id.startswith(expected_prefix)
            and self.run_id.removeprefix(expected_prefix) in allowed_suffixes
            and output.is_absolute()
            and output.parent == OUTPUT_ROOT.resolve()
            and output.name == self.run_id
            and not output.exists(),
            "AI-Hub full SFT output identity/freshness differs",
        )
        need((self.seed, self.max_length, self.learning_rate, self.weight_decay, self.warmup_ratio, self.num_train_epochs) == (2026072702, 2048, 5e-6, 0.01, 0.03, 1.0), "AI-Hub full SFT optimization differs")
        full_records = EXPECTED_COUNTS["refit_train"] * (1 if self.structure == "bundle" else 3)
        population = {
            "gpu0_smoke": (1, 1),
            "fsdp4_smoke": (1, 1),
            "fsdp4_numeric_smoke": (1024, 30),
            "fsdp4_fp32_numeric_smoke": (1024, 30),
            "full": (full_records, -1),
        }[self.phase]
        batch = (2, 4) if self.phase == "fsdp4_numeric_smoke" else ((1, 8) if self.phase in {"fsdp4_fp32_numeric_smoke", "full"} else (1, 1))
        need((self.train_limit, self.max_steps) == population, "AI-Hub full SFT population differs")
        need((self.per_device_train_batch_size, self.gradient_accumulation_steps) == batch, "AI-Hub full SFT batch differs")
        need(self.logging_steps == (10 if self.phase == "full" else 1), "AI-Hub full SFT logging differs")
        expected_dtype = {
            "gpu0_smoke": "bfloat16_direct",
            "fsdp4_smoke": "bfloat16_direct",
            "fsdp4_numeric_smoke": "float32_master_bfloat16_mixed_precision",
            "fsdp4_fp32_numeric_smoke": "float32_tf32",
            "full": "float32_tf32",
        }[self.phase]
        need(self.compute_dtype == expected_dtype, "AI-Hub full SFT compute dtype differs")


def _world_size() -> int:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("AI-Hub full rationale SFT requires .venv-standard") from exc
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    # torchrun exports WORLD_SIZE before Transformers/Accelerate initializes
    # the process group.  Reading the launcher contract here avoids falsely
    # classifying every rank as a single-process run during pre-initialization.
    raw = os.environ.get("WORLD_SIZE")
    if raw is None:
        return 1
    try:
        value = int(raw)
    except ValueError as exc:
        raise OfficialAIHubSFTError("AI-Hub full SFT WORLD_SIZE is invalid") from exc
    need(value > 0, "AI-Hub full SFT WORLD_SIZE is invalid")
    return value


def run(config: OfficialAIHubSFTConfig) -> dict[str, Any]:
    config.validate()
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("AI-Hub full rationale SFT requires .venv-standard") from exc
    set_seed(config.seed)
    world_size = _world_size()
    expected_world = 1 if config.phase == "gpu0_smoke" else 4
    need(world_size == expected_world, "AI-Hub full SFT world size differs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    need(visible == ("0" if config.phase == "gpu0_smoke" else "0,1,2,3"), "AI-Hub full SFT GPU scope differs")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        need(tokenizer.eos_token is not None, "AI-Hub full SFT tokenizer lacks pad/EOS")
        tokenizer.pad_token = tokenizer.eos_token
    template = _template_provenance(tokenizer)
    # Keep distributed master parameters and optimizer states in float32 while
    # SFTConfig.bf16 supplies bfloat16 autocast/FSDP mixed-precision compute.
    # Direct bfloat16 master parameters became non-finite after 20 full-data
    # updates; the single-GPU one-update smoke retains its original contract.
    master_dtype = torch.bfloat16 if config.compute_dtype == "bfloat16_direct" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False,
        dtype=master_dtype, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    examples = structure_sft_examples(config.structure, "refit_train", config.train_limit)
    dataset = Dataset.from_list(examples)
    fsdp = None
    fsdp_config = None
    if config.phase != "gpu0_smoke":
        # This is the same maintained Transformers/Accelerate FSDP2 contract
        # already proven by the repository's Qwen3 full-parameter arm.
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
        output_dir=config.output_dir,
        run_name=config.run_id,
        seed=config.seed,
        data_seed=config.seed,
        max_length=config.max_length,
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        save_strategy="no",
        bf16=config.compute_dtype != "float32_tf32",
        tf32=True,
        # FSDP's activation_checkpointing above is the maintained distributed
        # mechanism.  Transformers 5 rejects enabling its separate generic
        # gradient-checkpointing path at the same time.
        gradient_checkpointing=config.phase == "gpu0_smoke",
        optim="adamw_torch_fused",
        report_to=[],
        logging_nan_inf_filter=False,
        remove_unused_columns=False,
        dataset_num_proc=1,
        fsdp=fsdp,
        fsdp_config=fsdp_config,
    )
    class NonFiniteGuard(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
            for key, value in (logs or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    need(math.isfinite(float(value)), f"non-finite AI-Hub training log {key} at step {state.global_step}")
            return control

    trainer = SFTTrainer(model=model, args=arguments, train_dataset=dataset, processing_class=tokenizer, callbacks=[NonFiniteGuard()])
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    final_model = Path(config.output_dir) / "final_model"
    if config.phase == "full":
        # Full-state materialization is collective under FSDP2.
        trainer.save_model(str(final_model))
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(str(final_model))
        trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None
    failed = False
    if trainer.is_world_process_zero():
        try:
            metrics = finite_metrics(trained.metrics)
            model_files: list[dict[str, Any]] = []
            if config.phase == "full":
                weights = sorted(final_model.glob("*.safetensors"))
                need(bool(weights) and (final_model / "config.json").is_file(), "AI-Hub full SFT final state is unavailable")
                model_files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)} for path in weights]
            payload = {
                "schema_version": "mal2026-official-aihub-rationale-full-sft-complete-v1",
                "status": "completed",
                "run_id": config.run_id,
                "structure": config.structure,
                "phase": config.phase,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "world_size": world_size,
                "global_step": int(trainer.state.global_step),
                "train_records": len(examples),
                "train_metrics": metrics,
                "training_kind": "full_parameter",
                "training_dtype": config.compute_dtype,
                "template": template,
                "aihub_provenance": provenance(),
                "base_model_config_sha256": file_sha(MODEL_PATH / "config.json"),
                "final_model_files": model_files,
                "config": asdict(config),
                "privacy": "aggregate_only_no_rows_prompts_essays_feedback_rationales_ids_or_scores",
            }
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(config.output_dir) / "training_complete.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            failed = True
    state: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state, src=0)
    if state[0] or not isinstance(state[1], dict):
        raise OfficialAIHubSFTError("AI-Hub full SFT completion persistence failed")
    return state[1]
