"""Full-parameter AI-Hub integer pretraining for official decoder score arms."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .official_aihub_score_pretrain import CANONICAL_MANIFEST, load_integer_split
from .official_decoder_score import (
    ARCHITECTURES, AXES, COMPLETION_SCHEMA, MODEL_ID, MODEL_REVISION, ROOT,
    STATE_SCHEMA, _causal_collator, _causal_dataset, _head_collator,
    _head_dataset, file_sha256, generate_integer_predictions,
)
from .official_score_matrix import decode_logits, ordinal_targets, score_metrics


OUTPUT_ROOT = ROOT / "outputs" / "official-decoder-aihub-integer-score-full-pretrain-v1"


class DecoderAIHubPretrainError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise DecoderAIHubPretrainError(message)


@dataclass(frozen=True)
class DecoderAIHubConfig:
    schema_version: str
    run_id: str
    model_id: str
    model_revision: str
    model_path: str
    manifest_path: str
    output_root: str
    architectures: tuple[str, str, str]
    score_fields: tuple[str, str, str]
    integer_target_used: bool
    target_projection: str
    average_target_used: bool
    seed: int
    max_length: int
    training_method: str
    downstream_adaptation: str
    distributed_strategy: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_selection_epochs: float
    eval_steps: int
    early_stopping_patience: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    activation_checkpointing: bool
    fsdp_transformer_layer_class: str
    fsdp_state_dict_type: str
    training_dtype: str

    @classmethod
    def from_json(cls, path: Path, *, require_dependencies: bool = True) -> "DecoderAIHubConfig":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DecoderAIHubPretrainError("decoder AI-Hub config is unreadable") from exc
        _need(isinstance(raw, dict), "decoder AI-Hub config must be an object")
        for field in ("architectures", "score_fields"):
            _need(isinstance(raw.get(field), list), f"{field} must be a list")
            raw[field] = tuple(raw[field])
        _need(set(raw) == set(cls.__dataclass_fields__), "decoder AI-Hub config has missing or unknown fields")
        value = cls(**raw)
        value.validate(require_dependencies=require_dependencies)
        return value

    def validate(self, *, require_dependencies: bool = True) -> None:
        _need(self.schema_version == "mal2026-official-decoder-aihub-integer-full-pretrain-config-v1", "pretrain schema differs")
        _need(self.run_id == "official-decoder-aihub-integer-score-full-pretrain-v1-20260727-001", "pretrain run identity differs")
        _need((self.model_id, self.model_revision) == (MODEL_ID, MODEL_REVISION), "decoder model pin differs")
        _need(self.architectures == ARCHITECTURES and self.score_fields == AXES, "architecture/axis contract differs")
        _need(self.integer_target_used is True and self.target_projection == "official_half_up" and self.average_target_used is False, "integer target contract differs")
        _need(Path(self.manifest_path).resolve() == CANONICAL_MANIFEST.resolve(), "canonical AI-Hub manifest differs")
        _need(Path(self.output_root).resolve() == OUTPUT_ROOT.resolve(), "pretrain output root differs")
        _need((self.seed, self.max_length) == (2026072702, 2048), "seed/length contract differs")
        _need(self.training_method == "full_parameter" and self.downstream_adaptation == "fresh_MAL_LoRA", "full-to-LoRA lineage differs")
        _need(self.distributed_strategy == "fsdp_full_shard_auto_wrap" and self.optimizer == "adafactor", "distributed/optimizer contract differs")
        _need((self.learning_rate, self.weight_decay, self.warmup_ratio) == (1e-5, 0.01, 0.05), "optimization contract differs")
        _need((self.max_selection_epochs, self.eval_steps, self.early_stopping_patience) == (20.0, 100, 3), "selection contract differs")
        _need((self.per_device_train_batch_size, self.per_device_eval_batch_size, self.gradient_accumulation_steps) == (1, 2, 8), "batch contract differs")
        _need(self.activation_checkpointing is True and self.fsdp_transformer_layer_class == "Qwen2DecoderLayer" and self.fsdp_state_dict_type == "FULL_STATE_DICT", "FSDP contract differs")
        _need(self.training_dtype == "bfloat16", "training dtype differs")
        if require_dependencies:
            _need(Path(self.model_path).is_dir() and not Path(self.model_path).is_symlink(), "local decoder snapshot is unavailable")
            _need(Path(self.manifest_path).is_file() and not Path(self.manifest_path).is_symlink(), "AI-Hub manifest is unavailable")

    def identity(self, architecture: str) -> dict[str, Any]:
        _need(architecture in ARCHITECTURES, "unknown pretrain architecture")
        raw = asdict(self)
        for key in ("run_id", "output_root", "architectures"):
            raw.pop(key)
        raw["score_fields"] = list(raw["score_fields"])
        raw["architecture"] = architecture
        return raw


def select_event(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    _need(bool(events), "selection emitted no events")
    return min(events, key=lambda row: (
        float(row["macro_integer_rmse"]), -float(row["macro_integer_spearman"]),
        float(row["macro_continuous_rmse"]), int(row["global_step"]),
    ))


def build_full_model(config: DecoderAIHubConfig, architecture: str) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoModelForCausalLM
    _need(architecture in ARCHITECTURES, "unknown pretrain architecture")
    loader = AutoModelForCausalLM if architecture == "generative" else AutoModel
    backbone = loader.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    backbone.config.use_cache = False
    if architecture == "generative":
        model = backbone
    else:
        hidden = getattr(backbone.config, "hidden_size", None)
        _need(type(hidden) is int and hidden > 0, "decoder hidden size is unavailable")

        class FullDecoderScoreHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = backbone
                self.score_head = nn.Linear(hidden, 3 if architecture == "bounded_regression" else 12)

            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Mapping[str, Any] | None = None) -> None:
                self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

            def gradient_checkpointing_disable(self) -> None:
                self.backbone.gradient_checkpointing_disable()

            def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None, **_: Any) -> Mapping[str, Any]:
                hidden_state = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
                positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
                final = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
                _need(bool((final >= 0).all().item()), "decoder input has no non-padding token")
                logits = self.score_head(hidden_state[torch.arange(hidden_state.shape[0], device=hidden_state.device), final].float())
                result: dict[str, Any] = {"logits": logits}
                if labels is not None:
                    if architecture == "bounded_regression":
                        result["loss"] = functional.mse_loss(decode_logits(logits, architecture)[0], labels.float())
                    else:
                        result["loss"] = functional.binary_cross_entropy_with_logits(logits.reshape(-1, 3, 4), ordinal_targets(labels))
                return result
        model = FullDecoderScoreHead()
    _need(all(parameter.requires_grad for parameter in model.parameters()), "full decoder contains a frozen parameter")
    _need(not any("lora_" in name for name, _ in model.named_parameters()), "LoRA leaked into AI-Hub full pretraining")
    return model


def initialization_contract_sha256(config: DecoderAIHubConfig, architecture: str, model: Any) -> str:
    head_hash = None
    if architecture != "generative":
        digest = sha256()
        for name, tensor in sorted(model.score_head.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(json.dumps(list(value.shape)).encode()); digest.update(value.numpy().tobytes())
        head_hash = digest.hexdigest()
    payload = {"model_id": config.model_id, "model_revision": config.model_revision, "model_path": str(Path(config.model_path).resolve()), "architecture": architecture, "head_initial_sha256": head_hash, "seed": config.seed, "training_method": config.training_method}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact_inventory(directory: Path) -> tuple[list[dict[str, Any]], str]:
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            _need(not path.is_symlink(), "full decoder artifact contains a symlink")
            entries.append({"path": path.relative_to(directory).as_posix(), "size": path.stat().st_size, "sha256": file_sha256(path)})
    _need(bool(entries), "full decoder artifact is empty")
    return entries, sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def exported_tensor_contract(directory: Path, architecture: str) -> dict[str, Any]:
    from safetensors import safe_open
    names: set[str] = set()
    head_shapes: dict[str, list[int]] = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                _need(name not in names, "duplicate exported tensor name")
                names.add(name)
                if name.startswith("score_head."):
                    head_shapes[name] = list(handle.get_tensor(name).shape)
    _need(bool(names), "full decoder export has no tensors")
    if architecture == "generative":
        _need(not head_shapes and any(name.startswith("model.") for name in names), "generative full-state scope differs")
    else:
        width = 3 if architecture == "bounded_regression" else 12
        _need(head_shapes == {"score_head.weight": [width, 3584], "score_head.bias": [width]}, "dedicated full-state head shapes differ")
        _need(any(name.startswith("backbone.") for name in names), "dedicated full-state has no backbone")
    return {"tensor_count": len(names), "score_head_tensor_shapes": head_shapes, "complete_full_parameter_state": True}


def _wait_output(path: Path) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        _need(not path.exists(), f"refusing to reuse output: {path}")
        path.mkdir(parents=True)
    else:
        deadline = time.monotonic() + 60
        while not path.is_dir() and time.monotonic() < deadline:
            time.sleep(0.05)
        _need(path.is_dir(), "rank zero did not create output")


def _load_selection(config: DecoderAIHubConfig, architecture: str, path: Path) -> tuple[int, int, str, Mapping[str, Any]]:
    _need(path.is_file() and not path.is_symlink(), "selection completion is unavailable")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _need(payload.get("status") == "completed" and payload.get("phase") == "selection" and payload.get("architecture") == architecture, "selection identity differs")
    _need(payload.get("identity") == config.identity(architecture), "selection/refit configuration differs")
    selected = payload.get("selection", {}).get("selected_event", {})
    step, horizon = selected.get("global_step"), selected.get("selection_schedule_max_steps")
    initial = payload.get("initialization_contract_sha256")
    _need(type(step) is int and step > 0 and type(horizon) is int and horizon >= step, "selected-step contract differs")
    _need(isinstance(initial, str) and len(initial) == 64, "initialization contract differs")
    return step, horizon, initial, selected


def _distributed_generative_metrics(trainer: Any, tokenizer: Any, rows: Sequence[Any], config: DecoderAIHubConfig) -> dict[str, float]:
    import torch
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    world = torch.distributed.get_world_size() if distributed else 1
    local_rows = list(rows)[rank::world]
    wrapped = trainer.model_wrapped
    if distributed:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        _need(isinstance(wrapped, FSDP), "distributed generative evaluation requires FSDP")
        with FSDP.summon_full_params(wrapped, recurse=True, writeback=False, rank0_only=False):
            predictions, invalid = generate_integer_predictions(wrapped.module, tokenizer, local_rows, config)
    else:
        predictions, invalid = generate_integer_predictions(trainer.model, tokenizer, local_rows, config)
    gathered: list[Any] = [None] * world
    local = {"labels": [list(row.labels) for row in local_rows], "predictions": [list(row) for row in predictions], "invalid": invalid}
    if distributed:
        torch.distributed.all_gather_object(gathered, local)
    else:
        gathered[0] = local
    labels = [row for shard in gathered for row in shard["labels"]]
    predictions_all = [row for shard in gathered for row in shard["predictions"]]
    scored = score_metrics(labels, predictions_all, predictions_all)
    invalid_total = sum(shard["invalid"] for shard in gathered)
    return {
        "macro_integer_rmse": float(scored["macro_integer_rmse"]),
        "macro_integer_spearman": float(scored["macro_integer_spearman"]),
        "macro_continuous_rmse": float(scored["macro_continuous_rmse"]),
        "strict_parse_rate": 1.0 - invalid_total / len(labels),
        "invalid_output_count": float(invalid_total),
    }


def run_training(config: DecoderAIHubConfig, architecture: str, phase: str, *, smoke: bool = False, selection_metadata: Path | None = None) -> dict[str, Any]:
    config.validate(require_dependencies=True)
    _need(architecture in ARCHITECTURES and phase in {"selection", "refit"}, "pretrain architecture/phase differs")
    _need((phase == "selection") == (selection_metadata is None), "refit requires selection metadata")
    import torch
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

    output = Path(config.output_root) / config.run_id / (f"smoke-{architecture}-{phase}" if smoke else f"{architecture}-{phase}")
    _wait_output(output)
    train_split = "selection_train" if phase == "selection" else "refit_train"
    train_rows, train_sha = load_integer_split(train_split, Path(config.manifest_path))
    dev_rows = None
    dev_sha = None
    if phase == "selection":
        dev_rows, dev_sha = load_integer_split("selection_dev", Path(config.manifest_path))
    if smoke:
        train_rows = train_rows[:4]
        if dev_rows is not None: dev_rows = dev_rows[:4]

    selected_steps = schedule_horizon = None
    selected_event: Mapping[str, Any] | None = None
    selection_initial = None
    if phase == "refit":
        assert selection_metadata is not None
        selected_steps, schedule_horizon, selection_initial, selected_event = _load_selection(config, architecture, selection_metadata)

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = build_full_model(config, architecture)
    initial_contract = initialization_contract_sha256(config, architecture, model)
    if phase == "refit": _need(initial_contract == selection_initial, "selection/refit initialization replay differs")
    dataset = _causal_dataset(train_rows, tokenizer, config, "essay", None) if architecture == "generative" else _head_dataset(train_rows, tokenizer, config, "essay", None)
    dev_dataset = None
    if dev_rows is not None:
        dev_dataset = _causal_dataset(dev_rows, tokenizer, config, "essay", None) if architecture == "generative" else _head_dataset(dev_rows, tokenizer, config, "essay", None)
    events: list[dict[str, Any]] = []

    class SelectionCallback(TrainerCallback):
        def __init__(self) -> None:
            self.best: tuple[float, float, float, int] | None = None
            self.stale = 0
        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Mapping[str, Any] | None = None, **_: Any) -> Any:
            _need(metrics is not None, "selection evaluation has no metrics")
            event = {"global_step": int(state.global_step), "selection_schedule_max_steps": int(state.max_steps), "epoch": float(state.epoch), "macro_integer_rmse": float(metrics["eval_macro_integer_rmse"]), "macro_integer_spearman": float(metrics["eval_macro_integer_spearman"]), "macro_continuous_rmse": float(metrics["eval_macro_continuous_rmse"]), "strict_parse_rate": metrics.get("eval_strict_parse_rate")}
            events.append(event)
            key = (event["macro_integer_rmse"], -event["macro_integer_spearman"], event["macro_continuous_rmse"], event["global_step"])
            if self.best is None or key < self.best: self.best, self.stale = key, 0
            else:
                self.stale += 1
                if self.stale >= config.early_stopping_patience: control.should_training_stop = True
            return control

    class StopAtSelectedStep(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            if selected_steps is not None and int(state.global_step) >= selected_steps: control.should_training_stop = True
            return control

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    fsdp_config = {"transformer_layer_cls_to_wrap": [config.fsdp_transformer_layer_class], "activation_checkpointing": config.activation_checkpointing, "use_orig_params": True, "state_dict_type": config.fsdp_state_dict_type, "sync_module_states": True, "limit_all_gathers": True} if distributed else None
    args = TrainingArguments(
        output_dir=str(output / "trainer"), do_train=True, do_eval=phase == "selection", eval_strategy="steps" if phase == "selection" else "no", save_strategy="no",
        eval_steps=(1 if smoke else config.eval_steps) if phase == "selection" else None,
        max_steps=1 if smoke else (schedule_horizon if schedule_horizon is not None else -1), num_train_epochs=config.max_selection_epochs if phase == "selection" else 1.0,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio, optim=config.optimizer,
        per_device_train_batch_size=config.per_device_train_batch_size, per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=1 if smoke else config.gradient_accumulation_steps, bf16=True, tf32=True, report_to=[], remove_unused_columns=False,
        gradient_checkpointing=config.activation_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False}, fsdp="full_shard auto_wrap" if distributed else None, fsdp_config=fsdp_config,
        dataloader_num_workers=0, ddp_find_unused_parameters=False, logging_steps=1 if smoke else 5, save_only_model=True, seed=config.seed, data_seed=config.seed,
    )

    class GenerativeTrainer(Trainer):
        def evaluate(self, eval_dataset: Any = None, ignore_keys: Any = None, metric_key_prefix: str = "eval") -> dict[str, float]:
            _need(dev_rows is not None, "generative selection dev rows are unavailable")
            metrics = {f"{metric_key_prefix}_{key}": float(value) for key, value in _distributed_generative_metrics(self, tokenizer, dev_rows, config).items()}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            self.model.train()
            return metrics

    trainer_class = GenerativeTrainer if architecture == "generative" else Trainer
    trainer = trainer_class(model=model, args=args, train_dataset=dataset, eval_dataset=dev_dataset, data_collator=_causal_collator(tokenizer) if architecture == "generative" else _head_collator(tokenizer), compute_metrics=(lambda result: _head_metrics(result, architecture)) if architecture != "generative" and phase == "selection" else None, callbacks=[SelectionCallback()] if phase == "selection" else [StopAtSelectedStep()])
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    shared: list[Any] = [events if trainer.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized(): torch.distributed.broadcast_object_list(shared, src=0)
    events = shared[0]
    _need(isinstance(events, list), "selection event broadcast differs")
    if phase == "selection": selected_event = select_event(events)
    else: _need(int(trainer.state.global_step) == selected_steps, "refit did not stop at exact selected step")

    payload: dict[str, Any] = {
        "schema_version": COMPLETION_SCHEMA, "status": "completed", "mode": "gpu0_one_update_smoke" if smoke else "full", "run_id": config.run_id,
        "phase": phase, "architecture": architecture, "identity": config.identity(architecture), "initialization": "public", "input_view": "essay",
        "score_fields": list(AXES), "integer_target_used": True, "target_projection": "official_half_up", "average_read": False, "average_target_used": False,
        "rationale_output_used": False, "canonical_validation": None, "canonical_validation_access": False, "training_method": "full_parameter", "downstream_adaptation": "fresh_MAL_LoRA",
        "data": {"dataset": "aihub_human_feedback_v1", "split": train_split, "records": len(train_rows), "sha256": train_sha, "selection_dev_sha256": dev_sha},
        "initialization_contract_sha256": initial_contract, "selection": {"events": events, "selected_event": selected_event, "source": "AI-Hub selection_dev only"},
        "trainer": {"global_step": int(trainer.state.global_step), "scheduler_horizon_steps": int(trainer.state.max_steps), "exact_selected_step_stop": phase == "refit", "metrics": {key: float(value) for key, value in trained.metrics.items() if isinstance(value, (int, float))}},
        "reproducibility": {"seed": config.seed, "model_id": config.model_id, "model_revision": config.model_revision, "training_dtype": config.training_dtype, "visible_gpu_scope": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "privacy": "aggregate_only_no_rows_prompts_essays_feedback_ids_predictions_or_model_outputs_persisted",
    }
    if phase == "refit":
        if smoke:
            payload["state"] = {"schema_version": STATE_SCHEMA, "training_method": "full_parameter", "export_skipped": True, "reason": "one_update_smoke"}
        else:
            artifact = output / "full_model"
            trainer.save_model(str(artifact))
            trainer.accelerator.wait_for_everyone()
            metadata_path = output / "full_model_state.json"
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(artifact)
                inventory, artifact_sha = artifact_inventory(artifact)
                metadata = {"schema_version": STATE_SCHEMA, "architecture": architecture, "model_id": config.model_id, "model_revision": config.model_revision, "training_method": "full_parameter", "score_fields": list(AXES), "integer_target_used": True, "average_target_used": False, "state_scope": "complete_full_parameter_backbone_plus_matched_head", "artifact_path": str(artifact.resolve()), "artifact_sha256": artifact_sha, "inventory": inventory, **exported_tensor_contract(artifact, architecture)}
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                payload["state"] = {**metadata, "metadata_path": str(metadata_path.resolve()), "metadata_sha256": file_sha256(metadata_path)}
            state_shared: list[Any] = [payload.get("state") if trainer.is_world_process_zero() else None]
            if torch.distributed.is_available() and torch.distributed.is_initialized(): torch.distributed.broadcast_object_list(state_shared, src=0)
            payload["state"] = state_shared[0]
    completion = output / "training_complete.json"
    if trainer.is_world_process_zero(): completion.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trainer.accelerator.wait_for_everyone()
    return payload


def _head_metrics(result: Any, architecture: str) -> dict[str, float]:
    import torch
    continuous, integers, violations = decode_logits(torch.as_tensor(result.predictions), architecture)
    scored = score_metrics(result.label_ids.tolist(), continuous.tolist(), integers.tolist(), violations.tolist())
    return {
        "macro_integer_rmse": float(scored["macro_integer_rmse"]),
        "macro_integer_spearman": float(scored["macro_integer_spearman"]),
        "macro_continuous_rmse": float(scored["macro_continuous_rmse"]),
        "ordinal_monotonic_violation_rate": float(scored["ordinal_monotonic_violation_rate"]),
    }
