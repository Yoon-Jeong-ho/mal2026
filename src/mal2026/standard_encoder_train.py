"""Hugging Face Trainer lifecycle for the two encoder regressors.

Selection uses only the prepared human-feedback development split.  Refit uses
all prepared Training rows for the pre-recorded selected number of Trainer
updates.  The frozen validation set is intentionally not readable here; it is
reserved for :mod:`standard_encoder_eval` after a refit artifact exists.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from numbers import Real
import os
from pathlib import Path
from typing import Any, Mapping

from .metrics import compute_regression_metrics
from .standard_decoder_data import DEFAULT_MANIFEST, ROOT, SCORE_FIELDS, StandardDecoderContractError, load_prepared_split
from .standard_encoder_data import build_encoder_dataset, encoder_collator
from .standard_encoder_model import EncoderModelSpec, build_encoder_regressor, build_encoder_tokenizer

RUN_ROOT = ROOT / "outputs" / "standard-encoder-runs"


class StandardEncoderTrainingError(StandardDecoderContractError):
    """The immutable standard encoder lifecycle contract was violated."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise StandardEncoderTrainingError(message)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _direct_child(path: Path, parent: Path, label: str) -> Path:
    resolved_parent = parent.resolve()
    _need(path.is_absolute() and path.parent == resolved_parent and not path.exists(), f"{label} must be a new direct child of {resolved_parent}")
    return path


@dataclass(frozen=True)
class StandardEncoderConfig:
    run_id: str
    phase: str  # selection | refit
    backbone: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    model_path: str
    prepared_manifest: str
    output_dir: str
    max_length: int = 2048
    seed: int = 2026
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    num_train_epochs: float = 20.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 5
    early_stopping_patience: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    nv_snapshot_dir: str | None = None
    nv_review: Mapping[str, Any] | None = None
    selection_metadata_path: str | None = None
    wandb_project: str = "mal2026-korean-writing-scoring"
    wandb_entity: str | None = None

    @classmethod
    def from_json(cls, path: Path) -> "StandardEncoderConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        _need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "standard encoder config has missing or unknown fields")
        if isinstance(raw.get("lora_target_modules"), list):
            raw["lora_target_modules"] = tuple(raw["lora_target_modules"])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        _need(self.phase in {"selection", "refit"}, "phase must be selection or refit")
        _need(Path(self.prepared_manifest).resolve() == DEFAULT_MANIFEST.resolve(), "encoder must use the canonical aggregate prepared manifest")
        _need(bool(self.run_id) and Path(self.output_dir).is_absolute(), "run_id and absolute output_dir are required")
        _direct_child(Path(self.output_dir), RUN_ROOT, "output_dir")
        _need(self.max_length == 2048, "encoder max_length is frozen at 2048")
        _need(self.learning_rate > 0 and self.weight_decay >= 0 and 0 <= self.warmup_ratio < 1 and self.num_train_epochs > 0, "invalid optimization values")
        _need(self.per_device_train_batch_size > 0 and self.per_device_eval_batch_size > 0 and self.gradient_accumulation_steps > 0, "invalid batch settings")
        _need(self.logging_steps > 0 and self.early_stopping_patience > 0, "invalid logging/early-stopping settings")
        if self.phase == "selection":
            _need(self.eval_steps > 0 and self.save_steps == self.eval_steps and self.selection_metadata_path is None, "selection requires matching eval/save steps and no prior selection artifact")
        else:
            _need(self.selection_metadata_path is not None and self.eval_steps == 0 and self.save_steps == 0, "refit uses prior selection steps and must not evaluate")
        EncoderModelSpec.from_mapping(self.model_spec_mapping())

    def model_spec_mapping(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone, "model_id": self.model_id, "revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision, "model_path": self.model_path,
            "pooling": "last_nonpad" if self.backbone == "qwen3_embedding" else "remote_sentence_embedding",
            "normalize_embeddings": True, "lora_target_modules": list(self.lora_target_modules),
            "lora_r": self.lora_r, "lora_alpha": self.lora_alpha, "lora_dropout": self.lora_dropout,
            "nv_snapshot_dir": self.nv_snapshot_dir, "nv_review": self.nv_review,
        }


def _configure_wandb(config: StandardEncoderConfig) -> None:
    """Set the complete W&B routing contract before Trainer initializes callbacks."""
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_PROJECT"] = config.wandb_project
    os.environ["WANDB_RUN_NAME"] = config.run_id
    if config.wandb_entity:
        os.environ["WANDB_ENTITY"] = config.wandb_entity
    else:
        os.environ.pop("WANDB_ENTITY", None)


def _config_identity(config: StandardEncoderConfig) -> dict[str, Any]:
    """Fields that must remain unchanged between selection and refit."""
    data = json.loads(json.dumps(asdict(config), ensure_ascii=False))
    for key in ("run_id", "phase", "output_dir", "selection_metadata_path", "num_train_epochs", "eval_steps", "save_steps", "wandb_project", "wandb_entity"):
        data.pop(key)
    return data


def _load_selection_steps(config: StandardEncoderConfig) -> int:
    assert config.selection_metadata_path is not None
    candidate = Path(config.selection_metadata_path)
    _need(candidate.is_absolute() and candidate.name == "standard_encoder_training_complete.json", "selection metadata filename is invalid")
    expected_parent = RUN_ROOT.resolve()
    _need(candidate.parent.parent == expected_parent and candidate.parent.is_dir() and not candidate.is_symlink(), "selection metadata must reside under a standard encoder run")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StandardEncoderTrainingError("unable to read selection metadata") from exc
    _need(isinstance(payload, dict) and payload.get("phase") == "selection", "prior artifact is not a selection run")
    _need(payload.get("identity") == _config_identity(config), "selection/refit architecture or optimization contract differs")
    step = payload.get("selected_global_step")
    _need(isinstance(step, int) and step > 0, "selection artifact lacks a positive selected_global_step")
    return step


def _metric_function(eval_prediction: Any) -> dict[str, float]:
    predictions, labels = eval_prediction.predictions, eval_prediction.label_ids
    # Some Trainer configurations wrap model output in a tuple.
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    clipped = [[min(5.0, max(1.0, float(value))) for value in row] for row in predictions]
    targets = [{field: float(value) for field, value in zip(SCORE_FIELDS, row, strict=True)} for row in labels]
    predicted = [{field: float(value) for field, value in zip(SCORE_FIELDS, row, strict=True)} for row in clipped]
    result = compute_regression_metrics(targets, predicted)
    flattened: dict[str, float] = {}
    maes: list[float] = []
    for field in SCORE_FIELDS:
        metrics = result["per_target"][field]
        maes.append(float(metrics["mae"]))
        for name, value in metrics.items():
            if value is not None:
                flattened[f"{field}_{name}"] = float(value)
    flattened["primary_macro_mae"] = sum(maes) / len(maes)
    return flattened


def _write_complete(output: Path, payload: Mapping[str, Any]) -> None:
    path = output / "standard_encoder_training_complete.json"
    _need(not path.exists(), "training completion artifact already exists")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selection_metrics_at_best_step(trainer_or_state: Any, selected_global_step: int) -> dict[str, float]:
    """Read the distributed Trainer's already-recorded best-dev metrics.

    Calling ``Trainer.evaluate`` after ``train`` on world process zero alone
    would make Accelerate's evaluation gathers wait for ranks that are no
    longer participating. Selection evaluation has already been performed by
    Trainer on every rank at each configured evaluation step. The immutable
    best checkpoint identifies precisely which aggregate event to persist.
    """
    trainer_state = getattr(trainer_or_state, "state", trainer_or_state)
    history = getattr(trainer_state, "log_history", None)
    _need(isinstance(history, list), "Trainer log_history must be a list")
    for event in reversed(history):
        _need(isinstance(event, Mapping), "Trainer log_history event must be a mapping")
        if event.get("step") != selected_global_step or "eval_primary_macro_mae" not in event:
            continue
        metrics = {
            key: float(value)
            for key, value in event.items()
            if key.startswith("eval_") and isinstance(value, (int, float))
        }
        _need("eval_primary_macro_mae" in metrics, "best selection event lacks macro MAE")
        return metrics
    raise StandardEncoderTrainingError("best selection checkpoint has no recorded distributed evaluation metrics")


def _finite_metric(value: Any, label: str) -> float:
    """Convert one persisted metric to a finite JSON-safe float or fail closed."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StandardEncoderTrainingError(f"{label} must be a finite numeric metric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise StandardEncoderTrainingError(f"{label} must be finite")
    return parsed


def _persisted_finite_metrics(metrics: Any, label: str) -> dict[str, float]:
    """Return exactly the numeric metric subset allowed in completion JSON."""
    _need(isinstance(metrics, Mapping), f"{label} metrics must be a mapping")
    result: dict[str, float] = {}
    for key, value in metrics.items():
        # Match the completion artifact's numeric-only contract, but validate
        # every retained value before JSON serialization can spell NaN/Infinity.
        if isinstance(value, Real) and not isinstance(value, bool):
            result[str(key)] = _finite_metric(value, f"{label} metric {key}")
    return result


def _validate_encoder_metric_health(
    phase: str, trainer_state: Any, train_metrics: Any
) -> tuple[dict[str, float], int, dict[str, float], float | None]:
    """Gate all completion metrics before rank-zero exports a final model.

    This observes maintained Trainer state only. It does not change model
    optimization or DDP behavior. The returned mappings are the exact numeric
    mappings persisted in provenance, so a completed artifact cannot encode a
    non-finite metric.
    """
    _need(phase in {"selection", "refit"}, "invalid encoder health phase")
    serialized_train = _persisted_finite_metrics(train_metrics, "Trainer train")
    _finite_metric(serialized_train.get("train_loss"), "Trainer train_loss")
    global_step = getattr(trainer_state, "global_step", None)
    _need(isinstance(global_step, int) and not isinstance(global_step, bool) and global_step > 0, "Trainer global_step must be positive")
    if phase == "refit":
        return serialized_train, global_step, {}, None

    best_step = getattr(trainer_state, "best_global_step", None)
    _need(
        isinstance(best_step, int) and not isinstance(best_step, bool) and 0 < best_step <= global_step,
        "selection best_global_step must be a completed positive update",
    )
    best_checkpoint = getattr(trainer_state, "best_model_checkpoint", None)
    _need(isinstance(best_checkpoint, str) and bool(best_checkpoint.strip()), "selection best checkpoint is missing")
    best_metric = _finite_metric(getattr(trainer_state, "best_metric", None), "selection Trainer best_metric")
    selection_metrics = _persisted_finite_metrics(
        _selection_metrics_at_best_step(trainer_state, best_step),
        "selection",
    )
    primary = _finite_metric(selection_metrics.get("eval_primary_macro_mae"), "selection eval_primary_macro_mae")
    # Macro MAE over four 1--5 targets has this bounded range because encoder
    # metric computation clips predictions before calculating the score.
    _need(0.0 <= primary <= 4.0, "selection eval_primary_macro_mae is outside the score-contract range")
    _need(
        math.isclose(best_metric, primary, rel_tol=1e-9, abs_tol=1e-12),
        "selection best_metric does not match the recorded best macro MAE",
    )
    return serialized_train, best_step, selection_metrics, best_metric


def _rank_zero_finalize(
    trainer: Any,
    config: StandardEncoderConfig,
    output: Path,
    train_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Export and document the final model exactly once on world process zero."""
    # This health gate is deliberately before save_model and _write_complete:
    # no NaN/Infinity metric may produce a completed artifact or success signal.
    serialized_train, selected_global_step, selection_metrics, best_metric = _validate_encoder_metric_health(
        config.phase, trainer.state, train_metrics
    )
    final_dir = output / "final_model"
    trainer.save_model(str(final_dir))
    state_path = final_dir / "model.safetensors"
    _need(state_path.is_file(), "Trainer did not write a safe final model state")
    payload = {
        "status": "completed",
        "run_id": config.run_id,
        "phase": config.phase,
        "selected_global_step": selected_global_step,
        "trainer_global_step": int(trainer.state.global_step),
        "train_metrics": serialized_train,
        "selection_metrics": selection_metrics,
        "selection_best_metric": best_metric,
        "identity": _config_identity(config),
        "config": asdict(config),
        "model_state_sha256": _sha256(state_path),
        "privacy": "aggregate_only_no_rows_prompts_essays_feedback_ids_predictions_or_model_outputs_persisted",
    }
    _write_complete(output, payload)
    return payload


def _broadcast_rank_zero_finalization(
    torch: Any, trainer: Any, payload: dict[str, Any] | None, failed: bool
) -> dict[str, Any]:
    """Synchronize successful rank-zero publication or fail every DDP rank.

    A generic status rather than a caught exception string is broadcast so a
    local filesystem error cannot accidentally be turned into shared telemetry.
    All ranks enter this collective after training; a rank-zero save/write
    failure therefore fails workers promptly instead of leaving them blocked at
    a later barrier.
    """
    message: list[Any] = [bool(failed), payload]
    distributed = torch.distributed
    if distributed.is_available() and distributed.is_initialized():
        distributed.broadcast_object_list(message, src=0)
    if bool(message[0]):
        raise StandardEncoderTrainingError("world-process-zero finalization failed; inspect rank-zero stderr")
    _need(
        isinstance(message[1], dict) and message[1].get("status") == "completed",
        "rank-zero finalization returned no completion payload",
    )
    return message[1]


def run_standard_encoder(config: StandardEncoderConfig) -> dict[str, Any]:
    """Run only the maintained Transformers ``Trainer`` lifecycle."""
    config.validate()
    try:
        import torch
        from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover - runtime-only imports
        raise RuntimeError("standard encoder requires the project .venv-standard") from exc

    spec = EncoderModelSpec.from_mapping(config.model_spec_mapping())
    train_rows = load_prepared_split("selection_train" if config.phase == "selection" else "refit_train", Path(config.prepared_manifest))
    dev_rows = load_prepared_split("selection_dev", Path(config.prepared_manifest)) if config.phase == "selection" else None
    selected_steps = _load_selection_steps(config) if config.phase == "refit" else None
    tokenizer = build_encoder_tokenizer(spec)
    train_dataset = build_encoder_dataset(train_rows, tokenizer, config.max_length)
    eval_dataset = build_encoder_dataset(dev_rows, tokenizer, config.max_length) if dev_rows is not None else None
    model = build_encoder_regressor(spec)

    _configure_wandb(config)
    set_seed(config.seed)
    output = Path(config.output_dir)
    args = TrainingArguments(
        output_dir=str(output), overwrite_output_dir=False, do_train=True, do_eval=config.phase == "selection",
        eval_strategy="steps" if config.phase == "selection" else "no",
        save_strategy="steps" if config.phase == "selection" else "no",
        eval_steps=config.eval_steps if config.phase == "selection" else None,
        save_steps=config.save_steps if config.phase == "selection" else None,
        logging_steps=config.logging_steps, logging_strategy="steps", learning_rate=config.learning_rate,
        weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio, num_train_epochs=config.num_train_epochs,
        max_steps=selected_steps if selected_steps is not None else -1,
        per_device_train_batch_size=config.per_device_train_batch_size, per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps, bf16=True, tf32=True,
        save_total_limit=2 if config.phase == "selection" else None,
        load_best_model_at_end=config.phase == "selection", metric_for_best_model="primary_macro_mae" if config.phase == "selection" else None,
        greater_is_better=False if config.phase == "selection" else None,
        report_to=["wandb"], run_name=config.run_id, remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=True, ddp_find_unused_parameters=False,
        seed=config.seed, data_seed=config.seed, save_only_model=False,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)] if config.phase == "selection" else []
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=encoder_collator(tokenizer), compute_metrics=_metric_function if config.phase == "selection" else None,
        callbacks=callbacks,
    )
    train_result = trainer.train()
    # Trainer's train/eval/checkpoint collectives have completed on every rank.
    # Only the global main process may now touch the final-model/artifact paths.
    trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None
    failed = False
    if trainer.is_world_process_zero():
        try:
            payload = _rank_zero_finalize(trainer, config, output, train_result.metrics)
        except Exception:
            failed = True
    result = _broadcast_rank_zero_finalization(torch, trainer, payload, failed)
    # No process exits until it has received the rank-zero status.  On failure
    # every rank raises above; on success this makes the published files visible
    # before a launcher reclaims any worker.
    trainer.accelerator.wait_for_everyone()
    return result
