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

    os.environ.setdefault("WANDB_LOG_MODEL", "false")
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
    # Save the best selection model (or final refit model) through Trainer, not
    # a bespoke state/optimizer/DDP implementation.
    final_dir = output / "final_model"
    trainer.save_model(str(final_dir))
    _need((final_dir / "model.safetensors").is_file(), "Trainer did not write a safe final model state")
    metrics = trainer.evaluate() if config.phase == "selection" else {}
    selected_global_step = int(trainer.state.global_step)
    if config.phase == "selection" and trainer.state.best_global_step is not None:
        selected_global_step = int(trainer.state.best_global_step)
    _need(selected_global_step > 0, "Trainer completed without any optimizer update")
    payload = {
        "status": "completed", "run_id": config.run_id, "phase": config.phase, "selected_global_step": selected_global_step,
        "trainer_global_step": int(trainer.state.global_step), "train_metrics": {key: float(value) for key, value in train_result.metrics.items() if isinstance(value, (int, float))},
        "selection_metrics": {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        "identity": _config_identity(config), "config": asdict(config), "model_state_sha256": _sha256(final_dir / "model.safetensors"),
        "privacy": "aggregate_only_no_rows_prompts_essays_feedback_ids_predictions_or_model_outputs_persisted",
    }
    _write_complete(output, payload)
    return payload
