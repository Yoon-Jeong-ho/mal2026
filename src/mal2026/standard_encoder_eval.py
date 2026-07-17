"""Aggregate-only final/source evaluation for standard Trainer encoder states."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .standard_decoder_data import DEFAULT_MANIFEST, ROOT, StandardDecoderContractError, load_frozen_validation, load_prepared_split
from .standard_encoder_data import build_encoder_dataset, encoder_collator
from .standard_encoder_model import EncoderModelSpec, build_encoder_regressor, build_encoder_tokenizer
from .standard_encoder_train import RUN_ROOT, StandardEncoderConfig, _metric_function

EVAL_ROOT = ROOT / "outputs" / "standard-encoder-evals"


class StandardEncoderEvaluationError(StandardDecoderContractError):
    """Raised if an encoder evaluation would violate the frozen contract."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise StandardEncoderEvaluationError(message)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class StandardEncoderEvalConfig:
    run_id: str
    source: str  # selection_dev | frozen_validation
    training_metadata_path: str
    prepared_manifest: str
    validation_sha256: str
    output_dir: str
    per_device_eval_batch_size: int = 4
    wandb_project: str = "mal2026-korean-writing-scoring"
    wandb_entity: str | None = None

    @classmethod
    def from_json(cls, path: Path) -> "StandardEncoderEvalConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        _need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "standard encoder eval config has missing or unknown fields")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        _need(self.source in {"selection_dev", "frozen_validation"}, "invalid evaluation source")
        _need(Path(self.prepared_manifest).resolve() == DEFAULT_MANIFEST.resolve(), "evaluation must use canonical prepared manifest")
        output = Path(self.output_dir)
        _need(output.is_absolute() and output.parent == EVAL_ROOT.resolve() and not output.exists(), "evaluation output must be a new direct child of outputs/standard-encoder-evals")
        _need(self.per_device_eval_batch_size > 0, "per_device_eval_batch_size must be positive")
        if self.source == "frozen_validation":
            _need(isinstance(self.validation_sha256, str) and len(self.validation_sha256) == 64 and all(char in "0123456789abcdef" for char in self.validation_sha256), "frozen validation requires a SHA-256")


def _load_training_metadata(config: StandardEncoderEvalConfig) -> tuple[Mapping[str, Any], EncoderModelSpec, Path]:
    path = Path(config.training_metadata_path)
    _need(path.is_absolute() and path.name == "standard_encoder_training_complete.json", "training metadata filename is invalid")
    _need(path.parent.parent == RUN_ROOT.resolve() and path.parent.is_dir() and not path.parent.is_symlink(), "training metadata must be in a canonical standard encoder run")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StandardEncoderEvaluationError("unable to read training metadata") from exc
    _need(isinstance(metadata, dict) and metadata.get("status") == "completed", "training metadata is not completed")
    expected_phase = "selection" if config.source == "selection_dev" else "refit"
    _need(metadata.get("phase") == expected_phase, "evaluation source requires its corresponding training phase")
    saved_hash = metadata.get("model_state_sha256")
    state = path.parent / "final_model" / "model.safetensors"
    _need(isinstance(saved_hash, str) and state.is_file() and _sha256(state) == saved_hash, "final model state checksum failed")
    raw_config = metadata.get("config")
    _need(isinstance(raw_config, dict), "training metadata lacks config")
    if isinstance(raw_config.get("lora_target_modules"), list):
        raw_config["lora_target_modules"] = tuple(raw_config["lora_target_modules"])
    saved_config = StandardEncoderConfig(**raw_config)
    spec = EncoderModelSpec.from_mapping(saved_config.model_spec_mapping())
    return metadata, spec, state


def run_standard_encoder_evaluation(config: StandardEncoderEvalConfig) -> dict[str, Any]:
    """Use ``Trainer.predict`` and persist only aggregate metrics/provenance."""
    config.validate()
    metadata, spec, state_path = _load_training_metadata(config)
    rows = load_prepared_split("selection_dev", Path(config.prepared_manifest)) if config.source == "selection_dev" else load_frozen_validation(config.validation_sha256)
    try:
        from safetensors.torch import load_model
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover - runtime-only imports
        raise RuntimeError("standard encoder evaluation requires the project .venv-standard") from exc
    tokenizer = build_encoder_tokenizer(spec)
    model = build_encoder_regressor(spec)
    missing, unexpected = load_model(model, str(state_path), strict=False)
    _need(not missing and not unexpected, "saved standard Trainer model state does not match the encoder architecture")
    # Reserve the preflight-validated output before constructing Trainer.  Recent
    # Transformers versions create ``output_dir`` in ``TrainingArguments``;
    # leaving directory creation until after prediction would therefore turn a
    # successful evaluation into a FileExistsError.  ``exist_ok=False`` keeps
    # the no-overwrite contract fail-closed even if another process races this
    # evaluator after validation.
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    args = TrainingArguments(
        output_dir=str(output), do_train=False, do_eval=False, per_device_eval_batch_size=config.per_device_eval_batch_size,
        bf16=True, tf32=True, report_to=[], remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, data_collator=encoder_collator(tokenizer), compute_metrics=_metric_function)
    result = trainer.predict(build_encoder_dataset(rows, tokenizer, 2048), metric_key_prefix="eval")
    metrics = {key: float(value) for key, value in result.metrics.items() if isinstance(value, (int, float))}
    _need("eval_primary_macro_mae" in metrics, "Trainer prediction did not return macro MAE")
    payload = {
        "status": "completed", "run_id": config.run_id, "source": config.source,
        "training_run_id": metadata.get("run_id"), "model_state_sha256": _sha256(state_path),
        "metrics": metrics, "config": asdict(config),
        "privacy": "aggregate_only_no_rows_prompts_essays_feedback_ids_predictions_or_model_outputs_persisted",
    }
    (output / "aggregate_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        import wandb
        run = wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=config.run_id, config={"source": config.source, "training_run_id": metadata.get("run_id")})
        run.log({f"eval/{key.removeprefix('eval_')}": value for key, value in metrics.items()})
        run.finish()
    except ImportError:
        pass
    return payload
