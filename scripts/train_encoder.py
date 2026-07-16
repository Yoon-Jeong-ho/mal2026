#!/usr/bin/env python3
"""Accelerate/DDP training entry point for four-target encoder regression.

Expected config shape follows ``mal2026.config``'s encoder template.  In
addition to the template fields, ``model`` must explicitly set
``normalize_embeddings``, ``regression_loss: "mse"``, and
``loss_reduction: "mean"``. ``optimization`` must set ``weight_decay``,
``num_workers``, ``early_stopping_min_delta``, and
``early_stopping_patience``.  Qwen requires ``pooling: "last_nonpad"`` and
NV requires ``pooling: "remote_sentence_embedding"``.

The input JSONL contract is owned by ``mal2026.data_contract``: it contains
``prompt``, ``essay``, and a four-field ``score`` object.  This runner never
logs those fields to W&B and writes only aggregate development metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mal2026.encoder_modeling import (  # noqa: E402
    EncoderContractError,
    EncoderModelSpec,
    SCORE_FIELDS,
    load_nv_remote_code_review,
    verify_nv_snapshot,
)


class TrainingContractError(ValueError):
    """Raised before any model/data loading for an incomplete run config."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingContractError(message)


def _read_config(path: Path) -> tuple[Mapping[str, Any], str]:
    """Use the shared fail-closed config loader before encoder-specific checks."""

    from mal2026.config import load_experiment_config

    return load_experiment_config(path)


def _encoder_spec_from_config(value: Mapping[str, Any], args: argparse.Namespace) -> EncoderModelSpec:
    """Translate the documented shared config surface into this runner's spec."""

    kind = value.get("run_kind")
    _need(kind in {"encoder-qwen3", "encoder-nvembed"}, "train_encoder accepts only encoder run kinds")
    model = value["model"]
    adapter = value["adapter"]
    _need(isinstance(model, Mapping) and isinstance(adapter, Mapping), "validated config lost model/adapter objects")
    is_nv = kind == "encoder-nvembed"
    spec_value: dict[str, Any] = {
        "backbone": "nv_embed_v2" if is_nv else "qwen3_embedding",
        "model_id": model.get("id"),
        "revision": model.get("revision"),
        "tokenizer_revision": model.get("tokenizer_revision"),
        "pooling": model.get("pooling"),
        "normalize_embeddings": model.get("normalize_embeddings"),
        "lora_target_modules": adapter.get("target_modules"),
        "lora_rank": adapter.get("rank"),
        "lora_alpha": adapter.get("alpha"),
        "lora_dropout": adapter.get("dropout"),
        "regression_loss": model.get("regression_loss"),
        "loss_reduction": model.get("loss_reduction"),
    }
    if is_nv:
        spec_value["nv_snapshot_dir"] = args.nv_snapshot_dir
        spec_value["nv_review_path"] = args.nv_review_path
    return EncoderModelSpec.from_mapping(spec_value)


def _validate_training(value: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "seed",
        "epochs",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "max_sequence_length",
        "head_fraction",
        "num_workers",
        "early_stopping_min_delta",
        "early_stopping_patience",
    )
    missing = [key for key in required if key not in value]
    _need(not missing, f"missing training settings: {', '.join(missing)}")
    try:
        config = {
            "seed": int(value["seed"]),
            "epochs": int(value["epochs"]),
            "per_device_batch_size": int(value["per_device_batch_size"]),
            "gradient_accumulation_steps": int(value["gradient_accumulation_steps"]),
            "learning_rate": float(value["learning_rate"]),
            "weight_decay": float(value["weight_decay"]),
            "warmup_ratio": float(value["warmup_ratio"]),
            "max_length": int(value["max_sequence_length"]),
            "head_fraction": float(value["head_fraction"]),
            "num_workers": int(value["num_workers"]),
            "early_stopping_min_delta": float(value["early_stopping_min_delta"]),
            "early_stopping_patience": int(value["early_stopping_patience"]),
        }
    except (TypeError, ValueError) as error:
        raise TrainingContractError("training values have invalid types") from error
    _need(config["seed"] >= 0, "seed must be non-negative")
    _need(config["epochs"] > 0, "epochs must be positive")
    _need(config["per_device_batch_size"] > 0, "per_device_batch_size must be positive")
    _need(config["gradient_accumulation_steps"] > 0, "gradient_accumulation_steps must be positive")
    _need(config["learning_rate"] > 0 and config["weight_decay"] >= 0, "invalid optimizer settings")
    _need(0 <= config["warmup_ratio"] < 1, "warmup_ratio must be in [0, 1)")
    _need(config["max_length"] > 0, "max_length must be positive")
    _need(config["head_fraction"] == 0.75, "approved encoder head_fraction is exactly 0.75")
    _need(config["num_workers"] >= 0, "num_workers must be non-negative")
    _need(config["early_stopping_min_delta"] >= 0, "early_stopping_min_delta must be non-negative")
    _need(config["early_stopping_patience"] > 0, "early_stopping_patience must be positive")
    return config


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _format_encoder_input(record: Any) -> str:
    """Fixed score-blind input; IDs and labels are intentionally excluded."""

    from mal2026.formatting import format_encoder_input

    return format_encoder_input(record)


def _load_records(path: str, expected_sha256: str | None = None) -> list[Any]:
    from mal2026.data_contract import load_and_validate_jsonl

    return load_and_validate_jsonl(path, expected_sha256=expected_sha256)


def _extract_scores(record: Any) -> list[float]:
    source = getattr(record, "scores", None)
    if source is None:
        source = getattr(record, "score", None)
    if source is None:
        raise TrainingContractError("validated record lacks scores")
    values: list[float] = []
    for name in SCORE_FIELDS:
        value = getattr(source, name, None) if not isinstance(source, Mapping) else source.get(name)
        if value is None:
            raise TrainingContractError(f"validated record lacks {name} score")
        values.append(float(value))
    return values


def _build_dataset(records: list[Any], tokenizer: Any, max_length: int) -> Any:
    import torch
    from torch.utils.data import Dataset

    class EncoderDataset(Dataset[Any]):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> Mapping[str, Any]:
            record = records[index]
            encoded = tokenizer(_format_encoder_input(record), truncation=False, padding=False, return_attention_mask=True)
            input_ids = encoded["input_ids"]
            if len(input_ids) > max_length:
                head_length = (max_length * 3) // 4
                tail_length = max_length - head_length
                input_ids = input_ids[:head_length] + input_ids[-tail_length:]
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": torch.tensor(_extract_scores(record), dtype=torch.float32),
            }

    return EncoderDataset()


def _build_collator(tokenizer: Any) -> Any:
    import torch

    def collate(rows: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        features = [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in rows]
        padded = tokenizer.pad(features, padding=True, return_tensors="pt")
        padded["labels"] = torch.stack([row["labels"] for row in rows])
        return padded

    return collate


def _build_tokenizer(spec: EncoderModelSpec) -> Any:
    from transformers import AutoTokenizer

    if spec.backbone == "nv_embed_v2":
        review = load_nv_remote_code_review(spec.nv_review_path or "")
        _need(review.revision == spec.revision, "NV tokenizer review revision does not match model revision")
        verify_nv_snapshot(spec.nv_snapshot_dir or "", review)
        tokenizer = AutoTokenizer.from_pretrained(
            spec.nv_snapshot_dir,
            revision=spec.tokenizer_revision,
            trust_remote_code=True,
            local_files_only=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.tokenizer_revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        _need(tokenizer.eos_token_id is not None, "tokenizer requires an existing EOS token to define padding")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _metric_payload(true_values: Any, predicted_values: Any) -> dict[str, float]:
    """Use the shared aggregate-only metric contract; never pass source records."""

    from mal2026.metrics import compute_regression_metrics

    _need(len(true_values) == len(predicted_values), "development predictions and labels have different lengths")
    targets = [dict(zip(SCORE_FIELDS, map(float, row), strict=True)) for row in true_values]
    predictions = [dict(zip(SCORE_FIELDS, map(float, row), strict=True)) for row in predicted_values]
    result = compute_regression_metrics(targets, predictions)
    _need(isinstance(result, Mapping), "metric function must return an aggregate mapping")
    per_target = result.get("per_target")
    _need(isinstance(per_target, Mapping), "metric result does not contain per_target aggregates")
    flattened: dict[str, float] = {}
    for target, metric_values in per_target.items():
        _need(isinstance(metric_values, Mapping), "metric target result must be an aggregate mapping")
        for metric_name, value in metric_values.items():
            if value is not None:
                flattened[f"{target}_{metric_name}"] = float(value)
    _need("average_mae" in flattened, "metric result does not contain primary average MAE")
    return flattened


def _save_run_metadata(output_dir: Path, spec: EncoderModelSpec, training: Mapping[str, Any]) -> None:
    """Persist non-sensitive frozen config only; provenance manifest is added by launcher."""

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "encoder_frozen_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"model": asdict(spec), "training": dict(training)}, handle, ensure_ascii=False, indent=2, sort_keys=True)


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        from accelerate import Accelerator
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import get_cosine_schedule_with_warmup
    except ImportError as error:  # pragma: no cover - runtime dependency gate
        raise RuntimeError("encoder runner requires accelerate, torch, and transformers") from error

    raw_config, config_hash = _read_config(Path(args.config))
    spec = _encoder_spec_from_config(raw_config, args)
    optimization = raw_config["optimization"]
    data = raw_config["data"]
    _need(isinstance(optimization, Mapping) and isinstance(data, Mapping), "validated config lost optimization/data objects")
    training_source = dict(optimization)
    training_source["max_sequence_length"] = data.get("max_sequence_length")
    training_source["head_fraction"] = data.get("head_fraction")
    training = _validate_training(training_source)
    _need(training["max_length"] == 2048, "approved encoder max_length is exactly 2048")
    output_dir = Path(args.output_dir)
    _need(not output_dir.exists(), f"refusing to overwrite existing output directory: {output_dir}")
    accelerator = Accelerator(
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        mixed_precision="bf16",
    )
    _set_seed(training["seed"] + accelerator.process_index)
    if accelerator.is_main_process:
        _save_run_metadata(output_dir, spec, training)
    accelerator.wait_for_everyone()
    from mal2026.provenance import build_run_manifest, wandb_log_aggregates, wandb_rank_zero_init, write_local_run_manifest

    telemetry_config = {
            "run_kind": "encoder_regression",
            "backbone": spec.backbone,
            "model_id": spec.model_id,
            "revision": spec.revision,
            "tokenizer_revision": spec.tokenizer_revision,
            "pooling": spec.pooling,
            "normalize_embeddings": spec.normalize_embeddings,
            "lora_target_modules": ",".join(spec.lora_target_modules),
            "training": training,
        }
    wandb_run = wandb_rank_zero_init(
        project=args.wandb_project,
        run_id=args.run_id,
        config=telemetry_config,
        rank=accelerator.process_index,
    )

    train_records = _load_records(args.train_jsonl, args.train_sha256)
    dev_records = _load_records(args.dev_jsonl, args.dev_sha256)
    _need(bool(train_records) and bool(dev_records), "train and dev partitions must both be non-empty")
    if accelerator.is_main_process:
        manifest = build_run_manifest(
            run_id=args.run_id,
            config_hash=config_hash,
            data_contract={
                "train_sha256": args.train_sha256 or "unverified",
                "development_sha256": args.dev_sha256 or "unverified",
                "train_records": len(train_records),
                "development_records": len(dev_records),
            },
            command=" ".join(sys.argv),
            output_path=str(output_dir),
            extra={"backbone": spec.backbone, "model_revision": spec.revision, "tokenizer_revision": spec.tokenizer_revision},
        )
        write_local_run_manifest(output_dir / "run_manifest.json", manifest)
    tokenizer = _build_tokenizer(spec)
    from mal2026.encoder_modeling import build_encoder_regressor

    model = build_encoder_regressor(spec)
    train_loader = DataLoader(
        _build_dataset(train_records, tokenizer, training["max_length"]),
        batch_size=training["per_device_batch_size"],
        shuffle=True,
        num_workers=training["num_workers"],
        collate_fn=_build_collator(tokenizer),
        pin_memory=True,
    )
    dev_loader = DataLoader(
        _build_dataset(dev_records, tokenizer, training["max_length"]),
        batch_size=training["per_device_batch_size"],
        shuffle=False,
        num_workers=training["num_workers"],
        collate_fn=_build_collator(tokenizer),
        pin_memory=True,
    )
    optimizer = AdamW(model.parameters(), lr=training["learning_rate"], betas=(0.9, 0.95), weight_decay=training["weight_decay"])
    updates_per_epoch = (len(train_loader) + training["gradient_accumulation_steps"] - 1) // training["gradient_accumulation_steps"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(updates_per_epoch * training["epochs"] * training["warmup_ratio"]),
        num_training_steps=updates_per_epoch * training["epochs"],
    )
    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, dev_loader, scheduler)

    best_dev_mae = float("inf")
    stale_epochs = 0
    global_step = 0
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                result = model(**batch)
                loss = result["loss"]
                _need(bool(torch.isfinite(loss).item()), "non-finite encoder loss")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
        model.eval()
        gathered_true, gathered_predicted = [], []
        with torch.no_grad():
            for batch in dev_loader:
                result = model(**batch)
                gathered_true.append(accelerator.gather_for_metrics(batch["labels"]).float().cpu())
                gathered_predicted.append(accelerator.gather_for_metrics(result["predictions"]).float().cpu())
        if accelerator.is_main_process:
            true_values = torch.cat(gathered_true).numpy()
            predicted_values = torch.cat(gathered_predicted).numpy()
            metrics = _metric_payload(true_values, predicted_values)
            dev_mae = metrics["average_mae"]
            improved = (best_dev_mae - dev_mae) > training["early_stopping_min_delta"]
            if improved:
                best_dev_mae, stale_epochs = dev_mae, 0
                accelerator.save_state(output_dir / "best_checkpoint")
            else:
                stale_epochs += 1
            wandb_log_aggregates(wandb_run, {f"dev/{key}": value for key, value in metrics.items()} | {"train/epoch": epoch}, step=global_step)
        else:
            stale_epochs = 0
        stop_flag = torch.tensor([int(stale_epochs >= training["early_stopping_patience"])], device=accelerator.device)
        if accelerator.num_processes > 1:
            torch.distributed.broadcast(stop_flag, src=0)
        if bool(stop_flag.item()):
            break
    if wandb_run is not None:
        wandb_run.finish()
    accelerator.end_training()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="frozen encoder JSON config")
    parser.add_argument("--train-jsonl", required=True, help="restricted optimization-train JSONL path")
    parser.add_argument("--dev-jsonl", required=True, help="restricted development JSONL path")
    parser.add_argument("--train-sha256", default=None, help="expected train partition SHA-256")
    parser.add_argument("--dev-sha256", default=None, help="expected development partition SHA-256")
    parser.add_argument("--output-dir", required=True, help="new ignored outputs/runs/<run-id> directory")
    parser.add_argument("--run-id", required=True, help="unique non-secret run identifier")
    parser.add_argument("--nv-snapshot-dir", default=None, help="reviewed local NV model snapshot; mandatory for NV only")
    parser.add_argument("--nv-review-path", default=None, help="approved NV remote-code review JSON; mandatory for NV only")
    parser.add_argument("--wandb-project", default="mal2026-korean-writing-scoring")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except (EncoderContractError, TrainingContractError) as error:
        raise SystemExit(f"encoder contract error: {error}") from error
