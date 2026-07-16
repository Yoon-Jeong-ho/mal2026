#!/usr/bin/env python3
"""DDP-safe lifecycle runner for the two four-target encoder regressors.

This runner has three intentionally separate phases:
``selection`` chooses an update count on a deterministic prompt-disjoint
internal development partition derived from ``eval/train.jsonl``;
``refit`` trains on every record in that same hash-bound train file for the
selected count; and ``final-eval`` loads only that refit adapter and evaluates
the fixed, separately hash-bound ``eval/validation.jsonl``.  No phase writes
private text, IDs, predictions, or rationales to W&B or disk outside ignored
``outputs/runs/<run-id>``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
)


class TrainingContractError(ValueError):
    """Raised before any model/data loading for an incomplete lifecycle."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingContractError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_config(path: Path) -> tuple[Mapping[str, Any], str]:
    """Read a local, non-secret config without accepting mutable defaults.

    NV review fields are validated by :class:`EncoderModelSpec`, because that
    single typed record is later persisted verbatim in the local run manifest.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingContractError(f"unable to read encoder config: {path}") from error
    _need(isinstance(raw, Mapping), "encoder config must be a JSON object")
    _need(raw.get("run_kind") in {"encoder-qwen3", "encoder-nvembed"}, "unsupported encoder run_kind")
    for key in ("model", "adapter", "data", "optimization"):
        _need(isinstance(raw.get(key), Mapping), f"encoder config.{key} must be an object")
    return json.loads(json.dumps(raw)), _canonical_json_hash(raw)


def _encoder_spec_from_config(value: Mapping[str, Any], args: argparse.Namespace) -> EncoderModelSpec:
    kind = value["run_kind"]
    model, adapter = value["model"], value["adapter"]
    assert isinstance(model, Mapping) and isinstance(adapter, Mapping)
    spec_value: dict[str, Any] = {
        "backbone": "nv_embed_v2" if kind == "encoder-nvembed" else "qwen3_embedding",
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
    if kind == "encoder-nvembed":
        spec_value["nv_snapshot_dir"] = args.nv_snapshot_dir
        spec_value["nv_remote_code_review"] = value.get("remote_code_review")
    return EncoderModelSpec.from_mapping(spec_value)


def _validate_training(value: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "seed", "epochs", "per_device_batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay",
        "warmup_ratio", "max_sequence_length", "head_fraction", "dev_fraction", "num_workers",
        "early_stopping_min_delta", "early_stopping_patience",
    )
    missing = [key for key in required if key not in value]
    _need(not missing, f"missing training settings: {', '.join(missing)}")
    try:
        config = {
            "seed": int(value["seed"]), "epochs": int(value["epochs"]),
            "per_device_batch_size": int(value["per_device_batch_size"]),
            "gradient_accumulation_steps": int(value["gradient_accumulation_steps"]),
            "learning_rate": float(value["learning_rate"]), "weight_decay": float(value["weight_decay"]),
            "warmup_ratio": float(value["warmup_ratio"]), "max_length": int(value["max_sequence_length"]),
            "head_fraction": float(value["head_fraction"]), "dev_fraction": float(value["dev_fraction"]),
            "num_workers": int(value["num_workers"]), "early_stopping_min_delta": float(value["early_stopping_min_delta"]),
            "early_stopping_patience": int(value["early_stopping_patience"]),
        }
    except (TypeError, ValueError) as error:
        raise TrainingContractError("training values have invalid types") from error
    _need(config["seed"] >= 0 and config["epochs"] > 0, "invalid seed or epoch count")
    _need(config["per_device_batch_size"] > 0 and config["gradient_accumulation_steps"] > 0, "invalid batch policy")
    _need(config["learning_rate"] > 0 and config["weight_decay"] >= 0, "invalid optimizer settings")
    _need(0 <= config["warmup_ratio"] < 1 and 0 < config["dev_fraction"] < 1, "invalid warmup or dev fraction")
    _need(config["max_length"] == 2048 and config["head_fraction"] == 0.75, "frozen encoder input policy is 2048/75:25")
    _need(config["num_workers"] >= 0 and config["early_stopping_min_delta"] >= 0 and config["early_stopping_patience"] > 0, "invalid convergence policy")
    return config


def _resolve_run_dir(output_dir: str, run_id: str) -> Path:
    """Accept only a new child of resolved ``outputs/runs`` with no symlinks."""
    _need(bool(run_id) and Path(run_id).name == run_id, "run_id must be a single nonempty path component")
    project = PROJECT_ROOT.resolve()
    expected = project / "outputs" / "runs" / run_id
    provided = Path(output_dir)
    if not provided.is_absolute():
        provided = project / provided
    # Reject an existing symlink in any component rather than merely resolving through it.
    for ancestor in (project / "outputs", project / "outputs" / "runs", provided):
        _need(not ancestor.is_symlink(), "output path may not use symlinks")
    _need(provided.resolve(strict=False) == expected.resolve(strict=False), "output_dir must resolve to outputs/runs/<run-id>")
    _need(provided.parent.resolve(strict=False) == (project / "outputs" / "runs").resolve(strict=False), "output_dir escaped canonical runs root")
    _need(not provided.exists() and not provided.is_symlink(), "refusing to overwrite an existing run output")
    return expected


def _set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _format_encoder_input(record: Any) -> str:
    from mal2026.formatting import format_encoder_input
    return format_encoder_input(record)


def _load_records(path: str, expected_sha256: str) -> list[Any]:
    from mal2026.data_contract import load_and_validate_jsonl
    return load_and_validate_jsonl(path, expected_sha256=expected_sha256)


def _extract_scores(record: Any) -> list[float]:
    return [float(getattr(record.scores, name)) for name in SCORE_FIELDS]


def _build_dataset(records: list[Any], tokenizer: Any, max_length: int) -> Any:
    import torch
    from torch.utils.data import Dataset

    class EncoderDataset(Dataset[Any]):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> Mapping[str, Any]:
            encoded = tokenizer(_format_encoder_input(records[index]), truncation=False, padding=False, return_attention_mask=True)
            input_ids = encoded["input_ids"]
            if len(input_ids) > max_length:
                head_length = (max_length * 3) // 4
                input_ids = input_ids[:head_length] + input_ids[-(max_length - head_length):]
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": torch.tensor(_extract_scores(records[index]), dtype=torch.float32)}
    return EncoderDataset()


def _build_collator(tokenizer: Any) -> Any:
    import torch
    def collate(rows: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        padded = tokenizer.pad([{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in rows], padding=True, return_tensors="pt")
        padded["labels"] = torch.stack([row["labels"] for row in rows])
        return padded
    return collate


def _build_tokenizer(spec: EncoderModelSpec) -> Any:
    from transformers import AutoTokenizer
    if spec.backbone == "nv_embed_v2":
        # spec validation and snapshot hashing occur before this remote-code call.
        spec.validate_nv_runtime()
        tokenizer = AutoTokenizer.from_pretrained(spec.nv_snapshot_dir, revision=spec.tokenizer_revision, trust_remote_code=True, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.tokenizer_revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        _need(tokenizer.eos_token_id is not None, "tokenizer requires EOS for padding")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _metric_payload(true_values: Any, predicted_values: Any) -> dict[str, float]:
    from mal2026.metrics import compute_regression_metrics
    _need(len(true_values) == len(predicted_values), "predictions and labels differ in length")
    result = compute_regression_metrics(
        [dict(zip(SCORE_FIELDS, map(float, row), strict=True)) for row in true_values],
        [dict(zip(SCORE_FIELDS, map(float, row), strict=True)) for row in predicted_values],
    )
    flattened: dict[str, float] = {}
    for target, target_metrics in result["per_target"].items():
        for name, value in target_metrics.items():
            if value is not None:
                flattened[f"{target}_{name}"] = float(value)
    _need("average_mae" in flattened, "aggregate metric contract lacks average_mae")
    return flattened


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _need(not path.exists(), f"refusing to overwrite {path.name}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingContractError(f"unable to read required lifecycle metadata: {path}") from error
    _need(isinstance(value, Mapping), f"metadata must be a JSON object: {path.name}")
    return value


def _selection_metadata(path: str, *, config_hash: str, train_sha256: str) -> tuple[Mapping[str, Any], str]:
    location = Path(path)
    payload = _read_json(location)
    _need(payload.get("phase") == "selection", "selection metadata is not from selection phase")
    _need(payload.get("config_hash") == config_hash and payload.get("train_sha256") == train_sha256, "selection metadata config/data binding mismatch")
    _need(isinstance(payload.get("selected_updates"), int) and payload["selected_updates"] > 0, "selection metadata lacks selected_updates")
    return payload, _sha256_file(location)


def _save_trainable_adapter(accelerator: Any, model: Any, directory: Path) -> dict[str, str]:
    """Persist only LoRA/head trainables, not an 8B base checkpoint."""
    import torch
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        directory.mkdir(parents=True, exist_ok=False)
        unwrapped = accelerator.unwrap_model(model)
        state = {name: parameter.detach().cpu() for name, parameter in unwrapped.named_parameters() if parameter.requires_grad}
        _need(bool(state), "refit model has no trainable adapter/head parameters")
        torch.save({"schema_version": 1, "state_dict": state}, directory / "trainable_state.pt")
        hashes = {"trainable_state.pt": _sha256_file(directory / "trainable_state.pt")}
        _write_json(directory / "adapter_manifest.json", {"schema_version": 1, "files": hashes, "trainable_parameter_count": len(state)})
    accelerator.wait_for_everyone()
    return {"trainable_state.pt": _sha256_file(directory / "trainable_state.pt")} if accelerator.is_main_process else {}


def _load_trainable_adapter(accelerator: Any, model: Any, directory: Path, expected_files: Mapping[str, Any]) -> None:
    import torch
    _need(set(expected_files) == {"trainable_state.pt"}, "refit adapter manifest has unexpected files")
    state_path = directory / "trainable_state.pt"
    _need(state_path.is_file() and _sha256_file(state_path) == expected_files["trainable_state.pt"], "refit adapter checksum mismatch")
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    _need(isinstance(payload, Mapping) and payload.get("schema_version") == 1 and isinstance(payload.get("state_dict"), Mapping), "invalid refit adapter")
    unwrapped = accelerator.unwrap_model(model)
    expected_trainable = {name for name, parameter in unwrapped.named_parameters() if parameter.requires_grad}
    supplied = set(payload["state_dict"])
    _need(supplied == expected_trainable, "refit adapter parameters do not match current architecture")
    missing, unexpected = unwrapped.load_state_dict(dict(payload["state_dict"]), strict=False)
    _need(not unexpected and not (expected_trainable & set(missing)), "failed to load all refit adapter parameters")


def _evaluate(accelerator: Any, model: Any, loader: Any) -> dict[str, float] | None:
    import torch
    model.eval()
    true_values, predicted_values = [], []
    with torch.no_grad():
        for batch in loader:
            result = model(**batch)
            true_values.append(accelerator.gather_for_metrics(batch["labels"]).float().cpu())
            predicted_values.append(accelerator.gather_for_metrics(result["predictions"]).float().cpu())
    if not accelerator.is_main_process:
        return None
    return _metric_payload(torch.cat(true_values).numpy(), torch.cat(predicted_values).numpy())


def _make_loader(dataset: Any, *, batch_size: int, workers: int, collate: Any, accelerator: Any, shuffle: bool, seed: int) -> Any:
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=shuffle, seed=seed, drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=workers, collate_fn=collate, pin_memory=True, drop_last=False)


def _run_training(*, accelerator: Any, model: Any, optimizer: Any, train_loader: Any, dev_loader: Any | None, training: Mapping[str, Any], output_dir: Path, phase: str, wandb_run: Any) -> tuple[int, int | None, dict[str, float] | None]:
    """Train selection/refit; scheduler count is computed only after prepare."""
    import torch
    from transformers import get_cosine_schedule_with_warmup

    updates_per_epoch = math.ceil(len(train_loader) / training["gradient_accumulation_steps"])
    _need(updates_per_epoch > 0, "no optimizer updates per epoch")
    selected_updates = training.get("selected_updates") if phase == "refit" else None
    max_updates = int(selected_updates) if selected_updates is not None else updates_per_epoch * training["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=round(max_updates * training["warmup_ratio"]), num_training_steps=max_updates)
    scheduler = accelerator.prepare(scheduler)
    updates = 0
    best_updates: int | None = None
    best_metrics: dict[str, float] | None = None
    best_mae = float("inf")
    stale = 0
    epoch = 0
    while updates < max_updates:
        epoch += 1
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
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
                    updates += 1
            if updates >= max_updates:
                break
        if phase == "refit":
            continue
        assert dev_loader is not None
        metrics = _evaluate(accelerator, model, dev_loader)
        if accelerator.is_main_process:
            assert metrics is not None
            improved = (best_mae - metrics["average_mae"]) > training["early_stopping_min_delta"]
            if improved:
                best_mae, best_updates, best_metrics, stale = metrics["average_mae"], updates, metrics, 0
            else:
                stale += 1
            from mal2026.provenance import wandb_log_aggregates
            wandb_log_aggregates(wandb_run, {f"selection/dev/{key}": value for key, value in metrics.items()} | {"selection/epoch": epoch, "selection/updates": updates}, step=updates)
        else:
            improved, stale = False, 0
        improved_flag = torch.tensor([int(improved)], device=accelerator.device)
        stop_flag = torch.tensor([int(stale >= training["early_stopping_patience"])], device=accelerator.device)
        if accelerator.num_processes > 1:
            torch.distributed.broadcast(improved_flag, src=0)
            torch.distributed.broadcast(stop_flag, src=0)
        if bool(improved_flag.item()):
            accelerator.save_state(output_dir / "selection_checkpoint")
        accelerator.wait_for_everyone()
        if bool(stop_flag.item()):
            break
    _need(phase == "refit" or best_updates is not None, "selection produced no checkpoint")
    return updates, best_updates, best_metrics


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        from accelerate import Accelerator
        from torch.optim import AdamW
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("encoder runner requires torch, accelerate, and transformers") from error

    raw_config, config_hash = _read_config(Path(args.config))
    spec = _encoder_spec_from_config(raw_config, args)
    data, optimization = raw_config["data"], raw_config["optimization"]
    assert isinstance(data, Mapping) and isinstance(optimization, Mapping)
    training = _validate_training(dict(optimization) | {"max_sequence_length": data.get("max_sequence_length"), "head_fraction": data.get("head_fraction"), "dev_fraction": data.get("dev_fraction")})
    run_dir = _resolve_run_dir(args.output_dir, args.run_id)
    if args.phase == "final-eval":
        _need(args.eval_jsonl and args.eval_sha256 and args.refit_dir, "final-eval requires eval JSONL/hash and refit directory")
    else:
        _need(args.train_jsonl and args.train_sha256, "selection/refit requires train JSONL/hash")
        if args.phase == "refit":
            _need(args.selection_metadata, "refit requires selection metadata")

    accelerator = Accelerator(gradient_accumulation_steps=training["gradient_accumulation_steps"], mixed_precision="bf16")
    _set_seed(training["seed"] + accelerator.process_index)
    run_dir.mkdir(parents=True, exist_ok=False) if accelerator.is_main_process else None
    accelerator.wait_for_everyone()
    from mal2026.provenance import build_run_manifest, wandb_rank_zero_init, write_local_run_manifest
    telemetry = {"run_kind": "encoder_regression", "phase": args.phase, "backbone": spec.backbone, "model_id": spec.model_id, "model_revision": spec.revision, "tokenizer_revision": spec.tokenizer_revision, "pooling": spec.pooling, "training": training}
    wandb_run = wandb_rank_zero_init(project=args.wandb_project, run_id=args.run_id, config=telemetry, rank=accelerator.process_index)

    lifecycle: dict[str, Any] = {"phase": args.phase, "model": {"model_id": spec.model_id, "revision": spec.revision, "tokenizer_revision": spec.tokenizer_revision}, "config_hash": config_hash}
    if args.phase in {"selection", "refit"}:
        all_train = _load_records(args.train_jsonl, args.train_sha256)
        _need(len(all_train) == 2000, "selection/refit must bind all 2,000 canonical training records")
        from mal2026.data_contract import split_prompt_groups
        if args.phase == "selection":
            split = split_prompt_groups(all_train, training["dev_fraction"])
            train_records, dev_records = list(split.optimization_train), list(split.development)
            lifecycle["internal_development"] = split.manifest
        else:
            metadata, metadata_sha = _selection_metadata(args.selection_metadata, config_hash=config_hash, train_sha256=args.train_sha256)
            train_records, dev_records = all_train, None
            training["selected_updates"] = metadata["selected_updates"]
            lifecycle["selection_link"] = {"selection_metadata_sha256": metadata_sha, "selection_run_id": metadata.get("run_id"), "selected_updates": metadata["selected_updates"]}
        tokenizer = _build_tokenizer(spec)
        from mal2026.encoder_modeling import build_encoder_regressor
        model = build_encoder_regressor(spec)
        train_loader = _make_loader(_build_dataset(train_records, tokenizer, training["max_length"]), batch_size=training["per_device_batch_size"], workers=training["num_workers"], collate=_build_collator(tokenizer), accelerator=accelerator, shuffle=True, seed=training["seed"])
        dev_loader = _make_loader(_build_dataset(dev_records, tokenizer, training["max_length"]), batch_size=training["per_device_batch_size"], workers=training["num_workers"], collate=_build_collator(tokenizer), accelerator=accelerator, shuffle=False, seed=training["seed"]) if dev_records else None
        optimizer = AdamW(model.parameters(), lr=training["learning_rate"], betas=(0.9, 0.95), weight_decay=training["weight_decay"])
        if dev_loader is None:
            model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        else:
            model, optimizer, train_loader, dev_loader = accelerator.prepare(model, optimizer, train_loader, dev_loader)
        updates, selected_updates, best_metrics = _run_training(accelerator=accelerator, model=model, optimizer=optimizer, train_loader=train_loader, dev_loader=dev_loader, training=training, output_dir=run_dir, phase=args.phase, wandb_run=wandb_run)
        if args.phase == "refit":
            adapter_files = _save_trainable_adapter(accelerator, model, run_dir / "refit_adapter")
            lifecycle["refit"] = {"all_train_records": len(all_train), "updates": updates, "adapter_files": adapter_files}
        elif accelerator.is_main_process:
            metadata = {"schema_version": 1, "phase": "selection", "run_id": args.run_id, "config_hash": config_hash, "train_sha256": args.train_sha256, "selected_updates": selected_updates, "observed_updates": updates, "internal_development": lifecycle["internal_development"], "best_metrics": best_metrics}
            _write_json(run_dir / "selection_metadata.json", metadata)
            lifecycle["selection"] = {"selected_updates": selected_updates, "selection_metadata_sha256": _sha256_file(run_dir / "selection_metadata.json")}
        data_contract = {"train_sha256": args.train_sha256, "train_records": len(all_train), "optimization_records": len(train_records), "development_records": len(dev_records) if dev_records else 0}
    else:
        refit_dir = Path(args.refit_dir)
        refit_manifest = _read_json(refit_dir / "run_manifest.json")
        _need(refit_manifest.get("config_hash") == config_hash and refit_manifest.get("phase") == "refit", "refit manifest config/phase mismatch")
        refit_lifecycle = refit_manifest.get("lifecycle")
        _need(isinstance(refit_lifecycle, Mapping) and isinstance(refit_lifecycle.get("refit"), Mapping), "refit manifest lacks refit binding")
        _need(isinstance(refit_lifecycle.get("selection_link"), Mapping), "refit manifest lacks selection linkage")
        refit_model = refit_lifecycle.get("model")
        _need(refit_model == lifecycle["model"], "refit manifest model/tokenizer binding mismatch")
        refit_info = refit_lifecycle["refit"]
        _need(refit_info.get("all_train_records") == 2000, "refit manifest is not an all-2000-record refit")
        _need(isinstance(refit_info.get("adapter_files"), Mapping), "refit manifest lacks adapter checksums")
        eval_records = _load_records(args.eval_jsonl, args.eval_sha256)
        _need(len(eval_records) == 400, "final evaluation must use all 400 canonical validation records")
        tokenizer = _build_tokenizer(spec)
        from mal2026.encoder_modeling import build_encoder_regressor
        model = build_encoder_regressor(spec)
        eval_loader = _make_loader(_build_dataset(eval_records, tokenizer, training["max_length"]), batch_size=training["per_device_batch_size"], workers=training["num_workers"], collate=_build_collator(tokenizer), accelerator=accelerator, shuffle=False, seed=training["seed"])
        model, eval_loader = accelerator.prepare(model, eval_loader)
        _load_trainable_adapter(accelerator, model, refit_dir / "refit_adapter", refit_info["adapter_files"])
        metrics = _evaluate(accelerator, model, eval_loader)
        if accelerator.is_main_process:
            assert metrics is not None
            _write_json(run_dir / "final_metrics.json", {"schema_version": 1, "phase": "final-eval", "record_count": len(eval_records), "metrics": metrics})
        lifecycle["final_evaluation"] = {"validation_sha256": args.eval_sha256, "validation_records": len(eval_records), "refit_manifest_sha256": _sha256_file(refit_dir / "run_manifest.json"), "refit_adapter_files": dict(refit_info["adapter_files"]), "selection_link": refit_lifecycle.get("selection_link")}
        data_contract = {"validation_sha256": args.eval_sha256, "validation_records": len(eval_records)}
    if accelerator.is_main_process:
        manifest = build_run_manifest(run_id=args.run_id, config_hash=config_hash, data_contract=data_contract, command=" ".join(sys.argv), output_path=str(run_dir), extra={"phase": args.phase, "backbone": spec.backbone, "model_spec": asdict(spec), "lifecycle": lifecycle})
        # Promote lifecycle fields to top level so final-eval linkage is easily auditable.
        manifest["phase"] = args.phase
        manifest["lifecycle"] = lifecycle
        write_local_run_manifest(run_dir / "run_manifest.json", manifest)
    accelerator.wait_for_everyone()
    if wandb_run is not None:
        wandb_run.finish()
    accelerator.end_training()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("selection", "refit", "final-eval"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, help="new ignored outputs/runs/<run-id>")
    parser.add_argument("--train-jsonl", help="restricted eval/train.jsonl")
    parser.add_argument("--train-sha256", help="expected SHA-256 for eval/train.jsonl")
    parser.add_argument("--selection-metadata", help="selection output selection_metadata.json (refit only)")
    parser.add_argument("--eval-jsonl", help="restricted eval/validation.jsonl (final-eval only)")
    parser.add_argument("--eval-sha256", help="expected SHA-256 for eval/validation.jsonl")
    parser.add_argument("--refit-dir", help="completed refit output directory (final-eval only)")
    parser.add_argument("--nv-snapshot-dir", default=None, help="reviewed NV local snapshot named by its pinned revision")
    parser.add_argument("--wandb-project", default="mal2026-korean-writing-scoring")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except (EncoderContractError, TrainingContractError) as error:
        raise SystemExit(f"encoder contract error: {error}") from error
