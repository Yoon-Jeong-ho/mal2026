#!/usr/bin/env python3
"""DDP-safe lifecycle runner for the two four-target encoder regressors.

This runner has three intentionally separate phases:
``selection`` uses the precomputed human-feedback source train/development
partitions named by an aggregate-only ignored manifest; ``refit`` uses the
manifest's full eligible source-training partition for the selected count; and
``final-eval`` loads only that refit adapter and evaluates the fixed,
separately hash-bound ``eval/validation.jsonl``. No phase writes
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mal2026.decoder import (  # noqa: E402
    CANONICAL_VALIDATION_SHA256,
    ContractError,
    require_canonical_dataset,
    require_path_under_run,
    resolve_run_output_dir,
)
from mal2026.encoder_modeling import EncoderContractError, EncoderModelSpec, SCORE_FIELDS  # noqa: E402


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
    _need(config["dev_fraction"] == 0.20, "human-feedback source development fraction is frozen at 0.20")
    _need(config["num_workers"] >= 0 and config["early_stopping_min_delta"] >= 0 and config["early_stopping_patience"] > 0, "invalid convergence policy")
    return config


def _resolve_run_dir(output_dir: str, run_id: str) -> Path:
    """Create only a new, unsymlinked canonical ``outputs/runs/<run-id>`` child."""
    try:
        run_dir = resolve_run_output_dir(run_id, output_dir, must_exist=False)
    except ContractError as error:
        raise TrainingContractError(str(error)) from error
    _need(not run_dir.exists(), "refusing to overwrite an existing run output")
    return run_dir


def _require_final_validation(path: str, digest: str) -> tuple[str, str]:
    """Admit only the frozen canonical final-validation input."""
    _need(Path(path).is_absolute(), "canonical eval/validation.jsonl path must be absolute")
    try:
        resolved = require_canonical_dataset(path, "validation", digest)
    except ContractError as error:
        raise TrainingContractError(str(error)) from error
    _need(resolved.is_absolute(), "canonical validation dataset path must be absolute")
    return str(resolved), CANONICAL_VALIDATION_SHA256


def _require_prior_run_dir(path: str) -> Path:
    """Require an absolute canonical completed run directory, never a symlink."""
    candidate = Path(path)
    _need(candidate.is_absolute(), "prior run directory must be an absolute outputs/runs/<run-id> path")
    try:
        expected = resolve_run_output_dir(candidate.name, candidate, must_exist=True)
    except ContractError as error:
        raise TrainingContractError(str(error)) from error
    _need(candidate == expected, "prior run directory must be lexically canonical")
    return expected


def _require_prior_run_artifact(path: str | Path, relative_path: str) -> tuple[Path, str]:
    """Read metadata/artifacts only at a canonical relative path under one run."""
    candidate = Path(path)
    relative = Path(relative_path)
    _need(candidate.is_absolute() and not relative.is_absolute() and ".." not in relative.parts, "prior run artifact path must be canonical")
    _need(candidate.name == relative.name, f"prior run artifact must be named {relative.name}")
    _need(len(candidate.parents) >= len(relative.parts), "prior run artifact has invalid depth")
    run_dir = candidate
    for _ in relative.parts:
        run_dir = run_dir.parent
    run_id = run_dir.name
    expected = PROJECT_ROOT / "outputs" / "runs" / run_id / relative
    _need(candidate == expected, "prior run artifact must use canonical outputs/runs layout")
    try:
        resolved = require_path_under_run(candidate, run_id)
    except ContractError as error:
        raise TrainingContractError(str(error)) from error
    _need(resolved == expected.resolve(strict=True), "prior artifact path escaped canonical run root")
    return resolved, run_id


def _prepared_update_count(prepared_batches: int, accumulation_steps: int) -> int:
    """Update count after (not before) Accelerate has sharded the DataLoader."""
    _need(prepared_batches > 0 and accumulation_steps > 0, "prepared batches and accumulation must be positive")
    return math.ceil(prepared_batches / accumulation_steps)


def _rank_batch_coverage(record_count: int, batch_size: int, world_size: int) -> tuple[tuple[int, ...], ...]:
    """Dependency-free model of Accelerate's one-time batch sharding.

    A plain shuffled DataLoader is passed to ``accelerator.prepare``.  This
    helper documents the non-overlapping batch assignment and protects against
    reintroducing a manual DistributedSampler (which would double-shard).
    """
    _need(record_count > 0 and batch_size > 0 and world_size > 0, "invalid sharding dimensions")
    batches = [tuple(range(start, min(start + batch_size, record_count))) for start in range(0, record_count, batch_size)]
    return tuple(tuple(index for batch_index, batch in enumerate(batches) if batch_index % world_size == rank for index in batch) for rank in range(world_size))


def _set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _format_encoder_input(record: Any) -> str:
    from mal2026.formatting import format_encoder_input
    return format_encoder_input(record)


@dataclass(frozen=True)
class PreparedEncoderRecord:
    """In-memory restricted source record; feedback is intentionally excluded."""

    prompt: str
    essay: str
    scores: Any


@dataclass(frozen=True)
class PreparedPartition:
    path: Path
    sha256: str
    record_count: int


@dataclass(frozen=True)
class PreparedHumanFeedbackManifest:
    manifest_path: Path
    manifest_sha256: str
    source_fingerprint: str
    selection_train: PreparedPartition
    selection_dev: PreparedPartition
    refit_train: PreparedPartition


_FORBIDDEN_MANIFEST_KEYS = frozenset({"id", "document_id", "prompt", "essay", "response", "feedback", "records"})
_PREPARED_DATASET_ID = "aihub_human_feedback_v1"
_PREPARED_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "aihub_human_feedback_v1.json"
_PREPARED_ROOT = PROJECT_ROOT / "data" / "processed" / _PREPARED_DATASET_ID


def _require_prepared_root(path: Path) -> Path:
    """Resolve a restricted prepared partition beneath its fixed ignored root."""
    try:
        root = _PREPARED_ROOT.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise TrainingContractError("prepared rows must remain under fixed ignored data/processed root") from error
    return resolved


def _assert_aggregate_manifest(value: Any, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _need(isinstance(key, str), f"{path} uses non-string key")
            _need(key.casefold() not in _FORBIDDEN_MANIFEST_KEYS, f"{path} contains restricted raw-data field {key}")
            _assert_aggregate_manifest(nested, f"{path}.{key}")
    elif isinstance(value, list):
        # Aggregate manifest lists may contain fixed labels/checksums, never
        # nested row objects that could carry writing text or identifiers.
        _need(all(item is None or isinstance(item, (str, int, float, bool)) for item in value), f"{path} list must contain aggregate scalars only")
    elif value is None or isinstance(value, (str, int, float, bool)):
        return
    else:
        raise TrainingContractError(f"{path} contains unsupported value")


def _parse_partition(value: Any, expected_name: str) -> PreparedPartition:
    _need(isinstance(value, Mapping) and set(value) == {"filename", "sha256", "record_count"}, f"prepared partition {expected_name} has invalid fields")
    filename, digest, count = value["filename"], value["sha256"], value["record_count"]
    _need(filename == expected_name, f"prepared partition filename must be {expected_name}")
    _need(isinstance(digest, str) and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest), "prepared partition sha256 is invalid")
    _need(isinstance(count, int) and count > 0, "prepared partition record_count must be positive")
    path = _require_prepared_root(_PREPARED_ROOT / expected_name)
    _need(_sha256_file(path) == digest, f"prepared partition checksum mismatch: {expected_name}")
    return PreparedPartition(path=path, sha256=digest, record_count=count)


def _load_prepared_manifest(path: str) -> PreparedHumanFeedbackManifest:
    location = Path(path)
    _need(location.is_absolute(), "prepared manifest path must be absolute")
    try:
        resolved = location.resolve(strict=True)
        expected = _PREPARED_MANIFEST.resolve(strict=True)
    except OSError as error:
        raise TrainingContractError("prepared human-feedback manifest does not exist") from error
    _need(resolved == expected, "prepared manifest must be canonical data/manifests/aihub_human_feedback_v1.json")
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingContractError("unable to read prepared human-feedback manifest") from error
    _need(isinstance(raw, Mapping), "prepared manifest must be an object")
    _assert_aggregate_manifest(raw)
    required = {"schema_version", "dataset_id", "source", "eligibility", "score_contract", "feedback_contract", "split", "files"}
    _need(set(raw) == required, "prepared manifest has unknown or missing top-level fields")
    _need(raw["schema_version"] == 1 and raw["dataset_id"] == _PREPARED_DATASET_ID, "unsupported prepared human-feedback manifest dataset")
    source = raw["source"]
    _need(isinstance(source, Mapping) and isinstance(source.get("archive_list_sha256"), str) and len(source["archive_list_sha256"]) == 64 and all(char in "0123456789abcdef" for char in source["archive_list_sha256"]), "prepared manifest source archive fingerprint is invalid")
    _need(isinstance(raw["eligibility"], Mapping) and isinstance(raw["score_contract"], Mapping) and isinstance(raw["feedback_contract"], Mapping) and isinstance(raw["split"], Mapping), "prepared manifest requires aggregate source contracts")
    files = raw["files"]
    expected_files = {"selection_train", "selection_dev", "refit_train"}
    _need(isinstance(files, Mapping) and set(files) == expected_files, "prepared manifest must name exactly the three frozen partitions")
    return PreparedHumanFeedbackManifest(
        manifest_path=resolved,
        manifest_sha256=_sha256_file(resolved),
        source_fingerprint=source["archive_list_sha256"],
        selection_train=_parse_partition(files["selection_train"], "selection_train.jsonl"),
        selection_dev=_parse_partition(files["selection_dev"], "selection_dev.jsonl"),
        refit_train=_parse_partition(files["refit_train"], "refit_train.jsonl"),
    )


def _load_prepared_records(partition: PreparedPartition) -> list[PreparedEncoderRecord]:
    from mal2026.data_contract import ScoreVector

    rows: list[PreparedEncoderRecord] = []
    seen: set[str] = set()
    with partition.path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise TrainingContractError(f"prepared partition has invalid JSON at local line {number}") from error
            _need(isinstance(raw, Mapping), "prepared records must be objects")
            # Feedback is retained in restricted prepared rows for decoder SFT,
            # but never read or passed to this encoder input path.
            required = {"id", "prompt", "essay", "score", "feedback"}
            _need(required <= set(raw), "prepared record lacks required human-feedback contract fields")
            feedback = raw["feedback"]
            feedback_fields = {"holistic", "content_1", "content_2", "content_3", "organization_1", "organization_2", "expression_1", "expression_2", "task_1"}
            _need(isinstance(feedback, Mapping) and set(feedback) == feedback_fields and all(isinstance(item, str) and item.strip() for item in feedback.values()), "prepared human feedback must be complete and nonblank")
            identifier, prompt, essay, score = raw["id"], raw["prompt"], raw["essay"], raw["score"]
            _need(isinstance(identifier, str) and identifier and identifier not in seen, "prepared record id is blank or duplicate")
            _need(isinstance(prompt, str) and prompt.strip() and isinstance(essay, str) and essay.strip(), "prepared record prompt/essay must be nonblank")
            _need(isinstance(score, Mapping) and set(score) == set(SCORE_FIELDS), "prepared record scores must have four fields")
            values: dict[str, float] = {}
            for name in SCORE_FIELDS:
                value = score[name]
                _need(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 1.0 <= float(value) <= 5.0, "prepared score must be finite in [1,5]")
                values[name] = float(value)
            seen.add(identifier)
            rows.append(PreparedEncoderRecord(prompt=prompt, essay=essay, scores=ScoreVector(**values)))
    _need(len(rows) == partition.record_count, "prepared partition record_count mismatch")
    return rows


def _load_final_records(path: str, expected_sha256: str) -> list[Any]:
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
    # Set offline guards and verify the local review before importing the HF
    # loader that will execute a custom tokenizer class.
    if spec.backbone == "nv_embed_v2":
        spec.validate_nv_runtime()
    from transformers import AutoTokenizer
    if spec.backbone == "nv_embed_v2":
        tokenizer = AutoTokenizer.from_pretrained(str(Path(spec.nv_snapshot_dir or "").resolve(strict=True)), revision=spec.tokenizer_revision, trust_remote_code=True, local_files_only=True)
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
    _need(all(f"{field}_mae" in flattened for field in SCORE_FIELDS), "aggregate metric contract lacks per-score MAE")
    flattened["macro_mae"] = sum(flattened[f"{field}_mae"] for field in SCORE_FIELDS) / len(SCORE_FIELDS)
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


def _selection_metadata(path: str, *, config_hash: str, prepared_manifest_sha256: str, source_fingerprint: str) -> tuple[Mapping[str, Any], str]:
    location, run_id = _require_prior_run_artifact(path, "selection_metadata.json")
    payload = _read_json(location)
    _need(payload.get("phase") == "selection", "selection metadata is not from selection phase")
    _need(payload.get("run_id") == run_id, "selection metadata run_id does not bind its containing run")
    _need(payload.get("config_hash") == config_hash and payload.get("prepared_manifest_sha256") == prepared_manifest_sha256 and payload.get("source_fingerprint") == source_fingerprint, "selection metadata config/data binding mismatch")
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


def _make_loader(dataset: Any, *, batch_size: int, workers: int, collate: Any, shuffle: bool) -> Any:
    """Build an unsharded loader; Accelerate owns the sole DDP sharding step."""
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate, pin_memory=True, drop_last=False)


def _run_training(*, accelerator: Any, model: Any, optimizer: Any, train_loader: Any, dev_loader: Any | None, training: Mapping[str, Any], output_dir: Path, phase: str, wandb_run: Any) -> tuple[int, int | None, dict[str, float] | None]:
    """Train selection/refit; scheduler count is computed only after prepare."""
    import torch
    from transformers import get_cosine_schedule_with_warmup

    updates_per_epoch = _prepared_update_count(len(train_loader), training["gradient_accumulation_steps"])
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
        # DataLoaderShard exposes set_epoch; do not reach into a manual sampler.
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
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
            # Human-feedback source selection is frozen to four-score macro MAE,
            # not the independently-defined final-validation average field.
            improved = (best_mae - metrics["macro_mae"]) > training["early_stopping_min_delta"]
            if improved:
                best_mae, best_updates, best_metrics, stale = metrics["macro_mae"], updates, metrics, 0
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
    prepared: PreparedHumanFeedbackManifest | None = None
    canonical_eval_path: str | None = None
    canonical_eval_sha: str | None = None
    if args.phase == "final-eval":
        _need(args.eval_jsonl and args.eval_sha256 and args.refit_dir, "final-eval requires frozen validation JSONL/hash and refit directory")
        canonical_eval_path, canonical_eval_sha = _require_final_validation(args.eval_jsonl, args.eval_sha256)
    else:
        _need(args.prepared_manifest, "selection/refit requires the canonical aggregate human-feedback manifest")
        prepared = _load_prepared_manifest(args.prepared_manifest)
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
        assert prepared is not None
        if args.phase == "selection":
            train_records = _load_prepared_records(prepared.selection_train)
            dev_records = _load_prepared_records(prepared.selection_dev)
            lifecycle["prepared_source"] = {
                "manifest_sha256": prepared.manifest_sha256,
                "source_fingerprint": prepared.source_fingerprint,
                "selection_train_sha256": prepared.selection_train.sha256,
                "selection_train_records": prepared.selection_train.record_count,
                "selection_dev_sha256": prepared.selection_dev.sha256,
                "selection_dev_records": prepared.selection_dev.record_count,
            }
        else:
            metadata, metadata_sha = _selection_metadata(args.selection_metadata, config_hash=config_hash, prepared_manifest_sha256=prepared.manifest_sha256, source_fingerprint=prepared.source_fingerprint)
            train_records, dev_records = _load_prepared_records(prepared.refit_train), None
            training["selected_updates"] = metadata["selected_updates"]
            lifecycle["selection_link"] = {"selection_metadata_sha256": metadata_sha, "selection_run_id": metadata.get("run_id"), "selected_updates": metadata["selected_updates"]}
            lifecycle["prepared_source"] = {
                "manifest_sha256": prepared.manifest_sha256,
                "source_fingerprint": prepared.source_fingerprint,
                "refit_train_sha256": prepared.refit_train.sha256,
                "refit_train_records": prepared.refit_train.record_count,
            }
        tokenizer = _build_tokenizer(spec)
        from mal2026.encoder_modeling import build_encoder_regressor
        model = build_encoder_regressor(spec)
        train_loader = _make_loader(_build_dataset(train_records, tokenizer, training["max_length"]), batch_size=training["per_device_batch_size"], workers=training["num_workers"], collate=_build_collator(tokenizer), shuffle=True)
        dev_loader = _make_loader(_build_dataset(dev_records, tokenizer, training["max_length"]), batch_size=training["per_device_batch_size"], workers=training["num_workers"], collate=_build_collator(tokenizer), shuffle=False) if dev_records else None
        optimizer = AdamW(model.parameters(), lr=training["learning_rate"], betas=(0.9, 0.95), weight_decay=training["weight_decay"])
        if dev_loader is None:
            model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        else:
            model, optimizer, train_loader, dev_loader = accelerator.prepare(model, optimizer, train_loader, dev_loader)
        updates, selected_updates, best_metrics = _run_training(accelerator=accelerator, model=model, optimizer=optimizer, train_loader=train_loader, dev_loader=dev_loader, training=training, output_dir=run_dir, phase=args.phase, wandb_run=wandb_run)
        if args.phase == "refit":
            adapter_files = _save_trainable_adapter(accelerator, model, run_dir / "refit_adapter")
            lifecycle["refit"] = {"all_source_train_records": len(train_records), "updates": updates, "adapter_files": adapter_files}
        elif accelerator.is_main_process:
            metadata = {"schema_version": 1, "phase": "selection", "run_id": args.run_id, "config_hash": config_hash, "prepared_manifest_sha256": prepared.manifest_sha256, "source_fingerprint": prepared.source_fingerprint, "selected_updates": selected_updates, "observed_updates": updates, "prepared_source": lifecycle["prepared_source"], "best_metrics": best_metrics}
            _write_json(run_dir / "selection_metadata.json", metadata)
            lifecycle["selection"] = {"selected_updates": selected_updates, "selection_metadata_sha256": _sha256_file(run_dir / "selection_metadata.json")}
        data_contract = {"prepared_manifest_sha256": prepared.manifest_sha256, "source_fingerprint": prepared.source_fingerprint, "source_train_records": len(train_records), "source_development_records": len(dev_records) if dev_records else 0}
    else:
        refit_dir = _require_prior_run_dir(args.refit_dir)
        refit_manifest_path, refit_run_id = _require_prior_run_artifact(str(refit_dir / "run_manifest.json"), "run_manifest.json")
        refit_manifest = _read_json(refit_manifest_path)
        _need(refit_manifest.get("run_id") == refit_run_id, "refit manifest run_id does not bind its containing run")
        _need(refit_manifest.get("config_hash") == config_hash and refit_manifest.get("phase") == "refit", "refit manifest config/phase mismatch")
        refit_lifecycle = refit_manifest.get("lifecycle")
        _need(isinstance(refit_lifecycle, Mapping) and isinstance(refit_lifecycle.get("refit"), Mapping), "refit manifest lacks refit binding")
        _need(isinstance(refit_lifecycle.get("selection_link"), Mapping), "refit manifest lacks selection linkage")
        selection_link = refit_lifecycle["selection_link"]
        _need(isinstance(selection_link.get("selection_run_id"), str) and selection_link["selection_run_id"], "refit selection link lacks selection run_id")
        _need(isinstance(selection_link.get("selection_metadata_sha256"), str) and len(selection_link["selection_metadata_sha256"]) == 64, "refit selection link lacks metadata checksum")
        _need(isinstance(selection_link.get("selected_updates"), int) and selection_link["selected_updates"] > 0, "refit selection link lacks selected updates")
        refit_data = refit_manifest.get("data_contract")
        _need(isinstance(refit_data, Mapping) and isinstance(refit_data.get("prepared_manifest_sha256"), str) and len(refit_data["prepared_manifest_sha256"]) == 64, "refit manifest is not bound to a prepared human-feedback manifest")
        _need(isinstance(refit_data.get("source_fingerprint"), str) and len(refit_data["source_fingerprint"]) == 64, "refit manifest lacks source fingerprint")
        refit_model = refit_lifecycle.get("model")
        _need(refit_model == lifecycle["model"], "refit manifest model/tokenizer binding mismatch")
        refit_info = refit_lifecycle["refit"]
        _need(isinstance(refit_info.get("all_source_train_records"), int) and refit_info["all_source_train_records"] > 0, "refit manifest is not a full eligible source-data refit")
        _need(isinstance(refit_info.get("adapter_files"), Mapping), "refit manifest lacks adapter checksums")
        assert canonical_eval_path is not None and canonical_eval_sha is not None
        eval_records = _load_final_records(canonical_eval_path, canonical_eval_sha)
        _need(len(eval_records) == 400, "final evaluation must use all 400 canonical validation records")
        tokenizer = _build_tokenizer(spec)
        from mal2026.encoder_modeling import build_encoder_regressor
        model = build_encoder_regressor(spec)
        eval_loader = _make_loader(_build_dataset(eval_records, tokenizer, training["max_length"]), batch_size=training["per_device_batch_size"], workers=training["num_workers"], collate=_build_collator(tokenizer), shuffle=False)
        model, eval_loader = accelerator.prepare(model, eval_loader)
        adapter_state, adapter_run_id = _require_prior_run_artifact(str(refit_dir / "refit_adapter" / "trainable_state.pt"), "refit_adapter/trainable_state.pt")
        _need(adapter_run_id == refit_run_id, "refit adapter is not under the manifest run root")
        _load_trainable_adapter(accelerator, model, adapter_state.parent, refit_info["adapter_files"])
        metrics = _evaluate(accelerator, model, eval_loader)
        if accelerator.is_main_process:
            assert metrics is not None
            _write_json(run_dir / "final_metrics.json", {"schema_version": 1, "phase": "final-eval", "record_count": len(eval_records), "metrics": metrics})
        lifecycle["final_evaluation"] = {"validation_sha256": canonical_eval_sha, "validation_records": len(eval_records), "refit_run_id": refit_run_id, "refit_manifest_sha256": _sha256_file(refit_manifest_path), "refit_adapter_files": dict(refit_info["adapter_files"]), "selection_link": refit_lifecycle.get("selection_link")}
        data_contract = {"validation_sha256": canonical_eval_sha, "validation_records": len(eval_records), "refit_prepared_manifest_sha256": refit_data["prepared_manifest_sha256"], "refit_source_fingerprint": refit_data["source_fingerprint"]}
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
    parser.add_argument("--prepared-manifest", help="absolute canonical data/manifests/aihub_human_feedback_v1.json")
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
