"""Accelerate/DDP Qwen SFT for direct-score and human-feedback→score modes.

The runner reads only a prepared, ignored AI-Hub table plus a versioned
aggregate manifest.  Human feedback is an assistant target, never a user input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decoder import (
    ContractError,
    HUMAN_FEEDBACK_KEYS,
    SCORE_KEYS,
    assert_finite_loss,
    orderly_distributed_shutdown,
    parse_decoder_output,
    project_root,
    prompt_text,
    require_immutable_revision,
    require_tokenizer_chat_template,
    resolve_run_output_dir,
    sanitized_deterministic_generation_config,
    target_for_record,
    validate_lora_targets,
)

SYSTEM_MESSAGE = (
    "당신은 한국어 글쓰기 평가 모델입니다. 사용자 글만 근거로 채점하고, "
    "요청된 JSON 이외의 텍스트를 출력하지 마세요."
)
PREPARED_SCHEMA_VERSION = 1
PREPARED_DATASET_ID = "aihub_human_feedback_v1"


@dataclass(frozen=True)
class DecoderTrainConfig:
    run_id: str
    phase: str  # selection | refit
    mode: str  # direct | human_feedback
    model_id: str
    model_revision: str
    tokenizer_revision: str
    prepared_data_dir: str
    data_manifest_path: str
    data_manifest_sha256: str
    output_dir: str
    canonical_config_path: str
    selection_run_id: str | None = None
    selection_config_hash: str | None = None
    selected_updates: int | None = None
    seed: int = 20260717
    max_seq_length: int = 2048
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 12
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    wandb_project: str = "mal2026-korean-writing-scoring"
    wandb_entity: str | None = None
    max_new_tokens: int = 256

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DecoderTrainConfig":
        extra = sorted(set(raw) - {field.name for field in fields(cls)})
        if extra:
            raise ContractError(f"unknown decoder config fields: {extra}")
        return cls(**dict(raw))

    def validate(self) -> None:
        if self.phase not in {"selection", "refit"}:
            raise ContractError("phase must be selection or refit")
        if self.mode not in {"direct", "human_feedback"}:
            raise ContractError("mode must be direct or human_feedback")
        require_immutable_revision(self.model_revision, "model_revision")
        require_immutable_revision(self.tokenizer_revision, "tokenizer_revision")
        if not re.fullmatch(r"[0-9a-f]{64}", self.data_manifest_sha256):
            raise ContractError("data_manifest_sha256 must be a SHA-256")
        resolve_run_output_dir(self.run_id, self.output_dir)
        _prepared_data_dir(self.prepared_data_dir)
        _manifest_path(self.data_manifest_path)
        if self.phase == "refit":
            if not isinstance(self.selected_updates, int) or self.selected_updates < 1:
                raise ContractError("refit requires selected_updates >= 1 from its selection run")
            if not isinstance(self.selection_run_id, str) or not self.selection_run_id:
                raise ContractError("refit requires the immutable source selection_run_id")
            if not isinstance(self.selection_config_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", self.selection_config_hash):
                raise ContractError("refit requires selection_config_hash from verified selection metadata")
        elif self.selection_run_id is not None or self.selection_config_hash is not None:
            raise ContractError("selection runs must not claim a prior selection binding")
        expected_sequence, expected_new = (2048, 256) if self.mode == "direct" else (4096, 1536)
        if (self.max_seq_length, self.max_new_tokens) != (expected_sequence, expected_new):
            raise ContractError(f"{self.mode} requires max_seq_length={expected_sequence}, max_new_tokens={expected_new}")
        if (self.lora_rank, self.lora_alpha, self.lora_dropout) != (32, 64, 0.05):
            raise ContractError("frozen decoder LoRA config is rank=32, alpha=64, dropout=0.05")
        if self.per_device_batch_size != 1 or self.gradient_accumulation_steps != 8:
            raise ContractError("frozen decoder batch policy is one example/device with accumulation 8")
        if not 1 <= self.epochs <= 12:
            raise ContractError("frozen decoder cap is at most 12 epochs")
        if self.learning_rate != 2e-4 or self.weight_decay != 0.01 or self.warmup_ratio != 0.05:
            raise ContractError("frozen decoder optimizer is LR=2e-4, wd=0.01, warmup=0.05")


def _prepared_data_dir(value: str) -> Path:
    root = project_root()
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else root / candidate
    expected_root = (root / "data" / "processed").resolve(strict=False)
    if candidate.resolve(strict=False).parent != expected_root:
        raise ContractError("prepared_data_dir must be a direct child of ignored data/processed")
    if candidate.exists() and candidate.is_symlink():
        raise ContractError("prepared_data_dir may not be a symlink")
    return candidate


def _manifest_path(value: str) -> Path:
    root = project_root()
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else root / candidate
    expected_root = (root / "data" / "manifests").resolve(strict=False)
    if candidate.resolve(strict=False).parent != expected_root or candidate.suffix != ".json":
        raise ContractError("data_manifest_path must be a JSON direct child of data/manifests")
    if not candidate.is_file() or candidate.is_symlink():
        raise ContractError("data_manifest_path must be a readable non-symlink aggregate manifest")
    return candidate


def load_json_config(path: str) -> DecoderTrainConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("decoder config must be a JSON object") from exc
    if not isinstance(raw, dict):
        raise ContractError("decoder config must be a JSON object")
    config = DecoderTrainConfig.from_mapping(raw)
    config.validate()
    _validate_canonical_contract(config)
    return config


def _validate_canonical_contract(config: DecoderTrainConfig) -> tuple[dict[str, Any], str]:
    from .config import ConfigError, load_experiment_config

    try:
        contract, contract_hash = load_experiment_config(config.canonical_config_path)
    except ConfigError as exc:
        raise ContractError(f"invalid canonical decoder config: {exc}") from exc
    expected_kind = "decoder-direct" if config.mode == "direct" else "decoder-human-feedback-score"
    if contract["run_kind"] != expected_kind:
        raise ContractError("canonical run_kind does not match decoder mode")
    model, adapter, data, optimization = contract["model"], contract["adapter"], contract["data"], contract["optimization"]
    if (model["id"], model["revision"], model["tokenizer_revision"]) != (config.model_id, config.model_revision, config.tokenizer_revision):
        raise ContractError("runtime model fields do not match canonical config")
    if (adapter["rank"], adapter["alpha"], float(adapter["dropout"])) != (config.lora_rank, config.lora_alpha, config.lora_dropout):
        raise ContractError("runtime LoRA fields do not match canonical config")
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if adapter["target_modules"] != targets:
        raise ContractError("canonical decoder LoRA target modules do not match the frozen runtime")
    if (data["max_sequence_length"], data["max_new_tokens"], data["head_fraction"], data["dev_fraction"]) != (config.max_seq_length, config.max_new_tokens, 0.75, 0.20):
        raise ContractError("runtime decoder token/split policy does not match canonical config")
    if data.get("prepared_schema_version") != PREPARED_SCHEMA_VERSION:
        raise ContractError("canonical config does not bind prepared data schema version")
    if config.mode == "human_feedback" and data.get("feedback_target_max_tokens") != 1536:
        raise ContractError("human-feedback config must reserve exactly 1536 target tokens")
    if (optimization["seed"], optimization["learning_rate"], optimization["epochs"], optimization["per_device_batch_size"], optimization["gradient_accumulation_steps"], optimization["warmup_ratio"]) != (config.seed, config.learning_rate, config.epochs, config.per_device_batch_size, config.gradient_accumulation_steps, config.warmup_ratio):
        raise ContractError("runtime optimization fields do not match canonical config")
    return contract, contract_hash


def _runtime_config_hash(config: DecoderTrainConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(text: str, context: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"{context} contains duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        result = json.loads(text, object_pairs_hook=pairs)
    except (json.JSONDecodeError, ContractError) as exc:
        raise ContractError(f"invalid {context} JSON") from exc
    if not isinstance(result, dict):
        raise ContractError(f"{context} must be an object")
    return result


def _validate_source_row(row: Mapping[str, Any], line_number: int) -> dict[str, Any]:
    if tuple(row) != ("id", "prompt", "essay", "score", "feedback"):
        raise ContractError(f"prepared row {line_number} must have exactly ordered id,prompt,essay,score,feedback keys")
    for key in ("id", "prompt", "essay"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ContractError(f"prepared row {line_number} has blank {key}")
    score = row["score"]
    if not isinstance(score, Mapping) or tuple(score) != SCORE_KEYS:
        raise ContractError(f"prepared row {line_number} score keys/order are invalid")
    normalized_score: dict[str, float] = {}
    for key in SCORE_KEYS:
        value = row["score"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 1 <= float(value) <= 5:
            raise ContractError(f"prepared row {line_number} has invalid score.{key}")
        # The preparation contract quantizes emitted labels to two decimals.
        if abs(float(value) * 100 - round(float(value) * 100)) > 1e-8:
            raise ContractError(f"prepared row {line_number} score.{key} must have at most two decimals")
        normalized_score[key] = float(value)
    feedback = row["feedback"]
    if not isinstance(feedback, Mapping) or tuple(feedback) != HUMAN_FEEDBACK_KEYS:
        raise ContractError(f"prepared row {line_number} feedback keys/order are invalid")
    normalized_feedback: dict[str, str] = {}
    for key in HUMAN_FEEDBACK_KEYS:
        value = feedback[key]
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"prepared row {line_number} has blank feedback.{key}")
        normalized_feedback[key] = value
    return {"id": row["id"], "prompt": row["prompt"], "essay": row["essay"], "score": normalized_score, "feedback": normalized_feedback}


def _load_restricted_rows(path: Path, expected_sha256: str, expected_count: int) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ContractError("prepared data file is missing, symlinked, or hash-mismatched")
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ContractError("prepared JSONL may not contain blank lines")
            row = _validate_source_row(_strict_json_object(line, f"prepared row {line_number}"), line_number)
            if row["id"] in ids:
                raise ContractError("prepared split contains duplicate IDs")
            ids.add(row["id"])
            rows.append(row)
    if len(rows) != expected_count:
        raise ContractError("prepared data file record count does not match aggregate manifest")
    return rows


def _load_prepared_manifest(config: DecoderTrainConfig) -> tuple[dict[str, Any], Path]:
    path = _manifest_path(config.data_manifest_path)
    if _sha256(path) != config.data_manifest_sha256:
        raise ContractError("data_manifest_sha256 does not match aggregate manifest")
    raw = _strict_json_object(path.read_text(encoding="utf-8"), "prepared manifest")
    if raw.get("schema_version") != PREPARED_SCHEMA_VERSION or raw.get("dataset_id") != PREPARED_DATASET_ID:
        raise ContractError("prepared manifest schema/version is not approved")
    if not isinstance(raw.get("files"), Mapping) or set(raw["files"]) != {"selection_train", "selection_dev", "refit_train"}:
        raise ContractError("prepared manifest must declare exactly selection_train, selection_dev, refit_train")
    return raw, _prepared_data_dir(config.prepared_data_dir)


def _manifest_file(manifest: Mapping[str, Any], data_dir: Path, key: str) -> tuple[Path, str, int]:
    entry = manifest["files"][key]
    if not isinstance(entry, Mapping) or set(entry) != {"filename", "sha256", "record_count"}:
        raise ContractError(f"prepared manifest files.{key} has an invalid schema")
    filename, digest, count = entry["filename"], entry["sha256"], entry["record_count"]
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".jsonl"):
        raise ContractError(f"prepared manifest files.{key}.filename is unsafe")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or not isinstance(count, int) or count < 1:
        raise ContractError(f"prepared manifest files.{key} digest/count is invalid")
    return data_dir / filename, digest, count


def _records_as_mappings(rows: Any) -> list[dict[str, Any]]:
    """Compatibility conversion for frozen score-only final-evaluation rows."""
    return [{"id": row.id, "prompt": row.prompt, "essay": row.essay, "score": row.scores.as_dict()} for row in rows]


def _load_train_partitions(config: DecoderTrainConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load the preparation-owned source split; never derive it at runtime."""
    manifest, data_dir = _load_prepared_manifest(config)
    selected_keys = ("refit_train",) if config.phase == "refit" else ("selection_train", "selection_dev")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for key in selected_keys:
        path, digest, count = _manifest_file(manifest, data_dir, key)
        loaded[key] = _load_restricted_rows(path, digest, count)
    if config.phase == "refit":
        return loaded["refit_train"], [], manifest
    train_rows, dev_rows = loaded["selection_train"], loaded["selection_dev"]
    if {row["id"] for row in train_rows} & {row["id"] for row in dev_rows}:
        raise ContractError("prepared selection train/dev IDs overlap")
    # A prep artifact makes the source-dev split auditable; the runner merely
    # verifies and consumes it so all four architectures share the same rows.
    return train_rows, dev_rows, manifest


def _verify_refit_selection_binding(config: DecoderTrainConfig) -> None:
    if config.phase != "refit":
        return
    assert config.selection_run_id is not None and config.selection_config_hash is not None
    selection_dir = resolve_run_output_dir(config.selection_run_id, Path(config.output_dir).parent / config.selection_run_id, must_exist=True)
    try:
        metadata = _strict_json_object((selection_dir / "selection_metadata.json").read_text(encoding="utf-8"), "selection metadata")
    except OSError as exc:
        raise ContractError("refit source selection metadata is unavailable") from exc
    expected = {"status": "completed", "run_id": config.selection_run_id, "config_hash": config.selection_config_hash, "selected_updates": config.selected_updates, "mode": config.mode, "model_revision": config.model_revision, "tokenizer_revision": config.tokenizer_revision, "data_manifest_sha256": config.data_manifest_sha256}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ContractError("refit binding does not match verified selection metadata")


def _mean_sha256(mean: Mapping[str, float]) -> str:
    if set(mean) != set(SCORE_KEYS):
        raise ContractError("fallback mean must contain exactly the four score keys")
    return hashlib.sha256(json.dumps({key: float(mean[key]) for key in SCORE_KEYS}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _verified_selection_fallback_mean(config: DecoderTrainConfig) -> dict[str, float]:
    if config.phase != "refit" or config.selection_run_id is None:
        raise ContractError("only a refit may load a prior selection fallback mean")
    selection_dir = resolve_run_output_dir(config.selection_run_id, Path(config.output_dir).parent / config.selection_run_id, must_exist=True)
    try:
        metadata = _strict_json_object((selection_dir / "selection_metadata.json").read_text(encoding="utf-8"), "selection metadata")
        raw_mean = _strict_json_object((selection_dir / "fallback_mean.json").read_text(encoding="utf-8"), "fallback mean")
        mean = {key: float(raw_mean[key]) for key in SCORE_KEYS}
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ContractError("selection fallback mean/provenance is unavailable") from exc
    if _mean_sha256(mean) != metadata.get("fallback_mean_sha256"):
        raise ContractError("selection fallback mean checksum does not match selection metadata")
    return mean


def score_mean(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not records:
        raise ContractError("cannot compute a fallback mean from no records")
    return {key: sum(float(record["score"][key]) for record in records) / len(records) for key in SCORE_KEYS}


def updates_for_prepared_loader(loader_length: int, gradient_accumulation_steps: int) -> int:
    if not isinstance(loader_length, int) or loader_length < 1 or not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps < 1:
        raise ContractError("prepared loader length and gradient accumulation must be positive")
    return (loader_length + gradient_accumulation_steps - 1) // gradient_accumulation_steps


def accelerator_batch_assignment(batch_count: int, world_size: int) -> tuple[tuple[int, ...], ...]:
    if not isinstance(batch_count, int) or batch_count < 1 or not isinstance(world_size, int) or world_size < 1:
        raise ContractError("batch_count and world_size must be positive")
    return tuple(tuple(range(rank, batch_count, world_size)) for rank in range(world_size))


def head_tail_truncate(prefix_ids: list[int], target_ids: list[int], max_seq_length: int, eos_id: int) -> tuple[list[int], list[int], int]:
    """Fixed 75:25 input head:tail truncation; assistant targets never truncate."""
    if len(target_ids) + 1 >= max_seq_length:
        raise ContractError("assistant target exceeds rendered-chat budget; do not truncate it")
    available = max_seq_length - len(target_ids) - 1
    kept = prefix_ids if len(prefix_ids) <= available else prefix_ids[: (available * 3) // 4] + prefix_ids[len(prefix_ids) - (available - (available * 3) // 4) :]
    return kept + target_ids + [eos_id], [-100] * len(kept) + target_ids + [eos_id], len(prefix_ids) - len(kept)


def _input_token_cap(config: DecoderTrainConfig) -> int:
    # Reserve the *entire* bounded generation allocation before input truncation.
    cap = config.max_seq_length - config.max_new_tokens - 1
    if cap < 1:
        raise ContractError("decoder budget leaves no input tokens")
    return cap


def build_sft_example(tokenizer: Any, record: Mapping[str, Any], mode: str, max_seq_length: int, max_new_tokens: int | None = None) -> dict[str, Any]:
    user = prompt_text(record)
    prefix = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    target = target_for_record(record, mode)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if mode == "human_feedback" and len(target_ids) > 1536:
        raise ContractError("human-feedback target exceeds the common 1536-token eligibility gate")
    if max_new_tokens is not None and len(target_ids) > max_new_tokens:
        raise ContractError("assistant target exceeds the reserved generation budget")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ContractError("tokenizer must define eos_token_id")
    input_ids, labels, truncated = head_tail_truncate(prefix_ids, target_ids, max_seq_length, eos_id)
    return {"input_ids": input_ids, "labels": labels, "scores": [float(record["score"][key]) for key in SCORE_KEYS], "truncated_prompt_tokens": truncated}


class _SelectionGenerationDataset:
    """Development-only generation records; feedback stays target-only/local."""
    def __init__(self, tokenizer: Any, records: Sequence[Mapping[str, Any]], input_token_cap: int):
        self.items: list[dict[str, Any]] = []
        for record in records:
            user = prompt_text(record)
            prefix = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            if len(ids) > input_token_cap:
                head = (input_token_cap * 3) // 4
                ids = ids[:head] + ids[len(ids) - (input_token_cap - head) :]
            self.items.append({"input_ids": ids, "scores": [float(record["score"][key]) for key in SCORE_KEYS]})
    def __len__(self) -> int: return len(self.items)
    def __getitem__(self, index: int) -> dict[str, Any]: return self.items[index]


class _SFTDataset:
    def __init__(self, tokenizer: Any, records: Sequence[Mapping[str, Any]], config: DecoderTrainConfig):
        self.items = [build_sft_example(tokenizer, row, config.mode, config.max_seq_length, config.max_new_tokens) for row in records]
    def __len__(self) -> int: return len(self.items)
    def __getitem__(self, index: int) -> dict[str, Any]: return self.items[index]


def _train_collator(tokenizer: Any):
    import torch
    if tokenizer.pad_token_id is None: raise ContractError("tokenizer must have pad token set before collation")
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        n = max(len(row["input_ids"]) for row in batch); pad = tokenizer.pad_token_id
        return {"input_ids": torch.tensor([row["input_ids"] + [pad] * (n-len(row["input_ids"])) for row in batch], dtype=torch.long), "attention_mask": torch.tensor([[1]*len(row["input_ids"])+[0]*(n-len(row["input_ids"])) for row in batch], dtype=torch.long), "labels": torch.tensor([row["labels"] + [-100]*(n-len(row["labels"])) for row in batch], dtype=torch.long)}
    return collate


def _selection_generation_collator(tokenizer: Any):
    import torch
    if tokenizer.pad_token_id is None: raise ContractError("tokenizer must have pad token set before collation")
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        n=max(len(row["input_ids"]) for row in batch); pad=tokenizer.pad_token_id
        return {"input_ids": torch.tensor([[pad]*(n-len(row["input_ids"]))+row["input_ids"] for row in batch], dtype=torch.long), "attention_mask": torch.tensor([[0]*(n-len(row["input_ids"]))+[1]*len(row["input_ids"]) for row in batch], dtype=torch.long), "scores": torch.tensor([row["scores"] for row in batch], dtype=torch.float32)}
    return collate


def train(config: DecoderTrainConfig) -> None:
    config.validate(); _, canonical_config_hash = _validate_canonical_contract(config); _verify_refit_selection_binding(config)
    run_dir=resolve_run_output_dir(config.run_id, config.output_dir)
    if run_dir.exists(): raise ContractError(f"refusing to overwrite run output: {config.output_dir}")
    from accelerate import Accelerator
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
    import torch
    accelerator=Accelerator(mixed_precision="bf16", gradient_accumulation_steps=config.gradient_accumulation_steps)
    random.seed(config.seed+accelerator.process_index); torch.manual_seed(config.seed+accelerator.process_index)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed+accelerator.process_index)
    tokenizer=AutoTokenizer.from_pretrained(config.model_id, revision=config.tokenizer_revision, use_fast=True)
    contract,_=_validate_canonical_contract(config); require_tokenizer_chat_template(tokenizer, contract["model"]["chat_template_sha256"])
    tokenizer.padding_side="right"; tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, torch_dtype=torch.bfloat16); model.config.use_cache=False; model.gradient_checkpointing_enable()
    targets=validate_lora_targets((name for name,_ in model.named_modules()), ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"))
    model=get_peft_model(model, LoraConfig(r=config.lora_rank,lora_alpha=config.lora_alpha,lora_dropout=config.lora_dropout,target_modules=list(targets),task_type=TaskType.CAUSAL_LM,bias="none"))
    train_records,dev_records,prepared_manifest=_load_train_partitions(config)
    fallback_mean=_verified_selection_fallback_mean(config) if config.phase=="refit" else score_mean(train_records)
    train_loader=DataLoader(_SFTDataset(tokenizer,train_records,config),batch_size=1,shuffle=True,collate_fn=_train_collator(tokenizer),drop_last=False)
    dev_loader=DataLoader(_SelectionGenerationDataset(tokenizer,dev_records,_input_token_cap(config)),batch_size=1,shuffle=False,collate_fn=_selection_generation_collator(tokenizer),drop_last=False) if dev_records else None
    optimizer=AdamW(model.parameters(),lr=config.learning_rate,betas=(0.9,0.95),weight_decay=config.weight_decay)
    if dev_loader is None: model,optimizer,train_loader=accelerator.prepare(model,optimizer,train_loader)
    else: model,optimizer,train_loader,dev_loader=accelerator.prepare(model,optimizer,train_loader,dev_loader)
    updates_per_epoch=updates_for_prepared_loader(len(train_loader),config.gradient_accumulation_steps); total_updates=int(config.selected_updates) if config.phase=="refit" else updates_per_epoch*config.epochs
    if config.phase=="refit" and total_updates>updates_per_epoch*config.epochs: raise ContractError("selected_updates exceeds frozen refit epoch budget")
    scheduler=get_scheduler("cosine",optimizer=optimizer,num_warmup_steps=int(total_updates*config.warmup_ratio),num_training_steps=total_updates)
    if accelerator.is_main_process: run_dir.mkdir(parents=True,exist_ok=False); _write_json(run_dir/"config.json",asdict(config)); _write_json(run_dir/"fallback_mean.json",fallback_mean)
    accelerator.wait_for_everyone(); wandb_run=_wandb_init(accelerator,config); updates=0; best_mae=float("inf"); best_updates=None; best_valid=0.0; no_improvement=0
    try:
        for epoch in range(config.epochs):
            _set_loader_epoch(train_loader,epoch); model.train(); loss_sum=0.; loss_count=0
            for batch in train_loader:
                with accelerator.accumulate(model):
                    output=model(**batch); assert_finite_loss(output.loss); accelerator.backward(output.loss)
                    if accelerator.sync_gradients: accelerator.clip_grad_norm_(model.parameters(),1.0)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                loss_sum += float(output.loss.detach().float().item()); loss_count += 1
                if accelerator.sync_gradients:
                    updates += 1
                    if config.phase=="refit" and updates>=int(config.selected_updates): break
            metrics: dict[str,float|int]={"train/epoch":epoch+1,"train/loss":loss_sum/max(loss_count,1),"train/updates":updates}
            if dev_loader is not None:
                dev_mae,valid_rate=_evaluate_selection_dev(accelerator,model,tokenizer,dev_loader,config,fallback_mean)
                metrics.update({"dev/macro_mae":dev_mae,"dev/parse_valid_rate":valid_rate})
                if valid_rate>=0.99 and dev_mae < best_mae: best_mae,best_updates,best_valid=dev_mae,updates,valid_rate; no_improvement=0
                else: no_improvement += 1
            _log_aggregate(wandb_run,accelerator,metrics); _save_checkpoint(accelerator,model,run_dir,epoch+1,updates)
            if config.phase=="selection" and no_improvement>=3: break
            if config.phase=="refit" and updates>=int(config.selected_updates): break
        if config.phase=="refit" and updates != config.selected_updates: raise RuntimeError("refit ended before selected optimizer-update count")
        if config.phase=="selection" and best_updates is None: raise RuntimeError("no selection checkpoint met the >=0.99 strict parse-validity gate")
        if accelerator.is_main_process:
            selected=best_updates if config.phase=="selection" else config.selected_updates
            _write_json(run_dir/"training_complete.json", {"status":"completed","updates":updates,"selected_updates":selected,"best_dev_macro_mae":best_mae if best_updates else None,"best_dev_parse_valid_rate":best_valid if best_updates else None,"ended_at":datetime.now(UTC).isoformat()})
            if config.phase=="selection": _write_json(run_dir/"selection_metadata.json", {"status":"completed","run_id":config.run_id,"config_hash":_runtime_config_hash(config),"selected_updates":best_updates,"mode":config.mode,"model_revision":config.model_revision,"tokenizer_revision":config.tokenizer_revision,"data_manifest_sha256":config.data_manifest_sha256,"fallback_mean_sha256":_mean_sha256(fallback_mean),"selection_metric":"four_score_macro_mae","parse_validity_gate":0.99})
            _write_run_manifest(run_dir,config,canonical_config_hash,prepared_manifest,{"updates":updates,"selected_updates":selected or 0,"best_dev_macro_mae":best_mae if best_updates else -1.0,"best_dev_parse_valid_rate":best_valid if best_updates else -1.0},accelerator)
    finally:
        if wandb_run is not None: wandb_run.finish()
        orderly_distributed_shutdown()


def _evaluate_selection_dev(accelerator: Any,model: Any,tokenizer: Any,loader: Any,config: DecoderTrainConfig,fallback: Mapping[str,float]) -> tuple[float,float]:
    import torch
    model.eval(); errors=[0.,0.,0.,0.]; count=0; valid_count=0
    with torch.inference_mode():
        for batch in loader:
            unwrapped=accelerator.unwrap_model(model); generation_config=sanitized_deterministic_generation_config(unwrapped.generation_config)
            generated=unwrapped.generate(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],generation_config=generation_config,max_new_tokens=config.max_new_tokens,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
            texts=tokenizer.batch_decode(generated[:,batch["input_ids"].shape[1]:],skip_special_tokens=True)
            local=[]
            for text,truth in zip(texts,batch["scores"].tolist()):
                parsed=parse_decoder_output(text,config.mode,fallback); local.append(([abs(parsed.scores[key]-float(truth[i])) for i,key in enumerate(SCORE_KEYS)],int(parsed.valid)))
            error_tensor=torch.tensor([item[0] for item in local],device=accelerator.device,dtype=torch.float64); valid_tensor=torch.tensor([item[1] for item in local],device=accelerator.device,dtype=torch.float64)
            gathered_errors=accelerator.gather_for_metrics(error_tensor); gathered_valid=accelerator.gather_for_metrics(valid_tensor)
            if accelerator.is_main_process: errors=[a+b for a,b in zip(errors,gathered_errors.sum(dim=0).tolist())]; count+=int(gathered_errors.shape[0]); valid_count+=int(gathered_valid.sum().item())
    model.train(); values=accelerator.reduce(torch.tensor(errors+[count,valid_count],device=accelerator.device,dtype=torch.float64),reduction="sum")
    total=max(float(values[4].item()),1.0); return sum(float(values[i].item())/total for i in range(4))/4.0,float(values[5].item())/total


def _save_checkpoint(accelerator: Any,model: Any,run_dir: Path,epoch:int,updates:int)->None:
    checkpoint=run_dir/f"checkpoint-epoch-{epoch:02d}"; accelerator.wait_for_everyone(); accelerator.save_state(str(checkpoint/"accelerate"))
    if accelerator.is_main_process:
        adapter=checkpoint/"adapter"; accelerator.unwrap_model(model).save_pretrained(adapter,safe_serialization=True); _write_json(checkpoint/"metadata.json",{"epoch":epoch,"optimizer_updates":updates,"adapter_sha256":_directory_sha256(adapter)})
    accelerator.wait_for_everyone()


def _wandb_init(accelerator: Any,config: DecoderTrainConfig)->Any|None:
    from .provenance import wandb_rank_zero_init
    return wandb_rank_zero_init(project=config.wandb_project,run_id=config.run_id,rank=accelerator.process_index,config={"run_id":config.run_id,"phase":config.phase,"mode":config.mode,"model_id":config.model_id,"model_revision":config.model_revision,"seed":config.seed,"world_size":accelerator.num_processes})

def _log_aggregate(run:Any|None,accelerator:Any,values:Mapping[str,float|int])->None:
    if accelerator.is_main_process:
        from .provenance import wandb_log_aggregates
        wandb_log_aggregates(run,values,step=int(values.get("train/updates",0)))

def _set_loader_epoch(loader:Any,epoch:int)->None:
    for item in (loader,getattr(loader,"sampler",None),getattr(getattr(loader,"batch_sampler",None),"sampler",None)):
        if hasattr(item,"set_epoch"): item.set_epoch(epoch)

def _directory_sha256(path:Path)->str:
    if not path.is_dir(): raise ContractError("adapter directory is missing")
    digest=hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_symlink(): raise ContractError("adapter directory may not contain symlinks")
        if item.is_file():
            relative=item.relative_to(path).as_posix().encode(); digest.update(len(relative).to_bytes(8,"big")); digest.update(relative)
            with item.open("rb") as handle:
                while chunk:=handle.read(1024*1024): digest.update(chunk)
    return digest.hexdigest()

def _write_run_manifest(run_dir:Path,config:DecoderTrainConfig,canonical_config_hash:str,prepared_manifest:Mapping[str,Any],metrics:Mapping[str,float|int],accelerator:Any)->None:
    from .provenance import aggregate_only_payload,build_run_manifest
    gpu_name="unavailable"
    try:
        import torch
        if torch.cuda.is_available(): gpu_name=torch.cuda.get_device_name(accelerator.local_process_index)
    except Exception: pass
    split=prepared_manifest.get("split",{})
    manifest=build_run_manifest(run_id=config.run_id,config_hash=_runtime_config_hash(config),data_contract={"prepared_dataset":PREPARED_DATASET_ID,"prepared_manifest_sha256":config.data_manifest_sha256,"partition_algorithm":str(split.get("algorithm","preparation_owned")),"optimization_train_records":int(prepared_manifest["files"]["selection_train"]["record_count"] if config.phase=="selection" else prepared_manifest["files"]["refit_train"]["record_count"]),"development_records":int(prepared_manifest["files"]["selection_dev"]["record_count"] if config.phase=="selection" else 0)},command=" ".join(sys.argv),output_path=str(run_dir),extra={"canonical_config_hash":canonical_config_hash,"phase":config.phase,"mode":config.mode,"model_revision":config.model_revision,"tokenizer_revision":config.tokenizer_revision,"seed":config.seed,"world_size":accelerator.num_processes,"gpu_name":gpu_name,"metrics":dict(metrics),"deviations":"none"})
    (run_dir/"run_manifest.json").write_text(json.dumps(aggregate_only_payload(manifest),ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def _write_json(path:Path,value:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def main(argv:list[str]|None=None)->None:
    parser=argparse.ArgumentParser(description="Qwen2.5 direct-score or human-feedback→score SFT")
    parser.add_argument("--config",required=True,help="non-secret decoder JSON configuration")
    train(load_json_config(parser.parse_args(argv).config))

if __name__ == "__main__": main()
