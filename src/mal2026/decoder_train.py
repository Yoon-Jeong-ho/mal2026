"""Accelerate/DDP implementation for the two Qwen decoder SFT regimes.

No model is loaded at import time.  This runner is intentionally selection-only
or refit-only: final fixed-validation prediction belongs to ``decoder_eval``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .decoder import (
    CANONICAL_TRAIN_SHA256,
    ContractError,
    SCORE_KEYS,
    assert_finite_loss,
    prompt_text,
    require_canonical_dataset,
    require_immutable_revision,
    require_path_under_run,
    resolve_run_output_dir,
    target_for_record,
    validate_lora_targets,
    parse_decoder_output,
)

SYSTEM_MESSAGE = (
    "당신은 한국어 글쓰기 평가 모델입니다. 사용자 글만 근거로 채점하고, "
    "요청된 JSON 이외의 텍스트를 출력하지 마세요."
)


@dataclass(frozen=True)
class DecoderTrainConfig:
    run_id: str
    phase: str  # selection | refit
    mode: str  # direct | rationale
    model_id: str
    model_revision: str
    tokenizer_revision: str
    train_path: str
    output_dir: str
    canonical_config_path: str
    train_sha256: str = CANONICAL_TRAIN_SHA256
    rationale_run_id: str | None = None
    selected_updates: int | None = None
    seed: int = 20260716
    max_seq_length: int = 3072
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
    max_new_tokens: int = 512

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DecoderTrainConfig":
        known = {f.name for f in fields(cls)}
        extra = sorted(set(raw) - known)
        if extra:
            raise ContractError(f"unknown decoder config fields: {extra}")
        return cls(**dict(raw))

    def validate(self) -> None:
        if self.phase not in {"selection", "refit"}:
            raise ContractError("phase must be selection or refit")
        if self.mode not in {"direct", "rationale"}:
            raise ContractError("mode must be direct or rationale")
        require_immutable_revision(self.model_revision, "model_revision")
        require_immutable_revision(self.tokenizer_revision, "tokenizer_revision")
        resolve_run_output_dir(self.run_id, self.output_dir)
        if self.phase == "refit":
            if not isinstance(self.selected_updates, int) or self.selected_updates < 1:
                raise ContractError("refit requires selected_updates >= 1 from its selection run")
        require_canonical_dataset(self.train_path, "train", self.train_sha256)
        if self.mode == "rationale" and not self.rationale_run_id:
            raise ContractError("rationale mode requires a score-blind synthetic rationale run")
        if self.mode == "direct" and self.rationale_run_id:
            raise ContractError("direct mode must not consume synthetic rationale artifacts")
        if (self.lora_rank, self.lora_alpha, self.lora_dropout) != (32, 64, 0.05):
            raise ContractError("frozen decoder LoRA config is rank=32, alpha=64, dropout=0.05")
        if self.per_device_batch_size != 1 or self.gradient_accumulation_steps != 8:
            raise ContractError("frozen decoder batch policy is one example/device with accumulation 8")
        if self.max_seq_length != 3072 or self.epochs > 12 or self.epochs < 1:
            raise ContractError("frozen decoder cap is 3072 tokens and at most 12 epochs")
        if self.learning_rate != 2e-4 or self.weight_decay != 0.01 or self.warmup_ratio != 0.05:
            raise ContractError("frozen decoder optimizer is LR=2e-4, wd=0.01, warmup=0.05")


def load_json_config(path: str) -> DecoderTrainConfig:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ContractError("decoder config must be a JSON object")
    config = DecoderTrainConfig.from_mapping(raw)
    config.validate()
    _validate_canonical_contract(config)
    return config


def _validate_canonical_contract(config: DecoderTrainConfig) -> tuple[dict[str, Any], str]:
    """Bind the runtime fields to the versioned, fail-closed public template."""
    from .config import ConfigError, load_experiment_config

    try:
        contract, contract_hash = load_experiment_config(config.canonical_config_path)
    except ConfigError as exc:
        raise ContractError(f"invalid canonical decoder config: {exc}") from exc
    expected_kind = "decoder-direct" if config.mode == "direct" else "decoder-rationale-score"
    if contract["run_kind"] != expected_kind:
        raise ContractError("canonical run_kind does not match decoder mode")
    model, adapter, data, optimization = contract["model"], contract["adapter"], contract["data"], contract["optimization"]
    if (model["id"], model["revision"], model["tokenizer_revision"]) != (config.model_id, config.model_revision, config.tokenizer_revision):
        raise ContractError("runtime model fields do not match canonical config")
    if (adapter["rank"], adapter["alpha"], float(adapter["dropout"])) != (config.lora_rank, config.lora_alpha, config.lora_dropout):
        raise ContractError("runtime LoRA fields do not match canonical config")
    expected_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if adapter["target_modules"] != expected_targets:
        raise ContractError("canonical decoder LoRA target modules do not match the frozen runtime")
    if data["max_sequence_length"] != config.max_seq_length or optimization["seed"] != config.seed:
        raise ContractError("runtime sequence length or seed does not match canonical config")
    if (
        optimization["learning_rate"] != config.learning_rate
        or optimization["epochs"] != config.epochs
        or optimization["per_device_batch_size"] != config.per_device_batch_size
        or optimization["gradient_accumulation_steps"] != config.gradient_accumulation_steps
        or optimization["warmup_ratio"] != config.warmup_ratio
    ):
        raise ContractError("runtime optimization fields do not match canonical config")
    if config.mode == "rationale" and contract["teacher"]["id"] != config.model_id:
        # The current protocol freezes the Qwen teacher family; a separate
        # generator config records its immutable revision and settings.
        raise ContractError("canonical rationale teacher must be the Qwen decoder family")
    return contract, contract_hash


def _records_as_mappings(rows: Any) -> list[dict[str, Any]]:
    return [{"id": row.id, "prompt": row.prompt, "essay": row.essay, "score": row.scores.as_dict()} for row in rows]


def _load_train_partitions(config: DecoderTrainConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load once, hash-validate, and derive the frozen prompt-disjoint dev split."""
    from .data_contract import load_and_validate_jsonl, split_prompt_groups

    all_rows = load_and_validate_jsonl(config.train_path, expected_sha256=config.train_sha256)
    if config.phase == "refit":
        from .data_contract import stable_hash

        records = _records_as_mappings(all_rows)
        return records, [], {
            "schema_version": 1,
            "selection_algorithm": "refit_all_canonical_train_records",
            "total_records": len(records),
            "optimization_train_records": len(records),
            "development_records": 0,
            "optimization_record_id_sha256": stable_hash("\n".join(sorted(record["id"] for record in records))),
        }
    split = split_prompt_groups(all_rows, 0.10)
    return _records_as_mappings(split.optimization_train), _records_as_mappings(split.development), split.manifest


def _read_rationale_map(rationale_run_id: str, train_records: list[dict[str, Any]], train_partition: Mapping[str, Any], config: DecoderTrainConfig) -> dict[str, Any]:
    """Consume only a validated score-blind teacher run for this train partition."""
    from .rationale import RationaleValidationError, validate_rationale_payload

    rationale_dir = require_path_under_run(
        Path(config.output_dir).parent / rationale_run_id / "synthetic-rationales.jsonl", rationale_run_id
    ).parent
    path = rationale_dir / "synthetic-rationales.jsonl"
    provenance_path = rationale_dir / "rationale_provenance.json"
    if not provenance_path.is_file():
        raise ContractError("synthetic rationale artifact is missing aggregate provenance")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("unable to read synthetic rationale provenance") from exc
    expected = {
        "source_train_sha256": config.train_sha256,
        "partition_record_id_sha256": train_partition["optimization_record_id_sha256"],
        "partition_record_count": len(train_records),
        "nonempty_valid_rate_gate": 0.85,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ContractError("synthetic rationale provenance does not bind this canonical train partition")
    if provenance.get("status") != "passed" or not isinstance(provenance.get("teacher_revision"), str):
        raise ContractError("synthetic rationale generation did not pass its score-blind validation gate")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if provenance.get("artifact_sha256") != digest:
        raise ContractError("synthetic rationale artifact checksum does not match its provenance")
    by_id = {record["id"]: record for record in train_records}
    items: dict[str, Any] = {}
    nonempty_valid = 0
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            entry = json.loads(line)
            if not isinstance(entry, dict) or set(entry) != {"id", "rationale"} or not isinstance(entry["id"], str):
                raise ContractError("rationale JSONL rows must be exactly id and rationale (no scores or metadata)")
            identifier = entry["id"]
            if identifier not in by_id or identifier in items:
                raise ContractError("rationale rows must join one-to-one to the current train partition")
            try:
                result = validate_rationale_payload({"rationale": entry["rationale"]}, essay=by_id[identifier]["essay"])
            except RationaleValidationError as exc:
                raise ContractError(f"invalid synthetic rationale at local line {line_number}: {exc}") from exc
            nonempty_valid += int(result.nonempty_valid)
            items[identifier] = entry["rationale"]
    if set(items) != set(by_id):
        raise ContractError("rationale file must have exactly one row per current training record")
    if nonempty_valid / len(train_records) < 0.85:
        raise ContractError("synthetic rationale nonempty-valid rate is below frozen 85% no-go gate")
    return items


def head_tail_truncate(prefix_ids: list[int], target_ids: list[int], max_seq_length: int, eos_id: int) -> tuple[list[int], list[int], int]:
    """Freeze 75% head / 25% tail prompt truncation and assistant-only labels."""
    if len(target_ids) + 1 >= max_seq_length:
        raise ContractError("assistant target exceeds max_seq_length; do not silently truncate it")
    available = max_seq_length - len(target_ids) - 1
    if len(prefix_ids) <= available:
        kept_prefix = prefix_ids
    else:
        head = (available * 3) // 4
        kept_prefix = prefix_ids[:head] + prefix_ids[len(prefix_ids) - (available - head) :]
    ids = kept_prefix + target_ids + [eos_id]
    labels = [-100] * len(kept_prefix) + target_ids + [eos_id]
    return ids, labels, len(prefix_ids) - len(kept_prefix)


def build_sft_example(tokenizer: Any, record: Mapping[str, Any], mode: str, rationale: Any | None, max_seq_length: int) -> dict[str, Any]:
    """Use the model chat template and mask every user/system token from loss."""
    user = prompt_text(record)
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target = target_for_record(record, mode, rationale)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ContractError("tokenizer must define eos_token_id")
    input_ids, labels, truncated = head_tail_truncate(prefix_ids, target_ids, max_seq_length, eos_id)
    if labels[: len(input_ids) - len(target_ids) - 1] != [-100] * (len(input_ids) - len(target_ids) - 1):
        raise AssertionError("assistant-only loss masking invariant failed")
    return {"input_ids": input_ids, "labels": labels, "scores": [float(record["score"][k]) for k in SCORE_KEYS], "truncated_prompt_tokens": truncated}


class _SelectionGenerationDataset:
    """Development-only generation records; text stays process-local."""
    def __init__(self, tokenizer: Any, records: list[dict[str, Any]], max_seq_length: int):
        self.items: list[dict[str, Any]] = []
        for record in records:
            user = prompt_text(record)
            messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}]
            prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            if len(ids) > max_seq_length:
                head = (max_seq_length * 3) // 4
                ids = ids[:head] + ids[len(ids) - (max_seq_length - head) :]
            self.items.append({"input_ids": ids, "scores": [float(record["score"][key]) for key in SCORE_KEYS], "essay": record["essay"]})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class _SFTDataset:
    def __init__(self, tokenizer: Any, records: list[dict[str, Any]], mode: str, rationale_map: Mapping[str, Any] | None, max_seq_length: int):
        self.items = [
            build_sft_example(tokenizer, record, mode, None if rationale_map is None else rationale_map[record["id"]], max_seq_length)
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def _train_collator(tokenizer: Any):
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ContractError("tokenizer must have pad token set before collation")

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in batch)
        return {
            "input_ids": torch.tensor([row["input_ids"] + [pad_id] * (max_len - len(row["input_ids"])) for row in batch], dtype=torch.long),
            "attention_mask": torch.tensor([[1] * len(row["input_ids"]) + [0] * (max_len - len(row["input_ids"])) for row in batch], dtype=torch.long),
            "labels": torch.tensor([row["labels"] + [-100] * (max_len - len(row["labels"])) for row in batch], dtype=torch.long),
        }

    return collate


def _selection_generation_collator(tokenizer: Any):
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ContractError("tokenizer must have pad token set before collation")

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Qwen generation needs left padding to index final non-pad position.
        max_len = max(len(row["input_ids"]) for row in batch)
        ids = [[pad_id] * (max_len - len(row["input_ids"])) + row["input_ids"] for row in batch]
        mask = [[0] * (max_len - len(row["input_ids"])) + [1] * len(row["input_ids"]) for row in batch]
        # Keep the source essay process-local for rationale offset validation;
        # it is never sent to the model, W&B, or any output artifact.
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "scores": torch.tensor([row["scores"] for row in batch], dtype=torch.float32),
            "essays": [row["essay"] for row in batch],
        }

    return collate


def train(config: DecoderTrainConfig) -> None:
    """Run selection/refit SFT.  Calling this function requires declared ML deps."""
    config.validate()
    _, canonical_config_hash = _validate_canonical_contract(config)
    run_dir = resolve_run_output_dir(config.run_id, config.output_dir)
    if run_dir.exists():
        raise ContractError(f"refusing to overwrite run output: {config.output_dir}")
    from accelerate import Accelerator
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
    import torch

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=config.gradient_accumulation_steps)
    random.seed(config.seed + accelerator.process_index)
    torch.manual_seed(config.seed + accelerator.process_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + accelerator.process_index)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.tokenizer_revision, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Explicitly no device_map: Accelerator/DDP owns placement.
    model = AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    targets = validate_lora_targets((name for name, _ in model.named_modules()), ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    model = get_peft_model(model, LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=list(targets), task_type=TaskType.CAUSAL_LM, bias="none"))

    train_records, dev_records, split_manifest = _load_train_partitions(config)
    rationales = _read_rationale_map(config.rationale_run_id, train_records, split_manifest, config) if config.rationale_run_id else None
    train_dataset = _SFTDataset(tokenizer, train_records, config.mode, rationales, config.max_seq_length)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index,
        shuffle=True, seed=config.seed, drop_last=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.per_device_batch_size, sampler=train_sampler, collate_fn=_train_collator(tokenizer), drop_last=False)
    dev_loader = None
    if dev_records:
        dev_dataset = _SelectionGenerationDataset(tokenizer, dev_records, config.max_seq_length)
        dev_sampler = DistributedSampler(
            dev_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index,
            shuffle=False, seed=config.seed, drop_last=False,
        )
        dev_loader = DataLoader(dev_dataset, batch_size=config.per_device_batch_size, sampler=dev_sampler, collate_fn=_selection_generation_collator(tokenizer), drop_last=False)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=config.weight_decay)
    if dev_loader is None:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    else:
        model, optimizer, train_loader, dev_loader = accelerator.prepare(model, optimizer, train_loader, dev_loader)

    # Accelerate may shard/wrap the loader; calculate optimizer updates only
    # after that point so multi-GPU scheduler and refit counts stay exact.
    updates_per_epoch = (len(train_loader) + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps
    if updates_per_epoch < 1:
        raise ContractError("prepared train loader has no optimizer update")
    total_updates = int(config.selected_updates) if config.phase == "refit" else updates_per_epoch * config.epochs
    if config.phase == "refit" and total_updates > updates_per_epoch * config.epochs:
        raise ContractError("selected_updates exceeds the frozen refit epoch budget")
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=int(total_updates * config.warmup_ratio), num_training_steps=total_updates)

    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "config.json", asdict(config))
        _write_json(run_dir / "train_partition_mean.json", {k: sum(float(x["score"][k]) for x in train_records) / len(train_records) for k in SCORE_KEYS})
    accelerator.wait_for_everyone()
    wandb_run = _wandb_init(accelerator, config)
    updates = 0
    epoch_loss_sum = 0.0
    epoch_loss_count = 0
    best_dev_mae = float("inf")
    best_updates: int | None = None
    convergence_reference = float("inf")
    no_convergence_improvement = 0
    try:
        for epoch in range(config.epochs):
            _set_loader_epoch(train_loader, epoch)
            if dev_loader is not None:
                _set_loader_epoch(dev_loader, epoch)
            model.train()
            for batch in train_loader:
                with accelerator.accumulate(model):
                    output = model(**batch)
                    loss = output.loss
                    assert_finite_loss(loss)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                epoch_loss_sum += float(loss.detach().float().item())
                epoch_loss_count += 1
                if accelerator.sync_gradients:
                    updates += 1
                    if config.phase == "refit" and updates >= int(config.selected_updates):
                        break
            epoch_metrics: dict[str, float | int] = {"train/epoch": epoch + 1, "train/loss": epoch_loss_sum / max(epoch_loss_count, 1), "train/updates": updates}
            if dev_loader is not None:
                dev_mae, dev_parse_failure = _evaluate_selection_dev(accelerator, model, tokenizer, dev_loader, config)
                epoch_metrics["dev/average_mae"] = dev_mae
                epoch_metrics["dev/parse_failure_rate"] = dev_parse_failure
                # Checkpoint selection is strict-lowest MAE, ties retain earliest update.
                if dev_mae < best_dev_mae:
                    best_dev_mae, best_updates = dev_mae, updates
                if convergence_reference - dev_mae > 0.001:
                    convergence_reference = dev_mae
                    no_convergence_improvement = 0
                else:
                    no_convergence_improvement += 1
            _log_aggregate(wandb_run, accelerator, epoch_metrics)
            _save_checkpoint(accelerator, model, run_dir, epoch + 1, updates)
            if config.phase == "selection" and no_convergence_improvement >= 3:
                break
            if config.phase == "refit" and updates >= int(config.selected_updates):
                break
        if config.phase == "refit" and updates != config.selected_updates:
            raise RuntimeError("refit ended before the selected optimizer-update count")
        if config.phase == "selection" and best_updates is None:
            raise RuntimeError("selection produced no development checkpoint")
        if accelerator.is_main_process:
            summary = {"status": "completed", "updates": updates, "selected_updates": best_updates, "best_dev_average_mae": best_dev_mae if best_updates is not None else None, "ended_at": datetime.now(UTC).isoformat()}
            _write_json(run_dir / "training_complete.json", summary)
            _write_run_manifest(
                run_dir, config, canonical_config_hash, split_manifest,
                {"updates": updates, "selected_updates": best_updates or 0, "best_dev_average_mae": best_dev_mae if best_updates is not None else -1.0},
                accelerator,
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _evaluate_selection_dev(accelerator: Any, model: Any, tokenizer: Any, loader: Any, config: DecoderTrainConfig) -> tuple[float, float]:
    """Generation-based internal dev score used only for checkpoint selection."""
    import torch

    model.eval()
    absolute_error_sum = 0.0
    count = 0
    invalid = 0
    fallback = None  # initialized from local optimization train mean after directory exists
    with open(Path(config.output_dir) / "train_partition_mean.json", encoding="utf-8") as handle:
        fallback = json.load(handle)
    with torch.inference_mode():
        for batch in loader:
            generated = accelerator.unwrap_model(model).generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], do_sample=False, max_new_tokens=config.max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            texts = tokenizer.batch_decode(generated[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
            local_errors: list[float] = []
            local_invalid: list[int] = []
            for text, truth, essay in zip(texts, batch["scores"].tolist(), batch["essays"]):
                parsed = parse_decoder_output(text, config.mode, fallback, essay=None if config.mode == "direct" else essay)
                local_errors.append(abs(parsed.scores["average"] - float(truth[3])))
                local_invalid.append(int(not parsed.valid))
            gathered_error = accelerator.gather_for_metrics(torch.tensor(local_errors, device=accelerator.device, dtype=torch.float32))
            gathered_invalid = accelerator.gather_for_metrics(torch.tensor(local_invalid, device=accelerator.device, dtype=torch.float32))
            if accelerator.is_main_process:
                absolute_error_sum += float(gathered_error.sum().item())
                count += int(gathered_error.numel())
                invalid += int(gathered_invalid.sum().item())
    model.train()
    # ``reduce`` is all-rank and avoids a main-only gather/broadcast mismatch.
    values = accelerator.reduce(
        torch.tensor([absolute_error_sum, count, invalid], device=accelerator.device, dtype=torch.float64),
        reduction="sum",
    )
    return float(values[0].item() / max(values[1].item(), 1.0)), float(values[2].item() / max(values[1].item(), 1.0))


def _save_checkpoint(accelerator: Any, model: Any, run_dir: Path, epoch: int, updates: int) -> None:
    checkpoint = run_dir / f"checkpoint-epoch-{epoch:02d}"
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint / "accelerate"))
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(checkpoint / "adapter", safe_serialization=True)
        _write_json(checkpoint / "metadata.json", {"epoch": epoch, "optimizer_updates": updates})
    accelerator.wait_for_everyone()


def _wandb_init(accelerator: Any, config: DecoderTrainConfig) -> Any | None:
    from .provenance import wandb_rank_zero_init

    # Immutable W&B run ID and ``resume=never`` are enforced by the shared
    # guard.  Deliberately omit paths and all example-level fields.
    return wandb_rank_zero_init(
        project=config.wandb_project,
        run_id=config.run_id,
        rank=accelerator.process_index,
        config={
            "run_id": config.run_id,
            "phase": config.phase,
            "mode": config.mode,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "seed": config.seed,
            "world_size": accelerator.num_processes,
        },
    )


def _log_aggregate(wandb_run: Any | None, accelerator: Any, values: Mapping[str, float | int]) -> None:
    if accelerator.is_main_process:
        from .provenance import wandb_log_aggregates

        wandb_log_aggregates(wandb_run, values, step=int(values.get("train/updates", 0)))


def _set_loader_epoch(loader: Any, epoch: int) -> None:
    """Advance both ordinary and Accelerate-wrapped distributed samplers."""
    if hasattr(loader, "set_epoch"):
        loader.set_epoch(epoch)
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    batch_sampler = getattr(loader, "batch_sampler", None)
    nested_sampler = getattr(batch_sampler, "sampler", None)
    if hasattr(nested_sampler, "set_epoch"):
        nested_sampler.set_epoch(epoch)


def _write_run_manifest(
    run_dir: Path,
    config: DecoderTrainConfig,
    canonical_config_hash: str,
    split_manifest: Mapping[str, Any],
    metrics: Mapping[str, float | int],
    accelerator: Any,
) -> None:
    """Write one aggregate-only completion manifest in the checked run root."""
    from .provenance import aggregate_only_payload, build_run_manifest

    config_hash = hashlib.sha256(
        json.dumps(asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    gpu_name = "unavailable"
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(accelerator.local_process_index)
    except Exception:
        pass
    manifest = build_run_manifest(
        run_id=config.run_id,
        config_hash=config_hash,
        data_contract={
            "train_sha256": config.train_sha256,
            "partition_algorithm": str(split_manifest["selection_algorithm"]),
            "optimization_train_records": int(split_manifest["optimization_train_records"]),
            "development_records": int(split_manifest["development_records"]),
            "optimization_record_id_sha256": str(split_manifest["optimization_record_id_sha256"]),
        },
        command=" ".join(sys.argv),
        output_path=str(run_dir),
        extra={
            "canonical_config_hash": canonical_config_hash,
            "phase": config.phase,
            "mode": config.mode,
            "model_revision": config.model_revision,
            "tokenizer_revision": config.tokenizer_revision,
            "seed": config.seed,
            "world_size": accelerator.num_processes,
            "gpu_name": gpu_name,
            "metrics": dict(metrics),
            "deviations": "none",
        },
    )
    destination = run_dir / "run_manifest.json"
    if destination.exists():
        raise ContractError("refusing to overwrite run manifest")
    destination.write_text(json.dumps(aggregate_only_payload(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5 decoder SFT selection/refit runner")
    parser.add_argument("--config", required=True, help="non-secret decoder JSON configuration")
    args = parser.parse_args(argv)
    train(load_json_config(args.config))


if __name__ == "__main__":  # pragma: no cover
    main()
