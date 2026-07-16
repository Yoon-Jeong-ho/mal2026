"""Strict final-evaluation path for Qwen decoder score models.

It never performs selection or uses labels to alter generation.  Invalid decoder
text remains a scored prediction through the frozen optimization-train mean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .decoder import (
    CANONICAL_VALIDATION_SHA256,
    ContractError,
    SCORE_KEYS,
    parse_decoder_output,
    orderly_distributed_shutdown,
    prompt_text,
    require_canonical_dataset,
    require_immutable_revision,
    require_path_under_run,
    require_tokenizer_chat_template,
    resolve_run_output_dir,
    sanitized_deterministic_generation_config,
)
from .decoder_train import SYSTEM_MESSAGE, _directory_sha256, _records_as_mappings, _set_loader_epoch


@dataclass(frozen=True)
class DecoderEvalConfig:
    run_id: str
    mode: str  # direct | human_feedback
    model_id: str
    model_revision: str
    tokenizer_revision: str
    adapter_path: str
    evaluation_path: str
    output_dir: str
    fallback_mean: dict[str, float]
    selection_run_id: str
    refit_run_id: str
    canonical_config_path: str
    evaluation_sha256: str = CANONICAL_VALIDATION_SHA256
    max_seq_length: int = 2048
    per_device_batch_size: int = 1
    max_new_tokens: int = 256
    wandb_project: str = "mal2026-korean-writing-scoring"
    wandb_entity: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DecoderEvalConfig":
        known = {field.name for field in fields(cls)}
        extra = sorted(set(raw) - known)
        if extra:
            raise ContractError(f"unknown decoder evaluation config fields: {extra}")
        return cls(**dict(raw))

    def validate(self) -> None:
        if self.mode not in {"direct", "human_feedback"}:
            raise ContractError("mode must be direct or human_feedback")
        require_immutable_revision(self.model_revision, "model_revision")
        require_immutable_revision(self.tokenizer_revision, "tokenizer_revision")
        if not self.selection_run_id or not self.refit_run_id:
            raise ContractError("final evaluation requires separate selection_run_id and refit_run_id")
        expected_sequence, expected_new = (2048, 256) if self.mode == "direct" else (4096, 1536)
        if (self.max_seq_length, self.max_new_tokens) != (expected_sequence, expected_new) or self.per_device_batch_size != 1:
            raise ContractError("final decoder evaluation must use the mode-specific frozen token budget and batch size 1")
        run_dir = resolve_run_output_dir(self.run_id, self.output_dir)
        if self.run_id == self.refit_run_id:
            raise ContractError("final evaluation must use a distinct immutable run_id")
        require_canonical_dataset(self.evaluation_path, "validation", self.evaluation_sha256)
        adapter = require_path_under_run(self.adapter_path, self.refit_run_id)
        if not adapter.is_dir():
            raise ContractError("adapter_path must be a refit adapter directory")
        _require_completed_refit(self, adapter)
        resolve_run_output_dir(self.selection_run_id, Path(self.output_dir).parent / self.selection_run_id, must_exist=True)
        if set(self.fallback_mean) != set(SCORE_KEYS):
            raise ContractError("fallback_mean must be the saved optimization-train four-score mean")
        for key in SCORE_KEYS:
            value = float(self.fallback_mean[key])
            if not math.isfinite(value) or not 1 <= value <= 5:
                raise ContractError("fallback_mean values must be finite values in [1, 5]")


def load_json_config(path: str) -> DecoderEvalConfig:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ContractError("decoder eval config must be JSON object")
    config = DecoderEvalConfig.from_mapping(raw)
    config.validate()
    _validate_canonical_contract(config)
    return config


def _validate_canonical_contract(config: DecoderEvalConfig) -> tuple[dict[str, Any], str]:
    from .config import ConfigError, load_experiment_config

    try:
        contract, contract_hash = load_experiment_config(config.canonical_config_path)
    except ConfigError as exc:
        raise ContractError(f"invalid canonical decoder config: {exc}") from exc
    expected_kind = "decoder-direct" if config.mode == "direct" else "decoder-human-feedback-score"
    if contract["run_kind"] != expected_kind:
        raise ContractError("canonical run_kind does not match decoder evaluation mode")
    model, data = contract["model"], contract["data"]
    if (model["id"], model["revision"], model["tokenizer_revision"]) != (config.model_id, config.model_revision, config.tokenizer_revision):
        raise ContractError("evaluation model fields do not match canonical config")
    if (data["max_sequence_length"], data["max_new_tokens"], data["head_fraction"], data["dev_fraction"]) != (config.max_seq_length, config.max_new_tokens, 0.75, 0.20):
        raise ContractError("evaluation token/split policy does not match canonical config")
    return contract, contract_hash


def _require_completed_refit(config: DecoderEvalConfig, adapter: Path) -> None:
    """Bind fallback and adapter to a completed refit artifact, never selection."""
    refit_dir = resolve_run_output_dir(config.refit_run_id, Path(config.output_dir).parent / config.refit_run_id, must_exist=True)
    try:
        completed = json.loads((refit_dir / "training_complete.json").read_text(encoding="utf-8"))
        saved_config = json.loads((refit_dir / "config.json").read_text(encoding="utf-8"))
        saved_mean = json.loads((refit_dir / "fallback_mean.json").read_text(encoding="utf-8"))
        adapter_metadata = json.loads((adapter.parent / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("refit output is missing required completion provenance") from exc
    if completed.get("status") != "completed" or saved_config.get("phase") != "refit":
        raise ContractError("final evaluation accepts only a completed refit output")
    if saved_config.get("mode") != config.mode or saved_config.get("run_id") != config.refit_run_id:
        raise ContractError("refit output mode/run ID does not match final evaluation")
    if (
        saved_config.get("model_id"), saved_config.get("model_revision"), saved_config.get("tokenizer_revision")
    ) != (config.model_id, config.model_revision, config.tokenizer_revision):
        raise ContractError("final evaluation model fields do not match the completed refit output")
    # Re-check the persisted refit-to-selection cryptographic binding rather
    # than trusting that the refit process happened to check it at launch.
    from .decoder_train import DecoderTrainConfig, _verified_selection_fallback_mean, _verify_refit_selection_binding

    try:
        persisted_refit = DecoderTrainConfig.from_mapping(saved_config)
        persisted_refit.validate()
        _verify_refit_selection_binding(persisted_refit)
        if saved_mean != _verified_selection_fallback_mean(persisted_refit):
            raise ContractError("refit fallback mean was not carried from the verified selection partition")
    except (ContractError, TypeError) as exc:
        raise ContractError("completed refit has no valid immutable selection binding") from exc
    if saved_mean != config.fallback_mean:
        raise ContractError("fallback_mean must exactly equal the completed refit train mean")
    if adapter_metadata.get("optimizer_updates") != completed.get("selected_updates"):
        raise ContractError("named refit adapter must equal the completed selected-update count")
    if adapter_metadata.get("adapter_sha256") != _directory_sha256(adapter):
        raise ContractError("named refit adapter checksum does not match checkpoint metadata")


def _head_tail_prompt(ids: list[int], cap: int) -> list[int]:
    if len(ids) <= cap:
        return ids
    head = (cap * 3) // 4
    return ids[:head] + ids[len(ids) - (cap - head) :]


def build_generation_example(tokenizer: Any, record: Mapping[str, Any], input_token_cap: int) -> dict[str, Any]:
    user = prompt_text(record)
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    ids = _head_tail_prompt(ids, input_token_cap)
    return {"input_ids": ids, "scores": [float(record["score"][key]) for key in SCORE_KEYS], "id": record["id"]}


class _GenerationDataset:
    def __init__(self, tokenizer: Any, records: Sequence[Mapping[str, Any]], input_token_cap: int):
        self.items = [build_generation_example(tokenizer, record, input_token_cap) for record in records]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def _collator(tokenizer: Any):
    import torch

    if tokenizer.pad_token_id is None:
        raise ContractError("tokenizer needs a pad token")
    pad = tokenizer.pad_token_id

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        length = max(len(item["input_ids"]) for item in batch)
        return {
            "input_ids": torch.tensor([[pad] * (length - len(item["input_ids"])) + item["input_ids"] for item in batch], dtype=torch.long),
            "attention_mask": torch.tensor([[0] * (length - len(item["input_ids"])) + [1] * len(item["input_ids"]) for item in batch], dtype=torch.long),
            "scores": torch.tensor([item["scores"] for item in batch], dtype=torch.float32),
            "ids": [item["id"] for item in batch],
        }

    return collate


def _rankdata(values: Sequence[float]) -> list[float]:
    ranks = [0.0] * len(values)
    for _, group in _tie_groups(values):
        rank = (group[0] + group[-1] + 2) / 2.0  # one-indexed average rank
        for index in group:
            ranks[index] = rank
    return ranks


def _tie_groups(values: Sequence[float]):
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        yield values[ordered[cursor]], ordered[cursor:end]
        cursor = end


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx = sum((a - mx) ** 2 for a in x)
    dy = sum((b - my) ** 2 for b in y)
    if dx == 0 or dy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(dx * dy)


def _round_half_up_bin(value: float) -> int:
    # scores were already range-clamped by the strict parser/fallback policy.
    return min(5, max(1, int(math.floor(min(5.0, max(1.0, value)) + 0.5))))


def quadratic_weighted_kappa(target: Sequence[float], prediction: Sequence[float]) -> float:
    """Five-bin QWK with the predeclared half-up discretization."""
    if len(target) != len(prediction) or not target:
        return float("nan")
    observed = [[0.0] * 5 for _ in range(5)]
    target_hist = [0.0] * 5
    prediction_hist = [0.0] * 5
    for truth, pred in zip(target, prediction):
        i, j = _round_half_up_bin(truth) - 1, _round_half_up_bin(pred) - 1
        observed[i][j] += 1.0
        target_hist[i] += 1.0
        prediction_hist[j] += 1.0
    weight = lambda i, j: ((i - j) ** 2) / 16.0
    observed_error = sum(weight(i, j) * observed[i][j] for i in range(5) for j in range(5))
    expected_error = sum(weight(i, j) * target_hist[i] * prediction_hist[j] / len(target) for i in range(5) for j in range(5))
    return 1.0 - observed_error / expected_error if expected_error else float("nan")


def aggregate_metrics(target: Sequence[Sequence[float]], prediction: Sequence[Sequence[float]], valid: Sequence[bool]) -> dict[str, float]:
    if not target or len(target) != len(prediction):
        raise ContractError("nonempty aligned target/prediction arrays required")
    result: dict[str, float] = {"count": float(len(target)), "decoder/parse_failure_rate": 1.0 - sum(valid) / len(valid)}
    for index, key in enumerate(SCORE_KEYS):
        truth = [float(row[index]) for row in target]
        pred = [float(row[index]) for row in prediction]
        result[f"{key}/mae"] = sum(abs(a - b) for a, b in zip(truth, pred)) / len(truth)
        result[f"{key}/rmse"] = math.sqrt(sum((a - b) ** 2 for a, b in zip(truth, pred)) / len(truth))
        result[f"{key}/pearson"] = _pearson(truth, pred)
        result[f"{key}/spearman"] = _pearson(_rankdata(truth), _rankdata(pred))
        result[f"{key}/qwk"] = quadratic_weighted_kappa(truth, pred)
    result["primary/macro_mae"] = sum(result[f"{key}/mae"] for key in SCORE_KEYS) / len(SCORE_KEYS)
    return result


def evaluate(config: DecoderEvalConfig) -> None:
    config.validate()
    _, canonical_config_hash = _validate_canonical_contract(config)
    run_dir = resolve_run_output_dir(config.run_id, config.output_dir)
    if run_dir.exists():
        raise ContractError(f"refusing to overwrite run output: {config.output_dir}")
    from accelerate import Accelerator
    from peft import PeftModel
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    accelerator = Accelerator(mixed_precision="bf16")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.tokenizer_revision, use_fast=True)
    canonical_contract, _ = _validate_canonical_contract(config)
    require_tokenizer_chat_template(tokenizer, canonical_contract["model"]["chat_template_sha256"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Explicitly no device_map: Accelerator/DDP controls device placement.
    base = AutoModelForCausalLM.from_pretrained(config.model_id, revision=config.model_revision, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, config.adapter_path, is_trainable=False)
    model.config.use_cache = True
    from .data_contract import load_and_validate_jsonl
    records = _records_as_mappings(load_and_validate_jsonl(config.evaluation_path, expected_sha256=config.evaluation_sha256))
    dataset = _GenerationDataset(tokenizer, records, config.max_seq_length - config.max_new_tokens - 1)
    # Let Accelerator provide the one DDP sharding layer.
    loader = DataLoader(dataset, batch_size=config.per_device_batch_size, shuffle=False, collate_fn=_collator(tokenizer), drop_last=False)
    model, loader = accelerator.prepare(model, loader)
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "config.json", asdict(config))
    accelerator.wait_for_everyone()

    rank_path = run_dir / f"predictions-rank-{accelerator.process_index:03d}.jsonl"
    all_target: list[list[float]] = []
    all_pred: list[list[float]] = []
    all_valid: list[bool] = []
    with open(rank_path, "w", encoding="utf-8") as output:
        model.eval()
        _set_loader_epoch(loader, 0)
        with torch.inference_mode():
            for batch in loader:
                unwrapped = accelerator.unwrap_model(model)
                generation_config = sanitized_deterministic_generation_config(unwrapped.generation_config)
                generated = unwrapped.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], generation_config=generation_config, max_new_tokens=config.max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                generated_text = tokenizer.batch_decode(generated[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
                local_predictions: list[list[float]] = []
                local_valid: list[int] = []
                for identifier, text, truth in zip(batch["ids"], generated_text, batch["scores"].tolist()):
                    parsed = parse_decoder_output(text, config.mode, config.fallback_mean)
                    local_predictions.append([parsed.scores[key] for key in SCORE_KEYS])
                    local_valid.append(int(parsed.valid))
                    # Restricted prediction file contains no prompt/essay/rationale text.
                    output.write(json.dumps({"id": identifier, "prediction": parsed.scores, "parse_valid": parsed.valid, "parse_error": parsed.error}, ensure_ascii=False) + "\n")
                gathered_pred = accelerator.gather_for_metrics(torch.tensor(local_predictions, device=accelerator.device, dtype=torch.float32))
                gathered_truth = accelerator.gather_for_metrics(batch["scores"])
                gathered_valid = accelerator.gather_for_metrics(torch.tensor(local_valid, device=accelerator.device, dtype=torch.int64))
                if accelerator.is_main_process:
                    all_pred.extend(gathered_pred.cpu().tolist())
                    all_target.extend(gathered_truth.cpu().tolist())
                    all_valid.extend(bool(value) for value in gathered_valid.cpu().tolist())
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metrics = aggregate_metrics(all_target, all_pred, all_valid)
        _write_json(run_dir / "metrics.json", metrics)
        _wandb_log(config, metrics, accelerator)
        _write_run_manifest(run_dir, config, canonical_config_hash, metrics, accelerator)
    accelerator.wait_for_everyone()
    orderly_distributed_shutdown()


def _wandb_log(config: DecoderEvalConfig, metrics: Mapping[str, float], accelerator: Any) -> None:
    from .provenance import wandb_log_aggregates, wandb_rank_zero_init

    run = wandb_rank_zero_init(
        project=config.wandb_project,
        run_id=config.run_id,
        rank=accelerator.process_index,
        config={"run_id": config.run_id, "mode": config.mode, "selection_run_id": config.selection_run_id, "refit_run_id": config.refit_run_id},
    )
    try:
        wandb_log_aggregates(run, metrics, step=0)
    finally:
        if run is not None:
            run.finish()


def _write_run_manifest(run_dir: Path, config: DecoderEvalConfig, canonical_config_hash: str, metrics: Mapping[str, float], accelerator: Any) -> None:
    from .provenance import aggregate_only_payload, build_run_manifest

    config_hash = hashlib.sha256(json.dumps(asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
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
        data_contract={"validation_sha256": config.evaluation_sha256, "validation_records": int(metrics["count"])},
        command=" ".join(sys.argv),
        output_path=str(run_dir),
        extra={
            "canonical_config_hash": canonical_config_hash,
            "mode": config.mode,
            "model_revision": config.model_revision,
            "tokenizer_revision": config.tokenizer_revision,
            "selection_run_id": config.selection_run_id,
            "refit_run_id": config.refit_run_id,
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
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="strict final evaluation for Qwen decoder SFT")
    parser.add_argument("--config", required=True, help="non-secret decoder final-evaluation JSON configuration")
    args = parser.parse_args(argv)
    evaluate(load_json_config(args.config))


if __name__ == "__main__":  # pragma: no cover
    main()
