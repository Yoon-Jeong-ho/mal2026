#!/usr/bin/env python3
"""Frozen-validation, aggregate-only evaluation for the three pre-SFT heads.

This versioned evaluator deliberately accepts only the frozen validation split.
It loads scalar heads one at a time and writes only aggregate metrics and
provenance; no rows, identifiers, prompts, essays, predictions, or outputs are
persisted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any

from mal2026.metrics import compute_regression_metrics
from mal2026.standard_decoder_data import DEFAULT_MANIFEST, ROOT, StandardDecoderContractError, load_frozen_validation
from mal2026.standard_encoder_data import build_encoder_dataset, encoder_collator
from mal2026.standard_encoder_model import EncoderModelSpec, build_encoder_regressor, build_encoder_tokenizer

RUN_ROOT = ROOT / "outputs" / "standard-encoder-runs"
EVAL_ROOT = ROOT / "outputs" / "standard-encoder-evals"
HEAD_FIELDS = ("content", "organization", "expression")


class EnsembleValidationError(StandardDecoderContractError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise EnsembleValidationError(message)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@dataclass(frozen=True)
class Config:
    run_id: str
    source: str
    prepared_manifest: str
    validation_sha256: str
    output_dir: str
    head_completion_paths: dict[str, str]
    per_device_eval_batch_size: int = 1

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "ensemble config has missing or unknown fields")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        need(self.source == "frozen_validation", "this versioned ensemble may read only frozen_validation")
        need(Path(self.prepared_manifest).resolve() == DEFAULT_MANIFEST.resolve(), "must bind canonical aggregate prepared manifest")
        need(isinstance(self.validation_sha256, str) and len(self.validation_sha256) == 64 and all(ch in "0123456789abcdef" for ch in self.validation_sha256), "frozen validation SHA-256 is invalid")
        output = Path(self.output_dir)
        need(output.is_absolute() and output.parent == EVAL_ROOT.resolve() and not output.exists(), "output_dir must be a new direct child of standard encoder evals")
        need(self.per_device_eval_batch_size > 0, "invalid eval batch size")
        need(set(self.head_completion_paths) == set(HEAD_FIELDS), "head completion paths must be exactly content/organization/expression")
        identities: list[dict[str, Any]] = []
        for field in HEAD_FIELDS:
            path = Path(self.head_completion_paths[field])
            need(path.is_absolute() and path.name == "pre_sft_score_head_complete.json" and path.parent.parent == RUN_ROOT.resolve(), "head completion path is outside canonical run root")
            payload = json.loads(path.read_text(encoding="utf-8"))
            need(payload.get("status") == "completed" and payload.get("target_field") == field, f"{field} head is not completed")
            state = path.parent / "final_model" / "model.safetensors"
            need(state.is_file() and payload.get("model_state_sha256") == digest(state), f"{field} model checksum failed")
            raw = payload.get("config")
            need(isinstance(raw, dict), f"{field} config is missing")
            identities.append({key: raw.get(key) for key in ("backbone", "model_id", "model_revision", "tokenizer_revision", "model_path", "max_length", "prepared_manifest", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules")})
        need(all(value == identities[0] for value in identities[1:]), "head model/data architecture identities differ")


def completion(path: Path) -> tuple[dict[str, Any], EncoderModelSpec, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["config"]
    spec = EncoderModelSpec.from_mapping({
        "backbone": raw["backbone"], "model_id": raw["model_id"], "revision": raw["model_revision"],
        "tokenizer_revision": raw["tokenizer_revision"], "model_path": raw["model_path"],
        "pooling": "last_nonpad" if raw["backbone"] == "qwen3_embedding" else "remote_sentence_embedding",
        "normalize_embeddings": True, "lora_target_modules": raw["lora_target_modules"],
        "lora_r": raw["lora_r"], "lora_alpha": raw["lora_alpha"], "lora_dropout": raw["lora_dropout"],
        "nv_snapshot_dir": None, "nv_review": None,
    })
    return payload, spec, path.parent / "final_model" / "model.safetensors"


def run(config: Config) -> dict[str, Any]:
    config.validate()
    try:
        import torch
        from safetensors.torch import load_model
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ensemble evaluator requires project .venv-standard") from exc
    expected_gpu = os.environ.get("MAL2026_RESERVED_PHYSICAL_GPU")
    need(expected_gpu in {"0", "1", "2", "3"} and os.environ.get("CUDA_VISIBLE_DEVICES") == expected_gpu, "ensemble requires its watchdog-assigned CUDA_VISIBLE_DEVICES")
    completed = {field: completion(Path(config.head_completion_paths[field])) for field in HEAD_FIELDS}
    spec = completed["content"][1]
    tokenizer = build_encoder_tokenizer(spec)
    rows = load_frozen_validation(config.validation_sha256)
    dataset = build_encoder_dataset(rows, tokenizer, 2048, HEAD_FIELDS)
    loader = DataLoader(dataset, batch_size=config.per_device_eval_batch_size, shuffle=False, collate_fn=encoder_collator(tokenizer), num_workers=0)
    device = torch.device("cuda")
    scalar_outputs: dict[str, list[float]] = {field: [] for field in HEAD_FIELDS}
    labels_by_row: list[list[float]] = []
    for field, (_, head_spec, state) in completed.items():
        model = build_encoder_regressor(head_spec, (field,))
        missing, unexpected = load_model(model, str(state), strict=False)
        need(not missing and not unexpected, f"{field} saved state differs from scalar architecture")
        model = model.to(device).eval()
        seen_labels: list[list[float]] = []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels")
                inputs = {key: value.to(device) for key, value in batch.items()}
                scalar_outputs[field].extend(model(**inputs)["logits"].detach().float().cpu().reshape(-1).tolist())
                if field == "content":
                    seen_labels.extend(labels.detach().cpu().tolist())
        if field == "content":
            labels_by_row = seen_labels
        need(len(scalar_outputs[field]) == len(rows), f"{field} evaluator returned wrong prediction count")
        del model
        torch.cuda.empty_cache()
    targets: list[dict[str, float]] = []
    predictions: list[dict[str, float]] = []
    for index in range(len(rows)):
        target = {field: float(labels_by_row[index][pos]) for pos, field in enumerate(HEAD_FIELDS)}
        target["average"] = sum(target.values()) / 3.0
        predicted = {field: min(5.0, max(1.0, float(scalar_outputs[field][index]))) for field in HEAD_FIELDS}
        predicted["average"] = sum(predicted.values()) / 3.0
        targets.append(target)
        predictions.append(predicted)
    metrics = compute_regression_metrics(targets, predictions)
    need(len(targets) > 0 and all(math.isfinite(float(value)) for item in metrics["per_target"].values() for value in item.values() if value is not None), "ensemble metrics are non-finite")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    payload = {"status": "completed", "run_id": config.run_id, "source": config.source, "record_count": len(targets), "metrics": metrics,
               "head_completion_sha256": {field: digest(Path(path)) for field, path in config.head_completion_paths.items()},
               "head_model_state_sha256": {field: completed[field][0]["model_state_sha256"] for field in HEAD_FIELDS},
               "average_policy": "predicted_average=(content+organization+expression)/3 outside_model; average has no learned head",
               "config": asdict(config),
               "privacy": "aggregate_only_no_rows_prompts_essays_feedback_ids_predictions_or_model_outputs_persisted"}
    (output / "aggregate_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-config", action="store_true")
    args = parser.parse_args()
    config = Config.from_json(args.config)
    if args.validate_config:
        print(json.dumps({"status": "validated", "run_id": config.run_id, "gpu_free": True}, sort_keys=True))
        return
    print(json.dumps(run(config), sort_keys=True))


if __name__ == "__main__":
    main()
