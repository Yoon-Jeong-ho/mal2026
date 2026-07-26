#!/usr/bin/env python3
"""Evaluate fixed uniform prediction/state ensembles of R0 epochs 1--4."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import EXPECTED_ESSAYS, SOURCE_SHA256  # noqa: E402
from mal2026.rlaif_qwen3_embedding import AXES, MODEL_ID, MODEL_REVISION, _collator, _examples, _sha, _tokenized, build_model  # noqa: E402
from mal2026.rlaif_qwen3_epoch_sweep import EpochSweepTrainConfig  # noqa: E402
from mal2026.rlaif_top3_encoder import three_axis_metrics  # noqa: E402


SOURCE = ROOT / "outputs" / "rlaif-qwen3-embedding-epoch-sweep-v1" / "rlaif-qwen3-embedding-epoch-sweep-v1-full-003" / "training_complete.json"
OUTPUT = ROOT / "outputs" / "rlaif-qwen3-embedding-improvement-evals-v1" / "rlaif-qwen3-improvement-eval-v1-r0-ensemble-full-006"
REPORT = OUTPUT / "ensemble_metrics.json"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    try:
        import numpy as np
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("epoch ensemble requires .venv-standard") from exc
    need(SOURCE.is_file() and not OUTPUT.exists(), "ensemble source/output freshness differs")
    training = json.loads(SOURCE.read_text(encoding="utf-8"))
    raw = dict(training["config"])
    raw["score_fields"] = tuple(raw["score_fields"])
    config = EpochSweepTrainConfig(**raw)
    config.validate(require_fresh_output=False)
    checkpoints = training.get("checkpoints")
    need(isinstance(checkpoints, list) and [item.get("epoch") for item in checkpoints[:4]] == [1, 2, 3, 4], "R0 epoch 1--4 sequence differs")
    checkpoints = checkpoints[:4]
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = _examples("validation", EXPECTED_ESSAYS["validation"])
    dataset = _tokenized(examples, tokenizer, config.max_length, include_source=True)
    model, _ = build_model(config)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    OUTPUT.mkdir(parents=True)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(OUTPUT), do_train=False, do_eval=False, per_device_eval_batch_size=8, bf16=True, tf32=True, report_to=[], remove_unused_columns=False), data_collator=_collator(tokenizer))
    truth = [[float(value) for value in item["labels"]] for item in examples]
    predictions: list[list[list[float]]] = []
    states: list[dict[str, Any]] = []
    lineage = []
    for checkpoint in checkpoints:
        state_path = Path(checkpoint["trainable_state_path"])
        need(state_path.is_file() and _sha(state_path) == checkpoint["trainable_state_sha256"], "R0 checkpoint checksum differs")
        state = load_file(str(state_path), device="cpu")
        need(set(state) == trainable_names and all(tensor.is_floating_point() for tensor in state.values()), "R0 checkpoint tensors cannot be uniformly averaged")
        incompatible = model.load_state_dict(state, strict=False)
        need(not incompatible.unexpected_keys and not (trainable_names & set(incompatible.missing_keys)), "R0 checkpoint load differs")
        raw_prediction = trainer.predict(dataset).predictions
        values = raw_prediction.tolist() if isinstance(raw_prediction, np.ndarray) else raw_prediction
        need(len(values) == len(truth), "R0 prediction count differs")
        predictions.append([[float(value) for value in row] for row in values])
        states.append(state)
        lineage.append({"epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"], "trainable_state_sha256": checkpoint["trainable_state_sha256"]})
    prediction_mean = [[sum(predictions[epoch][row][axis] for epoch in range(4)) / 4 for axis in range(3)] for row in range(len(truth))]
    prediction_metrics = three_axis_metrics(truth, prediction_mean)
    soup = {name: sum((state[name].to(torch.float32) for state in states), start=torch.zeros_like(states[0][name], dtype=torch.float32)).div_(4).to(states[0][name].dtype) for name in sorted(trainable_names)}
    incompatible = model.load_state_dict(soup, strict=False)
    need(not incompatible.unexpected_keys and not (trainable_names & set(incompatible.missing_keys)), "R0 soup load differs")
    raw_soup = trainer.predict(dataset).predictions
    soup_values = raw_soup.tolist() if isinstance(raw_soup, np.ndarray) else raw_soup
    soup_metrics = three_axis_metrics(truth, [[float(value) for value in row] for row in soup_values])
    need(all(math.isfinite(float(metrics[axis][name])) for metrics in (prediction_metrics, soup_metrics) for axis in AXES for name in ("rmse", "spearman")), "ensemble metric is non-finite")
    payload = {
        "schema_version": "mal2026-rlaif-qwen3-epoch-ensemble-v1", "status": "completed",
        "run_id": "rlaif-qwen3-improvement-eval-v1-r0-ensemble-full-006",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "score_fields": list(AXES),
        "average_target_used": False, "source_training_metadata": str(SOURCE.resolve()),
        "source_training_metadata_sha256": file_sha(SOURCE), "checkpoints": lineage,
        "prediction_ensemble": {"rule": "uniform arithmetic mean of four epoch prediction vectors", "metrics": prediction_metrics},
        "state_soup": {"rule": "uniform arithmetic mean of corresponding floating trainable tensors", "metrics": soup_metrics},
        "validation": {"unique_essays": len(truth), "predictions_per_essay_for_prediction_ensemble": 4},
        "canonical_source_sha256": dict(SOURCE_SHA256),
        "selection_caveat": "validation was previously exposed; descriptive development evidence only",
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
    }
    trainer.accelerator.wait_for_everyone()
    failed = False
    if trainer.is_world_process_zero():
        try:
            need(not REPORT.exists(), "ensemble report already exists")
            REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            failed = True
    message: list[Any] = [failed]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(message, src=0)
    if message[0]:
        raise RuntimeError("rank-zero ensemble persistence failed")
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
