#!/usr/bin/env python3
"""Recompute existing Qwen3 score candidates under the official integer contract.

Per-row predictions are written only to the ignored restricted data root so a
later score-conditioned rationale generator can consume the exact deployed
integers.  Public outputs contain aggregate metrics and provenance only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import EXPECTED_ESSAYS, SOURCE_SHA256  # noqa: E402
from mal2026.official_writing_contract import AXES, integerize_score  # noqa: E402
from mal2026.rlaif_qwen3_embedding import (  # noqa: E402
    _collator as r0_collator,
    _examples as r0_examples,
    _tokenized as r0_tokenized,
    build_model as r0_build_model,
)
from mal2026.rlaif_qwen3_epoch_sweep import EpochSweepTrainConfig  # noqa: E402
from mal2026.rlaif_qwen3_improvement import (  # noqa: E402
    ImprovementTrainConfig,
    collator as improvement_collator,
    examples as improvement_examples,
    model_for_arm,
    tokenized as improvement_tokenized,
)
from mal2026.rlaif_top3_encoder import three_axis_metrics  # noqa: E402


R0_TRAINING = ROOT / "outputs/rlaif-qwen3-embedding-epoch-sweep-v1/rlaif-qwen3-embedding-epoch-sweep-v1-full-003/training_complete.json"
IMPROVEMENT_TRAINING = {
    "essay_only": ROOT / "outputs/rlaif-qwen3-embedding-improvement-v1/rlaif-qwen3-improvement-v1-essay_only-full-006/training_complete.json",
    "essay_instruction": ROOT / "outputs/rlaif-qwen3-embedding-improvement-v1/rlaif-qwen3-improvement-v1-essay_instruction-full-006/training_complete.json",
    "rationale_instruction": ROOT / "outputs/rlaif-qwen3-embedding-improvement-v1/rlaif-qwen3-improvement-v1-rationale_instruction-full-007/training_complete.json",
}
RESTRICTED_ROOT = ROOT / "data/processed/restricted/official_prompt_alignment_v1/score_predictions"
AGGREGATE_ROOT = ROOT / "outputs/official-prompt-alignment-v1/score-metrics"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = rank
        start = end
    return ranks


def tolerant_spearman(truth: Sequence[float], predicted: Sequence[float]) -> float | None:
    left, right = average_ranks(truth), average_ranks(predicted)
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0 or right_ss == 0:
        return None
    return sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True)) / math.sqrt(left_ss * right_ss)


def official_metrics(truth: Sequence[Sequence[float]], predicted: Sequence[Sequence[float]]) -> tuple[dict[str, Any], list[list[int]]]:
    need(len(truth) == len(predicted) and len(truth) > 0, "official metric rows must align")
    projected = [[integerize_score(float(value)) for value in row] for row in predicted]
    result: dict[str, Any] = {}
    rmses: list[float] = []
    correlations: list[float] = []
    for index, axis in enumerate(AXES):
        observed = [float(row[index]) for row in truth]
        emitted = [int(row[index]) for row in projected]
        rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(observed, emitted, strict=True)) / len(observed))
        spearman = tolerant_spearman(observed, emitted)
        result[axis] = {
            "rmse": rmse,
            "spearman": spearman,
            "integer_histogram": {str(score): count for score, count in sorted(Counter(emitted).items())},
            "unique_predictions": len(set(emitted)),
        }
        rmses.append(rmse)
        if spearman is not None:
            correlations.append(spearman)
    result["three_axis_macro_rmse"] = sum(rmses) / len(rmses)
    result["three_axis_macro_spearman"] = sum(correlations) / len(correlations) if len(correlations) == len(AXES) else None
    result["undefined_spearman_axes"] = [axis for axis in AXES if result[axis]["spearman"] is None]
    return result, projected


def load_json(path: Path) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink(), f"missing provenance: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict) and value.get("status") == "completed", f"incomplete provenance: {path}")
    return value


def load_state(model: Any, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    from safetensors.torch import load_file

    state_path = Path(str(checkpoint["trainable_state_path"]))
    need(state_path.is_file() and file_sha(state_path) == checkpoint["trainable_state_sha256"], "checkpoint checksum differs")
    state = load_file(str(state_path), device="cpu")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    need(set(state) == trainable, "checkpoint trainable tensors differ")
    incompatible = model.load_state_dict(state, strict=False)
    need(not incompatible.unexpected_keys and not (trainable & set(incompatible.missing_keys)), "checkpoint load differs")
    return state


def write_private_predictions(
    path: Path,
    source_ids: Sequence[str],
    continuous: Sequence[Sequence[float]],
    projected: Sequence[Sequence[int]],
) -> str:
    need(len(source_ids) == len(continuous) == len(projected), "private prediction rows differ")
    with path.open("x", encoding="utf-8") as handle:
        for source_id, raw, integer in zip(source_ids, continuous, projected, strict=True):
            row = {
                "source_id": source_id,
                "continuous_prediction": {axis: float(raw[index]) for index, axis in enumerate(AXES)},
                "emitted_integer_prediction": {axis: int(integer[index]) for index, axis in enumerate(AXES)},
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return file_sha(path)


def trainer_for(model: Any, output: Path, batch_size: int, collator: Any) -> Any:
    from transformers import Trainer, TrainingArguments

    return Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output / "trainer"),
            do_train=False,
            do_eval=False,
            per_device_eval_batch_size=batch_size,
            bf16=True,
            tf32=True,
            report_to=[],
            remove_unused_columns=False,
        ),
        data_collator=collator,
    )


def evaluate_r0(limit: int, output: Path, private: Path) -> list[dict[str, Any]]:
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    training = load_json(R0_TRAINING)
    raw_config = dict(training["config"])
    raw_config["score_fields"] = tuple(raw_config["score_fields"])
    config = EpochSweepTrainConfig(**raw_config)
    config.validate(require_fresh_output=False)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    items = r0_examples("validation", limit)
    dataset = r0_tokenized(items, tokenizer, config.max_length, include_source=True)
    model, _ = r0_build_model(config)
    trainer = trainer_for(model, output, 8, r0_collator(tokenizer))
    truth = [[float(value) for value in item["labels"]] for item in items]
    source_ids = [str(item["source_id"]) for item in items]
    states: list[dict[str, Any]] = []
    predictions: list[list[list[float]]] = []
    for checkpoint in training["checkpoints"][:4]:
        states.append(load_state(model, checkpoint))
        raw_prediction = trainer.predict(dataset).predictions
        values = raw_prediction.tolist() if isinstance(raw_prediction, np.ndarray) else raw_prediction
        predictions.append([[float(value) for value in row] for row in values])
    mean_prediction = [[sum(predictions[epoch][row][axis] for epoch in range(4)) / 4.0 for axis in range(3)] for row in range(limit)]
    continuous_metrics = three_axis_metrics(truth, mean_prediction)
    integer_metrics, projected = official_metrics(truth, mean_prediction)
    prediction_path = private / "r0_prediction_ensemble.jsonl"
    prediction_sha = write_private_predictions(prediction_path, source_ids, mean_prediction, projected)
    trainable = sorted(states[0])
    soup = {
        name: sum((state[name].to(torch.float32) for state in states), start=torch.zeros_like(states[0][name], dtype=torch.float32)).div_(4).to(states[0][name].dtype)
        for name in trainable
    }
    incompatible = model.load_state_dict(soup, strict=False)
    need(not incompatible.unexpected_keys, "state soup load differs")
    raw_soup = trainer.predict(dataset).predictions
    soup_values = raw_soup.tolist() if isinstance(raw_soup, np.ndarray) else raw_soup
    soup_prediction = [[float(value) for value in row] for row in soup_values]
    soup_continuous = three_axis_metrics(truth, soup_prediction)
    soup_integer, soup_projected = official_metrics(truth, soup_prediction)
    soup_path = private / "r0_state_soup.jsonl"
    soup_sha = write_private_predictions(soup_path, source_ids, soup_prediction, soup_projected)
    lineage = [{"epoch": int(item["epoch"]), "trainable_state_sha256": item["trainable_state_sha256"]} for item in training["checkpoints"][:4]]
    return [
        {"candidate": "r0_prediction_ensemble", "continuous_metrics": continuous_metrics, "official_integer_metrics": integer_metrics, "prediction_sha256": prediction_sha, "checkpoint_lineage": lineage},
        {"candidate": "r0_state_soup", "continuous_metrics": soup_continuous, "official_integer_metrics": soup_integer, "prediction_sha256": soup_sha, "checkpoint_lineage": lineage},
    ]


def evaluate_improvement(arm: str, limit: int, output: Path, private: Path) -> list[dict[str, Any]]:
    import numpy as np
    from transformers import AutoTokenizer

    training_path = IMPROVEMENT_TRAINING[arm]
    training = load_json(training_path)
    raw_config = dict(training["config"])
    raw_config["score_fields"] = tuple(raw_config["score_fields"])
    config = ImprovementTrainConfig(**raw_config)
    # These immutable checkpoints were produced by recovery lineages -006 and
    # -007.  The current training module intentionally accepts only a fresh
    # -007 launch, so calling its launch validator would incorrectly reject
    # the completed -006 essay-only artifact.  Bind every scientific/model
    # field here instead of weakening the historical trainer contract.
    expected_run = (
        f"rlaif-qwen3-improvement-v1-{arm}-full-006"
        if arm in {"essay_only", "essay_instruction"}
        else "rlaif-qwen3-improvement-v1-rationale_instruction-full-007"
    )
    need(config.schema_version == "mal2026-rlaif-qwen3-improvement-train-v1", "historical improvement schema differs")
    need(config.run_id == expected_run and Path(config.output_dir).name == expected_run, "historical improvement run identity differs")
    need(config.arm == arm and config.phase == "full" and config.score_fields == AXES, "historical improvement task differs")
    need((config.model_id, config.model_revision) == ("Qwen/Qwen3-Embedding-8B", "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"), "historical improvement model differs")
    need((config.seed, config.max_length, config.learning_rate, config.weight_decay, config.warmup_ratio) == (2026072601, 2048, 1e-4, 0.01, 0.05), "historical improvement optimization differs")
    need((config.num_train_epochs, config.max_steps, config.essay_limit) == (4.0, -1, 2000), "historical improvement schedule differs")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    items = improvement_examples(arm, "validation", limit)
    dataset = improvement_tokenized(items, tokenizer, config.max_length)
    model, _ = model_for_arm(config)
    trainer = trainer_for(model, output, 8, improvement_collator(tokenizer))
    truth = [[float(value) for value in item["labels"]] for item in items]
    source_ids = [str(item["source_id"]) for item in items]
    results: list[dict[str, Any]] = []
    for checkpoint in training["checkpoints"]:
        load_state(model, checkpoint)
        raw_prediction = trainer.predict(dataset).predictions
        values = raw_prediction.tolist() if isinstance(raw_prediction, np.ndarray) else raw_prediction
        predicted = [[float(value) for value in row] for row in values]
        continuous_metrics = three_axis_metrics(truth, predicted)
        integer_metrics, projected = official_metrics(truth, predicted)
        epoch = int(checkpoint["epoch"])
        prediction_path = private / f"{arm}_epoch_{epoch:02d}.jsonl"
        prediction_sha = write_private_predictions(prediction_path, source_ids, predicted, projected)
        results.append({
            "candidate": f"{arm}_epoch_{epoch:02d}",
            "continuous_metrics": continuous_metrics,
            "official_integer_metrics": integer_metrics,
            "prediction_sha256": prediction_sha,
            "checkpoint_lineage": [{"epoch": epoch, "trainable_state_sha256": checkpoint["trainable_state_sha256"]}],
        })
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("r0_ensemble", "essay_only", "essay_instruction", "rationale_instruction"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    need(0 < args.limit <= EXPECTED_ESSAYS["validation"], "limit must be in [1,400]")
    need(args.run_id.replace("-", "").replace("_", "").isalnum(), "run id contains unsupported characters")
    output = AGGREGATE_ROOT / args.run_id
    private = RESTRICTED_ROOT / args.run_id
    need(not output.exists() and not private.exists(), "official score output must be fresh")
    output.mkdir(parents=True)
    private.mkdir(mode=0o700, parents=True)
    results = evaluate_r0(args.limit, output, private) if args.arm == "r0_ensemble" else evaluate_improvement(args.arm, args.limit, output, private)
    need(results and all(result["official_integer_metrics"]["three_axis_macro_rmse"] >= 0 for result in results), "official metrics are incomplete")
    report = {
        "schema_version": "mal2026-official-score-metrics-v1",
        "status": "completed",
        "run_id": args.run_id,
        "arm": args.arm,
        "score_projection": "clip_[1,5]_then_decimal_ROUND_HALF_UP",
        "validation_rows": args.limit,
        "official_output_scores": "integer_1_to_5",
        "results": results,
        "canonical_source_sha256": dict(SOURCE_SHA256),
        "restricted_prediction_directory": str(private.resolve()),
        "selection_caveat": "validation was previously exposed; descriptive development evidence only",
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_in_this_report",
    }
    atomic_json(output / "aggregate_metrics.json", report)
    print(json.dumps({"status": "completed", "run_id": args.run_id, "arm": args.arm, "rows": args.limit, "candidates": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
