#!/usr/bin/env python3
"""Materialize the eight hash-bound score-encoder arm configurations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import SOURCE_SHA256, sha256_file  # noqa: E402


MODELS = {
    "qwen3_embedding_8b": {
        "slug": "qwen3-embedding-8b",
        "id": "Qwen/Qwen3-Embedding-8B",
        "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "path": ROOT / "outputs/model-cache/Qwen--Qwen3-Embedding-8B-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "completion": ROOT / "outputs/official-aihub-integer-score-full-pretrain-v1/official-aihub-integer-score-full-pretrain-v1-20260728-002/bounded_regression-refit/training_complete.json",
        "max_length": 2560, "train_batch": 8, "eval_batch": 8, "accumulation": 1, "dtype": "bfloat16",
    },
    "kure_v1": {
        "slug": "kure-v1", "id": "nlpai-lab/KURE-v1",
        "revision": "d14c8a9423946e268a0c9952fecf3a7aabd73bd9",
        "path": ROOT / "outputs/model-cache/nlpai-lab--KURE-v1-d14c8a9423946e268a0c9952fecf3a7aabd73bd9",
        "completion": ROOT / "outputs/official-kure-aihub-score-full-pretrain-v1/official-kure-aihub-score-full-pretrain-v1-20260729-003/training_complete.json",
        "max_length": 2048, "train_batch": 16, "eval_batch": 32, "accumulation": 1, "dtype": "float32",
    },
}


def need(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--handoff", type=Path, required=True); parser.add_argument("--suffix", default="20260807-001"); parser.add_argument("--score-balance-mode", choices=("none", "per_axis_exact_inverse_frequency_loss"), default="none"); parser.add_argument("--model-key", choices=tuple(MODELS), action="append"); parser.add_argument("--training-protocol", choices=("select_then_refit", "fixed_full_train"), default="select_then_refit"); parser.add_argument("--fixed-epoch-map", type=Path); args = parser.parse_args()
    setproctitle("mal2026:prepare-score-encoder-configs")
    handoff = json.loads(args.handoff.read_text(encoding="utf-8")); need(handoff.get("schema_version") == "mal2026-rationale-pipeline-encoder-ratio-handoff-v2" and handoff.get("status") == "completed", "score encoder handoff differs")
    ratio = str(handoff.get("arm")); need(ratio in {"1to1", "1to2", "1to3"}, "score encoder handoff ratio differs")
    selected_models = set(args.model_key or MODELS)
    need(selected_models and selected_models <= set(MODELS), "score encoder model filter differs")
    if args.training_protocol == "fixed_full_train":
        need(args.fixed_epoch_map is not None and args.fixed_epoch_map.is_file(), "fixed epoch map unavailable")
        fixed_epoch_map = json.loads(args.fixed_epoch_map.read_text(encoding="utf-8"))
        need(isinstance(fixed_epoch_map, dict), "fixed epoch map differs")
    else:
        need(args.fixed_epoch_map is None, "select-then-refit received a fixed epoch map")
        fixed_epoch_map = {}
    created = []
    for model_key, model in MODELS.items():
        if model_key not in selected_models:
            continue
        completion_path = Path(model["completion"]); completion = json.loads(completion_path.read_text(encoding="utf-8")); state = completion.get("state")
        need(completion.get("status") == "completed" and isinstance(state, dict), "AI-Hub encoder completion differs")
        artifact = Path(state["artifact_path"]); artifact_sha = state["artifact_sha256"]
        for objective in ("bounded_regression", "categorical_5class"):
            for initialization in ("base", "aihub"):
                balance_slug = "" if args.score_balance_mode == "none" else "-balanced-exact-loss"
                epoch_key = f"{model_key}|{objective}|{initialization}"
                if args.training_protocol == "fixed_full_train":
                    fixed = fixed_epoch_map.get(epoch_key)
                    need(isinstance(fixed, dict) and isinstance(fixed.get("epochs"), int) and 1 <= int(fixed["epochs"]) <= 8 and isinstance(fixed.get("source"), str) and bool(fixed["source"].strip()), f"fixed epoch entry differs: {epoch_key}")
                    fixed_epochs = int(fixed["epochs"]); fixed_epoch_source = str(fixed["source"])
                    protocol_slug = f"-fixed-full-e{fixed_epochs}"
                else:
                    fixed_epochs = None; fixed_epoch_source = None; protocol_slug = ""
                run_id = f"rationale-pipeline-score-encoder-{ratio}{balance_slug}{protocol_slug}-{model['slug']}-{objective.replace('_', '-')}-{initialization}-{args.suffix}"
                destination = ROOT / "configs" / f"{run_id}.json"; need(not destination.exists(), f"score encoder config exists: {destination.name}")
                authorization = (
                    "2026-08-10: user explicitly instructed removing duplicate epoch-selection plus refit training and continuing faster with a predeclared fixed full-data epoch count on GPUs 0-3"
                    if args.training_protocol == "fixed_full_train"
                    else ("2026-08-10: user explicitly instructed evaluation after Qwen 1:2 followed by a score-distribution-balanced experiment on GPUs 0-3" if args.score_balance_mode != "none" else "2026-08-07: user explicitly authorized GPUs 0-3 for the rationale and score training pipeline")
                )
                value = {
                    "schema_version": "mal2026-rationale-pipeline-score-encoder-v1", "run_id": run_id,
                    "model_key": model_key, "model_id": model["id"], "model_revision": model["revision"], "model_path": str(Path(model["path"]).resolve()),
                    "objective": objective, "initialization": initialization,
                    "aihub_completion_path": str(completion_path.resolve()) if initialization == "aihub" else None,
                    "aihub_completion_sha256": sha256_file(completion_path) if initialization == "aihub" else None,
                    "aihub_artifact_path": str(artifact.resolve()) if initialization == "aihub" else None,
                    "aihub_artifact_sha256": artifact_sha if initialization == "aihub" else None,
                    "train_path": str((ROOT / "eval/train.jsonl").resolve()), "train_sha256": SOURCE_SHA256["train"],
                    "validation_path": str((ROOT / "eval/validation.jsonl").resolve()), "validation_sha256": SOURCE_SHA256["validation"],
                    "rationale_handoff_path": str(args.handoff.resolve()), "rationale_handoff_sha256": sha256_file(args.handoff),
                    "rationale_ratio": ratio,
                    "seed": 2026080707, "epochs": list(range(1, 9)), "learning_rate": 1e-4, "weight_decay": 0.01, "warmup_ratio": 0.05,
                    "max_length": model["max_length"], "per_device_train_batch_size": model["train_batch"], "per_device_eval_batch_size": model["eval_batch"], "gradient_accumulation_steps": model["accumulation"],
                    "gradient_checkpointing": True,
                    "selective_gradient_checkpointing_stride": None,
                    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "training_dtype": model["dtype"],
                    "classification_weighting": "inverse_sqrt_train_class_frequency_normalized_per_axis" if args.score_balance_mode == "none" else "per_example_per_axis_exact_inverse_frequency",
                    "score_balance_mode": args.score_balance_mode,
                    "training_protocol": args.training_protocol,
                    "fixed_epochs": fixed_epochs,
                    "fixed_epoch_source": fixed_epoch_source,
                    "gpu_scope": [0, 1, 2, 3], "user_authorization": authorization,
                }
                destination.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); created.append(str(destination.resolve()))
    print(json.dumps({"status": "completed", "configs": created}, sort_keys=True))


if __name__ == "__main__": main()
