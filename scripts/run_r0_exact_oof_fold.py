#!/usr/bin/env python3
"""Train and predict one exact R0 five-fold OOF fold.

This runner keeps the historical R0 model, AI-Hub warm start, rationale input,
optimizer constants, and epoch-1--4 prediction ensemble.  It changes only the
training population: four train folds are used to predict the fifth.  Canonical
validation data is never loaded.  Row-level predictions remain under the
ignored restricted data root; outputs contain checkpoints, aggregate metrics,
and immutable provenance only.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import SOURCE_SHA256, TRAIN_SOURCE  # noqa: E402
from mal2026.official_writing_contract import integerize_score  # noqa: E402
from mal2026.rlaif_qwen3_embedding import (  # noqa: E402
    AXES,
    MODEL_ID,
    MODEL_PATH,
    MODEL_REVISION,
    RATIONALE_SOURCE,
    WARMSTART_METADATA,
    _collator,
    _examples,
    _sha,
    _tokenized,
    _trainable_state,
    build_model,
    warmstart_provenance,
)
from mal2026.rlaif_qwen3_epoch_sweep import (  # noqa: E402
    checkpoint_dir,
    training_config as r0_training_config,
)
from mal2026.rlaif_top3_encoder import generation_dir, three_axis_metrics  # noqa: E402
from mal2026.solar_consensus_pilot import stratified_fold_assignments  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "r0-exact-oof-v1"
RESTRICTED_ROOT = ROOT / "data" / "processed" / "restricted" / "r0_exact_oof_v1"
FOLDS = 5
FOLD_SEED = 2026073101
TRAIN_RECORDS = 2000
EPOCHS = (1, 2, 3, 4)
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 8
# Historical R0 used 4 GPUs x batch 4 x accumulation 4 = effective batch 64.
# A single physical-GPU fold therefore uses accumulation 16.
GRADIENT_ACCUMULATION_STEPS = 16
EXPECTED_TRAIN_RECORDS_PER_FOLD = 1600
EXPECTED_HELDOUT_RECORDS_PER_FOLD = 400
EXPECTED_STEPS_PER_EPOCH = 25
RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{2,127}")


class R0ExactOOFError(RuntimeError):
    """Raised when the fixed OOF protocol or an artifact binding differs."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise R0ExactOOFError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assignment_fingerprint(assignments: Mapping[str, int]) -> str:
    payload = "\n".join(
        f"{identifier}:{fold}" for identifier, fold in sorted(assignments.items())
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def exact_protocol() -> dict[str, Any]:
    """Return the old R0 constants, rejecting accidental protocol drift."""
    source = r0_training_config("full")
    need(source["arm"] == "qwen3_aihub_warmstart", "R0 initialization arm differs")
    need(source["source_key"] == RATIONALE_SOURCE, "R0 rationale source differs")
    need(source["score_fields"] == list(AXES) and "average" not in source["score_fields"],
         "R0 target axes differ")
    need(source["model_id"] == MODEL_ID and source["model_revision"] == MODEL_REVISION,
         "R0 model identity differs")
    need(Path(source["model_path"]).resolve() == MODEL_PATH.resolve(),
         "R0 model snapshot differs")
    need(source["initialization"] == "aihub_48016_warmstart" and
         Path(source["warmstart_metadata_path"]).resolve() == WARMSTART_METADATA.resolve(),
         "R0 warm-start binding differs")
    need((source["seed"], source["max_length"], source["learning_rate"],
          source["weight_decay"], source["warmup_ratio"]) ==
         (2026072601, 2048, 1e-4, 0.01, 0.05), "R0 optimizer constants differ")
    need((source["lora_r"], source["lora_alpha"], source["lora_dropout"],
          source["training_dtype"]) == (16, 32, 0.05, "bfloat16"),
         "R0 LoRA or numeric constants differ")
    return source


def expected_checkpoint_steps() -> dict[int, int]:
    return {epoch: epoch * EXPECTED_STEPS_PER_EPOCH for epoch in EPOCHS}


def _read_train_lineage() -> list[SimpleNamespace]:
    """Read only canonical train lineage needed for document-safe folds."""
    need(TRAIN_SOURCE.is_file() and not TRAIN_SOURCE.is_symlink(),
         "canonical train source is unavailable")
    need(file_sha256(TRAIN_SOURCE) == SOURCE_SHA256["train"],
         "canonical train source checksum differs")
    expected = {"id", "document_id", "prompt_num", "prompt", "essay", "score"}
    rows: list[SimpleNamespace] = []
    with TRAIN_SOURCE.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            need(bool(line.strip()), f"blank train row at line {line_number}")
            raw = json.loads(line)
            need(isinstance(raw, dict) and set(raw) == expected,
                 "canonical train schema differs")
            identifier, document_id, prompt_num = (
                raw["id"], raw["document_id"], raw["prompt_num"]
            )
            need(all(isinstance(value, (str, int)) and str(value).strip()
                     for value in (identifier, document_id, prompt_num)),
                 "canonical train lineage differs")
            rows.append(SimpleNamespace(
                identifier=str(identifier),
                document_id=str(document_id),
                prompt_num=str(prompt_num),
            ))
    need(len(rows) == TRAIN_RECORDS, "canonical train population differs")
    need(len({row.identifier for row in rows}) == TRAIN_RECORDS,
         "canonical train source IDs differ")
    need(len({row.document_id for row in rows}) == TRAIN_RECORDS,
         "canonical train document IDs differ")
    return rows


def prepare_fold_population(
    examples: Sequence[Mapping[str, Any]],
    lineage: Sequence[SimpleNamespace],
    fold: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, int], dict[str, Any]]:
    """Assign and split one source/document-safe train-only fold."""
    need(0 <= fold < FOLDS, "fold index differs")
    need(len(examples) == len(lineage) == TRAIN_RECORDS, "OOF population differs")
    by_id = {str(item["source_id"]): item for item in examples}
    need(len(by_id) == TRAIN_RECORDS and set(by_id) == {row.identifier for row in lineage},
         "rationale examples and canonical train lineage differ")
    fold_rows: list[SimpleNamespace] = []
    for row in lineage:
        item = by_id[row.identifier]
        labels = item.get("labels")
        need(isinstance(labels, list) and len(labels) == len(AXES) and
             all(type(value) in {int, float} and math.isfinite(float(value))
                 for value in labels), "three-axis OOF labels differ")
        fold_rows.append(SimpleNamespace(
            identifier=row.identifier,
            document_id=row.document_id,
            prompt_num=row.prompt_num,
            labels=tuple(float(value) for value in labels),
        ))
    assignments = stratified_fold_assignments(fold_rows, FOLDS, FOLD_SEED)
    train = [item for item in examples if assignments[str(item["source_id"])] != fold]
    heldout = [item for item in examples if assignments[str(item["source_id"])] == fold]
    train_ids = {str(item["source_id"]) for item in train}
    heldout_ids = {str(item["source_id"]) for item in heldout}
    documents = {row.identifier: row.document_id for row in lineage}
    train_documents = {documents[identifier] for identifier in train_ids}
    heldout_documents = {documents[identifier] for identifier in heldout_ids}
    need(len(train) == EXPECTED_TRAIN_RECORDS_PER_FOLD and
         len(heldout) == EXPECTED_HELDOUT_RECORDS_PER_FOLD,
         "OOF fold sizes differ")
    need(train_ids.isdisjoint(heldout_ids) and
         train_documents.isdisjoint(heldout_documents),
         "OOF source/document leakage detected")
    need(train_ids | heldout_ids == set(by_id), "OOF fold coverage differs")
    gate = {
        "source_id_disjoint": True,
        "document_id_disjoint": True,
        "complete_train_2000_coverage": True,
        "train_records": len(train),
        "heldout_records": len(heldout),
    }
    return train, heldout, assignments, gate


def uniform_epoch_mean(
    predictions: Sequence[Sequence[Sequence[float]]],
) -> list[list[float]]:
    need(len(predictions) == len(EPOCHS), "epoch prediction count differs")
    row_count = len(predictions[0])
    need(row_count > 0 and all(len(epoch) == row_count for epoch in predictions),
         "epoch prediction populations differ")
    result: list[list[float]] = []
    for row_index in range(row_count):
        vectors = [epoch[row_index] for epoch in predictions]
        need(all(len(vector) == len(AXES) for vector in vectors),
             "epoch prediction axis count differs")
        averaged = [
            sum(float(vector[axis_index]) for vector in vectors) / len(EPOCHS)
            for axis_index in range(len(AXES))
        ]
        need(all(math.isfinite(value) for value in averaged),
             "epoch prediction ensemble is non-finite")
        result.append(averaged)
    return result


def row_predictions(
    heldout: Sequence[Mapping[str, Any]], continuous: Sequence[Sequence[float]], fold: int,
) -> list[dict[str, Any]]:
    need(len(heldout) == len(continuous), "OOF row predictions differ")
    rows: list[dict[str, Any]] = []
    for item, values in zip(heldout, continuous, strict=True):
        labels = item["labels"]
        need(len(values) == len(labels) == len(AXES), "OOF row axes differ")
        rows.append({
            "source_id": str(item["source_id"]),
            "fold": fold,
            "continuous_prediction": {
                axis: float(values[index]) for index, axis in enumerate(AXES)
            },
            "half_up_integer_prediction": {
                axis: integerize_score(float(values[index]))
                for index, axis in enumerate(AXES)
            },
            "reference_score": {
                axis: float(labels[index]) for index, axis in enumerate(AXES)
            },
        })
    return rows


def write_jsonl_fresh(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    need(not path.exists(), "restricted OOF prediction output must be fresh")
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return file_sha256(path)


def write_json_fresh(path: Path, payload: Mapping[str, Any]) -> str:
    need(not path.exists(), f"output must be fresh: {path.name}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return file_sha256(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fold", required=True, type=int, choices=range(FOLDS))
    parser.add_argument("--physical-gpu", required=True, type=int, choices=(0, 1, 2, 3))
    return parser.parse_args()


def validate_runtime_args(args: argparse.Namespace) -> tuple[Path, Path]:
    need(RUN_ID_PATTERN.fullmatch(args.run_id) is not None,
         "run ID must contain only lowercase letters, digits, and hyphens")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.physical_gpu),
         "CUDA visibility does not match the declared physical GPU")
    output = OUTPUT_ROOT / args.run_id / f"fold-{args.fold:02d}"
    restricted = RESTRICTED_ROOT / args.run_id / f"fold-{args.fold:02d}"
    need(not output.exists() and not restricted.exists(), "OOF fold outputs must be fresh")
    return output, restricted


def main() -> None:
    args = parse_args()
    output, restricted = validate_runtime_args(args)
    protocol = exact_protocol()
    warmstart = warmstart_provenance()
    rationale_path = generation_dir(RATIONALE_SOURCE, "train", "full") / "generated_rationales.jsonl"
    need(rationale_path.is_file() and not rationale_path.is_symlink(),
         "R0 train rationale artifact is unavailable")

    try:
        import numpy as np
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("exact R0 OOF requires the existing .venv-standard environment") from exc

    need(torch.cuda.is_available() and torch.cuda.device_count() == 1,
         "one and only one visible CUDA GPU is required")
    need(torch.cuda.current_device() == 0, "visible CUDA device must be logical GPU 0")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    restricted.mkdir(mode=0o700, parents=True, exist_ok=False)

    examples = _examples("train", TRAIN_RECORDS)
    lineage = _read_train_lineage()
    train_rows, heldout_rows, assignments, leakage_gate = prepare_fold_population(
        examples, lineage, args.fold
    )
    seed = int(protocol["seed"])
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model_path"], revision=protocol["model_revision"],
        local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    if tokenizer.pad_token is None:
        need(tokenizer.eos_token is not None, "R0 tokenizer lacks pad/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = _tokenized(
        train_rows, tokenizer, int(protocol["max_length"]), include_source=False
    )
    heldout_dataset = _tokenized(
        heldout_rows, tokenizer, int(protocol["max_length"]), include_source=False
    )

    fold_config = SimpleNamespace(**{
        **protocol,
        "arm": "qwen3_aihub_warmstart",
        "score_fields": AXES,
        "num_train_epochs": float(len(EPOCHS)),
        "train_record_limit": len(train_rows),
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "output_dir": str(output.resolve()),
    })
    model, initialization = build_model(fold_config)  # type: ignore[arg-type]
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    expected_steps = expected_checkpoint_steps()

    class FoldCheckpointCallback(TrainerCallback):
        def on_epoch_end(self, _args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            epoch = int(round(float(state.epoch or 0.0)))
            need(epoch in expected_steps and int(state.global_step) == expected_steps[epoch],
                 "OOF epoch checkpoint boundary differs")
            root = checkpoint_dir(output, epoch)
            need(not root.exists(), "OOF epoch checkpoint must be fresh")
            root.mkdir(parents=True, exist_ok=False)
            state_path = root / "trainable_model.safetensors"
            save_file(_trainable_state(kwargs["model"]), str(state_path))
            metadata = {
                "schema_version": "mal2026-r0-exact-oof-checkpoint-v1",
                "run_id": args.run_id,
                "fold": args.fold,
                "epoch": epoch,
                "global_step": int(state.global_step),
                "trainable_state_sha256": file_sha256(state_path),
                "source_train_sha256": SOURCE_SHA256["train"],
                "rationale_sha256": file_sha256(rationale_path),
                "fold_assignment_fingerprint": assignment_fingerprint(assignments),
                "average_target_used": False,
            }
            write_json_fresh(root / "checkpoint_metadata.json", metadata)
            return control

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output / "trainer"),
            run_name=f"{args.run_id}-fold-{args.fold:02d}",
            do_train=True,
            do_eval=False,
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=int(protocol["logging_steps"]),
            learning_rate=float(protocol["learning_rate"]),
            weight_decay=float(protocol["weight_decay"]),
            warmup_ratio=float(protocol["warmup_ratio"]),
            num_train_epochs=float(len(EPOCHS)),
            max_steps=-1,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            bf16=True,
            tf32=True,
            # Keep the historical R0 wrapper contract: its custom Regressor does
            # not expose the Transformers gradient-checkpointing delegation API.
            # Enabling Trainer-side checkpointing here is both a protocol drift
            # and an integration error for this model.
            gradient_checkpointing=False,
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            max_grad_norm=1.0,
            optim="adamw_torch",
            logging_nan_inf_filter=False,
            seed=seed,
            data_seed=seed,
        ),
        train_dataset=train_dataset,
        data_collator=_collator(tokenizer),
        callbacks=[FoldCheckpointCallback()],
    )
    trained = trainer.train()
    need(math.isfinite(float(trained.metrics["train_loss"])), "OOF train loss is non-finite")
    need(int(trainer.state.global_step) == expected_steps[EPOCHS[-1]],
         "OOF final optimizer step differs")

    epoch_predictions: list[list[list[float]]] = []
    checkpoint_lineage: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        root = checkpoint_dir(output, epoch)
        state_path = root / "trainable_model.safetensors"
        metadata_path = root / "checkpoint_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        need(metadata.get("epoch") == epoch and
             metadata.get("global_step") == expected_steps[epoch] and
             metadata.get("trainable_state_sha256") == file_sha256(state_path),
             "OOF checkpoint metadata differs")
        state = load_file(str(state_path), device="cpu")
        need(set(state) == trainable_names, "OOF checkpoint tensor names differ")
        incompatible = model.load_state_dict(state, strict=False)
        need(not incompatible.unexpected_keys and
             not (trainable_names & set(incompatible.missing_keys)),
             "OOF checkpoint load differs")
        raw_prediction = trainer.predict(heldout_dataset).predictions
        values = raw_prediction.tolist() if isinstance(raw_prediction, np.ndarray) else raw_prediction
        need(len(values) == len(heldout_rows), "OOF heldout prediction count differs")
        epoch_predictions.append([
            [float(value) for value in vector] for vector in values
        ])
        checkpoint_lineage.append({
            "epoch": epoch,
            "global_step": expected_steps[epoch],
            "trainable_state_path": str(state_path.resolve()),
            "trainable_state_sha256": metadata["trainable_state_sha256"],
            "checkpoint_metadata_sha256": file_sha256(metadata_path),
        })

    continuous = uniform_epoch_mean(epoch_predictions)
    private_rows = row_predictions(heldout_rows, continuous, args.fold)
    prediction_path = restricted / "oof_predictions.jsonl"
    prediction_sha = write_jsonl_fresh(prediction_path, private_rows)
    truth = [[float(value) for value in item["labels"]] for item in heldout_rows]
    integers = [
        [integerize_score(value) for value in vector] for vector in continuous
    ]
    continuous_metrics = three_axis_metrics(truth, continuous)
    integer_metrics = three_axis_metrics(truth, integers)
    need(all(math.isfinite(float(metrics[axis][name]))
             for metrics in (continuous_metrics, integer_metrics)
             for axis in AXES for name in ("rmse", "spearman")),
         "OOF aggregate metric is non-finite")

    aggregate = {
        "schema_version": "mal2026-r0-exact-oof-fold-aggregate-v1",
        "fold": args.fold,
        "folds": FOLDS,
        "heldout_records": len(heldout_rows),
        "continuous_metrics": continuous_metrics,
        "half_up_integer_metrics": integer_metrics,
        "average_target_used": False,
        "validation_rows_loaded": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions",
    }
    aggregate_path = output / "aggregate_metrics.json"
    aggregate_sha = write_json_fresh(aggregate_path, aggregate)
    result = {
        "schema_version": "mal2026-r0-exact-oof-fold-result-v1",
        "status": "completed",
        "run_id": args.run_id,
        "fold": args.fold,
        "folds": FOLDS,
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "initialization": initialization,
        "rationale_source": RATIONALE_SOURCE,
        "score_fields": list(AXES),
        "average_target_used": False,
        "validation_rows_loaded": False,
        "validation_rows_directly_scored": False,
        "train_records": len(train_rows),
        "heldout_records": len(heldout_rows),
        "fold_seed": FOLD_SEED,
        "fold_assignment_fingerprint": assignment_fingerprint(assignments),
        "leakage_gate": leakage_gate,
        "ensemble": {
            "epochs": list(EPOCHS),
            "rule": "uniform arithmetic mean of exact epoch prediction vectors",
            "integer_rule": "clip_to_1_5_then_decimal_round_half_up",
            "predictions_per_heldout_row": len(EPOCHS),
        },
        "training": {
            "seed": seed,
            "epochs": len(EPOCHS),
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": (
                PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
            ),
            "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
            "global_steps": expected_steps[EPOCHS[-1]],
            "train_loss": float(trained.metrics["train_loss"]),
            "learning_rate": float(protocol["learning_rate"]),
            "weight_decay": float(protocol["weight_decay"]),
            "warmup_ratio": float(protocol["warmup_ratio"]),
            "max_length": int(protocol["max_length"]),
        },
        "checkpoints": checkpoint_lineage,
        "artifacts": {
            "restricted_oof_path": str(prediction_path.resolve()),
            "restricted_oof_sha256": prediction_sha,
            "aggregate_path": str(aggregate_path.resolve()),
            "aggregate_sha256": aggregate_sha,
        },
        "provenance": {
            "canonical_train_sha256": SOURCE_SHA256["train"],
            "rationale_generation_path": str(rationale_path.resolve()),
            "rationale_generation_sha256": file_sha256(rationale_path),
            "warmstart": warmstart,
            "historical_r0_protocol": {
                key: protocol[key] for key in (
                    "arm", "source_key", "model_id", "model_revision", "initialization",
                    "seed", "max_length", "learning_rate", "weight_decay", "warmup_ratio",
                    "lora_r", "lora_alpha", "lora_dropout", "training_dtype",
                )
            },
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "worktree_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "torch_version": torch.__version__,
        },
        "privacy": "row_level_oof_restricted; outputs_are_aggregate_and_provenance_only",
    }
    result_path = output / "result.json"
    result_sha = write_json_fresh(result_path, result)
    print(json.dumps({
        "status": "completed",
        "run_id": args.run_id,
        "fold": args.fold,
        "heldout_records": len(heldout_rows),
        "result_path": str(result_path.resolve()),
        "result_sha256": result_sha,
        "restricted_oof_sha256": prediction_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
