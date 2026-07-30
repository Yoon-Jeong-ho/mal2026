#!/usr/bin/env python3
"""Train one leakage-controlled OOF scorer fold and predict Solar candidates."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.evaluation_prompt_score_encoder import (  # noqa: E402
    EvaluationPromptScoreEncoderConfig,
    make_dataset,
    token_length_audit,
)
from mal2026.official_score_matrix import AXES, decode_logits, file_sha256, score_metrics  # noqa: E402
from mal2026.rationale_aware_encoder import (  # noqa: E402
    ContinuousScoreRow,
    build_model,
    collator,
    load_continuous_rows,
    trainable_state,
    verify_artifact_inventory,
)
from mal2026.solar_consensus_pilot import stratified_fold_assignments  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs/solar-consensus-oof-v1"
RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_consensus_oof_v1"
PILOT_RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_consensus_pilot_v1"
FOLDS = 5
FOLD_SEED = 2026073002
MODEL_PROTOCOL = {
    "qwen3_embedding_8b": {
        "config": ROOT / "configs/evaluation_prompt_score_encoder.qwen3_embedding_8b.direct.v1.json",
        "epochs": 3,
        "gradient_accumulation": 8,
        "prior_result": ROOT / "outputs/evaluation-prompt-score-encoder-v1/evaluation-prompt-score-encoder-v1-qwen3-embedding-8b-direct-20260729-001/result.json",
    },
    "kure_v1": {
        "config": ROOT / "configs/evaluation_prompt_score_encoder.kure_v1.direct.v1.json",
        "epochs": 5,
        "gradient_accumulation": 4,
        "prior_result": ROOT / "outputs/evaluation-prompt-score-encoder-v1/evaluation-prompt-score-encoder-v1-kure-v1-direct-20260729-001/result.json",
    },
}


class SolarConsensusOOFError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SolarConsensusOOFError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    need(path.is_file() and not path.is_symlink(), f"input is unavailable: {path.name}")
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    need(values and all(isinstance(value, dict) for value in values), f"input differs: {path.name}")
    return values


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    need(not path.exists(), "prediction output must be fresh")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return file_sha256(path)


def candidate_rows(
    raw_rows: Sequence[Mapping[str, Any]], assignments: Mapping[str, int], fold: int,
) -> tuple[list[ContinuousScoreRow], list[Mapping[str, Any]]]:
    selected: list[Mapping[str, Any]] = []
    converted: list[ContinuousScoreRow] = []
    seen: set[str] = set()
    for raw in raw_rows:
        candidate_id = raw.get("candidate_id")
        source_id = raw.get("source_id")
        score = raw.get("score")
        need(isinstance(candidate_id, str) and candidate_id not in seen and
             isinstance(source_id, str) and source_id in assignments and
             isinstance(score, dict), "candidate OOF lineage differs")
        seen.add(candidate_id)
        if assignments[source_id] != fold:
            continue
        labels = tuple(float(score[axis]) for axis in AXES)
        converted.append(ContinuousScoreRow(
            identifier=candidate_id,
            document_id=str(raw.get("source_document_id", source_id)),
            prompt_num="synthetic_candidate",
            prompt=str(raw["prompt"]),
            essay=str(raw["essay"]),
            labels=labels,  # labels are carried only because Trainer predict expects the field
        ))
        selected.append(raw)
    need(len(converted) == len(selected), "candidate conversion differs")
    return converted, selected


def prediction_rows(
    rows: Sequence[ContinuousScoreRow], continuous: Sequence[Sequence[float]],
    integers: Sequence[Sequence[int]], *, include_reference: bool,
    source_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    need(len(rows) == len(continuous) == len(integers), "OOF prediction population differs")
    result: list[dict[str, Any]] = []
    for index, (row, raw_values, integer_values) in enumerate(
        zip(rows, continuous, integers, strict=True)
    ):
        value: dict[str, Any] = {
            "record_id": row.identifier,
            "continuous_prediction": {
                axis: float(raw_values[axis_index]) for axis_index, axis in enumerate(AXES)
            },
            "integer_prediction": {
                axis: int(integer_values[axis_index]) for axis_index, axis in enumerate(AXES)
            },
        }
        if include_reference:
            value["reference_score"] = {
                axis: float(row.labels[axis_index]) for axis_index, axis in enumerate(AXES)
            }
        else:
            assert source_rows is not None
            value.update({
                "candidate_id": row.identifier,
                "source_id": source_rows[index]["source_id"],
                "solar_modal_score": source_rows[index]["score"],
            })
        result.append(value)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pilot-run-id", required=True)
    parser.add_argument("--model-key", choices=tuple(MODEL_PROTOCOL), required=True)
    parser.add_argument("--fold", type=int, choices=range(FOLDS), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.physical_gpu),
         "visible GPU does not match declared physical GPU")
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    need(torch.cuda.device_count() == 1, "OOF fold must see exactly one GPU")
    protocol = MODEL_PROTOCOL[args.model_key]
    config = EvaluationPromptScoreEncoderConfig.from_json(protocol["config"])
    prior_result_path = protocol["prior_result"]
    need(prior_result_path.is_file(), "prior fixed-epoch result is unavailable")
    prior = json.loads(prior_result_path.read_text(encoding="utf-8"))
    need(prior.get("status") == "completed" and
         int(prior["selection"]["selected"]["epoch"]) == int(protocol["epochs"]),
         "fixed OOF epoch provenance differs")
    output = OUTPUT_ROOT / args.run_id / args.model_key / (
        f"smoke-fold-{args.fold:02d}" if args.smoke else f"fold-{args.fold:02d}"
    )
    restricted = RESTRICTED_ROOT / args.run_id / args.model_key / (
        f"smoke-fold-{args.fold:02d}" if args.smoke else f"fold-{args.fold:02d}"
    )
    need(not output.exists() and not restricted.exists(), "OOF outputs must be fresh")
    output.mkdir(mode=0o700, parents=True)
    restricted.mkdir(mode=0o700, parents=True)

    all_train = load_continuous_rows(Path(config.train_path), config.train_sha256, 2000)
    assignments = stratified_fold_assignments(all_train, FOLDS, FOLD_SEED)
    train_rows = [row for row in all_train if assignments[row.identifier] != args.fold]
    heldout_rows = [row for row in all_train if assignments[row.identifier] == args.fold]
    need(len(train_rows) + len(heldout_rows) == 2000 and
         set(row.identifier for row in train_rows).isdisjoint(
             row.identifier for row in heldout_rows
         ), "OOF train/heldout isolation differs")
    candidate_path = (
        PILOT_RESTRICTED_ROOT / args.pilot_run_id / "stable_modal_candidates.jsonl"
    )
    raw_candidates = read_jsonl(candidate_path)
    candidates, candidate_sources = candidate_rows(raw_candidates, assignments, args.fold)
    need(candidates, "OOF fold has no pilot candidates")

    seed = int(config.seed) + args.fold
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        revision=config.model_revision,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        need(tokenizer.eos_token is not None, "OOF tokenizer has no pad token")
        tokenizer.pad_token = tokenizer.eos_token
    train_audit = token_length_audit(train_rows, config, None, tokenizer)
    heldout_audit = token_length_audit(heldout_rows, config, None, tokenizer)
    candidate_audit = token_length_audit(candidates, config, None, tokenizer)
    verified = verify_artifact_inventory(config)  # type: ignore[arg-type]
    model, initialization = build_model(config)  # type: ignore[arg-type]
    if args.smoke:
        train_rows = train_rows[:4]
        heldout_rows = heldout_rows[:4]
        candidates = candidates[:4]
        candidate_sources = candidate_sources[:4]
    train_dataset = make_dataset(train_rows, config, None, tokenizer)
    heldout_dataset = make_dataset(heldout_rows, config, None, tokenizer)
    candidate_dataset = make_dataset(candidates, config, None, tokenizer)
    training_args = TrainingArguments(
        output_dir=str(output / "trainer"),
        do_train=True,
        do_eval=False,
        eval_strategy="no",
        save_strategy="no",
        num_train_epochs=float(protocol["epochs"]),
        max_steps=1 if args.smoke else -1,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        optim="adamw_torch_fused",
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=1 if args.smoke else int(protocol["gradient_accumulation"]),
        bf16=config.training_dtype == "bfloat16",
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        logging_steps=1 if args.smoke else 5,
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator(tokenizer),
    )
    training = trainer.train()
    need(math.isfinite(float(training.metrics["train_loss"])), "OOF train loss is non-finite")
    heldout_prediction = trainer.predict(heldout_dataset)
    candidate_prediction = trainer.predict(candidate_dataset)
    heldout_continuous, heldout_integer, heldout_violations = decode_logits(
        torch.as_tensor(heldout_prediction.predictions), "bounded_regression"
    )
    candidate_continuous, candidate_integer, candidate_violations = decode_logits(
        torch.as_tensor(candidate_prediction.predictions), "bounded_regression"
    )
    heldout_metrics = score_metrics(
        heldout_prediction.label_ids.tolist(),
        heldout_continuous.tolist(),
        heldout_integer.tolist(),
        heldout_violations.tolist(),
    )
    need(math.isfinite(float(heldout_metrics["macro_continuous_rmse"])),
         "OOF heldout metric is non-finite")
    state_path = output / "trainable_state.safetensors"
    save_file(trainable_state(model), str(state_path))
    original_path = restricted / "original_oof_predictions.jsonl"
    candidate_prediction_path = restricted / "candidate_predictions.jsonl"
    original_hash = write_jsonl(
        original_path,
        prediction_rows(
            heldout_rows, heldout_continuous.tolist(), heldout_integer.tolist(),
            include_reference=True,
        ),
    )
    candidate_hash = write_jsonl(
        candidate_prediction_path,
        prediction_rows(
            candidates, candidate_continuous.tolist(), candidate_integer.tolist(),
            include_reference=False, source_rows=candidate_sources,
        ),
    )
    assignment_fingerprint = sha256("\n".join(
        f"{identifier}:{fold}" for identifier, fold in sorted(assignments.items())
    ).encode()).hexdigest()
    result = {
        "schema_version": "mal2026-solar-consensus-oof-fold-result-v1",
        "status": "completed",
        "mode": "one_update_smoke" if args.smoke else "full_fold",
        "completed_at": now(),
        "run_id": args.run_id,
        "pilot_run_id": args.pilot_run_id,
        "model_key": args.model_key,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "fold": args.fold,
        "folds": FOLDS,
        "physical_gpu": args.physical_gpu,
        "train_records": len(train_rows),
        "heldout_original_records": len(heldout_rows),
        "candidate_records": len(candidates),
        "fixed_epochs": int(protocol["epochs"]),
        "fixed_epoch_source_result": str(prior_result_path.resolve()),
        "fixed_epoch_source_result_sha256": file_sha256(prior_result_path),
        "effective_batch_size": (
            config.per_device_train_batch_size *
            (1 if args.smoke else int(protocol["gradient_accumulation"]))
        ),
        "fold_seed": FOLD_SEED,
        "fold_assignment_fingerprint": assignment_fingerprint,
        "average_target_used": False,
        "validation_rows_loaded": False,
        "validation_rows_directly_scored": False,
        "fixed_epoch_source_may_reflect_prior_validation_selection": True,
        "candidate_solar_labels_used_for_training": False,
        "heldout_original_metrics": heldout_metrics,
        "token_length_audits": {
            "train": train_audit,
            "heldout_original": heldout_audit,
            "candidates": candidate_audit,
        },
        "warmstart_verification": verified,
        "initialization": initialization,
        "trainable_state_sha256": file_sha256(state_path),
        "prediction_outputs": {
            "original_oof_path": str(original_path.resolve()),
            "original_oof_sha256": original_hash,
            "candidate_path": str(candidate_prediction_path.resolve()),
            "candidate_sha256": candidate_hash,
        },
        "bindings": {
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "worktree_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "base_config_sha256": file_sha256(protocol["config"]),
            "train_sha256": config.train_sha256,
            "candidate_input_sha256": file_sha256(candidate_path),
            "environment": {
                "torch": torch.__version__,
            },
        },
        "privacy": "aggregate contains no essay, prompt, rationale, identifier, or individual prediction",
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"], "mode": result["mode"],
        "model_key": args.model_key, "fold": args.fold,
        "heldout_macro_rmse": heldout_metrics["macro_continuous_rmse"],
        "candidate_records": len(candidates),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
