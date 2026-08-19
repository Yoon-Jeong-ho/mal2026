#!/usr/bin/env python3
"""Verify and merge five completed exact-R0 OOF folds.

All row-level material is read from and written to the ignored restricted
root.  Public outputs contain aggregate metrics, numeric counts, fixed
contracts, and cryptographic bindings only.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import SOURCE_SHA256, TRAIN_SOURCE  # noqa: E402
from mal2026.official_writing_contract import integerize_score  # noqa: E402
from mal2026.r0_ordinal_residual import (  # noqa: E402
    BasePredictionContract,
    GOLD_LABEL_POLICY,
)
from mal2026.rlaif_qwen3_embedding import AXES  # noqa: E402
from mal2026.rlaif_top3_encoder import three_axis_metrics  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "r0-exact-oof-v1"
RESTRICTED_ROOT = ROOT / "data" / "processed" / "restricted" / "r0_exact_oof_v1"
FOLDS = 5
ROWS_PER_FOLD = 400
TOTAL_ROWS = 2000
EPOCHS = (1, 2, 3, 4)
EXPECTED_GLOBAL_STEPS = {1: 25, 2: 50, 3: 75, 4: 100}
RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{2,127}")
ROW_FIELDS = {
    "source_id", "fold", "continuous_prediction",
    "half_up_integer_prediction", "reference_score",
}


class R0ExactOOFAggregateError(RuntimeError):
    """Raised when a fold, checksum, provenance, or merge contract differs."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise R0ExactOOFAggregateError(message)


def file_sha256(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"ordinary file required: {path.name}")
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R0ExactOOFAggregateError(f"JSON is unreadable: {path.name}") from exc
    need(isinstance(value, dict), f"JSON object required: {path.name}")
    return value


def canonical_source_ids() -> list[str]:
    """Load only IDs from the immutable canonical train population."""
    need(file_sha256(TRAIN_SOURCE) == SOURCE_SHA256["train"],
         "canonical train checksum differs")
    identifiers: list[str] = []
    with TRAIN_SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            need(isinstance(raw, dict) and isinstance(raw.get("id"), str),
                 "canonical train ID schema differs")
            identifiers.append(raw["id"])
    need(len(identifiers) == TOTAL_ROWS and len(set(identifiers)) == TOTAL_ROWS,
         "canonical train ID population differs")
    return identifiers


def train_residual_contract() -> dict[str, Any]:
    value = {
        "split_role": "train",
        "base_prediction_origin": "oof",
        "base_model_fit_excludes_split": True,
        "evaluation_only": False,
        "gold_label_policy": GOLD_LABEL_POLICY,
        "contains_average_target": False,
    }
    BasePredictionContract.from_mapping(value)
    return value


def _axis_mapping(value: Any, label: str, *, integer: bool) -> dict[str, float | int]:
    need(isinstance(value, dict) and set(value) == set(AXES), f"{label} axes differ")
    result: dict[str, float | int] = {}
    for axis in AXES:
        raw = value[axis]
        if integer:
            need(type(raw) is int and 1 <= raw <= 5, f"{label} must be integer 1..5")
            result[axis] = raw
        else:
            need(type(raw) in {int, float} and not isinstance(raw, bool),
                 f"{label} must be numeric")
            parsed = float(raw)
            need(math.isfinite(parsed) and 1.0 <= parsed <= 5.0,
                 f"{label} must be finite within [1,5]")
            result[axis] = parsed
    return result


def read_fold_rows(path: Path, fold: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            need(bool(line.strip()), f"blank OOF row in fold {fold}")
            raw = json.loads(line)
            need(isinstance(raw, dict) and set(raw) == ROW_FIELDS,
                 f"OOF row schema differs in fold {fold}")
            need(isinstance(raw["source_id"], str) and raw["source_id"].strip(),
                 "OOF source ID differs")
            need(raw["fold"] == fold, "OOF row fold differs")
            continuous = _axis_mapping(
                raw["continuous_prediction"], "continuous prediction", integer=False
            )
            integer = _axis_mapping(
                raw["half_up_integer_prediction"], "integer prediction", integer=True
            )
            reference = _axis_mapping(raw["reference_score"], "reference score", integer=False)
            need(all(integer[axis] == integerize_score(float(continuous[axis])) for axis in AXES),
                 "stored OOF integer is not exact half-up projection")
            rows.append({
                "source_id": raw["source_id"],
                "fold": fold,
                "continuous_prediction": continuous,
                "half_up_integer_prediction": integer,
                "reference_score": reference,
            })
    need(len(rows) == ROWS_PER_FOLD, f"fold {fold} row count differs")
    need(len({row["source_id"] for row in rows}) == ROWS_PER_FOLD,
         f"fold {fold} has duplicate source IDs")
    return rows


def _expected_checkpoint_paths(fold_output: Path, epoch: int) -> tuple[Path, Path]:
    root = fold_output / "epoch_checkpoints" / f"epoch-{epoch:02d}"
    return root / "trainable_model.safetensors", root / "checkpoint_metadata.json"


def verify_fold(
    run_id: str, fold: int, output_root: Path, restricted_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_output = output_root / run_id / f"fold-{fold:02d}"
    fold_restricted = restricted_root / run_id / f"fold-{fold:02d}"
    result_path = fold_output / "result.json"
    result = read_json(result_path)
    need(result.get("schema_version") == "mal2026-r0-exact-oof-fold-result-v1" and
         result.get("status") == "completed" and result.get("run_id") == run_id and
         result.get("fold") == fold and result.get("folds") == FOLDS,
         f"fold {fold} result identity differs")
    need(result.get("score_fields") == list(AXES) and
         result.get("average_target_used") is False and
         result.get("validation_rows_loaded") is False and
         result.get("validation_rows_directly_scored") is False,
         f"fold {fold} target or validation contract differs")
    need(result.get("train_records") == 1600 and result.get("heldout_records") == ROWS_PER_FOLD,
         f"fold {fold} population differs")
    gate = result.get("leakage_gate")
    need(isinstance(gate, dict) and gate.get("source_id_disjoint") is True and
         gate.get("document_id_disjoint") is True and
         gate.get("complete_train_2000_coverage") is True,
         f"fold {fold} leakage gate differs")
    ensemble = result.get("ensemble")
    need(isinstance(ensemble, dict) and ensemble.get("epochs") == list(EPOCHS) and
         ensemble.get("predictions_per_heldout_row") == len(EPOCHS),
         f"fold {fold} ensemble contract differs")

    artifacts = result.get("artifacts")
    need(isinstance(artifacts, dict), f"fold {fold} artifact inventory differs")
    prediction_path = fold_restricted / "oof_predictions.jsonl"
    aggregate_path = fold_output / "aggregate_metrics.json"
    need(Path(str(artifacts.get("restricted_oof_path", ""))).resolve() == prediction_path.resolve() and
         artifacts.get("restricted_oof_sha256") == file_sha256(prediction_path),
         f"fold {fold} restricted prediction checksum differs")
    need(Path(str(artifacts.get("aggregate_path", ""))).resolve() == aggregate_path.resolve() and
         artifacts.get("aggregate_sha256") == file_sha256(aggregate_path),
         f"fold {fold} aggregate checksum differs")
    aggregate = read_json(aggregate_path)
    need(aggregate.get("schema_version") == "mal2026-r0-exact-oof-fold-aggregate-v1" and
         aggregate.get("fold") == fold and aggregate.get("folds") == FOLDS and
         aggregate.get("heldout_records") == ROWS_PER_FOLD and
         aggregate.get("average_target_used") is False and
         aggregate.get("validation_rows_loaded") is False,
         f"fold {fold} aggregate contract differs")

    provenance = result.get("provenance")
    need(isinstance(provenance, dict) and
         provenance.get("canonical_train_sha256") == SOURCE_SHA256["train"] and
         isinstance(provenance.get("rationale_generation_sha256"), str) and
         isinstance(provenance.get("script_sha256"), str) and
         isinstance(provenance.get("git_sha"), str),
         f"fold {fold} provenance differs")

    checkpoints = result.get("checkpoints")
    need(isinstance(checkpoints, list) and
         [item.get("epoch") for item in checkpoints if isinstance(item, dict)] == list(EPOCHS),
         f"fold {fold} checkpoint sequence differs")
    checkpoint_bindings: list[dict[str, Any]] = []
    for item, epoch in zip(checkpoints, EPOCHS, strict=True):
        need(isinstance(item, dict), f"fold {fold} checkpoint entry differs")
        need(item.get("global_step") == EXPECTED_GLOBAL_STEPS[epoch],
             f"fold {fold} epoch {epoch} global step differs")
        state_path, metadata_path = _expected_checkpoint_paths(fold_output, epoch)
        need(Path(str(item.get("trainable_state_path", ""))).resolve() == state_path.resolve() and
             item.get("trainable_state_sha256") == file_sha256(state_path),
             f"fold {fold} epoch {epoch} state checksum differs")
        need(item.get("checkpoint_metadata_sha256") == file_sha256(metadata_path),
             f"fold {fold} epoch {epoch} metadata checksum differs")
        metadata = read_json(metadata_path)
        need(metadata.get("schema_version") == "mal2026-r0-exact-oof-checkpoint-v1" and
             metadata.get("run_id") == run_id and metadata.get("fold") == fold and
             metadata.get("epoch") == epoch and metadata.get("average_target_used") is False and
             metadata.get("global_step") == EXPECTED_GLOBAL_STEPS[epoch] and
             metadata.get("source_train_sha256") == SOURCE_SHA256["train"] and
             metadata.get("rationale_sha256") == provenance.get("rationale_generation_sha256") and
             metadata.get("fold_assignment_fingerprint") == result.get("fold_assignment_fingerprint") and
             metadata.get("trainable_state_sha256") == item.get("trainable_state_sha256"),
             f"fold {fold} epoch {epoch} checkpoint provenance differs")
        checkpoint_bindings.append({
            "epoch": epoch,
            "state_sha256": item["trainable_state_sha256"],
            "metadata_sha256": item["checkpoint_metadata_sha256"],
        })

    rows = read_fold_rows(prediction_path, fold)
    binding = {
        "fold": fold,
        "result_sha256": file_sha256(result_path),
        "aggregate_sha256": artifacts["aggregate_sha256"],
        "prediction_sha256": artifacts["restricted_oof_sha256"],
        "fold_assignment_fingerprint": result["fold_assignment_fingerprint"],
        "provenance": provenance,
        "model_id": result.get("model_id"),
        "model_revision": result.get("model_revision"),
        "rationale_source": result.get("rationale_source"),
        "initialization": result.get("initialization"),
        "checkpoints": checkpoint_bindings,
    }
    return rows, binding


def collect_verified_rows(
    run_id: str,
    expected_source_ids: Sequence[str],
    *,
    output_root: Path = OUTPUT_ROOT,
    restricted_root: Path = RESTRICTED_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    need(RUN_ID_PATTERN.fullmatch(run_id) is not None, "run ID differs")
    need(len(expected_source_ids) == TOTAL_ROWS and len(set(expected_source_ids)) == TOTAL_ROWS,
         "expected source population differs")
    all_rows: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        rows, binding = verify_fold(run_id, fold, output_root, restricted_root)
        all_rows.extend(rows)
        bindings.append(binding)
    fingerprints = {binding["fold_assignment_fingerprint"] for binding in bindings}
    need(len(fingerprints) == 1, "fold assignment fingerprints differ")
    provenance_keys = (
        "provenance", "model_id", "model_revision", "rationale_source", "initialization"
    )
    need(all(
        all(binding[key] == bindings[0][key] for key in provenance_keys)
        for binding in bindings[1:]
    ), "fold model provenance differs")
    identifiers = [row["source_id"] for row in all_rows]
    need(len(identifiers) == TOTAL_ROWS and len(set(identifiers)) == TOTAL_ROWS,
         "OOF merge requires exactly 2,000 unique source IDs")
    expected = set(expected_source_ids)
    need(set(identifiers) == expected, "OOF source IDs differ from canonical train")
    by_id = {row["source_id"]: row for row in all_rows}
    ordered = [by_id[identifier] for identifier in expected_source_ids]
    return ordered, bindings, next(iter(fingerprints))


def public_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    need(len(rows) == TOTAL_ROWS, "aggregate row count differs")
    truth = [[float(row["reference_score"][axis]) for axis in AXES] for row in rows]
    continuous = [
        [float(row["continuous_prediction"][axis]) for axis in AXES] for row in rows
    ]
    integers = [
        [int(row["half_up_integer_prediction"][axis]) for axis in AXES] for row in rows
    ]
    return {
        "schema_version": "mal2026-r0-exact-oof-aggregate-v1",
        "folds": FOLDS,
        "records": TOTAL_ROWS,
        "unique_source_ids": TOTAL_ROWS,
        "predictions_per_source": 1,
        "continuous_metrics": three_axis_metrics(truth, continuous),
        "half_up_integer_metrics": three_axis_metrics(truth, integers),
        "average_target_used": False,
        "validation_rows_loaded": False,
    }


def write_json_fresh(path: Path, value: Mapping[str, Any], *, private: bool = False) -> str:
    need(not path.exists(), f"fresh output required: {path.name}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if private:
        os.chmod(path, 0o600)
    return file_sha256(path)


def write_jsonl_fresh(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    need(not path.exists(), "fresh merged OOF output required")
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return file_sha256(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    need(RUN_ID_PATTERN.fullmatch(args.run_id) is not None, "run ID differs")
    output = OUTPUT_ROOT / args.run_id / "merged"
    restricted = RESTRICTED_ROOT / args.run_id / "merged"
    need(not output.exists() and not restricted.exists(), "merged outputs must be fresh")
    rows, bindings, fingerprint = collect_verified_rows(
        args.run_id, canonical_source_ids()
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    restricted.mkdir(mode=0o700, parents=True, exist_ok=False)
    merged_path = restricted / "oof_predictions.jsonl"
    merged_sha = write_jsonl_fresh(merged_path, rows)
    aggregate = public_aggregate(rows)
    aggregate_path = output / "aggregate_metrics.json"
    aggregate_sha = write_json_fresh(aggregate_path, aggregate)
    contract = train_residual_contract()
    contract_path = output / "train_residual_contract.json"
    contract_sha = write_json_fresh(contract_path, contract)
    result = {
        "schema_version": "mal2026-r0-exact-oof-merge-result-v1",
        "status": "completed",
        "folds": FOLDS,
        "records": TOTAL_ROWS,
        "unique_source_ids": TOTAL_ROWS,
        "predictions_per_source": 1,
        "fold_assignment_fingerprint": fingerprint,
        "canonical_train_sha256": SOURCE_SHA256["train"],
        "merged_restricted_oof_sha256": merged_sha,
        "aggregate_sha256": aggregate_sha,
        "train_residual_contract_sha256": contract_sha,
        "base_prediction_origin_oof": 1,
        "fold_bindings": [{
            "fold": binding["fold"],
            "result_sha256": binding["result_sha256"],
            "aggregate_sha256": binding["aggregate_sha256"],
            "prediction_sha256": binding["prediction_sha256"],
            "checkpoint_state_sha256": [
                item["state_sha256"] for item in binding["checkpoints"]
            ],
            "checkpoint_metadata_sha256": [
                item["metadata_sha256"] for item in binding["checkpoints"]
            ],
        } for binding in bindings],
        "shared_provenance_sha256": sha256(json.dumps(
            bindings[0]["provenance"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "privacy_row_level_outputs_restricted": 1,
    }
    result_path = output / "result.json"
    result_sha = write_json_fresh(result_path, result)
    print(json.dumps({
        "status": "completed",
        "records": TOTAL_ROWS,
        "result_sha256": result_sha,
        "merged_restricted_oof_sha256": merged_sha,
        "aggregate_sha256": aggregate_sha,
        "train_residual_contract_sha256": contract_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
