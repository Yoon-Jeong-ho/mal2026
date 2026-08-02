"""Aggregate-only diagnostics for the train-only ordinal tail program."""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .iterative_tail_metrics import AXES, compute_iterative_tail_metrics, half_up_band
from .r0_ordinal_residual import load_embedding_artifact


class OrdinalTailDiagnosticError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise OrdinalTailDiagnosticError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OrdinalTailDiagnosticError(f"invalid JSONL at line {line_number}") from exc
            need(isinstance(value, Mapping), f"JSONL row {line_number} must be an object")
            yield value


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {name: float(np.quantile(values, quantile)) for name, quantile in (("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95))}


def _prediction_diagnostics(gold: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    gold_band = np.vectorize(half_up_band)(gold)
    pred_band = np.vectorize(half_up_band)(prediction)
    axes: dict[str, Any] = {}
    for column, axis in enumerate(AXES):
        actual, estimate = gold[:, column], prediction[:, column]
        actual_band, estimate_band = gold_band[:, column], pred_band[:, column]
        confusion = [[int(np.sum((actual_band == left) & (estimate_band == right))) for right in range(1, 6)] for left in range(1, 6)]
        per_gold: dict[str, Any] = {}
        for score in range(1, 6):
            mask = actual_band == score
            values = estimate[mask]
            per_gold[str(score)] = {
                "count": int(mask.sum()),
                "prediction_mean": float(values.mean()),
                "prediction_std": float(values.std()),
                "bias_vs_raw_gold": float(np.mean(values - actual[mask])),
                "bias_vs_integer_band": float(np.mean(values - score)),
                "quantiles": _quantiles(values),
                "central_3_4_capture_rate": float(np.mean(np.isin(estimate_band[mask], (3, 4)))),
            }
        axes[axis] = {
            "gold_band_histogram": {str(score): int(np.sum(actual_band == score)) for score in range(1, 6)},
            "prediction_band_histogram": {str(score): int(np.sum(estimate_band == score)) for score in range(1, 6)},
            "confusion_rows_gold_columns_prediction_1_to_5": confusion,
            "prediction_mean": float(estimate.mean()),
            "prediction_std": float(estimate.std()),
            "per_gold_band": per_gold,
        }
    return {"axes": axes}


def _raw_score_profile(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    prompts: dict[str, Any] = {}
    prompt_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        prompt_rows[str(row["prompt_num"])].append(row)
    for axis in AXES:
        values = np.asarray([float(row["score"][axis]) for row in records], dtype=float)
        axes[axis] = {
            "raw_minimum": float(values.min()),
            "raw_maximum": float(values.max()),
            "raw_mean": float(values.mean()),
            "raw_std": float(values.std()),
            "fractional_label_count": int(np.sum(values != np.floor(values))),
            "unique_raw_score_count": int(len(np.unique(values))),
            "band_counts": {str(score): int(np.sum(np.vectorize(half_up_band)(values) == score)) for score in range(1, 6)},
        }
    for prompt, rows in sorted(prompt_rows.items()):
        prompt_axis = {}
        for axis in AXES:
            bands = [half_up_band(float(row["score"][axis])) for row in rows]
            counts = Counter(bands)
            prompt_axis[axis] = {
                "band_counts": {str(score): counts.get(score, 0) for score in range(1, 6)},
                "adjacent_pair_capacity": {f"{score}_{score + 1}": counts.get(score, 0) * counts.get(score + 1, 0) for score in range(1, 5)},
                "far_pair_capacity": int(sum(counts.get(left, 0) * counts.get(right, 0) for left in range(1, 6) for right in range(left + 2, 6))),
            }
        prompts[prompt] = {"records": len(rows), "axes": prompt_axis}
    return {"records": len(records), "axes": axes, "prompts": prompts}


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict) and value.get("schema_version") == "mal2026-ordinal-tail-program-v1", "ordinal tail config schema differs")
    need(value.get("axes") == list(AXES), "ordinal tail axes differ")
    need(value.get("average_target_forbidden") is True, "average target must be forbidden")
    need(value.get("validation_selection_forbidden") is True, "validation selection must be forbidden")
    need(value.get("authorized_gpus") == [0, 1, 2, 3] and value.get("smoke_gpu") == 0, "GPU scope differs")
    need(value.get("outer_folds") == 5 and value.get("inner_folds") == 4, "nested fold contract differs")
    return value


def run_diagnostics(config_path: Path, *, root: Path = Path(".")) -> dict[str, Any]:
    config = load_config(config_path)
    train_path = root / config["train_path"]
    manifest_path = root / config["r0_embedding_manifest_path"]
    rows_path = root / config["r0_embedding_rows_path"]
    need(train_path.is_file() and file_sha256(train_path) == config["train_sha256"], "canonical train differs")
    need(rows_path.is_file() and file_sha256(rows_path) == config["r0_embedding_rows_sha256"], "R0 embedding rows differ")
    records = list(_jsonl(train_path))
    need(len(records) == 2000, "canonical train must contain 2,000 rows")
    required = {"id", "document_id", "prompt_num", "prompt", "essay", "score"}
    need(all(set(row) == required and set(row["score"]) == set(AXES) | {"average"} for row in records), "canonical train schema differs")
    manifest, embedding_rows = load_embedding_artifact(manifest_path, rows_path)
    need(len(embedding_rows) == len(records) == 2000 and manifest.split_role == "train", "R0 train artifact population differs")
    raw_gold = np.asarray([row.raw_labels for row in embedding_rows], dtype=float)
    base = np.asarray([row.base_predictions for row in embedding_rows], dtype=float)
    metrics = compute_iterative_tail_metrics(raw_gold, base)
    need(abs(float(metrics["macro"]["rmse"]) - float(config["baseline_macro_oof_rmse"])) < 1e-9, "R0 OOF baseline differs")
    fold_counts = Counter(row.oof_fold for row in embedding_rows)
    group_folds: dict[str, int] = {}
    for row in embedding_rows:
        prior = group_folds.setdefault(row.group_id, int(row.oof_fold))
        need(prior == row.oof_fold, "a document group spans OOF folds")
    result = {
        "schema_version": "mal2026-ordinal-tail-diagnostic-v1",
        "status": "completed",
        "run_id": config["run_id"],
        "selection_population": "canonical_train_only",
        "validation_rows_loaded": False,
        "average_target_used": False,
        "train_sha256": config["train_sha256"],
        "r0_embedding_rows_sha256": config["r0_embedding_rows_sha256"],
        "records": 2000,
        "document_groups": len(group_folds),
        "oof_fold_counts": {str(fold): fold_counts[fold] for fold in range(5)},
        "raw_score_profile": _raw_score_profile(records),
        "r0_oof_metrics": metrics,
        "r0_prediction_diagnostics": _prediction_diagnostics(raw_gold, base),
        "stretch_target": {
            "macro_oof_rmse": float(config["stretch_macro_oof_rmse"]),
            "baseline_macro_oof_rmse": float(config["baseline_macro_oof_rmse"]),
            "absolute_improvement_required": float(config["baseline_macro_oof_rmse"] - config["stretch_macro_oof_rmse"]),
            "relative_improvement_required": float(1.0 - config["stretch_macro_oof_rmse"] / config["baseline_macro_oof_rmse"]),
            "role": "stretch_not_a_promotion_threshold",
        },
        "privacy": "aggregate_only_no_essay_prompt_text_identifier_embedding_or_row_prediction_persisted",
    }
    output_root = root / config["output_root"]
    need(not (output_root / "diagnostic.json").exists(), "refusing to overwrite diagnostic output")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "diagnostic.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    task_card = {
        "schema_version": "mal2026-ordinal-tail-task-card-v1",
        "run_id": config["run_id"],
        "authorized_by": "user-approved end-to-end ordinal tail plan with RMSE approximately 0.4 stretch target",
        "resource_scope": "GPUs 0-3; GPU0 first for smoke",
        "selection": "five-outer-by-four-inner train-only nested OOF",
        "validation_selection": False,
        "average_target_used": False,
        "baseline": {"name": "exact R0 OOF", "macro_rmse": config["baseline_macro_oof_rmse"]},
        "completion_predicate": "diagnostic, fixed-feature screen, top-two end-to-end KURE/cRT, NPCR, OOF calibration/ensemble, frozen refit and descriptive validation all complete",
        "allowed_recovery": "bounded launcher, serialization, schema, memory, and environment integration repair without scientific-variable changes",
        "escalation": "GPU conflict, destructive overwrite, invalid metric/data, external cost, or unapproved scientific change",
    }
    (output_root / "task_card.json").write_text(json.dumps(task_card, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger = output_root / "ledger.jsonl"
    event = {
        "run_id": config["run_id"], "stage": "diagnostic", "event": "next_stage_complete",
        "failure_family": "none", "repair_iteration": 0, "evidence_ref": str((output_root / "diagnostic.json").resolve()),
        "command_ref": "scripts/run_ordinal_tail_diagnostics.py", "resource_scope": "none",
        "gpu_scope_authorization": "default GPUs 0-3; user explicitly approved this end-to-end plan",
        "decision": "continue",
    }
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return result
