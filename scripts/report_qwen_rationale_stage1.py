#!/usr/bin/env python3
"""Compare a completed Stage1 exact OOF ablation with the frozen R0 OOF."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.qwen_rationale_oof import AXES, Config  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "outputs/r0-exact-oof-v1/r0-exact-oof-20260731-002/merged/aggregate_metrics.json",
    )
    args = parser.parse_args()
    config = Config.from_json(args.config, verify_validation_hash=False)
    aggregate_path = config.output_root / "stage1" / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if aggregate.get("status") != "completed" or aggregate.get("records") != 2000:
        raise SystemExit("Stage1 exact OOF aggregate is incomplete")
    if baseline.get("records") != 2000 or baseline.get("folds") != 5:
        raise SystemExit("R0 exact OOF baseline is incompatible")

    baseline_continuous = float(baseline["continuous_metrics"]["three_axis_macro_rmse"])
    baseline_integer = float(baseline["half_up_integer_metrics"]["three_axis_macro_rmse"])
    arms: dict[str, object] = {}
    for arm, summary in aggregate["arms"].items():
        metrics = summary["metrics"]
        continuous = float(metrics["macro_continuous_rmse_raw_decimal_gold"])
        integer = float(metrics["macro_integer_rmse"])
        axis_tail_recall = {
            axis: {
                score: metrics["axes"][axis]["per_gold_recall"][score]
                for score in ("1", "2", "5")
            }
            for axis in AXES
        }
        arms[arm] = {
            "input_variant": summary["input_variant"],
            "macro_continuous_rmse_raw_decimal_gold": continuous,
            "macro_integer_rmse": integer,
            "macro_integer_spearman": float(metrics["macro_integer_spearman"]),
            "macro_tail_1_2_5_rmse": float(metrics["macro_tail_rmse"]),
            "delta_continuous_rmse_vs_r0": continuous - baseline_continuous,
            "delta_integer_rmse_vs_r0": integer - baseline_integer,
            "tail_recall_by_axis": axis_tail_recall,
        }
    best_continuous = min(arms, key=lambda arm: (arms[arm]["macro_continuous_rmse_raw_decimal_gold"], arm))
    best_integer = min(arms, key=lambda arm: (arms[arm]["macro_integer_rmse"], arm))
    report = {
        "schema_version": "mal2026-qwen-rationale-oof-stage1-report-v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "run_id": config["run_id"],
        "gpu_scope": config["gpu_scope"],
        "records": 2000,
        "folds": 5,
        "selection_data": "train_only_exact_oof",
        "validation_access": False,
        "average_target_used": False,
        "r0_exact_oof_baseline": {
            "run_id": "r0-exact-oof-20260731-002",
            "macro_continuous_rmse_raw_decimal_gold": baseline_continuous,
            "macro_half_up_integer_rmse": baseline_integer,
        },
        "arms": arms,
        "best_continuous_arm": best_continuous,
        "best_integer_arm": best_integer,
        "target_rmse": float(config["target_rmse"]),
        "target_reached": bool(arms[best_continuous]["macro_continuous_rmse_raw_decimal_gold"] <= float(config["target_rmse"])),
    }
    for arm in arms.values():
        if not all(math.isfinite(float(arm[key])) for key in (
            "macro_continuous_rmse_raw_decimal_gold", "macro_integer_rmse", "macro_integer_spearman", "macro_tail_1_2_5_rmse"
        )):
            raise SystemExit("non-finite Stage1 metric")
    output = config.output_root / "stage1" / "report.json"
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
