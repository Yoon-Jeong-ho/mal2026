#!/usr/bin/env python3
"""Aggregate the six axis-wise KURE contrastive results without row data."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/kure-axis-ordinal-contrastive-v1"
ARMS = ("base", "aihub_full_backbone")
AXES = ("content", "organization", "expression")
METHODS = ("continuous_head", "prototype_soft", "hybrid", "center_0.5", "center_0.1", "cluster_k2")
FIELDS = (
    "continuous_rmse", "continuous_spearman", "integer_rmse", "integer_accuracy",
    "one_off_accuracy", "low_1_2_rmse", "score5_rmse", "gold34_balanced_accuracy",
    "gold3_to4_rate", "gold4_to3_rate",
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_id(arm: str, axis: str) -> str:
    tag = "aihub" if arm == "aihub_full_backbone" else "base"
    return f"kure-axis-contrastive-{tag}-{axis}-20260802-001"


def read_result(root: Path, arm: str, axis: str) -> dict[str, Any]:
    path = root / run_id(arm, axis) / "result.json"
    need(path.is_file(), f"missing result: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    need(value.get("status") == "completed" and value.get("mode") == "full", f"incomplete result: {path}")
    need((value.get("arm"), value.get("axis")) == (arm, axis), f"result identity differs: {path}")
    need(value.get("average_read") is False and value.get("average_target_used") is False, "average contract differs")
    return value


def selected_methods(result: Mapping[str, Any]) -> Mapping[str, Any]:
    epoch = int(result["selection"]["selected_epoch"])
    events = [event for event in result["selection"]["events"] if int(event["epoch"]) == epoch]
    need(len(events) == 1, "selected epoch event differs")
    return events[0]["evaluation"]["methods"]


def aggregate_methods(axis_methods: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS:
        per_axis = {axis: axis_methods[axis][method] for axis in AXES}
        macro = {field: sum(float(per_axis[axis][field]) for axis in AXES) / 3.0 for field in FIELDS}
        need(all(math.isfinite(value) for value in macro.values()), "aggregate metric is non-finite")
        result[method] = {"macro": macro, "axes": per_axis}
    return result


def aggregate(root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    source: dict[str, Any] = {}
    for arm in ARMS:
        values = {axis: read_result(root, arm, axis) for axis in AXES}
        selection = aggregate_methods({axis: selected_methods(values[axis]) for axis in AXES})
        validation = aggregate_methods({axis: values[axis]["canonical_validation"]["methods"] for axis in AXES})
        arms[arm] = {
            "selection_dev": selection, "canonical_validation_descriptive": validation,
            "selected_epochs": {axis: int(values[axis]["selection"]["selected_epoch"]) for axis in AXES},
            "primary_method": "hybrid",
        }
        source[arm] = {
            axis: {"path": str((root / run_id(arm, axis) / "result.json").resolve()),
                   "sha256": file_sha(root / run_id(arm, axis) / "result.json")}
            for axis in AXES
        }
    base = arms["base"]["selection_dev"]["hybrid"]["macro"]
    aihub = arms["aihub_full_backbone"]["selection_dev"]["hybrid"]["macro"]
    difference = {field: float(aihub[field]) - float(base[field]) for field in FIELDS}
    winner = min(ARMS, key=lambda arm: float(arms[arm]["selection_dev"]["hybrid"]["macro"]["continuous_rmse"]))
    payload = {
        "schema_version": "mal2026-kure-axis-ordinal-contrastive-aggregate-v1", "status": "completed",
        "run_id": "kure-axis-ordinal-contrastive-v1-20260802-001", "axes": list(AXES),
        "average_read": False, "average_target_used": False, "primary_method": "hybrid",
        "arms": arms, "selection_dev_aihub_minus_base": difference,
        "selection_dev_winner": winner,
        "historical_context": {
            "kure_direct_validation_macro_rmse": 0.641855939488718,
            "exact_r0_oof_macro_rmse_not_comparable_to_validation": 0.5687802169918456,
        },
        "sources": source,
        "interpretation": "train-internal selection and repeatedly exposed validation are descriptive, not independent confirmation",
    }
    destination = root / "aggregate.json"
    need(not destination.exists(), f"refusing to overwrite aggregate: {destination}")
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    aggregate(args.output_root)


if __name__ == "__main__":
    main()
