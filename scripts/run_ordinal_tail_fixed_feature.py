#!/usr/bin/env python3
"""Run one immutable stage of the ordinal-tail frozen-feature screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mal2026.ordinal_tail_fixed_feature import FixedFeatureConfig, run


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/ordinal_tail_program.v1.json"),
    )
    parser.add_argument("--mode", required=True, choices=("smoke", "outer_fold", "full"))
    parser.add_argument("--outer-fold", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FixedFeatureConfig.from_json(args.config, root=ROOT)
    result = run(config, mode=args.mode, outer_fold=args.outer_fold)
    summary = {
        "status": result["status"],
        "mode": result["mode"],
        "run_id": result["run_id"],
        "outer_fold": result.get("outer_fold"),
        "selected_candidate": result.get("selected_candidate"),
        "candidate_macro_rmse": result.get("candidate_metrics", result.get("outer_metrics", {}))
        .get("macro", {})
        .get("rmse"),
        "protected_output": result.get("protected_output"),
        "protected_output_manifest_sha256": result.get("protected_output_manifest_sha256"),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
