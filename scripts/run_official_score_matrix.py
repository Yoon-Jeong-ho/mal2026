#!/usr/bin/env python3
"""Run or dry-check one official Qwen3-Embedding score-matrix arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mal2026.official_score_matrix import MatrixConfig, arm_names, run_arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=arm_names(), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = MatrixConfig.from_json(args.config, require_dependencies=False)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "arm": args.arm, "smoke": args.smoke, "dependencies_checked": False, "gpu_started": False}, sort_keys=True))
        return
    run_arm(config, args.arm, smoke=args.smoke)


if __name__ == "__main__":
    main()
