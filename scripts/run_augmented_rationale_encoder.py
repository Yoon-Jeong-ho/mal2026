#!/usr/bin/env python3
"""Run one source-disjoint Solar-augmented rationale-aware encoder arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.augmented_rationale_encoder import AugmentedEncoderConfig, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    config = AugmentedEncoderConfig.from_json(args.config, require_dependencies=True)
    result = run(config, smoke=args.mode == "smoke")
    if int(__import__("os").environ.get("RANK", "0")) == 0:
        print(json.dumps({
            "status": result["status"], "mode": result["mode"],
            "run_id": result.get("run_id", config.run_id), "model_key": config.model_key,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
