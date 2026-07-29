#!/usr/bin/env python3
"""Run final selected rationale-aware encoder on train plus validation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.final_rationale_encoder import FinalEncoderConfig, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final_rationale_aware_score_encoder.v1.json")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    config = FinalEncoderConfig.from_json(args.config, require_dependencies=True)
    result = run(config, smoke=args.mode == "smoke")
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({
            "status": result["status"], "mode": result["mode"],
            "run_id": config.run_id, "selected": result["selected"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
