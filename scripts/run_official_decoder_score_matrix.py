#!/usr/bin/env python3
"""Run, smoke, or dry-check official decoder integer-score arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mal2026.official_decoder_score import (
    DecoderScoreConfig, arm_names, experiment_contract, parse_arm, run_target_arm,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=arm_names())
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-arms", action="store_true")
    args = parser.parse_args()
    if not (args.dry_run or args.list_arms or args.arm):
        parser.error("choose --arm, --dry-run, or --list-arms")
    # Phase-specific runners validate only the dependencies they are allowed
    # to access (notably, AI-Hub pretraining never touches target validation).
    config = DecoderScoreConfig.from_json(args.config, require_dependencies=False)
    payload = experiment_contract(config)
    payload.update({"status": "dry_run_passed", "gpu_started": False})
    if args.arm:
        architecture, initialization, input_view = parse_arm(args.arm)
        payload["requested_arm"] = {"name": args.arm, "architecture": architecture, "initialization": initialization, "input_view": input_view}
        payload["unresolved_dependencies"] = [
            *( ["selected rationale train/validation paths and checksums"] if input_view == "rationale" else []),
            *( [f"matched AI-Hub {architecture} completion/state checksums"] if initialization == "aihub_matched" else []),
        ]
    if args.dry_run or args.list_arms:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        assert args.arm is not None
        run_target_arm(config, args.arm, smoke=args.smoke)


if __name__ == "__main__":
    main()
