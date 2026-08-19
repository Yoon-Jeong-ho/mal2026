#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.qwen_rationale_oof import (  # noqa: E402
    Config, aggregate_phase, audit_aihub_tail, calibrate, final_report,
    preflight, refit_and_validate, run_fold,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the staged Qwen rationale-aware exact OOF experiment.")
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--preflight", action="store_true")
    commands.add_argument("--fold", type=int, choices=range(5))
    commands.add_argument("--aggregate", choices=("stage1", "stage2", "stage4"))
    commands.add_argument("--calibrate", choices=("stage2", "stage4"))
    commands.add_argument("--audit-aihub-tail", action="store_true")
    commands.add_argument("--refit-validate", action="store_true")
    commands.add_argument("--final-report", action="store_true")
    parser.add_argument("--phase", choices=("stage1", "stage2", "stage4"))
    parser.add_argument("--arm")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = Config.from_json(args.config)
    if args.preflight:
        result = preflight(config)
    elif args.fold is not None:
        if not args.phase or not args.arm:
            parser.error("--fold requires --phase and --arm")
        if args.smoke and (args.phase, args.fold) != ("stage1", 0):
            parser.error("smoke is fixed to stage1 fold 0")
        result = run_fold(config, args.phase, args.arm, args.fold, smoke=args.smoke)
    elif args.aggregate:
        result = aggregate_phase(config, args.aggregate)
    elif args.calibrate:
        result = calibrate(config, args.calibrate)
    elif args.audit_aihub_tail:
        result = audit_aihub_tail(config)
    elif args.refit_validate:
        result = refit_and_validate(config)
    else:
        result = final_report(config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
