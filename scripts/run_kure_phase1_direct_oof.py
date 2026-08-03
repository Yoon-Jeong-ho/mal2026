#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from mal2026.kure_phase1_direct_oof import (
    KUREPhase1DirectOOFConfig, _atomic_public_json, aggregate, run, scheduler_state_conflict,
    set_process_title, summarize_gpu_telemetry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved Stage3 CORAL phase-1 heads without training.")
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--outer-fold", type=int, choices=range(5))
    group.add_argument("--aggregate", action="store_true")
    group.add_argument("--check-authorization", action="store_true")
    group.add_argument("--telemetry-csv", type=Path)
    group.add_argument("--scheduler-state", type=Path)
    parser.add_argument("--scheduler-run-id")
    parser.add_argument("--telemetry-summary", type=Path)
    parser.add_argument("--selected-gpus")
    parser.add_argument("--minimum-samples", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.outer_fold is not None:
        set_process_title(f"cli:{'smoke' if args.smoke else 'oof'}:f{args.outer_fold}")
    elif args.aggregate:
        set_process_title("cli:aggregate")
    elif args.telemetry_csv:
        set_process_title("cli:telemetry-summary")
    elif args.scheduler_state:
        set_process_title("cli:scheduler-gate")
    else:
        set_process_title("cli:authorization-gate")
    config = KUREPhase1DirectOOFConfig.from_json(args.config)
    if args.scheduler_state:
        if not args.selected_gpus or not args.scheduler_run_id or args.validate_only or args.smoke:
            parser.error("scheduler validation requires --selected-gpus and --scheduler-run-id")
        selected = tuple(int(item) for item in args.selected_gpus.split(","))
        if args.scheduler_state.exists():
            try: state = json.loads(args.scheduler_state.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc: raise SystemExit("active scheduler state is unreadable") from exc
            reason = scheduler_state_conflict(
                state, selected, age_seconds=time.time() - args.scheduler_state.stat().st_mtime,
                expected_run_id=args.scheduler_run_id,
            )
            if reason: raise SystemExit(reason)
        result = {"status": "scheduler_safe", "selected_gpus": selected}
    elif args.telemetry_csv:
        if not args.telemetry_summary or not args.selected_gpus or args.minimum_samples is None or args.validate_only or args.smoke:
            parser.error("telemetry summarization requires --telemetry-summary, --selected-gpus, and --minimum-samples")
        selected = tuple(int(item) for item in args.selected_gpus.split(","))
        result = summarize_gpu_telemetry(args.telemetry_csv, selected, args.minimum_samples)
        _atomic_public_json(args.telemetry_summary, result)
    elif args.check_authorization:
        if args.validate_only or args.smoke:
            parser.error("--check-authorization cannot be combined with --validate-only or --smoke")
        config.require_execution_authorization()
        result = {"status": "authorized", "task_card_sha256": config.task_card_sha256}
    elif args.aggregate:
        if args.validate_only or args.smoke:
            parser.error("--aggregate cannot be combined with --validate-only or --smoke")
        result = aggregate(config)
    else:
        if args.smoke and args.outer_fold != 0:
            parser.error("--smoke requires --outer-fold 0")
        result = run(config, outer_fold=args.outer_fold, validate_only=args.validate_only, smoke=args.smoke)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
