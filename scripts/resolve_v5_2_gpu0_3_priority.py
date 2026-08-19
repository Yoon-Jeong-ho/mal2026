#!/usr/bin/env python3
"""Resolve v5.2 GPU ownership through the documented utilization plan.

Only a plan that explicitly declares a v5.2 priority job can be yielded.  The
current score_utilization workers are non-yieldable by design, so this resolver
returns the authorized GPU0-only fallback without signaling or touching them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = ROOT / "outputs/reservations/utilization-only-20260720-001"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--gpus", nargs="+", type=int, required=True); parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.gpus or args.gpus[0] != 0 or len(set(args.gpus)) != len(args.gpus) or any(gpu not in {0, 1, 2, 3} for gpu in args.gpus):
        raise SystemExit("requested topology must be an ordered subset of GPUs 0--3 beginning with GPU0")
    plan_path = args.runtime / "utilization_only_plan.json"; ledger_path = args.runtime / "utilization_only_ledger.json"
    result = {"schema_version": "mal2026-v5_2-priority-resolution-v1", "requested_physical_gpus": args.gpus, "selected_physical_gpus": args.gpus, "status": "all_requested_gpus_available_to_priority", "ownership_evidence": "documented utilization priority plan", "release_verified": False}
    if len(args.gpus) > 1:
        if not plan_path.is_file() or not ledger_path.is_file():
            result.update({"status": "fallback_gpu0_only", "selected_physical_gpus": [0], "reason": "utilization priority plan/ledger unavailable; no safe yield authority"})
        else:
            plan = json.loads(plan_path.read_text(encoding="utf-8")); entries = plan.get("gpus", {})
            non_yieldable = []
            for gpu in args.gpus[1:]:
                entry = entries.get(str(gpu), {}); priority = entry.get("priority", []); util = entry.get("utilization", {})
                if not any(job.get("kind") == "judge_v5_2_repeat" and job.get("run_purpose") == "higher_priority_research" for job in priority) or util.get("kind") != "backfill":
                    non_yieldable.append(gpu)
            if non_yieldable:
                result.update({"status": "fallback_gpu0_only", "selected_physical_gpus": [0], "reason": "documented utilization jobs are non-yieldable score_utilization; no SIGTERM or overwrite issued", "non_yieldable_gpus": non_yieldable})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
