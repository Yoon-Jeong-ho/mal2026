#!/usr/bin/env python3
"""Fail-closed preregistration validator for future train-only candidate use.

It intentionally cannot select candidates, construct SFT data, or access a
candidate payload.  It records whether a supplied aggregate judge-v3 manifest
has crossed the prerequisite gate for a separately authorized production
selection protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/openai_distribution_utilization.v1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--judge-v3-manifest", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "mal2026-openai-distribution-utilization-v1":
        raise SystemExit("invalid utilization preregistration schema")
    required = config.get("hard_preconditions", {})
    judge_passed = False
    if args.judge_v3_manifest is not None:
        manifest = json.loads(args.judge_v3_manifest.read_text(encoding="utf-8"))
        judge_passed = bool(manifest.get("pilot_passed_hard_gates") is True and str(manifest.get("schema_version", "")).startswith("qwen36-gguf-judge-v3"))
    payload = {
        "status": "ready_for_separate_production_selection_protocol" if judge_passed else "blocked_pending_judge_v3",
        "judge_v3_global_hard_gates_passed": judge_passed,
        "selection_performed": False,
        "sft_constructed": False,
        "validation_used": False,
        "separate_post_v3_production_selection_protocol_required": required.get("separate_post_v3_production_selection_protocol_required") is True,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
