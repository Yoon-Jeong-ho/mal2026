#!/usr/bin/env python3
"""v5.2 GPU0--3 synthetic gate; it never addresses an unselected GPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/openai_explanation_repeat_distribution.v5_2.gpu0_3.json"
SPEC = importlib.util.spec_from_file_location("v5_wire", ROOT / "scripts/preflight_openai_repeat_v5_synthetic.py")
assert SPEC and SPEC.loader
WIRE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(WIRE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8")); runtime = cfg["runtime"]
    allowed = runtime["physical_gpus"]
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(gpu not in allowed for gpu in args.gpus):
        raise SystemExit("selected GPUs are outside the v5.2 GPU0--3 allowlist")
    ports = {gpu: int(runtime["ports"][str(gpu)]) for gpu in args.gpus}
    WIRE.SCHEMA = cfg["schema_version"]
    WIRE.GPUS = tuple(args.gpus)
    WIRE.PORTS = ports
    WIRE.PARALLEL = int(runtime["parallel_requests_per_server"])
    WIRE.CONTEXT_SIZE = int(runtime["context_size"])
    WIRE.MAX_TOKENS = int(cfg["request"]["max_tokens"])
    WIRE.SAFETY_MARGIN = int(runtime["context_safety_margin"])
    WIRE.RETRY_ATTEMPTS = int(cfg["retry"]["max_attempts"])
    report = WIRE.run(args.run_dir)
    report.update({"config_sha256": sha(CONFIG), "selected_physical_gpus": args.gpus,
                   "parallel_requests_per_server": WIRE.PARALLEL,
                   "slot_context_size": WIRE.CONTEXT_SIZE // WIRE.PARALLEL,
                   "gpu_ownership": "project-owned GPUs 0--3 only; GPUs 4--7 never queried or used"})
    (args.run_dir / "aggregate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
