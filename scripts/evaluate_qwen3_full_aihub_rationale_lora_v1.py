#!/usr/bin/env python3
"""Evaluate all epochs of the full-AI-Hub then rationale-LoRA arm."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.qwen3_full_aihub_then_lora import FullRationaleConfig, run_rationale_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--essay-limit", type=int, required=True)
    parser.add_argument("--per-device-batch-size", type=int, required=True)
    args = parser.parse_args()
    run_rationale_evaluation(FullRationaleConfig.from_json(args.config), args.output.resolve(), args.essay_limit, args.per_device_batch_size)


if __name__ == "__main__":
    main()

