#!/usr/bin/env python3
"""Continue the full AI-Hub Qwen3 model with rationale-stage LoRA."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.qwen3_full_aihub_then_lora import FullRationaleConfig, run_rationale_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_rationale_training(FullRationaleConfig.from_json(args.config))


if __name__ == "__main__":
    main()

