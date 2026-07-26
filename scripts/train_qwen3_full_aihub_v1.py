#!/usr/bin/env python3
"""Run one native-FSDP full-parameter AI-Hub phase."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.qwen3_full_aihub_then_lora import FullTrainConfig, run_full_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_full_training(FullTrainConfig.from_json(args.config))


if __name__ == "__main__":
    main()

