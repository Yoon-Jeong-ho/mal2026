#!/usr/bin/env python3
"""Train the Qwen3-Embedding warm-start and save every epoch state."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.rlaif_qwen3_epoch_sweep import EpochSweepTrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_training(EpochSweepTrainConfig.from_json(args.config))


if __name__ == "__main__":
    main()
