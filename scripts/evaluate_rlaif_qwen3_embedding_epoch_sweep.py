#!/usr/bin/env python3
"""Evaluate all saved Qwen3-Embedding epoch states."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.rlaif_qwen3_epoch_sweep import EpochSweepEvalConfig, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_evaluation(EpochSweepEvalConfig.from_json(args.config))


if __name__ == "__main__":
    main()
