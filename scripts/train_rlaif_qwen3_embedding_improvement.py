#!/usr/bin/env python3
"""Train one frozen Qwen3-Embedding improvement arm."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.rlaif_qwen3_improvement import ImprovementTrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_training(ImprovementTrainConfig.from_json(args.config))


if __name__ == "__main__":
    main()

