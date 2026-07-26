#!/usr/bin/env python3
"""Train one Qwen3-Embedding three-axis rationale score regressor."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.rlaif_qwen3_embedding import Qwen3EmbeddingTrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_training(Qwen3EmbeddingTrainConfig.from_json(args.config))


if __name__ == "__main__":
    main()
