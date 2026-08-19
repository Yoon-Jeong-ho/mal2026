#!/usr/bin/env python3
"""Maintained Trainer runner for leakage-safe R0 ordinal residual training."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.r0_ordinal_residual import (  # noqa: E402
    ResidualRunConfig,
    run_residual_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a three-axis five-way residual classifier from frozen embeddings and leakage-safe R0 scores.",
        epilog=(
            "The config is the exact ResidualRunConfig JSON schema. Its train/validation manifests use "
            "EmbeddingArtifactManifest; JSONL rows contain source_id, group_id, shared_embedding, "
            "base_continuous_prediction, raw_continuous_gold, and oof_fold. Train requires exact "
            "folds 0..4; held-out validation requires null and is evaluated once after selection."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_residual_experiment(ResidualRunConfig.from_json(args.config))


if __name__ == "__main__":
    main()
