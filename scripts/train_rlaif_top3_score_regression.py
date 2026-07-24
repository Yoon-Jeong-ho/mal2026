#!/usr/bin/env python3
"""Train one three-axis Qwen2.5 encoder for one RLAIF rationale source."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.rlaif_top3_encoder import RLAIFTop3RegressionConfig, run_score_regression


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_score_regression(RLAIFTop3RegressionConfig.from_json(args.config))


if __name__ == "__main__":
    main()
