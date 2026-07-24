#!/usr/bin/env python3
"""Evaluate one completed top-three encoder on frozen validation writings."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.rlaif_top3_encoder import RLAIFTop3RegressionEvalConfig, run_score_regression_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_score_regression_evaluation(RLAIFTop3RegressionEvalConfig.from_json(args.config))


if __name__ == "__main__":
    main()
