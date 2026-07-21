#!/usr/bin/env python3
"""Train one declared score-regression condition with Transformers Trainer."""
from __future__ import annotations
import argparse
from pathlib import Path
from mal2026.api_score_regression import APIScoreRegressionConfig, run_api_score_regression


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); args = parser.parse_args()
    run_api_score_regression(APIScoreRegressionConfig.from_json(args.config))


if __name__ == "__main__": main()
