#!/usr/bin/env python3
"""Evaluate one completed score-regression run on all frozen validation essays."""
from __future__ import annotations
import argparse
from pathlib import Path
from mal2026.api_score_regression import APIScoreRegressionEvalConfig, run_api_score_regression_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); args = parser.parse_args()
    run_api_score_regression_evaluation(APIScoreRegressionEvalConfig.from_json(args.config))


if __name__ == "__main__": main()
