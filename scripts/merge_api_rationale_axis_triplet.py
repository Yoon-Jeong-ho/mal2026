#!/usr/bin/env python3
"""Merge three restricted single-axis generation artifacts for exact v6 judging."""
from __future__ import annotations
import argparse
from pathlib import Path
from mal2026.api_rationale_merge import APIRationaleMergeConfig, run_api_rationale_merge


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True,type=Path);args=parser.parse_args()
    run_api_rationale_merge(APIRationaleMergeConfig.from_json(args.config))


if __name__=="__main__":main()
