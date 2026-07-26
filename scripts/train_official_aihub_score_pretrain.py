#!/usr/bin/env python3
"""Train one AI-Hub integer three-axis selection/refit stage."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.official_aihub_score_pretrain import PretrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--head", required=True, choices=("bounded_regression", "ordinal_cumulative"))
    parser.add_argument("--phase", required=True, choices=("selection", "refit"))
    parser.add_argument("--selection-metadata", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_training(
        PretrainConfig.from_json(args.config), args.head, args.phase,
        smoke=args.smoke, selection_metadata=args.selection_metadata,
    )


if __name__ == "__main__":
    main()
