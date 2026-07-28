#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.official_kure_score import KUREScoreConfig, run_arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--view", choices=("essay", "rationale"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_arm(KUREScoreConfig.from_json(args.config, require_rationales=args.view == "rationale"), args.view, smoke=args.smoke)


if __name__ == "__main__":
    main()
