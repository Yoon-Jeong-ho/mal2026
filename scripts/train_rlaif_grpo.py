#!/usr/bin/env python3
"""Launch one declared train-only RLAIF/GRPO continuation."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.rlaif_grpo import RLAIFRunConfig, run_rlaif_grpo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_rlaif_grpo(RLAIFRunConfig.from_json(args.config))


if __name__ == "__main__":
    main()
