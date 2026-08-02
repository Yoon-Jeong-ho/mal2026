#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.kure_axis_contrastive import AxisContrastiveConfig, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(AxisContrastiveConfig.from_json(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
