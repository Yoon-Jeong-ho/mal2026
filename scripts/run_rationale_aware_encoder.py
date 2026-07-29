#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.rationale_aware_encoder import RationaleEncoderConfig, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(RationaleEncoderConfig.from_json(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
