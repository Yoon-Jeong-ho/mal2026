#!/usr/bin/env python3
"""Load the full Qwen3 regressor and run one GPU0 forward pass."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.qwen3_full_aihub_then_lora import run_gpu0_construction_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_gpu0_construction_gate(args.output.resolve())


if __name__ == "__main__":
    main()

