#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.ordinal_tail_diagnostics import run_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ordinal_tail_program.v1.json"))
    args = parser.parse_args()
    result = run_diagnostics(args.config)
    print(result["run_id"], result["status"], result["r0_oof_metrics"]["macro"]["rmse"])


if __name__ == "__main__":
    main()
