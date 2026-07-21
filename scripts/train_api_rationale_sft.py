#!/usr/bin/env python3
"""Train one API-rationale decoder task with maintained TRL SFTTrainer."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.api_rationale_sft import APIRationaleSFTConfig, run_api_rationale_sft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_api_rationale_sft(APIRationaleSFTConfig.from_json(args.config))


if __name__ == "__main__":
    main()
