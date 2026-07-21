#!/usr/bin/env python3
"""Apply the fixed v6 Qwen pointwise judge to generated rationale artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.api_rationale_judge import APIRationaleJudgeConfig, run_api_rationale_judge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--server-attestation", required=True, type=Path)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    run_api_rationale_judge(APIRationaleJudgeConfig.from_json(args.config), args.endpoint, args.server_attestation, args.model)


if __name__ == "__main__":
    main()
