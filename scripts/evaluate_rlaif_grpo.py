#!/usr/bin/env python3
"""Generate and score one post-RLAIF frozen-validation rationale system."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.rlaif_evaluation import RLAIFEvaluationConfig, generate_validation, judge_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "judge"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--server-attestation", required=True, type=Path)
    args = parser.parse_args()
    config = RLAIFEvaluationConfig.from_json(args.config)
    if args.command == "generate":
        generate_validation(config, args.endpoint, args.server_attestation)
    else:
        judge_validation(config, args.endpoint, args.server_attestation)


if __name__ == "__main__":
    main()
