#!/usr/bin/env python3
"""Call a declared local vLLM decoder endpoint for rationale-only generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.api_rationale_generation import APIRationaleGenerationConfig, run_api_rationale_generation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--server-attestation", required=True, type=Path)
    args = parser.parse_args()
    run_api_rationale_generation(APIRationaleGenerationConfig.from_json(args.config), args.endpoint, args.server_attestation)


if __name__ == "__main__":
    main()
