#!/usr/bin/env python3
"""Generate one private, independent top-three RLAIF rationale source."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.rlaif_top3_encoder import RLAIFTop3GenerationConfig, run_rationale_generation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--server-attestation", required=True, type=Path)
    args = parser.parse_args()
    run_rationale_generation(RLAIFTop3GenerationConfig.from_json(args.config), args.endpoint, args.server_attestation)


if __name__ == "__main__":
    main()
