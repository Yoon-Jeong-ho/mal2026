#!/usr/bin/env python3
"""Prepare or run the validation-only human-audited decoder prompt ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.decoder_human_audited_prompt_validation import Config, prepare, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "run"), required=True)
    args = parser.parse_args()
    config = Config.from_json(args.config)
    result = prepare(config, args.config) if args.stage == "prepare" else run(config, args.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
