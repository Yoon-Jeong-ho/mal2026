#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.evaluation_prompt_rationale_sft import EvaluationPromptRationaleSFTConfig, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(EvaluationPromptRationaleSFTConfig.from_json(args.config), smoke=args.smoke)
    print(json.dumps({key: result[key] for key in ("status", "mode", "run_id", "prompt_kind", "global_step")}, sort_keys=True))


if __name__ == "__main__":
    main()
