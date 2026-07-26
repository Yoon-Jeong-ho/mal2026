#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_aihub_continuation_sft import OfficialAIHubContinuationConfig, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(OfficialAIHubContinuationConfig.from_json(args.config))
    print(json.dumps({"status": result["status"], "run_id": result["run_id"], "structure": result["structure"], "task": result["task"], "global_step": result["global_step"]}, sort_keys=True))


if __name__ == "__main__":
    main()
