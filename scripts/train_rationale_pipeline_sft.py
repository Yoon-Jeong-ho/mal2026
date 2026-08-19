#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.rationale_pipeline_sft import RationalePipelineSFTConfig, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = RationalePipelineSFTConfig.from_json(args.config)
    setproctitle(f"mal2026:rationale-pipeline-sft:{config.stage}:rank{os.environ.get('LOCAL_RANK', '0')}"[:255])
    result = run(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({key: result[key] for key in ("status", "run_id", "stage", "global_step", "train_records")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
