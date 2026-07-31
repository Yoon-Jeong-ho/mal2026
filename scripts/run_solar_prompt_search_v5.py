#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mal2026.solar_prompt_search_v5 import SearchConfigV5, preflight, prepare, run_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "preflight", "run"), required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), default="discovery")
    args = parser.parse_args()
    config = SearchConfigV5.from_json(args.config)
    if args.stage == "prepare":
        result = prepare(config, args.config)
    elif args.stage == "preflight":
        result = preflight(config)
    else:
        result = run_split(config, args.split)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
