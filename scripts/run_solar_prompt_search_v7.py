#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mal2026.solar_prompt_search_v7 import SearchConfigV7, aggregate_confirmation, aggregate_discovery, preflight, prepare, run_features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "preflight", "run-features", "aggregate-discovery", "aggregate-confirmation"), required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), default="discovery")
    args = parser.parse_args()
    config = SearchConfigV7.from_json(args.config)
    if args.stage == "prepare":
        result = prepare(config, args.config)
    elif args.stage == "preflight":
        result = preflight(config)
    elif args.stage == "run-features":
        result = run_features(config, args.split)
    elif args.stage == "aggregate-discovery":
        result = aggregate_discovery(config)
    else:
        result = aggregate_confirmation(config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
