#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from mal2026.solar_prompt_search_v3 import SearchConfigV3, aggregate_discovery, preflight, prepare, run_candidate

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "preflight", "run", "aggregate-discovery"), required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--split", choices=("discovery", "confirmation"), default="discovery")
    args = parser.parse_args()
    config = SearchConfigV3.from_json(args.config)
    if args.stage == "prepare": result = prepare(config, args.config)
    elif args.stage == "preflight": result = preflight(config)
    elif args.stage == "run":
        if args.candidate is None: parser.error("--candidate required")
        result = run_candidate(config, args.candidate, args.split)
    else: result = aggregate_discovery(config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0

if __name__ == "__main__": raise SystemExit(main())
