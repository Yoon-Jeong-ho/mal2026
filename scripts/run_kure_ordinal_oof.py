#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mal2026.kure_ordinal_oof import KUREOrdinalOOFConfig, aggregate, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one exact train-only KURE ordinal outer fold.")
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--outer-fold", type=int, choices=range(5))
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Validate CPU contracts; never initialize KURE or CUDA.")
    parser.add_argument("--smoke", action="store_true", help="GPU0 gate: first method/content, two phase1 and one cRT steps.")
    args = parser.parse_args()
    config = KUREOrdinalOOFConfig.from_json(args.config, require_dependencies=not args.validate_only)
    if args.aggregate:
        if args.validate_only or args.smoke:
            parser.error("--aggregate cannot be combined with --validate-only or --smoke")
        result = aggregate(args.config)
    else:
        if args.smoke and args.outer_fold != 0:
            parser.error("--smoke requires --outer-fold 0")
        result = run(args.config, outer_fold=args.outer_fold, validate_only=args.validate_only, smoke=args.smoke)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
