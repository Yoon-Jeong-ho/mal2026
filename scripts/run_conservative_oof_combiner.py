#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mal2026.conservative_oof_combiner import CombinerConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or validate the validation-free conservative OOF combiner.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run(CombinerConfig.from_json(args.config), validate_only=args.validate_only)
    print(json.dumps({key: result.get(key) for key in ("status", "records", "protected_output", "validation_rows_loaded")}, sort_keys=True))


if __name__ == "__main__":
    main()
