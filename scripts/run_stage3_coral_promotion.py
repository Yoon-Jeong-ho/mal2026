#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mal2026.stage3_coral_promotion import PromotionConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the preregistered Stage3 coral-natural promotion gate.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run(PromotionConfig.from_json(args.config), validate_only=args.validate_only)
    print(json.dumps({key: result.get(key) for key in
                      ("status", "records", "folds", "method", "eligible", "rps_eligible")}, sort_keys=True))


if __name__ == "__main__":
    main()
