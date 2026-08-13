#!/usr/bin/env python3
"""Generate the public aggregate profile from canonical train only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.train_score_band_profile import EXPECTED_RECORDS, build_profile, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "eval/train.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/reports/train_score_band_profile_v1.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical_train = (ROOT / "eval/train.jsonl").resolve()
    train = args.train.resolve()
    if train != canonical_train or not train.is_file() or train.is_symlink():
        raise RuntimeError("input must be the canonical regular file eval/train.jsonl")
    records = []
    with train.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                records.append(json.loads(line))
            if len(records) > EXPECTED_RECORDS:
                raise RuntimeError("canonical train contains more than 2,000 records")
    profile = build_profile(records, source_sha256=sha256_file(train))
    output = args.output.resolve()
    expected_output = (ROOT / "data/reports/train_score_band_profile_v1.json").resolve()
    if output != expected_output:
        raise RuntimeError("output must be data/reports/train_score_band_profile_v1.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "records": len(records), "source_sha256": profile["source"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
