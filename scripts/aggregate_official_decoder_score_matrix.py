#!/usr/bin/env python3
"""Aggregate and rank all twelve official decoder score arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mal2026.official_decoder_score import DecoderScoreConfig, aggregate_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = DecoderScoreConfig.from_json(args.config, require_dependencies=False)
    payload = aggregate_results(config)
    output = args.output or Path(config.output_root) / "aggregate.json"
    if output.resolve().parent != Path(config.output_root).resolve():
        raise ValueError("aggregate output must remain under the ignored decoder output root")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(output.resolve()), "winner": payload["winner"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
