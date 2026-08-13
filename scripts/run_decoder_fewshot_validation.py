#!/usr/bin/env python3
"""Prepare, execute, or aggregate the validation-only decoder few-shot matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.decoder_fewshot_validation import (  # noqa: E402
    FewshotConfig,
    aggregate_models,
    finalize_partial_length_retry,
    prepare_protocol,
    retry_length_failures,
    run_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "model", "retry-length", "finalize-partial-retry", "aggregate"), required=True)
    parser.add_argument("--model-key")
    args = parser.parse_args()
    config = FewshotConfig.from_json(args.config)
    if args.stage == "prepare":
        if args.model_key is not None:
            parser.error("prepare does not accept --model-key")
        result = prepare_protocol(config, args.config)
    elif args.stage in {"model", "retry-length", "finalize-partial-retry"}:
        if args.model_key is None:
            parser.error(f"{args.stage} requires --model-key")
        if args.stage == "model":
            result = run_model(config, args.config, args.model_key)
        elif args.stage == "retry-length":
            result = retry_length_failures(config, args.config, args.model_key)
        else:
            result = finalize_partial_length_retry(config, args.config, args.model_key)
    else:
        if args.model_key is not None:
            parser.error("aggregate does not accept --model-key")
        result = aggregate_models(config, args.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
