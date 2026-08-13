#!/usr/bin/env python3
"""Run Solar/OpenAI extensions of the fixed decoder few-shot validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.decoder_fewshot_external import (  # noqa: E402
    ExternalConfig,
    aggregate,
    api_finalize,
    api_poll,
    api_smoke,
    api_submit,
    prepare,
    repair_api_assistant_content,
    solar_run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "repair-api-requests", "api-smoke", "api-submit", "api-poll", "api-finalize", "solar-run", "aggregate"), required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    config = ExternalConfig.from_json(args.config)
    if args.stage.startswith("api-"):
        if args.model is None:
            parser.error(f"{args.stage} requires --model")
        function = {"api-smoke": api_smoke, "api-submit": api_submit, "api-poll": api_poll, "api-finalize": api_finalize}[args.stage]
        result = function(config, args.model)
    else:
        if args.model is not None:
            parser.error(f"{args.stage} does not accept --model")
        if args.stage == "prepare":
            result = prepare(config, args.config)
        elif args.stage == "repair-api-requests":
            result = repair_api_assistant_content(config)
        elif args.stage == "solar-run":
            result = solar_run(config)
        else:
            result = aggregate(config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
