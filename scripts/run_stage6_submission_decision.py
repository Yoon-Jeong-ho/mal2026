#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mal2026.stage6_submission_decision import DecisionConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the preregistered Stage6 submission decision.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(DecisionConfig.from_json(args.config))
    print(json.dumps({key: result[key] for key in ("status", "submission_slots", "pending_deploy_artifacts")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
