#!/usr/bin/env python3
"""Run one train-only NPCR outer fold or aggregate completed folds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mal2026.prompt_reference_npcr import NPCRConfig, load_rows, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("outer_fold", "full"))
    parser.add_argument("--outer-fold", type=int, choices=range(5))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = NPCRConfig.from_json(args.config)
    if args.validate_only:
        rows = load_rows(config)
        print(json.dumps({"status": "validated", "run_id": config.run_id, "records": len(rows),
                          "validation_rows_loaded": False, "average_target_used": False, "r0_score_feature_used": False}, sort_keys=True))
        return
    os.environ["MAL2026_NPCR_COMMAND"] = " ".join(os.sys.argv)
    result = run(config, mode=args.mode, outer_fold=args.outer_fold, device=args.device)
    print(json.dumps({key: result.get(key) for key in ("status", "mode", "run_id", "outer_fold", "selected_candidate")}, sort_keys=True))


if __name__ == "__main__":
    main()
