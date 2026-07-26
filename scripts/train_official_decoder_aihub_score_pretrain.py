#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from mal2026.official_decoder_aihub_pretrain import ARCHITECTURES, DecoderAIHubConfig, run_training

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--phase", choices=("selection", "refit"), required=True)
    parser.add_argument("--selection-metadata", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = DecoderAIHubConfig.from_json(args.config, require_dependencies=not args.dry_run)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "gpu_started": False, "architecture": args.architecture, "phase": args.phase, "training_method": "full_parameter", "canonical_validation_access": False}, sort_keys=True))
        return
    run_training(config, args.architecture, args.phase, smoke=args.smoke, selection_metadata=args.selection_metadata)

if __name__ == "__main__": main()
