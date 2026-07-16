#!/usr/bin/env python3
"""Launch direct or human-feedback Qwen SFT through TRL SFTTrainer only."""
from __future__ import annotations
import argparse
from pathlib import Path
from mal2026.standard_decoder_train import StandardSFTConfig, run_sft

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=Path)
args = parser.parse_args()
run_sft(StandardSFTConfig.from_json(args.config))
