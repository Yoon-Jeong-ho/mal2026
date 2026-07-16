#!/usr/bin/env python3
"""Train a standard Hugging Face Trainer encoder selection/refit run."""
from __future__ import annotations
import argparse
from pathlib import Path
from mal2026.standard_encoder_train import StandardEncoderConfig, run_standard_encoder

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=Path)
args = parser.parse_args()
run_standard_encoder(StandardEncoderConfig.from_json(args.config))
