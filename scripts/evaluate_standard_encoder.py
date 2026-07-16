#!/usr/bin/env python3
"""Evaluate a standard Trainer encoder state without persisting row outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
from mal2026.standard_encoder_eval import StandardEncoderEvalConfig, run_standard_encoder_evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=Path)
args = parser.parse_args()
run_standard_encoder_evaluation(StandardEncoderEvalConfig.from_json(args.config))
