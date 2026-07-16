#!/usr/bin/env python3
"""Select the vLLM macro-MAE winner among completed Trainer checkpoints."""
from __future__ import annotations
import argparse
from pathlib import Path
from mal2026.standard_decoder_selection import select_checkpoint
parser = argparse.ArgumentParser()
parser.add_argument("--selection-run-dir", required=True, type=Path)
parser.add_argument("--evaluation-metrics", required=True, action="append", type=Path)
args = parser.parse_args()
select_checkpoint(args.selection_run_dir, args.evaluation_metrics)
