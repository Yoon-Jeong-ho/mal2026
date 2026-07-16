#!/usr/bin/env python3
"""Evaluate a standard Qwen LoRA adapter via vLLM offline batched generation."""
from __future__ import annotations
import argparse
from pathlib import Path
from mal2026.standard_decoder_vllm import VLLMEvalConfig, run_vllm_evaluation
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=Path)
args = parser.parse_args()
run_vllm_evaluation(VLLMEvalConfig.from_json(args.config))
