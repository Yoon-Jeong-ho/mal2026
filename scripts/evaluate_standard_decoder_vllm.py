#!/usr/bin/env python3
"""Evaluate a standard Qwen LoRA adapter via vLLM offline batched generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.standard_decoder_vllm import VLLMEvalConfig, run_vllm_evaluation


def main(argv: list[str] | None = None) -> None:
    """Run the vLLM evaluator only from the CLI entry point.

    vLLM's worker launcher uses Python ``spawn``.  A module-level evaluator
    call would therefore be replayed while a spawned child imports this
    script, violating Python's safe-import contract and recursively starting
    another evaluator.  Keep argument parsing and execution behind the usual
    ``__main__`` guard so spawned workers can import the module safely.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    run_vllm_evaluation(VLLMEvalConfig.from_json(args.config))


if __name__ == "__main__":
    main()
