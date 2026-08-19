#!/usr/bin/env python3
"""Immutable v5.2 lineage with the parser bound to the emitted schema."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("repeat_v5",ROOT/"scripts/run_openai_explanation_repeat_distribution_v5.py")
V5=importlib.util.module_from_spec(SPEC); assert SPEC and SPEC.loader; SPEC.loader.exec_module(V5)
V5.CONFIG_PATH=ROOT/"configs/openai_explanation_repeat_distribution.v5_2.pilot.json"
V5.SCHEMA="mal2026-openai-explanation-repeat-distribution-v5_2"
V5.WIRE.SCHEMA=V5.SCHEMA
if __name__ == "__main__": V5.main()
