#!/usr/bin/env python3
"""Immutable v5.1 lineage: v5 transport repair plus explicit verdict iff rule."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repeat_v5", ROOT / "scripts/run_openai_explanation_repeat_distribution_v5.py")
V5 = importlib.util.module_from_spec(SPEC); assert SPEC and SPEC.loader; SPEC.loader.exec_module(V5)
V5.CONFIG_PATH = ROOT / "configs/openai_explanation_repeat_distribution.v5_1.pilot.json"
V5.SCHEMA = "mal2026-openai-explanation-repeat-distribution-v5_1"

if __name__ == "__main__": V5.main()
