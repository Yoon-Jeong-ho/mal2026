#!/usr/bin/env bash
# Versioned wrapper: five score-blind prompt types × ten seed repeats each.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export MAL2026_DIST100_CONFIG="$ROOT/configs/qwen36_native_fp8_vllm_essay_only_prompt5x10.v3.json"
export MAL2026_DIST100_SCHEMA="mal2026-qwen36-native-fp8-vllm-essay-only-prompt5x10-v3"
export MAL2026_DIST100_RUN_PREFIX="qwen36-native-fp8-essay-only-prompt5x10-v3-"
export MAL2026_DIST100_TRAIN_OUTPUT_SUBDIR="judge_runs_essay_only_prompt5x10_v3"
export MAL2026_DIST100_OUTPUT_ROOT="native-fp8-vllm-essay-only-prompt5x10-v3"
exec bash "$ROOT/scripts/run_qwen36_native_fp8_vllm_distribution100_essay_only_v2_smoke.sh" "$@"
