#!/usr/bin/env bash
# Versioned wrapper: score-only train first, then frozen validation.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export MAL2026_DIST100_CONFIG="$ROOT/configs/qwen36_native_fp8_vllm_essay_only_score5x10.v4.json"
export MAL2026_DIST100_SCHEMA="mal2026-qwen36-native-fp8-vllm-essay-only-score5x10-v4"
export MAL2026_DIST100_RUN_PREFIX="qwen36-native-fp8-essay-only-score5x10-v4-"
export MAL2026_DIST100_TRAIN_OUTPUT_SUBDIR="judge_runs_essay_only_score5x10_v4"
export MAL2026_DIST100_VALIDATION_OUTPUT_SUBDIR="frozen_validation_judge_runs_essay_only_score5x10_v4"
export MAL2026_DIST100_OUTPUT_ROOT="native-fp8-vllm-essay-only-score5x10-v4"
exec bash "$ROOT/scripts/run_qwen36_native_fp8_vllm_distribution100_essay_only_v2_full.sh"
