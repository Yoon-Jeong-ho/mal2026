#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export MAL2026_REPEAT_CONFIG="$ROOT/configs/openai_explanation_repeat_distribution.v5_1.pilot.json"
export MAL2026_REPEAT_RUNNER="$ROOT/scripts/run_openai_explanation_repeat_distribution_v5_1.py"
exec "$ROOT/scripts/run_openai_explanation_repeat_distribution_v5.sh" "$@"
