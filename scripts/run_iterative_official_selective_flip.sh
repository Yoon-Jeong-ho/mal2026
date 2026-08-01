#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec .venv-standard/bin/python scripts/run_iterative_official_selective_flip.py "$@"
