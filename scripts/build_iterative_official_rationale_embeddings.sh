#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv-standard/bin/python}"
SCRIPT="${ROOT}/scripts/build_iterative_official_rationale_embeddings.py"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {smoke|shard N|launch|merge|progress} [builder arguments...]" >&2
  exit 2
fi

action="$1"
shift
case "${action}" in
  smoke)
    CUDA_VISIBLE_DEVICES=0 exec "${PYTHON_BIN}" "${SCRIPT}" --smoke --physical-gpu 0 "$@"
    ;;
  shard)
    if [[ $# -lt 1 || ! "$1" =~ ^[0-3]$ ]]; then
      echo "shard requires one index in 0..3" >&2
      exit 2
    fi
    shard="$1"
    shift
    CUDA_VISIBLE_DEVICES="${shard}" exec "${PYTHON_BIN}" "${SCRIPT}" \
      --shard "${shard}" --physical-gpu "${shard}" "$@"
    ;;
  launch)
    exec "${PYTHON_BIN}" "${SCRIPT}" --launch "$@"
    ;;
  merge)
    exec "${PYTHON_BIN}" "${SCRIPT}" --merge "$@"
    ;;
  progress)
    exec "${PYTHON_BIN}" "${SCRIPT}" --progress "$@"
    ;;
  *)
    echo "unknown action: ${action}" >&2
    exit 2
    ;;
esac
