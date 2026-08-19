#!/usr/bin/env bash
# Immutable preflight wrapper for the GPU-0-only train-only judge-v2 runner.
set -Eeuo pipefail
if [[ $# -lt 4 || $# -gt 5 ]]; then echo "usage: $0 BATCH_RUN_ID JUDGE_RUN_ID MODEL_GGUF LLAMA_SERVER [PORT]" >&2; exit 2; fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="$ROOT/.venv-standard/bin/python"
[[ -x "$PYTHON" ]] || { echo "documented interpreter missing" >&2; exit 1; }
BATCH="$1"; JUDGE="$2"; MODEL="$3"; SERVER="$4"; PORT="${5:-18084}"
CONFIG="$ROOT/configs/qwen36_gguf_judge.v2.pilot.json"
DERIVED="$ROOT/data/processed/restricted/openai_rationale_batches/$BATCH/derived/train-only-candidates-v1-20260719-001"
LLAMA_REPO="$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/llama.cpp"
[[ "$BATCH" == "openai-rationale-terra-full-20260719-001" ]] || { echo "unapproved judge batch lineage" >&2; exit 1; }
[[ "$JUDGE" =~ ^qwen36-judge-v2-pilot-20260720-[0-9]{3}$ ]] || { echo "judge run ID is not fresh/canonical" >&2; exit 1; }
[[ "$MODEL" == "$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" ]] || { echo "GGUF path differs from immutable plan" >&2; exit 1; }
[[ "$SERVER" == "$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server" ]] || { echo "llama-server path differs from immutable plan" >&2; exit 1; }
[[ -r "$MODEL" && -x "$SERVER" && -f "$DERIVED/candidates.train.jsonl" && -f "$DERIVED/candidates.train.manifest.json" ]] || { echo "judge prerequisite file is absent" >&2; exit 1; }
[[ "$(stat -c %s "$MODEL")" == "22285080192" ]] || { echo "GGUF byte-size gate failed" >&2; exit 1; }
[[ "$(sha256sum "$MODEL" | awk '{print $1}')" == "b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772" ]] || { echo "GGUF checksum gate failed" >&2; exit 1; }
[[ "$(git -C "$LLAMA_REPO" rev-parse HEAD)" == "571d0d540df04f25298d0e159e520d9fc62ed121" ]] || { echo "llama.cpp revision gate failed" >&2; exit 1; }
[[ "$(git -C "$LLAMA_REPO" describe --tags --exact-match)" == "b10068" ]] || { echo "llama.cpp release-tag gate failed" >&2; exit 1; }
PYTHONPATH="$ROOT/src" "$PYTHON" - "$CONFIG" "$DERIVED/candidates.train.manifest.json" "$DERIVED/candidates.train.jsonl" "$BATCH" <<'PYCODE'
import hashlib
import json
import sys
from pathlib import Path
config_path, manifest_path, candidate_path = map(Path, sys.argv[1:4])
batch = sys.argv[4]
config = json.loads(config_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
if not (manifest.get("status") == "completed" and manifest.get("batch_run_id") == batch and manifest.get("split") == "train" and manifest.get("row_count") == 6000 and manifest.get("candidate_file_sha256") == digest):
    raise SystemExit("derived train-only candidate binding gate failed")
selection, protocol, runtime = config.get("selection", {}), config.get("protocol", {}), config.get("runtime", {})
if not (selection.get("required_candidate_artifact") == "derived/train-only-candidates-v1-20260719-001/candidates.train.jsonl" and selection.get("required_candidate_manifest") == "derived/train-only-candidates-v1-20260719-001/candidates.train.manifest.json" and runtime.get("gpu_allowlist") == [0] and protocol.get("validation_policy") == "never_load_validation_source_rows_or_construct_validation_requests" and protocol.get("selection_artifact_permitted") is False):
    raise SystemExit("judge isolation/config binding gate failed")
PYCODE
if [[ "${MAL2026_JUDGE_STATIC_ONLY:-}" == "1" ]]; then exit 0; fi
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0" && "${MAL2026_RESERVED_PHYSICAL_GPU:-}" == "0" ]] || { echo "judge wrapper requires exactly watchdog-assigned GPU 0" >&2; exit 1; }
exec bash "$ROOT/scripts/run_qwen36_judge_v2_pilot.sh" "$BATCH" "$JUDGE" "$MODEL" "$SERVER" "$PORT"
