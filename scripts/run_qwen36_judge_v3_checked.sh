#!/usr/bin/env bash
# Static/provenance gate for the versioned v3 train-only remediation pilot.
set -Eeuo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 BATCH JUDGE MODEL SERVER PORT" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"; PYTHON="$ROOT/.venv-standard/bin/python"
BATCH="$1"; JUDGE="$2"; MODEL="$3"; SERVER="$4"; PORT="$5"; CONFIG="$ROOT/configs/qwen36_gguf_judge.v3.pilot.json"
DERIVED="$ROOT/data/processed/restricted/openai_rationale_batches/$BATCH/derived/train-only-candidates-v1-20260719-001"; LLAMA_REPO="$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/llama.cpp"
[[ -x "$PYTHON" && "$BATCH" == "openai-rationale-terra-full-20260719-001" && "$JUDGE" =~ ^qwen36-judge-v3-pilot-20260720-[0-9]{3}$ ]] || { echo "v3 lineage/run ID gate failed" >&2; exit 1; }
[[ "$MODEL" == "$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" && "$SERVER" == "$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server" && -r "$MODEL" && -x "$SERVER" ]] || { echo "v3 immutable executable gate failed" >&2; exit 1; }
[[ "$(stat -c %s "$MODEL")" == "22285080192" && "$(sha256sum "$MODEL" | awk '{print $1}')" == "b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772" ]] || { echo "v3 GGUF gate failed" >&2; exit 1; }
[[ "$(git -C "$LLAMA_REPO" rev-parse HEAD)" == "571d0d540df04f25298d0e159e520d9fc62ed121" && "$(git -C "$LLAMA_REPO" describe --tags --exact-match)" == "b10068" ]] || { echo "v3 llama revision gate failed" >&2; exit 1; }
env MAL2026_JUDGE_CONFIG="$CONFIG" MAL2026_JUDGE_SCHEMA="qwen36-gguf-judge-v3-pilot" PYTHONPATH="$ROOT/src" "$PYTHON" - "$CONFIG" "$DERIVED/candidates.train.manifest.json" "$DERIVED/candidates.train.jsonl" "$BATCH" <<'PY'
import hashlib, json, sys
from pathlib import Path
config_path, manifest_path, candidate_path = map(Path, sys.argv[1:4])
batch = sys.argv[4]
config = json.loads(config_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
if not (config.get("schema_version") == "qwen36-gguf-judge-v3-pilot" and config.get("runtime", {}).get("gpu_allowlist") == [0] and config["runtime"].get("parallel_requests") == 4 and config.get("selection", {}).get("split") == "train" and config["selection"].get("max_essays") == 32 and config.get("protocol", {}).get("validation_policy") == "never_load_validation_source_rows_or_construct_validation_requests" and config["protocol"].get("selection_artifact_permitted") is False and manifest.get("status") == "completed" and manifest.get("batch_run_id") == batch and manifest.get("split") == "train" and manifest.get("row_count") == 6000 and manifest.get("candidate_file_sha256") == digest):
    raise SystemExit("v3 split/config/parallel binding gate failed")
PY
[[ "${MAL2026_JUDGE_STATIC_ONLY:-}" == "1" ]] && exit 0
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0" && "${MAL2026_RESERVED_PHYSICAL_GPU:-}" == "0" ]] || { echo "v3 wrapper requires watchdog GPU 0" >&2; exit 1; }
exec bash "$ROOT/scripts/run_qwen36_judge_v3_pilot.sh" "$BATCH" "$JUDGE" "$MODEL" "$SERVER" "$PORT"
