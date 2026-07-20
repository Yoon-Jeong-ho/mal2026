#!/usr/bin/env bash
# Durable score-blind collection: train first, then frozen validation.  One
# vLLM DP4 endpoint owns GPUs 0--3; it is never a serial GPU loop.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PATH="$ROOT/.venv-standard/bin:$PATH"
PY="$ROOT/.venv-standard/bin/python"; VLLM="$ROOT/.venv-standard/bin/vllm"
CFG="${MAL2026_DIST100_CONFIG:-$ROOT/configs/qwen36_native_fp8_vllm_distribution100_essay_only.v2.json}"
export MAL2026_DIST100_CONFIG="$CFG"
export MAL2026_DIST100_SCHEMA="${MAL2026_DIST100_SCHEMA:-mal2026-qwen36-native-fp8-vllm-distribution100-essay-only-v2}"
MODEL="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-FP8-native-fp8-v1"; MODEL_ID="Qwen/Qwen3.6-35B-A3B-FP8"
SCORE="$ROOT/scripts/score_rationale_distribution_vllm_dp4.py"; DERIVE="$ROOT/scripts/derive_validation_only_candidates.py"
BATCH="$ROOT/data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001"
RUN_PREFIX="${MAL2026_DIST100_RUN_PREFIX:-qwen36-native-fp8-dist100-essay-only-v2-}"
TRAIN_SUBDIR="${MAL2026_DIST100_TRAIN_OUTPUT_SUBDIR:-judge_runs_essay_only_v2}"
VALIDATION_SUBDIR="${MAL2026_DIST100_VALIDATION_OUTPUT_SUBDIR:-frozen_validation_judge_runs_essay_only_v2}"
OUTPUT_ROOT="${MAL2026_DIST100_OUTPUT_ROOT:-native-fp8-vllm-distribution100-essay-only-v2}"
TRAIN_ID="${MAL2026_DIST100_TRAIN_ID:-${RUN_PREFIX}train-20260720-full-001}"
VALIDATION_ID="${MAL2026_DIST100_VALIDATION_ID:-${RUN_PREFIX}validation-20260720-full-001}"
GPU0_SMOKE_ID="${MAL2026_DIST100_GPU0_SMOKE_ID:-${MAL2026_DIST100_ESSAY_ONLY_GPU0_SMOKE_ID:-${RUN_PREFIX}train-20260720-gpu0_smoke-001}}"
DP4_SMOKE_ID="${MAL2026_DIST100_DP4_SMOKE_ID:-${MAL2026_DIST100_ESSAY_ONLY_DP4_SMOKE_ID:-${RUN_PREFIX}train-20260720-dp4_smoke-001}}"
TRAIN_DEST="$BATCH/$TRAIN_SUBDIR/$TRAIN_ID"; VALIDATION_DEST="$BATCH/$VALIDATION_SUBDIR/$VALIDATION_ID"
OUT="$ROOT/outputs/$OUTPUT_ROOT/$TRAIN_ID"; PORT=18340; PID=""
[[ -x "$PY" && -x "$VLLM" && -x "$SCORE" && -x "$DERIVE" && -f "$CFG" && -d "$MODEL" ]] || { echo "essay-only distribution runtime prerequisite is unavailable" >&2; exit 1; }
read -r GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS < <("$PY" - "$CFG" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))['runtime']
print(r['gpu_memory_utilization'],r['max_num_seqs_per_dp_rank'],r['max_num_batched_tokens'])
PY
)
"$PY" - "$BATCH/$TRAIN_SUBDIR/$GPU0_SMOKE_ID/aggregate_score_report.json" "$BATCH/$TRAIN_SUBDIR/$DP4_SMOKE_ID/aggregate_score_report.json" <<'PY'
import json,sys
for p in sys.argv[1:]:
 d=json.load(open(p)); assert d['status']=='passed' and all(d['hard_gates'].values())
PY
VALIDATION_ARTIFACT="$BATCH/derived/validation-only-candidates-v1-20260720-001/candidates.validation.jsonl"
if [[ ! -f "$VALIDATION_ARTIFACT" ]]; then "$PY" "$DERIVE" --derived-run-id validation-only-candidates-v1-20260720-001; fi
for gpu in 0 1 2 3; do
  line="$(nvidia-smi --id="$gpu" --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)"
  IFS=, read -r index memory util temp <<<"$line"
  [[ "${index// /}" == "$gpu" && "${memory// /}" == 0 && "${util// /}" == 0 && "${temp// /}" -le 80 ]] || { echo "GPU $gpu is not idle/cool" >&2; exit 1; }
done
if [[ ! -e "$TRAIN_DEST" ]]; then "$PY" "$SCORE" prepare --split train --run-id "$TRAIN_ID" --phase full; fi
mkdir -p "$OUT"
cleanup() { rc=$?; [[ -z "$PID" ]] || kill "$PID" 2>/dev/null || true; [[ -z "$PID" ]] || wait "$PID" 2>/dev/null || true; exit "$rc"; }
trap cleanup EXIT INT TERM
env CUDA_VISIBLE_DEVICES=0,1,2,3 MAL2026_RESERVED_PHYSICAL_GPUS=0,1,2,3 "$VLLM" serve "$MODEL" \
  --served-model-name "$MODEL_ID" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 --data-parallel-size 4 --max-model-len 4096 --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --gdn-prefill-backend triton --generation-config vllm --enable-prefix-caching --enforce-eager \
  >"$OUT/server.log" 2>&1 & PID="$!"
for _ in $(seq 1 240); do curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null
attest() {
  local dest="$1"
  "$PY" - "$dest/server_attestation.json" "$CFG" "$PORT" "$PID" "$MAX_NUM_SEQS" <<'PY'
import hashlib,json,sys
from pathlib import Path
path,cfg,port,pid,max_num_seqs=sys.argv[1:]
visible=open(f'/proc/{pid}/environ','rb').read(); assert b'CUDA_VISIBLE_DEVICES=0,1,2,3' in visible
payload={'schema_version':'mal2026-native-fp8-vllm-distribution100-server-attestation-v1','server_host':'127.0.0.1','server_port':int(port),'physical_gpus':[0,1,2,3],'tensor_parallel_size':1,'data_parallel_size':4,'max_model_len':4096,'max_num_seqs_per_dp_rank':int(max_num_seqs),'config_sha256':hashlib.sha256(Path(cfg).read_bytes()).hexdigest(),'server_process_environment_verified':True}
Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
}
attest "$TRAIN_DEST"
"$PY" "$SCORE" execute --split train --run-id "$TRAIN_ID" --phase full --endpoint "http://127.0.0.1:$PORT" --server-attestation "$TRAIN_DEST/server_attestation.json" --model "$MODEL_ID"
if [[ ! -e "$VALIDATION_DEST" ]]; then "$PY" "$SCORE" prepare --split validation --run-id "$VALIDATION_ID" --phase full; fi
attest "$VALIDATION_DEST"
"$PY" "$SCORE" execute --split validation --run-id "$VALIDATION_ID" --phase full --endpoint "http://127.0.0.1:$PORT" --server-attestation "$VALIDATION_DEST/server_attestation.json" --model "$MODEL_ID"
