#!/usr/bin/env bash
# One actual-candidate, score-blind 100-score gate.  The source writing score
# is neither read nor supplied to the judge under this v2 protocol.
set -Eeuo pipefail
[[ $# -eq 1 && ( "$1" == "gpu0" || "$1" == "dp4" ) ]] || { echo "usage: $0 gpu0|dp4" >&2; exit 2; }
MODE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PATH="$ROOT/.venv-standard/bin:$PATH"
PY="$ROOT/.venv-standard/bin/python"; VLLM="$ROOT/.venv-standard/bin/vllm"
CFG="${MAL2026_DIST100_CONFIG:-$ROOT/configs/qwen36_native_fp8_vllm_distribution100_essay_only.v2.json}"
export MAL2026_DIST100_CONFIG="$CFG"
export MAL2026_DIST100_SCHEMA="${MAL2026_DIST100_SCHEMA:-mal2026-qwen36-native-fp8-vllm-distribution100-essay-only-v2}"
MODEL="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-FP8-native-fp8-v1"
MODEL_ID="Qwen/Qwen3.6-35B-A3B-FP8"; SCORE="$ROOT/scripts/score_rationale_distribution_vllm_dp4.py"
RUN_PREFIX="${MAL2026_DIST100_RUN_PREFIX:-qwen36-native-fp8-dist100-essay-only-v2-}"
TRAIN_SUBDIR="${MAL2026_DIST100_TRAIN_OUTPUT_SUBDIR:-judge_runs_essay_only_v2}"
OUTPUT_ROOT="${MAL2026_DIST100_OUTPUT_ROOT:-native-fp8-vllm-distribution100-essay-only-v2}"
SUFFIX="${MAL2026_DIST100_SMOKE_SUFFIX:-${MAL2026_DIST100_ESSAY_ONLY_SMOKE_SUFFIX:-001}}"
[[ "$SUFFIX" =~ ^[0-9]{3}$ ]] || { echo "smoke suffix must be three digits" >&2; exit 2; }
[[ -x "$PY" && -x "$VLLM" && -x "$SCORE" && -f "$CFG" && -d "$MODEL" ]] || { echo "native FP8 runtime prerequisite is unavailable" >&2; exit 1; }
read -r GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS EAGER_FLAG < <("$PY" - "$CFG" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))['runtime']
assert isinstance(r.get('enforce_eager'), bool)
print(r['gpu_memory_utilization'],r['max_num_seqs_per_dp_rank'],r['max_num_batched_tokens'], int(r['enforce_eager']))
PY
)
[[ "$EAGER_FLAG" == 0 || "$EAGER_FLAG" == 1 ]] || { echo "invalid eager runtime flag" >&2; exit 2; }
EAGER_ARGS=(); [[ "$EAGER_FLAG" == 1 ]] && EAGER_ARGS=(--enforce-eager)
if [[ "$MODE" == gpu0 ]]; then
  PHASE=gpu0_smoke; RUN_ID="${RUN_PREFIX}train-20260720-gpu0_smoke-$SUFFIX"; PORT=18343; CVD=0; DP=1; GPUS=(0)
else
  PHASE=dp4_smoke; RUN_ID="${RUN_PREFIX}train-20260720-dp4_smoke-$SUFFIX"; PORT=18344; CVD=0,1,2,3; DP=4; GPUS=(0 1 2 3)
fi
DEST="$ROOT/data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001/$TRAIN_SUBDIR/$RUN_ID"
OUT="$ROOT/outputs/$OUTPUT_ROOT/$RUN_ID"
# The launcher log directory may be pre-created before durable tmux redirection.
# The restricted destination, however, is the immutable lineage guard.
[[ ! -e "$DEST" ]] || { echo "refusing to overwrite existing smoke lineage" >&2; exit 1; }
for gpu in "${GPUS[@]}"; do
  line="$(nvidia-smi --id="$gpu" --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)"
  IFS=, read -r index memory util temp <<<"$line"
  [[ "${index// /}" == "$gpu" && "${memory// /}" == 0 && "${util// /}" == 0 && "${temp// /}" -le 80 ]] || { echo "GPU $gpu is not idle/cool" >&2; exit 1; }
done
"$PY" "$SCORE" prepare --split train --run-id "$RUN_ID" --phase "$PHASE" --limit-candidates 1
mkdir -p "$OUT"
PID=""
cleanup() { rc=$?; [[ -z "$PID" ]] || kill "$PID" 2>/dev/null || true; [[ -z "$PID" ]] || wait "$PID" 2>/dev/null || true; exit "$rc"; }
trap cleanup EXIT INT TERM
env CUDA_VISIBLE_DEVICES="$CVD" MAL2026_RESERVED_PHYSICAL_GPUS="$CVD" "$VLLM" serve "$MODEL" \
  --served-model-name "$MODEL_ID" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 --data-parallel-size "$DP" --max-model-len 4096 --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --gdn-prefill-backend triton --generation-config vllm --enable-prefix-caching "${EAGER_ARGS[@]}" \
  >"$OUT/server.log" 2>&1 & PID="$!"
for _ in $(seq 1 240); do curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null
"$PY" - "$DEST/server_attestation.json" "$CFG" "$PORT" "$PID" "$CVD" "$DP" "$MAX_NUM_SEQS" <<'PY'
import hashlib,json,sys
from pathlib import Path
path,cfg,port,pid,cvd,dp,max_num_seqs=map(str,sys.argv[1:])
visible=open(f'/proc/{pid}/environ','rb').read()
assert f'CUDA_VISIBLE_DEVICES={cvd}'.encode() in visible
gpus=[int(x) for x in cvd.split(',')]
payload={'schema_version':'mal2026-native-fp8-vllm-distribution100-server-attestation-v1','server_host':'127.0.0.1','server_port':int(port),'physical_gpus':gpus,'tensor_parallel_size':1,'data_parallel_size':int(dp),'max_model_len':4096,'max_num_seqs_per_dp_rank':int(max_num_seqs),'config_sha256':hashlib.sha256(Path(cfg).read_bytes()).hexdigest(),'server_process_environment_verified':True}
Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
"$PY" "$SCORE" execute --split train --run-id "$RUN_ID" --phase "$PHASE" --endpoint "http://127.0.0.1:$PORT" --server-attestation "$DEST/server_attestation.json" --model "$MODEL_ID"
