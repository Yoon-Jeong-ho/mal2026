#!/usr/bin/env bash
# Actual-input contrastive validity calibration for the fixed v6 score-only
# rationale judge.  GPU0 is an execution gate; full uses one DP=4 endpoint.
set -Eeuo pipefail
[[ $# -eq 1 && ( "$1" == "gpu0" || "$1" == "full" ) ]] || { echo "usage: $0 gpu0|full" >&2; exit 2; }
MODE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PATH="$ROOT/.venv-standard/bin:$PATH"
PY="$ROOT/.venv-standard/bin/python"; VLLM="$ROOT/.venv-standard/bin/vllm"
CFG="$ROOT/configs/qwen36_native_fp8_vllm_rationale_only_score5x10.v6.json"
export MAL2026_DIST100_CONFIG="$CFG"
export MAL2026_DIST100_SCHEMA="mal2026-qwen36-native-fp8-vllm-rationale-only-score5x10-v6"
MODEL="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-FP8-native-fp8-v1"
MODEL_ID="Qwen/Qwen3.6-35B-A3B-FP8"
EVALUATE="$ROOT/scripts/evaluate_rationale_judge_contrastive_v1.py"
BATCH="$ROOT/data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001"
DEST_ROOT="$BATCH/judge_contrastive_validity_v1"
OUT_ROOT="$ROOT/outputs/native-fp8-vllm-rationale-contrastive-v1"
[[ -x "$PY" && -x "$VLLM" && -x "$EVALUATE" && -f "$CFG" && -d "$MODEL" ]] || { echo "contrastive runtime prerequisite is unavailable" >&2; exit 1; }
read -r GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS EAGER_FLAG < <("$PY" - "$CFG" <<'PY'
import json,sys
runtime=json.load(open(sys.argv[1], encoding='utf-8'))['runtime']
assert isinstance(runtime.get('enforce_eager'), bool)
print(runtime['gpu_memory_utilization'], runtime['max_num_seqs_per_dp_rank'], runtime['max_num_batched_tokens'], int(runtime['enforce_eager']))
PY
)
[[ "$EAGER_FLAG" == 0 || "$EAGER_FLAG" == 1 ]] || { echo "invalid eager runtime flag" >&2; exit 2; }
EAGER_ARGS=(); [[ "$EAGER_FLAG" == 1 ]] && EAGER_ARGS=(--enforce-eager)
if [[ "$MODE" == "gpu0" ]]; then
  PHASE="gpu0_smoke"; RUN_ID="qwen36-native-fp8-rationale-contrastive-v1-train-20260720-gpu0_smoke-001"
  PORT=18348; CVD="0"; DP=1; GPUS=(0)
else
  PHASE="full"; RUN_ID="qwen36-native-fp8-rationale-contrastive-v1-train-20260720-full-001"
  PORT=18349; CVD="0,1,2,3"; DP=4; GPUS=(0 1 2 3)
  "$PY" - "$DEST_ROOT/qwen36-native-fp8-rationale-contrastive-v1-train-20260720-gpu0_smoke-001/aggregate_contrastive_report.json" <<'PY'
import json,sys
report=json.load(open(sys.argv[1], encoding='utf-8'))
assert report['status'] == 'completed' and all(report['hard_gates'].values())
PY
fi
DEST="$DEST_ROOT/$RUN_ID"; OUT="$OUT_ROOT/$RUN_ID"
[[ ! -e "$DEST" ]] || { echo "refusing to overwrite contrastive lineage" >&2; exit 1; }
for gpu in "${GPUS[@]}"; do
  line="$(nvidia-smi --id="$gpu" --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)"
  IFS=, read -r index memory util temp <<<"$line"
  [[ "${index// /}" == "$gpu" && "${memory// /}" == 0 && "${util// /}" == 0 && "${temp// /}" -le 80 ]] || { echo "GPU $gpu is not idle/cool" >&2; exit 1; }
done
"$PY" "$EVALUATE" prepare --run-id "$RUN_ID" --phase "$PHASE"
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
visible=Path(f'/proc/{pid}/environ').read_bytes()
assert f'CUDA_VISIBLE_DEVICES={cvd}'.encode() in visible
payload={'schema_version':'mal2026-rationale-contrastive-v1-server-attestation-v1','server_host':'127.0.0.1','server_port':int(port),'physical_gpus':[int(gpu) for gpu in cvd.split(',')],'tensor_parallel_size':1,'data_parallel_size':int(dp),'max_model_len':4096,'max_num_seqs_per_dp_rank':int(max_num_seqs),'config_sha256':hashlib.sha256(Path(cfg).read_bytes()).hexdigest(),'server_process_environment_verified':True}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n', encoding='utf-8')
PY
"$PY" "$EVALUATE" execute --run-id "$RUN_ID" --phase "$PHASE" --endpoint "http://127.0.0.1:$PORT" --server-attestation "$DEST/server_attestation.json" --model "$MODEL_ID"
