#!/usr/bin/env bash
# One actual-candidate, 100-score gate.  GPU0 runs first; DP4 is allowed only
# after this script's GPU0 invocation has passed and recorded its gate.
set -Eeuo pipefail
[[ $# -eq 1 && ( "$1" == "gpu0" || "$1" == "dp4" ) ]] || { echo "usage: $0 gpu0|dp4" >&2; exit 2; }
MODE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PATH="$ROOT/.venv-standard/bin:$PATH"
PY="$ROOT/.venv-standard/bin/python"; VLLM="$ROOT/.venv-standard/bin/vllm"
CFG="$ROOT/configs/qwen36_native_fp8_vllm_distribution100.v1.json"
MODEL="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-FP8-native-fp8-v1"
MODEL_ID="Qwen/Qwen3.6-35B-A3B-FP8"; SCORE="$ROOT/scripts/score_rationale_distribution_vllm_dp4.py"
SUFFIX="${MAL2026_DIST100_SMOKE_SUFFIX:-001}"
[[ "$SUFFIX" =~ ^[0-9]{3}$ ]] || { echo "smoke suffix must be three digits" >&2; exit 2; }
[[ -x "$PY" && -x "$VLLM" && -d "$MODEL" ]] || { echo "native FP8 runtime prerequisite is unavailable" >&2; exit 1; }
if [[ "$MODE" == gpu0 ]]; then
  PHASE=gpu0_smoke; RUN_ID=qwen36-native-fp8-dist100-train-20260720-gpu0_smoke-"$SUFFIX"; PORT=18341; CVD=0; DP=1; GPUS=(0)
else
  PHASE=dp4_smoke; RUN_ID=qwen36-native-fp8-dist100-train-20260720-dp4_smoke-"$SUFFIX"; PORT=18342; CVD=0,1,2,3; DP=4; GPUS=(0 1 2 3)
fi
DEST="$ROOT/data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001/judge_runs/$RUN_ID"
OUT="$ROOT/outputs/native-fp8-vllm-distribution100/$RUN_ID"
[[ ! -e "$DEST" && ! -e "$OUT" ]] || { echo "refusing to overwrite existing smoke lineage" >&2; exit 1; }
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
  --tensor-parallel-size 1 --data-parallel-size "$DP" --max-model-len 4096 --max-num-seqs 64 --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.80 --gdn-prefill-backend triton --generation-config vllm --enable-prefix-caching --enforce-eager \
  >"$OUT/server.log" 2>&1 & PID="$!"
for _ in $(seq 1 240); do curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null
"$PY" - "$DEST/server_attestation.json" "$CFG" "$PORT" "$PID" "$CVD" "$DP" <<'PY'
import hashlib,json,sys
from pathlib import Path
path,cfg,port,pid,cvd,dp=map(str,sys.argv[1:])
visible=open(f'/proc/{pid}/environ','rb').read()
assert f'CUDA_VISIBLE_DEVICES={cvd}'.encode() in visible
gpus=[int(x) for x in cvd.split(',')]
payload={'schema_version':'mal2026-native-fp8-vllm-distribution100-server-attestation-v1','server_host':'127.0.0.1','server_port':int(port),'physical_gpus':gpus,'tensor_parallel_size':1,'data_parallel_size':int(dp),'max_model_len':4096,'max_num_seqs_per_dp_rank':64,'config_sha256':hashlib.sha256(Path(cfg).read_bytes()).hexdigest(),'server_process_environment_verified':True}
Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
"$PY" "$SCORE" execute --split train --run-id "$RUN_ID" --phase "$PHASE" --endpoint "http://127.0.0.1:$PORT" --server-attestation "$DEST/server_attestation.json" --model "$MODEL_ID"
