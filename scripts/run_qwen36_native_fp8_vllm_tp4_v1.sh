#!/usr/bin/env bash
# One-command TP=4 vLLM synthetic throughput gate on project GPUs 0--3.
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 RUN_ID" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PATH="$ROOT/.venv-standard/bin:$PATH"
RUN_ID="$1"; PY="$ROOT/.venv-standard/bin/python"; VLLM="$ROOT/.venv-standard/bin/vllm"
CFG="${MAL2026_TP4_CONFIG:-$ROOT/configs/qwen36_native_fp8_vllm.tp4.v1.json}"
MODEL="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-FP8-native-fp8-v1"
MANIFEST="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-FP8-native-fp8-v1.manifest.json"
OUT="$ROOT/outputs/native-fp8-vllm-tp4/$RUN_ID"; SYN="$ROOT/scripts/preflight_qwen36_native_fp8_vllm_synthetic.py"
MODEL_ID="Qwen/Qwen3.6-35B-A3B-FP8"; PID=""; PORT=""; EAGER=()
[[ "$RUN_ID" =~ ^native-fp8-vllm-tp4-20260720-[0-9]{3}$ ]] || { echo "run id is outside TP4 lineage" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite TP4 run" >&2; exit 1; }
cleanup() { rc=$?; [[ -z "$PID" ]] || kill "$PID" 2>/dev/null || true; [[ -z "$PID" ]] || wait "$PID" 2>/dev/null || true; exit "$rc"; }
trap cleanup EXIT INT TERM
[[ "$(dirname "$CFG")" == "$ROOT/configs" && -f "$CFG" && ! -L "$CFG" ]] || { echo "TP4 config is outside canonical config root" >&2; exit 2; }
read -r PORT EAGER_FLAG < <("$PY" - "$MANIFEST" "$CFG" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2]))
assert m['repository']=='Qwen/Qwen3.6-35B-A3B-FP8' and len(m['files'])==56
r=c['runtime']; assert r['physical_gpus']==[0,1,2,3] and r['tensor_parallel_size']==4 and r['data_parallel_size']==1
assert r['max_num_seqs']==64 and r['max_num_batched_tokens']==32768
assert c['request']=={'chat_template_kwargs':{'enable_thinking':False},'max_tokens':192,'top_p':1.0}
print(r['port'], '1' if r.get('enforce_eager') is True else '0')
PY
)
[[ "$EAGER_FLAG" == 0 || "$EAGER_FLAG" == 1 ]] || { echo "invalid eager runtime flag" >&2; exit 2; }
[[ "$EAGER_FLAG" == 1 ]] && EAGER=(--enforce-eager)
for gpu in 0 1 2 3; do
  x="$(nvidia-smi --id="$gpu" --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)"
  IFS=, read -r index memory util temp <<<"$x"
  [[ "${index// /}" == "$gpu" && "${memory// /}" == 0 && "${util// /}" == 0 && "${temp// /}" -le 80 ]] || { echo "GPU $gpu is not idle/cool" >&2; exit 1; }
done
mkdir -p "$OUT"
env CUDA_VISIBLE_DEVICES=0,1,2,3 MAL2026_RESERVED_PHYSICAL_GPUS=0,1,2,3 "$VLLM" serve "$MODEL" \
  --served-model-name "$MODEL_ID" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 4 --data-parallel-size 1 --max-model-len 4096 \
  --max-num-seqs 64 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.80 \
  --gdn-prefill-backend triton "${EAGER[@]}" >"$OUT/server-tp4.log" 2>&1 & PID="$!"
for _ in $(seq 1 180); do curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null
"$PY" - "$PID" <<'PY'
import sys
value=open(f'/proc/{sys.argv[1]}/environ','rb').read()
assert b'CUDA_VISIBLE_DEVICES=0,1,2,3' in value
PY
"$PY" "$SYN" --endpoint "http://127.0.0.1:$PORT" --model "$MODEL_ID" --gpu 0 --config "$CFG" --run-dir "$OUT/synthetic"
"$PY" - "$CFG" "$OUT/synthetic/aggregate.json" "$OUT/attestation.json" <<'PY'
import hashlib,json,sys
c,a,o=sys.argv[1:]; aggregate=json.load(open(a)); cfg=json.load(open(c)); r=cfg['runtime']
assert aggregate['status']=='passed'
json.dump({'schema_version':'mal2026-vllm-tp4-attestation-v1','physical_gpus':r['physical_gpus'],'tensor_parallel_size':r['tensor_parallel_size'],'data_parallel_size':r['data_parallel_size'],'max_num_seqs':r['max_num_seqs'],'max_num_batched_tokens':r['max_num_batched_tokens'],'config_sha256':hashlib.sha256(open(c,'rb').read()).hexdigest(),'synthetic_aggregate_sha256':hashlib.sha256(open(a,'rb').read()).hexdigest(),'raw_payloads_or_outputs_persisted':False},open(o,'x'),indent=2,sort_keys=True)
PY
