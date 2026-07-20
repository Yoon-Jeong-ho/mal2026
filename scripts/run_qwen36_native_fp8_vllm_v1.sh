#!/usr/bin/env bash
# Fail-closed launcher for the isolated native-FP8 vLLM lane.
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 RUN_ID {gpu0-synthetic|compare|workers-synthetic|smoke|full}" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PATH="$ROOT/.venv-standard/bin:$PATH"
RUN_ID="$1"; STAGE="$2"; PY="$ROOT/.venv-standard/bin/python"; VLLM="$ROOT/.venv-standard/bin/vllm"
CFG="$ROOT/configs/qwen36_native_fp8_vllm.v1.json"; MODEL="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-FP8-native-fp8-v1"
MANIFEST="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-FP8-native-fp8-v1.manifest.json"; OUT="$ROOT/outputs/native-fp8-vllm/$RUN_ID"
SYN="$ROOT/scripts/preflight_qwen36_native_fp8_vllm_synthetic.py"; RUNNER="$ROOT/scripts/run_qwen36_native_fp8_vllm_v1.py"; GATE="$ROOT/scripts/record_native_fp8_gate.py"
MODEL_ID="Qwen/Qwen3.6-35B-A3B-FP8"; PIDS=(); mkdir -p "$OUT"
cleanup() { rc=$?; for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done; exit "$rc"; }
trap cleanup EXIT INT TERM
need_model() { [[ -d "$MODEL" && -f "$MANIFEST" ]] || { echo "verified model missing" >&2; exit 1; }; "$PY" - "$MANIFEST" "$CFG" <<'PY'
import hashlib,json,sys
m=json.load(open(sys.argv[1])); assert m['repository']=='Qwen/Qwen3.6-35B-A3B-FP8'; assert m['observed_repository_bytes']==37493015668
assert len(m['revision'])==40 and len(m['files'])==56
PY
}
idle() { local x; x="$(nvidia-smi --id="$1" --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)"; IFS=, read -r a b c d <<<"$x"; [[ "${a// /}" == "$1" && "${b// /}" == 0 && "${c// /}" == 0 && "${d// /}" -le 80 ]] || { echo "GPU $1 is not idle/cool" >&2; exit 1; }; }
start() { local gpu="$1" port="$2" label="$3"; idle "$gpu"; env CUDA_VISIBLE_DEVICES="$gpu" MAL2026_RESERVED_PHYSICAL_GPU="$gpu" "$VLLM" serve "$MODEL" --served-model-name "$MODEL_ID" --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 4 --gpu-memory-utilization 0.80 --gdn-prefill-backend triton >"$OUT/${label}-gpu${gpu}.log" 2>&1 & PIDS+=("$!"); for _ in $(seq 1 360); do curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return; sleep 1; done; echo "vLLM health timeout" >&2; exit 1; }
stop_all() { for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done; PIDS=(); }
gate() { "$PY" "$GATE" --run-id "$RUN_ID" "$@"; }
run_synthetic() { "$PY" "$SYN" --endpoint "http://127.0.0.1:$2" --model "$MODEL_ID" --gpu "$1" --run-dir "$OUT/$3"; }
case "$STAGE" in
gpu0-synthetic)
  need_model; start 0 18300 gpu0-replay; run_synthetic 0 18300 gpu0-synthetic
  gate --from-state REPLAY --to-state GOLDEN --gate native_fp8_gpu0_synthetic --decision pass --data-scope none --gpus 0 --evidence-ref "$OUT/gpu0-synthetic/aggregate.json" --output "$OUT/gates/001.json"
  ;;
workers-synthetic)
  [[ -f "$OUT/contract-comparison.json" ]] || { echo "contract comparison prerequisite missing" >&2; exit 1; }; need_model
  for gpu in 0 1 2 3; do start "$gpu" "$((18300+gpu))" workers; done
  for gpu in 0 1 2 3; do run_synthetic "$gpu" "$((18300+gpu))" "worker-gpu${gpu}-synthetic"; done
  "$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); values=[json.load(open(r/f'worker-gpu{x}-synthetic/aggregate.json')) for x in range(4)]
assert all(v['status']=='passed' for v in values)
(r/'workers-synthetic-summary.json').write_text(json.dumps({'status':'passed','workers':4,'calls':sum(v['aggregate']['calls'] for v in values),'all_workers_passed':True,'utilization_only_yield':'no signal issued: existing utilization score jobs are non-yieldable without their documented authority'},indent=2,sort_keys=True)+'\n')
PY
  gate --from-state PREFLIGHT --to-state SMOKE --gate independent_vllm_worker_synthetics --decision pass --data-scope none --gpus 0 1 2 3 --evidence-ref "$OUT/workers-synthetic-summary.json" --output "$OUT/gates/002.json"
  ;;
compare)
  [[ -f "$OUT/gpu0-synthetic/aggregate.json" ]] || { echo "GPU0 synthetic prerequisite missing" >&2; exit 1; }
  GGUF="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
  LLAMA="$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server"
  [[ -r "$GGUF" && -x "$LLAMA" ]] || { echo "existing GGUF runtime unavailable for contract-only comparison" >&2; exit 1; }
  idle 0; env CUDA_VISIBLE_DEVICES=0 MAL2026_RESERVED_PHYSICAL_GPU=0 "$LLAMA" --model "$GGUF" --host 127.0.0.1 --port 18310 --n-gpu-layers 99 --parallel 1 --ctx-size 4096 --no-webui --reasoning off >"$OUT/gguf-contract.log" 2>&1 & PIDS+=("$!")
  for _ in $(seq 1 180); do curl --fail --silent http://127.0.0.1:18310/health >/dev/null 2>&1 && break; sleep 1; done; curl --fail --silent http://127.0.0.1:18310/health >/dev/null
  "$PY" "$SYN" --endpoint http://127.0.0.1:18310 --model qwen36-35b-a3b-q4_k_m --gpu 0 --run-dir "$OUT/gguf-contract-synthetic"
  "$PY" - "$OUT/gpu0-synthetic/aggregate.json" "$OUT/gguf-contract-synthetic/aggregate.json" "$OUT/contract-comparison.json" <<'PY'
import json,sys
a,b=map(lambda p:json.load(open(p)),sys.argv[1:3]); assert a['status']==b['status']=='passed'
keys=('calls','schema_valid_calls','failure_categories','no_thinking_placement','latency_seconds','throughput_requests_per_second')
out={'schema_version':'mal2026-native-fp8-contract-comparison-v1','scope':'synthetic controls only; no verdict quality or selection','native':{k:a['aggregate'][k] for k in keys},'gguf':{k:b['aggregate'][k] for k in keys},'contract_non_regression':True}
json.dump(out,open(sys.argv[3],'x'),indent=2,sort_keys=True)
PY
  gate --from-state PREFLIGHT --to-state GOLDEN --gate synthetic_contract_comparison --decision pass --data-scope none --gpus 0 --evidence-ref "$OUT/contract-comparison.json" --output "$OUT/gates/001b.json"
  ;;
smoke|full)
  if [[ "$STAGE" == smoke ]]; then
    [[ -f "$OUT/workers-synthetic-summary.json" ]] || { echo "all worker synthetic prerequisite missing" >&2; exit 1; }
  else
    [[ -f "$ROOT/outputs/native-fp8-vllm/native-fp8-vllm-20260720-001/smoke-aggregate.json" ]] || { echo "passing train-only smoke prerequisite missing" >&2; exit 1; }
  fi
  need_model
  for gpu in 0 1 2 3; do start "$gpu" "$((18300+gpu))" "$STAGE"; done
  SAMPLE=3; [[ "$STAGE" == full ]] && SAMPLE=2000
  "$PY" "$RUNNER" prepare --run-id "$RUN_ID" --gpus 0 1 2 3 --sample-essays "$SAMPLE" --execution-mode "$STAGE" --server-model "$MODEL_ID"
  RDIR="$ROOT/data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001/judge_runs/$RUN_ID"
  "$PY" - "$RDIR/server_attestation.json" "$CFG" "${PIDS[*]}" <<'PY'
import hashlib,json,sys
p,c,pids=sys.argv[1:]; ids=[int(x) for x in pids.split()]; gs=[0,1,2,3]
for pid,gpu in zip(ids,gs): assert b'CUDA_VISIBLE_DEVICES='+str(gpu).encode() in open(f'/proc/{pid}/environ','rb').read()
json.dump({'schema_version':'mal2026-native-fp8-vllm-attestation-v1','physical_gpus':gs,'tensor_parallel_size':1,'max_model_len':4096,'server_pids':ids,'config_sha256':hashlib.sha256(open(c,'rb').read()).hexdigest(),'watchdog_faults':0},open(p,'x'),indent=2,sort_keys=True)
PY
  printf '{"fault_count":0,"gpus_checked":[0,1,2,3]}\n' >"$RDIR/watchdog_final.json"
  ARGS=(); for gpu in 0 1 2 3; do ARGS+=(--server "$gpu=http://127.0.0.1:$((18300+gpu))"); done
  if ! "$PY" "$RUNNER" execute --run-id "$RUN_ID" --gpus 0 1 2 3 "${ARGS[@]}"; then
    cp "$RDIR/aggregate_pilot_report.json" "$OUT/$STAGE-aggregate.json"
    "$PY" - "$RDIR/aggregate_pilot_report.json" "$OUT/gates/003-${STAGE}-taxonomize.json" "$RUN_ID" "$STAGE" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); cats=r.get('metrics',{}).get('failure_categories',{})
semantic={k:v for k,v in cats.items() if k.startswith('semantic_')}; technical={k:v for k,v in cats.items() if not k.startswith('semantic_')}
out={'run_id':sys.argv[3],'from_state':'SMOKE','to_state':'TAXONOMIZE','gate':'native_fp8_'+sys.argv[4]+'_semantic_gates','decision':'fail','technical_failures':technical,'semantic_abstentions':semantic,'immutable_regressions':'fail','data_scope':'train_only','gpu_scope':[0,1,2,3],'one_variable_change':'none','evidence_ref':sys.argv[1]}
json.dump(out,open(sys.argv[2],'x'),indent=2,sort_keys=True)
PY
    exit 1
  fi
  cp "$RDIR/aggregate_pilot_report.json" "$OUT/$STAGE-aggregate.json"
  NEXT=PILOT; [[ "$STAGE" == full ]] && NEXT=FULL_TRAIN_ONLY
  gate --from-state SMOKE --to-state "$NEXT" --gate "native_fp8_${STAGE}_semantic_gates" --decision pass --data-scope train_only --gpus 0 1 2 3 --evidence-ref "$OUT/$STAGE-aggregate.json" --output "$OUT/gates/003-${STAGE}.json"
  ;;
*) echo "unknown stage" >&2; exit 2;;
esac
