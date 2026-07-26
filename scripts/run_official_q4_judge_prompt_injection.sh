#!/usr/bin/env bash
# Launch the exact frozen Q4 proxy judge for one train-only prompt-injection variant.
set -Eeuo pipefail

[[ $# -eq 3 ]] || { echo "usage: $0 RUN_ID CASE_FILE EXPECTED" >&2; exit 2; }
RUN_ID="$1"; CASE_FILE="$2"; EXPECTED="$3"
[[ "$EXPECTED" -ge 8 && "$EXPECTED" -le 400 ]] || { echo "injection audit must contain 8--400 train rows" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="$ROOT/.venv-standard/bin/python"
SERVER="$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server"
REPO="$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/llama.cpp"
MODEL="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
MODEL_SHA="b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772"
REVISION="571d0d540df04f25298d0e159e520d9fc62ed121"; TAG="b10068"
CONTRACT="$ROOT/src/mal2026/official_writing_contract.py"
CONTRACT_SHA="7b04149227a44852ca78bd65f5ec70245b284503256374debf2735f17ca69e50"
GPUS=(0 1 2 3); PORTS=(19100 19101 19102 19103)
RUN_ROOT="$ROOT/outputs/official-prompt-alignment-v1/q4-judge-runtime/$RUN_ID"
ATTESTATION="$RUN_ROOT/server_attestation.json"

[[ -x "$PY" && -x "$SERVER" && -r "$MODEL" && -r "$CASE_FILE" ]] || { echo "injection judge prerequisite failed" >&2; exit 1; }
[[ ! -e "$RUN_ROOT" ]] || { echo "injection judge runtime output exists" >&2; exit 1; }
[[ "$(sha256sum "$MODEL" | awk '{print $1}')" == "$MODEL_SHA" ]] || { echo "official GGUF checksum differs" >&2; exit 1; }
[[ "$(sha256sum "$CONTRACT" | awk '{print $1}')" == "$CONTRACT_SHA" ]] || { echo "frozen judge prompt contract changed" >&2; exit 1; }
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$REVISION" && "$(git -C "$REPO" describe --tags --exact-match)" == "$TAG" ]] || { echo "official llama.cpp revision differs" >&2; exit 1; }
for gpu in "${GPUS[@]}"; do
  line="$(nvidia-smi --id="$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  IFS=, read -r index used util <<<"$line"
  [[ "${index// /}" == "$gpu" && "${used// /}" == 0 && "${util// /}" == 0 ]] || { echo "GPU $gpu is not idle" >&2; exit 1; }
done

mkdir -p "$RUN_ROOT/logs"; PIDS=()
cleanup() {
  rc=$?
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  exit "$rc"
}
trap cleanup EXIT INT TERM
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"; port="${PORTS[$i]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$SERVER" --model "$MODEL" --host 127.0.0.1 --port "$port" \
    --n-gpu-layers 99 --parallel 4 --ctx-size 32768 --batch-size 2048 --ubatch-size 512 \
    --no-webui --reasoning off >"$RUN_ROOT/logs/llama-server-gpu$gpu.log" 2>&1 &
  PIDS+=("$!")
done
for port in "${PORTS[@]}"; do
  healthy=0
  for _ in $(seq 1 240); do
    if curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null 2>&1; then healthy=1; break; fi
    sleep 1
  done
  [[ "$healthy" == 1 ]] || { echo "injection judge server health timeout on $port" >&2; exit 1; }
done

"$PY" - "$ATTESTATION" "$MODEL" "$SERVER" "$REPO" "$MODEL_SHA" "$REVISION" "$TAG" "${GPUS[*]}" "${PORTS[*]}" "${PIDS[*]}" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
from urllib.request import urlopen
out,model,server,repo,model_sha,revision,tag,gpus,ports,pids=sys.argv[1:]
gs=[int(x) for x in gpus.split()]; ps=[int(x) for x in ports.split()]; ids=[int(x) for x in pids.split()]
assert hashlib.sha256(Path(model).read_bytes()).hexdigest()==model_sha
for gpu,port,pid in zip(gs,ps,ids,strict=True):
    env=Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    visible=next(x.split(b'=',1)[1].decode() for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES='))
    props=json.loads(urlopen(f'http://127.0.0.1:{port}/props',timeout=5).read().decode())
    assert visible==str(gpu) and props.get('total_slots')==4
    assert props.get('default_generation_settings',{}).get('n_ctx')==8192
value={'schema_version':'mal2026-official-q4-judge-server-attestation-v1','physical_gpus':gs,'server_endpoints':[f'http://127.0.0.1:{p}' for p in ps],'server_pids':ids,'parallel_per_server':4,'context_per_slot':8192,'model_sha256':model_sha,'llama_server_sha256':hashlib.sha256(Path(server).read_bytes()).hexdigest(),'llama_revision':subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip(),'llama_tag':subprocess.check_output(['git','-C',repo,'describe','--tags','--exact-match'],text=True).strip()}
Path(out).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY

ARGS=(); for port in "${PORTS[@]}"; do ARGS+=(--endpoint "http://127.0.0.1:$port"); done
PYTHONPATH="$ROOT/src" "$PY" "$ROOT/scripts/evaluate_official_q4_judge_prompt_injection.py" \
  --run-id "$RUN_ID" --case-file "$CASE_FILE" --expected "$EXPECTED" --max-inflight 16 \
  --server-attestation "$ATTESTATION" "${ARGS[@]}"
