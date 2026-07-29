#!/usr/bin/env bash
# Launch exact Q4_K_M judge replicas on GPU0 (smoke) or GPUs 0--3 (full).
set -Eeuo pipefail

[[ $# -eq 5 || $# -eq 6 ]] || { echo "usage: $0 MODE(smoke|audit|full) RUN_ID SPLIT PARTICIPANT_FILE EXPECTED [SYSTEM_PROMPT_FILE]" >&2; exit 2; }
MODE="$1"; RUN_ID="$2"; SPLIT="$3"; PARTICIPANT_FILE="$4"; EXPECTED="$5"
SYSTEM_PROMPT_FILE="${6:-}"
[[ "$MODE" == smoke || "$MODE" == audit || "$MODE" == full ]] || { echo "mode must be smoke, audit, or full" >&2; exit 2; }
[[ "$SPLIT" == train || "$SPLIT" == validation ]] || { echo "split differs" >&2; exit 2; }
if [[ "$MODE" == smoke ]]; then GPUS=(0); PORTS=(19100); [[ "$SPLIT" == train && "$EXPECTED" -le 4 ]] || { echo "smoke must be train-only and at most four rows" >&2; exit 2; }
elif [[ "$MODE" == audit ]]; then GPUS=(0 1 2 3); PORTS=(19100 19101 19102 19103); [[ "$SPLIT" == train && "$EXPECTED" -ge 8 && "$EXPECTED" -le 400 ]] || { echo "audit must be train-only with 8--400 rows" >&2; exit 2; }
else GPUS=(0 1 2 3); PORTS=(19100 19101 19102 19103); [[ "$SPLIT" == validation && "$EXPECTED" == 400 ]] || { echo "full judge must be the 400-row validation run" >&2; exit 2; }; fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="$ROOT/.venv-standard/bin/python"
SERVER="$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server"
REPO="$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/llama.cpp"
MODEL="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
MODEL_SHA="b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772"
REVISION="571d0d540df04f25298d0e159e520d9fc62ed121"; TAG="b10068"
RUN_ROOT="$ROOT/outputs/official-prompt-alignment-v1/q4-judge-runtime/$RUN_ID"
ATTESTATION="$RUN_ROOT/server_attestation.json"

[[ -x "$PY" && -x "$SERVER" && -r "$MODEL" && -r "$PARTICIPANT_FILE" ]] || { echo "official judge prerequisite failed" >&2; exit 1; }
[[ ! -e "$RUN_ROOT" ]] || { echo "official judge runtime output exists" >&2; exit 1; }
[[ "$(sha256sum "$MODEL" | awk '{print $1}')" == "$MODEL_SHA" ]] || { echo "official GGUF checksum differs" >&2; exit 1; }
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$REVISION" && "$(git -C "$REPO" describe --tags --exact-match)" == "$TAG" ]] || { echo "official llama.cpp revision differs" >&2; exit 1; }
if [[ -n "$SYSTEM_PROMPT_FILE" ]]; then
  [[ -r "$SYSTEM_PROMPT_FILE" ]] || { echo "system prompt file is unavailable" >&2; exit 1; }
  JUDGE_PROMPT_SHA="$(sha256sum "$SYSTEM_PROMPT_FILE" | awk '{print $1}')"
else
  JUDGE_PROMPT_SHA="$(PYTHONPATH="$ROOT/src" "$PY" - <<'PY'
from hashlib import sha256
from mal2026.official_writing_contract import FROZEN_PROXY_JUDGE_SYSTEM_PROMPT
print(sha256(FROZEN_PROXY_JUDGE_SYSTEM_PROMPT.encode('utf-8')).hexdigest())
PY
)"
fi
for gpu in "${GPUS[@]}"; do
  line="$(nvidia-smi --id="$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  IFS=, read -r index used util <<<"$line"
  [[ "${index// /}" == "$gpu" && "${used// /}" == 0 && "${util// /}" == 0 ]] || { echo "GPU $gpu is not idle" >&2; exit 1; }
done

mkdir -p "$RUN_ROOT/logs"
PIDS=()
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
  [[ "$healthy" == 1 ]] || { echo "official judge server health timeout on $port" >&2; exit 1; }
done

"$PY" - "$ATTESTATION" "$MODEL" "$SERVER" "$REPO" "$MODEL_SHA" "$REVISION" "$TAG" "$JUDGE_PROMPT_SHA" "${GPUS[*]}" "${PORTS[*]}" "${PIDS[*]}" <<'PY'
import hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import urlopen
out,model,server,repo,model_sha,revision,tag,judge_prompt_sha,gpus,ports,pids=sys.argv[1:]
gs=[int(x) for x in gpus.split()]; ps=[int(x) for x in ports.split()]; ids=[int(x) for x in pids.split()]
assert hashlib.sha256(Path(model).read_bytes()).hexdigest()==model_sha
for gpu,port,pid in zip(gs,ps,ids,strict=True):
    env=Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    visible=next(x.split(b'=',1)[1].decode() for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES='))
    props=json.loads(urlopen(f'http://127.0.0.1:{port}/props',timeout=5).read().decode())
    assert visible==str(gpu) and props.get('total_slots')==4
    assert props.get('default_generation_settings',{}).get('n_ctx')==8192
value={'schema_version':'mal2026-official-q4-judge-server-attestation-v1','created_at':datetime.now(timezone.utc).replace(microsecond=0).isoformat(),'physical_gpus':gs,'server_endpoints':[f'http://127.0.0.1:{p}' for p in ps],'server_pids':ids,'parallel_per_server':4,'context_per_slot':8192,'model_sha256':model_sha,'llama_server_sha256':hashlib.sha256(Path(server).read_bytes()).hexdigest(),'llama_revision':subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip(),'llama_tag':subprocess.check_output(['git','-C',repo,'describe','--tags','--exact-match'],text=True).strip(),'judge_prompt_sha256':judge_prompt_sha}
Path(out).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY

ARGS=()
for port in "${PORTS[@]}"; do ARGS+=(--endpoint "http://127.0.0.1:$port"); done
if [[ -n "$SYSTEM_PROMPT_FILE" ]]; then
  ARGS+=(--system-prompt-file "$SYSTEM_PROMPT_FILE")
fi
PYTHONPATH="$ROOT/src" "$PY" "$ROOT/scripts/evaluate_official_q4_judge.py" \
  --run-id "$RUN_ID" --split "$SPLIT" --participant-file "$PARTICIPANT_FILE" --expected "$EXPECTED" \
  --max-inflight "$((4 * ${#GPUS[@]}))" --server-attestation "$ATTESTATION" "${ARGS[@]}"
