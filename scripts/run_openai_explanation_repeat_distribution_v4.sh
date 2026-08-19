#!/usr/bin/env bash
# Isolated v4 repeat-distribution pilot.  Only physical GPUs 4-7 are queried or used.
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 RUN_ID" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="$ROOT/.venv-standard/bin/python"; CONFIG="$ROOT/configs/openai_explanation_repeat_distribution.v4.pilot.json"
MODEL="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
SERVER="$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server"
LLAMA_REPO="$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/llama.cpp"
RUN_ID="$1"; RUN_DIR="$ROOT/data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001/judge_runs/$RUN_ID"
[[ -x "$PYTHON" && -x "$SERVER" && -r "$MODEL" && ! -e "$RUN_DIR" ]] || { echo "runtime prerequisite or clean run-dir gate failed" >&2; exit 1; }
[[ "$(stat -c %s "$MODEL")" == "22285080192" && "$(sha256sum "$MODEL" | awk '{print $1}')" == "b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772" ]] || { echo "pinned Q4_K_M gate failed" >&2; exit 1; }
[[ "$(git -C "$LLAMA_REPO" rev-parse HEAD)" == "571d0d540df04f25298d0e159e520d9fc62ed121" && "$(git -C "$LLAMA_REPO" describe --tags --exact-match)" == "b10068" ]] || { echo "pinned llama.cpp revision gate failed" >&2; exit 1; }
for GPU in 4 5 6 7; do
  LINE="$(nvidia-smi --id="$GPU" --query-gpu=index,memory.total,memory.used,temperature.gpu,utilization.gpu --format=csv,noheader,nounits)"
  IFS=, read -r INDEX TOTAL USED TEMP UTIL <<<"$LINE"; INDEX="${INDEX// /}"; USED="${USED// /}"; TEMP="${TEMP// /}"; UTIL="${UTIL// /}"
  [[ "$INDEX" == "$GPU" && "$USED" == 0 && "$UTIL" == 0 && "$TEMP" -le 80 ]] || { echo "GPU $GPU is not an idle/cool pilot resource" >&2; exit 1; }
done
export MAL2026_REPEAT_CONFIG="$CONFIG"
cd "$ROOT"; "$PYTHON" scripts/run_openai_explanation_repeat_distribution_v4.py prepare --run-id "$RUN_ID"
mkdir -p "$RUN_DIR/logs"
PORTS=(18084 18085 18086 18087); PIDS=(); WATCHDOG_PID=""
cleanup() { local status=$?; for PID in "${PIDS[@]:-}"; do kill "$PID" 2>/dev/null || true; done; for PID in "${PIDS[@]:-}"; do wait "$PID" 2>/dev/null || true; done; [[ -n "$WATCHDOG_PID" ]] && { kill "$WATCHDOG_PID" 2>/dev/null || true; wait "$WATCHDOG_PID" 2>/dev/null || true; }; exit "$status"; }
trap cleanup EXIT INT TERM
for OFFSET in 0 1 2 3; do
  GPU=$((4 + OFFSET)); PORT="${PORTS[$OFFSET]}"
  env CUDA_VISIBLE_DEVICES="$GPU" "$SERVER" --model "$MODEL" --host 127.0.0.1 --port "$PORT" --n-gpu-layers 99 --parallel 4 --ctx-size 4096 --no-webui --reasoning off >"$RUN_DIR/logs/llama-server-gpu$GPU.log" 2>&1 & PIDS+=("$!")
done
for OFFSET in 0 1 2 3; do
  PORT="${PORTS[$OFFSET]}"; for _ in $(seq 1 180); do curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done; curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null
done
for OFFSET in 0 1 2 3; do
  GPU=$((4 + OFFSET)); SERVER_CVD="$(tr '\0' '\n' <"/proc/${PIDS[$OFFSET]}/environ" | awk -F= '$1 == "CUDA_VISIBLE_DEVICES" {print $2}')"
  [[ "$SERVER_CVD" == "$GPU" ]] || { echo "GPU visibility attestation failed" >&2; exit 1; }
done
"$PYTHON" - "$RUN_DIR/server_attestation.json" "$CONFIG" "${PIDS[@]}" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
path,config,*pids=sys.argv[1:]
Path(path).write_text(json.dumps({"schema_version":"mal2026-repeat-server-attestation-v1","created_at":datetime.now(timezone.utc).replace(microsecond=0).isoformat(),"physical_gpus":[4,5,6,7],"parallel_requests_per_server":4,"server_host":"127.0.0.1","server_ports":[18084,18085,18086,18087],"server_pids":[int(x) for x in pids],"config_sha256":hashlib.sha256(Path(config).read_bytes()).hexdigest(),"watchdog_faults":0},sort_keys=True,indent=2)+"\n")
PY
"$PYTHON" - "$RUN_DIR/watchdog_final.json" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"finished_at":datetime.now(timezone.utc).replace(microsecond=0).isoformat(),"fault_count":0,"gpus_checked":[4,5,6,7]},sort_keys=True,indent=2)+"\n")
PY
(
  FAULTS=0
  while :; do
    for GPU in 4 5 6 7; do
      LINE="$(nvidia-smi --id="$GPU" --query-gpu=index,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true)"; IFS=, read -r INDEX TOTAL USED TEMP <<<"$LINE"; TOTAL="${TOTAL// /}"; USED="${USED// /}"; TEMP="${TEMP// /}"
      if [[ -z "$TOTAL" || -z "$USED" || -z "$TEMP" || "$TEMP" -gt 85 || "$USED" -gt $((TOTAL * 75 / 100)) ]]; then FAULTS=$((FAULTS + 1)); for PID in "${PIDS[@]}"; do kill "$PID" 2>/dev/null || true; done; break 2; fi
    done
    sleep 5
  done
  "$PYTHON" - "$RUN_DIR/watchdog_final.json" "$FAULTS" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"finished_at":datetime.now(timezone.utc).replace(microsecond=0).isoformat(),"fault_count":int(sys.argv[2]),"gpus_checked":[4,5,6,7]},sort_keys=True,indent=2)+"\n")
PY
) &
WATCHDOG_PID="$!"
"$PYTHON" scripts/run_openai_explanation_repeat_distribution_v4.py execute --run-id "$RUN_ID" --server 4=http://127.0.0.1:18084 --server 5=http://127.0.0.1:18085 --server 6=http://127.0.0.1:18086 --server 7=http://127.0.0.1:18087
kill "$WATCHDOG_PID" 2>/dev/null || true; wait "$WATCHDOG_PID" 2>/dev/null || true; WATCHDOG_PID=""
"$PYTHON" - "$RUN_DIR/watchdog_final.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); value=json.loads(p.read_text()); assert value["fault_count"]==0
PY
