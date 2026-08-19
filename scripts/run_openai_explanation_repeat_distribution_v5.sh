#!/usr/bin/env bash
# V5 local runner. It can address only debug GPUs 4-7 and never exposes raw payloads.
set -Eeuo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 RUN_ID MODE(smoke|pilot) SAMPLE_ESSAYS" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"; RUN_ID="$1"; MODE="$2"; SAMPLE="$3"
PYTHON="$ROOT/.venv-standard/bin/python"; CONFIG="${MAL2026_REPEAT_CONFIG:-$ROOT/configs/openai_explanation_repeat_distribution.v5.pilot.json}"; RUNNER="${MAL2026_REPEAT_RUNNER:-$ROOT/scripts/run_openai_explanation_repeat_distribution_v5.py}"; MODEL="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"; SERVER="$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server"; LLAMA_REPO="$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/llama.cpp"
[[ "$MODE" == smoke && "$SAMPLE" == 3 || "$MODE" == pilot && "$SAMPLE" == 96 ]] || { echo "v5 permits only 3-essay smoke or 96-essay capped pilot" >&2; exit 2; }
[[ -x "$PYTHON" && -x "$SERVER" && -r "$MODEL" ]] || { echo "v5 runtime prerequisite failed" >&2; exit 1; }
[[ "$(stat -c %s "$MODEL")" == 22285080192 && "$(sha256sum "$MODEL" | awk '{print $1}')" == b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772 && "$(git -C "$LLAMA_REPO" rev-parse HEAD)" == 571d0d540df04f25298d0e159e520d9fc62ed121 ]] || { echo "v5 pinned-runtime gate failed" >&2; exit 1; }
for GPU in 4 5 6 7; do line="$(nvidia-smi --id="$GPU" --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)"; IFS=, read -r index used util temp <<<"$line"; [[ "${index// /}" == "$GPU" && "${used// /}" == 0 && "${util// /}" == 0 && "${temp// /}" -le 80 ]] || { echo "debug GPU preflight failed" >&2; exit 1; }; done
export MAL2026_REPEAT_CONFIG="$CONFIG"; cd "$ROOT"; "$PYTHON" "$RUNNER" prepare --run-id "$RUN_ID" --sample-essays "$SAMPLE" --execution-mode "$MODE"
RUN_DIR="$ROOT/data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001/judge_runs/$RUN_ID"; mkdir -p "$RUN_DIR/logs"; PIDS=(); WATCHDOG=""
cleanup() { rc=$?; for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done; [[ -n "$WATCHDOG" ]] && { kill "$WATCHDOG" 2>/dev/null || true; wait "$WATCHDOG" 2>/dev/null || true; }; exit "$rc"; }; trap cleanup EXIT INT TERM
for offset in 0 1 2 3; do gpu=$((4+offset)); port=$((18184+offset)); env CUDA_VISIBLE_DEVICES="$gpu" "$SERVER" --model "$MODEL" --host 127.0.0.1 --port "$port" --n-gpu-layers 99 --parallel 1 --ctx-size 4096 --no-webui --reasoning off >"$RUN_DIR/logs/llama-server-gpu$gpu.log" 2>&1 & PIDS+=("$!"); done
for port in 18184 18185 18186 18187; do for _ in $(seq 1 180); do curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done; curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null; done
"$PYTHON" - "$RUN_DIR/server_attestation.json" "$CONFIG" "${PIDS[@]}" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import urlopen
p,c,*pids=sys.argv[1:]; pids=[int(v) for v in pids]
for gpu,port,pid in zip([4,5,6,7],[18184,18185,18186,18187],pids):
    env=Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    visible=next((x.split(b"=",1)[1].decode() for x in env if x.startswith(b"CUDA_VISIBLE_DEVICES=")), "")
    props=json.loads(urlopen(f"http://127.0.0.1:{port}/props",timeout=5).read().decode())
    assert visible == str(gpu) and props.get("total_slots") == 1 and props.get("default_generation_settings",{}).get("n_ctx") == 4096
Path(p).write_text(json.dumps({"schema_version":"mal2026-repeat-v5-server-attestation-v1","created_at":datetime.now(timezone.utc).replace(microsecond=0).isoformat(),"physical_gpus":[4,5,6,7],"parallel_requests_per_server":1,"slot_context":4096,"server_host":"127.0.0.1","server_ports":[18184,18185,18186,18187],"server_pids":pids,"config_sha256":hashlib.sha256(Path(c).read_bytes()).hexdigest()},sort_keys=True,indent=2)+"\n")
PY
"$PYTHON" - "$RUN_DIR/watchdog_final.json" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"fault_count":0,"gpus_checked":[4,5,6,7]},sort_keys=True)+"\n")
PY
"$PYTHON" "$RUNNER" execute --run-id "$RUN_ID" --server 4=http://127.0.0.1:18184 --server 5=http://127.0.0.1:18185 --server 6=http://127.0.0.1:18186 --server 7=http://127.0.0.1:18187
