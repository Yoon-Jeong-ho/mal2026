#!/usr/bin/env bash
# Run the bounded v2 pilot end-to-end in an owned tmux/scheduler job.
# It intentionally has no selection, SFT, DPO, or GRPO stage.
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 BATCH_RUN_ID JUDGE_RUN_ID MODEL_GGUF LLAMA_SERVER [PORT]" >&2
  exit 2
fi

BATCH_RUN_ID="$1"
JUDGE_RUN_ID="$2"
MODEL_GGUF="$3"
LLAMA_SERVER="$4"
PORT="${5:-18084}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv-standard/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python"
RUN_DIR="$ROOT/data/processed/restricted/openai_rationale_batches/$BATCH_RUN_ID/judge_runs/$JUDGE_RUN_ID"

[[ -r "$MODEL_GGUF" ]] || { echo "configured GGUF is unreadable" >&2; exit 1; }
[[ -x "$LLAMA_SERVER" ]] || { echo "configured llama-server is not executable" >&2; exit 1; }

# Address only physical GPU 0.  A query failure or non-idle device is a stop,
# not permission to broaden discovery to any other GPU.
GPU0_INFO="$(nvidia-smi --id=0 --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader)"
GPU0_USED="$(awk -F, '{gsub(/ MiB/, "", $4); gsub(/ /, "", $4); print $4}' <<<"$GPU0_INFO")"
if [[ "$GPU0_USED" != "0" ]]; then
  echo "physical GPU 0 is not idle; v2 pilot not started" >&2
  exit 1
fi

cd "$ROOT"
"$PYTHON" scripts/judge_feedback_candidates_v2.py prepare \
  --batch-run-id "$BATCH_RUN_ID" --judge-run-id "$JUDGE_RUN_ID"

mkdir -p "$RUN_DIR/logs"
"$PYTHON" - "$RUN_DIR/runner_provenance.json" "$GPU0_INFO" "$PORT" <<'PY'
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
path, gpu0, port = map(str, sys.argv[1:])
record = {
    "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "command": "scripts/run_qwen36_judge_v2_pilot.sh (prepare -> synthetic smoke -> execute)",
    "cuda_visible_devices": "0",
    "physical_gpu": 0,
    "gpu0_observation": gpu0,
    "server_port": int(port),
    "selection_artifact_constructed": False,
}
Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

env CUDA_VISIBLE_DEVICES=0 "$LLAMA_SERVER" \
  --model "$MODEL_GGUF" --host 127.0.0.1 --port "$PORT" \
  --n-gpu-layers 99 --parallel 1 --ctx-size 4096 --no-webui --reasoning off \
  >"$RUN_DIR/logs/llama-server-gpu0.log" 2>&1 &
SERVER_PID="$!"
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 180); do
  if curl --fail --silent --show-error "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl --fail --silent --show-error "http://127.0.0.1:$PORT/health" >/dev/null

SERVER_CVD="$(tr '\0' '\n' <"/proc/$SERVER_PID/environ" | awk -F= '$1 == "CUDA_VISIBLE_DEVICES" {print $2}')"
[[ "$SERVER_CVD" == "0" ]] || { echo "owned server GPU-0 environment attestation failed" >&2; exit 1; }
"$PYTHON" - "$RUN_DIR/server_attestation.json" "$PORT" "$SERVER_PID" "$SERVER_CVD" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path.cwd()
path, port, pid, cvd = sys.argv[1:]
server_exe = Path(f"/proc/{pid}/exe").resolve()
record = {
    "schema_version": "qwen36-gguf-judge-v2-server-attestation-v1",
    "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "server_host": "127.0.0.1",
    "server_port": int(port),
    "physical_gpu": 0,
    "cuda_visible_devices": cvd,
    "server_pid": int(pid),
    "server_executable_sha256": hashlib.sha256(server_exe.read_bytes()).hexdigest(),
    "config_sha256": hashlib.sha256((root / "configs/qwen36_gguf_judge.v2.pilot.json").read_bytes()).hexdigest(),
    "server_process_environment_verified": True,
}
Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

env CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/judge_feedback_candidates_v2.py smoke \
  --batch-run-id "$BATCH_RUN_ID" --judge-run-id "$JUDGE_RUN_ID" --server "http://127.0.0.1:$PORT" \
  --server-attestation "$RUN_DIR/server_attestation.json"
env CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/judge_feedback_candidates_v2.py execute \
  --batch-run-id "$BATCH_RUN_ID" --judge-run-id "$JUDGE_RUN_ID" --server "http://127.0.0.1:$PORT" \
  --server-attestation "$RUN_DIR/server_attestation.json"
