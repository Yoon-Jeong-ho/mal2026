#!/usr/bin/env bash
# Versioned, capped train-only v3 remediation pilot; no selection/SFT/DPO/GRPO.
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 BATCH JUDGE MODEL SERVER PORT" >&2; exit 2; }
BATCH="$1"; JUDGE="$2"; MODEL="$3"; SERVER="$4"; PORT="$5"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PYTHON="$ROOT/.venv-standard/bin/python"
RUN_DIR="$ROOT/data/processed/restricted/openai_rationale_batches/$BATCH/judge_runs/$JUDGE"
[[ -x "$PYTHON" && -r "$MODEL" && -x "$SERVER" ]] || { echo "v3 runtime prerequisite absent" >&2; exit 1; }
GPU0_INFO="$(nvidia-smi --id=0 --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader)"
GPU0_USED="$(awk -F, '{gsub(/ MiB/, "", $4); gsub(/ /, "", $4); print $4}' <<<"$GPU0_INFO")"
[[ "$GPU0_USED" == "0" ]] || { echo "physical GPU 0 is not idle; v3 not started" >&2; exit 1; }
cd "$ROOT"
export MAL2026_JUDGE_CONFIG="$ROOT/configs/qwen36_gguf_judge.v3.pilot.json"
export MAL2026_JUDGE_SCHEMA="qwen36-gguf-judge-v3-pilot"
"$PYTHON" scripts/judge_feedback_candidates_v2.py prepare --batch-run-id "$BATCH" --judge-run-id "$JUDGE"
mkdir -p "$RUN_DIR/logs"
"$PYTHON" - "$RUN_DIR/runner_provenance.json" "$GPU0_INFO" "$PORT" <<'PY'
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
path, gpu0, port = sys.argv[1:]
Path(path).write_text(json.dumps({"created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "command": "scripts/run_qwen36_judge_v3_pilot.sh (prepare -> synthetic smoke -> execute; parallel=4)", "cuda_visible_devices": "0", "physical_gpu": 0, "gpu0_observation": gpu0, "server_port": int(port), "server_parallel": 4, "selection_artifact_constructed": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
env CUDA_VISIBLE_DEVICES=0 "$SERVER" --model "$MODEL" --host 127.0.0.1 --port "$PORT" --n-gpu-layers 99 --parallel 4 --ctx-size 4096 --no-webui --reasoning off >"$RUN_DIR/logs/llama-server-gpu0.log" 2>&1 &
PID="$!"; cleanup() { kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; }; trap cleanup EXIT
for _ in $(seq 1 180); do curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null
SERVER_CVD="$(tr '\0' '\n' <"/proc/$PID/environ" | awk -F= '$1 == "CUDA_VISIBLE_DEVICES" {print $2}')"
[[ "$SERVER_CVD" == "0" ]] || { echo "GPU-0 server attestation failed" >&2; exit 1; }
"$PYTHON" - "$RUN_DIR/server_attestation.json" "$PORT" "$PID" "$SERVER_CVD" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path, port, pid, cvd = sys.argv[1:]
root = Path.cwd(); exe = Path(f"/proc/{pid}/exe").resolve()
Path(path).write_text(json.dumps({"schema_version": "qwen36-gguf-judge-v2-server-attestation-v1", "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), "server_host": "127.0.0.1", "server_port": int(port), "physical_gpu": 0, "cuda_visible_devices": cvd, "server_pid": int(pid), "server_executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(), "config_sha256": hashlib.sha256((root / "configs/qwen36_gguf_judge.v3.pilot.json").read_bytes()).hexdigest(), "server_process_environment_verified": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
env CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/judge_feedback_candidates_v2.py smoke --batch-run-id "$BATCH" --judge-run-id "$JUDGE" --server "http://127.0.0.1:$PORT" --server-attestation "$RUN_DIR/server_attestation.json"
env CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/judge_feedback_candidates_v2.py execute --batch-run-id "$BATCH" --judge-run-id "$JUDGE" --server "http://127.0.0.1:$PORT" --server-attestation "$RUN_DIR/server_attestation.json"
