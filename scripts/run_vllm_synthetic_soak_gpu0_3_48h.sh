#!/usr/bin/env bash
# Durable 24/48/120-hour, aggregate-only synthetic vLLM soak on physical GPUs 0--3.
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 RUN_ID" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PATH="$ROOT/.venv-standard/bin:$PATH"
RUN_ID="$1"
[[ "$RUN_ID" =~ ^vllm-soak-gpu0-3-(24h|48h|120h)-[0-9]{8}-[0-9]{3}$ ]] || { echo "invalid run ID" >&2; exit 2; }
PY="$ROOT/.venv-standard/bin/python"
VLLM="$ROOT/scripts/run_vllm_with_title.py"
CFG="${MAL2026_VLLM_SOAK_CONFIG:-$ROOT/configs/vllm_synthetic_soak_gpu0_3_48h.v1.json}"
CLIENT="$ROOT/scripts/run_vllm_synthetic_soak.py"
MODEL="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-FP8-native-fp8-v1"
MODEL_ID="Qwen/Qwen3.6-35B-A3B-FP8"
OUT="$ROOT/outputs/legacy/vllm-synthetic-soak/$RUN_ID"
PORT=18360
PID=""

[[ -x "$PY" && -r "$VLLM" && -r "$CFG" && -r "$CLIENT" && -d "$MODEL" ]] || { echo "canonical local runtime is unavailable" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
"$PY" - "$CFG" "$MODEL" "$RUN_ID" <<'PY'
import json, pathlib, sys, vllm
c=json.load(open(sys.argv[1])); r=c["runtime"]
durations={"mal2026-vllm-synthetic-soak-gpu0-3-48h-v1":172800,"mal2026-vllm-synthetic-soak-gpu0-3-24h-v1":86400,"mal2026-vllm-synthetic-soak-gpu0-3-120h-v1":432000}
assert c["schema_version"] in durations and r["duration_seconds"]==durations[c["schema_version"]]
assert vllm.__version__==r["required_vllm_version"]=="0.25.1"
assert r["physical_gpus"]==[0,1,2,3]
assert r["data_parallel_size"]==4 and r["tensor_parallel_size"]==1
assert pathlib.Path(sys.argv[2],"config.json").is_file()
assert f"-{r['duration_seconds']//3600}h-" in sys.argv[3]
PY
for gpu in 0 1 2 3; do
  line="$(nvidia-smi --id="$gpu" --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)"
  IFS=, read -r index memory util temp <<<"$line"
  [[ "${index// /}" == "$gpu" && "${memory// /}" == 0 && "${util// /}" == 0 && "${temp// /}" -le 80 ]] || {
    echo "GPU $gpu is not idle/cool: $line" >&2; exit 1;
  }
done
"$PY" - "$PORT" <<'PY'
import socket,sys
port=int(sys.argv[1])
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    if sock.connect_ex(("127.0.0.1",port)) == 0:
        raise SystemExit(f"localhost port {port} is occupied")
PY
mkdir -p "$OUT"
cleanup() {
  rc=$?
  [[ -z "$PID" ]] || kill "$PID" 2>/dev/null || true
  [[ -z "$PID" ]] || wait "$PID" 2>/dev/null || true
  printf '%s | %s | exit=%s\n' "$(date --iso-8601=seconds)" "$RUN_ID" "$rc" >>"$OUT/append_only_ledger.log"
  exit "$rc"
}
trap cleanup EXIT INT TERM
{
  printf '%s | %s | launch | GPUs 0-3 explicitly authorized; duration bound by config | git=%s | config_sha256=%s\n' \
    "$(date --iso-8601=seconds)" "$RUN_ID" "$(git -C "$ROOT" rev-parse HEAD)" "$(sha256sum "$CFG" | awk '{print $1}')"
} >>"$OUT/append_only_ledger.log"
env SPT_NOENV=1 CUDA_VISIBLE_DEVICES=0,1,2,3 MAL2026_RESERVED_PHYSICAL_GPUS=0,1,2,3 \
  "$PY" "$VLLM" serve "$MODEL" \
  --served-model-name "$MODEL_ID" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 --data-parallel-size 4 \
  --max-model-len 4096 --max-num-seqs 192 --max-num-batched-tokens 65536 \
  --gpu-memory-utilization 0.90 --gdn-prefill-backend triton \
  --generation-config vllm --enable-prefix-caching \
  >"$OUT/server.log" 2>&1 &
PID="$!"
for _ in $(seq 1 600); do
  curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$PID" 2>/dev/null || { tail -100 "$OUT/server.log" >&2; exit 1; }
  sleep 1
done
curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null
"$PY" - "$PID" "$OUT/server_attestation.json" "$CFG" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
pid,out,cfg=sys.argv[1:]
environment=Path(f"/proc/{pid}/environ").read_bytes()
assert b"CUDA_VISIBLE_DEVICES=0,1,2,3" in environment
Path(out).write_text(json.dumps({
  "schema_version":"mal2026-vllm-synthetic-soak-server-attestation-v1",
  "created_at":datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
  "physical_gpus":[0,1,2,3],"tensor_parallel_size":1,"data_parallel_size":4,
  "server_pid":int(pid),"server_host":"127.0.0.1","server_port":18360,
  "config_sha256":hashlib.sha256(Path(cfg).read_bytes()).hexdigest()
},indent=2,sort_keys=True)+"\n")
PY
"$PY" "$CLIENT" --config "$CFG" --run-dir "$OUT" --endpoint "http://127.0.0.1:$PORT" --model "$MODEL_ID"
