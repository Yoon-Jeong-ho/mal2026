#!/usr/bin/env bash
# v5.2 migration launcher.  It only ever queries and starts selected GPUs 0--3.
set -Eeuo pipefail
[[ $# -ge 4 ]] || { echo "usage: $0 RUN_ID MODE(smoke|full) SAMPLE_ESSAYS GPU [GPU ...]" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"; RUN_ID="$1"; MODE="$2"; SAMPLE="$3"; shift 3; GPUS=("$@")
PYTHON="$ROOT/.venv-standard/bin/python"; RUNNER="$ROOT/scripts/run_openai_explanation_repeat_distribution_v5_2_gpu0_3.py"; CONFIG="$ROOT/configs/openai_explanation_repeat_distribution.v5_2.gpu0_3.json"
PREFLIGHT="$ROOT/scripts/preflight_openai_repeat_v5_2_gpu0_3_synthetic.py"; RESOLVER="$ROOT/scripts/resolve_v5_2_gpu0_3_priority.py"; MODEL="$ROOT/outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
SERVER="$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server"; LLAMA_REPO="$ROOT/outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/llama.cpp"
[[ "$MODE" == smoke && "$SAMPLE" == 3 || "$MODE" == full && "$SAMPLE" == 2000 ]] || { echo "v5.2 migration permits only 3-essay smoke or 2,000-essay full train-only run" >&2; exit 2; }
[[ -x "$PYTHON" && -x "$SERVER" && -r "$MODEL" ]] || { echo "v5.2 runtime prerequisite failed" >&2; exit 1; }
[[ "$(stat -c %s "$MODEL")" == 22285080192 && "$(sha256sum "$MODEL" | awk '{print $1}')" == b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772 && "$(git -C "$LLAMA_REPO" rev-parse HEAD)" == 571d0d540df04f25298d0e159e520d9fc62ed121 && "$(git -C "$LLAMA_REPO" describe --tags --exact-match)" == b10068 ]] || { echo "v5.2 pinned-runtime gate failed" >&2; exit 1; }
[[ ${#GPUS[@]} -ge 1 && ${#GPUS[@]} -le 4 ]] || { echo "invalid GPU count" >&2; exit 2; }
declare -A SEEN=(); for gpu in "${GPUS[@]}"; do [[ "$gpu" =~ ^[0-3]$ && -z "${SEEN[$gpu]:-}" ]] || { echo "GPU selection must be unique and within 0--3" >&2; exit 2; }; SEEN[$gpu]=1; done
if [[ "$MODE" == smoke && "${GPUS[*]}" != "0" ]]; then echo "smoke is GPU0-only" >&2; exit 2; fi
verify_gate_report() {
  local report="$1"; local label="$2"
  [[ -n "$report" && -f "$report" ]] || { echo "$label aggregate report is required" >&2; exit 1; }
  "$PYTHON" - "$report" "$CONFIG" "$label" <<'PY'
import hashlib,json,sys
report=json.load(open(sys.argv[1],encoding="utf-8")); expected=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest()
assert report.get("status") == "passed", f"{sys.argv[3]} gate did not pass"
assert report.get("config_sha256") == expected, f"{sys.argv[3]} config lineage mismatch"
assert all(report.get("hard_gates",{}).values()), f"{sys.argv[3]} hard-gate report is incomplete"
PY
}
verify_gate_report "${MAL2026_V52_PREFLIGHT_REPORT:-}" "synthetic preflight"
if [[ "$MODE" == full ]]; then
  verify_gate_report "${MAL2026_V52_SMOKE_REPORT:-}" "three-essay smoke"
  RESOLUTION="$ROOT/outputs/debug/v5_2-priority-resolution-$RUN_ID.json"; [[ ! -e "$RESOLUTION" ]] || { echo "priority-resolution artifact already exists" >&2; exit 1; }
  "$PYTHON" "$RESOLVER" --gpus "${GPUS[@]}" --output "$RESOLUTION" >/dev/null
  mapfile -t GPUS < <("$PYTHON" - "$RESOLUTION" <<'PY'
import json,sys
print("\n".join(str(gpu) for gpu in json.load(open(sys.argv[1],encoding="utf-8"))["selected_physical_gpus"]))
PY
)
fi
for gpu in "${GPUS[@]}"; do
  line="$(nvidia-smi --id="$gpu" --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)"; IFS=, read -r index used util temp <<<"$line"
  [[ "${index// /}" == "$gpu" && "${used// /}" == 0 && "${util// /}" == 0 && "${temp// /}" -le 80 ]] || { echo "selected GPU $gpu is not idle/cool" >&2; exit 1; }
done
RUN_DIR="$ROOT/data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001/judge_runs/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || { echo "run directory already exists; refusing overwrite" >&2; exit 1; }
cd "$ROOT"; "$PYTHON" "$RUNNER" prepare --run-id "$RUN_ID" --gpus "${GPUS[@]}" --sample-essays "$SAMPLE" --execution-mode "$MODE"
if [[ "$MODE" == full ]]; then
  "$PYTHON" - "$RUN_DIR/manifest.json" "$RESOLUTION" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); value=json.loads(p.read_text(encoding="utf-8")); resolution=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
value["priority_resolution"]={key:resolution[key] for key in ("status","requested_physical_gpus","selected_physical_gpus","release_verified","ownership_evidence")}
p.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
fi
mkdir -p "$RUN_DIR/logs"; PIDS=(); WATCHDOG=""
cleanup() { rc=$?; for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done; [[ -n "$WATCHDOG" ]] && { kill "$WATCHDOG" 2>/dev/null || true; wait "$WATCHDOG" 2>/dev/null || true; }; exit "$rc"; }; trap cleanup EXIT INT TERM
mapfile -t PORTS < <("$PYTHON" - "$CONFIG" "${GPUS[@]}" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1],encoding="utf-8")); print("\n".join(str(cfg["runtime"]["ports"][gpu]) for gpu in sys.argv[2:]))
PY
)
"$PYTHON" - "${PORTS[@]}" <<'PY'
import socket,sys
for raw in sys.argv[1:]:
    port=int(raw)
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        if sock.connect_ex(("127.0.0.1",port)) == 0: raise SystemExit(f"localhost port {port} is occupied")
PY
for offset in "${!GPUS[@]}"; do gpu="${GPUS[$offset]}"; port="${PORTS[$offset]}"; env CUDA_VISIBLE_DEVICES="$gpu" MAL2026_RESERVED_PHYSICAL_GPU="$gpu" "$SERVER" --model "$MODEL" --host 127.0.0.1 --port "$port" --n-gpu-layers 99 --parallel 4 --ctx-size 16384 --no-webui --reasoning off >"$RUN_DIR/logs/llama-server-gpu$gpu.log" 2>&1 & PIDS+=("$!"); done
for port in "${PORTS[@]}"; do for _ in $(seq 1 180); do curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 1; done; curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null; done
"$PYTHON" - "$RUN_DIR/server_attestation.json" "$CONFIG" "$MODEL" "$SERVER" "$LLAMA_REPO" "${GPUS[*]}" "${PORTS[*]}" "${PIDS[*]}" <<'PY'
import hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import urlopen
p,c,model,server,repo,gpus,ports,pids=sys.argv[1:]; gs=[int(x) for x in gpus.split()]; ps=[int(x) for x in ports.split()]; ids=[int(x) for x in pids.split()]
for gpu,port,pid in zip(gs,ps,ids,strict=True):
    env=Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"); visible=next((x.split(b"=",1)[1].decode() for x in env if x.startswith(b"CUDA_VISIBLE_DEVICES=")), "")
    props=json.loads(urlopen(f"http://127.0.0.1:{port}/props",timeout=5).read().decode())
    assert visible == str(gpu) and props.get("total_slots") == 4 and props.get("default_generation_settings",{}).get("n_ctx") == 4096
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
Path(p).write_text(json.dumps({"schema_version":"mal2026-repeat-v5_2-gpu0-3-server-attestation-v1","created_at":datetime.now(timezone.utc).replace(microsecond=0).isoformat(),"physical_gpus":gs,"parallel_requests_per_server":4,"slot_context":4096,"server_host":"127.0.0.1","server_ports":ps,"server_pids":ids,"config_sha256":digest(c),"runner_sha256":digest(Path(c).parents[1]/"scripts/run_openai_explanation_repeat_distribution_v5_2_gpu0_3.py"),"preflight_sha256":digest(Path(c).parents[1]/"scripts/preflight_openai_repeat_v5_2_gpu0_3_synthetic.py"),"model_sha256":digest(model),"llama_server_sha256":digest(server),"llama_revision":subprocess.check_output(["git","-C",repo,"rev-parse","HEAD"],text=True).strip(),"git_sha":subprocess.check_output(["git","-C",str(Path(c).parents[1]),"rev-parse","HEAD"],text=True).strip(),"watchdog_faults":0,"gpu_ownership":"project-owned GPUs 0--3 only; GPUs 4--7 never queried or used"},sort_keys=True,indent=2)+"\n",encoding="utf-8")
PY
"$PYTHON" - "$RUN_DIR/watchdog_final.json" "${GPUS[@]}" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"fault_count":0,"gpus_checked":[int(x) for x in sys.argv[2:]]},sort_keys=True)+"\n")
PY
(
  faults=0
  while :; do
    for gpu in "${GPUS[@]}"; do line="$(nvidia-smi --id="$gpu" --query-gpu=index,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true)"; IFS=, read -r index total used temp <<<"$line"; total="${total// /}"; used="${used// /}"; temp="${temp// /}"; if [[ -z "$total" || -z "$used" || -z "$temp" || "$temp" -gt 85 || "$used" -gt $((total * 75 / 100)) ]]; then faults=$((faults+1)); for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done; break 2; fi; done
    sleep 5
  done
  "$PYTHON" - "$RUN_DIR/watchdog_final.json" "$faults" "${GPUS[@]}" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"fault_count":int(sys.argv[2]),"gpus_checked":[int(x) for x in sys.argv[3:]]},sort_keys=True)+"\n")
PY
) & WATCHDOG="$!"
SERVER_ARGS=(); for offset in "${!GPUS[@]}"; do SERVER_ARGS+=(--server "${GPUS[$offset]}=http://127.0.0.1:${PORTS[$offset]}"); done
"$PYTHON" "$RUNNER" execute --run-id "$RUN_ID" --gpus "${GPUS[@]}" "${SERVER_ARGS[@]}"
kill "$WATCHDOG" 2>/dev/null || true; wait "$WATCHDOG" 2>/dev/null || true; WATCHDOG=""
"$PYTHON" - "$RUN_DIR/watchdog_final.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1],encoding="utf-8"))["fault_count"] == 0
PY
