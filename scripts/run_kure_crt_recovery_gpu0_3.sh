#!/usr/bin/env bash
# GPU0 smoke, then five fixed train-only recovery folds on GPUs 0--3.
set -Eeuo pipefail

[[ $# -eq 1 && ( "$1" == "smoke" || "$1" == "full" ) ]] || { echo "usage: $0 smoke|full" >&2; exit 2; }
MODE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="$ROOT/.venv-standard/bin/python"
CONFIG="$ROOT/configs/kure_crt_recovery.v1.json"
RUNNER="$ROOT/scripts/run_kure_crt_recovery.py"
MODULE="$ROOT/src/mal2026/kure_crt_recovery.py"
LAUNCHER="$ROOT/scripts/run_kure_crt_recovery_gpu0_3.sh"
TEST="$ROOT/tests/test_kure_crt_recovery.py"
RUN_DIR="$ROOT/outputs/kure-ordinal-crt-recovery-v1/kure-ordinal-crt-recovery-v1-20260803-001"
LOG_DIR="$RUN_DIR/logs"
LEDGER="$RUN_DIR/ledger.jsonl"
CHECK_GPUS=(0)
[[ "$MODE" == "full" ]] && CHECK_GPUS=(0 1 2 3)

for path in "$PYTHON" "$CONFIG" "$RUNNER" "$MODULE" "$LAUNCHER" "$TEST"; do [[ -e "$path" ]] || { echo "missing runtime dependency: $path" >&2; exit 1; }; done
mkdir -p "$LOG_DIR"
for gpu in "${CHECK_GPUS[@]}"; do
  processes="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  [[ -z "$processes" ]] || { echo "GPU $gpu has a pre-existing compute process" >&2; exit 1; }
  IFS=, read -r memory utilization < <(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  memory="${memory// /}"; utilization="${utilization// /}"
  # NVIDIA's driver can reserve a few MiB even when no compute context exists.
  # The process query above is authoritative; tolerate only that small idle
  # footprint and still require zero measured utilization.
  (( memory <= 16 )) && [[ "$utilization" == 0 ]] || { echo "GPU $gpu is not fully idle" >&2; exit 1; }
done

if [[ "$MODE" == "full" ]]; then
  "$PYTHON" - "$RUN_DIR/smoke/outer-00.json" "$RUN_DIR/smoke/attestation.json" "$CONFIG" "$RUNNER" "$MODULE" "$LAUNCHER" "$TEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
report_path, attestation_path, *paths = map(Path, sys.argv[1:])
report, attestation = json.loads(report_path.read_text()), json.loads(attestation_path.read_text())
assert report["status"] == "completed" and report["mode"] == "smoke" and report["nonselectable"] is True
assert report["validation_rows_loaded"] is False and report["average_target_used"] is False
assert attestation["physical_gpu"] == 0 and attestation["nonselectable"] is True
assert attestation["smoke_report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
for name, path in zip(("config", "runner", "module", "launcher", "test"), paths, strict=True):
    assert attestation[f"{name}_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
PY
fi

"$PYTHON" - "$LEDGER" "$MODE" "$CONFIG" <<'PY'
import hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
ledger, mode, config = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
scope = [0] if mode == "smoke" else [0, 1, 2, 3]
workers = ([{"physical_gpu": 0, "outer_fold": 0, "smoke": True}] if mode == "smoke" else
           [{"physical_gpu": 0, "outer_folds": [0, 4]}, {"physical_gpu": 1, "outer_folds": [1]},
            {"physical_gpu": 2, "outer_folds": [2]}, {"physical_gpu": 3, "outer_folds": [3]}])
hardware = []
for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"], text=True).splitlines():
    index, uuid, name, total = [item.strip() for item in line.split(",", 3)]
    if int(index) in scope:
        hardware.append({"physical_gpu": int(index), "uuid": uuid, "name": name, "memory_total_mib": int(total)})
event = {"run_id": "kure-ordinal-crt-recovery-v1-20260803-001", "event": "stage_launch", "stage": "kure_crt_recovery", "mode": mode,
         "started_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
         "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
         "config_file_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
         "exact_launcher_command": f"bash scripts/run_kure_crt_recovery_gpu0_3.sh {mode}",
         "resource_scope": "physical GPU0 only" if mode == "smoke" else "physical GPUs 0,1,2,3",
         "gpu_authorization": "user-authorized default MAL2026 GPUs 0-3; GPU0 smoke first",
         "worker_mapping": workers, "hardware": hardware,
         "environment": {"python_executable": sys.executable, "python": platform.python_version(), "platform": platform.platform()},
         "method": "coral-natural", "validation_selection": False, "average_target_used": False, "negative_stage3_preserved": True}
ledger.parent.mkdir(parents=True, exist_ok=True)
with ledger.open("a", encoding="utf-8") as stream: stream.write(json.dumps(event, sort_keys=True) + "\n")
PY

PIDS=(); TELEMETRY_PID=""
cleanup() { rc=$?; [[ -n "$TELEMETRY_PID" ]] && kill "$TELEMETRY_PID" 2>/dev/null || true; for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; exit "$rc"; }
trap cleanup EXIT INT TERM
TELEMETRY="$LOG_DIR/gpu-telemetry-$MODE.csv"
[[ ! -e "$TELEMETRY" ]] || { echo "refusing to overwrite $TELEMETRY" >&2; exit 1; }
(
  echo "timestamp,index,utilization_gpu,memory_used,memory_total,temperature_gpu"
  while :; do nvidia-smi --id="$(IFS=,; echo "${CHECK_GPUS[*]}")" --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits || true; sleep 30; done
) >"$TELEMETRY" 2>&1 & TELEMETRY_PID="$!"

cd "$ROOT"
if [[ "$MODE" == "smoke" ]]; then
  LOG="$LOG_DIR/smoke-gpu0.log"; [[ ! -e "$LOG" ]] || { echo "refusing to overwrite $LOG" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --outer-fold 0 --smoke >"$LOG" 2>&1
  "$PYTHON" - "$RUN_DIR/smoke/outer-00.json" "$RUN_DIR/smoke/attestation.json" "$CONFIG" "$RUNNER" "$MODULE" "$LAUNCHER" "$TEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
report, output, *paths = map(Path, sys.argv[1:])
assert not output.exists(), "refusing to overwrite smoke attestation"
value = {"schema_version": "mal2026-kure-crt-recovery-smoke-attestation-v1", "physical_gpu": 0, "nonselectable": True,
         "smoke_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(), "validation_rows_loaded": False, "average_target_used": False}
for name, path in zip(("config", "runner", "module", "launcher", "test"), paths, strict=True): value[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
else
  LOGS=("$LOG_DIR/full-gpu0-folds0-4.log" "$LOG_DIR/full-gpu1-fold1.log" "$LOG_DIR/full-gpu2-fold2.log" "$LOG_DIR/full-gpu3-fold3.log" "$LOG_DIR/full-aggregate.log")
  for log in "${LOGS[@]}"; do [[ ! -e "$log" ]] || { echo "refusing to overwrite $log" >&2; exit 1; }; done
  (CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --outer-fold 0; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --outer-fold 4) >"${LOGS[0]}" 2>&1 & PIDS+=("$!")
  for gpu in 1 2 3; do CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --outer-fold "$gpu" >"$LOG_DIR/full-gpu${gpu}-fold${gpu}.log" 2>&1 & PIDS+=("$!"); done
  failure=0; for pid in "${PIDS[@]}"; do wait "$pid" || failure=1; done; PIDS=()
  [[ "$failure" == 0 ]] || { echo "one or more recovery workers failed" >&2; exit 1; }
  PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --aggregate >"${LOGS[4]}" 2>&1
fi
kill "$TELEMETRY_PID" 2>/dev/null || true; wait "$TELEMETRY_PID" 2>/dev/null || true; TELEMETRY_PID=""
"$PYTHON" - "$LEDGER" "$MODE" "$RUN_DIR" "$TELEMETRY" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
ledger, mode, root, telemetry = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
evidence = root / ("smoke/outer-00.json" if mode == "smoke" else "aggregate.json")
report = json.loads(evidence.read_text())
event = {"run_id": "kure-ordinal-crt-recovery-v1-20260803-001", "event": "stage_complete", "stage": "kure_crt_recovery", "mode": mode,
         "status": report["status"], "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
         "evidence_ref": str(evidence.resolve()), "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
         "telemetry_ref": str(telemetry.resolve()), "telemetry_sha256": hashlib.sha256(telemetry.read_bytes()).hexdigest(),
         "validation_selection": False, "average_target_used": False}
with ledger.open("a", encoding="utf-8") as stream: stream.write(json.dumps(event, sort_keys=True) + "\n")
PY
trap - EXIT INT TERM
