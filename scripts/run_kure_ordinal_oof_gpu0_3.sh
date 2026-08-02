#!/usr/bin/env bash
# GPU0 smoke followed by exact five-fold KURE OOF on authorized GPUs 0--3.
set -Eeuo pipefail

[[ $# -eq 1 && ( "$1" == "smoke" || "$1" == "full" ) ]] || {
  echo "usage: $0 smoke|full" >&2
  exit 2
}

MODE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="$ROOT/.venv-standard/bin/python"
CONFIG="$ROOT/configs/kure_ordinal_oof.v1.json"
RUNNER="$ROOT/scripts/run_kure_ordinal_oof.py"
MODULE="$ROOT/src/mal2026/kure_ordinal_oof.py"
LAUNCHER="$ROOT/scripts/run_kure_ordinal_oof_gpu0_3.sh"
TEST="$ROOT/tests/test_kure_ordinal_oof.py"
RUN_DIR="$ROOT/outputs/kure-ordinal-oof-v1/kure-ordinal-oof-v1-20260803-001"
LOG_DIR="$RUN_DIR/logs"
LEDGER="$RUN_DIR/ledger.jsonl"
GPUS=(0 1 2 3)
CHECK_GPUS=(0)
[[ "$MODE" == "full" ]] && CHECK_GPUS=("${GPUS[@]}")

for path in "$PYTHON" "$CONFIG" "$RUNNER" "$MODULE" "$LAUNCHER" "$TEST"; do
  [[ -e "$path" ]] || { echo "missing runtime dependency: $path" >&2; exit 1; }
done
mkdir -p "$LOG_DIR"

for gpu in "${CHECK_GPUS[@]}"; do
  processes="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  [[ -z "$processes" ]] || { echo "GPU $gpu has a pre-existing compute process" >&2; exit 1; }
  IFS=, read -r memory utilization < <(
    nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
  )
  memory="${memory// /}"; utilization="${utilization// /}"
  [[ "$memory" == 0 && "$utilization" == 0 ]] || { echo "GPU $gpu is not fully idle" >&2; exit 1; }
done

if [[ "$MODE" == "full" ]]; then
  "$PYTHON" - "$RUN_DIR/smoke/outer-00.json" "$RUN_DIR/smoke/attestation.json" \
    "$CONFIG" "$RUNNER" "$MODULE" "$LAUNCHER" "$TEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
report_path, attest_path, *code_paths = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
attest = json.loads(attest_path.read_text(encoding="utf-8"))
assert report["status"] == "completed" and report["mode"] == "smoke"
assert report["outer_fold"] == 0 and report["records"] == 8
assert report["validation_rows_loaded"] is False and report["average_target_used"] is False
assert report["smoke_not_reusable_for_selection_or_scientific_results"] is True
assert attest["physical_gpu"] == 0 and attest["validation_rows_loaded"] is False
assert attest["average_target_used"] is False
names = ("config", "runner", "module", "launcher", "test")
for name, path in zip(names, code_paths, strict=True):
    assert attest[f"{name}_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
assert attest["smoke_report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
PY
fi

"$PYTHON" - "$LEDGER" "$MODE" "$CONFIG" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
ledger, mode, config = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
event = {
    "run_id": "kure-ordinal-oof-v1-20260803-001",
    "event": "stage_launch", "stage": "kure_ordinal_oof", "mode": mode,
    "started_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "config_file_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "resource_scope": "GPUs 0-3; GPU0 smoke first",
    "methods": ["coral-natural", "rps-natural"],
    "validation_selection": False, "average_target_used": False,
}
ledger.parent.mkdir(parents=True, exist_ok=True)
with ledger.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(event, sort_keys=True) + "\n")
PY

PIDS=(); TELEMETRY_PID=""
cleanup() {
  rc=$?
  [[ -n "$TELEMETRY_PID" ]] && kill "$TELEMETRY_PID" 2>/dev/null || true
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  exit "$rc"
}
trap cleanup EXIT INT TERM

TELEMETRY="$LOG_DIR/gpu-telemetry-$MODE.csv"
[[ ! -e "$TELEMETRY" ]] || { echo "refusing to overwrite $TELEMETRY" >&2; exit 1; }
(
  echo "timestamp,index,utilization_gpu,memory_used,memory_total,temperature_gpu"
  while :; do
    nvidia-smi --id="$(IFS=,; echo "${CHECK_GPUS[*]}")" \
      --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu \
      --format=csv,noheader,nounits || true
    sleep 30
  done
) >"$TELEMETRY" 2>&1 & TELEMETRY_PID="$!"

cd "$ROOT"
if [[ "$MODE" == "smoke" ]]; then
  LOG="$LOG_DIR/smoke-gpu0.log"
  [[ ! -e "$LOG" ]] || { echo "refusing to overwrite $LOG" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" "$RUNNER" \
    --config "$CONFIG" --outer-fold 0 --smoke >"$LOG" 2>&1
  "$PYTHON" - "$RUN_DIR/smoke/outer-00.json" "$RUN_DIR/smoke/attestation.json" \
    "$CONFIG" "$RUNNER" "$MODULE" "$LAUNCHER" "$TEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
report, output, *code_paths = map(Path, sys.argv[1:])
assert not output.exists(), f"refusing to overwrite {output}"
names = ("config", "runner", "module", "launcher", "test")
value = {
    "schema_version": "mal2026-kure-ordinal-oof-smoke-attestation-v1",
    "smoke_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    "physical_gpu": 0, "validation_rows_loaded": False, "average_target_used": False,
}
for name, path in zip(names, code_paths, strict=True):
    value[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
else
  LOGS=("$LOG_DIR/full-gpu0-folds0-4.log" "$LOG_DIR/full-gpu1-fold1.log" \
        "$LOG_DIR/full-gpu2-fold2.log" "$LOG_DIR/full-gpu3-fold3.log" "$LOG_DIR/full-aggregate.log")
  for log in "${LOGS[@]}"; do [[ ! -e "$log" ]] || { echo "refusing to overwrite $log" >&2; exit 1; }; done
  (
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --outer-fold 0
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --outer-fold 4
  ) >"${LOGS[0]}" 2>&1 & PIDS+=("$!")
  for gpu in 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$PYTHON" "$RUNNER" \
      --config "$CONFIG" --outer-fold "$gpu" >"$LOG_DIR/full-gpu${gpu}-fold${gpu}.log" 2>&1 &
    PIDS+=("$!")
  done
  failure=0
  for pid in "${PIDS[@]}"; do wait "$pid" || failure=1; done
  PIDS=()
  [[ "$failure" == 0 ]] || { echo "one or more KURE outer-fold workers failed" >&2; exit 1; }
  PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --aggregate >"${LOGS[4]}" 2>&1
fi

kill "$TELEMETRY_PID" 2>/dev/null || true; wait "$TELEMETRY_PID" 2>/dev/null || true; TELEMETRY_PID=""
"$PYTHON" - "$LEDGER" "$MODE" "$RUN_DIR" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
ledger, mode, root = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
evidence = root / ("smoke/outer-00.json" if mode == "smoke" else "aggregate.json")
report = json.loads(evidence.read_text(encoding="utf-8"))
event = {"run_id": "kure-ordinal-oof-v1-20260803-001", "event": "stage_complete",
         "stage": "kure_ordinal_oof", "mode": mode, "status": report["status"],
         "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
         "evidence_ref": str(evidence.resolve()), "validation_selection": False,
         "average_target_used": False}
with ledger.open("a", encoding="utf-8") as stream: stream.write(json.dumps(event, sort_keys=True) + "\n")
PY
trap - EXIT INT TERM
