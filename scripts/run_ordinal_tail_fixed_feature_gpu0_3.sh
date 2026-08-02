#!/usr/bin/env bash
# Immutable GPU0--3 launcher for the ordinal-tail frozen-feature screen.
set -Eeuo pipefail

[[ $# -eq 1 && ( "$1" == "smoke" || "$1" == "full" ) ]] || {
  echo "usage: $0 smoke|full" >&2
  exit 2
}

MODE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="$ROOT/.venv-standard/bin/python"
RUNNER="$ROOT/scripts/run_ordinal_tail_fixed_feature.py"
MODULE="$ROOT/src/mal2026/ordinal_tail_fixed_feature.py"
LAUNCHER="$ROOT/scripts/run_ordinal_tail_fixed_feature_gpu0_3.sh"
TEST="$ROOT/tests/test_ordinal_tail_fixed_feature.py"
CONFIG="$ROOT/configs/ordinal_tail_program.v1.json"
RUN_DIR="$ROOT/outputs/ordinal-tail-program-v1/ordinal-tail-program-v1-20260803-001"
LOG_DIR="$RUN_DIR/fixed_feature/logs"
LEDGER="$RUN_DIR/ledger.jsonl"
GPUS=(0 1 2 3)
CHECK_GPUS=(0)
[[ "$MODE" == "full" ]] && CHECK_GPUS=("${GPUS[@]}")

[[ -x "$PYTHON" && -r "$RUNNER" && -r "$CONFIG" ]] || {
  echo "fixed-feature runtime prerequisite failed" >&2
  exit 1
}
mkdir -p "$LOG_DIR"

# Read only the explicitly authorized GPU set. Never displace an existing job.
for gpu in "${CHECK_GPUS[@]}"; do
  processes="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  [[ -z "$processes" ]] || {
    echo "GPU $gpu has a pre-existing compute process; refusing to start" >&2
    exit 1
  }
  IFS=, read -r memory utilization < <(
    nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
  )
  memory="${memory// /}"; utilization="${utilization// /}"
  [[ "$memory" == "0" && "$utilization" == "0" ]] || {
    echo "GPU $gpu is not fully idle; refusing to start" >&2
    exit 1
  }
done

if [[ "$MODE" == "full" ]]; then
  "$PYTHON" - "$RUN_DIR/fixed_feature/smoke/outer-00.json" \
    "$RUN_DIR/fixed_feature/smoke/attestation.json" "$CONFIG" "$RUNNER" \
    "$MODULE" "$LAUNCHER" "$TEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
report_path, attestation_path, config_path, runner_path, module_path, launcher_path, test_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
assert report["status"] == "completed" and report["mode"] == "smoke"
assert report["candidate_count"] == 10 and report["validation_rows_loaded"] is False
assert report["average_target_used"] is False
assert report["candidate_inventory"] == [
    "ce-natural", "rps-natural", "coral-natural", "corn-natural",
    "slace-a0.5", "slace-a1", "slace-a2", "ce-effective-b0.99",
    "ce-effective-b0.999", "ce-sqrt-sampler",
]
assert attestation["smoke_report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
assert attestation["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
assert attestation["runner_sha256"] == hashlib.sha256(runner_path.read_bytes()).hexdigest()
assert attestation["module_sha256"] == hashlib.sha256(module_path.read_bytes()).hexdigest()
assert attestation["launcher_sha256"] == hashlib.sha256(launcher_path.read_bytes()).hexdigest()
assert attestation["test_sha256"] == hashlib.sha256(test_path.read_bytes()).hexdigest()
assert attestation["physical_gpu"] == 0
assert attestation["validation_selection"] is False
assert attestation["average_target_used"] is False
assert report["config_sha256"] == attestation["config_sha256"]
config = json.loads(config_path.read_text(encoding="utf-8"))
assert report["embedding_rows_sha256"] == config["r0_embedding_rows_sha256"]
assert report["r0_oof_prediction_sha256"] == config["r0_oof_prediction_sha256"]
PY
fi

"$PYTHON" - "$LEDGER" "$MODE" "$CONFIG" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
ledger, mode, config = map(Path, sys.argv[1:])
event = {
    "run_id": "ordinal-tail-program-v1-20260803-001",
    "event": "stage_launch",
    "stage": "fixed_feature_screen",
    "mode": str(mode),
    "started_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "resource_scope": "GPUs 0-3; GPU0 smoke first",
    "validation_selection": False,
    "average_target_used": False,
}
with ledger.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(event, sort_keys=True) + "\n")
PY

PIDS=()
TELEMETRY_PID=""
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
) >"$TELEMETRY" 2>&1 &
TELEMETRY_PID="$!"

cd "$ROOT"
if [[ "$MODE" == "smoke" ]]; then
  [[ ! -e "$LOG_DIR/smoke-gpu0.log" ]] || { echo "refusing to overwrite smoke log" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" "$RUNNER" \
    --config "$CONFIG" --mode smoke --outer-fold 0 \
    >"$LOG_DIR/smoke-gpu0.log" 2>&1
  "$PYTHON" - "$RUN_DIR/fixed_feature/smoke/outer-00.json" \
    "$RUN_DIR/fixed_feature/smoke/attestation.json" "$CONFIG" "$RUNNER" \
    "$MODULE" "$LAUNCHER" "$TEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
report, output, config, runner, module, launcher, test = map(Path, sys.argv[1:])
assert not output.exists(), f"refusing to overwrite {output}"
value = {
    "schema_version": "mal2026-ordinal-tail-fixed-feature-smoke-attestation-v1",
    "smoke_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
    "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
    "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
    "test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
    "physical_gpu": 0,
    "validation_selection": False,
    "average_target_used": False,
}
output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
else
  for log in "$LOG_DIR/full-gpu0-folds0-4.log" "$LOG_DIR/full-gpu1-fold1.log" \
             "$LOG_DIR/full-gpu2-fold2.log" "$LOG_DIR/full-gpu3-fold3.log" \
             "$LOG_DIR/full-aggregate.log"; do
    [[ ! -e "$log" ]] || { echo "refusing to overwrite $log" >&2; exit 1; }
  done
  # Five outer folds are scheduled as four independent workers. GPU0 executes
  # folds 0 and 4 serially; GPUs 1--3 execute one fold each.
  (
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --mode outer_fold --outer-fold 0
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --mode outer_fold --outer-fold 4
  ) >"$LOG_DIR/full-gpu0-folds0-4.log" 2>&1 & PIDS+=("$!")
  for gpu in 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$PYTHON" "$RUNNER" \
      --config "$CONFIG" --mode outer_fold --outer-fold "$gpu" \
      >"$LOG_DIR/full-gpu${gpu}-fold${gpu}.log" 2>&1 & PIDS+=("$!")
  done
  failure=0
  for pid in "${PIDS[@]}"; do wait "$pid" || failure=1; done
  PIDS=()
  [[ "$failure" -eq 0 ]] || { echo "one or more outer-fold workers failed" >&2; exit 1; }
  PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --mode full \
    >"$LOG_DIR/full-aggregate.log" 2>&1
fi

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
TELEMETRY_PID=""

"$PYTHON" - "$LEDGER" "$MODE" "$RUN_DIR" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
ledger, mode, root = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
evidence = root / "fixed_feature" / ("smoke/outer-00.json" if mode == "smoke" else "aggregate.json")
report = json.loads(evidence.read_text(encoding="utf-8"))
event = {
    "run_id": "ordinal-tail-program-v1-20260803-001",
    "event": "stage_complete",
    "stage": "fixed_feature_screen",
    "mode": mode,
    "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "status": report["status"],
    "evidence_ref": str(evidence.resolve()),
    "validation_selection": False,
    "average_target_used": False,
}
with ledger.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(event, sort_keys=True) + "\n")
PY

trap - EXIT INT TERM
