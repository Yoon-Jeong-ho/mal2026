#!/usr/bin/env bash
# GPU0 plumbing smoke followed by five outer-fold NPCR jobs on GPUs 0--3.
set -Eeuo pipefail

[[ $# -eq 1 && ( "$1" == "smoke" || "$1" == "full" ) ]] || { echo "usage: $0 smoke|full" >&2; exit 2; }
MODE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="$ROOT/.venv-standard/bin/python"
CONFIG="$ROOT/configs/prompt_reference_npcr.v1.json"
RUNNER="$ROOT/scripts/run_prompt_reference_npcr.py"
MODULE="$ROOT/src/mal2026/prompt_reference_npcr.py"
LAUNCHER="$ROOT/scripts/run_prompt_reference_npcr_gpu0_3.sh"
TEST="$ROOT/tests/test_prompt_reference_npcr.py"
RUN_ID="prompt-reference-npcr-v1-20260803-001"
RUN_DIR="$ROOT/outputs/prompt-reference-npcr-v1/$RUN_ID"
RESTRICTED_DIR="$ROOT/data/processed/restricted/prompt_reference_npcr_v1/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
LEDGER="$RUN_DIR/ledger.jsonl"
CHECK_GPUS=(0); [[ "$MODE" == full ]] && CHECK_GPUS=(0 1 2 3)

for path in "$PYTHON" "$CONFIG" "$RUNNER" "$MODULE" "$LAUNCHER" "$TEST"; do [[ -e "$path" ]] || { echo "missing NPCR dependency: $path" >&2; exit 1; }; done
mkdir -p "$LOG_DIR"
for gpu in "${CHECK_GPUS[@]}"; do
  processes="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  [[ -z "$processes" ]] || { echo "GPU $gpu has a pre-existing compute process" >&2; exit 1; }
  IFS=, read -r memory utilization < <(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  memory="${memory// /}"; utilization="${utilization// /}"
  [[ "$memory" == 0 && "$utilization" == 0 ]] || { echo "GPU $gpu is not fully idle" >&2; exit 1; }
done

if [[ "$MODE" == full ]]; then
  "$PYTHON" - "$RUN_DIR/smoke/outer-00.json" "$RUN_DIR/smoke/attestation.json" "$CONFIG" "$RUNNER" "$MODULE" "$LAUNCHER" "$TEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
report_path, attest_path, *paths = map(Path, sys.argv[1:])
report, attestation = json.loads(report_path.read_text()), json.loads(attest_path.read_text())
assert report["status"] == "completed" and report["mode"] == "smoke" and report["records"] == 8
assert report["validation_rows_loaded"] is False and report["average_target_used"] is False
assert report["smoke_not_reusable_for_selection_or_scientific_results"] is True
assert attestation["smoke_report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
for name, path in zip(("config", "runner", "module", "launcher", "test"), paths, strict=True):
    assert attestation[f"{name}_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
assert attestation["physical_gpu"] == 0 and attestation["validation_rows_loaded"] is False
PY
fi

"$PYTHON" - "$LEDGER" "$MODE" "$CONFIG" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
ledger, mode, config = map(Path, sys.argv[1:])
ledger.parent.mkdir(parents=True, exist_ok=True)
event = {"run_id":"prompt-reference-npcr-v1-20260803-001", "event":"stage_launch", "stage":"prompt_reference_npcr",
 "mode":str(mode), "started_at":datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
 "git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(), "config_sha256":hashlib.sha256(config.read_bytes()).hexdigest(),
 "resource_scope":"GPUs 0-3; GPU0 smoke first", "validation_rows_loaded":False, "average_target_used":False, "r0_score_feature_used":False}
with ledger.open("a",encoding="utf-8") as f: f.write(json.dumps(event,sort_keys=True)+"\n")
PY

PIDS=(); TELEMETRY_PID=""
cleanup() {
  rc=$?
  [[ -n "$TELEMETRY_PID" ]] && kill "$TELEMETRY_PID" 2>/dev/null || true
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  [[ -n "$TELEMETRY_PID" ]] && wait "$TELEMETRY_PID" 2>/dev/null || true
  for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  if [[ "$rc" -ne 0 ]]; then
    "$PYTHON" - "$LEDGER" "$MODE" "$rc" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
ledger, mode, code = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
with ledger.open("a",encoding="utf-8") as f:
    f.write(json.dumps({"run_id":"prompt-reference-npcr-v1-20260803-001","event":"stage_failed","stage":"prompt_reference_npcr","mode":mode,"exit_code":code,"completed_at":datetime.now(timezone.utc).replace(microsecond=0).isoformat()},sort_keys=True)+"\n")
PY
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM
TELEMETRY="$LOG_DIR/gpu-telemetry-$MODE.csv"; [[ ! -e "$TELEMETRY" ]] || { echo "refusing to overwrite telemetry" >&2; exit 1; }
(
  echo "timestamp,index,utilization_gpu,memory_used,memory_total,temperature_gpu"
  while :; do nvidia-smi --id="$(IFS=,; echo "${CHECK_GPUS[*]}")" --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits || true; sleep 30; done
) >"$TELEMETRY" 2>&1 & TELEMETRY_PID="$!"

cd "$ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$MODE" == smoke ]]; then
  [[ ! -e "$LOG_DIR/smoke-gpu0.log" ]] || { echo "refusing to overwrite smoke log" >&2; exit 1; }
  PYTHONPATH=src "$PYTHON" -m unittest "$TEST" >"$LOG_DIR/smoke-tests.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" - "$CONFIG" "$RUN_DIR/smoke/outer-00.json" "$RESTRICTED_DIR/smoke/outer-00.jsonl" <<'PY' >"$LOG_DIR/smoke-gpu0.log" 2>&1
import json, os, sys
from dataclasses import replace
from pathlib import Path
from mal2026.prompt_reference_npcr import NPCRConfig, _atomic_json, _atomic_jsonl_private, _derived_seed, _fit_predict, load_rows, outer_and_inner_indices
config, public, restricted = NPCRConfig.from_json(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
assert not public.exists() and not restricted.exists()
rows = load_rows(config); _, inner = outer_and_inner_indices(rows, 0); fit, dev = inner[min(inner)]
smoke = replace(config, epochs=1)
prediction, pairs = _fit_predict(rows, fit, dev[:8], 0, config.candidates[0], smoke, seed=_derived_seed(config.seed, 0, config.candidates[0].identifier, 0, "gpu0-smoke"), device="cuda")
private_sha = _atomic_jsonl_private(restricted, ({"source_id": rows[index].source_id, "outer_fold": 0, "row_prediction": {"content": float(prediction[pos])}} for pos, index in enumerate(dev[:8])))
report = {"schema_version":"mal2026-prompt-reference-npcr-smoke-v1", "status":"completed", "mode":"smoke", "run_id":config.run_id,
 "records":8, "physical_gpu":0, "candidate":config.candidates[0].identifier, "axis":"content", "epochs":1, "pair_count":pairs,
 "restricted_predictions_sha256":private_sha, "validation_rows_loaded":False, "average_target_used":False, "r0_score_feature_used":False,
 "smoke_not_reusable_for_selection_or_scientific_results":True}
_atomic_json(public, report); print(json.dumps(report,sort_keys=True))
PY
  "$PYTHON" - "$RUN_DIR/smoke/outer-00.json" "$RUN_DIR/smoke/attestation.json" "$CONFIG" "$RUNNER" "$MODULE" "$LAUNCHER" "$TEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
report, output, *paths = map(Path, sys.argv[1:]); assert not output.exists()
value={"schema_version":"mal2026-prompt-reference-npcr-smoke-attestation-v1", "smoke_report_sha256":hashlib.sha256(report.read_bytes()).hexdigest(), "physical_gpu":0, "validation_rows_loaded":False, "average_target_used":False, "r0_score_feature_used":False}
for name,path in zip(("config","runner","module","launcher","test"),paths,strict=True): value[f"{name}_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
output.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
else
  for log in "$LOG_DIR/full-gpu0-folds0-4.log" "$LOG_DIR/full-gpu1-fold1.log" "$LOG_DIR/full-gpu2-fold2.log" "$LOG_DIR/full-gpu3-fold3.log" "$LOG_DIR/full-aggregate.log"; do [[ ! -e "$log" ]] || { echo "refusing to overwrite log" >&2; exit 1; }; done
  (CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --mode outer_fold --outer-fold 0 --device cuda; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --mode outer_fold --outer-fold 4 --device cuda) >"$LOG_DIR/full-gpu0-folds0-4.log" 2>&1 & PIDS+=("$!")
  for gpu in 1 2 3; do CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --mode outer_fold --outer-fold "$gpu" --device cuda >"$LOG_DIR/full-gpu${gpu}-fold${gpu}.log" 2>&1 & PIDS+=("$!"); done
  failure=0; for pid in "${PIDS[@]}"; do wait "$pid" || failure=1; done; PIDS=(); [[ "$failure" == 0 ]] || { echo "NPCR outer fold failure" >&2; exit 1; }
  PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --mode full --device cpu >"$LOG_DIR/full-aggregate.log" 2>&1
fi
kill "$TELEMETRY_PID" 2>/dev/null || true; wait "$TELEMETRY_PID" 2>/dev/null || true; TELEMETRY_PID=""
"$PYTHON" - "$LEDGER" "$MODE" "$RUN_DIR" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
ledger, mode, root = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
evidence = root / ("smoke/outer-00.json" if mode == "smoke" else "aggregate.json")
report = json.loads(evidence.read_text(encoding="utf-8"))
with ledger.open("a",encoding="utf-8") as f:
    f.write(json.dumps({"run_id":"prompt-reference-npcr-v1-20260803-001","event":"stage_complete","stage":"prompt_reference_npcr","mode":mode,"status":report["status"],"evidence_ref":str(evidence.resolve()),"completed_at":datetime.now(timezone.utc).replace(microsecond=0).isoformat(),"validation_rows_loaded":False,"average_target_used":False},sort_keys=True)+"\n")
PY
trap - EXIT INT TERM
