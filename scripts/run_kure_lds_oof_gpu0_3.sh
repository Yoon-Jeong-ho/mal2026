#!/usr/bin/env bash
# Fail-closed GPU0 smoke, then exact five-fold LDS training on GPUs 0--3.
set -Eeuo pipefail
[[ $# -eq 1 && ( "$1" == "smoke" || "$1" == "full" ) ]] || { echo "usage: $0 smoke|full" >&2; exit 2; }
MODE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"  # side-effect-free normalization before every gate
PYTHON="$ROOT/.venv-standard/bin/python"
CONFIG="$ROOT/configs/kure_lds_oof.v1.json"
RUNNER="$ROOT/scripts/run_kure_lds_oof.py"
MODULE="$ROOT/src/mal2026/kure_lds_oof.py"
PREPARER="$ROOT/scripts/prepare_kure_lds_inputs.py"
LAUNCHER="$ROOT/scripts/run_kure_lds_oof_gpu0_3.sh"
TEST="$ROOT/tests/test_kure_lds_oof.py"
RUN_DIR="$ROOT/outputs/kure-coral-lds-oof-v1/kure-coral-lds-oof-v1-20260803-001"
LOG_DIR="$RUN_DIR/logs"; LEDGER="$RUN_DIR/ledger.jsonl"
for path in "$PYTHON" "$CONFIG" "$RUNNER" "$MODULE" "$PREPARER" "$LAUNCHER" "$TEST"; do [[ -e "$path" ]] || { echo "missing runtime dependency: $path" >&2; exit 1; }; done
# First executable gate: pending configs cannot create artifacts, query GPUs, or acquire locks.
PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --check-authorization >/dev/null
lexists() { [[ -e "$1" || -L "$1" ]]; }
ATTEMPT_TAG="smoke-one-shot"
if [[ "$MODE" == "full" ]]; then
  ATTEMPT_TAG="${MAL2026_ATTEMPT_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM}"
  [[ "$ATTEMPT_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || {
    echo "unsafe MAL2026_ATTEMPT_TAG" >&2; exit 2;
  }
fi
if [[ "$MODE" == "full" ]] && lexists "$RUN_DIR/aggregate.json"; then
  echo "complete LDS aggregate exists; scientific rerun is forbidden" >&2; exit 1
fi
if [[ "$MODE" == "smoke" ]]; then
  for sentinel in "$RUN_DIR/smoke/outer-00.json" "$RUN_DIR/smoke/attestation.json" \
                  "$RUN_DIR/logs/gpu-telemetry-smoke.csv" "$RUN_DIR/logs/gpu-telemetry-smoke-summary.json" \
                  "$RUN_DIR/logs/smoke-gpu0.log"; do
    lexists "$sentinel" && { echo "immutable smoke/output sentinel already exists; refusing to overwrite" >&2; exit 1; }
  done
else
  for sentinel in "$RUN_DIR/logs/gpu-telemetry-full-${ATTEMPT_TAG}.csv" \
                  "$RUN_DIR/logs/gpu-telemetry-full-${ATTEMPT_TAG}-summary.json"; do
    lexists "$sentinel" && { echo "immutable full output sentinel already exists; refusing to overwrite" >&2; exit 1; }
  done
fi
TELEMETRY_STOP="$LOG_DIR/.telemetry-stop-$MODE-$$"
! lexists "$TELEMETRY_STOP" || { echo "telemetry stop sentinel already exists" >&2; exit 1; }
MISSING=()
if [[ "$MODE" == "full" ]]; then
  for fold in 0 1 2 3 4; do
    state="$(PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --fold-status "$fold")"
    if [[ "$state" == *'"status": "missing"'* ]]; then MISSING+=("$fold")
    elif [[ "$state" != *'"status": "completed_immutable"'* ]]; then
      echo "unknown fold continuation state for fold $fold" >&2; exit 1
    fi
  done
  for gpu in 0 1 2 3; do
    planned=(); for fold in "${MISSING[@]}"; do [[ "$fold" == "$gpu" || ( "$gpu" == 0 && "$fold" == 4 ) ]] && planned+=("$fold"); done
    [[ "${#planned[@]}" -gt 0 ]] || continue
    planned_log="$LOG_DIR/full-gpu${gpu}-folds$(IFS=-; echo "${planned[*]}").attempt-${ATTEMPT_TAG}.log"
    ! lexists "$planned_log" || { echo "full attempt log sentinel already exists" >&2; exit 1; }
  done
  ! lexists "$LOG_DIR/full-aggregate.attempt-${ATTEMPT_TAG}.log" || {
    echo "full aggregate log sentinel already exists" >&2; exit 1;
  }
fi

CHECK_GPUS=(0); [[ "$MODE" == "full" ]] && CHECK_GPUS=(0 1 2 3)
COORD_DIR="$ROOT/outputs/reservations/gpu0-3-watchdog-coordination-v1"; mkdir -p "$COORD_DIR"
LOCK_FDS=()
for gpu in "${CHECK_GPUS[@]}"; do
  fd=$((200 + gpu)); eval "exec ${fd}>\"$COORD_DIR/gpu${gpu}.lock\""
  flock -n "$fd" || { echo "coordination lock is held for physical GPU $gpu" >&2; exit 1; }
  LOCK_FDS+=("$fd")
done
# Non-destructive gates for the stopped 004 lineage and the explicitly named
# delayed 005 successor. The latter observes the shared coordination locks.
IFS=,; SELECTED_CSV="${CHECK_GPUS[*]}"; unset IFS
for scheduler in \
  "outputs/legacy/vllm-idle-scheduler/vllm-idle-arm-gpu0-3-20260803-004/state.json|vllm-soak-gpu0-3-120h-20260803-004" \
  "outputs/legacy/vllm-idle-scheduler/vllm-idle-arm-gpu0-3-20260803-005/state.json|vllm-soak-gpu0-3-120h-20260803-005"; do
  state="${scheduler%%|*}"; run_id="${scheduler#*|}"
  [[ ! -e "$state" && ! -L "$state" ]] || PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" \
    --scheduler-state "$state" --scheduler-run-id "$run_id" --selected-gpus "$SELECTED_CSV" >/dev/null
done
for gpu in "${CHECK_GPUS[@]}"; do
  processes="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  [[ -z "$processes" ]] || { echo "GPU $gpu has a pre-existing compute process" >&2; exit 1; }
done
mkdir -p "$LOG_DIR"
PIDS=(); TELEMETRY_PID=""; FAILED_ARMED=1
remove_tracked_pid() {
  local target="$1" pid; local remaining=()
  for pid in "${PIDS[@]}"; do [[ "$pid" == "$target" ]] || remaining+=("$pid"); done
  PIDS=("${remaining[@]}")
}
wait_tracked_pid() {
  local pid="$1" rc=0
  wait "$pid" || rc=$?
  remove_tracked_pid "$pid"
  return "$rc"
}
append_event() {
  local kind="$1" rc="${2:-0}"
  "$PYTHON" - "$LEDGER" "$kind" "$rc" "$MODE" "$RUN_DIR" "$ATTEMPT_TAG" <<'PY'
import hashlib,json,os,sys
import setproctitle
from datetime import datetime,timezone
from pathlib import Path
ledger,kind,rc,mode,root,attempt=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]),sys.argv[4],Path(sys.argv[5]),sys.argv[6]
setproctitle.setproctitle(f'mal2026:lds:ledger:{mode}:{kind}')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
files=[]
for path in sorted((root/'logs').glob('*')) if (root/'logs').exists() else []:
 if path.is_file() and not path.is_symlink(): files.append({'path':str(path.resolve()),'sha256':sha(path),'bytes':path.stat().st_size})
e={'event':kind,'mode':mode,'exit_code':rc,'at':datetime.now(timezone.utc).replace(microsecond=0).isoformat(),'existing_artifacts':files}
config_path=Path('configs/kure_lds_oof.v1.json'); config=json.loads(config_path.read_text())
e.update({'run_id':config['run_id'],'git_sha':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'config_file_sha256':sha(config_path),'task_card_sha256':config['task_card_sha256'],
          'exact_launcher_command':f'bash scripts/run_kure_lds_oof_gpu0_3.sh {mode}',
          'training_performed':True,'integration_recovery_policy':config['integration_recovery_policy'],
          'attempt_tag':attempt,'validation_selection':False,'average_target_used':False})
telemetry=(root/'logs'/('gpu-telemetry-smoke.csv' if mode=='smoke' else f'gpu-telemetry-full-{attempt}.csv'))
summary=(root/'logs'/('gpu-telemetry-smoke-summary.json' if mode=='smoke' else f'gpu-telemetry-full-{attempt}-summary.json'))
e['telemetry_path']=str(telemetry.resolve()); e['telemetry_summary_path']=str(summary.resolve())
if summary.is_file() and not summary.is_symlink():
 e['telemetry_summary']=json.loads(summary.read_text()); e['telemetry_summary_sha256']=sha(summary)
if kind == 'stage_complete':
 evidence=root/('smoke/outer-00.json' if mode == 'smoke' else 'aggregate.json')
 if not evidence.is_file() or evidence.is_symlink(): raise SystemExit('stage-complete evidence is missing')
 e['evidence_ref']=str(evidence.resolve()); e['evidence_sha256']=sha(evidence)
ledger.parent.mkdir(parents=True,exist_ok=True)
fd=os.open(ledger,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o644)
with os.fdopen(fd,'a') as f: f.write(json.dumps(e,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
PY
}
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  local pid; local live_output; declare -A live_jobs=()
  live_output="$(jobs -pr)"
  while IFS= read -r pid; do [[ -n "$pid" ]] && live_jobs["$pid"]=1; done <<<"$live_output"
  for pid in "${PIDS[@]}"; do
    if [[ -n "${live_jobs[$pid]:-}" ]]; then
      pkill -TERM -P "$pid" 2>/dev/null || :
      kill "$pid" 2>/dev/null || :
    fi
  done
  if [[ -n "$TELEMETRY_PID" ]]; then touch "$TELEMETRY_STOP"; wait "$TELEMETRY_PID" 2>/dev/null || :; fi
  rm -f "$TELEMETRY_STOP"
  if (( FAILED_ARMED )); then append_event stage_failed "$rc" || :; fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$MODE" == "full" ]]; then
  "$PYTHON" - "$RUN_DIR/smoke/outer-00.json" "$RUN_DIR/smoke/attestation.json" "$RUN_DIR/logs/gpu-telemetry-smoke-summary.json" "$CONFIG" "$RUNNER" "$MODULE" "$PREPARER" "$LAUNCHER" "$TEST" <<'PY'
import hashlib,json,sys
import setproctitle
from pathlib import Path
report_path,attestation_path,summary_path,*paths=map(Path,sys.argv[1:]); report=json.loads(report_path.read_text()); attestation=json.loads(attestation_path.read_text()); config=json.loads(paths[0].read_text())
setproctitle.setproctitle('mal2026:lds:smoke-attestation-check')
assert report["status"]=="completed" and report["mode"]=="smoke" and report["nonselectable"] is True
assert report["task_card_sha256"]==attestation["task_card_sha256"]==config["task_card_sha256"]
assert report["config_file_sha256"]==attestation["config_sha256"]==hashlib.sha256(paths[0].read_bytes()).hexdigest()
assert report["validation_rows_loaded"] is attestation["validation_rows_loaded"] is False
assert report["average_target_used"] is attestation["average_target_used"] is False
assert attestation["smoke_report_sha256"]==hashlib.sha256(report_path.read_bytes()).hexdigest()
assert attestation["telemetry_summary_sha256"]==hashlib.sha256(summary_path.read_bytes()).hexdigest()
for name,path in zip(("config","runner","module","preparer","launcher","test"),paths,strict=True): assert attestation[f"{name}_sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
PY
fi
append_event stage_launch 0
if [[ "$MODE" == "smoke" ]]; then
  TELEMETRY="$LOG_DIR/gpu-telemetry-smoke.csv"; SUMMARY="$LOG_DIR/gpu-telemetry-smoke-summary.json"
else
  TELEMETRY="$LOG_DIR/gpu-telemetry-full-${ATTEMPT_TAG}.csv"; SUMMARY="$LOG_DIR/gpu-telemetry-full-${ATTEMPT_TAG}-summary.json"
fi
! lexists "$TELEMETRY" && ! lexists "$SUMMARY" || { echo "refusing to overwrite telemetry" >&2; exit 1; }
"$PYTHON" - "$TELEMETRY" "$TELEMETRY_STOP" "${CHECK_GPUS[*]}" "$$" <<'PY' &
import csv,os,signal,subprocess,sys,time
import setproctitle
from pathlib import Path
out,stop,raw,parent=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]); selected={int(x) for x in raw.split()}
setproctitle.setproctitle(f'mal2026:lds:telemetry:{"-".join(map(str,sorted(selected)))}')
fields=['timestamp','index','uuid','name','memory.total','driver_version','utilization.gpu','memory.used']
try:
 with out.open('x',newline='') as f:
  w=csv.writer(f); w.writerow(fields); f.flush(); os.fsync(f.fileno())
  while not stop.exists():
   text=subprocess.check_output(['nvidia-smi',f'--id={",".join(map(str,sorted(selected)))}',f'--query-gpu={",".join(fields)}','--format=csv,noheader,nounits'],text=True,timeout=15)
   rows=list(csv.reader(text.splitlines())); observed={int(row[1].strip()) for row in rows}
   if observed != selected or len(rows) != len(selected): raise RuntimeError('selected GPU telemetry coverage differs')
   w.writerows(rows); f.flush(); os.fsync(f.fileno()); time.sleep(30)
except BaseException:
 os.kill(parent,signal.SIGTERM)
 raise
PY
TELEMETRY_PID="$!"

if [[ "$MODE" == "smoke" ]]; then
  LOG="$LOG_DIR/smoke-gpu0.log"; ! lexists "$LOG" || { echo "refusing to overwrite $LOG" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --outer-fold 0 --smoke >"$LOG" 2>&1 & PIDS+=("$!")
  wait_tracked_pid "${PIDS[0]}"
else
  [[ "${#MISSING[@]}" -gt 0 ]] || echo "all five immutable folds already complete; aggregating only"
  for gpu in 0 1 2 3; do
    folds=(); for fold in "${MISSING[@]}"; do [[ "${fold}" == "$gpu" || ( "$gpu" == 0 && "$fold" == 4 ) ]] && folds+=("$fold"); done
    [[ "${#folds[@]}" -gt 0 ]] || continue
    log="$LOG_DIR/full-gpu${gpu}-folds$(IFS=-; echo "${folds[*]}").attempt-${ATTEMPT_TAG}.log"
    ! lexists "$log" || { echo "refusing to overwrite $log" >&2; exit 1; }
    (for fold in "${folds[@]}"; do CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --outer-fold "$fold"; done) >"$log" 2>&1 & PIDS+=("$!")
  done
  worker_pids=("${PIDS[@]}"); failure=0
  for pid in "${worker_pids[@]}"; do wait_tracked_pid "$pid" || failure=1; done
  [[ "$failure" == 0 ]] || { echo "one or more workers failed" >&2; exit 1; }
  aggregate_log="$LOG_DIR/full-aggregate.attempt-${ATTEMPT_TAG}.log"; ! lexists "$aggregate_log" || exit 1
  PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --aggregate >"$aggregate_log" 2>&1 & PIDS+=("$!")
  wait_tracked_pid "${PIDS[0]}"
fi
touch "$TELEMETRY_STOP"; wait "$TELEMETRY_PID"; TELEMETRY_PID=""; rm -f "$TELEMETRY_STOP"
MIN_SAMPLES=1; [[ "$MODE" == "full" ]] && MIN_SAMPLES=2
PYTHONPATH=src "$PYTHON" "$RUNNER" --config "$CONFIG" --telemetry-csv "$TELEMETRY" --telemetry-summary "$SUMMARY" --selected-gpus "$SELECTED_CSV" --minimum-samples "$MIN_SAMPLES" >/dev/null

if [[ "$MODE" == "smoke" ]]; then
  "$PYTHON" - "$RUN_DIR/smoke/outer-00.json" "$RUN_DIR/smoke/attestation.json" "$SUMMARY" "$CONFIG" "$RUNNER" "$MODULE" "$PREPARER" "$LAUNCHER" "$TEST" <<'PY'
import hashlib,json,os,sys,tempfile
import setproctitle
from pathlib import Path
report,output,summary,*paths=map(Path,sys.argv[1:]); assert not (output.exists() or output.is_symlink()); result=json.loads(report.read_text()); config=json.loads(paths[0].read_text())
setproctitle.setproctitle('mal2026:lds:smoke-attestation-write')
assert result['task_card_sha256']==config['task_card_sha256'] and result['config_file_sha256']==hashlib.sha256(paths[0].read_bytes()).hexdigest()
assert result['validation_rows_loaded'] is False and result['average_target_used'] is False
value={'schema_version':'mal2026-kure-coral-lds-smoke-attestation-v1','physical_gpu':0,'nonselectable':True,'smoke_report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'telemetry_summary_sha256':hashlib.sha256(summary.read_bytes()).hexdigest(),'task_card_sha256':result['task_card_sha256'],'config_sha256':result['config_file_sha256'],'validation_rows_loaded':False,'average_target_used':False}
for name,path in zip(('config','runner','module','preparer','launcher','test'),paths,strict=True): value[f'{name}_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
output.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f'.{output.name}.',dir=output.parent,text=True)
with os.fdopen(fd,'w') as f: f.write(json.dumps(value,indent=2,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
os.link(tmp,output); os.unlink(tmp); d=os.open(output.parent,os.O_DIRECTORY); os.fsync(d); os.close(d)
PY
fi
append_event stage_complete 0
FAILED_ARMED=0
trap - EXIT INT TERM
