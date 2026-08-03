#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from mal2026.kure_lds_oof import (KURELDSOOFConfig,_atomic_public_json,aggregate,fold_status,run,
    scheduler_state_conflict,set_process_title,summarize_gpu_telemetry)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True); g.add_argument("--outer-fold",type=int,choices=range(5)); g.add_argument("--aggregate",action="store_true")
    g.add_argument("--check-authorization",action="store_true"); g.add_argument("--telemetry-csv",type=Path); g.add_argument("--scheduler-state",type=Path)
    g.add_argument("--fold-status",type=int,choices=range(5))
    p.add_argument("--telemetry-summary",type=Path); p.add_argument("--selected-gpus"); p.add_argument("--minimum-samples",type=int)
    p.add_argument("--scheduler-run-id"); p.add_argument("--validate-only",action="store_true"); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
    if a.outer_fold is not None: set_process_title(f"cli:{'smoke' if a.smoke else 'oof'}:f{a.outer_fold}")
    elif a.aggregate: set_process_title("cli:aggregate")
    elif a.telemetry_csv: set_process_title("cli:telemetry-summary")
    elif a.scheduler_state: set_process_title("cli:scheduler-gate")
    elif a.fold_status is not None: set_process_title(f"cli:fold-status:f{a.fold_status}")
    else: set_process_title("cli:authorization-gate")
    c=KURELDSOOFConfig.from_json(a.config)
    if a.scheduler_state:
        if not a.selected_gpus or not a.scheduler_run_id: p.error("scheduler check requires --selected-gpus and --scheduler-run-id")
        selected=tuple(map(int,a.selected_gpus.split(",")))
        if a.scheduler_state.exists() or a.scheduler_state.is_symlink():
            state=json.loads(a.scheduler_state.read_text()); reason=scheduler_state_conflict(
                state,selected,age_seconds=time.time()-a.scheduler_state.stat().st_mtime,
                expected_run_id=a.scheduler_run_id)
            if reason: raise SystemExit(reason)
        result={"status":"scheduler_safe","selected_gpus":selected}
    elif a.telemetry_csv:
        if not a.telemetry_summary or not a.selected_gpus or a.minimum_samples is None: p.error("telemetry arguments incomplete")
        result=summarize_gpu_telemetry(a.telemetry_csv,tuple(map(int,a.selected_gpus.split(","))),a.minimum_samples); _atomic_public_json(a.telemetry_summary,result)
    elif a.check_authorization: c.require_execution_authorization(preflight_all_fit=True); result={"status":"authorized","task_card_sha256":c.task_card_sha256}
    elif a.fold_status is not None: c.require_execution_authorization(); result={"outer_fold":a.fold_status,"status":fold_status(c,a.fold_status)}
    elif a.aggregate: result=aggregate(c)
    else: result=run(c,outer_fold=a.outer_fold,validate_only=a.validate_only,smoke=a.smoke)
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__": main()
