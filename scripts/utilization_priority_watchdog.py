#!/usr/bin/env python3
"""GPU 0--3 utilization queue with safe priority and shared ownership."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

from gpu_watchdog_coordination import GpuLease, clear_priority_request, coordination_dir, gpu_has_compute_process, request_priority

ALLOWED = {0, 1, 2, 3}; STOP = False


def now() -> str: return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def stop_handler(*_: object) -> None:
    global STOP; STOP = True
def need(value: bool, message: str) -> None:
    if not value: raise SystemExit(message)
def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8")); need(isinstance(value, dict), "plan must be an object"); return value


def command_ok(command: Any, gpu: int, runtime: Path, kind: str) -> bool:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command): return False
    if kind == "backfill": return command[1:3] == ["scripts/gpu_local_utilization_backfill.py", "--physical-gpu"] and str(gpu) in command
    if kind == "score_utilization": return len(command) == 4 and command[1:3] == ["scripts/train_pre_sft_score_head.py", "--config"] and Path(command[3]).resolve().parent == (runtime / "configs").resolve()
    if kind == "judge_v3": return len(command) == 7 and command[:2] == ["bash", "scripts/run_qwen36_judge_v3_checked.sh"]
    if kind == "frozen_aggregation": return len(command) == 4 and command[1:3] == ["scripts/evaluate_pre_sft_score_ensemble_validation.py", "--config"] and Path(command[3]).resolve().parent == (runtime / "configs").resolve()
    return False


def validate(plan: dict[str, Any]) -> None:
    runtime = Path(plan.get("runtime_dir", "")).resolve(); root = Path(plan.get("project_root", "")).resolve()
    need(plan.get("schema_version") == "mal2026-utilization-priority-v1", "unsupported plan schema")
    need(root.is_dir() and runtime.parent == root / "outputs" / "reservations", "runtime containment failed")
    need(plan.get("allowed_physical_gpus") == [0, 1, 2, 3], "GPU allowlist must be exactly 0--3")
    need(plan.get("python") == str(root / ".venv-standard" / "bin" / "python"), "project interpreter mismatch")
    need(isinstance(plan.get("end_at_utc"), str) and isinstance(plan.get("poll_seconds"), int) and 5 <= plan["poll_seconds"] <= 60 and isinstance(plan.get("max_temperature_c"), int) and 40 <= plan["max_temperature_c"] <= 90, "invalid time/poll/temperature policy")
    entries = plan.get("gpus"); need(isinstance(entries, dict) and {int(item) for item in entries} == ALLOWED, "must define exactly GPUs 0--3")
    for raw_gpu, entry in entries.items():
        gpu = int(raw_gpu); need(isinstance(entry, dict) and entry.get("gpu") == gpu, "GPU entry mismatch")
        util = entry.get("utilization"); need(isinstance(util, dict) and util.get("run_purpose") == "utilization_only" and util.get("gpu") == gpu and command_ok(util.get("command"), gpu, runtime, str(util.get("kind"))), "invalid utilization command")
        priorities = entry.get("priority", []); need(isinstance(priorities, list), "priority queue must be a list")
        for job in priorities: need(isinstance(job, dict) and job.get("gpu") == gpu and job.get("run_purpose") == "higher_priority_research" and isinstance(job.get("ready_file"), str) and Path(job["ready_file"]).resolve().is_relative_to(root / "outputs") and command_ok(job.get("command"), gpu, runtime, str(job.get("kind"))), "invalid priority command")


def health(gpu: int) -> dict[str, int]:
    result = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=index,temperature.gpu", "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True, timeout=15)
    need(result.returncode == 0 and result.stdout.strip(), f"GPU {gpu} health query failed"); parts = [item.strip() for item in result.stdout.strip().split(",")]
    need(len(parts) == 2 and parts[0] == str(gpu), f"GPU {gpu} health identity failed"); return {"index": int(parts[0]), "temperature_c": int(parts[1])}


def launch(job: dict[str, Any], gpu: int, root: Path, log: Path) -> subprocess.Popen[str]:
    environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = str(gpu); environment["MAL2026_RESERVED_PHYSICAL_GPU"] = str(gpu); environment["PYTHONPATH"] = str(root / "src")
    with log.open("a", encoding="utf-8") as handle: handle.write(json.dumps({"at": now(), "run_purpose": job["run_purpose"], "gpu": gpu, "kind": job["kind"]}, sort_keys=True) + "\n")
    return subprocess.Popen(job["command"], cwd=root, env=environment, stdin=subprocess.DEVNULL, stdout=log.open("a", encoding="utf-8"), stderr=subprocess.STDOUT, start_new_session=True, text=True)


def yield_backfill(process: subprocess.Popen[str]) -> None:
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: pass
def write(path: Path, value: dict[str, Any]) -> None:
    value["updated_at"] = now(); temporary = path.with_suffix(".tmp"); temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--plan", type=Path, required=True); parser.add_argument("--coordination-dir", type=Path); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); plan = load(args.plan); validate(plan)
    if args.dry_run: print(json.dumps({"status": "validated", "run_purpose": "utilization_only", "plan": str(args.plan)}, sort_keys=True)); return
    root = Path(plan["project_root"]).resolve(); runtime = Path(plan["runtime_dir"]).resolve(); coordinator = coordination_dir(root, args.coordination_dir or plan.get("coordination_dir")); logs = runtime / "logs"; logs.mkdir(parents=True, exist_ok=True)
    state_path = runtime / "utilization_only_state.json"; state: dict[str, Any] = {"schema_version": "mal2026-utilization-only-state-v2", "run_purpose": "utilization_only", "coordination_dir": str(coordinator), "gpus": {str(g): {"status": "pending", "pinning": {"CUDA_VISIBLE_DEVICES": str(g), "MAL2026_RESERVED_PHYSICAL_GPU": str(g)}} for g in sorted(ALLOWED)}}
    active: dict[int, tuple[dict[str, Any], subprocess.Popen[str], GpuLease]] = {}; used: set[int] = set(); signal.signal(signal.SIGTERM, stop_handler); signal.signal(signal.SIGINT, stop_handler); deadline = datetime.fromisoformat(plan["end_at_utc"])
    while not STOP and datetime.now(timezone.utc) < deadline:
        for gpu in sorted(ALLOWED):
            entry = plan["gpus"][str(gpu)]
            try:
                state["gpus"][str(gpu)]["health"] = health(gpu)
                if state["gpus"][str(gpu)]["health"]["temperature_c"] > plan["max_temperature_c"]: raise RuntimeError("temperature threshold exceeded")
            except Exception as exc:
                state["gpus"][str(gpu)]["status"] = "safety_stop"; state["gpus"][str(gpu)]["reason"] = str(exc); used.add(gpu); continue
            ready = next((job for job in entry["priority"] if Path(job["ready_file"]).is_file()), None)
            if gpu in active:
                job, process, lease = active[gpu]
                if ready is not None and job["kind"] == "backfill": yield_backfill(process); state["gpus"][str(gpu)]["status"] = "backfill_yield_requested"; state["gpus"][str(gpu)]["yield_job"] = ready["kind"]; continue
                if process.poll() is None:
                    state["gpus"][str(gpu)]["status"] = "priority_waiting_for_non_yieldable_utilization" if ready is not None and job["run_purpose"] == "utilization_only" else job["run_purpose"]
                    if ready is not None: state["gpus"][str(gpu)]["yield_job"] = ready["kind"]
                    continue
                active.pop(gpu); lease.release(); state["gpus"][str(gpu)]["status"] = "utilization_exited" if job["run_purpose"] == "utilization_only" else "priority_exited"; used.add(gpu)
            if ready is not None:
                if gpu_has_compute_process(gpu): request_priority(coordinator, gpu, "utilization_priority_watchdog", ready["kind"]); state["gpus"][str(gpu)]["status"] = "external_owner_waiting_priority"; continue
                lease = GpuLease.acquire(coordinator, gpu, "utilization_priority_watchdog", "higher_priority_research", ready["kind"])
                if lease is None: request_priority(coordinator, gpu, "utilization_priority_watchdog", ready["kind"]); state["gpus"][str(gpu)]["status"] = "coordination_owner_waiting_priority"; continue
                active[gpu] = (ready, launch(ready, gpu, root, logs / f"gpu{gpu}-priority.log"), lease); clear_priority_request(coordinator, gpu, "utilization_priority_watchdog"); state["gpus"][str(gpu)]["status"] = "higher_priority_research"; used.add(gpu); continue
            if gpu not in used:
                util = entry["utilization"]
                if gpu_has_compute_process(gpu): state["gpus"][str(gpu)]["status"] = "external_owner_waiting_utilization"; continue
                lease = GpuLease.acquire(coordinator, gpu, "utilization_priority_watchdog", "utilization_only", util["kind"])
                if lease is None: state["gpus"][str(gpu)]["status"] = "coordination_owner_waiting_utilization"; continue
                active[gpu] = (util, launch(util, gpu, root, logs / f"gpu{gpu}-utilization_only.log"), lease); state["gpus"][str(gpu)]["status"] = "utilization_only"
        write(state_path, state); time.sleep(plan["poll_seconds"])
    for _, process, lease in active.values():
        if lease.kind == "backfill": yield_backfill(process)
        lease.release()
    for gpu in ALLOWED: clear_priority_request(coordinator, gpu, "utilization_priority_watchdog")
    state["status"] = "stopped" if STOP else "deadline_reached"; write(state_path, state)


if __name__ == "__main__": main()
