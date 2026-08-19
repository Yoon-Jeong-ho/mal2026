#!/usr/bin/env python3
"""GPU 0--3 reservation watchdog with shared exclusive ownership leases."""
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
from zoneinfo import ZoneInfo

from gpu_watchdog_coordination import GpuLease, clear_priority_request, coordination_dir, gpu_has_compute_process, request_priority

ALLOWED = {0, 1, 2, 3}
KST = ZoneInfo("Asia/Seoul")
STOP = False


def stop_handler(*_: object) -> None:
    global STOP
    STOP = True


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), "reservation plan must be a JSON object")
    return value


def validate(plan: dict[str, Any]) -> None:
    need(plan.get("schema_version") == "mal2026-gpu-reservation-v1", "unsupported plan schema")
    need(plan.get("allowed_physical_gpus") == [0, 1, 2, 3], "plan must be exactly GPUs 0-3")
    need(plan.get("not_before_kst") == "2026-07-20T00:00:00+09:00", "plan must start at the approved midnight KST")
    need(plan.get("reservation_end_kst") == "2026-07-21T00:00:00+09:00", "plan must have the fixed bounded reservation end")
    root = Path(plan.get("project_root", "")).resolve(); runtime = Path(plan.get("runtime_dir", "")).resolve()
    need(root.is_dir() and runtime.parent == root / "outputs" / "reservations", "plan root/runtime containment failed")
    need(plan.get("python") == str(root / ".venv-standard" / "bin" / "python"), "plan must use the documented project interpreter")
    need(isinstance(plan.get("poll_seconds"), int) and 5 <= plan["poll_seconds"] <= 60, "invalid polling interval")
    backfill = plan.get("backfill")
    need(isinstance(backfill, dict) and set(backfill) == {"seconds", "memory_fraction", "max_temperature_c", "max_cycles_per_gpu"}, "invalid bounded backfill policy")
    need(isinstance(backfill["seconds"], int) and 1 <= backfill["seconds"] <= 900 and isinstance(backfill["memory_fraction"], (int, float)) and 0.05 <= float(backfill["memory_fraction"]) <= 0.40 and isinstance(backfill["max_temperature_c"], int) and 40 <= backfill["max_temperature_c"] <= 90 and isinstance(backfill["max_cycles_per_gpu"], int) and 1 <= backfill["max_cycles_per_gpu"] <= 288, "backfill policy is outside the bounded safety envelope")
    queues = plan.get("queues")
    need(isinstance(queues, dict) and {int(key) for key in queues} == ALLOWED, "plan queues must be exactly 0-3")
    seen: set[str] = set()
    for raw_gpu, jobs in queues.items():
        gpu = int(raw_gpu); need(gpu in ALLOWED and isinstance(jobs, list), "invalid GPU queue")
        for job in jobs:
            need(isinstance(job, dict) and isinstance(job.get("id"), str) and job["id"] not in seen, "invalid/duplicate job ID")
            seen.add(job["id"]); command = job.get("command")
            need(job.get("gpu") == gpu and isinstance(command, list) and all(isinstance(x, str) for x in command), "job violates GPU-local command contract")
            need(job.get("kind") in {"judge_v2", "pre_sft_score_head", "ensemble", "replication", "ablation"}, "unapproved research job kind")
            need(isinstance(job.get("requires_files", []), list) and all(isinstance(x, str) and x.startswith("/") for x in job.get("requires_files", [])), "invalid job prerequisite")
            need(not {"scripts/train_standard_decoder_sft.py", "dpo", "grpo"}.intersection(command), "SFT/DPO/GRPO command is forbidden")
            if job["kind"] == "judge_v2":
                need(command[:2] in (["bash", "scripts/run_qwen36_judge_v2_checked.sh"], ["bash", "scripts/run_qwen36_judge_v3_checked.sh"]) and len(command) in {6, 7}, "invalid judge command")
            else:
                expected = "scripts/train_pre_sft_score_head.py" if job["kind"] in {"pre_sft_score_head", "replication", "ablation"} else None
                if expected is None:
                    need(command[:2] in ([plan["python"], "scripts/evaluate_pre_sft_score_ensemble.py"], [plan["python"], "scripts/evaluate_pre_sft_score_ensemble_validation.py"]) and len(command) == 4 and command[2] == "--config", "invalid ensemble command")
                else:
                    need(command[:2] == [plan["python"], expected] and len(command) == 4 and command[2] == "--config", "invalid score-model command")
                need(Path(command[3]).resolve().parent == runtime / "configs", "research config escapes reservation runtime")
            need(all(Path(item).resolve().is_relative_to(root / "outputs") for item in job.get("requires_files", [])), "job prerequisite escapes ignored outputs")


def write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now(); temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(temporary, path)


def gpu_health(gpu: int) -> dict[str, str]:
    result = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True, timeout=15)
    if result.returncode != 0 or not result.stdout.strip(): raise RuntimeError(f"GPU {gpu} health query failed")
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    if len(fields) != 4 or fields[0] != str(gpu): raise RuntimeError(f"GPU {gpu} health identity was not attested")
    return {"index": fields[0], "utilization": fields[1], "memory_used_mib": fields[2], "temperature_c": fields[3]}


def ready(job: dict[str, Any]) -> bool:
    return all(Path(item).is_file() for item in job.get("requires_files", []))


def launch(command: list[str], gpu: int, log: Path, cwd: str) -> subprocess.Popen[str]:
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["MAL2026_RESERVED_PHYSICAL_GPU"] = str(gpu); env["PYTHONPATH"] = str(Path(cwd) / "src")
    with log.open("a", encoding="utf-8") as handle: handle.write(f"[{utc_now()}] launch gpu={gpu} command={json.dumps(command)}\n")
    return subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log.open("a", encoding="utf-8"), stderr=subprocess.STDOUT, start_new_session=True, text=True)


def yield_backfill(process: subprocess.Popen[str]) -> None:
    """Non-destructive yield: bounded backfill handles SIGTERM itself; never SIGKILL."""
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: pass


def parse_kst(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None: raise SystemExit("KST timestamp requires timezone offset")
    return parsed.astimezone(KST)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--plan", type=Path, required=True); parser.add_argument("--coordination-dir", type=Path); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); plan = load(args.plan); validate(plan)
    if args.dry_run:
        print(json.dumps({"status": "validated", "gpu_queries": 0, "plan": str(args.plan), "jobs": sum(len(v) for v in plan["queues"].values())}, sort_keys=True)); return
    not_before = parse_kst(plan["not_before_kst"]); reservation_end = parse_kst(plan["reservation_end_kst"])
    if datetime.now(KST) < not_before: raise SystemExit("refusing pre-reservation execution")
    root_path = Path(plan["project_root"]).resolve(); root = str(root_path); coordinator = coordination_dir(root_path, args.coordination_dir or plan.get("coordination_dir"))
    runtime = Path(plan["runtime_dir"]); logs = runtime / "logs"; heartbeats = runtime / "heartbeats"; logs.mkdir(parents=True, exist_ok=True); heartbeats.mkdir(parents=True, exist_ok=True)
    state_path = runtime / "watchdog_state.json"; state: dict[str, Any] = {"schema_version": "mal2026-gpu-reservation-state-v2", "plan": str(args.plan), "coordination_dir": str(coordinator), "gpus": {str(g): {"status": "pending", "completed_jobs": [], "failed_jobs": []} for g in sorted(ALLOWED)}}
    active: dict[int, tuple[dict[str, Any], subprocess.Popen[str], str, GpuLease]] = {}; cursor = {gpu: 0 for gpu in ALLOWED}; cycles = {gpu: 0 for gpu in ALLOWED}; blocked: set[int] = set(); poll = int(plan["poll_seconds"])
    signal.signal(signal.SIGTERM, stop_handler); signal.signal(signal.SIGINT, stop_handler)
    while not STOP and datetime.now(KST) < reservation_end:
        all_done = True
        for gpu in sorted(ALLOWED):
            if gpu in blocked: continue
            jobs = plan["queues"][str(gpu)]
            try:
                health = gpu_health(gpu); state["gpus"][str(gpu)]["health"] = health
                if int(health["temperature_c"]) > int(plan["backfill"]["max_temperature_c"]): raise RuntimeError("temperature threshold exceeded")
            except Exception as exc:
                state["gpus"][str(gpu)]["status"] = "safety_stop"; state["gpus"][str(gpu)]["reason"] = str(exc); blocked.add(gpu); continue
            if gpu in active:
                job, process, mode, lease = active[gpu]
                if mode == "backfill" and (lease.priority_requested() or (cursor[gpu] < len(jobs) and ready(jobs[cursor[gpu]]))):
                    yield_backfill(process); state["gpus"][str(gpu)]["status"] = "backfill_yield_requested"; all_done = False; continue
                code = process.poll()
                if code is None:
                    state["gpus"][str(gpu)]["status"] = mode; all_done = False; continue
                active.pop(gpu); lease.release(); state["gpus"][str(gpu)]["status"] = "idle"
                if mode == "research":
                    (state["gpus"][str(gpu)]["completed_jobs"] if code == 0 else state["gpus"][str(gpu)]["failed_jobs"]).append(job["id"] if code == 0 else {"id": job["id"], "exit_code": code}); cursor[gpu] += 1
            if cursor[gpu] < len(jobs) and ready(jobs[cursor[gpu]]):
                job = jobs[cursor[gpu]]
                if gpu_has_compute_process(gpu): request_priority(coordinator, gpu, "reservation_watchdog", job["kind"]); state["gpus"][str(gpu)]["status"] = "external_owner_waiting_research"; all_done = False; continue
                lease = GpuLease.acquire(coordinator, gpu, "reservation_watchdog", "higher_priority_research", job["kind"])
                if lease is None: request_priority(coordinator, gpu, "reservation_watchdog", job["kind"]); state["gpus"][str(gpu)]["status"] = "coordination_owner_waiting_research"; all_done = False; continue
                active[gpu] = (job, launch(job["command"], gpu, logs / f"{job['id']}.log", root), "research", lease); clear_priority_request(coordinator, gpu, "reservation_watchdog"); state["gpus"][str(gpu)]["status"] = "research"; state["gpus"][str(gpu)]["active_job"] = job["id"]; all_done = False; continue
            waiting = cursor[gpu] < len(jobs)
            if gpu_has_compute_process(gpu): state["gpus"][str(gpu)]["status"] = "external_owner_waiting" if waiting else "external_owner_after_queue"; all_done = False; continue
            if cycles[gpu] >= int(plan["backfill"]["max_cycles_per_gpu"]): state["gpus"][str(gpu)]["status"] = "backfill_budget_exhausted"; blocked.add(gpu); continue
            lease = GpuLease.acquire(coordinator, gpu, "reservation_watchdog", "utilization_only", "backfill")
            if lease is None: state["gpus"][str(gpu)]["status"] = "coordination_owner_waiting_backfill"; all_done = False; continue
            command = [plan["python"], "scripts/gpu_local_utilization_backfill.py", "--physical-gpu", str(gpu), "--seconds", str(plan["backfill"]["seconds"]), "--memory-fraction", str(plan["backfill"]["memory_fraction"]), "--heartbeat", str(heartbeats / f"gpu{gpu}.json")]
            active[gpu] = ({"id": "backfill"}, launch(command, gpu, logs / f"gpu{gpu}-backfill.log", root), "backfill", lease); cycles[gpu] += 1; state["gpus"][str(gpu)]["status"] = "backfill_waiting_research" if waiting else "backfill_after_queue"; all_done = False
        write_state(state_path, state)
        if all_done: break
        time.sleep(poll)
    for _, process, mode, lease in active.values():
        if mode == "backfill": yield_backfill(process)
        lease.release()
    for gpu in ALLOWED: clear_priority_request(coordinator, gpu, "reservation_watchdog")
    state["status"] = "stopped" if STOP else "completed"; write_state(state_path, state)


if __name__ == "__main__": main()
