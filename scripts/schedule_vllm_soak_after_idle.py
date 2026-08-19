#!/usr/bin/env python3
"""Arm a bounded GPU-idle trigger for the synthetic vLLM soak."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("SPT_NOENV", "1")
import setproctitle


ROOT = Path(__file__).resolve().parents[1]
GPUS = (0, 1, 2, 3)


def now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def gpu_snapshot() -> list[dict[str, int]]:
    rows = []
    for gpu in GPUS:
        output = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}",
             "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=10,
        ).strip()
        index, memory, utilization, temperature = [int(x.strip()) for x in output.split(",")]
        if index != gpu:
            raise RuntimeError("selected-GPU telemetry mismatch")
        rows.append({"gpu": gpu, "memory_used_mib": memory,
                     "utilization_percent": utilization, "temperature_c": temperature})
    return rows


def main() -> None:
    setproctitle.setproctitle("(D)_vllm")
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--delay-hours", type=int, required=True)
    parser.add_argument("--arming-hours", type=int, required=True)
    parser.add_argument("--idle-minutes", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if (args.delay_hours, args.arming_hours, args.idle_minutes) != (0, 48, 60):
        raise SystemExit("scheduler permits exactly immediate monitoring, a 48-hour arm, and 60-minute idle gate")
    if args.schedule_id != "vllm-idle-arm-gpu0-3-20260807-007":
        raise SystemExit("unexpected schedule lineage")
    if args.run_id != "vllm-soak-gpu0-3-120h-20260807-007":
        raise SystemExit("unexpected five-day run lineage")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if (cfg.get("schema_version") != "mal2026-vllm-synthetic-soak-gpu0-3-120h-v1"
            or cfg["runtime"].get("duration_seconds") != 432000
            or cfg["runtime"].get("physical_gpus") != list(GPUS)):
        raise SystemExit("five-day config is outside the authorized envelope")

    runtime = ROOT / "outputs/legacy/vllm-idle-scheduler" / args.schedule_id
    runtime.mkdir(parents=True, exist_ok=False)
    state_path = runtime / "state.json"
    ledger_path = runtime / "append_only_ledger.log"
    started = now()
    monitoring_starts = started + timedelta(hours=args.delay_hours)
    deadline = monitoring_starts + timedelta(hours=args.arming_hours)
    idle_since: datetime | None = None
    observations = 0
    faults = 0
    with ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(f"{started.isoformat()} | delayed | GPUs 0-3 | delay_hours=0 | arm_hours=48 | idle_minutes=60 | run_hours=120\n")

    while now() < monitoring_starts:
        atomic_json(state_path, {
            "schema_version": "mal2026-vllm-idle-arm-state-v1", "status": "delayed",
            "started_at": started.isoformat(), "monitoring_starts_at": monitoring_starts.isoformat(),
            "arming_deadline": deadline.isoformat(), "updated_at": now().isoformat(),
            "physical_gpus": list(GPUS), "idle_required_seconds": 3600,
            "consecutive_idle_seconds": 0, "observations": 0, "telemetry_faults": 0,
            "planned_run_id": args.run_id, "planned_duration_seconds": 432000,
        })
        time.sleep(min(30, max(1, int((monitoring_starts - now()).total_seconds()))))
    with ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(f"{now().isoformat()} | monitoring_started | deadline={deadline.isoformat()}\n")

    while now() < deadline:
        observed_at = now()
        try:
            snapshot = gpu_snapshot()
            safe = all(row["temperature_c"] <= 80 for row in snapshot)
            idle = safe and all(row["memory_used_mib"] == 0 and row["utilization_percent"] == 0 for row in snapshot)
            if idle:
                idle_since = idle_since or observed_at
            else:
                idle_since = None
            observations += 1
            idle_seconds = 0 if idle_since is None else (observed_at - idle_since).total_seconds()
            atomic_json(state_path, {
                "schema_version": "mal2026-vllm-idle-arm-state-v1", "status": "armed",
                "started_at": started.isoformat(), "arming_deadline": deadline.isoformat(),
                "updated_at": observed_at.isoformat(), "physical_gpus": list(GPUS),
                "idle_required_seconds": 3600, "consecutive_idle_seconds": idle_seconds,
                "observations": observations, "telemetry_faults": faults, "last_gpu_snapshot": snapshot,
                "planned_run_id": args.run_id, "planned_duration_seconds": 432000,
            })
            if idle_seconds >= 3600:
                session = "mal2026-vllm-soak-gpu0-3-120h-20260807-007"
                if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
                    raise RuntimeError("planned run session already exists")
                launcher = ROOT / "scripts/run_vllm_synthetic_soak_gpu0_3_48h.sh"
                launch_log = ROOT / "outputs/legacy-vllm-soak-launch-20260807-007.log"
                command = (
                    f"cd {ROOT} && export MAL2026_VLLM_SOAK_CONFIG={args.config.resolve()} && "
                    f"exec bash {launcher} {args.run_id} > {launch_log} 2>&1"
                )
                subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)
                with ledger_path.open("a", encoding="utf-8") as ledger:
                    ledger.write(f"{now().isoformat()} | launched | session={session} | idle_seconds={idle_seconds}\n")
                atomic_json(state_path, {
                    "schema_version": "mal2026-vllm-idle-arm-state-v1", "status": "launched",
                    "launched_at": now().isoformat(), "physical_gpus": list(GPUS),
                    "qualifying_idle_seconds": idle_seconds, "observations": observations,
                    "telemetry_faults": faults, "run_id": args.run_id, "tmux_session": session,
                })
                return
        except Exception as exc:
            faults += 1
            idle_since = None
            with ledger_path.open("a", encoding="utf-8") as ledger:
                ledger.write(f"{now().isoformat()} | telemetry_or_launch_fault | category={type(exc).__name__}\n")
        time.sleep(30)

    with ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(f"{now().isoformat()} | expired_without_launch | observations={observations} | faults={faults}\n")
    atomic_json(state_path, {
        "schema_version": "mal2026-vllm-idle-arm-state-v1", "status": "expired_without_launch",
        "expired_at": now().isoformat(), "physical_gpus": list(GPUS),
        "observations": observations, "telemetry_faults": faults, "planned_run_id": args.run_id,
    })


if __name__ == "__main__":
    main()
