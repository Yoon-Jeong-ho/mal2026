#!/usr/bin/env python3
"""Durable fixed orchestrator for the six preregistered KURE axis jobs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/kure-axis-ordinal-contrastive-v1"
RUNTIME = OUTPUT_ROOT / "runtime"
LEDGER = RUNTIME / "ledger.jsonl"
GPU_SAMPLES = RUNTIME / "gpu_samples.jsonl"
PYTHON = ROOT / ".venv-standard/bin/python"
AXES = ("content", "organization", "expression")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def append(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": now(), **value}, sort_keys=True) + "\n")


def ledger(event: str, **fields: Any) -> None:
    append(LEDGER, {"event": event, **fields})


def gpu_snapshot() -> list[dict[str, Any]]:
    command = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True)
    rows = []
    for line in output.splitlines():
        index, uuid, memory, utilization = [part.strip() for part in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "memory_used_mib": int(memory), "utilization_percent": int(utilization)})
    return rows


def preflight() -> None:
    rows = gpu_snapshot()
    selected = [row for row in rows if row["index"] in {0, 1, 2, 3}]
    if len(selected) != 4 or any(row["memory_used_mib"] != 0 for row in selected):
        raise RuntimeError(f"GPU0-3 conflict: {selected}")
    expected = [
        OUTPUT_ROOT / f"kure-axis-contrastive-{'aihub' if arm == 'aihub_full_backbone' else 'base'}-{axis}-20260802-001"
        for arm in ("base", "aihub_full_backbone") for axis in AXES
    ]
    if any(path.exists() for path in expected):
        raise RuntimeError("a full output directory already exists; refusing overwrite")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    ledger("full_preflight_completed", git_sha=git_sha, gpu_scope=[0, 1, 2, 3],
           gpu_scope_authorization="repository default plus current user experiment authorization", gpu_state=selected)


def config(arm: str, axis: str) -> Path:
    tag = "aihub" if arm == "aihub_full_backbone" else "base"
    return ROOT / f"configs/kure_axis_contrastive.{tag}.{axis}.v1.json"


def launch_batch(tasks: Sequence[tuple[int, str, str]]) -> None:
    running: list[tuple[subprocess.Popen[str], Any, int, str, str, Path]] = []
    for gpu, arm, axis in tasks:
        log = RUNTIME / f"full-{'aihub' if arm == 'aihub_full_backbone' else 'base'}-{axis}.log"
        command = [str(PYTHON), "scripts/run_kure_axis_contrastive.py", "--config", str(config(arm, axis))]
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "MAL2026_RESERVED_PHYSICAL_GPU": str(gpu),
                            "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
        handle = log.open("x", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        running.append((process, handle, gpu, arm, axis, log))
        ledger("axis_job_started", gpu=gpu, arm=arm, axis=axis, pid=process.pid, command=command, log=str(log.relative_to(ROOT)))
    last_sample = 0.0
    while running:
        current = time.monotonic()
        if current - last_sample >= 30:
            append(GPU_SAMPLES, {"event": "sample", "running": [{"gpu": gpu, "arm": arm, "axis": axis, "pid": process.pid} for process, _, gpu, arm, axis, _ in running], "gpus": gpu_snapshot()})
            last_sample = current
        failures = []
        remaining = []
        for item in running:
            process, handle, gpu, arm, axis, log = item
            code = process.poll()
            if code is None:
                remaining.append(item)
                continue
            handle.close()
            ledger("axis_job_finished", gpu=gpu, arm=arm, axis=axis, pid=process.pid, exit_code=code, log=str(log.relative_to(ROOT)))
            if code:
                failures.append((arm, axis, code, log))
        running = remaining
        if failures:
            # Do not terminate successful or still-running sibling research jobs.
            while running:
                time.sleep(10)
                next_running = []
                for item in running:
                    process, handle, gpu, arm, axis, log = item
                    code = process.poll()
                    if code is None:
                        next_running.append(item)
                    else:
                        handle.close(); ledger("axis_job_finished_after_peer_failure", gpu=gpu, arm=arm, axis=axis, pid=process.pid, exit_code=code, log=str(log.relative_to(ROOT)))
                running = next_running
            raise RuntimeError(f"axis job failure: {failures}")
        if running:
            time.sleep(10)


def main() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    try:
        preflight()
        launch_batch(((0, "base", "content"), (1, "base", "organization"), (2, "base", "expression"), (3, "aihub_full_backbone", "content")))
        launch_batch(((0, "aihub_full_backbone", "organization"), (1, "aihub_full_backbone", "expression")))
        command = [str(PYTHON), "scripts/aggregate_kure_axis_contrastive.py"]
        completed = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": "src"}, text=True, capture_output=True)
        ledger("aggregate_finished", exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        if completed.returncode:
            raise RuntimeError("aggregate failed")
        aggregate = json.loads((OUTPUT_ROOT / "aggregate.json").read_text(encoding="utf-8"))
        ledger("experiment_completed", selection_dev_winner=aggregate["selection_dev_winner"],
               base_rmse=aggregate["arms"]["base"]["selection_dev"]["hybrid"]["macro"]["continuous_rmse"],
               aihub_rmse=aggregate["arms"]["aihub_full_backbone"]["selection_dev"]["hybrid"]["macro"]["continuous_rmse"])
    except Exception as exc:
        ledger("experiment_failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
