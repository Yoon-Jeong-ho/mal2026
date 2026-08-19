#!/usr/bin/env python3
"""Run OOF fold tasks on the config-authorized GPUs without idle gaps."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from setproctitle import setproctitle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mal2026.qwen_rationale_oof import Config, STAGE1_ARMS, STAGE2_ARMS  # noqa: E402


def gpu_rows() -> dict[int, dict[str, int]]:
    command = ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result: dict[int, dict[str, int]] = {}
    for line in completed.stdout.splitlines():
        index, memory, utilization = [int(item.strip()) for item in line.split(",")]
        result[index] = {"memory_used_mib": memory, "utilization_gpu": utilization}
    return result


def telemetry(stop: threading.Event, path: Path, selected: tuple[int, ...]) -> None:
    fields = ["timestamp", "index", "memory.total", "memory.used", "utilization.gpu"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(fields)
        while not stop.is_set():
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True,
            )
            timestamp = time.time()
            for line in completed.stdout.splitlines():
                values = [item.strip() for item in line.split(",")]
                if int(values[0]) in selected: writer.writerow([timestamp, *values])
            handle.flush(); stop.wait(30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("stage1", "stage2", "stage4"), required=True)
    args = parser.parse_args(); config = Config.from_json(args.config)
    setproctitle(f"mal2026:qwen-oof-scheduler:{args.phase}")
    selected = tuple(config["gpu_scope"]); status = gpu_rows()
    conflicts = {gpu: status[gpu] for gpu in selected if status[gpu]["memory_used_mib"] > 1024 or status[gpu]["utilization_gpu"] > 10}
    if conflicts:
        raise SystemExit(f"GPU scope conflict; existing processes were not touched: {conflicts}")
    if args.phase == "stage1": arms = tuple(STAGE1_ARMS)
    elif args.phase == "stage2": arms = tuple(STAGE2_ARMS)
    else:
        audit = json.loads((config.output_root / "stage4" / "aihub_audit.json").read_text(encoding="utf-8"))
        if audit.get("status") != "admitted":
            print(json.dumps({"status": "not_admitted", "phase": args.phase, "reason": audit.get("reason")}, sort_keys=True)); return
        arms = ("aihub_tail",)
    tasks = [(arm, fold) for arm in arms for fold in range(5)]
    work: queue.Queue[tuple[str, int]] = queue.Queue()
    for task in tasks: work.put(task)
    logs = config.output_root / args.phase / "scheduler_logs"; logs.mkdir(parents=True, exist_ok=True)
    stop = threading.Event(); telemetry_path = config.output_root / args.phase / "gpu_telemetry.csv"
    monitor = threading.Thread(target=telemetry, args=(stop, telemetry_path, selected), daemon=True); monitor.start()

    def worker(gpu: int) -> list[dict[str, object]]:
        completed_tasks: list[dict[str, object]] = []
        while True:
            try: arm, fold = work.get_nowait()
            except queue.Empty: return completed_tasks
            result_path = config.output_root / args.phase / arm / f"fold-{fold:02d}" / "result.json"
            if result_path.is_file():
                completed_tasks.append({"arm": arm, "fold": fold, "gpu": gpu, "status": "skipped_completed"}); work.task_done(); continue
            command = [str(ROOT / ".venv-standard/bin/python"), str(ROOT / "scripts/run_qwen_rationale_oof.py"), "--config", str(args.config), "--fold", str(fold), "--phase", args.phase, "--arm", arm]
            environment = dict(os.environ); environment["CUDA_VISIBLE_DEVICES"] = str(gpu); environment["TOKENIZERS_PARALLELISM"] = "false"
            log_path = logs / f"{arm}.fold-{fold:02d}.gpu-{gpu}.log"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("COMMAND " + " ".join(command) + "\n"); handle.flush()
                process = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            completed_tasks.append({"arm": arm, "fold": fold, "gpu": gpu, "status": "completed" if process.returncode == 0 else "failed", "exit_code": process.returncode, "log": str(log_path)})
            work.task_done()
            if process.returncode != 0: return completed_tasks

    all_results: list[dict[str, object]] = []
    try:
        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in selected]
            for future in as_completed(futures): all_results.extend(future.result())
    finally:
        stop.set(); monitor.join(timeout=35)
    failed = [item for item in all_results if item["status"] == "failed"]
    summary = {"status": "failed" if failed else "completed", "phase": args.phase, "tasks": all_results, "telemetry_path": str(telemetry_path)}
    (config.output_root / args.phase / "scheduler_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if failed: raise SystemExit(1)


if __name__ == "__main__":
    main()
