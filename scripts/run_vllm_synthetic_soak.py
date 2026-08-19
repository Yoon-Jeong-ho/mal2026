#!/usr/bin/env python3
"""Bounded synthetic vLLM load client with aggregate-only telemetry.

The client never reads project data and never persists prompts or completions.
It records only counters, latency summaries, and selected-GPU telemetry.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

os.environ.setdefault("SPT_NOENV", "1")
import setproctitle


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> None:
    setproctitle.setproctitle("(D)_vllm")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    runtime, request_cfg = cfg["runtime"], cfg["request"]
    authorized_durations = {
        "mal2026-vllm-synthetic-soak-gpu0-3-48h-v1": 172800,
        "mal2026-vllm-synthetic-soak-gpu0-3-24h-v1": 86400,
        "mal2026-vllm-synthetic-soak-gpu0-3-120h-v1": 432000,
    }
    if (
        cfg.get("schema_version") not in authorized_durations
        or runtime.get("physical_gpus") != [0, 1, 2, 3]
        or runtime.get("duration_seconds") != authorized_durations[cfg["schema_version"]]
        or runtime.get("client_concurrency") != 768
    ):
        raise SystemExit("soak configuration escaped its authorized GPU0--3 duration envelope")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_dir / "client_metrics.json"
    telemetry_path = args.run_dir / "gpu_telemetry.csv"
    stop = threading.Event()
    lock = threading.Lock()
    started = time.monotonic()
    deadline = started + int(runtime["duration_seconds"])
    counters = {"completed": 0, "failed": 0, "output_tokens": 0}
    latencies: list[float] = []
    errors: dict[str, int] = {}
    utilization: dict[int, list[int]] = {gpu: [] for gpu in runtime["physical_gpus"]}
    temperatures: dict[int, list[int]] = {gpu: [] for gpu in runtime["physical_gpus"]}

    def snapshot(status: str) -> dict:
        elapsed = max(0.001, time.monotonic() - started)
        with lock:
            latency_copy = list(latencies)
            value = {
                "schema_version": "mal2026-vllm-synthetic-soak-aggregate-v1",
                "status": status,
                "updated_at": now(),
                "elapsed_seconds": elapsed,
                "planned_duration_seconds": runtime["duration_seconds"],
                "physical_gpus": runtime["physical_gpus"],
                "client_concurrency": runtime["client_concurrency"],
                "completed_requests": counters["completed"],
                "failed_requests": counters["failed"],
                "output_tokens": counters["output_tokens"],
                "requests_per_second": counters["completed"] / elapsed,
                "output_tokens_per_second": counters["output_tokens"] / elapsed,
                "latency_seconds": {
                    "p50": percentile(latency_copy, 0.50),
                    "p95": percentile(latency_copy, 0.95),
                    "p99": percentile(latency_copy, 0.99),
                },
                "gpu_utilization_percent": {
                    str(gpu): {
                        "samples": len(utilization[gpu]),
                        "mean": statistics.fmean(utilization[gpu]) if utilization[gpu] else None,
                        "p50": percentile([float(x) for x in utilization[gpu]], 0.50),
                        "p95": percentile([float(x) for x in utilization[gpu]], 0.95),
                    }
                    for gpu in runtime["physical_gpus"]
                },
                "gpu_temperature_c": {
                    str(gpu): {
                        "max": max(temperatures[gpu]) if temperatures[gpu] else None,
                        "mean": statistics.fmean(temperatures[gpu]) if temperatures[gpu] else None,
                    }
                    for gpu in runtime["physical_gpus"]
                },
                "error_categories": dict(sorted(errors.items())),
                "raw_prompts_or_responses_persisted": False,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            }
        atomic_json(metrics_path, value)
        return value

    def telemetry_worker() -> None:
        telemetry_path.write_text(
            "timestamp_utc,gpu_index,utilization_gpu_percent,memory_used_mib,temperature_c\n",
            encoding="utf-8",
        )
        while not stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi", "--id=0,1,2,3",
                        "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=10,
                )
                timestamp = now()
                rows = []
                for line in output.splitlines():
                    gpu, util, memory, temp = [int(item.strip()) for item in line.split(",")]
                    if gpu not in utilization:
                        raise RuntimeError("telemetry returned an unauthorized GPU")
                    rows.append(f"{timestamp},{gpu},{util},{memory},{temp}\n")
                    with lock:
                        utilization[gpu].append(util)
                        temperatures[gpu].append(temp)
                    if temp > int(runtime["max_temperature_c"]):
                        errors["temperature_limit"] = errors.get("temperature_limit", 0) + 1
                        stop.set()
                with telemetry_path.open("a", encoding="utf-8") as handle:
                    handle.writelines(rows)
            except Exception as exc:  # fail closed on missing safety telemetry
                with lock:
                    name = f"telemetry:{type(exc).__name__}"
                    errors[name] = errors.get(name, 0) + 1
                stop.set()
            stop.wait(int(runtime["telemetry_interval_seconds"]))

    synthetic_base = (
        "다음은 연구 서버의 합성 처리량 점검입니다. 실제 작문이나 개인 정보는 없습니다. "
        "한국어로 서로 다른 짧은 문장을 계속 생성하되, 번호나 표를 쓰지 말고 정확히 충분한 길이로 작성하세요. "
    )

    def worker(worker_id: int) -> None:
        rng = random.Random(int(cfg["seed"]) + worker_id)
        while not stop.is_set() and time.monotonic() < deadline:
            nonce = rng.getrandbits(64)
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": f"{synthetic_base} 합성 키: {nonce:016x}"}],
                **request_cfg,
            }
            started_call = time.monotonic()
            try:
                req = Request(
                    args.endpoint.rstrip("/") + "/v1/chat/completions",
                    data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req, timeout=900) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not result.get("choices"):
                    raise RuntimeError("missing_choices")
                used = int(result.get("usage", {}).get("completion_tokens", 0))
                with lock:
                    counters["completed"] += 1
                    counters["output_tokens"] += used
                    latencies.append(time.monotonic() - started_call)
                    if len(latencies) > 20000:
                        del latencies[:10000]
            except Exception as exc:
                with lock:
                    counters["failed"] += 1
                    name = type(exc).__name__
                    errors[name] = errors.get(name, 0) + 1
                if not stop.is_set():
                    time.sleep(1)

    telemetry = threading.Thread(target=telemetry_worker, name="gpu-telemetry", daemon=True)
    telemetry.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(runtime["client_concurrency"])) as pool:
        futures = [pool.submit(worker, index) for index in range(int(runtime["client_concurrency"]))]
        try:
            while time.monotonic() < deadline and not stop.wait(30):
                snapshot("running")
        finally:
            stop.set()
            concurrent.futures.wait(futures)
    telemetry.join(timeout=15)
    final = snapshot("completed" if time.monotonic() >= deadline and not errors.get("temperature_limit") else "stopped")
    if final["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
