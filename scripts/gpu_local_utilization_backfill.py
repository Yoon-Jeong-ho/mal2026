#!/usr/bin/env python3
"""Bounded, data-free, GPU-local utilization backfill for a reservation watchdog."""
from __future__ import annotations
import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

stop = False

def on_signal(*_: object) -> None:
    global stop
    stop = True

def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", required=True, type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--seconds", required=True, type=int)
    parser.add_argument("--memory-fraction", type=float, default=0.30)
    parser.add_argument("--heartbeat", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise SystemExit("backfill requires exactly its assigned physical CUDA_VISIBLE_DEVICES")
    if not 1 <= args.seconds <= 900 or not 0.05 <= args.memory_fraction <= 0.40:
        raise SystemExit("invalid bounded backfill duration or memory fraction")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("backfill requires exactly one visible CUDA GPU")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    torch.manual_seed(20260720 + args.physical_gpu)
    # Deliberately data-free and modest in memory.  The per-process fraction is
    # enforced before allocation, and this tensor is far below it on all hosts.
    size = 4096
    left = torch.randn((size, size), device="cuda", dtype=torch.float16)
    right = torch.randn((size, size), device="cuda", dtype=torch.float16)
    deadline = time.monotonic() + args.seconds
    iteration = 0
    args.heartbeat.parent.mkdir(parents=True, exist_ok=True)
    while not stop and time.monotonic() < deadline:
        left = torch.matmul(left, right).relu_()
        iteration += 1
        if iteration % 10 == 0:
            torch.cuda.synchronize()
            args.heartbeat.write_text(json.dumps({"at": now(), "physical_gpu": args.physical_gpu, "iterations": iteration, "memory_fraction_cap": args.memory_fraction, "data_access": False}) + "\n", encoding="utf-8")
    torch.cuda.synchronize()
    args.heartbeat.write_text(json.dumps({"at": now(), "physical_gpu": args.physical_gpu, "iterations": iteration, "ended": True, "yielded": stop, "memory_fraction_cap": args.memory_fraction, "data_access": False}) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
