#!/usr/bin/env python3
"""Durable GPU0--3 launcher for the sealed v2 nested remediation run."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.iterative_tail_remediation_protocol import (  # noqa: E402
    CONFIG_PATH,
    RUN_ID,
    load_protocol,
    validate_bound_inputs,
)
from mal2026.iterative_tail_remediation_runner import (  # noqa: E402
    PUBLIC_ROOT,
    RESTRICTED_ROOT,
    aggregate_outer_results,
    gpu0_smoke,
    run_outer_fold,
)


PYTHON = ROOT / ".venv-standard" / "bin" / "python"
AUTHORIZED_GPUS = (0, 1, 2, 3)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ledger(event: str, **fields: object) -> None:
    path = PUBLIC_ROOT / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at_epoch": time.time(), "event": event, **fields}, sort_keys=True) + "\n")


def _visible_physical_gpu() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {str(gpu) for gpu in AUTHORIZED_GPUS}:
        raise RuntimeError("worker requires exactly one CUDA_VISIBLE_DEVICES value in authorized GPUs 0..3")
    return int(visible)


def _gpu_snapshot_and_conflicts() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    snapshots: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    for gpu in AUTHORIZED_GPUS:
        state = subprocess.run(
            [
                "nvidia-smi", "-i", str(gpu),
                "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout.strip()
        processes = subprocess.run(
            [
                "nvidia-smi", "-i", str(gpu),
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout.strip()
        snapshots.append({"gpu": gpu, "state": state, "compute_processes": processes or None})
        if processes:
            conflicts.append({"gpu": gpu, "compute_processes": processes})
    return snapshots, conflicts


def _write_task_card(snapshot: list[dict[str, object]]) -> None:
    protocol = load_protocol()
    audit = validate_bound_inputs(protocol)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    _atomic_json(PUBLIC_ROOT / "task_card.json", {
        "schema_version": "mal2026-iterative-tail-remediation-task-card-v2",
        "run_id": RUN_ID,
        "hypothesis": "leakage-safe conditional calibration can improve both score tails and 3-vs-4 separation over exact R0 OOF",
        "git_sha_at_launch": git_sha,
        "config_path": str(CONFIG_PATH),
        "config_sha256": _sha256(ROOT / CONFIG_PATH),
        "seed": protocol.raw["execution"]["initialization_seed"],
        "canonical_train_sha256": protocol.lineage["canonical_train_sha256"],
        "baseline_oof_sha256": audit.baseline_sha256,
        "environment": ".venv-standard",
        "gpu_scope": list(AUTHORIZED_GPUS),
        "gpu_authorization": "user explicitly authorized active use of GPUs 0,1,2,3 for this named experiment",
        "gpu_preflight": snapshot,
        "external_api_calls_allowed": False,
        "external_api_calls": 0,
        "public_output_root": str(PUBLIC_ROOT),
        "restricted_output_root": str(RESTRICTED_ROOT),
        "exact_command": f"{PYTHON} {Path(__file__).resolve()} --full",
        "protocol_amendment_before_execution": (
            "independent pre-execution reviews required split-specific 2-of-3 inner teachers, "
            "baseline-relative global selection, exact five-knot WLS, registered tail constants, "
            "and public knot scrubbing; no v2 outer result existed before amendment"
        ),
        "validation_loaded": False,
        "average_target_used": False,
    })


def _completed_outer(outer_fold: int) -> bool:
    result = PUBLIC_ROOT / f"outer-{outer_fold}" / "result.json"
    prediction = RESTRICTED_ROOT / f"outer-{outer_fold}" / "predictions.jsonl"
    if not result.is_file() or not prediction.is_file():
        return False
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "completed" and payload.get("restricted_prediction_sha256") == _sha256(prediction)


def _spawn(mode: str, value: int, physical_gpu: int) -> tuple[subprocess.Popen[str], object]:
    log_path = PUBLIC_ROOT / "logs" / f"{mode}-{value}.gpu{physical_gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    flag = "--outer-worker" if mode == "outer" else "--smoke-worker"
    command = [str(PYTHON), str(Path(__file__).resolve()), flag]
    if mode == "outer":
        command.append(str(value))
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True,
    )
    return process, log


def _run_smoke_controller() -> None:
    process, log = _spawn("smoke", 0, 0)
    _ledger("smoke_started", gpu=0, pid=process.pid)
    code = process.wait()
    log.close()
    if code:
        _ledger("smoke_failed", gpu=0, exit_code=code)
        raise RuntimeError(f"GPU0 smoke failed with exit code {code}")
    _ledger("smoke_completed", gpu=0)


def _run_outer_controller() -> None:
    pending = [fold for fold in range(5) if not _completed_outer(fold)]
    running: dict[int, tuple[subprocess.Popen[str], object, int]] = {}
    available = list(AUTHORIZED_GPUS)
    failures: list[dict[str, int]] = []
    while pending or running:
        while pending and available:
            fold = pending.pop(0)
            gpu = available.pop(0)
            process, log = _spawn("outer", fold, gpu)
            running[fold] = (process, log, gpu)
            _ledger("outer_started", outer_fold=fold, gpu=gpu, pid=process.pid)
        finished = []
        for fold, (process, log, gpu) in running.items():
            code = process.poll()
            if code is None:
                continue
            log.close()
            finished.append(fold)
            available.append(gpu)
            available.sort()
            if code or not _completed_outer(fold):
                failures.append({"outer_fold": fold, "gpu": gpu, "exit_code": int(code or 1)})
                _ledger("outer_failed", outer_fold=fold, gpu=gpu, exit_code=code)
            else:
                _ledger("outer_completed", outer_fold=fold, gpu=gpu)
        for fold in finished:
            del running[fold]
        if failures:
            # Preserve in-flight work and negative results, but do not start a
            # new fold after a declared stage failure.
            pending.clear()
        if running and not finished:
            time.sleep(5)
    if failures:
        raise RuntimeError(f"outer stage failed: {failures}")


def _full() -> None:
    load_protocol()
    snapshot, conflicts = _gpu_snapshot_and_conflicts()
    _write_task_card(snapshot)
    _ledger("preflight_completed", gpu_scope=list(AUTHORIZED_GPUS), conflicts=conflicts)
    if conflicts:
        raise RuntimeError(f"pre-existing GPU compute process detected; refusing to launch: {conflicts}")
    if not (PUBLIC_ROOT / "smoke.json").is_file():
        _run_smoke_controller()
    _run_outer_controller()
    aggregate = aggregate_outer_results()
    _ledger(
        "aggregate_completed",
        macro_rmse=aggregate["selected_metrics"]["macro"]["rmse"],
        improvement=aggregate["macro_rmse_improvement"],
        final_gate_pass=aggregate["final_gate_pass"],
    )
    print(json.dumps({
        "run_id": RUN_ID,
        "status": "completed",
        "final_gate_pass": aggregate["final_gate_pass"],
        "final_selection": aggregate["final_selection"],
        "macro_rmse_improvement": aggregate["macro_rmse_improvement"],
    }, ensure_ascii=False, indent=2))


def _progress() -> dict[str, object]:
    outer = []
    for fold in range(5):
        path = PUBLIC_ROOT / f"outer-{fold}" / "result.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            outer.append({
                "outer_fold": fold,
                "status": payload.get("status"),
                "selected_candidate": payload.get("selected_candidate"),
            })
        else:
            outer.append({"outer_fold": fold, "status": "pending"})
    completion = PUBLIC_ROOT / "completion.json"
    return {
        "run_id": RUN_ID,
        "smoke_completed": (PUBLIC_ROOT / "smoke.json").is_file(),
        "outer": outer,
        "completion": json.loads(completion.read_text(encoding="utf-8")) if completion.is_file() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--outer-worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.full:
        _full()
    elif args.smoke:
        snapshot, conflicts = _gpu_snapshot_and_conflicts()
        if any(item["gpu"] == 0 for item in conflicts):
            raise RuntimeError(f"GPU0 conflict: {conflicts}")
        _write_task_card(snapshot)
        _run_smoke_controller()
    elif args.smoke_worker:
        if _visible_physical_gpu() != 0:
            raise RuntimeError("smoke worker is bound to physical GPU0")
        print(json.dumps(gpu0_smoke(device="cuda:0"), ensure_ascii=False, indent=2))
    elif args.outer_worker is not None:
        physical_gpu = _visible_physical_gpu()
        if args.outer_worker not in range(5):
            raise RuntimeError("outer fold must be 0..4")
        print(f"outer={args.outer_worker} physical_gpu={physical_gpu} start", flush=True)
        print(json.dumps(run_outer_fold(args.outer_worker, device="cuda:0"), ensure_ascii=False, indent=2))
    elif args.aggregate:
        print(json.dumps(aggregate_outer_results(), ensure_ascii=False, indent=2))
    elif args.progress:
        print(json.dumps(_progress(), ensure_ascii=False, indent=2))
    else:
        parser.error("choose --full, --smoke, --aggregate, or --progress")


if __name__ == "__main__":
    main()
