#!/usr/bin/env python3
"""Durable GPU0--3 launcher for the final nested v4 router study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.iterative_tail_router_protocol import (  # noqa: E402
    CONFIG_PATH,
    RUN_ID,
    load_protocol,
    validate_bound_inputs,
    validate_model_inventory,
)
from mal2026.iterative_tail_router_runner import (  # noqa: E402
    PUBLIC_ROOT,
    RESTRICTED_ROOT,
    aggregate_outer_results,
    gpu0_smoke,
    run_outer_fold,
)


PYTHON = ROOT / ".venv-standard" / "bin" / "python"
AUTHORIZED_GPUS = (0, 1, 2, 3)
TMUX_SESSION = "mal2026-v4-router"


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


def _visible_gpu() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {str(gpu) for gpu in AUTHORIZED_GPUS}:
        raise RuntimeError("worker requires exactly one authorized CUDA_VISIBLE_DEVICES value 0..3")
    return int(visible)


def _gpu_preflight() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    snapshots, conflicts = [], []
    for gpu in AUTHORIZED_GPUS:
        state = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout.strip()
        processes = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout.strip()
        snapshots.append({"gpu": gpu, "state": state, "compute_processes": processes or None})
        if processes:
            conflicts.append({"gpu": gpu, "compute_processes": processes})
    return snapshots, conflicts


def _write_task_card(snapshot: list[dict[str, object]]) -> None:
    protocol = load_protocol()
    audit = validate_bound_inputs(protocol)
    validate_model_inventory(require_available=True)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    config_sha = _sha256(ROOT / CONFIG_PATH)
    timestamp = datetime.now(timezone.utc).isoformat()
    evidence = [
        {
            "agent_role": "metrics/statistics",
            "timestamp": timestamp,
            "hypothesis": "score-routed identity anchors may separate low-tail protection from score5 and 3/4 gains",
            "predicted_direction": "low_1_2_and_score5_rmse_down_with_gold3_4_balanced_accuracy_up",
            "falsification_condition": "no route passes the seven-gate inner AND or final nested gate",
            "accepted_or_rejected_rationale": "accepted_for_one_final_adaptive_nested_study_with_fail_closed_stop",
            "config_sha256": config_sha,
        },
        {
            "agent_role": "modeling",
            "timestamp": timestamp,
            "hypothesis": "five bounded score-only routing families can preserve exact identity where expert evidence is unsafe",
            "predicted_direction": "joint_macro_equal_group_low_high_and_mid_boundary_improvement",
            "falsification_condition": "all twenty fixed routes fail the inner gate or fresh outer refit loses the signal",
            "accepted_or_rejected_rationale": "accepted_as_exact_twenty_route_inventory_before_outer_results",
            "config_sha256": config_sha,
        },
        {
            "agent_role": "scientific_protocol",
            "timestamp": timestamp,
            "hypothesis": "nested S/D/O isolation permits descriptive same-train testing without outer selection leakage",
            "predicted_direction": "outer_predictions_once_after_route_and_weights_freeze",
            "falsification_condition": "any D or O target reaches component fitting or route selection, or concatenated outer results trigger selection",
            "accepted_or_rejected_rationale": "accepted_with_no_independent_or_generalization_claim_and_mandatory_freeze_on_failure",
            "config_sha256": config_sha,
        },
    ]
    _atomic_json(PUBLIC_ROOT / "task_card.json", {
        "schema_version": "mal2026-iterative-tail-router-task-card-v4",
        "run_id": RUN_ID,
        "git_sha_at_launch": git_sha,
        "config_path": str(CONFIG_PATH),
        "config_sha256": config_sha,
        "model_source_sha256": _sha256(ROOT / "src/mal2026/iterative_tail_router_models.py"),
        "protocol_source_sha256": _sha256(ROOT / "src/mal2026/iterative_tail_router_protocol.py"),
        "runner_source_sha256": _sha256(ROOT / "src/mal2026/iterative_tail_router_runner.py"),
        "launcher_source_sha256": _sha256(Path(__file__).resolve()),
        "seed": protocol.raw["execution"]["initialization_seed"],
        "canonical_train_sha256": protocol.raw["lineage"]["canonical_train_sha256"],
        "baseline_oof_sha256": audit.baseline_sha256,
        "evidence_cache_sha256": audit.evidence_cache_sha256,
        "v2_historical_aggregate_sha256": audit.historical_v2_aggregate_sha256,
        "v3_historical_aggregate_sha256": audit.historical_v3_aggregate_sha256,
        "environment": ".venv-standard",
        "gpu_scope": list(AUTHORIZED_GPUS),
        "gpu_authorization": "user explicitly authorized active use of GPUs 0,1,2,3 for the iterative program",
        "gpu_preflight": snapshot,
        "route_count_per_outer": 20,
        "outer_fold_count": 5,
        "inner_fold_count_per_outer": 4,
        "agent_preregistration_evidence": evidence,
        "agent_evidence_used_as_model_feature_or_weight": False,
        "adaptive_after_v2_and_v3_observed": True,
        "confirmatory_or_generalization_claim_allowed": False,
        "external_api_calls": 0,
        "validation_loaded": False,
        "average_target_used": False,
        "public_output_root": str(PUBLIC_ROOT),
        "restricted_output_root": str(RESTRICTED_ROOT),
        "exact_command": f"{PYTHON} {Path(__file__).resolve()} --full",
    })


def _outer_complete(outer: int) -> bool:
    result_path = PUBLIC_ROOT / f"outer-{outer}" / "result.json"
    prediction_path = RESTRICTED_ROOT / f"outer-{outer}" / "predictions.jsonl"
    if not result_path.is_file() or not prediction_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("status") == "completed" and result.get("route_count") == 20
        and result.get("restricted_prediction_sha256") == _sha256(prediction_path)
    )


def _spawn(mode: str, value: int, gpu: int):
    log_path = PUBLIC_ROOT / "logs" / f"{mode}-{value}.gpu{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [str(PYTHON), str(Path(__file__).resolve()), "--smoke-worker" if mode == "smoke" else "--outer-worker"]
    if mode == "outer":
        command.append(str(value))
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
    return process, log


def _smoke_controller() -> None:
    process, log = _spawn("smoke", 0, 0)
    _ledger("smoke_started", gpu=0, pid=process.pid)
    code = process.wait()
    log.close()
    if code:
        _ledger("smoke_failed", gpu=0, exit_code=code)
        raise RuntimeError(f"GPU0 smoke failed: exit {code}")
    _ledger("smoke_completed", gpu=0)


def _outer_controller() -> None:
    pending = [outer for outer in range(5) if not _outer_complete(outer)]
    running: dict[int, tuple[subprocess.Popen, object, int]] = {}
    available = list(AUTHORIZED_GPUS)
    failures = []
    while pending or running:
        while pending and available and not failures:
            outer, gpu = pending.pop(0), available.pop(0)
            process, log = _spawn("outer", outer, gpu)
            running[outer] = (process, log, gpu)
            _ledger("outer_started", outer_fold=outer, gpu=gpu, pid=process.pid)
        finished = []
        for outer, (process, log, gpu) in running.items():
            code = process.poll()
            if code is None:
                continue
            log.close()
            finished.append(outer)
            available.append(gpu)
            available.sort()
            if code or not _outer_complete(outer):
                failures.append({"outer_fold": outer, "gpu": gpu, "exit_code": int(code or 1)})
                _ledger("outer_failed", outer_fold=outer, gpu=gpu, exit_code=code)
            else:
                _ledger("outer_completed", outer_fold=outer, gpu=gpu)
        for outer in finished:
            del running[outer]
        if failures:
            pending.clear()
        if running and not finished:
            time.sleep(5)
    if failures:
        raise RuntimeError(f"outer stage failed: {failures}")


def _full() -> None:
    snapshot, conflicts = _gpu_preflight()
    _write_task_card(snapshot)
    _ledger("preflight_completed", gpu_scope=list(AUTHORIZED_GPUS), conflicts=conflicts)
    if conflicts:
        raise RuntimeError(f"pre-existing compute process detected; launch refused: {conflicts}")
    if not (PUBLIC_ROOT / "smoke.json").is_file():
        _smoke_controller()
    _outer_controller()
    aggregate = aggregate_outer_results()
    _ledger(
        "aggregate_completed", final_gate_pass=aggregate["final_gate_pass"],
        final_selection=aggregate["final_selection"],
        same_train_search_frozen=aggregate["same_train_search_frozen"],
    )
    print(json.dumps({
        "run_id": RUN_ID,
        "status": "completed",
        "final_gate_pass": aggregate["final_gate_pass"],
        "final_selection": aggregate["final_selection"],
        "same_train_search_frozen": aggregate["same_train_search_frozen"],
    }, ensure_ascii=False, indent=2))


def _progress() -> dict[str, object]:
    outers = []
    for outer in range(5):
        path = PUBLIC_ROOT / f"outer-{outer}" / "result.json"
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            outers.append({
                "outer_fold": outer, "status": raw.get("status"),
                "selected_route": raw.get("selected_route"), "route_count": raw.get("route_count"),
            })
        else:
            outers.append({"outer_fold": outer, "status": "pending"})
    completion_path = PUBLIC_ROOT / "completion.json"
    return {
        "run_id": RUN_ID,
        "smoke_completed": (PUBLIC_ROOT / "smoke.json").is_file(),
        "outers": outers,
        "completion": json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.is_file() else None,
    }


def _launch_tmux() -> None:
    existing = subprocess.run(["tmux", "has-session", "-t", TMUX_SESSION], capture_output=True)
    if existing.returncode == 0:
        raise RuntimeError(f"tmux session already exists: {TMUX_SESSION}")
    command = f"cd {ROOT} && {PYTHON} {Path(__file__).resolve()} --full"
    subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SESSION, command], check=True)
    print(json.dumps({"launched": True, "tmux_session": TMUX_SESSION, "command": command}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--outer-worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.launch:
        _launch_tmux()
    elif args.full:
        _full()
    elif args.smoke:
        snapshot, conflicts = _gpu_preflight()
        if any(item["gpu"] == 0 for item in conflicts):
            raise RuntimeError(f"GPU0 conflict: {conflicts}")
        _write_task_card(snapshot)
        _smoke_controller()
    elif args.smoke_worker:
        if _visible_gpu() != 0:
            raise RuntimeError("smoke worker must use physical GPU0")
        print(json.dumps(gpu0_smoke(device="cuda:0"), ensure_ascii=False, indent=2))
    elif args.outer_worker is not None:
        gpu = _visible_gpu()
        if args.outer_worker not in range(5):
            raise RuntimeError("outer fold must be 0..4")
        print(f"outer={args.outer_worker} physical_gpu={gpu} start", flush=True)
        print(json.dumps(run_outer_fold(args.outer_worker, device="cuda:0"), ensure_ascii=False, indent=2))
    elif args.aggregate:
        print(json.dumps(aggregate_outer_results(), ensure_ascii=False, indent=2))
    elif args.progress:
        print(json.dumps(_progress(), ensure_ascii=False, indent=2))
    else:
        parser.error("choose one action")


if __name__ == "__main__":
    main()
