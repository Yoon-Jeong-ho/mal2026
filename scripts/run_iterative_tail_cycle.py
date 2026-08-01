#!/usr/bin/env python3
"""Durable GPU0--3 launcher for the fixed v3 twenty-cycle discovery."""
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

from mal2026.iterative_tail_cycle_protocol import (  # noqa: E402
    CONFIG_PATH,
    RUN_ID,
    load_protocol,
    validate_bound_inputs,
    validate_model_inventory,
)
from mal2026.iterative_tail_cycle_runner import (  # noqa: E402
    PUBLIC_ROOT,
    RESTRICTED_ROOT,
    aggregate_results,
    gpu0_smoke,
    run_fold,
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
    evidence = [
        {
            "agent_role": "metrics/statistics",
            "timestamp": "2026-08-01T00:00:00+09:00",
            "hypothesis": "five-band ordinal and conditional routing must separate the R17 low-tail loss from its score5/3-4 gain",
            "predicted_direction": "low1_2_and_score5_rmse_down_with_gold3_4_balanced_accuracy_up",
            "falsification_condition": "strict seven-gate AND fails on complete five-fold OOF",
            "accepted_or_rejected_rationale": "accepted_as_cycles_1_to_20_fixed_before_v3_results",
            "config_sha256": config_sha,
        },
        {
            "agent_role": "modeling",
            "timestamp": "2026-08-01T00:00:00+09:00",
            "hypothesis": "score-routed identity anchors and band-specific objectives outperform global residual/calibration trade-offs",
            "predicted_direction": "protect_low_tail_while_retaining_high_and_mid_boundary_signal",
            "falsification_condition": "no fixed candidate passes all gates or best macro candidate sacrifices either tail",
            "accepted_or_rejected_rationale": "accepted_as_five_families_four_variants_each",
            "config_sha256": config_sha,
        },
        {
            "agent_role": "scientific_protocol",
            "timestamp": "2026-08-01T00:00:00+09:00",
            "hypothesis": "all 20 candidates can be compared descriptively only after complete heldout OOF generation",
            "predicted_direction": "no_selection_leakage_and_fail_closed_baseline",
            "falsification_condition": "any heldout gold enters fit or any selection occurs before all 100 predictions complete",
            "accepted_or_rejected_rationale": "accepted_with_adaptive_nonconfirmatory_claim_boundary",
            "config_sha256": config_sha,
        },
    ]
    _atomic_json(PUBLIC_ROOT / "task_card.json", {
        "schema_version": "mal2026-iterative-tail-cycle-task-card-v3",
        "run_id": RUN_ID,
        "git_sha_at_launch": git_sha,
        "config_path": str(CONFIG_PATH),
        "config_sha256": config_sha,
        "model_source_sha256": _sha256(ROOT / "src/mal2026/iterative_tail_cycle_models.py"),
        "protocol_source_sha256": _sha256(ROOT / "src/mal2026/iterative_tail_cycle_protocol.py"),
        "runner_source_sha256": _sha256(ROOT / "src/mal2026/iterative_tail_cycle_runner.py"),
        "seed": protocol.raw["execution"]["initialization_seed"],
        "canonical_train_sha256": protocol.raw["lineage"]["canonical_train_sha256"],
        "baseline_oof_sha256": audit.baseline_sha256,
        "v2_historical_aggregate_sha256": audit.historical_v2_aggregate_sha256,
        "environment": ".venv-standard",
        "gpu_scope": list(AUTHORIZED_GPUS),
        "gpu_authorization": "user explicitly authorized active use of GPUs 0,1,2,3 for the iterative program",
        "gpu_preflight": snapshot,
        "cycle_count": 20,
        "fold_cycle_prediction_count": 100,
        "agent_preregistration_evidence": evidence,
        "agent_evidence_used_as_model_feature_or_weight": False,
        "adaptive_after_v2_outer_observed": True,
        "confirmatory_claim_allowed": False,
        "external_api_calls": 0,
        "validation_loaded": False,
        "average_target_used": False,
        "public_output_root": str(PUBLIC_ROOT),
        "restricted_output_root": str(RESTRICTED_ROOT),
        "exact_command": f"{PYTHON} {Path(__file__).resolve()} --full",
    })


def _fold_complete(fold: int) -> bool:
    result_path = PUBLIC_ROOT / f"fold-{fold}" / "result.json"
    prediction_path = RESTRICTED_ROOT / f"fold-{fold}" / "predictions.jsonl"
    if not result_path.is_file() or not prediction_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("status") == "completed"
        and result.get("cycle_count") == 20
        and result.get("restricted_prediction_sha256") == _sha256(prediction_path)
    )


def _spawn(mode: str, value: int, gpu: int):
    log_path = PUBLIC_ROOT / "logs" / f"{mode}-{value}.gpu{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [str(PYTHON), str(Path(__file__).resolve()), "--smoke-worker" if mode == "smoke" else "--fold-worker"]
    if mode == "fold":
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


def _fold_controller() -> None:
    pending = [fold for fold in range(5) if not _fold_complete(fold)]
    running = {}
    available = list(AUTHORIZED_GPUS)
    failures = []
    while pending or running:
        while pending and available and not failures:
            fold, gpu = pending.pop(0), available.pop(0)
            process, log = _spawn("fold", fold, gpu)
            running[fold] = (process, log, gpu)
            _ledger("fold_started", fold=fold, gpu=gpu, pid=process.pid)
        finished = []
        for fold, (process, log, gpu) in running.items():
            code = process.poll()
            if code is None:
                continue
            log.close(); finished.append(fold); available.append(gpu); available.sort()
            if code or not _fold_complete(fold):
                failures.append({"fold": fold, "gpu": gpu, "exit_code": int(code or 1)})
                _ledger("fold_failed", fold=fold, gpu=gpu, exit_code=code)
            else:
                _ledger("fold_completed", fold=fold, gpu=gpu)
        for fold in finished:
            del running[fold]
        if failures:
            pending.clear()
        if running and not finished:
            time.sleep(5)
    if failures:
        raise RuntimeError(f"fold stage failed: {failures}")


def _full() -> None:
    snapshot, conflicts = _gpu_preflight()
    _write_task_card(snapshot)
    _ledger("preflight_completed", gpu_scope=list(AUTHORIZED_GPUS), conflicts=conflicts)
    if conflicts:
        raise RuntimeError(f"pre-existing compute process detected; launch refused: {conflicts}")
    if not (PUBLIC_ROOT / "smoke.json").is_file():
        _smoke_controller()
    _fold_controller()
    aggregate = aggregate_results()
    _ledger(
        "aggregate_completed", cycle_count=aggregate["cycle_count"],
        strict_gate_pass=aggregate["strict_discovery_gate_pass"],
        strict_selection=aggregate["strict_selection"],
        best_exploratory=aggregate["best_exploratory_variant"],
    )
    print(json.dumps({
        "run_id": RUN_ID,
        "status": "completed",
        "strict_gate_pass": aggregate["strict_discovery_gate_pass"],
        "strict_selection": aggregate["strict_selection"],
        "best_exploratory": aggregate["best_exploratory_variant"],
    }, ensure_ascii=False, indent=2))


def _progress() -> dict[str, object]:
    folds = []
    for fold in range(5):
        path = PUBLIC_ROOT / f"fold-{fold}" / "result.json"
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            folds.append({"fold": fold, "status": raw.get("status"), "cycle_count": raw.get("cycle_count")})
        else:
            folds.append({"fold": fold, "status": "pending"})
    completion_path = PUBLIC_ROOT / "completion.json"
    return {
        "run_id": RUN_ID,
        "smoke_completed": (PUBLIC_ROOT / "smoke.json").is_file(),
        "folds": folds,
        "completion": json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.is_file() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fold-worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.full:
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
    elif args.fold_worker is not None:
        gpu = _visible_gpu()
        if args.fold_worker not in range(5):
            raise RuntimeError("fold must be 0..4")
        print(f"fold={args.fold_worker} physical_gpu={gpu} start", flush=True)
        print(json.dumps(run_fold(args.fold_worker, device="cuda:0"), ensure_ascii=False, indent=2))
    elif args.aggregate:
        print(json.dumps(aggregate_results(), ensure_ascii=False, indent=2))
    elif args.progress:
        print(json.dumps(_progress(), ensure_ascii=False, indent=2))
    else:
        parser.error("choose --full, --smoke, --aggregate, or --progress")


if __name__ == "__main__":
    main()
