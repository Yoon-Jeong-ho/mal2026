#!/usr/bin/env python3
"""Orchestrate the fixed train-only 20-round iterative tail experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.iterative_tail_runner import (  # noqa: E402
    PUBLIC_ROOT,
    RESTRICTED_ROOT,
    aggregate_candidate_round,
    apply_sequential_promotion,
    initialize_baseline_round,
    load_experiment_data,
    prepare_score_blind_cache,
    public_progress,
    run_candidate_fold,
    run_round19_ensemble,
    run_round20_calibration,
    variants_for_round,
)


def _write_ledger(event: str, **fields: object) -> None:
    path = PUBLIC_ROOT / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"at_epoch": time.time(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _worker(tasks_path: Path, device: str) -> None:
    data = load_experiment_data()
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    for task in tasks:
        round_number = int(task["round"])
        fold = int(task["fold"])
        variants = {variant.variant_id: variant for variant in variants_for_round(round_number)}
        variant = variants[task["variant"]]
        _, result_path = __import__(
            "mal2026.iterative_tail_runner", fromlist=["fold_result_paths"]
        ).fold_result_paths(variant, fold)
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "completed":
                continue
        print(f"start round={round_number} variant={variant.variant_id} fold={fold}", flush=True)
        run_candidate_fold(data, variant, fold, device=device)
        print(f"complete round={round_number} variant={variant.variant_id} fold={fold}", flush=True)


def _stage_tasks(rounds: range | tuple[int, ...]) -> list[dict[str, object]]:
    return [
        {"round": round_number, "variant": variant.variant_id, "fold": fold}
        for round_number in rounds
        for variant in variants_for_round(round_number)
        for fold in range(5)
    ]


def _launch_workers(tasks: list[dict[str, object]], stage: str) -> None:
    assignments = [[] for _ in range(4)]
    # Interleave fold-heavy candidates so every GPU receives comparable work.
    for index, task in enumerate(tasks):
        assignments[index % 4].append(task)
    processes = []
    log_handles = []
    for gpu, assigned in enumerate(assignments):
        task_path = RESTRICTED_ROOT / "task_queues" / f"{stage}.gpu{gpu}.json"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_path.write_text(json.dumps(assigned, indent=2) + "\n", encoding="utf-8")
        log_path = PUBLIC_ROOT / "logs" / f"{stage}.gpu{gpu}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            str(ROOT / ".venv-standard" / "bin" / "python"), str(Path(__file__).resolve()),
            "--worker", str(task_path), "--device", "cuda:0",
        ]
        processes.append(subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log_handle, stderr=subprocess.STDOUT))
        log_handles.append(log_handle)
    _write_ledger("workers_started", stage=stage, gpus=[0, 1, 2, 3], task_count=len(tasks))
    failures = []
    for gpu, process in enumerate(processes):
        code = process.wait()
        if code:
            failures.append({"gpu": gpu, "exit_code": code})
    for handle in log_handles:
        handle.close()
    if failures:
        _write_ledger("workers_failed", stage=stage, failures=failures)
        raise RuntimeError(f"worker stage {stage} failed: {failures}")
    _write_ledger("workers_completed", stage=stage, gpus=[0, 1, 2, 3], task_count=len(tasks))


def _full() -> None:
    prepare_score_blind_cache()
    data = load_experiment_data()
    initialize_baseline_round(data)
    _write_ledger("baseline_reproduced", macro_rmse=0.5687802162500409)

    smoke_variant = variants_for_round(3)[-1]
    smoke = run_candidate_fold(data, smoke_variant, 0, device="cuda:0", smoke=True)
    _write_ledger("gpu0_smoke_completed", macro_rmse=smoke["macro_rmse"], records=smoke["records"])

    _launch_workers(_stage_tasks(range(2, 17)), "rounds02-16")
    for round_number in range(2, 17):
        aggregate_candidate_round(data, round_number)
        _write_ledger("round_aggregated", round=round_number)

    _launch_workers(_stage_tasks((17, 18)), "rounds17-18")
    for round_number in (17, 18):
        aggregate_candidate_round(data, round_number)
        _write_ledger("round_aggregated", round=round_number)

    promotion18 = apply_sequential_promotion(data, through_round=18, final_bootstrap=False)
    run_round19_ensemble(data, promotion18["promoted_rounds"])
    promotion19 = apply_sequential_promotion(data, through_round=19, final_bootstrap=False)
    run_round20_calibration(data, int(promotion19["incumbent_round"]))
    final = apply_sequential_promotion(data, through_round=20)
    _write_ledger(
        "twenty_rounds_completed", incumbent_round=final["incumbent_round"],
        macro_rmse=final["incumbent_macro_rmse"], final_gate_pass=final["final_gate_pass"],
    )
    print(json.dumps(public_progress(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.prepare:
        print(json.dumps(prepare_score_blind_cache(), indent=2))
    elif args.worker:
        _worker(args.worker, args.device)
    elif args.smoke:
        prepare_score_blind_cache()
        data = load_experiment_data()
        initialize_baseline_round(data)
        print(json.dumps(run_candidate_fold(data, variants_for_round(3)[-1], 0, device=args.device, smoke=True), indent=2))
    elif args.full:
        _full()
    elif args.progress:
        print(json.dumps(public_progress(), indent=2))
    else:
        parser.error("choose --prepare, --smoke, --worker, --full, or --progress")


if __name__ == "__main__":
    main()
