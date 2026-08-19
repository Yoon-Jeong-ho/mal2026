#!/usr/bin/env python3
"""Run four encoder arms with GPU0 smokes then one arm per GPU 0--3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-standard/bin/python"
TRAINER = ROOT / "scripts/train_rationale_pipeline_score_encoder.py"
OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-score-encoder-matrix-v1"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8")); need(value.get("schema_version") == "mal2026-rationale-pipeline-score-encoder-v1", "score encoder matrix config differs"); return value


def gpu_state() -> list[dict[str, int]]:
    raw = subprocess.check_output(["nvidia-smi", "--id=0,1,2,3", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    return [{"index": int(a), "memory_used_mib": int(b), "utilization_percent": int(c)} for a, b, c in ([part.strip() for part in line.split(",")] for line in raw.splitlines())]


def require_idle(gpus: tuple[int, ...]) -> None:
    raw = subprocess.run(["nvidia-smi", f"--id={','.join(map(str, gpus))}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True, capture_output=True, check=True).stdout.strip()
    state = {row["index"]: row for row in gpu_state()}
    need(not raw and all(state[gpu]["memory_used_mib"] <= 16 and state[gpu]["utilization_percent"] == 0 for gpu in gpus), "score encoder matrix GPU scope is not idle; no process was altered")


def wait_idle(gpus: tuple[int, ...], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            require_idle(gpus); return
        except RuntimeError:
            if time.monotonic() >= deadline: raise
            time.sleep(1)


def telemetry(stop: threading.Event, path: Path) -> None:
    while not stop.is_set():
        try:
            with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps({"at": now(), "gpus": gpu_state()}, separators=(",", ":")) + "\n")
        except Exception: pass
        stop.wait(1)


def command(path: Path, gpu: int, smoke: bool) -> tuple[list[str], dict[str, str]]:
    value = [str(PYTHON), str(TRAINER), "--config", str(path)]
    if smoke: value.append("--smoke")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MAL2026_PHYSICAL_GPU": str(gpu), "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    return value, environment


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--config", type=Path, action="append", required=True); args = parser.parse_args()
    setproctitle(f"mal2026:score-encoder-matrix:{args.run_id}"[:255])
    need(len(args.config) == 4 and len({str(path.resolve()) for path in args.config}) == 4, "score encoder matrix requires four unique configs")
    values = [config(path) for path in args.config]
    need(len({value["model_key"] for value in values}) == 1 and {(value["objective"], value["initialization"]) for value in values} == {(a, b) for a in ("bounded_regression", "categorical_5class") for b in ("base", "aihub")}, "score encoder matrix arms differ")
    need(len({value.get("rationale_ratio") for value in values}) == 1 and values[0].get("rationale_ratio") in {"1to1", "1to2", "1to3"}, "score encoder matrix rationale ratio differs")
    output = OUTPUT_PARENT / args.run_id; need(not output.exists(), "score encoder matrix output must be fresh"); output.mkdir(parents=True)
    (output / "launch.json").write_text(json.dumps({"schema_version": "mal2026-rationale-pipeline-score-encoder-matrix-launch-v1", "status": "running", "run_id": args.run_id, "created_at": now(), "model_key": values[0]["model_key"], "rationale_ratio": values[0]["rationale_ratio"], "gpu_scope": [0, 1, 2, 3], "configs": [str(path.resolve()) for path in args.config], "protocol": "sequential_gpu0_one_update_smokes_then_staged_authorized_single_gpu_full_arms_with_completed_result_reuse_effective_batch_preserved"}, indent=2, sort_keys=True) + "\n")

    # Repository policy requires GPU0 for the smallest real preflight.
    for index, (path, value) in enumerate(zip(args.config, values, strict=True)):
        prior = ROOT / "outputs/rationale-pipeline-score-encoder-v1" / f"smoke-{value['run_id']}" / "smoke_complete.json"
        if prior.is_file():
            completed = json.loads(prior.read_text(encoding="utf-8"))
            need(completed.get("status") == "completed" and completed.get("run_id") == value["run_id"] and completed.get("rationale_ratio") == value["rationale_ratio"] and completed.get("model_key") == value["model_key"] and completed.get("objective") == value["objective"] and completed.get("initialization") == value["initialization"] and completed.get("training_protocol", "select_then_refit") == value.get("training_protocol", "select_then_refit"), "prior score encoder smoke differs")
            (output / f"smoke-{index}-{path.stem}-reused.json").write_text(json.dumps({"status": "reused_completed_smoke", "artifact": str(prior.resolve())}, indent=2, sort_keys=True) + "\n")
            continue
        wait_idle((0,)); log = (output / f"smoke-{index}-{path.stem}.log").open("x", encoding="utf-8")
        cmd, env = command(path, 0, True); result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT); log.close(); need(result.returncode == 0, f"score encoder smoke failed: {path}")
        wait_idle((0,))

    pending: list[tuple[int, Path]] = []
    for gpu, (path, value) in enumerate(zip(args.config, values, strict=True)):
        prior = ROOT / "outputs/rationale-pipeline-score-encoder-v1" / value["run_id"] / "result.json"
        if prior.is_file():
            completed = json.loads(prior.read_text(encoding="utf-8"))
            need(completed.get("status") == "completed" and completed.get("run_id") == value["run_id"] and completed.get("rationale_ratio") == value["rationale_ratio"] and completed.get("model_key") == value["model_key"] and completed.get("objective") == value["objective"] and completed.get("initialization") == value["initialization"] and completed.get("physical_gpu") in {0, 1, 2, 3} and completed.get("training_protocol", "select_then_refit") == value.get("training_protocol", "select_then_refit"), "prior score encoder full result differs")
            (output / f"full-gpu{gpu}-{path.stem}-reused.json").write_text(json.dumps({"status": "reused_completed_full_result", "artifact": str(prior.resolve())}, indent=2, sort_keys=True) + "\n")
        else:
            pending.append((gpu, path))
    if pending: wait_idle(tuple(gpu for gpu, _ in pending))
    stop = threading.Event(); thread = threading.Thread(target=telemetry, args=(stop, output / "telemetry.jsonl"), daemon=True); thread.start()
    processes: list[tuple[subprocess.Popen[str], Any, Path]] = []
    try:
        for gpu, path in pending:
            log = (output / f"full-gpu{gpu}-{path.stem}.log").open("x", encoding="utf-8"); cmd, env = command(path, gpu, False)
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True); processes.append((process, log, path))
        failures = []
        for process, log, path in processes:
            returncode = process.wait(); log.close()
            if returncode != 0: failures.append({"config": str(path), "returncode": returncode})
        need(not failures, f"score encoder full arms failed: {failures}")
    finally:
        stop.set(); thread.join(timeout=5)
    results = []
    for value in values:
        path = ROOT / "outputs/rationale-pipeline-score-encoder-v1" / value["run_id"] / "result.json"
        need(path.is_file(), "score encoder matrix result unavailable")
        result = json.loads(path.read_text(encoding="utf-8")); need(result.get("status") == "completed", "score encoder matrix result incomplete")
        results.append({"run_id": value["run_id"], "objective": value["objective"], "initialization": value["initialization"], "training_protocol": result.get("training_protocol", "select_then_refit"), "fixed_epochs": result.get("fixed_epochs"), "physical_gpu": result["physical_gpu"], "macro_integer_rmse": result["canonical_validation"]["metrics"]["macro_integer_rmse"], "overall_integer_rmse": result["canonical_validation"]["metrics"]["overall_integer_rmse"], "macro_integer_spearman": result["canonical_validation"]["metrics"]["macro_integer_spearman"], "result_path": str(path.resolve())})
    winner = min(results, key=lambda row: (row["macro_integer_rmse"], -row["macro_integer_spearman"], row["run_id"]))
    payload: Mapping[str, Any] = {"schema_version": "mal2026-rationale-pipeline-score-encoder-matrix-result-v2", "status": "completed", "run_id": args.run_id, "completed_at": now(), "model_key": values[0]["model_key"], "rationale_ratio": values[0]["rationale_ratio"], "training_protocols": sorted({value.get("training_protocol", "select_then_refit") for value in values}), "gpu_scope": [0, 1, 2, 3], "arms": results, "winner": winner, "validation_use": "descriptive_comparison_of_predeclared_arms_not_epoch_selection", "average_used": False, "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_or_predictions"}
    (output / "aggregate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    launch = json.loads((output / "launch.json").read_text()); launch.update({"status": "completed", "completed_at": now()}); (output / "launch.json").write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "run_id": args.run_id, "winner": winner}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
