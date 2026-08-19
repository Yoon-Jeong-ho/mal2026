#!/usr/bin/env python3
"""Wait for encoder campaign, then train/evaluate the final joint decoder."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-standard/bin/python"
TORCHRUN = ROOT / ".venv-standard/bin/torchrun"
TRAINER = ROOT / "scripts/train_rationale_pipeline_joint_decoder.py"
EVALUATOR = ROOT / "scripts/evaluate_rationale_pipeline_joint_decoder.py"
ENCODER_CAMPAIGN_PARENT = ROOT / "outputs/rationale-pipeline-ratio-encoder-campaign-v1"
JOINT_PARENT = ROOT / "outputs/rationale-pipeline-joint-decoder-v1"
JOINT_EVALUATION_PARENT = ROOT / "outputs/rationale-pipeline-joint-decoder-evaluation-v1"
OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-joint-followup-v1"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def pid_alive(pid: int, marker: str) -> bool:
    path = Path(f"/proc/{pid}/cmdline")
    return path.is_file() and marker in path.read_bytes().replace(b"\0", b" ").decode(errors="replace")


def require_idle() -> None:
    raw = subprocess.run(["nvidia-smi", "--id=0,1,2,3", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True, capture_output=True, check=True).stdout.strip()
    state = subprocess.check_output(["nvidia-smi", "--id=0,1,2,3", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    parsed = [tuple(int(part.strip()) for part in line.split(",")) for line in state.splitlines()]
    need(not raw and all(memory <= 16 and utilization == 0 for _, memory, utilization in parsed), "joint followup GPU scope is not idle; no process was altered")


def wait_idle(timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            require_idle(); return
        except RuntimeError:
            if time.monotonic() >= deadline: raise
            time.sleep(1)


def run(command: list[str], log: Path, environment: Mapping[str, str]) -> None:
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, env=dict(environment), stdout=handle, stderr=subprocess.STDOUT, text=True)
    need(result.returncode == 0, f"joint followup command failed: {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--encoder-campaign-run-id", required=True)
    parser.add_argument("--encoder-campaign-pid", type=int, required=True)
    parser.add_argument("--joint-config", type=Path, required=True)
    args = parser.parse_args()
    setproctitle(f"mal2026:joint-followup:{args.run_id}"[:255])
    output = OUTPUT_PARENT / args.run_id; need(not output.exists(), "joint followup output must be fresh"); output.mkdir(parents=True)
    config = json.loads(args.joint_config.read_text(encoding="utf-8")); joint_run_id = str(config["run_id"])
    state: dict[str, Any] = {"schema_version": "mal2026-rationale-pipeline-joint-followup-v1", "status": "waiting_encoder_campaign", "run_id": args.run_id, "created_at": now(), "encoder_campaign_run_id": args.encoder_campaign_run_id, "joint_run_id": joint_run_id, "gpu_scope": [0, 1, 2, 3], "average_used": False}
    atomic_json(output / "state.json", state)
    encoder_aggregate_path = ENCODER_CAMPAIGN_PARENT / args.encoder_campaign_run_id / "aggregate.json"
    while not encoder_aggregate_path.is_file():
        need(pid_alive(args.encoder_campaign_pid, args.encoder_campaign_run_id), "encoder campaign exited without aggregate")
        time.sleep(60)
    encoder = json.loads(encoder_aggregate_path.read_text(encoding="utf-8")); need(encoder.get("status") == "completed", "encoder campaign aggregate differs")
    wait_idle()
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    state.update({"status": "joint_smoke", "joint_smoke_started_at": now()}); atomic_json(output / "state.json", state)
    run([str(PYTHON), str(TRAINER), "--config", str(args.joint_config), "--smoke"], output / "joint-smoke.log", {**environment, "CUDA_VISIBLE_DEVICES": "0"})
    smoke = JOINT_PARENT / f"smoke-{joint_run_id}" / "smoke_complete.json"; need(smoke.is_file(), "joint followup smoke artifact unavailable")
    wait_idle()
    state.update({"status": "joint_full", "joint_full_started_at": now()}); atomic_json(output / "state.json", state)
    run([str(TORCHRUN), "--standalone", "--nproc_per_node=4", str(TRAINER), "--config", str(args.joint_config)], output / "joint-full.log", {**environment, "CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    completion = JOINT_PARENT / joint_run_id / "training_complete.json"; adapter = JOINT_PARENT / joint_run_id / "adapter"
    need(completion.is_file() and (adapter / "adapter_model.safetensors").is_file(), "joint followup training artifacts unavailable")
    wait_idle()
    evaluation_run_id = f"{joint_run_id}-evaluation"
    state.update({"status": "joint_evaluation", "joint_evaluation_started_at": now()}); atomic_json(output / "state.json", state)
    run([str(PYTHON), str(EVALUATOR), "--run-id", evaluation_run_id, "--base-model", str(config["model_path"]), "--adapter", str(adapter), "--training-completion", str(completion)], output / "joint-evaluation.log", environment)
    evaluation_path = JOINT_EVALUATION_PARENT / evaluation_run_id / "aggregate.json"; need(evaluation_path.is_file(), "joint followup evaluation unavailable")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")); need(evaluation.get("status") == "completed", "joint followup evaluation differs")
    result = {**state, "status": "completed", "completed_at": now(), "encoder_campaign_aggregate": str(encoder_aggregate_path.resolve()), "encoder_summaries": encoder["summaries"], "joint_training_completion": str(completion.resolve()), "joint_evaluation": str(evaluation_path.resolve()), "joint_score_metrics": evaluation["score_metrics"], "joint_judge": evaluation["judge"], "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights"}
    atomic_json(output / "aggregate.json", result); atomic_json(output / "state.json", result)
    print(json.dumps({"status": "completed", "run_id": args.run_id, "joint_macro_integer_rmse": evaluation["score_metrics"]["macro_integer_rmse"], "joint_judge_macro": evaluation["judge"]["macro_mean"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
