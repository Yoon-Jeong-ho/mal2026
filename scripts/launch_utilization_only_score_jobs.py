#!/usr/bin/env python3
"""Create and dry-validate the isolated, non-scientific utilization plan."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-standard" / "bin" / "python"
MANIFEST = ROOT / "data/manifests/aihub_human_feedback_v1.json"
QWEN3 = ROOT / "outputs/model-cache/Qwen--Qwen3-Embedding-8B-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
BATCH = "openai-rationale-terra-full-20260719-001"
MODEL = ROOT / "outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
SERVER = ROOT / "outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server"


def need(value: bool, message: str) -> None:
    if not value:
        raise SystemExit(message)


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score_config(run_id: str, field: str, gpu: int) -> dict:
    return {
        "run_id": run_id, "target_field": field, "backbone": "qwen3_embedding", "model_id": "Qwen/Qwen3-Embedding-8B",
        "model_revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", "tokenizer_revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "model_path": str(QWEN3), "prepared_manifest": str(MANIFEST), "output_dir": str(ROOT / "outputs/standard-encoder-runs" / run_id),
        "max_length": 2048, "seed": 20260720 + gpu, "learning_rate": 0.0001, "weight_decay": 0.01, "warmup_ratio": 0.05,
        "num_train_epochs": 10000.0, "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1, "gradient_accumulation_steps": 16,
        "eval_steps": 250000, "save_steps": 250000, "logging_steps": 5000, "early_stopping_patience": 1,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "wandb_project": "mal2026-utilization-only", "wandb_entity": None, "utilization_only": True, "utilization_label": "utilization_only",
    }


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True, check=False)
    if result.returncode:
        raise SystemExit(f"guard failed: {command!r}\n{result.stdout}\n{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--hours", type=int, default=23)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    need(args.run_id.startswith("utilization-only-20260720-") and 1 <= args.hours <= 23, "run ID/hours are outside approved utilization envelope")
    need(PYTHON.is_file() and MANIFEST.is_file() and QWEN3.is_dir(), "canonical interpreter, manifest, or local score model missing")
    runtime = ROOT / "outputs/reservations" / args.run_id
    need(not runtime.exists(), "unique runtime already exists")
    runtime.mkdir(parents=True); configs = runtime / "configs"; configs.mkdir()
    jobs: dict[str, dict] = {}
    for gpu, field in ((1, "content"), (2, "organization"), (3, "expression")):
        run_id = f"{args.run_id}-{field}-gpu{gpu}-utilization_only"
        config = configs / f"{field}-gpu{gpu}-utilization_only.json"
        write(config, score_config(run_id, field, gpu))
        run([str(PYTHON), "scripts/train_pre_sft_score_head.py", "--config", str(config), "--validate-config"])
        jobs[str(gpu)] = {"gpu": gpu, "utilization": {"run_purpose": "utilization_only", "gpu": gpu, "kind": "score_utilization", "command": [str(PYTHON), "scripts/train_pre_sft_score_head.py", "--config", str(config)]}, "priority": []}
    # GPU 0 stays judge-v3-exclusive.  The data-free backfill yields as soon as
    # a separately validated preparation process creates this aggregate marker.
    heartbeat = runtime / "heartbeats/gpu0-utilization_only.json"
    jobs["0"] = {"gpu": 0, "utilization": {"run_purpose": "utilization_only", "gpu": 0, "kind": "backfill", "command": [str(PYTHON), "scripts/gpu_local_utilization_backfill.py", "--physical-gpu", "0", "--seconds", "900", "--memory-fraction", "0.30", "--heartbeat", str(heartbeat)]}, "priority": [{"run_purpose": "higher_priority_research", "gpu": 0, "kind": "judge_v3", "ready_file": str(runtime / "judge_v3_validated.ready.json"), "command": ["bash", "scripts/run_qwen36_judge_v3_checked.sh", BATCH, "qwen36-judge-v3-pilot-20260720-002", str(MODEL), str(SERVER), "18084"]}]}
    end_at = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=args.hours)
    plan = {"schema_version": "mal2026-utilization-priority-v1", "run_purpose": "utilization_only", "project_root": str(ROOT), "runtime_dir": str(runtime), "python": str(PYTHON), "allowed_physical_gpus": [0, 1, 2, 3], "poll_seconds": 15, "max_temperature_c": 80, "end_at_utc": end_at.isoformat(), "gpus": jobs}
    plan_path = runtime / "utilization_only_plan.json"; write(plan_path, plan)
    write(runtime / "utilization_only_ledger.json", {"run_purpose": "utilization_only", "gpu_pinning": {str(g): {"CUDA_VISIBLE_DEVICES": str(g), "MAL2026_RESERVED_PHYSICAL_GPU": str(g)} for g in range(4)}, "epoch_cap": 10000, "validation_metrics_policy": "never loaded, reported, or used as evidence", "yield_policy": "GPU0 data-free backfill SIGTERMs when judge_v3_validated.ready.json exists; GPUs1-3 remain isolated score utilization jobs", "concurrent_aggregation": "not queued or consumed"})
    run([str(PYTHON), "-m", "py_compile", "scripts/train_pre_sft_score_head.py", "scripts/utilization_priority_watchdog.py"])
    run([str(PYTHON), "scripts/utilization_priority_watchdog.py", "--plan", str(plan_path), "--dry-run"])
    if args.launch:
        session = args.run_id
        need(subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode != 0, "tmux session already exists")
        subprocess.run(["tmux", "new-session", "-d", "-s", session, f"cd {ROOT} && exec {PYTHON} scripts/utilization_priority_watchdog.py --plan {plan_path}"], check=True)
    print(json.dumps({"status": "launched" if args.launch else "dry_run_passed", "run_purpose": "utilization_only", "runtime": str(runtime), "plan": str(plan_path), "tmux_session": args.run_id if args.launch else None}, sort_keys=True))


if __name__ == "__main__":
    main()
