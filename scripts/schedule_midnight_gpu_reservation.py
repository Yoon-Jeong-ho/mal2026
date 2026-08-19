#!/usr/bin/env python3
"""Validate and schedule the approved 2026-07-20 KST 0--3 reservation.

The dry-run and scheduling paths are GPU-free.  The durable tmux payload only
starts the watchdog at midnight KST; it does not query or initialize CUDA
before then.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-standard" / "bin" / "python"
RUNTIME = ROOT / "outputs" / "reservations" / "gpu0-3-20260720-0000-kst-001"
SESSION = "mal2026-resv-20260720-0000-kst"
NOT_BEFORE = "2026-07-20T00:00:00+09:00"
RESERVATION_END = "2026-07-21T00:00:00+09:00"
BATCH = "openai-rationale-terra-full-20260719-001"
JUDGE = "qwen36-judge-v2-pilot-20260720-001"
MODEL = ROOT / "outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
LLAMA = ROOT / "outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server"
LLAMA_REPO = ROOT / "outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/llama.cpp"
QWEN3 = ROOT / "outputs/model-cache/Qwen--Qwen3-Embedding-8B-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
MANIFEST = ROOT / "data/manifests/aihub_human_feedback_v1.json"
DERIVED = ROOT / "data/processed/restricted/openai_rationale_batches" / BATCH / "derived/train-only-candidates-v1-20260719-001"
CONFIG = ROOT / "configs/qwen36_gguf_judge.v2.pilot.json"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(command: list[str]) -> str:
    environment = {**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True, env=environment)
    if result.returncode:
        raise SystemExit(f"dry-run command failed: {command!r}\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def static_judge_gate() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    model = config["model"]
    runtime = config["runtime"]
    need(MODEL.is_file() and MODEL.stat().st_size == model["bytes"], "judge GGUF bytes gate failed")
    need(digest(MODEL) == model["sha256"], "judge GGUF sha256 gate failed")
    need(LLAMA.is_file() and LLAMA.stat().st_mode & 0o111, "llama-server executable gate failed")
    need(LLAMA_REPO.is_dir(), "llama.cpp source root is absent")
    need(run(["git", "-C", str(LLAMA_REPO), "rev-parse", "HEAD"]) == runtime["revision"], "llama.cpp immutable revision gate failed")
    tag = run(["git", "-C", str(LLAMA_REPO), "describe", "--tags", "--exact-match"])
    need(tag == runtime["release_tag"], "llama.cpp release-tag gate failed")
    manifest_path = DERIVED / "candidates.train.manifest.json"
    candidate_path = DERIVED / "candidates.train.jsonl"
    need(manifest_path.is_file() and candidate_path.is_file(), "train-only derived judge artifact is absent")
    derived = json.loads(manifest_path.read_text(encoding="utf-8"))
    need(derived.get("status") == "completed" and derived.get("batch_run_id") == BATCH and derived.get("split") == "train" and derived.get("row_count") == 6000 and derived.get("candidate_file_sha256") == digest(candidate_path), "derived train-only judge binding gate failed")
    need(config["selection"]["required_candidate_artifact"] == "derived/train-only-candidates-v1-20260719-001/candidates.train.jsonl" and config["selection"]["required_candidate_manifest"] == "derived/train-only-candidates-v1-20260719-001/candidates.train.manifest.json", "judge config no longer binds verified train-only artifact")
    need(config["runtime"]["gpu_allowlist"] == [0] and config["protocol"]["validation_policy"] == "never_load_validation_source_rows_or_construct_validation_requests" and not config["protocol"]["selection_artifact_permitted"], "judge isolation/GPU hard gate failed")
    run(["bash", "-n", "scripts/run_qwen36_judge_v2_pilot.sh"])
    run(["bash", "-n", "scripts/run_qwen36_judge_v2_checked.sh"])
    run(["env", "MAL2026_JUDGE_STATIC_ONLY=1", "bash", "scripts/run_qwen36_judge_v2_checked.sh", BATCH, JUDGE, str(MODEL), str(LLAMA), "18084"])
    return {"gguf_sha256": model["sha256"], "gguf_bytes": model["bytes"], "llama_revision": runtime["revision"], "llama_release_tag": tag, "derived_candidate_sha256": derived["candidate_file_sha256"], "derived_row_count": derived["row_count"]}


def head_config(run_id: str, field: str, seed: int, lora_r: int = 16) -> dict[str, Any]:
    return {
        "run_id": run_id, "target_field": field, "backbone": "qwen3_embedding", "model_id": "Qwen/Qwen3-Embedding-8B",
        "model_revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", "tokenizer_revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "model_path": str(QWEN3), "prepared_manifest": str(MANIFEST), "output_dir": str(ROOT / "outputs/standard-encoder-runs" / run_id),
        "max_length": 2048, "seed": seed, "learning_rate": 0.0001, "weight_decay": 0.01, "warmup_ratio": 0.05,
        "num_train_epochs": 20.0, "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1, "gradient_accumulation_steps": 16,
        "eval_steps": 100, "save_steps": 100, "logging_steps": 5, "early_stopping_patience": 3,
        "lora_r": lora_r, "lora_alpha": 32, "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "wandb_project": "mal2026-korean-writing-scoring", "wandb_entity": None,
    }


def write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare() -> tuple[Path, dict[str, Any]]:
    need(PYTHON.is_file() and PYTHON.stat().st_mode & 0o111, "documented .venv-standard interpreter is unavailable")
    need(QWEN3.is_dir() and MANIFEST.is_file(), "score-model local model or manifest is absent")
    need(not RUNTIME.exists(), f"reservation runtime already exists: {RUNTIME}")
    RUNTIME.mkdir(parents=True)
    configs = RUNTIME / "configs"; configs.mkdir()
    primary = {
        "content": "pre-sft-20260720-content-primary", "organization": "pre-sft-20260720-organization-primary", "expression": "pre-sft-20260720-expression-primary",
    }
    replicas = {
        "content": "pre-sft-20260720-content-replication-seed2027", "organization": "pre-sft-20260720-organization-ablation-r8",
    }
    paths: dict[str, Path] = {}
    for field, run_id in primary.items():
        path = configs / f"{field}-primary.json"; write(path, head_config(run_id, field, 2026)); paths[f"{field}_primary"] = path
    content_rep = configs / "content-replication.json"; write(content_rep, head_config(replicas["content"], "content", 2027)); paths["content_rep"] = content_rep
    org_ablation = configs / "organization-ablation-r8.json"; write(org_ablation, head_config(replicas["organization"], "organization", 2026, 8)); paths["org_ablation"] = org_ablation
    completion = {field: str(ROOT / "outputs/standard-encoder-runs" / run_id / "pre_sft_score_head_complete.json") for field, run_id in primary.items()}
    ensemble = configs / "ensemble-primary.json"
    write(ensemble, {"run_id": "pre-sft-20260720-primary-ensemble-selection-dev", "source": "selection_dev", "prepared_manifest": str(MANIFEST), "output_dir": str(ROOT / "outputs/standard-encoder-evals/pre-sft-20260720-primary-ensemble-selection-dev"), "head_completion_paths": completion, "per_device_eval_batch_size": 1})
    judge = static_judge_gate()
    for path in paths.values():
        run([str(PYTHON), "scripts/train_pre_sft_score_head.py", "--config", str(path), "--validate-config"])
    run([str(PYTHON), "scripts/evaluate_pre_sft_score_ensemble.py", "--config", str(ensemble), "--validate-plan"])
    run([str(PYTHON), "scripts/gpu_reservation_watchdog.py", "--help"])
    plan = {
        "schema_version": "mal2026-gpu-reservation-v1", "project_root": str(ROOT), "runtime_dir": str(RUNTIME), "python": str(PYTHON),
        "allowed_physical_gpus": [0, 1, 2, 3], "not_before_kst": NOT_BEFORE, "reservation_end_kst": RESERVATION_END,
        "poll_seconds": 15, "backfill": {"seconds": 300, "memory_fraction": 0.30, "max_temperature_c": 80, "max_cycles_per_gpu": 144},
        "queues": {
            "0": [{"id": "judge-v2-train-only", "kind": "judge_v2", "gpu": 0, "requires_files": [], "command": ["bash", "scripts/run_qwen36_judge_v2_checked.sh", BATCH, JUDGE, str(MODEL), str(LLAMA), "18084"]}],
            "1": [{"id": "content-primary", "kind": "pre_sft_score_head", "gpu": 1, "requires_files": [], "command": [str(PYTHON), "scripts/train_pre_sft_score_head.py", "--config", str(paths["content_primary"])]}, {"id": "content-replication-seed2027", "kind": "replication", "gpu": 1, "requires_files": [completion["content"]], "command": [str(PYTHON), "scripts/train_pre_sft_score_head.py", "--config", str(paths["content_rep"])]}],
            "2": [{"id": "organization-primary", "kind": "pre_sft_score_head", "gpu": 2, "requires_files": [], "command": [str(PYTHON), "scripts/train_pre_sft_score_head.py", "--config", str(paths["organization_primary"])]}, {"id": "organization-ablation-r8", "kind": "ablation", "gpu": 2, "requires_files": [completion["organization"]], "command": [str(PYTHON), "scripts/train_pre_sft_score_head.py", "--config", str(paths["org_ablation"])]}],
            "3": [{"id": "expression-primary", "kind": "pre_sft_score_head", "gpu": 3, "requires_files": [], "command": [str(PYTHON), "scripts/train_pre_sft_score_head.py", "--config", str(paths["expression_primary"])]}, {"id": "primary-external-average-ensemble", "kind": "ensemble", "gpu": 3, "requires_files": list(completion.values()), "command": [str(PYTHON), "scripts/evaluate_pre_sft_score_ensemble.py", "--config", str(ensemble)]}],
        },
    }
    plan_path = RUNTIME / "reservation_plan.json"; write(plan_path, plan)
    run([str(PYTHON), "scripts/gpu_reservation_watchdog.py", "--plan", str(plan_path), "--dry-run"])
    metadata = {"schema_version": "mal2026-reservation-schedule-metadata-v1", "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), "tmux_session": SESSION, "not_before_kst": NOT_BEFORE, "reservation_end_kst": RESERVATION_END, "gpu_scope": [0, 1, 2, 3], "judge_static_gate": judge, "score_model": {"backbone": "Qwen/Qwen3-Embedding-8B", "heads": ["content", "organization", "expression"], "average": "external_ensemble_only", "validation_source": "selection_dev"}, "queue_job_ids": [job["id"] for jobs in plan["queues"].values() for job in jobs], "dry_run": "passed_without_gpu_queries_or_cuda_initialization"}
    write(RUNTIME / "schedule_metadata.json", metadata)
    return plan_path, metadata


def tmux_schedule(plan: Path) -> None:
    result = subprocess.run(["tmux", "has-session", "-t", SESSION], check=False, capture_output=True)
    need(result.returncode != 0, f"tmux session already exists: {SESSION}")
    # The child shell sleeps until the exact KST boundary.  No watchdog/GPU
    # command runs before that point, and the watchdog repeats the time gate.
    command = f"cd {ROOT} && while [ \"$(TZ=Asia/Seoul date +%Y-%m-%dT%H:%M:%S%z)\" '<' \"2026-07-20T00:00:00+0900\" ]; do sleep 15; done; exec {PYTHON} scripts/gpu_reservation_watchdog.py --plan {plan}"
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, command], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="write a fresh ignored plan and validate every command without GPU access")
    parser.add_argument("--schedule", action="store_true", help="create plan then launch the persistent tmux sleeper")
    args = parser.parse_args()
    need(args.dry_run ^ args.schedule, "select exactly one of --dry-run or --schedule")
    if args.schedule and RUNTIME.exists():
        plan = RUNTIME / "reservation_plan.json"
        metadata_path = RUNTIME / "schedule_metadata.json"
        need(plan.is_file() and metadata_path.is_file(), "existing reservation runtime is incomplete; refusing overwrite")
        # Re-attest the immutable judge binding and all already-generated plan
        # commands before scheduling the same dry-run lineage.
        static_judge_gate()
        existing = json.loads(plan.read_text(encoding="utf-8"))
        for jobs in existing["queues"].values():
            for job in jobs:
                command = job["command"]
                if job["kind"] in {"pre_sft_score_head", "replication", "ablation"}:
                    run([command[0], command[1], command[2], command[3], "--validate-config"])
                elif job["kind"] == "ensemble":
                    run([command[0], command[1], command[2], command[3], "--validate-plan"])
        run([str(PYTHON), "scripts/gpu_reservation_watchdog.py", "--plan", str(plan), "--dry-run"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        plan, metadata = prepare()
    if args.schedule:
        tmux_schedule(plan)
        metadata["tmux_scheduled"] = True
        write(RUNTIME / "schedule_metadata.json", metadata)
    print(json.dumps({"status": "scheduled" if args.schedule else "dry_run_passed", "runtime": str(RUNTIME), "plan": str(plan), "tmux_session": SESSION if args.schedule else None}, sort_keys=True))

if __name__ == "__main__":
    main()
