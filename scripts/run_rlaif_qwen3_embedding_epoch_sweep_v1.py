#!/usr/bin/env python3
"""Durable GPU0 gate -> DDP4 train -> twelve-checkpoint evaluation runner."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.rlaif_qwen3_embedding import (  # noqa: E402
    AXES,
    EXPECTED_WARMSTART_SHA256,
    MODEL_ID,
    MODEL_PATH,
    MODEL_REVISION,
    RATIONALE_SOURCE,
    SOURCE_SHA256,
    WARMSTART_STATE,
    warmstart_provenance,
)
from mal2026.rlaif_qwen3_epoch_sweep import (  # noqa: E402
    FULL_EPOCHS,
    checkpoint_dir,
    evaluation_config,
    evaluation_dir,
    expected_checkpoint_steps,
    training_config,
    training_dir,
)


RUNTIME_ID = "20260726-003"
RUN_ID = "rlaif-qwen3-embedding-epoch-sweep-v1-20260726-003"
RUN_ROOT = ROOT / "outputs" / "rlaif-qwen3-embedding-epoch-sweep-v1" / RUNTIME_ID
CONFIG_ROOT = RUN_ROOT / "configs"
LOG_ROOT = RUN_ROOT / "logs"
LEDGER = RUN_ROOT / "ledger.jsonl"
MANIFEST = RUN_ROOT / "manifest.json"
FINAL = ROOT / "outputs" / "aggregate-reports" / f"{RUN_ID}.final-summary.json"
PREVIOUS = ROOT / "outputs" / "aggregate-reports" / "rlaif-qwen3-embedding-comparison-v1-20260726-002.final-summary.json"
PYTHON = ROOT / ".venv-standard" / "bin" / "python"


class RunnerError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"unreadable JSON: {path}") from exc
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), f"refusing to replace {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rewrite_manifest(value: Mapping[str, Any]) -> None:
    temporary = MANIFEST.with_suffix(".json.tmp")
    need(not temporary.exists(), "manifest temporary already exists")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def ledger(value: Mapping[str, Any]) -> None:
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": now(), **value}, ensure_ascii=False, sort_keys=True) + "\n")


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def environment() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "datasets", "safetensors")}


def gpu_processes(gpus: Sequence[int]) -> list[str]:
    completed = subprocess.run(["nvidia-smi", f"--id={','.join(str(gpu) for gpu in gpus)}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], cwd=ROOT, text=True, capture_output=True, check=True)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def wait_idle(gpus: Sequence[int], timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while True:
        processes = gpu_processes(gpus)
        if not processes:
            return
        if time.monotonic() >= deadline:
            raise RunnerError(f"GPU boundary conflict on {list(gpus)}; existing processes were not altered")
        time.sleep(5)


def config_path(kind: str, phase: str) -> Path:
    return CONFIG_ROOT / f"{kind}-{phase}.json"


def run_stage(stage: str, command: Sequence[str], gpus: Sequence[int]) -> None:
    log = LOG_ROOT / f"{stage}.log"
    need(not log.exists(), f"stage log already exists: {stage}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
    env["TOKENIZERS_PARALLELISM"] = "false"
    scope = "GPU0" if list(gpus) == [0] else "GPUs 0,1,2,3"
    ledger({"stage": stage, "event": "start", "command": list(command), "resource_scope": scope})
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(list(command), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    ledger({"stage": stage, "event": "completed" if completed.returncode == 0 else "failed", "exit_code": completed.returncode, "log": str(log.relative_to(ROOT)), "resource_scope": scope})
    need(completed.returncode == 0, f"stage failed: {stage}")


def validate_training(phase: str) -> dict[str, Any]:
    path = training_dir(phase) / "training_complete.json"
    value = read_json(path)
    expected = expected_checkpoint_steps(phase)
    records = 2000 if phase == "full" else 4
    need(value.get("status") == "completed" and value.get("phase") == phase, f"{phase} training identity differs")
    need(value.get("score_fields") == list(AXES) and value.get("average_target_used") is False, f"{phase} training score targets differ")
    need(value.get("train_records") == records and value.get("global_step") == max(expected.values()), f"{phase} training count/update differs")
    checkpoints = value.get("checkpoints")
    need(isinstance(checkpoints, list) and [item.get("epoch") for item in checkpoints] == list(expected), f"{phase} checkpoint sequence differs")
    for item in checkpoints:
        epoch = int(item["epoch"])
        state = checkpoint_dir(training_dir(phase), epoch) / "trainable_model.safetensors"
        need(item.get("global_step") == expected[epoch] and state.is_file() and item.get("trainable_state_sha256") == file_sha(state), f"{phase}/epoch-{epoch} checkpoint differs")
    metrics = value.get("train_metrics")
    need(isinstance(metrics, dict) and "train_loss" in metrics and all(isinstance(number, (int, float)) and math.isfinite(float(number)) for number in metrics.values()), f"{phase} training metrics differ")
    ledger({"stage": f"train-{phase}", "event": "aggregate", "global_step": max(expected.values()), "checkpoint_count": len(expected), "train_loss": metrics["train_loss"], "completion_sha256": file_sha(path), "resource_scope": "none"})
    return value


def validate_evaluation(phase: str) -> dict[str, Any]:
    path = evaluation_dir(phase) / "epoch_sweep_metrics.json"
    value = read_json(path)
    expected = expected_checkpoint_steps(phase)
    population = 400 if phase == "full" else 4
    need(value.get("status") == "completed" and value.get("phase") == phase, f"{phase} evaluation identity differs")
    need(value.get("score_fields") == list(AXES) and value.get("average_target_used") is False, f"{phase} evaluation score targets differ")
    rows = value.get("epoch_results")
    need(isinstance(rows, list) and [row.get("epoch") for row in rows] == list(expected), f"{phase} epoch result sequence differs")
    for row in rows:
        metrics = row.get("metrics")
        need(isinstance(metrics, dict) and set(metrics) == {*AXES, "three_axis_macro_rmse", "three_axis_macro_spearman"}, f"{phase}/epoch metrics fields differ")
        need(all(math.isfinite(float(metrics[axis][metric])) for axis in AXES for metric in ("rmse", "spearman")), f"{phase}/epoch metric is non-finite")
    need(value.get("validation") == {"unique_essays": population, "input_records": population, "predictions_per_essay_per_checkpoint": 1, "checkpoint_evaluations": len(expected), "rationale_sources_combined": 0}, f"{phase} validation population differs")
    best = value.get("best_epoch_by_validation_macro_rmse_then_spearman")
    need(isinstance(best, dict) and best.get("epoch") in expected, f"{phase} best epoch differs")
    ledger({"stage": f"evaluate-{phase}", "event": "aggregate", "checkpoint_count": len(rows), "best_epoch": best["epoch"], "best_metrics": best["metrics"], "report_sha256": file_sha(path), "resource_scope": "none"})
    return value


def prepare() -> dict[str, Any]:
    need(PYTHON.is_file() and MODEL_PATH.is_dir() and not MODEL_PATH.is_symlink(), "epoch-sweep environment/model is unavailable")
    need(not RUN_ROOT.exists() and not FINAL.exists(), "epoch-sweep runtime/final already exists")
    RUN_ROOT.mkdir(parents=True)
    CONFIG_ROOT.mkdir()
    LOG_ROOT.mkdir()
    warm = warmstart_provenance()
    need(file_sha(WARMSTART_STATE) == EXPECTED_WARMSTART_SHA256, "epoch-sweep warm-start bytes differ")
    for phase in ("gpu0_preflight", "full"):
        write_new_json(config_path("train", phase), training_config(phase))
        write_new_json(config_path("eval", phase), evaluation_config(phase))
    manifest = {
        "schema_version": "mal2026-rlaif-qwen3-embedding-epoch-sweep-run-v1",
        "run_id": RUN_ID,
        "status": "running",
        "started_at": now(),
        "completed_at": None,
        "failed_at": None,
        "failure": None,
        "git_sha": git_sha(),
        "resource_scope": {"preflight": [0], "full": [0, 1, 2, 3], "authorization": "default MAL2026 GPU scope"},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "path": str(MODEL_PATH.resolve())},
        "arm": "qwen3_aihub_warmstart",
        "rationale_source": RATIONALE_SOURCE,
        "score_fields": list(AXES),
        "average_target_used": False,
        "canonical_source_sha256": dict(SOURCE_SHA256),
        "warmstart": warm,
        "epoch_checkpoints": list(range(1, FULL_EPOCHS + 1)),
        "validation_use": "explicit_user_authorized_descriptive_epoch_sweep_on_previously_exposed_validation",
        "environment": environment(),
        "commands": {"runner": f"PYTHONPATH=src .venv-standard/bin/python scripts/{Path(__file__).name}"},
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
    }
    write_new_json(MANIFEST, manifest)
    ledger({"stage": "runner", "event": "start", "resource_scope": "none", "gpu_scope_authorization": "default", "git_sha": manifest["git_sha"]})
    return manifest


def final_summary(manifest: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    previous = read_json(PREVIOUS)
    previous_best = previous.get("best_qwen3_arm")
    need(isinstance(previous_best, dict), "previous Qwen3 summary differs")
    return {
        "schema_version": "mal2026-rlaif-qwen3-embedding-epoch-sweep-final-v1",
        "status": "completed",
        "run_id": RUN_ID,
        "git_sha": manifest["git_sha"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "arm": "qwen3_aihub_warmstart",
        "rationale_source": RATIONALE_SOURCE,
        "score_fields": list(AXES),
        "average_target_used": False,
        "epoch_results": evaluation["epoch_results"],
        "best_epoch_by_validation_macro_rmse_then_spearman": evaluation["best_epoch_by_validation_macro_rmse_then_spearman"],
        "previous_fixed_epoch12_result": previous_best,
        "selection_rule": "lower three-axis macro RMSE, then higher macro Spearman, then earlier epoch",
        "selection_caveat": evaluation["selection_caveat"],
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
    }


def main() -> None:
    manifest = prepare()
    try:
        wait_idle([0])
        run_stage("train-gpu0-preflight", [str(PYTHON), "scripts/train_rlaif_qwen3_embedding_epoch_sweep.py", "--config", str(config_path("train", "gpu0_preflight"))], [0])
        validate_training("gpu0_preflight")
        wait_idle([0])
        run_stage("evaluate-gpu0-preflight", [str(PYTHON), "scripts/evaluate_rlaif_qwen3_embedding_epoch_sweep.py", "--config", str(config_path("eval", "gpu0_preflight"))], [0])
        validate_evaluation("gpu0_preflight")
        ledger({"stage": "gpu0-preflight", "event": "smoke_pass", "resource_scope": "GPU0", "decision": "continue to fixed full DDP4 epoch sweep"})
        wait_idle([0, 1, 2, 3])
        run_stage("train-full", [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/train_rlaif_qwen3_embedding_epoch_sweep.py", "--config", str(config_path("train", "full"))], [0, 1, 2, 3])
        validate_training("full")
        wait_idle([0, 1, 2, 3])
        run_stage("evaluate-full", [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/evaluate_rlaif_qwen3_embedding_epoch_sweep.py", "--config", str(config_path("eval", "full"))], [0, 1, 2, 3])
        evaluation = validate_evaluation("full")
        summary = final_summary(manifest, evaluation)
        write_new_json(FINAL, summary)
        rewrite_manifest({**manifest, "status": "completed", "completed_at": now()})
        best = summary["best_epoch_by_validation_macro_rmse_then_spearman"]
        ledger({"stage": "final-summary", "event": "completed", "evidence_ref": str(FINAL.relative_to(ROOT)), "best_epoch": best["epoch"], "best_three_axis_macro_rmse": best["metrics"]["three_axis_macro_rmse"], "resource_scope": "none", "decision": "complete"})
    except Exception as exc:
        rewrite_manifest({**manifest, "status": "failed", "failed_at": now(), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        ledger({"stage": "runner", "event": "failed", "failure_type": type(exc).__name__, "failure_message": str(exc), "resource_scope": "none", "decision": "stop and preserve"})
        raise


if __name__ == "__main__":
    main()
