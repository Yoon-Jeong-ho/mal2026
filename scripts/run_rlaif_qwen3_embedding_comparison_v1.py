#!/usr/bin/env python3
"""Durable GPU0-preflight -> DDP4 Qwen3-Embedding two-arm comparison."""
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
    ARMS,
    AXES,
    EVAL_ROOT,
    MODEL_ID,
    MODEL_PATH,
    MODEL_REVISION,
    RATIONALE_SOURCE,
    SOURCE_SHA256,
    TRAIN_ROOT,
    WARMSTART_METADATA,
    WARMSTART_STATE,
    EXPECTED_WARMSTART_SHA256,
    evaluation_config,
    evaluation_dir,
    training_config,
    training_dir,
    warmstart_provenance,
)


RUNTIME_ID = "20260726-001"
RUN_ID = "rlaif-qwen3-embedding-comparison-v1-20260726-001"
RUN_ROOT = ROOT / "outputs" / "rlaif-qwen3-embedding-comparison-v1" / RUNTIME_ID
CONFIG_ROOT = RUN_ROOT / "configs"
LOG_ROOT = RUN_ROOT / "logs"
LEDGER = RUN_ROOT / "ledger.jsonl"
MANIFEST = RUN_ROOT / "manifest.json"
FINAL = ROOT / "outputs" / "aggregate-reports" / f"{RUN_ID}.final-summary.json"
PYTHON = ROOT / ".venv-standard" / "bin" / "python"
QWEN25_BASELINE = ROOT / "outputs" / "aggregate-reports" / "rlaif-top3-encoder-v1-20260725-001.final-summary.json"


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
    entry = {"timestamp": now(), **value}
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def environment() -> dict[str, str]:
    packages = ("torch", "transformers", "peft", "accelerate", "datasets", "safetensors")
    return {name: importlib.metadata.version(name) for name in packages}


def gpu_processes(gpus: Sequence[int]) -> list[str]:
    command = ["nvidia-smi", f"--id={','.join(str(gpu) for gpu in gpus)}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
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


def config_path(kind: str, arm: str, phase: str) -> Path:
    return CONFIG_ROOT / f"{kind}-{arm}-{phase}.json"


def run_stage(stage: str, command: Sequence[str], gpus: Sequence[int]) -> None:
    log = LOG_ROOT / f"{stage}.log"
    need(not log.exists(), f"stage log already exists: {stage}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
    env["TOKENIZERS_PARALLELISM"] = "false"
    ledger({"stage": stage, "event": "start", "command": list(command), "resource_scope": "GPU0" if list(gpus) == [0] else "GPUs 0,1,2,3"})
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(list(command), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    ledger({"stage": stage, "event": "completed" if completed.returncode == 0 else "failed", "exit_code": completed.returncode, "log": str(log.relative_to(ROOT)), "resource_scope": "GPU0" if list(gpus) == [0] else "GPUs 0,1,2,3"})
    need(completed.returncode == 0, f"stage failed: {stage}")


def validate_training(arm: str, phase: str) -> dict[str, Any]:
    path = training_dir(arm, phase) / "training_complete.json"
    value = read_json(path)
    expected_records, expected_steps = (4, 1) if phase == "gpu0_preflight" else (2000, 384)
    need(value.get("status") == "completed" and value.get("arm") == arm and value.get("phase") == phase, f"{arm}/{phase} completion identity differs")
    need(value.get("score_fields") == list(AXES) and value.get("average_target_used") is False, f"{arm}/{phase} score targets differ")
    need(value.get("train_records") == expected_records and value.get("global_step") == expected_steps, f"{arm}/{phase} update/count differs")
    metrics = value.get("train_metrics")
    need(isinstance(metrics, dict) and "train_loss" in metrics and all(isinstance(metric, (int, float)) and math.isfinite(float(metric)) for metric in metrics.values()), f"{arm}/{phase} metrics are non-finite")
    state = training_dir(arm, phase) / "trainable_model.safetensors"
    need(state.is_file() and value.get("trainable_state_sha256") == file_sha(state), f"{arm}/{phase} state checksum differs")
    ledger({"stage": f"train-{arm}-{phase}", "event": "aggregate", "global_step": expected_steps, "train_loss": metrics["train_loss"], "completion_sha256": file_sha(path), "resource_scope": "none"})
    return value


def validate_evaluation(arm: str) -> dict[str, Any]:
    path = evaluation_dir(arm) / "aggregate_metrics.json"
    value = read_json(path)
    need(value.get("status") == "completed" and value.get("arm") == arm and value.get("score_fields") == list(AXES), f"{arm} evaluation identity differs")
    need(value.get("average_target_used") is False, f"{arm} evaluation used average")
    need(value.get("validation") == {"unique_essays": 400, "input_records": 400, "predictions_per_essay": 1, "rationale_sources_combined": 0}, f"{arm} validation population differs")
    metrics = value.get("metrics")
    expected = {*AXES, "three_axis_macro_rmse", "three_axis_macro_spearman"}
    need(isinstance(metrics, dict) and set(metrics) == expected, f"{arm} metric fields differ")
    for axis in AXES:
        need(set(metrics[axis]) == {"rmse", "spearman"} and all(math.isfinite(float(number)) for number in metrics[axis].values()), f"{arm}/{axis} metrics differ")
    need(math.isfinite(float(metrics["three_axis_macro_rmse"])) and math.isfinite(float(metrics["three_axis_macro_spearman"])), f"{arm} macro metrics differ")
    ledger({"stage": f"evaluate-{arm}-validation", "event": "aggregate", "metrics": metrics, "report_sha256": file_sha(path), "resource_scope": "none"})
    return value


def prepare() -> dict[str, Any]:
    need(PYTHON.is_file(), ".venv-standard Python is unavailable")
    need(MODEL_PATH.is_dir() and not MODEL_PATH.is_symlink(), "Qwen3 model snapshot is unavailable")
    need(not RUN_ROOT.exists(), f"runtime already exists: {RUN_ROOT}")
    need(not FINAL.exists(), f"final summary already exists: {FINAL}")
    RUN_ROOT.mkdir(parents=True)
    CONFIG_ROOT.mkdir()
    LOG_ROOT.mkdir()
    warm = warmstart_provenance()
    need(file_sha(WARMSTART_STATE) == EXPECTED_WARMSTART_SHA256, "AI-Hub Qwen3 warm-start bytes differ from the recorded checksum")
    for arm in ARMS:
        for phase in ("gpu0_preflight", "full"):
            write_new_json(config_path("train", arm, phase), training_config(arm, phase))
        write_new_json(config_path("eval", arm, "validation"), evaluation_config(arm))
    manifest = {
        "schema_version": "mal2026-rlaif-qwen3-embedding-comparison-run-v1",
        "run_id": RUN_ID,
        "status": "running",
        "started_at": now(),
        "completed_at": None,
        "failed_at": None,
        "failure": None,
        "git_sha": git_sha(),
        "resource_scope": {"preflight": [0], "full": [0, 1, 2, 3], "authorization": "default MAL2026 GPU scope"},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "path": str(MODEL_PATH.resolve())},
        "arms": list(ARMS),
        "rationale_source": RATIONALE_SOURCE,
        "score_fields": list(AXES),
        "average_target_used": False,
        "canonical_source_sha256": dict(SOURCE_SHA256),
        "warmstart": warm,
        "environment": environment(),
        "commands": {"runner": f"PYTHONPATH=src .venv-standard/bin/python scripts/{Path(__file__).name}"},
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
    }
    write_new_json(MANIFEST, manifest)
    ledger({"stage": "runner", "event": "start", "resource_scope": "none", "gpu_scope_authorization": "default", "git_sha": manifest["git_sha"]})
    return manifest


def final_summary(manifest: Mapping[str, Any], evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prior = read_json(QWEN25_BASELINE)
    prior_best = prior.get("best_by_three_axis_macro_rmse_then_spearman")
    need(isinstance(prior_best, dict), "Qwen2.5 baseline summary differs")
    rows = [{
        "arm": value["arm"],
        "initialization": value["initialization"],
        "metrics": value["metrics"],
        "training_run_id": value["training_run_id"],
        "trainable_state_sha256": value["trainable_state_sha256"],
        "validation": value["validation"],
    } for value in evaluations]
    best = min(rows, key=lambda row: (float(row["metrics"]["three_axis_macro_rmse"]), -float(row["metrics"]["three_axis_macro_spearman"]), row["arm"]))
    return {
        "schema_version": "mal2026-rlaif-qwen3-embedding-comparison-final-v1",
        "status": "completed",
        "run_id": RUN_ID,
        "git_sha": manifest["git_sha"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "rationale_source": RATIONALE_SOURCE,
        "score_fields": list(AXES),
        "average_target_used": False,
        "arms": rows,
        "best_qwen3_arm": {"arm": best["arm"], "metrics": best["metrics"]},
        "prior_qwen25_rationale_baseline": {"source_key": prior_best.get("source_key"), "metrics": prior_best.get("metrics")},
        "selection_rule": "lower three-axis macro RMSE, then higher macro Spearman; comparison is descriptive on the already-exposed canonical validation split",
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
    }


def main() -> None:
    manifest = prepare()
    try:
        wait_idle([0])
        for arm in ARMS:
            stage = f"train-{arm}-gpu0_preflight"
            run_stage(stage, [str(PYTHON), "scripts/train_rlaif_qwen3_embedding.py", "--config", str(config_path("train", arm, "gpu0_preflight"))], [0])
            validate_training(arm, "gpu0_preflight")
            wait_idle([0])
        ledger({"stage": "gpu0-preflight", "event": "smoke_pass", "arms": list(ARMS), "resource_scope": "GPU0", "decision": "continue to fixed full DDP4 stage"})
        wait_idle([0, 1, 2, 3])
        evaluations: list[dict[str, Any]] = []
        for arm in ARMS:
            train_stage = f"train-{arm}-full"
            run_stage(train_stage, [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/train_rlaif_qwen3_embedding.py", "--config", str(config_path("train", arm, "full"))], [0, 1, 2, 3])
            validate_training(arm, "full")
            wait_idle([0, 1, 2, 3])
            eval_stage = f"evaluate-{arm}-validation"
            run_stage(eval_stage, [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/evaluate_rlaif_qwen3_embedding.py", "--config", str(config_path("eval", arm, "validation"))], [0, 1, 2, 3])
            evaluations.append(validate_evaluation(arm))
            wait_idle([0, 1, 2, 3])
        summary = final_summary(manifest, evaluations)
        write_new_json(FINAL, summary)
        completed = {**manifest, "status": "completed", "completed_at": now()}
        rewrite_manifest(completed)
        ledger({"stage": "final-summary", "event": "completed", "evidence_ref": str(FINAL.relative_to(ROOT)), "best_qwen3_arm": summary["best_qwen3_arm"]["arm"], "best_three_axis_macro_rmse": summary["best_qwen3_arm"]["metrics"]["three_axis_macro_rmse"], "resource_scope": "none", "decision": "complete"})
    except Exception as exc:
        failed = {**manifest, "status": "failed", "failed_at": now(), "failure": {"type": type(exc).__name__, "message": str(exc)}}
        rewrite_manifest(failed)
        ledger({"stage": "runner", "event": "failed", "failure_type": type(exc).__name__, "failure_message": str(exc), "resource_scope": "none", "decision": "stop and preserve"})
        raise


if __name__ == "__main__":
    main()
