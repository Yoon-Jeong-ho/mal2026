#!/usr/bin/env python3
"""Durable GPU0 gates and GPU0--3 Qwen3 input/view improvement runner."""
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

from mal2026.rlaif_qwen3_embedding import AXES, MODEL_ID, MODEL_PATH, MODEL_REVISION, warmstart_provenance  # noqa: E402
from mal2026.rlaif_qwen3_improvement import (  # noqa: E402
    ARMS, EVAL_ROOT, PHASES, TRAIN_ROOT, evaluation_config, evaluation_dir,
    expected_steps, training_config, training_dir,
)


RUNTIME_ID = "20260726-006"
RUN_ID = "rlaif-qwen3-embedding-improvement-v1-20260726-006"
RUN_ROOT = TRAIN_ROOT / RUNTIME_ID
CONFIG_ROOT = RUN_ROOT / "configs"
LOG_ROOT = RUN_ROOT / "logs"
LEDGER = RUN_ROOT / "ledger.jsonl"
MANIFEST = RUN_ROOT / "manifest.json"
FINAL = ROOT / "outputs" / "aggregate-reports" / f"{RUN_ID}.final-summary.json"
ENSEMBLE_REPORT = EVAL_ROOT / "rlaif-qwen3-improvement-eval-v1-r0-ensemble-full-006" / "ensemble_metrics.json"
PYTHON = ROOT / ".venv-standard" / "bin" / "python"
GPU0_GATES = ("essay_only", "trait_specific")


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


def gpu_processes(gpus: Sequence[int]) -> list[str]:
    result = subprocess.run(["nvidia-smi", f"--id={','.join(map(str, gpus))}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], cwd=ROOT, text=True, capture_output=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def wait_idle(gpus: Sequence[int], timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while True:
        processes = gpu_processes(gpus)
        if not processes:
            return
        if time.monotonic() >= deadline:
            raise RunnerError(f"GPU conflict on {list(gpus)}; existing processes were not altered")
        time.sleep(5)


def config_path(kind: str, arm: str, phase: str) -> Path:
    return CONFIG_ROOT / f"{kind}-{arm}-{phase}.json"


def run_stage(stage: str, command: Sequence[str], gpus: Sequence[int]) -> None:
    log = LOG_ROOT / f"{stage}.log"
    need(not log.exists(), f"stage log already exists: {stage}")
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    scope = "GPU0" if list(gpus) == [0] else "GPUs 0,1,2,3"
    ledger({"stage": stage, "event": "start", "command": list(command), "resource_scope": scope})
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(list(command), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    ledger({"stage": stage, "event": "completed" if completed.returncode == 0 else "failed", "exit_code": completed.returncode, "log": str(log.relative_to(ROOT)), "resource_scope": scope})
    need(completed.returncode == 0, f"stage failed: {stage}")


def validate_train(arm: str, phase: str) -> dict[str, Any]:
    path = training_dir(arm, phase) / "training_complete.json"
    value = read_json(path)
    expected = expected_steps(arm, phase)
    essays = 2000 if phase == "full" else 4
    views = 3 if arm in {"trait_specific", "multi_rationale"} else 1
    need(value.get("status") == "completed" and value.get("arm") == arm and value.get("phase") == phase, "training identity differs")
    need(value.get("score_fields") == list(AXES) and value.get("average_target_used") is False, "training targets differ")
    need(value.get("unique_train_essays") == essays and value.get("train_input_records") == essays * views and value.get("global_step") == max(expected.values()), "training counts differ")
    metrics = value.get("train_metrics")
    need(isinstance(metrics, dict) and "train_loss" in metrics and all(math.isfinite(float(number)) for number in metrics.values()), "training metrics differ")
    ledger({"stage": f"validate-train-{arm}-{phase}", "event": "aggregate", "global_step": value["global_step"], "train_loss": metrics["train_loss"], "completion_sha256": file_sha(path), "resource_scope": "none"})
    return value


def validate_eval(arm: str, phase: str) -> dict[str, Any]:
    path = evaluation_dir(arm, phase) / "epoch_metrics.json"
    value = read_json(path)
    need(value.get("status") == "completed" and value.get("arm") == arm and value.get("phase") == phase, "evaluation identity differs")
    need(value.get("score_fields") == list(AXES) and value.get("average_target_used") is False, "evaluation targets differ")
    rows = value.get("epoch_results")
    need(isinstance(rows, list) and [row.get("epoch") for row in rows] == list(expected_steps(arm, phase)), "evaluation epoch sequence differs")
    best = value.get("best_epoch_by_validation_macro_rmse_then_spearman")
    need(isinstance(best, dict) and math.isfinite(float(best["metrics"]["three_axis_macro_rmse"])), "evaluation best result differs")
    ledger({"stage": f"validate-eval-{arm}-{phase}", "event": "aggregate", "best_epoch": best["epoch"], "best_metrics": best["metrics"], "report_sha256": file_sha(path), "resource_scope": "none"})
    return value


def prepare() -> dict[str, Any]:
    need(PYTHON.is_file() and MODEL_PATH.is_dir() and not RUN_ROOT.exists() and not FINAL.exists(), "runner environment/output freshness differs")
    RUN_ROOT.mkdir(parents=True)
    CONFIG_ROOT.mkdir()
    LOG_ROOT.mkdir()
    for arm in ARMS:
        write_new_json(config_path("train", arm, "full"), training_config(arm, "full"))
        write_new_json(config_path("eval", arm, "full"), evaluation_config(arm, "full"))
    for arm in GPU0_GATES:
        write_new_json(config_path("train", arm, "gpu0_preflight"), training_config(arm, "gpu0_preflight"))
        write_new_json(config_path("eval", arm, "gpu0_preflight"), evaluation_config(arm, "gpu0_preflight"))
    manifest = {
        "schema_version": "mal2026-rlaif-qwen3-improvement-run-v1", "run_id": RUN_ID,
        "status": "running", "started_at": now(), "completed_at": None, "failed_at": None, "failure": None,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "resource_scope": {"preflight": [0], "full": [0, 1, 2, 3], "authorization": "default MAL2026 GPU scope"},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "path": str(MODEL_PATH.resolve())},
        "arms": list(ARMS), "gpu0_gate_arms": list(GPU0_GATES), "score_fields": list(AXES), "average_target_used": False,
        "warmstart": warmstart_provenance(),
        "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "datasets", "safetensors")},
        "protocol_record": "docs/experiment_records/rlaif_qwen3_embedding_improvement_program_v1_20260726_004.md",
        "validation_use": "predeclared descriptive comparison on previously exposed validation",
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
    }
    write_new_json(MANIFEST, manifest)
    ledger({"stage": "runner", "event": "start", "git_sha": manifest["git_sha"], "resource_scope": "none", "gpu_scope_authorization": "default"})
    return manifest


def main() -> None:
    manifest = prepare()
    results: dict[str, Any] = {}
    try:
        for arm in GPU0_GATES:
            wait_idle([0])
            run_stage(f"train-{arm}-gpu0-preflight", [str(PYTHON), "scripts/train_rlaif_qwen3_embedding_improvement.py", "--config", str(config_path("train", arm, "gpu0_preflight"))], [0])
            validate_train(arm, "gpu0_preflight")
            wait_idle([0])
            run_stage(f"eval-{arm}-gpu0-preflight", [str(PYTHON), "scripts/evaluate_rlaif_qwen3_embedding_improvement.py", "--config", str(config_path("eval", arm, "gpu0_preflight"))], [0])
            validate_eval(arm, "gpu0_preflight")
        ledger({"stage": "gpu0-preflight", "event": "smoke_pass", "decision": "continue through fixed GPU0-3 program", "resource_scope": "GPU0"})
        wait_idle([0, 1, 2, 3])
        run_stage("evaluate-r0-epoch-ensembles", [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/evaluate_rlaif_qwen3_epoch_ensemble_v1.py"], [0, 1, 2, 3])
        ensemble = read_json(ENSEMBLE_REPORT)
        need(ensemble.get("status") == "completed" and ensemble.get("average_target_used") is False, "ensemble validation differs")
        ledger({"stage": "validate-r0-epoch-ensembles", "event": "aggregate", "prediction_metrics": ensemble["prediction_ensemble"]["metrics"], "soup_metrics": ensemble["state_soup"]["metrics"], "report_sha256": file_sha(ENSEMBLE_REPORT), "resource_scope": "none"})
        for arm in ARMS:
            wait_idle([0, 1, 2, 3])
            run_stage(f"train-{arm}-full", [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/train_rlaif_qwen3_embedding_improvement.py", "--config", str(config_path("train", arm, "full"))], [0, 1, 2, 3])
            validate_train(arm, "full")
            wait_idle([0, 1, 2, 3])
            run_stage(f"eval-{arm}-full", [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/evaluate_rlaif_qwen3_embedding_improvement.py", "--config", str(config_path("eval", arm, "full"))], [0, 1, 2, 3])
            results[arm] = validate_eval(arm, "full")
        candidates = []
        for arm, result in results.items():
            candidates.append({"arm": arm, **result["best_epoch_by_validation_macro_rmse_then_spearman"]})
        candidates.extend([
            {"arm": "r0_epoch1_4_prediction_ensemble", "epoch": None, "metrics": ensemble["prediction_ensemble"]["metrics"]},
            {"arm": "r0_epoch1_4_state_soup", "epoch": None, "metrics": ensemble["state_soup"]["metrics"]},
        ])
        ranked = sorted(candidates, key=lambda row: (float(row["metrics"]["three_axis_macro_rmse"]), -float(row["metrics"]["three_axis_macro_spearman"]), 10**9 if row["epoch"] is None else int(row["epoch"]), row["arm"]))
        summary = {
            "schema_version": "mal2026-rlaif-qwen3-improvement-final-v1", "status": "completed", "run_id": RUN_ID,
            "git_sha": manifest["git_sha"], "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "score_fields": list(AXES), "average_target_used": False, "ranked_results": ranked,
            "arm_epoch_results": {arm: result["epoch_results"] for arm, result in results.items()},
            "selection_rule": "lower macro RMSE, then higher macro Spearman, then earlier epoch, then arm name",
            "selection_caveat": "validation was previously exposed; descriptive development evidence only",
            "target_macro_rmse": 0.4213,
            "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
        }
        write_new_json(FINAL, summary)
        rewrite_manifest({**manifest, "status": "completed", "completed_at": now()})
        ledger({"stage": "final-summary", "event": "completed", "best": ranked[0], "evidence_ref": str(FINAL.relative_to(ROOT)), "resource_scope": "none", "decision": "continue to separately frozen full-parameter AI-Hub arm"})
    except Exception as exc:
        rewrite_manifest({**manifest, "status": "failed", "failed_at": now(), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        ledger({"stage": "runner", "event": "failed", "failure_type": type(exc).__name__, "failure_message": str(exc), "resource_scope": "none", "decision": "stop and preserve"})
        raise


if __name__ == "__main__":
    main()
