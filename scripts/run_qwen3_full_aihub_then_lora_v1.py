#!/usr/bin/env python3
"""Sequential construction -> FSDP selection/refit -> rationale LoRA runner."""
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

from mal2026.qwen3_full_aihub_then_lora import (  # noqa: E402
    AXES, EVAL_ROOT, FULL_FINAL_STATE, FULL_REFIT_METADATA, FULL_ROOT,
    FULL_SELECTION_METADATA, MODEL_ID, MODEL_PATH, MODEL_REVISION,
    RATIONALE_ROOT, full_config, full_dir, rationale_config, rationale_dir,
    rationale_eval_dir, rationale_expected_steps,
)


RUNTIME_ID = "20260726-005"
RUN_ID = "qwen3-full-aihub-then-rationale-lora-v1-20260726-005"
RUN_ROOT = FULL_ROOT / RUNTIME_ID
CONFIG_ROOT = RUN_ROOT / "configs"
LOG_ROOT = RUN_ROOT / "logs"
LEDGER = RUN_ROOT / "ledger.jsonl"
MANIFEST = RUN_ROOT / "manifest.json"
CONSTRUCTION = FULL_ROOT / "qwen3-full-aihub-v1-gpu0-construction-005"
FINAL = ROOT / "outputs" / "aggregate-reports" / f"{RUN_ID}.final-summary.json"
PREVIOUS_FINAL = ROOT / "outputs" / "aggregate-reports" / "rlaif-qwen3-embedding-improvement-v1-20260726-006.final-summary.json"
PREVIOUS_MANIFEST = ROOT / "outputs" / "rlaif-qwen3-embedding-improvement-v1" / "20260726-006" / "manifest.json"
EPOCH_SWEEP_FINAL = ROOT / "outputs" / "aggregate-reports" / "rlaif-qwen3-embedding-epoch-sweep-v1-20260726-003.final-summary.json"
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
    need(not temporary.exists(), "manifest temporary exists")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def ledger(value: Mapping[str, Any]) -> None:
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": now(), **value}, ensure_ascii=False, sort_keys=True) + "\n")


def gpu_processes(gpus: Sequence[int]) -> list[str]:
    result = subprocess.run(["nvidia-smi", f"--id={','.join(map(str, gpus))}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], cwd=ROOT, text=True, capture_output=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def wait_idle(gpus: Sequence[int], timeout: int = 86400) -> None:
    deadline = time.monotonic() + timeout
    while True:
        processes = gpu_processes(gpus)
        if not processes:
            return
        if time.monotonic() >= deadline:
            raise RunnerError(f"GPU conflict on {list(gpus)}; existing processes were not altered")
        time.sleep(15)


def wait_previous(timeout: int = 86400) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        if PREVIOUS_FINAL.is_file():
            value = read_json(PREVIOUS_FINAL)
            need(value.get("status") == "completed", "prior improvement summary is not complete")
            return value
        if PREVIOUS_MANIFEST.is_file():
            state = read_json(PREVIOUS_MANIFEST).get("status")
            if state == "failed":
                raise RunnerError("prior fixed improvement program failed; full arm was not started")
        if time.monotonic() >= deadline:
            raise RunnerError("timed out waiting for prior fixed improvement program")
        time.sleep(30)


def config_path(kind: str, phase: str) -> Path:
    return CONFIG_ROOT / f"{kind}-{phase}.json"


def run_stage(stage: str, command: Sequence[str], gpus: Sequence[int]) -> None:
    log = LOG_ROOT / f"{stage}.log"
    need(not log.exists(), f"stage log exists: {stage}")
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    scope = "GPU0" if list(gpus) == [0] else "GPUs 0,1,2,3"
    ledger({"stage": stage, "event": "start", "command": list(command), "resource_scope": scope})
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(list(command), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    ledger({"stage": stage, "event": "completed" if completed.returncode == 0 else "failed", "exit_code": completed.returncode, "log": str(log.relative_to(ROOT)), "resource_scope": scope})
    need(completed.returncode == 0, f"stage failed: {stage}")


def validate_full(phase: str) -> dict[str, Any]:
    path = full_dir(phase) / "full_aihub_training_complete.json"
    value = read_json(path)
    need(value.get("status") == "completed" and value.get("phase") == phase and value.get("score_fields") == ["content", "organization", "expression", "average"], "full completion identity differs")
    need(value.get("average_target_used") is True and isinstance(value.get("selected_global_step"), int), "full target/update contract differs")
    metrics = value.get("train_metrics")
    need(isinstance(metrics, dict) and "train_loss" in metrics and all(math.isfinite(float(number)) for number in metrics.values()), "full metrics differ")
    if phase == "refit":
        need(FULL_FINAL_STATE.is_file() and value.get("model_state_sha256") == file_sha(FULL_FINAL_STATE), "refit full state differs")
    ledger({"stage": f"validate-{phase}", "event": "aggregate", "selected_global_step": value["selected_global_step"], "trainer_global_step": value["trainer_global_step"], "train_loss": metrics["train_loss"], "completion_sha256": file_sha(path), "resource_scope": "none"})
    return value


def validate_rationale(phase: str) -> dict[str, Any]:
    train_path = rationale_dir(phase) / "training_complete.json"
    eval_path = rationale_eval_dir(phase) / "epoch_metrics.json"
    train, evaluation = read_json(train_path), read_json(eval_path)
    need(train.get("status") == evaluation.get("status") == "completed" and train.get("score_fields") == evaluation.get("score_fields") == list(AXES), "rationale completion identity differs")
    need(train.get("average_target_used") is False and evaluation.get("average_target_used") is False, "rationale average target leaked")
    expected = rationale_expected_steps(phase)
    need(train.get("global_step") == max(expected.values()) and [row.get("epoch") for row in evaluation["epoch_results"]] == list(expected), "rationale epoch/update contract differs")
    best = evaluation["best_epoch_by_validation_macro_rmse_then_spearman"]
    need(math.isfinite(float(best["metrics"]["three_axis_macro_rmse"])), "rationale best metric differs")
    ledger({"stage": f"validate-rationale-{phase}", "event": "aggregate", "best": best, "training_sha256": file_sha(train_path), "evaluation_sha256": file_sha(eval_path), "resource_scope": "none"})
    return evaluation


def prepare() -> dict[str, Any]:
    need(PYTHON.is_file() and MODEL_PATH.is_dir() and not RUN_ROOT.exists() and not FINAL.exists(), "full runner environment/output freshness differs")
    RUN_ROOT.mkdir(parents=True)
    CONFIG_ROOT.mkdir()
    LOG_ROOT.mkdir()
    for phase in ("fsdp_gate", "selection", "refit"):
        write_new_json(config_path("full", phase), full_config(phase))
    for phase in ("gpu0_preflight", "full"):
        write_new_json(config_path("rationale", phase), rationale_config(phase))
    manifest = {
        "schema_version": "mal2026-qwen3-full-aihub-then-lora-run-v1", "run_id": RUN_ID,
        "status": "waiting_for_prior_program", "started_at": now(), "completed_at": None, "failed_at": None, "failure": None,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "resource_scope": {"construction": [0], "fsdp_and_full": [0, 1, 2, 3], "rationale_preflight": [0], "authorization": "default MAL2026 GPU scope"},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "path": str(MODEL_PATH.resolve())},
        "protocol_record": "docs/experiment_records/rlaif_qwen3_embedding_improvement_program_v1_20260726_004.md",
        "full_aihub_score_fields": ["content", "organization", "expression", "average"],
        "rationale_score_fields": list(AXES), "rationale_average_target_used": False,
        "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "datasets", "safetensors")},
        "privacy": "aggregate_only_no_rows_prompts_essays_feedback_rationales_ids_or_predictions_persisted",
    }
    write_new_json(MANIFEST, manifest)
    ledger({"stage": "runner", "event": "start", "git_sha": manifest["git_sha"], "resource_scope": "none", "gpu_scope_authorization": "default"})
    return manifest


def main() -> None:
    manifest = prepare()
    try:
        previous = wait_previous()
        rewrite_manifest({**manifest, "status": "running", "prior_program_sha256": file_sha(PREVIOUS_FINAL)})
        ledger({"stage": "prior-program", "event": "completed", "evidence_ref": str(PREVIOUS_FINAL.relative_to(ROOT)), "sha256": file_sha(PREVIOUS_FINAL), "resource_scope": "none"})
        wait_idle([0])
        run_stage("gpu0-construction", [str(PYTHON), "scripts/preflight_qwen3_full_aihub_construction.py", "--output", str(CONSTRUCTION)], [0])
        construction = read_json(CONSTRUCTION / "construction_gate.json")
        need(construction.get("status") == "completed" and construction.get("finite_loss") is True, "GPU0 construction gate differs")
        ledger({"stage": "gpu0-construction", "event": "smoke_pass", "report_sha256": file_sha(CONSTRUCTION / "construction_gate.json"), "decision": "continue to real FSDP4 update", "resource_scope": "GPU0"})
        wait_idle([0, 1, 2, 3])
        run_stage("fsdp4-one-update", [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/train_qwen3_full_aihub_v1.py", "--config", str(config_path("full", "fsdp_gate"))], [0, 1, 2, 3])
        validate_full("fsdp_gate")
        ledger({"stage": "fsdp4-one-update", "event": "smoke_pass", "decision": "continue to fixed selection/refit", "resource_scope": "GPUs 0,1,2,3"})
        wait_idle([0, 1, 2, 3])
        run_stage("full-aihub-selection", [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/train_qwen3_full_aihub_v1.py", "--config", str(config_path("full", "selection"))], [0, 1, 2, 3])
        selection = validate_full("selection")
        wait_idle([0, 1, 2, 3])
        run_stage("full-aihub-refit", [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/train_qwen3_full_aihub_v1.py", "--config", str(config_path("full", "refit"))], [0, 1, 2, 3])
        refit = validate_full("refit")
        wait_idle([0])
        run_stage("rationale-lora-gpu0-preflight", [str(PYTHON), "scripts/train_qwen3_full_aihub_rationale_lora_v1.py", "--config", str(config_path("rationale", "gpu0_preflight"))], [0])
        wait_idle([0])
        run_stage("rationale-eval-gpu0-preflight", [str(PYTHON), "scripts/evaluate_qwen3_full_aihub_rationale_lora_v1.py", "--config", str(config_path("rationale", "gpu0_preflight")), "--output", str(rationale_eval_dir("gpu0_preflight")), "--essay-limit", "4", "--per-device-batch-size", "4"], [0])
        validate_rationale("gpu0_preflight")
        ledger({"stage": "rationale-lora-gpu0-preflight", "event": "smoke_pass", "decision": "continue to fixed DDP4 rationale arm", "resource_scope": "GPU0"})
        wait_idle([0, 1, 2, 3])
        run_stage("rationale-lora-full", [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/train_qwen3_full_aihub_rationale_lora_v1.py", "--config", str(config_path("rationale", "full"))], [0, 1, 2, 3])
        wait_idle([0, 1, 2, 3])
        run_stage("rationale-eval-full", [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/evaluate_qwen3_full_aihub_rationale_lora_v1.py", "--config", str(config_path("rationale", "full")), "--output", str(rationale_eval_dir("full")), "--essay-limit", "400", "--per-device-batch-size", "8"], [0, 1, 2, 3])
        evaluation = validate_rationale("full")
        epoch_sweep = read_json(EPOCH_SWEEP_FINAL)
        prior_best = previous["ranked_results"][0]
        full_best = evaluation["best_epoch_by_validation_macro_rmse_then_spearman"]
        candidates = [
            {"arm": "full_parameter_aihub_then_rationale_lora", **full_best},
            {"arm": prior_best["arm"], "epoch": prior_best.get("epoch"), "metrics": prior_best["metrics"]},
            {"arm": "previous_aihub_lora_r0", **epoch_sweep["best_epoch_by_validation_macro_rmse_then_spearman"]},
        ]
        ranked = sorted(candidates, key=lambda row: (float(row["metrics"]["three_axis_macro_rmse"]), -float(row["metrics"]["three_axis_macro_spearman"]), 10**9 if row.get("epoch") is None else int(row["epoch"]), row["arm"]))
        summary = {
            "schema_version": "mal2026-qwen3-full-aihub-then-lora-final-v1", "status": "completed", "run_id": RUN_ID,
            "git_sha": manifest["git_sha"], "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "selection": {"selected_global_step": selection["selected_global_step"], "trainer_global_step": selection["trainer_global_step"], "selection_metrics": selection["selection_metrics"]},
            "refit": {"global_step": refit["trainer_global_step"], "model_state_sha256": refit["model_state_sha256"]},
            "full_parameter_arm_epoch_results": evaluation["epoch_results"], "comparison_ranked": ranked,
            "score_fields": list(AXES), "average_target_used_in_rationale_stage": False,
            "selection_caveat": "validation was previously exposed; descriptive development evidence only",
            "target_macro_rmse": 0.4213,
            "privacy": "aggregate_only_no_rows_prompts_essays_feedback_rationales_ids_or_predictions_persisted",
        }
        write_new_json(FINAL, summary)
        rewrite_manifest({**manifest, "status": "completed", "completed_at": now(), "prior_program_sha256": file_sha(PREVIOUS_FINAL)})
        ledger({"stage": "final-summary", "event": "completed", "best": ranked[0], "evidence_ref": str(FINAL.relative_to(ROOT)), "resource_scope": "none", "decision": "complete"})
    except Exception as exc:
        rewrite_manifest({**manifest, "status": "failed", "failed_at": now(), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        ledger({"stage": "runner", "event": "failed", "failure_type": type(exc).__name__, "failure_message": str(exc), "resource_scope": "none", "decision": "stop and preserve"})
        raise


if __name__ == "__main__":
    main()
