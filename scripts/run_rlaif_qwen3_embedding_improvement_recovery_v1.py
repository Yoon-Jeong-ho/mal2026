#!/usr/bin/env python3
"""OOM recovery for the three fixed long-input Qwen3 improvement arms."""
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
from mal2026.rlaif_qwen3_improvement import EVAL_ROOT, TRAIN_ROOT, evaluation_config, evaluation_dir, expected_steps, training_config, training_dir  # noqa: E402


RUNTIME_ID = "20260726-007"
RUN_ID = "rlaif-qwen3-embedding-improvement-v1-20260726-007"
RUN_ROOT = TRAIN_ROOT / RUNTIME_ID
CONFIG_ROOT = RUN_ROOT / "configs"
LOG_ROOT = RUN_ROOT / "logs"
LEDGER = RUN_ROOT / "ledger.jsonl"
MANIFEST = RUN_ROOT / "manifest.json"
FINAL = ROOT / "outputs" / "aggregate-reports" / f"{RUN_ID}.final-summary.json"
FAILED_MANIFEST = TRAIN_ROOT / "20260726-006" / "manifest.json"
ENSEMBLE_REPORT = EVAL_ROOT / "rlaif-qwen3-improvement-eval-v1-r0-ensemble-full-006" / "ensemble_metrics.json"
COMPLETED_REPORTS = {
    arm: EVAL_ROOT / f"rlaif-qwen3-improvement-eval-v1-{arm}-full-006" / "epoch_metrics.json"
    for arm in ("essay_only", "essay_instruction")
}
PENDING = ("rationale_instruction", "trait_specific", "multi_rationale")
PYTHON = ROOT / ".venv-standard" / "bin" / "python"


class RecoveryError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
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


def wait_idle(timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while True:
        result = subprocess.run(["nvidia-smi", "--id=0,1,2,3", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], cwd=ROOT, text=True, capture_output=True, check=True)
        if not [line for line in result.stdout.splitlines() if line.strip()]:
            return
        if time.monotonic() >= deadline:
            raise RecoveryError("GPU conflict on 0--3; existing processes were not altered")
        time.sleep(5)


def config_path(kind: str, arm: str) -> Path:
    return CONFIG_ROOT / f"{kind}-{arm}-full.json"


def run_stage(stage: str, command: Sequence[str]) -> None:
    log = LOG_ROOT / f"{stage}.log"
    need(not log.exists(), f"stage log exists: {stage}")
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "0,1,2,3", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    ledger({"stage": stage, "event": "start", "command": list(command), "resource_scope": "GPUs 0,1,2,3"})
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(list(command), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    ledger({"stage": stage, "event": "completed" if completed.returncode == 0 else "failed", "exit_code": completed.returncode, "log": str(log.relative_to(ROOT)), "resource_scope": "GPUs 0,1,2,3"})
    need(completed.returncode == 0, f"stage failed: {stage}")


def validate_arm(arm: str) -> dict[str, Any]:
    train_path = training_dir(arm, "full") / "training_complete.json"
    eval_path = evaluation_dir(arm, "full") / "epoch_metrics.json"
    train, evaluation = read_json(train_path), read_json(eval_path)
    need(train.get("status") == evaluation.get("status") == "completed" and train.get("arm") == evaluation.get("arm") == arm, "recovery arm identity differs")
    need(train.get("score_fields") == evaluation.get("score_fields") == list(AXES) and train.get("average_target_used") is False and evaluation.get("average_target_used") is False, "recovery score fields differ")
    need(train.get("global_step") == max(expected_steps(arm, "full").values()), "recovery update count differs")
    cfg = train["config"]
    need(cfg.get("per_device_train_batch_size") == 2 and cfg.get("gradient_accumulation_steps") == 8, "OOM recovery batch contract differs")
    best = evaluation["best_epoch_by_validation_macro_rmse_then_spearman"]
    need(math.isfinite(float(best["metrics"]["three_axis_macro_rmse"])), "recovery best metric differs")
    ledger({"stage": f"validate-{arm}", "event": "aggregate", "best": best, "training_sha256": file_sha(train_path), "evaluation_sha256": file_sha(eval_path), "resource_scope": "none"})
    return evaluation


def prepare() -> dict[str, Any]:
    need(PYTHON.is_file() and MODEL_PATH.is_dir() and not RUN_ROOT.exists() and not FINAL.exists(), "recovery environment/output freshness differs")
    failed = read_json(FAILED_MANIFEST)
    need(failed.get("status") == "failed" and failed.get("failure", {}).get("message") == "stage failed: train-rationale_instruction-full", "expected OOM source runtime differs")
    need(ENSEMBLE_REPORT.is_file() and all(path.is_file() for path in COMPLETED_REPORTS.values()), "completed source reports are unavailable")
    RUN_ROOT.mkdir(parents=True)
    CONFIG_ROOT.mkdir()
    LOG_ROOT.mkdir()
    for arm in PENDING:
        write_new_json(config_path("train", arm), training_config(arm, "full"))
        write_new_json(config_path("eval", arm), evaluation_config(arm, "full"))
    manifest = {"schema_version": "mal2026-rlaif-qwen3-improvement-recovery-run-v1", "run_id": RUN_ID, "status": "running", "started_at": now(), "completed_at": None, "failed_at": None, "failure": None, "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "resource_scope": {"full": [0, 1, 2, 3], "authorization": "default MAL2026 GPU scope"}, "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "path": str(MODEL_PATH.resolve())}, "reused_completed_arms": list(COMPLETED_REPORTS), "recovered_arms": list(PENDING), "recovery": {"source_runtime": "20260726-006", "reason": "rationale_input_cuda_oom_at_per_device_batch_4", "per_device_batch": 2, "gradient_accumulation": 8, "global_batch": 64, "allocator": "expandable_segments"}, "score_fields": list(AXES), "average_target_used": False, "warmstart": warmstart_provenance(), "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "datasets", "safetensors")}, "protocol_record": "docs/experiment_records/rlaif_qwen3_embedding_improvement_program_v1_20260726_004.md", "validation_use": "predeclared descriptive comparison on previously exposed validation", "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted"}
    write_new_json(MANIFEST, manifest)
    ledger({"stage": "runner", "event": "start", "git_sha": manifest["git_sha"], "resource_scope": "none", "recovery": manifest["recovery"]})
    return manifest


def main() -> None:
    manifest = prepare()
    recovered: dict[str, Any] = {}
    try:
        for arm in PENDING:
            wait_idle()
            run_stage(f"train-{arm}-full", [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/train_rlaif_qwen3_embedding_improvement.py", "--config", str(config_path("train", arm))])
            wait_idle()
            run_stage(f"eval-{arm}-full", [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/evaluate_rlaif_qwen3_embedding_improvement.py", "--config", str(config_path("eval", arm))])
            recovered[arm] = validate_arm(arm)
        all_results = {arm: read_json(path) for arm, path in COMPLETED_REPORTS.items()}
        all_results.update(recovered)
        ensemble = read_json(ENSEMBLE_REPORT)
        candidates = [{"arm": arm, **result["best_epoch_by_validation_macro_rmse_then_spearman"]} for arm, result in all_results.items()]
        candidates.extend([{"arm": "r0_epoch1_4_prediction_ensemble", "epoch": None, "metrics": ensemble["prediction_ensemble"]["metrics"]}, {"arm": "r0_epoch1_4_state_soup", "epoch": None, "metrics": ensemble["state_soup"]["metrics"]}])
        ranked = sorted(candidates, key=lambda row: (float(row["metrics"]["three_axis_macro_rmse"]), -float(row["metrics"]["three_axis_macro_spearman"]), 10**9 if row["epoch"] is None else int(row["epoch"]), row["arm"]))
        summary = {"schema_version": "mal2026-rlaif-qwen3-improvement-final-v1", "status": "completed", "run_id": RUN_ID, "git_sha": manifest["git_sha"], "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "score_fields": list(AXES), "average_target_used": False, "ranked_results": ranked, "arm_epoch_results": {arm: result["epoch_results"] for arm, result in all_results.items()}, "recovery": manifest["recovery"], "reused_evidence_sha256": {"ensemble": file_sha(ENSEMBLE_REPORT), **{arm: file_sha(path) for arm, path in COMPLETED_REPORTS.items()}}, "selection_rule": "lower macro RMSE, then higher macro Spearman, then earlier epoch, then arm name", "selection_caveat": "validation was previously exposed; descriptive development evidence only", "target_macro_rmse": 0.4213, "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted"}
        write_new_json(FINAL, summary)
        rewrite_manifest({**manifest, "status": "completed", "completed_at": now()})
        ledger({"stage": "final-summary", "event": "completed", "best": ranked[0], "evidence_ref": str(FINAL.relative_to(ROOT)), "resource_scope": "none", "decision": "continue to full-parameter AI-Hub arm"})
    except Exception as exc:
        rewrite_manifest({**manifest, "status": "failed", "failed_at": now(), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        ledger({"stage": "runner", "event": "failed", "failure_type": type(exc).__name__, "failure_message": str(exc), "resource_scope": "none", "decision": "stop and preserve"})
        raise


if __name__ == "__main__":
    main()

