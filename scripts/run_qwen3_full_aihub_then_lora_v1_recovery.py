#!/usr/bin/env python3
"""Resume runtime 009 after its evaluation-CLI freshness assertion failure."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
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
    AXES,
    FULL_FINAL_STATE,
    FULL_REFIT_METADATA,
    MODEL_ID,
    MODEL_PATH,
    MODEL_REVISION,
    FullRationaleConfig,
    rationale_checkpoint_dir,
    rationale_dir,
    rationale_eval_dir,
    rationale_expected_steps,
)


SOURCE_ROOT = ROOT / "outputs" / "qwen3-full-aihub-v1" / "20260726-009"
RUN_ID = "qwen3-full-aihub-then-rationale-lora-v1-recovery-20260726-010"
RUN_ROOT = ROOT / "outputs" / "qwen3-full-aihub-v1" / "20260726-010"
LOG_ROOT = RUN_ROOT / "logs"
LEDGER = RUN_ROOT / "ledger.jsonl"
MANIFEST = RUN_ROOT / "manifest.json"
FINAL = ROOT / "outputs" / "aggregate-reports" / f"{RUN_ID}.final-summary.json"
PRIOR_FINAL = ROOT / "outputs" / "aggregate-reports" / "rlaif-qwen3-embedding-improvement-v1-20260726-007.final-summary.json"
EPOCH_SWEEP_FINAL = ROOT / "outputs" / "aggregate-reports" / "rlaif-qwen3-embedding-epoch-sweep-v1-20260726-003.final-summary.json"
PYTHON = ROOT / ".venv-standard" / "bin" / "python"


class RecoveryError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"unreadable JSON: {path}") from exc
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), f"refusing to replace {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rewrite_manifest(value: Mapping[str, Any]) -> None:
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def ledger(value: Mapping[str, Any]) -> None:
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": now(), **value}, ensure_ascii=False, sort_keys=True) + "\n")


def gpu_processes(gpus: Sequence[int]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", f"--id={','.join(map(str, gpus))}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def wait_idle(gpus: Sequence[int], timeout: int = 86400) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if not gpu_processes(gpus):
            return
        if time.monotonic() >= deadline:
            raise RecoveryError(f"GPU conflict on {list(gpus)}; existing processes were not altered")
        time.sleep(15)


def run_stage(stage: str, command: Sequence[str], gpus: Sequence[int]) -> None:
    log = LOG_ROOT / f"{stage}.log"
    need(not log.exists(), f"stage log exists: {stage}")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    scope = "GPU0" if list(gpus) == [0] else "GPUs 0,1,2,3"
    ledger({"stage": stage, "event": "start", "command": list(command), "resource_scope": scope})
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(list(command), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    ledger({"stage": stage, "event": "completed" if completed.returncode == 0 else "failed", "exit_code": completed.returncode, "log": str(log.relative_to(ROOT)), "resource_scope": scope})
    need(completed.returncode == 0, f"stage failed: {stage}")


def config_path(phase: str) -> Path:
    return SOURCE_ROOT / "configs" / f"rationale-{phase}.json"


def validate_training(phase: str) -> dict[str, Any]:
    config = FullRationaleConfig.from_json(config_path(phase), require_fresh_output=False)
    path = Path(config.output_dir) / "training_complete.json"
    value = read_json(path)
    expected = rationale_expected_steps(phase)
    need(value.get("status") == "completed" and value.get("phase") == phase, "rationale training identity differs")
    need(value.get("score_fields") == list(AXES) and value.get("average_target_used") is False, "rationale target contract differs")
    need(value.get("global_step") == max(expected.values()), "rationale update count differs")
    for epoch, step in expected.items():
        checkpoint = rationale_checkpoint_dir(Path(config.output_dir), epoch) / "checkpoint_metadata.json"
        metadata = read_json(checkpoint)
        need(metadata.get("global_step") == step and metadata.get("average_target_used") is False, "rationale checkpoint differs")
    return value


def validate_evaluation(phase: str) -> dict[str, Any]:
    path = rationale_eval_dir(phase) / "epoch_metrics.json"
    value = read_json(path)
    expected = rationale_expected_steps(phase)
    need(value.get("status") == "completed" and value.get("phase") == phase, "rationale evaluation identity differs")
    need(value.get("score_fields") == list(AXES) and value.get("average_target_used") is False, "evaluation target contract differs")
    need([row.get("epoch") for row in value["epoch_results"]] == list(expected), "evaluated epochs differ")
    best = value["best_epoch_by_validation_macro_rmse_then_spearman"]
    need(math.isfinite(float(best["metrics"]["three_axis_macro_rmse"])), "rationale best metric is non-finite")
    ledger({"stage": f"validate-{phase}", "event": "aggregate", "best": best, "evaluation_sha256": file_sha(path), "resource_scope": "none"})
    return value


def prepare() -> dict[str, Any]:
    need(PYTHON.is_file() and MODEL_PATH.is_dir(), "recovery environment differs")
    need(not RUN_ROOT.exists() and not FINAL.exists(), "recovery output freshness differs")
    source_manifest = read_json(SOURCE_ROOT / "manifest.json")
    failed_log = SOURCE_ROOT / "logs" / "rationale-eval-gpu0-preflight.log"
    need(source_manifest.get("status") == "failed", "source runtime did not fail")
    need("rationale output freshness differs" in failed_log.read_text(encoding="utf-8"), "source failure differs")
    refit = read_json(FULL_REFIT_METADATA)
    need(refit.get("status") == "completed" and refit.get("model_state_sha256") == file_sha(FULL_FINAL_STATE), "completed refit provenance differs")
    RUN_ROOT.mkdir(parents=True)
    LOG_ROOT.mkdir()
    manifest = {
        "schema_version": "mal2026-qwen3-full-aihub-then-lora-recovery-v1",
        "run_id": RUN_ID,
        "status": "running",
        "started_at": now(),
        "completed_at": None,
        "failed_at": None,
        "failure": None,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_runtime": "20260726-009",
        "source_failure": "evaluation CLI applied training-only output freshness assertion",
        "reused_stages": ["full-aihub-selection", "full-aihub-refit", "rationale-lora-gpu0-preflight"],
        "resource_scope": {"preflight_evaluation": [0], "full_rationale_and_evaluation": [0, 1, 2, 3], "authorization": "default MAL2026 GPU scope"},
        "score_fields": list(AXES),
        "average_target_used": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
    }
    write_new_json(MANIFEST, manifest)
    ledger({"stage": "runner", "event": "start", "git_sha": manifest["git_sha"], "source_runtime": "20260726-009", "resource_scope": "none"})
    return manifest


def main() -> None:
    manifest = prepare()
    try:
        preflight_training = validate_training("gpu0_preflight")
        ledger({"stage": "reuse-gpu0-preflight", "event": "completed", "global_step": preflight_training["global_step"], "resource_scope": "none"})
        wait_idle([0])
        run_stage(
            "rationale-eval-gpu0-preflight",
            [str(PYTHON), "scripts/evaluate_qwen3_full_aihub_rationale_lora_v1.py", "--config", str(config_path("gpu0_preflight")), "--output", str(rationale_eval_dir("gpu0_preflight")), "--essay-limit", "4", "--per-device-batch-size", "4"],
            [0],
        )
        validate_evaluation("gpu0_preflight")
        ledger({"stage": "rationale-eval-gpu0-preflight", "event": "smoke_pass", "decision": "continue to fixed DDP4 rationale arm", "resource_scope": "GPU0"})
        wait_idle([0, 1, 2, 3])
        run_stage(
            "rationale-lora-full",
            [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/train_qwen3_full_aihub_rationale_lora_v1.py", "--config", str(config_path("full"))],
            [0, 1, 2, 3],
        )
        validate_training("full")
        wait_idle([0, 1, 2, 3])
        run_stage(
            "rationale-eval-full",
            [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/evaluate_qwen3_full_aihub_rationale_lora_v1.py", "--config", str(config_path("full")), "--output", str(rationale_eval_dir("full")), "--essay-limit", "400", "--per-device-batch-size", "8"],
            [0, 1, 2, 3],
        )
        evaluation = validate_evaluation("full")
        prior = read_json(PRIOR_FINAL)["ranked_results"][0]
        baseline = read_json(EPOCH_SWEEP_FINAL)["best_epoch_by_validation_macro_rmse_then_spearman"]
        full_best = evaluation["best_epoch_by_validation_macro_rmse_then_spearman"]
        candidates = [
            {"arm": "full_parameter_aihub_then_rationale_lora", **full_best},
            {"arm": prior["arm"], "epoch": prior.get("epoch"), "metrics": prior["metrics"]},
            {"arm": "previous_aihub_lora_r0", **baseline},
        ]
        ranked = sorted(candidates, key=lambda row: (float(row["metrics"]["three_axis_macro_rmse"]), -float(row["metrics"]["three_axis_macro_spearman"]), int(row.get("epoch") or 10**9), row["arm"]))
        refit = read_json(FULL_REFIT_METADATA)
        summary = {
            "schema_version": "mal2026-qwen3-full-aihub-then-lora-final-v1",
            "status": "completed",
            "run_id": RUN_ID,
            "git_sha": manifest["git_sha"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_runtime": "20260726-009",
            "recovery": "evaluation config is loaded with existing training output allowed; no scientific protocol change",
            "refit": {"global_step": refit["trainer_global_step"], "model_state_sha256": refit["model_state_sha256"]},
            "full_parameter_arm_epoch_results": evaluation["epoch_results"],
            "comparison_ranked": ranked,
            "score_fields": list(AXES),
            "average_target_used_in_rationale_stage": False,
            "selection_caveat": "validation was previously exposed; descriptive development evidence only",
            "target_macro_rmse": 0.4213,
            "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
        }
        write_new_json(FINAL, summary)
        rewrite_manifest({**manifest, "status": "completed", "completed_at": now()})
        ledger({"stage": "final-summary", "event": "completed", "best": ranked[0], "evidence_ref": str(FINAL.relative_to(ROOT)), "resource_scope": "none", "decision": "complete"})
    except Exception as exc:
        rewrite_manifest({**manifest, "status": "failed", "failed_at": now(), "failure": {"type": type(exc).__name__, "message": str(exc)}})
        ledger({"stage": "runner", "event": "failed", "failure_type": type(exc).__name__, "failure_message": str(exc), "resource_scope": "none", "decision": "stop and preserve"})
        raise


if __name__ == "__main__":
    main()
