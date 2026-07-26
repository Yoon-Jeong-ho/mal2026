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
    EVAL_ROOT,
    FULL_FINAL_STATE,
    FULL_REFIT_METADATA,
    MODEL_ID,
    MODEL_PATH,
    MODEL_REVISION,
    FullRationaleConfig,
    rationale_checkpoint_dir,
    rationale_dir,
    rationale_expected_steps,
)


SOURCE_ROOT = ROOT / "outputs" / "qwen3-full-aihub-v1" / "20260726-009"
FAILED_RECOVERY_ROOT = ROOT / "outputs" / "qwen3-full-aihub-v1" / "20260726-010"
FAILED_PERSISTENCE_ROOT = ROOT / "outputs" / "qwen3-full-aihub-v1" / "20260726-011"
FAILED_FULL_EVAL_ROOT = ROOT / "outputs" / "qwen3-full-aihub-v1" / "20260726-012"
RUN_ID = "qwen3-full-aihub-then-rationale-lora-v1-recovery-20260726-013"
RUN_ROOT = ROOT / "outputs" / "qwen3-full-aihub-v1" / "20260726-013"
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


def recovery_eval_dir(phase: str) -> Path:
    suffix = "012" if phase == "gpu0_preflight" else "013"
    return EVAL_ROOT / f"qwen3-full-aihub-rationale-lora-eval-v1-{phase}-{suffix}"


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
    path = recovery_eval_dir(phase) / "epoch_metrics.json"
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
    recovery_manifest = read_json(FAILED_RECOVERY_ROOT / "manifest.json")
    persistence_manifest = read_json(FAILED_PERSISTENCE_ROOT / "manifest.json")
    full_eval_manifest = read_json(FAILED_FULL_EVAL_ROOT / "manifest.json")
    failed_log = SOURCE_ROOT / "logs" / "rationale-eval-gpu0-preflight.log"
    recovery_log = FAILED_RECOVERY_ROOT / "logs" / "rationale-eval-gpu0-preflight.log"
    persistence_log = FAILED_PERSISTENCE_ROOT / "logs" / "rationale-eval-gpu0-preflight.log"
    full_eval_log = FAILED_FULL_EVAL_ROOT / "logs" / "rationale-eval-full.log"
    need(source_manifest.get("status") == "failed", "source runtime did not fail")
    need("rationale output freshness differs" in failed_log.read_text(encoding="utf-8"), "source failure differs")
    need(recovery_manifest.get("status") == "failed", "first recovery did not fail")
    need("Spearman is undefined for constant ranks" in recovery_log.read_text(encoding="utf-8"), "first recovery failure differs")
    need(persistence_manifest.get("status") == "failed", "second recovery did not fail")
    need("evaluation persistence failed" in persistence_log.read_text(encoding="utf-8"), "second recovery failure differs")
    need(full_eval_manifest.get("status") == "failed", "third recovery did not fail")
    need("Spearman is undefined for constant ranks" in full_eval_log.read_text(encoding="utf-8"), "third recovery failure differs")
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
        "source_failures": ["evaluation CLI applied training-only output freshness assertion", "four-row finite-output smoke had undefined Spearman from constant prediction ranks", "Trainer created the fresh evaluation directory before aggregate persistence", "a full-arm checkpoint had mathematically undefined Spearman from constant prediction ranks"],
        "reused_stages": ["full-aihub-selection", "full-aihub-refit", "rationale-lora-gpu0-preflight", "rationale-eval-gpu0-preflight", "rationale-lora-full"],
        "resource_scope": {"full_evaluation": [0, 1, 2, 3], "authorization": "default MAL2026 GPU scope"},
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
        validate_evaluation("gpu0_preflight")
        full_training = validate_training("full")
        ledger({"stage": "reuse-rationale-lora-full", "event": "completed", "global_step": full_training["global_step"], "resource_scope": "none"})
        wait_idle([0, 1, 2, 3])
        run_stage(
            "rationale-eval-full",
            [str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/evaluate_qwen3_full_aihub_rationale_lora_v1.py", "--config", str(config_path("full")), "--output", str(recovery_eval_dir("full")), "--essay-limit", "400", "--per-device-batch-size", "8"],
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
        def comparison_key(row: Mapping[str, Any]) -> tuple[float, float, int, str]:
            correlation = row["metrics"]["three_axis_macro_spearman"]
            return (float(row["metrics"]["three_axis_macro_rmse"]), -float(correlation) if correlation is not None else math.inf, int(row.get("epoch") or 10**9), str(row["arm"]))
        ranked = sorted(candidates, key=comparison_key)
        refit = read_json(FULL_REFIT_METADATA)
        summary = {
            "schema_version": "mal2026-qwen3-full-aihub-then-lora-final-v1",
            "status": "completed",
            "run_id": RUN_ID,
            "git_sha": manifest["git_sha"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_runtime": "20260726-009",
            "recovery": "completed training is reused; mathematically undefined Spearman is recorded as null and never wins a tie; RMSE and the full evaluation data are unchanged",
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
