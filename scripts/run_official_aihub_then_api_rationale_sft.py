#!/usr/bin/env python3
"""Durable AI-Hub full SFT -> official API LoRA stage for the selected structure."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-standard/bin/python"
TRAIN_FULL = ROOT / "scripts/train_official_aihub_rationale_sft.py"
TRAIN_CONTINUATION = ROOT / "scripts/train_official_aihub_continuation_sft.py"
FULL_CONFIG = ROOT / "configs/official_aihub_rationale_full_sft_ax4_axis_triplet.full.v1.json"
FSDP_GATE_CONFIG = ROOT / "configs/official_aihub_rationale_full_sft_ax4_axis_triplet.fsdp4_smoke.v1.json"
NUMERIC_GATE_CONFIG = ROOT / "configs/official_aihub_rationale_full_sft_ax4_axis_triplet.fsdp4_numeric_smoke.v1.json"
FP32_NUMERIC_GATE_CONFIG = ROOT / "configs/official_aihub_rationale_full_sft_ax4_axis_triplet.fsdp4_fp32_numeric_smoke.v1.json"
CONTINUATION_SMOKE = ROOT / "configs/official_aihub_then_api_rationale_lora_ax4_axis_triplet_content.gpu0_smoke.v1.json"
CONTINUATION_FULL = {
    "content": ROOT / "configs/official_aihub_then_api_rationale_lora_ax4_axis_triplet_content.full.v1.json",
    "organization": ROOT / "configs/official_aihub_then_api_rationale_lora_ax4_axis_triplet_organization.full.v1.json",
    "expression": ROOT / "configs/official_aihub_then_api_rationale_lora_ax4_axis_triplet_expression.full.v1.json",
}
FULL_OUTPUT = ROOT / "outputs/official-aihub-rationale-full-sft-v1/official-aihub-rationale-full-sft-v1-ax4-axis_triplet-full-002"
SMOKE_OUTPUT = ROOT / "outputs/official-aihub-rationale-full-sft-v1/official-aihub-rationale-full-sft-v1-ax4-axis_triplet-gpu0_smoke-001"
FSDP_GATE_OUTPUT = ROOT / "outputs/official-aihub-rationale-full-sft-v1/official-aihub-rationale-full-sft-v1-ax4-axis_triplet-fsdp4_smoke-001"
NUMERIC_GATE_OUTPUT = ROOT / "outputs/official-aihub-rationale-full-sft-v1/official-aihub-rationale-full-sft-v1-ax4-axis_triplet-fsdp4_numeric_smoke-001"
FP32_NUMERIC_GATE_OUTPUT = ROOT / "outputs/official-aihub-rationale-full-sft-v1/official-aihub-rationale-full-sft-v1-ax4-axis_triplet-fsdp4_fp32_numeric_smoke-001"
CONTINUATION_ROOT = ROOT / "outputs/official-aihub-then-api-rationale-lora-v1"
RUN_ID = "official-aihub-then-api-rationale-sft-v1-20260727-005"
RUN_ROOT = ROOT / "outputs/official-prompt-alignment-v1/aihub-then-api-rationale-pipeline" / RUN_ID
LEDGER = ROOT / "outputs/official-prompt-alignment-v1/20260727-001/ledger.jsonl"


class PipelineError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def ledger(stage: str, event: str, evidence: str, command: Sequence[str] | str, resource: str, *, failure: str = "none", decision: str = "continue", deviation: str = "none") -> None:
    row = {
        "timestamp": now(),
        "run_id": "official-prompt-alignment-v1-20260727-001",
        "stage": stage,
        "event": event,
        "command_ref": list(command) if not isinstance(command, str) else command,
        "resource_scope": resource,
        "gpu_scope_authorization": "repository default GPUs 0-3; user explicitly requested GPUs 0-3",
        "failure_family": failure,
        "repair_iteration": 0,
        "decision": decision,
        "deviation": deviation,
        "evidence_ref": evidence,
    }
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def base_env(cuda_visible_devices: str) -> dict[str, str]:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "PYTHONPATH": str(ROOT / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "1",
        "NCCL_DEBUG": "WARN",
    }


def command_full() -> list[str]:
    return [
        str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4",
        str(TRAIN_FULL), "--config", str(FULL_CONFIG),
    ]


def command_continuation(config: Path) -> list[str]:
    return [str(PYTHON), str(TRAIN_CONTINUATION), "--config", str(config)]


def run_one(stage: str, command: Sequence[str], env: Mapping[str, str], log_name: str, resource: str) -> None:
    log = RUN_ROOT / "logs" / log_name
    need(not log.exists(), f"stage log already exists: {log.name}")
    ledger(stage, "start", str(log.relative_to(ROOT)), command, resource)
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, env=dict(env), stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        ledger(stage, "failed", str(log.relative_to(ROOT)), command, resource, failure="training_process_nonzero", decision="escalate")
        raise PipelineError(f"stage failed ({completed.returncode}): {stage}")


def completion(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    need(path.is_file(), f"completion is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    for key, wanted in expected.items():
        need(value.get(key) == wanted, f"completion field differs: {key}")
    metrics = value.get("train_metrics")
    need(isinstance(metrics, dict) and isinstance(metrics.get("train_loss"), (int, float)), "completion train loss is unavailable")
    return value


def main() -> None:
    need(PYTHON.is_file() and TRAIN_FULL.is_file() and TRAIN_CONTINUATION.is_file(), "existing environment or trainer is unavailable")
    need((SMOKE_OUTPUT / "training_complete.json").is_file(), "GPU0 full-parameter smoke did not pass")
    need((FSDP_GATE_OUTPUT / "training_complete.json").is_file(), "four-GPU FSDP one-update gate did not pass")
    need((FP32_NUMERIC_GATE_OUTPUT / "training_complete.json").is_file(), "four-GPU FSDP float32 numerical gate did not pass")
    need(not RUN_ROOT.exists(), "pipeline run root must be fresh")
    need(not FULL_OUTPUT.exists(), "full AI-Hub output must be fresh")
    for task in CONTINUATION_FULL:
        expected = CONTINUATION_ROOT / f"official-aihub-then-api-rationale-lora-v1-ax4-axis_triplet-{task}-full-001"
        need(not expected.exists(), f"continuation output must be fresh: {task}")
    smoke_continuation = CONTINUATION_ROOT / "official-aihub-then-api-rationale-lora-v1-ax4-axis_triplet-content-gpu0_smoke-001"
    need(not smoke_continuation.exists(), "continuation smoke output must be fresh")
    RUN_ROOT.mkdir(parents=True)
    (RUN_ROOT / "logs").mkdir()
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    inputs = [FULL_CONFIG, FSDP_GATE_CONFIG, NUMERIC_GATE_CONFIG, FP32_NUMERIC_GATE_CONFIG, CONTINUATION_SMOKE, *CONTINUATION_FULL.values(), TRAIN_FULL, TRAIN_CONTINUATION, ROOT / "src/mal2026/official_aihub_rationale_sft.py", ROOT / "src/mal2026/official_aihub_continuation_sft.py", ROOT / "data/manifests/aihub_argumentative_official_rationale_v1.json"]
    manifest: dict[str, Any] = {
        "schema_version": "mal2026-official-aihub-then-api-rationale-pipeline-v1",
        "status": "running",
        "run_id": RUN_ID,
        "created_at": now(),
        "git_sha": git_sha,
        "git_worktree_note": "tracked reproducibility inputs may be uncommitted; exact input hashes are authoritative",
        "seed": {"aihub_full": 2026072702, "api_lora": 2026072701},
        "gpu_scope": "GPUs 0-3; GPU0 continuation smoke; GPUs 0-2 parallel continuation",
        "gpu_scope_authorization": "repository default GPUs 0-3; user explicitly requested GPUs 0-3",
        "inputs": {str(path.relative_to(ROOT)): file_sha(path) for path in inputs},
        "privacy": "aggregate_only_no_rows_prompts_essays_feedback_rationales_ids_scores_or_model_weights_in_manifest",
    }
    atomic_json(RUN_ROOT / "manifest.json", manifest)
    try:
        full_command = command_full()
        run_one("official_aihub_rationale_full_sft", full_command, base_env("0,1,2,3"), "aihub-full-axis-triplet.log", "GPUs 0-3")
        full = completion(FULL_OUTPUT / "training_complete.json", {"status": "completed", "structure": "axis_triplet", "phase": "full", "training_kind": "full_parameter", "train_records": 48030, "world_size": 4})
        ledger("official_aihub_rationale_full_sft", "next_stage_complete", str((FULL_OUTPUT / "training_complete.json").relative_to(ROOT)), full_command, "GPUs 0-3")

        smoke_command = command_continuation(CONTINUATION_SMOKE)
        run_one("official_aihub_then_api_rationale_lora_gpu0_smoke", smoke_command, base_env("0"), "continuation-content-gpu0-smoke.log", "GPU0")
        smoke = completion(smoke_continuation / "training_complete.json", {"status": "completed", "structure": "axis_triplet", "task": "content", "phase": "gpu0_smoke", "training_kind": "lora_after_aihub_full_parameter", "train_records": 1})
        ledger("official_aihub_then_api_rationale_lora_gpu0_smoke", "smoke_pass", str((smoke_continuation / "training_complete.json").relative_to(ROOT)), smoke_command, "GPU0")

        jobs: dict[str, tuple[subprocess.Popen[str], Any, Path, list[str], str]] = {}
        for task, config in CONTINUATION_FULL.items():
            physical_gpu = {"content": "0", "organization": "1", "expression": "2"}[task]
            command = command_continuation(config)
            log = RUN_ROOT / "logs" / f"continuation-{task}-full.log"
            ledger(f"official_aihub_then_api_rationale_lora_{task}", "start", str(log.relative_to(ROOT)), command, f"GPU{physical_gpu}")
            handle = log.open("x", encoding="utf-8")
            process = subprocess.Popen(command, cwd=ROOT, env=base_env(physical_gpu), stdout=handle, stderr=subprocess.STDOUT, text=True)
            jobs[task] = (process, handle, log, command, physical_gpu)
        failed: list[str] = []
        for task, (process, handle, log, command, physical_gpu) in jobs.items():
            code = process.wait()
            handle.close()
            if code:
                failed.append(task)
                ledger(f"official_aihub_then_api_rationale_lora_{task}", "failed", str(log.relative_to(ROOT)), command, f"GPU{physical_gpu}", failure="training_process_nonzero", decision="escalate")
        need(not failed, "continuation full jobs failed: " + ",".join(failed))
        continuations: dict[str, Any] = {}
        for task, config in CONTINUATION_FULL.items():
            output = CONTINUATION_ROOT / f"official-aihub-then-api-rationale-lora-v1-ax4-axis_triplet-{task}-full-001"
            value = completion(output / "training_complete.json", {"status": "completed", "structure": "axis_triplet", "task": task, "phase": "full", "training_kind": "lora_after_aihub_full_parameter", "train_records": 6000})
            continuations[task] = value
            command = command_continuation(config)
            physical_gpu = {"content": "0", "organization": "1", "expression": "2"}[task]
            ledger(f"official_aihub_then_api_rationale_lora_{task}", "next_stage_complete", str((output / "training_complete.json").relative_to(ROOT)), command, f"GPU{physical_gpu}")

        report = {
            "schema_version": "mal2026-official-aihub-then-api-rationale-sft-aggregate-v1",
            "status": "completed",
            "run_id": RUN_ID,
            "completed_at": now(),
            "structure": "axis_triplet",
            "selection_basis": "frozen_macro_then_worst_cell_from_initial_exact_q4_comparison",
            "aihub_full": {"global_step": full["global_step"], "train_records": full["train_records"], "train_loss": full["train_metrics"]["train_loss"], "completion_sha256": file_sha(FULL_OUTPUT / "training_complete.json")},
            "continuation_smoke": {"global_step": smoke["global_step"], "train_loss": smoke["train_metrics"]["train_loss"]},
            "api_lora": {task: {"global_step": value["global_step"], "train_records": value["train_records"], "train_loss": value["train_metrics"]["train_loss"], "completion_sha256": file_sha(CONTINUATION_ROOT / f"official-aihub-then-api-rationale-lora-v1-ax4-axis_triplet-{task}-full-001/training_complete.json")} for task, value in continuations.items()},
            "human_or_reference_validation_score_used": False,
            "privacy": "aggregate_only_no_rows_prompts_essays_feedback_rationales_ids_scores_or_model_weights",
        }
        atomic_json(RUN_ROOT / "aggregate_training_report.json", report)
        manifest.update({"status": "completed", "completed_at": now(), "aggregate_report_sha256": file_sha(RUN_ROOT / "aggregate_training_report.json")})
        atomic_json(RUN_ROOT / "manifest.json", manifest)
        ledger("official_aihub_then_api_rationale_sft_pipeline", "next_stage_complete", str((RUN_ROOT / "aggregate_training_report.json").relative_to(ROOT)), str(Path(__file__).relative_to(ROOT)), "GPUs 0-3")
        print(json.dumps({"status": "completed", "run_id": RUN_ID, "report": str(RUN_ROOT / "aggregate_training_report.json")}, sort_keys=True))
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at": now(), "failure_type": type(exc).__name__, "failure_message": str(exc)})
        atomic_json(RUN_ROOT / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
