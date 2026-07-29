#!/usr/bin/env python3
"""Durably run Solar augmentation through the final score encoder.

This runner never pulls an image or installs a package. The official Solar
image must already be local; each successful smoke continues immediately to
the corresponding full four-GPU stage.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_rl_servers import assert_gpus_idle  # noqa: E402


RUN_ID = "remaining-solar-encoder-pipeline-v1-20260729-001"
OUTPUT = ROOT / "outputs/remaining-solar-encoder-pipeline-v1" / RUN_ID
PYTHON = ROOT / ".venv-standard/bin/python"
TORCHRUN = ROOT / ".venv-standard/bin/torchrun"


class RemainingPipelineError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RemainingPipelineError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Phase:
    name: str
    command: tuple[str, ...]
    result: Path
    gpus: tuple[int, ...]


def phases() -> tuple[Phase, ...]:
    qwen_config = ROOT / "configs/augmented_rationale_aware_qwen3_embedding_8b.v1.json"
    kure_config = ROOT / "configs/augmented_rationale_aware_kure_v1.v1.json"
    final_config = ROOT / "configs/final_rationale_aware_score_encoder.v1.json"
    qwen_run = "augmented-rationale-aware-qwen3-embedding-8b-v1-20260729-001"
    kure_run = "augmented-rationale-aware-kure-v1-v1-20260729-001"
    final_run = "final-rationale-aware-score-encoder-v1-20260729-001"
    return (
        Phase(
            "solar_augmentation",
            (str(PYTHON), str(ROOT / "scripts/run_solar_axis_augmentation.py")),
            ROOT / "outputs/solar-axis-degradation-v1/solar-open2-axis-degradation-train-v1-20260729-004/result.json",
            (0, 1, 2, 3),
        ),
        Phase(
            "augmented_bundle_rationales",
            (str(PYTHON), str(ROOT / "scripts/run_solar_augmented_bundle_rationales.py")),
            ROOT / "outputs/solar-augmented-bundle-rationale-v1/official-dpo-bundle-solar-augmented-rationales-v1-20260729-001/result.json",
            (0, 1, 2, 3),
        ),
        Phase(
            "qwen_augmented_smoke",
            (str(PYTHON), str(ROOT / "scripts/run_augmented_rationale_encoder.py"), "--config", str(qwen_config), "--mode", "smoke"),
            ROOT / "outputs/augmented-rationale-aware-encoder-v1" / f"smoke-{qwen_run}/smoke_complete.json",
            (0,),
        ),
        Phase(
            "qwen_augmented_full",
            (str(TORCHRUN), "--standalone", "--nproc_per_node=4", str(ROOT / "scripts/run_augmented_rationale_encoder.py"), "--config", str(qwen_config), "--mode", "full"),
            ROOT / "outputs/augmented-rationale-aware-encoder-v1" / f"{qwen_run}/result.json",
            (0, 1, 2, 3),
        ),
        Phase(
            "kure_augmented_smoke",
            (str(PYTHON), str(ROOT / "scripts/run_augmented_rationale_encoder.py"), "--config", str(kure_config), "--mode", "smoke"),
            ROOT / "outputs/augmented-rationale-aware-encoder-v1" / f"smoke-{kure_run}/smoke_complete.json",
            (0,),
        ),
        Phase(
            "kure_augmented_full",
            (str(TORCHRUN), "--standalone", "--nproc_per_node=4", str(ROOT / "scripts/run_augmented_rationale_encoder.py"), "--config", str(kure_config), "--mode", "full"),
            ROOT / "outputs/augmented-rationale-aware-encoder-v1" / f"{kure_run}/result.json",
            (0, 1, 2, 3),
        ),
        Phase(
            "final_winner_smoke",
            (str(PYTHON), str(ROOT / "scripts/run_final_rationale_encoder.py"), "--config", str(final_config), "--mode", "smoke"),
            ROOT / "outputs/final-rationale-aware-score-encoder-v1" / f"smoke-{final_run}/smoke_complete.json",
            (0,),
        ),
        Phase(
            "final_winner_full",
            (str(TORCHRUN), "--standalone", "--nproc_per_node=4", str(ROOT / "scripts/run_final_rationale_encoder.py"), "--config", str(final_config), "--mode", "full"),
            ROOT / "outputs/final-rationale-aware-score-encoder-v1" / f"{final_run}/result.json",
            (0, 1, 2, 3),
        ),
    )


def completed_result(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(result, dict) and result.get("status") == "completed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    need(args.run_id == RUN_ID, "remaining pipeline run identity differs")
    plan = phases()
    if args.dry_run:
        print(json.dumps({
            "run_id": RUN_ID,
            "implicit_docker_pull": False,
            "phases": [{"name": phase.name, "gpus": list(phase.gpus), "command": list(phase.command), "result": str(phase.result)} for phase in plan],
        }, indent=2, sort_keys=True))
        return

    OUTPUT.mkdir(mode=0o700, parents=True, exist_ok=True)
    logs = OUTPUT / "logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    manifest_path = OUTPUT / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        need(manifest.get("run_id") == RUN_ID, "remaining pipeline manifest differs")
    else:
        manifest = {
            "schema_version": "mal2026-remaining-solar-encoder-pipeline-v1",
            "status": "running", "run_id": RUN_ID, "created_at": now(),
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "gpu_scope": [0, 1, 2, 3],
            "gpu_authorization": "repository default GPUs0-3 and explicit user request; GPUs4-7 not queried or used",
            "implicit_docker_pull": False, "phases": {},
        }
        atomic_json(manifest_path, manifest)

    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    for phase in plan:
        if completed_result(phase.result):
            manifest["phases"][phase.name] = {
                **manifest["phases"].get(phase.name, {}),
                "status": "completed", "result": str(phase.result.resolve()), "skipped_existing": True,
            }
            atomic_json(manifest_path, manifest)
            continue
        state = manifest["phases"].get(phase.name, {})
        attempt = int(state.get("attempts", 0)) + 1
        log = logs / f"{phase.name}-attempt-{attempt:03d}.log"
        need(not log.exists(), "remaining pipeline log must be fresh")
        assert_gpus_idle(phase.gpus)
        visible = ",".join(str(gpu) for gpu in phase.gpus)
        phase_env = {**environment, "CUDA_VISIBLE_DEVICES": visible, "MAL2026_RESERVED_PHYSICAL_GPUS": visible}
        manifest["phases"][phase.name] = {
            "status": "running", "attempts": attempt, "started_at": now(),
            "gpus": list(phase.gpus), "command": list(phase.command), "log": str(log.resolve()),
        }
        atomic_json(manifest_path, manifest)
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(phase.command, cwd=ROOT, env=phase_env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        if completed.returncode != 0 or not completed_result(phase.result):
            manifest["status"] = "failed"
            manifest["failed_at"] = now()
            manifest["phases"][phase.name].update({"status": "failed", "returncode": completed.returncode, "failed_at": now()})
            atomic_json(manifest_path, manifest)
            raise RemainingPipelineError(f"phase failed: {phase.name}; log={log}")
        manifest["phases"][phase.name].update({
            "status": "completed", "returncode": 0, "completed_at": now(), "result": str(phase.result.resolve()),
        })
        manifest["status"] = "running"
        manifest.pop("failed_at", None)
        atomic_json(manifest_path, manifest)
    manifest.update({"status": "completed", "completed_at": now()})
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "completed", "run_id": RUN_ID, "manifest": str(manifest_path.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
