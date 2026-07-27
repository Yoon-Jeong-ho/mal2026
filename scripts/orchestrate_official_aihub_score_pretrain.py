#!/usr/bin/env python3
"""GPU0 one-update gates followed by FSDP4 full-backbone selection/refit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_aihub_score_pretrain import (  # noqa: E402
    HEADS, PretrainConfig, downstream_target_contract, file_sha256,
)
from mal2026.official_score_prompt import provenance as score_prompt_provenance  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _python() -> str:
    return str(ROOT / ".venv-standard" / "bin" / "python")


def _train_command(config_path: Path, head: str, phase: str, *, smoke: bool, selection_metadata: Path | None = None) -> list[str]:
    command = [_python(), "scripts/train_official_aihub_score_pretrain.py", "--config", str(config_path), "--head", head, "--phase", phase]
    if selection_metadata is not None:
        command += ["--selection-metadata", str(selection_metadata)]
    if smoke:
        command.append("--smoke")
    return command


def plan(config_path: Path, config: PretrainConfig) -> list[dict[str, object]]:
    root = Path(config.output_root) / config.run_id
    stages: list[dict[str, object]] = []
    for head in HEADS:
        selection = root / f"{head}-selection" / "training_complete.json"
        smoke_selection = root / f"smoke-{head}-selection" / "training_complete.json"
        stages.extend([
            {"head": head, "phase": "selection", "stage": "gpu0_one_update_smoke", "gpus": [0], "command": _train_command(config_path, head, "selection", smoke=True)},
            {"head": head, "phase": "refit", "stage": "gpu0_one_update_smoke", "gpus": [0], "command": _train_command(config_path, head, "refit", smoke=True, selection_metadata=smoke_selection)},
            {"head": head, "phase": "selection", "stage": "fsdp4_full_parameter", "gpus": [0, 1, 2, 3], "command": [_python(), "-m", "torch.distributed.run", "--nproc_per_node=4", *_train_command(config_path, head, "selection", smoke=False)[1:]]},
            {"head": head, "phase": "refit", "stage": "fsdp4_full_parameter", "gpus": [0, 1, 2, 3], "command": [_python(), "-m", "torch.distributed.run", "--nproc_per_node=4", *_train_command(config_path, head, "refit", smoke=False, selection_metadata=selection)[1:]]},
        ])
    return stages


def _gpu_processes(gpus: list[int]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", f"--id={','.join(map(str, gpus))}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = PretrainConfig.from_json(args.config, require_dependencies=not args.dry_run)
    stages = plan(args.config, config)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_passed", "gpu_started": False,
            "gpu_scope": {"smoke": [0], "full": [0, 1, 2, 3]},
            "heads": list(HEADS), "average_read": False,
            "training_method": "full_parameter",
            "distributed_strategy": "fsdp_full_shard_auto_wrap",
            "fsdp_version": config.fsdp_version,
            **downstream_target_contract(config),
            **score_prompt_provenance(config.score_prompt_kind),
            "canonical_validation_access": False, "plan": stages,
        }, indent=2, sort_keys=True))
        return

    run_root = Path(config.output_root) / config.run_id
    if run_root.exists():
        raise RuntimeError(f"refusing to reuse run root: {run_root}")
    run_root.mkdir(parents=True)
    logs = run_root / "logs"
    logs.mkdir()
    ledger = run_root / "ledger.jsonl"
    manifest_path = run_root / "manifest.json"
    manifest = {
        "schema_version": "mal2026-aihub-integer-score-pretrain-run-v2",
        "status": "running", "run_id": config.run_id, "started_at": now(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "gpu_scope": {"smoke": [0], "full": [0, 1, 2, 3], "authorization": "default MAL2026 GPU scope"},
        "heads": list(config.heads), **downstream_target_contract(config),
        **score_prompt_provenance(config.score_prompt_kind),
        "training_method": "full_parameter",
        "distributed_strategy": config.distributed_strategy,
        "fsdp_version": config.fsdp_version,
        "average_read": False,
        "canonical_validation_access": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["TOKENIZERS_PARALLELISM"] = "false"
    for index, stage in enumerate(stages):
        gpus = list(stage["gpus"])
        processes = _gpu_processes(gpus)
        if processes:
            raise RuntimeError(f"GPU boundary conflict on {gpus}; no process was altered: {processes}")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        event = {"timestamp": now(), "event": "start", **stage}
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        log = logs / f"{index:02d}-{stage['head']}-{stage['phase']}-{stage['stage']}.log"
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(stage["command"], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": now(), "event": "completed" if completed.returncode == 0 else "failed", "exit_code": completed.returncode, "log": str(log), **stage}, sort_keys=True) + "\n")
        if completed.returncode:
            raise RuntimeError(f"stage failed: {stage['head']} {stage['phase']} {stage['stage']}")

    results = []
    for head in HEADS:
        completion = run_root / f"{head}-refit" / "training_complete.json"
        payload = json.loads(completion.read_text(encoding="utf-8"))
        results.append({
            "head": head,
            "selected_global_step": payload["selection"]["selected_event"]["global_step"],
            "selection_metrics": {key: payload["selection"]["selected_event"][key] for key in ("macro_integer_rmse", "macro_integer_spearman", "macro_continuous_rmse")},
            "artifact_path": payload["state"]["artifact_path"], "artifact_sha256": payload["state"]["artifact_sha256"],
            "backbone_tensor_count": payload["state"]["backbone_tensor_count"],
            "score_head_state_sha256": payload["state"]["score_head_state_sha256"],
            "state_metadata_path": payload["state"]["metadata_path"],
            "state_metadata_sha256": payload["state"]["metadata_sha256"],
            "completion_path": str(completion.resolve()),
            "completion_sha256": file_sha256(completion),
        })
    aggregate = {
        "schema_version": "mal2026-aihub-integer-score-pretrain-aggregate-v2",
        "status": "completed", "run_id": config.run_id,
        **downstream_target_contract(config), **score_prompt_provenance(config.score_prompt_kind), "average_read": False,
        "training_method": "full_parameter",
        "downstream_adaptation": "fresh_MAL2026_LoRA_on_selected_full_AIHub_backbone_with_matched_head_retained",
        "selection_source": "AI-Hub selection_dev only", "refit_records": 48016,
        "canonical_validation_access": False, "results": results,
        "privacy": "aggregate_only_no_rows_prompts_essays_feedback_ids_predictions_or_model_outputs_persisted",
    }
    aggregate_path = run_root / "aggregate_results.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    manifest.update({"status": "completed", "completed_at": now(), "aggregate_sha256": file_sha256(aggregate_path)})
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)


if __name__ == "__main__":
    main()
