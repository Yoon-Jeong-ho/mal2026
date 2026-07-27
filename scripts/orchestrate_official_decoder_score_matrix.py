#!/usr/bin/env python3
"""GPU0-gated, GPU0--3 orchestration for decoder integer-score experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_decoder_score import ARCHITECTURES, DecoderScoreConfig, arm_names, file_sha256, parse_arm
from mal2026.official_decoder_aihub_pretrain import DecoderAIHubConfig
from mal2026.official_score_matrix import select_bootstrap_candidate
from scripts.orchestrate_official_decoder_aihub_score_pretrain import plan as aihub_plan


PYTHON = ROOT / ".venv-standard" / "bin" / "python"
TORCHRUN = ROOT / ".venv-standard" / "bin" / "torchrun"
RUNNER = ROOT / "scripts" / "run_official_decoder_score_matrix.py"
AGGREGATOR = ROOT / "scripts" / "aggregate_official_decoder_score_matrix.py"
GPU_SCOPE = "0,1,2,3"
DEFAULT_AIHUB_CONFIG = ROOT / "configs" / "official_decoder_aihub_integer_score_pretrain.v1.json"


def _command(config: Path, *, architecture: str | None = None, arm: str | None = None, smoke: bool, full: bool) -> list[str]:
    args = [str(RUNNER), "--config", str(config)]
    args += ["--aihub-pretrain", architecture] if architecture else ["--arm", str(arm)]
    if smoke:
        args.append("--smoke")
    return ([str(TORCHRUN), "--standalone", "--nproc_per_node=4"] if full else [str(PYTHON)]) + args


def target_arms(input_view: str) -> list[str]:
    if input_view not in {"all", "essay", "rationale"}:
        raise ValueError("target input view differs")
    return [arm for arm in arm_names() if input_view == "all" or parse_arm(arm)[2] == input_view]


def command_plan(
    config: Path,
    aihub_config_path: Path = DEFAULT_AIHUB_CONFIG,
    target_input_view: str = "all",
) -> list[dict[str, object]]:
    aihub_config = DecoderAIHubConfig.from_json(aihub_config_path, require_dependencies=False)
    plan: list[dict[str, object]] = list(aihub_plan(aihub_config_path, aihub_config))
    resolved = Path("RESOLVED_CONFIG_AFTER_AIHUB_PRETRAIN.json")
    for arm in target_arms(target_input_view):
        plan.append({"stage": "target_smoke", "arm": arm, "gpus": [0], "command": _command(resolved, arm=arm, smoke=True, full=False)})
        plan.append({"stage": "target_full", "arm": arm, "gpus": [0, 1, 2, 3], "command": _command(resolved, arm=arm, smoke=False, full=True)})
    return plan


def _assert_gpu_scope_idle(scope: str) -> None:
    if scope not in {"0", GPU_SCOPE}:
        raise RuntimeError("GPU scope is outside the authorized 0--3 contract")
    result = subprocess.run(
        ["nvidia-smi", "-i", scope, "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("GPUs 0--3 have a pre-existing compute process; refusing to launch or alter it")


def _run(command: list[str], visible: str) -> None:
    _assert_gpu_scope_idle(visible)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": visible, "PYTHONPATH": str(ROOT / "src"), "WANDB_DISABLED": "true"})
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _resolve_config(source: Path, config: DecoderScoreConfig, aihub_config: DecoderAIHubConfig, target_input_view: str = "all") -> Path:
    raw = json.loads(source.read_text(encoding="utf-8"))
    aggregate_path = Path(aihub_config.output_root) / aihub_config.run_id / "aggregate_results.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if aggregate.get("schema_version") != "mal2026-official-decoder-aihub-pretrain-aggregate-v1" or aggregate.get("status") != "completed" or aggregate.get("run_id") != aihub_config.run_id:
        raise RuntimeError("decoder AI-Hub aggregate identity differs")
    results = aggregate.get("results")
    if not isinstance(results, list) or {item.get("architecture") for item in results if isinstance(item, dict)} != set(ARCHITECTURES):
        raise RuntimeError("decoder AI-Hub aggregate architecture coverage differs")
    by_architecture = {item["architecture"]: item for item in results}
    for architecture in ARCHITECTURES:
        result = by_architecture[architecture]
        completion = Path(result["completion_path"])
        metadata = Path(result["state_metadata_path"])
        artifact = Path(result["artifact_path"])
        if file_sha256(completion) != result.get("completion_sha256") or file_sha256(metadata) != result.get("state_metadata_sha256"):
            raise RuntimeError(f"AI-Hub {architecture} aggregate checksums differ")
        payload = json.loads(completion.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or payload.get("architecture") != architecture or payload.get("phase") != "refit" or payload.get("training_method") != "full_parameter":
            raise RuntimeError(f"AI-Hub {architecture} completion is invalid")
        if payload.get("state", {}).get("artifact_sha256") != result.get("artifact_sha256"):
            raise RuntimeError(f"AI-Hub {architecture} artifact checksum binding differs")
        raw["aihub_artifacts"][architecture] = {
            "completion_path": str(completion.resolve()), "completion_sha256": file_sha256(completion),
            "artifact_path": str(artifact.resolve()), "artifact_sha256": result["artifact_sha256"],
            "state_metadata_path": str(metadata.resolve()), "state_metadata_sha256": file_sha256(metadata),
        }
    destination = Path(config.output_root) / f"resolved_config.{target_input_view}.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != raw:
            raise RuntimeError("refusing to replace a different resolved decoder config")
    else:
        destination.write_text(json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resolved = DecoderScoreConfig.from_json(destination, require_dependencies=False)
    for architecture in ARCHITECTURES:
        resolved.validate_warm_artifact(architecture)
    return destination


def _write_essay_bootstrap_aggregate(config: DecoderScoreConfig, arms: list[str]) -> Path:
    candidates = []
    for arm in arms:
        completion = Path(config.output_root) / arm / "training_complete.json"
        payload = json.loads(completion.read_text(encoding="utf-8"))
        selected = payload["selection"]["selected_event"]
        candidates.append({
            "arm": arm,
            **{key: selected[key] for key in ("epoch", "macro_integer_rmse", "macro_integer_spearman", "macro_continuous_rmse")},
            "canonical_validation": payload["canonical_validation"]["metrics"],
            "completion_path": str(completion.resolve()),
            "completion_sha256": file_sha256(completion),
        })
    winner = select_bootstrap_candidate(candidates)
    ranking = sorted(candidates, key=lambda row: (
        row["canonical_validation"]["macro_integer_rmse"],
        -row["canonical_validation"]["macro_integer_spearman"],
        row["canonical_validation"]["macro_continuous_rmse"],
        row["arm"],
    ))
    aggregate = {
        "schema_version": "mal2026-official-decoder-score-essay-bootstrap-aggregate-v1",
        "status": "completed",
        "run_id": config.run_id,
        "score_prompt_kind": config.score_prompt_kind,
        "selection_source": "train_internal_dev_only",
        "canonical_validation_used_for_selection": False,
        "selected_arm": winner["arm"],
        "candidates": candidates,
        "descriptive_canonical_validation_ranking": [row["arm"] for row in ranking],
        "ranking_caveat": "canonical validation is descriptive only and did not choose the bootstrap arm",
    }
    output = Path(config.output_root) / "essay_bootstrap_aggregate.json"
    if output.exists():
        raise RuntimeError("refusing to replace decoder essay bootstrap aggregate")
    output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--aihub-config", type=Path, default=DEFAULT_AIHUB_CONFIG)
    parser.add_argument("--reuse-completed-aihub-pretrain", action="store_true")
    parser.add_argument("--target-input-view", choices=("all", "essay", "rationale"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = DecoderScoreConfig.from_json(args.config, require_dependencies=False)
    aihub_config = DecoderAIHubConfig.from_json(args.aihub_config, require_dependencies=not args.dry_run)
    if args.dry_run:
        stages = command_plan(args.config, args.aihub_config, args.target_input_view)
        if args.reuse_completed_aihub_pretrain:
            stages = [stage for stage in stages if stage.get("stage") not in {"gpu0_one_update_smoke", "fsdp4_one_update_preflight", "fsdp4_full_parameter"}]
        print(json.dumps({"status": "dry_run_passed", "gpu_started": False, "authorized_gpu_scope": [0, 1, 2, 3], "reuse_completed_aihub_pretrain": args.reuse_completed_aihub_pretrain, "target_input_view": args.target_input_view, "stages": stages}, ensure_ascii=False, indent=2))
        return
    manifest_path = Path(config.output_root) / f"orchestration_manifest.{args.target_input_view}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "mal2026-official-decoder-score-orchestration-v1", "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(), "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "gpu_scope": [0, 1, 2, 3], "authorization": "repository default MAL2026 GPU scope", "config_path": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config), "target_input_view": args.target_input_view, "completed_stages": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.reuse_completed_aihub_pretrain:
        aggregate = Path(aihub_config.output_root) / aihub_config.run_id / "aggregate_results.json"
        payload = json.loads(aggregate.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "mal2026-official-decoder-aihub-pretrain-aggregate-v1" or payload.get("status") != "completed":
            raise RuntimeError("reused decoder AI-Hub pretraining aggregate differs")
        manifest["completed_stages"].append("aihub:reused_completed_full_parameter_pretrain")
        manifest["reused_aihub_pretrain_aggregate_sha256"] = file_sha256(aggregate)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    else:
        for stage in aihub_plan(args.aihub_config, aihub_config):
            _run(list(stage["command"]), ",".join(map(str, stage["gpus"])))
            manifest["completed_stages"].append(f"aihub:{stage['architecture']}:{stage['phase']}:{stage['stage']}")
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    resolved = _resolve_config(args.config, config, aihub_config, args.target_input_view)
    manifest["resolved_config_path"] = str(resolved.resolve())
    manifest["resolved_config_sha256"] = file_sha256(resolved)
    selected_target_arms = target_arms(args.target_input_view)
    for arm in selected_target_arms:
        _run(_command(resolved, arm=arm, smoke=True, full=False), "0")
        manifest["completed_stages"].append(f"target_smoke:{arm}")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _run(_command(resolved, arm=arm, smoke=False, full=True), GPU_SCOPE)
        manifest["completed_stages"].append(f"target_full:{arm}")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.target_input_view == "essay":
        aggregate_path = _write_essay_bootstrap_aggregate(config, selected_target_arms)
        manifest["completed_stages"].append("aggregate:decoder_essay_bootstrap")
    else:
        aggregate_command = [str(PYTHON), str(AGGREGATOR), "--config", str(resolved)]
        subprocess.run(aggregate_command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=True)
        aggregate_path = Path(config.output_root) / "aggregate.json"
        manifest["completed_stages"].append("aggregate:decoder_12_arm_ranking")
    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["aggregate_path"] = str(aggregate_path.resolve())
    manifest["aggregate_sha256"] = file_sha256(aggregate_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
