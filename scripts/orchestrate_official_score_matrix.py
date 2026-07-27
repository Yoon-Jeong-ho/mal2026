#!/usr/bin/env python3
"""GPU0-smoke then DDP4 orchestration for the fixed eight-arm score matrix."""
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

from mal2026.official_score_matrix import MatrixConfig, arm_names, file_sha256, parse_arm, select_bootstrap_candidate  # noqa: E402
from mal2026.official_score_prompt import provenance as score_prompt_provenance  # noqa: E402


def command(config: Path, arm: str, smoke: bool) -> list[str]:
    python = str(ROOT / ".venv-standard" / "bin" / "python")
    base = [python, "scripts/run_official_score_matrix.py", "--config", str(config), "--arm", arm]
    return base + (["--smoke"] if smoke else [])


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def gpu_processes(gpus: list[int]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", f"--id={','.join(map(str, gpus))}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def write_new_json(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to replace {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("bootstrap", "rationale", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = MatrixConfig.from_json(args.config, require_dependencies=False)
    if not args.dry_run:
        if args.stage == "all":
            raise RuntimeError("stage A and B require external rationale generation; run --stage bootstrap, bind its SHA/rationales, then --stage rationale")
        config.validate_dependencies(args.stage)
    selected_arms = [arm for arm in arm_names() if parse_arm(arm)[2] == ("essay" if args.stage == "bootstrap" else "rationale")]
    if args.stage == "all":
        selected_arms = list(arm_names())
    plan = []
    for arm in selected_arms:
        plan.append({"arm": arm, "stage": "gpu0_smoke", "gpus": [0], "command": command(args.config, arm, True)})
        plan.append({"arm": arm, "stage": "ddp4_full", "gpus": [0, 1, 2, 3], "command": [str(ROOT / ".venv-standard" / "bin" / "python"), "-m", "torch.distributed.run", "--nproc_per_node=4", *command(args.config, arm, False)[1:]]})
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "stage": args.stage, "gpu_started": False, "arm_count": len(selected_arms), "plan": plan, "external_handoff_required_between_stages": True}, indent=2, sort_keys=True))
        return
    output_root = Path(config.output_root)
    bootstrap_path = output_root / "bootstrap_selection.json"
    manifest_path = output_root / "manifest.json"
    if args.stage == "bootstrap":
        if output_root.exists():
            raise RuntimeError(f"refusing to reuse matrix output root: {output_root}")
        output_root.mkdir(parents=True)
    else:
        if not output_root.is_dir() or not bootstrap_path.is_file() or file_sha256(bootstrap_path) != config.bootstrap_selection_sha256:
            raise RuntimeError("stage B bootstrap selection binding is unavailable")
    logs = output_root / "logs"
    logs.mkdir(exist_ok=args.stage == "rationale")
    ledger = output_root / "ledger.jsonl"
    if args.stage == "bootstrap":
        manifest = {
            "schema_version": "mal2026-official-score-matrix-run-v1", "status": "stage_a_running", "run_id": config.run_id,
            "started_at": now(), "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "resource_scope": {"preflight": [0], "full": [0, 1, 2, 3], "authorization": "default MAL2026 GPU scope"},
            "arm_count": 8, "stage_a_arm_count": 4, "score_fields": ["content", "organization", "expression"], "average_target_used": False,
            **score_prompt_provenance(config.score_prompt_kind),
            "aihub_full_pretrain": {
                "bounded_completion_sha256": config.aihub_bounded_completion_sha256, "bounded_artifact_sha256": config.aihub_bounded_artifact_sha256,
                "ordinal_completion_sha256": config.aihub_ordinal_completion_sha256, "ordinal_artifact_sha256": config.aihub_ordinal_artifact_sha256,
                "load_semantics": config.aihub_warmstart_load_mode,
            },
            "selection_source": "deterministic train-internal 80/20 only; integer RMSE then integer Spearman then continuous RMSE then earlier epoch", "canonical_validation_use": "single final descriptive evaluation per arm",
        }
        write_new_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "stage_a_completed":
            raise RuntimeError("stage A manifest is not complete")
        manifest.update({"status": "stage_b_running", "stage_b_started_at": now()})
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["TOKENIZERS_PARALLELISM"] = "false"
    # The orchestrator just verified the large warmstate once.  Propagate that
    # exact digest so four DDP ranks do not concurrently reread the same file.
    env["MAL2026_VERIFIED_AIHUB_BOUNDED_REGRESSION_SHA256"] = config.aihub_bounded_artifact_sha256
    env["MAL2026_VERIFIED_AIHUB_ORDINAL_CUMULATIVE_SHA256"] = config.aihub_ordinal_artifact_sha256
    for stage in plan:
        processes = gpu_processes(stage["gpus"])
        if processes:
            raise RuntimeError(f"GPU boundary conflict on {stage['gpus']}; existing processes were not altered: {processes}")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, stage["gpus"]))
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": now(), "event": "start", **stage}, sort_keys=True) + "\n")
        log = logs / f"{stage['stage']}--{stage['arm']}.log"
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(stage["command"], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": now(), "event": "completed" if completed.returncode == 0 else "failed", "exit_code": completed.returncode, "log": str(log), **stage}, sort_keys=True) + "\n")
        if completed.returncode:
            raise RuntimeError(f"matrix stage failed: {stage['stage']} / {stage['arm']}")
    if args.stage == "bootstrap":
        candidates = []
        for arm in selected_arms:
            path = output_root / arm / "result.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            selected_event = value["selection"]["selected_event"]
            candidates.append({"arm": arm, **{key: selected_event[key] for key in ("epoch", "macro_integer_rmse", "macro_integer_spearman", "macro_continuous_rmse")}, "result_path": str(path.resolve()), "result_sha256": file_sha256(path), "score_files": value["emitted_integer_score_files"]})
        winner = select_bootstrap_candidate(candidates)
        bootstrap = {
            "schema_version": "mal2026-official-score-bootstrap-selection-v1", "status": "stage_a_completed", "run_id": config.run_id,
            "selection_source": "train_internal_dev_only", "canonical_validation_used_for_selection": False,
            "selection_rule": "lowest macro integer RMSE, then highest macro integer Spearman, then lowest continuous RMSE, then arm name",
            "candidates": candidates, "selected_arm": winner["arm"], "selected_result_path": winner["result_path"], "selected_result_sha256": winner["result_sha256"],
            "selected_score_files": winner["score_files"], "average_target_used": False,
            **score_prompt_provenance(config.score_prompt_kind),
        }
        write_new_json(bootstrap_path, bootstrap)
        manifest.update({"status": "stage_a_completed", "stage_a_completed_at": now(), "bootstrap_selection_sha256": file_sha256(bootstrap_path), "external_handoff": "generate final restricted train/validation rationales from selected emitted integer score files, then bind all SHAs in config"})
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
        return

    results = []
    for arm in arm_names():
        path = output_root / arm / "result.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        results.append({"arm": arm, "selected_epoch": value["selection"]["selected_epoch"], "canonical_validation": value["canonical_validation"]["metrics"], "result_sha256": file_sha256(path)})
    ranking = sorted(results, key=lambda row: (
        row["canonical_validation"]["macro_integer_rmse"],
        -row["canonical_validation"]["macro_integer_spearman"],
        row["canonical_validation"]["macro_continuous_rmse"],
        row["arm"],
    ))
    summary = {
        "schema_version": "mal2026-official-score-matrix-aggregate-v1", "status": "completed", "run_id": config.run_id,
        "score_fields": ["content", "organization", "expression"], "average_target_used": False, "arms": results,
        **score_prompt_provenance(config.score_prompt_kind),
        "descriptive_validation_ranking": [row["arm"] for row in ranking],
        "descriptive_validation_ranking_rule": "lowest macro integer RMSE, then highest macro integer Spearman, then lowest macro continuous RMSE, then arm name",
        "ranking_caveat": "canonical validation is descriptive and was not used for epoch selection or refit",
    }
    write_new_json(output_root / "aggregate_results.json", summary)
    manifest.update({"status": "completed", "completed_at": now(), "aggregate_sha256": file_sha256(output_root / "aggregate_results.json"), "bootstrap_selection_sha256": config.bootstrap_selection_sha256, "rationale_manifest_sha256": config.rationale_manifest_sha256})
    temporary = output_root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_root / "manifest.json")


if __name__ == "__main__":
    main()
