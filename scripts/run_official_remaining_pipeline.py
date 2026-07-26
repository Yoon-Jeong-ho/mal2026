#!/usr/bin/env python3
"""Resume-safe outer runner for the remaining rationale/score experiment.

This runner only sequences already-declared producers.  Each producer keeps
its own GPU0 smoke gate, GPU 0--3 scope, immutable artifacts, and fail-closed
validation.  A completed stage is skipped only after its authoritative
completion artifact validates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_decoder_score import DecoderScoreConfig  # noqa: E402
from mal2026.official_rationale_candidate_evaluation import resolve_handoff  # noqa: E402
from mal2026.official_rationale_handoff import AXES, HandoffConfig  # noqa: E402
from mal2026.official_remaining_pipeline import (  # noqa: E402
    EMBEDDING_PRETRAIN_ROOT, RATIONALE_RESTRICTED_ROOT, RL_ROOT,
    build_candidate_bindings, file_sha256, read_json,
    resolve_decoder_score_config, resolve_embedding_score_config,
)
from mal2026.official_score_matrix import MatrixConfig  # noqa: E402


PYTHON = ROOT / ".venv-standard/bin/python"
RUN_ID = "official-remaining-pipeline-v1-20260727-001"
RUN_ROOT = ROOT / "outputs/official-remaining-pipeline-v1" / RUN_ID
CONFIG_ROOT = ROOT / "outputs/official-runtime-configs-v1" / RUN_ID
EMBED_PRETRAIN = ROOT / "configs/official_aihub_integer_score_pretrain.repair1.v1.json"
EMBED_TEMPLATE = ROOT / "configs/official_score_matrix.v1.json"
HANDOFF_TEMPLATE = ROOT / "configs/official_rationale_handoff.v1.json"
DECODER_TEMPLATE = ROOT / "configs/official_decoder_score_matrix.v1.json"
DECODER_PRETRAIN = ROOT / "configs/official_decoder_aihub_integer_score_pretrain.v1.json"
DECODER_PRETRAIN_ROOT = (
    ROOT / "outputs/official-decoder-aihub-integer-score-full-pretrain-v1"
    / "official-decoder-aihub-integer-score-full-pretrain-v1-20260727-001"
)
RL_SAFETY_GATE = (
    ROOT / "outputs/official-prompt-alignment-v1/judge-prompt-injection"
    / "official-judge-prompt-injection-train32-001/aggregate_rl_safety_gate.json"
)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_bound_config(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"resolved config differs from existing artifact: {path}")
        return
    path.write_text(encoded, encoding="utf-8")


def command(argv: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise RuntimeError(f"stage log already exists without a completed stage report: {log}")
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(
            argv, cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "TOKENIZERS_PARALLELISM": "false", "WANDB_DISABLED": "true"},
            stdout=handle, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"stage command failed with {result.returncode}: {' '.join(argv)}")


class Runner:
    def __init__(self) -> None:
        self.manifest = RUN_ROOT / "manifest.json"
        self.ledger = RUN_ROOT / "ledger.jsonl"
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if self.manifest.is_file():
            value = read_json(self.manifest, "remaining-pipeline manifest")
            if value.get("schema_version") != "mal2026-official-remaining-pipeline-v1" or value.get("run_id") != RUN_ID:
                raise RuntimeError("remaining-pipeline manifest differs")
            resumptions = value.setdefault("resumptions", [])
            if not isinstance(resumptions, list):
                raise RuntimeError("remaining-pipeline resumptions differ")
            resumptions.append({"resumed_at": now(), "git_sha": git_sha})
            value["status"] = "running"
            atomic_json(self.manifest, value)
        else:
            atomic_json(self.manifest, {
                "schema_version": "mal2026-official-remaining-pipeline-v1", "status": "running",
                "run_id": RUN_ID, "started_at": now(),
                "git_sha": git_sha, "resumptions": [],
                "gpu_scope": [0, 1, 2, 3], "gpu_authorization": "default MAL2026 scope and explicit user request",
                "scientific_protocol": "declared public-spec rationale RL then fixed embedding/decoder score matrices",
            })

    def event(self, stage: str, event: str, evidence: Mapping[str, Any]) -> None:
        with self.ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": now(), "stage": stage, "event": event, "evidence": dict(evidence)}, sort_keys=True) + "\n")

    def stage(self, name: str, validate: Callable[[], Mapping[str, Any]], execute: Callable[[], None]) -> Mapping[str, Any]:
        report = RUN_ROOT / "stages" / f"{name}.json"
        if report.is_file():
            value = read_json(report, f"stage {name}")
            if value.get("status") != "completed":
                raise RuntimeError(f"stage report is not completed: {name}")
            evidence = dict(validate())
            if evidence != value.get("evidence"):
                raise RuntimeError(f"completed stage evidence changed: {name}")
            return evidence
        self.event(name, "started", {})
        execute()
        evidence = dict(validate())
        atomic_json(report, {"schema_version": "mal2026-official-remaining-stage-v1", "status": "completed", "stage": name, "completed_at": now(), "evidence": evidence})
        self.event(name, "completed", {"report_sha256": file_sha256(report)})
        return evidence

    def finish(self) -> None:
        value = read_json(self.manifest, "remaining-pipeline manifest")
        value.update({"status": "completed", "completed_at": now()})
        atomic_json(self.manifest, value)


def completed_json(path: Path, schema: str | None = None) -> dict[str, Any]:
    value = read_json(path, str(path))
    if value.get("status") != "completed" or (schema is not None and value.get("schema_version") != schema):
        raise RuntimeError(f"completion artifact differs: {path}")
    return value


def wait_for_rl() -> None:
    manifest = RL_ROOT / "manifest.json"
    while True:
        if RL_SAFETY_GATE.is_file():
            gate = read_json(RL_SAFETY_GATE, "RL safety gate")
            if gate.get("status") == "failed_gates" or gate.get("rl_allowed") is False:
                raise RuntimeError("fixed proxy-judge safety gate failed; RL remains fail-closed")
        if not manifest.is_file():
            time.sleep(30)
            continue
        value = read_json(manifest, "RL manifest")
        status = value.get("status")
        if status == "completed":
            # The RL runner atomically marks its manifest immediately before
            # writing the aggregate.  Wait for both artifacts so the outer
            # stage cannot observe that short publication interval.
            if (RL_ROOT / "aggregate_experiment.json").is_file():
                return
            time.sleep(2)
            continue
        if status == "failed":
            raise RuntimeError("RL did not complete successfully: failed")
        if status != "running":
            raise RuntimeError(f"RL manifest status differs: {status}")
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = [
        "embedding_aihub_full_pretrain", "embedding_score_bootstrap_4_arms", "decoder_aihub_full_pretrain",
        "wait_for_completed_dpo_grpo",
        "resolve_9_rationale_candidates", "candidate_generation_and_fixed_q4_repeated_evaluation",
        "select_winner_and_generate_train_validation_rationales", "embedding_rationale_4_arms",
        "decoder_aihub_full_pretrain_and_12_arm_matrix", "final_aggregate",
    ]
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "gpu_started": False, "gpu_scope": [0, 1, 2, 3], "plan": plan}, indent=2))
        return

    runner = Runner()
    # These score-only stages are independent of the rationale winner and RL.
    # Run them first so a fail-closed judge gate does not leave authorized
    # full-parameter pretraining or the essay-only bootstrap idle.
    runner.stage(
        "embedding_aihub_full_pretrain",
        lambda: {"aggregate_sha256": file_sha256(EMBEDDING_PRETRAIN_ROOT / "aggregate_results.json")},
        lambda: command([str(PYTHON), "scripts/orchestrate_official_aihub_score_pretrain.py", "--config", str(EMBED_PRETRAIN)], RUN_ROOT / "logs/embedding-aihub-full-pretrain-repair1.log"),
    )

    embed_template = read_json(EMBED_TEMPLATE, "embedding score template")
    bootstrap_config = CONFIG_ROOT / "official_score_matrix.bootstrap.resolved.json"
    resolved_bootstrap = resolve_embedding_score_config(embed_template, include_rationales=False)
    write_bound_config(bootstrap_config, resolved_bootstrap)
    MatrixConfig.from_json(bootstrap_config).validate_dependencies("bootstrap")
    score_root = ROOT / "outputs/official-score-matrix-v1"
    runner.stage(
        "embedding_score_bootstrap",
        lambda: {"bootstrap_sha256": file_sha256(score_root / "bootstrap_selection.json")},
        lambda: command([str(PYTHON), "scripts/orchestrate_official_score_matrix.py", "--config", str(bootstrap_config), "--stage", "bootstrap"], RUN_ROOT / "logs/embedding-score-bootstrap.log"),
    )
    runner.stage(
        "decoder_aihub_full_pretrain",
        lambda: {"aggregate_sha256": file_sha256(DECODER_PRETRAIN_ROOT / "aggregate_results.json")},
        lambda: command([str(PYTHON), "scripts/orchestrate_official_decoder_aihub_score_pretrain.py", "--config", str(DECODER_PRETRAIN)], RUN_ROOT / "logs/decoder-aihub-full-pretrain.log"),
    )
    runner.stage(
        "rl_complete",
        lambda: {"manifest_sha256": file_sha256(RL_ROOT / "manifest.json"), "aggregate_sha256": file_sha256(RL_ROOT / "aggregate_experiment.json")},
        wait_for_rl,
    )

    handoff_template = read_json(HANDOFF_TEMPLATE, "rationale handoff template")
    bindings_path = CONFIG_ROOT / "official_rationale_candidate_bindings.resolved.json"
    bindings = build_candidate_bindings(handoff_template)
    write_bound_config(bindings_path, bindings)
    pending_handoff_path = CONFIG_ROOT / "official_rationale_handoff.pending_evaluations.json"
    pending = resolve_handoff(handoff_template, bindings, require_evaluations=False)
    write_bound_config(pending_handoff_path, pending)
    HandoffConfig.from_json(pending_handoff_path)
    candidate_root = ROOT / "outputs/official-rationale-candidate-evaluations-v1/official-rationale-handoff-v1-20260727-001"
    runner.stage(
        "rationale_candidate_evaluations",
        lambda: {"evaluations": {candidate["key"]: file_sha256(candidate_root / candidate["key"] / "evaluation.json") for candidate in pending["candidates"]}},
        lambda: command([str(PYTHON), "scripts/run_official_rationale_candidate_evaluations.py", "--config", str(pending_handoff_path)], RUN_ROOT / "logs/rationale-candidate-evaluations.log"),
    )

    final_handoff_path = CONFIG_ROOT / "official_rationale_handoff.resolved.json"
    resolved_handoff = resolve_handoff(handoff_template, bindings, require_evaluations=True)
    write_bound_config(final_handoff_path, resolved_handoff)
    HandoffConfig.from_json(final_handoff_path).validate_dependencies()
    handoff_runtime = ROOT / "outputs/official-rationale-handoff-v1/official-rationale-handoff-v1-20260727-001"
    runner.stage(
        "final_rationale_handoff",
        lambda: {
            "runtime_manifest_sha256": file_sha256(handoff_runtime / "manifest.json"),
            "restricted_manifest_sha256": file_sha256(RATIONALE_RESTRICTED_ROOT / "aggregate_handoff_manifest.json"),
        },
        lambda: command([str(PYTHON), "scripts/run_official_rationale_handoff.py", "--config", str(final_handoff_path)], RUN_ROOT / "logs/final-rationale-handoff.log"),
    )

    rationale_config = CONFIG_ROOT / "official_score_matrix.rationale.resolved.json"
    resolved_rationale = resolve_embedding_score_config(embed_template, include_rationales=True)
    write_bound_config(rationale_config, resolved_rationale)
    MatrixConfig.from_json(rationale_config).validate_dependencies("rationale")
    runner.stage(
        "embedding_score_rationale",
        lambda: {"aggregate_sha256": file_sha256(score_root / "aggregate_results.json")},
        lambda: command([str(PYTHON), "scripts/orchestrate_official_score_matrix.py", "--config", str(rationale_config), "--stage", "rationale"], RUN_ROOT / "logs/embedding-score-rationale.log"),
    )

    decoder_template = read_json(DECODER_TEMPLATE, "decoder score template")
    decoder_config = CONFIG_ROOT / "official_decoder_score_matrix.rationale.resolved.json"
    write_bound_config(decoder_config, resolve_decoder_score_config(decoder_template))
    DecoderScoreConfig.from_json(decoder_config, require_dependencies=False)
    decoder_root = ROOT / "outputs/official-decoder-score-matrix-v1"
    runner.stage(
        "decoder_score_matrix",
        lambda: {"aggregate_sha256": file_sha256(decoder_root / "aggregate.json")},
        lambda: command([
            str(PYTHON), "scripts/orchestrate_official_decoder_score_matrix.py", "--config", str(decoder_config),
            "--aihub-config", str(DECODER_PRETRAIN), "--reuse-completed-aihub-pretrain",
        ], RUN_ROOT / "logs/decoder-score-matrix.log"),
    )

    final = {
        "schema_version": "mal2026-official-rationale-score-final-aggregate-v1", "status": "completed",
        "run_id": RUN_ID, "score_fields": list(AXES), "average_target_used": False,
        "rationale_handoff_sha256": file_sha256(RATIONALE_RESTRICTED_ROOT / "aggregate_handoff_manifest.json"),
        "embedding_score_aggregate_sha256": file_sha256(score_root / "aggregate_results.json"),
        "decoder_score_aggregate_sha256": file_sha256(decoder_root / "aggregate.json"),
    }
    atomic_json(RUN_ROOT / "aggregate_results.json", final)
    runner.finish()


if __name__ == "__main__":
    main()
