#!/usr/bin/env python3
"""Compose bundle/axis participants and compare them with the fixed Q4 judge."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-standard/bin/python"
COMPOSE = ROOT / "scripts/compose_official_participants.py"
JUDGE = ROOT / "scripts/run_official_q4_judge.sh"
RESTRICTED = ROOT / "data/processed/restricted/official_prompt_alignment_v1"
GENERATION = RESTRICTED / "rationale_generation"
PARTICIPANTS = RESTRICTED / "participants"
SCORES = RESTRICTED / "score_predictions/official-score-essay-only-full-20260727-002/essay_only_epoch_04.jsonl"
SMOKE_SCORES = GENERATION / "official-api-candidate-train-smoke-scores-001.jsonl"
REPORT_ROOT = ROOT / "outputs/official-prompt-alignment-v1/structure-comparison"
RUN_ID = "official-rationale-structure-comparison-v1-20260727-001"


class ComparisonError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


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


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    need(not log.exists(), f"comparison log exists: {log.name}")
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, stdout=handle, stderr=subprocess.STDOUT, text=True)
    need(completed.returncode == 0, f"comparison command failed: {log.name}")


def compose(run_id: str, score_file: Path, generation_dirs: list[Path], output: Path, aggregate: Path, expected: int) -> None:
    command = [
        str(PYTHON), str(COMPOSE), "--run-id", run_id, "--score-file", str(score_file),
        "--output-file", str(output), "--aggregate-file", str(aggregate), "--expected", str(expected),
    ]
    for path in generation_dirs:
        command.extend(["--generation-dir", str(path)])
    run(command, REPORT_ROOT / "logs" / f"compose-{run_id}.log")


def judge(mode: str, run_id: str, split: str, participant: Path, expected: int) -> None:
    run([str(JUDGE), mode, run_id, split, str(participant), str(expected)], REPORT_ROOT / "logs" / f"judge-{run_id}.log")


def main() -> None:
    report = REPORT_ROOT / RUN_ID / "aggregate_structure_comparison.json"
    need(not report.exists(), "structure comparison output must be fresh")
    bundle_generation = GENERATION / "official-rationale-generation-v1-ax4-bundle-validation-004"
    axis_generations = [GENERATION / f"official-rationale-generation-v1-ax4-{axis}-validation-004" for axis in ("content", "organization", "expression")]
    smoke_generation = GENERATION / "official-rationale-generation-v1-ax4-bundle-train-smoke-004"
    for path in [SCORES, SMOKE_SCORES, smoke_generation, bundle_generation, *axis_generations]:
        need(path.exists(), f"comparison prerequisite is unavailable: {path.name}")
    PARTICIPANTS.mkdir(mode=0o700, parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    smoke_participant = PARTICIPANTS / "official-rationale-ax4-bundle-train-smoke-001.jsonl"
    bundle_participant = PARTICIPANTS / "official-rationale-ax4-bundle-validation-001.jsonl"
    axis_participant = PARTICIPANTS / "official-rationale-ax4-axis-triplet-validation-001.jsonl"
    outputs = [smoke_participant, bundle_participant, axis_participant]
    need(not any(path.exists() for path in outputs), "participant composition output must be fresh")
    compose(
        "official-rationale-ax4-bundle-train-smoke-001", SMOKE_SCORES, [smoke_generation], smoke_participant,
        report.parent / "participant-bundle-train-smoke.json", 1,
    )
    compose(
        "official-rationale-ax4-bundle-validation-001", SCORES, [bundle_generation], bundle_participant,
        report.parent / "participant-bundle-validation.json", 400,
    )
    compose(
        "official-rationale-ax4-axis-triplet-validation-001", SCORES, axis_generations, axis_participant,
        report.parent / "participant-axis-triplet-validation.json", 400,
    )
    judge("smoke", "official-q4-judge-ax4-bundle-train-smoke-001", "train", smoke_participant, 1)
    judge("full", "official-q4-judge-ax4-bundle-validation-001", "validation", bundle_participant, 400)
    judge("full", "official-q4-judge-ax4-axis-triplet-validation-001", "validation", axis_participant, 400)
    judge_root = ROOT / "outputs/official-prompt-alignment-v1/q4-judge"
    paths = {
        "bundle": judge_root / "official-q4-judge-ax4-bundle-validation-001/aggregate_judge_report.json",
        "axis_triplet": judge_root / "official-q4-judge-ax4-axis-triplet-validation-001/aggregate_judge_report.json",
    }
    values = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    for name, value in values.items():
        need(value.get("status") == "completed" and all(value.get("hard_gates", {}).values()), f"Q4 judge aggregate gate failed: {name}")
    winner = max(values, key=lambda name: (float(values[name]["macro_mean"]), float(values[name]["worst_cell_mean"])))
    payload = {
        "schema_version": "mal2026-official-rationale-structure-comparison-v1",
        "status": "completed",
        "run_id": RUN_ID,
        "completed_at": now(),
        "methods": {
            name: {
                "macro_mean": value["macro_mean"],
                "worst_cell_mean": value["worst_cell_mean"],
                "score_1_or_2_rate": value["score_1_or_2_rate"],
                "axis_means": value["axis_means"],
                "dimension_means": value["dimension_means"],
                "cell_means": value["cell_means"],
                "judge_report_sha256": file_sha(paths[name]),
                "participant_sha256": value["participant_sha256"],
            }
            for name, value in values.items()
        },
        "winner_by_frozen_macro_then_worst_cell": winner,
        "score_file_identical_across_methods": True,
        "score_file_sha256": file_sha(SCORES),
        "judge_prompt_kind": "single_frozen_public-spec-aligned_proxy_not_verbatim_organizer_prompt",
        "exact_q4_model_sha256": values["bundle"]["model_sha256"],
        "human_or_reference_score_read_or_prompted": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_evidence_or_predictions",
    }
    atomic_json(report, payload)
    print(json.dumps({"status": "completed", "winner": winner, "macro_means": {name: value["macro_mean"] for name, value in values.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
