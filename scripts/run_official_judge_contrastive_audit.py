#!/usr/bin/env python3
"""Run the predeclared train-only directional gate for the frozen proxy judge."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
JUDGE = ROOT / "scripts/run_official_q4_judge.sh"
PREP_ID = "official-judge-contrastive-train32-001"
PARTICIPANT_ROOT = ROOT / "data/processed/restricted/official_prompt_alignment_v1/judge_contrastive" / PREP_ID
RUN_ROOT = ROOT / "outputs/official-prompt-alignment-v1/judge-contrastive" / PREP_ID
AIHUB_COMPARISON = ROOT / "outputs/official-prompt-alignment-v1/aihub-comparison/official-aihub-rationale-comparison-v1-20260727-001/aggregate_aihub_comparison.json"
JUDGE_AGGREGATE_ROOT = ROOT / "outputs/official-prompt-alignment-v1/q4-judge"
JUDGE_RECORD_ROOT = ROOT / "data/processed/restricted/official_prompt_alignment_v1/q4_judge"
LEDGER = ROOT / "outputs/official-prompt-alignment-v1/20260727-001/ledger.jsonl"
VARIANTS = ("base", "axis_swapped", "score_perturbed", "unsupported")
RUN_IDS = {name: f"official-q4-judge-contrastive-{name.replace('_', '-')}-train32-001" for name in VARIANTS}
MIN_MEAN_DECREASE = 0.25
MIN_PAIRED_DECREASE_RATE = 0.50


class ContrastiveError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ContrastiveError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def ledger(event: str, evidence: str, *, decision: str, deviation: str) -> None:
    row = {
        "timestamp": now(), "run_id": "official-prompt-alignment-v1-20260727-001",
        "stage": "official_proxy_judge_contrastive_gate", "event": event,
        "command_ref": str(Path(__file__).relative_to(ROOT)), "resource_scope": "GPUs 0-3",
        "gpu_scope_authorization": "repository default GPUs 0-3; user explicitly requested GPUs 0-3",
        "failure_family": "none" if decision == "continue" else "judge_directional_gate_failed",
        "repair_iteration": 0, "decision": decision, "deviation": deviation, "evidence_ref": evidence,
    }
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def wait_for_aihub_comparison() -> None:
    manifest = AIHUB_COMPARISON.parent / "manifest.json"
    for _ in range(1440):
        if AIHUB_COMPARISON.is_file():
            value = json.loads(AIHUB_COMPARISON.read_text(encoding="utf-8")); need(value.get("status") == "completed", "AI-Hub comparison differs"); return
        if manifest.is_file():
            value = json.loads(manifest.read_text(encoding="utf-8")); need(value.get("status") != "failed", "AI-Hub comparison failed before judge audit")
        time.sleep(30)
    raise ContrastiveError("AI-Hub comparison wait timeout")


def run_variant(name: str) -> None:
    participant = PARTICIPANT_ROOT / f"{name}.jsonl"; run_id = RUN_IDS[name]
    report = JUDGE_AGGREGATE_ROOT / run_id / "aggregate_judge_report.json"
    need(participant.is_file() and not report.parent.exists(), f"contrastive prerequisite/freshness differs: {name}")
    log = RUN_ROOT / "logs" / f"judge-{name}.log"; need(not log.exists(), f"contrastive log exists: {name}")
    command = [str(JUDGE), "audit", run_id, "train", str(participant), "32"]
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    need(completed.returncode == 0, f"contrastive Q4 process failed: {name}")
    value = json.loads(report.read_text(encoding="utf-8"))
    need(value.get("status") == "completed" and all(value.get("hard_gates", {}).values()), f"contrastive Q4 aggregate failed: {name}")


def target_values(run_id: str, dimension: str) -> list[tuple[str, float]]:
    path = JUDGE_RECORD_ROOT / run_id / "judge_records.jsonl"; values: list[tuple[str, float]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); output = row.get("judge_output"); need(output is not None, "contrastive judge row invalid")
            scores = [int(output[axis][dimension]["score"]) for axis in ("content", "organization", "expression")]
            values.append((str(row["source_id"]), statistics.fmean(scores)))
    need(len(values) == 32, "contrastive judge population differs")
    return values


def comparison(corruption: str, dimension: str) -> dict[str, Any]:
    base = target_values(RUN_IDS["base"], dimension); changed = target_values(RUN_IDS[corruption], dimension)
    need([x[0] for x in base] == [x[0] for x in changed], "contrastive paired IDs differ")
    differences = [a[1] - b[1] for a, b in zip(base, changed, strict=True)]
    mean_decrease = statistics.fmean(differences); decrease = sum(value > 0 for value in differences)
    result = {
        "target_dimension": dimension, "mean_decrease": mean_decrease,
        "paired_decrease_count": decrease, "paired_tie_count": sum(value == 0 for value in differences),
        "paired_increase_count": sum(value < 0 for value in differences), "paired_decrease_rate": decrease / len(differences),
    }
    result["passed"] = mean_decrease >= MIN_MEAN_DECREASE and result["paired_decrease_rate"] >= MIN_PAIRED_DECREASE_RATE
    return result


def main() -> None:
    aggregate = RUN_ROOT / "aggregate_contrastive_gate.json"; need((RUN_ROOT / "preparation_report.json").is_file() and not aggregate.exists(), "contrastive preparation/output differs")
    (RUN_ROOT / "logs").mkdir()
    wait_for_aihub_comparison()
    for name in VARIANTS: run_variant(name)
    tests = {
        "axis_swap_domain_match": comparison("axis_swapped", "domain_match"),
        "score_perturbation_consistency": comparison("score_perturbed", "score_rationale_consistency"),
        "unsupported_groundedness": comparison("unsupported", "groundedness"),
    }
    passed = all(value["passed"] for value in tests.values())
    payload = {
        "schema_version": "mal2026-official-proxy-judge-contrastive-gate-v1", "status": "passed" if passed else "failed_gates",
        "run_id": PREP_ID, "completed_at": now(), "records_per_variant": 32,
        "thresholds_frozen_before_results": {"minimum_target_mean_decrease": MIN_MEAN_DECREASE, "minimum_paired_decrease_rate": MIN_PAIRED_DECREASE_RATE},
        "tests": tests, "rl_with_this_proxy_judge_allowed": passed,
        "judge_prompt_kind": "single_frozen_public-spec-aligned_proxy_not_verbatim_organizer_prompt",
        "human_or_reference_score_read_or_prompted": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_predictions",
    }
    atomic_json(aggregate, payload)
    ledger("smoke_pass" if passed else "failed", str(aggregate.relative_to(ROOT)), decision="continue" if passed else "escalate", deviation="RL may proceed" if passed else "RL with this saturated fixed proxy judge is skipped by the predeclared directional gate")
    print(json.dumps({"status": payload["status"], "rl_allowed": passed, "tests": tests}, sort_keys=True))
    # Persist the complete negative result above, then fail closed so a caller
    # cannot mistake a scientifically failed directional gate for success by
    # checking only the process exit status.
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
