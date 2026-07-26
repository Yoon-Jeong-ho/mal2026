#!/usr/bin/env python3
"""Wait for AI-Hub->API SFT, then generate, judge, and compare the selected structure."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-standard/bin/python"
TRAIN_RUN = ROOT / "outputs/official-prompt-alignment-v1/aihub-then-api-rationale-pipeline/official-aihub-then-api-rationale-sft-v1-20260727-005"
TRAIN_REPORT = TRAIN_RUN / "aggregate_training_report.json"
TRAIN_MANIFEST = TRAIN_RUN / "manifest.json"
GENERATOR = ROOT / "scripts/run_official_aihub_then_api_rationale_generation.py"
COMPOSER = ROOT / "scripts/compose_official_participants.py"
JUDGE = ROOT / "scripts/run_official_q4_judge.sh"
SCORE_FILE = ROOT / "data/processed/restricted/official_prompt_alignment_v1/score_predictions/official-score-essay-only-full-20260727-002/essay_only_epoch_04.jsonl"
GEN_ROOT = ROOT / "data/processed/restricted/official_prompt_alignment_v1/rationale_generation"
PARTICIPANT = ROOT / "data/processed/restricted/official_prompt_alignment_v1/participants/official-aihub-then-api-rationale-ax4-axis-triplet-validation-001.jsonl"
RUN_ID = "official-aihub-rationale-comparison-v1-20260727-001"
RUN_ROOT = ROOT / "outputs/official-prompt-alignment-v1/aihub-comparison" / RUN_ID
PARTICIPANT_REPORT = RUN_ROOT / "participant-aihub-then-api-axis-triplet-validation.json"
NEW_JUDGE_ID = "official-q4-judge-aihub-then-api-ax4-axis-triplet-validation-001"
NEW_JUDGE_REPORT = ROOT / "outputs/official-prompt-alignment-v1/q4-judge" / NEW_JUDGE_ID / "aggregate_judge_report.json"
OLD_JUDGE_ID = "official-q4-judge-ax4-axis-triplet-validation-001"
OLD_JUDGE_REPORT = ROOT / "outputs/official-prompt-alignment-v1/q4-judge" / OLD_JUDGE_ID / "aggregate_judge_report.json"
OLD_STRUCTURE_REPORT = ROOT / "outputs/official-prompt-alignment-v1/structure-comparison/official-rationale-structure-comparison-v1-20260727-001/aggregate_structure_comparison.json"
JUDGE_RECORD_ROOT = ROOT / "data/processed/restricted/official_prompt_alignment_v1/q4_judge"
LEDGER = ROOT / "outputs/official-prompt-alignment-v1/20260727-001/ledger.jsonl"


class EvaluationError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


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
        "timestamp": now(), "run_id": "official-prompt-alignment-v1-20260727-001", "stage": stage, "event": event,
        "command_ref": list(command) if not isinstance(command, str) else command, "resource_scope": resource,
        "gpu_scope_authorization": "repository default GPUs 0-3; user explicitly requested GPUs 0-3",
        "failure_family": failure, "repair_iteration": 0, "decision": decision, "deviation": deviation, "evidence_ref": evidence,
    }
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(command: Sequence[str], log: Path, stage: str, resource: str) -> None:
    need(not log.exists(), f"evaluation log exists: {log.name}")
    ledger(stage, "start", str(log.relative_to(ROOT)), command, resource)
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        ledger(stage, "failed", str(log.relative_to(ROOT)), command, resource, failure="process_nonzero", decision="escalate")
        raise EvaluationError(f"evaluation stage failed ({completed.returncode}): {stage}")


def wait_for_training() -> None:
    for _ in range(1440):
        if TRAIN_REPORT.is_file():
            value = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
            need(value.get("status") == "completed" and value.get("structure") == "axis_triplet", "training aggregate differs")
            return
        if TRAIN_MANIFEST.is_file():
            manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
            need(manifest.get("status") != "failed", "training pipeline failed before evaluation")
        time.sleep(30)
    raise EvaluationError("training wait timeout")


def wait_for_gpus_idle() -> None:
    for _ in range(120):
        values = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-i", "0,1,2,3"], text=True).splitlines()
        parsed = [tuple(int(part.strip()) for part in line.split(",")) for line in values]
        if parsed == [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]:
            return
        time.sleep(2)
    raise EvaluationError("GPUs 0-3 did not become idle after owned training stage")


def judge_histogram(run_id: str) -> tuple[dict[str, Any], list[tuple[str, float]]]:
    path = JUDGE_RECORD_ROOT / run_id / "judge_records.jsonl"
    histogram: Counter[int] = Counter(); per_essay: list[tuple[str, float]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); output = row.get("judge_output"); need(output is not None, "judge row is invalid")
            values = [int(output[axis][dimension]["score"]) for axis in ("content", "organization", "expression") for dimension in ("domain_match", "score_rationale_consistency", "specificity", "groundedness")]
            histogram.update(values); per_essay.append((str(row["source_id"]), statistics.fmean(values)))
    need(len(per_essay) == 400 and sum(histogram.values()) == 4800, "judge distribution population differs")
    result = {
        "judge_cells": 4800, "score_histogram": {str(score): histogram[score] for score in range(1, 6)},
        "score_percentages": {str(score): histogram[score] / 4800 for score in range(1, 6)},
        "perfect_12_of_12_essays": sum(value == 5 for _, value in per_essay),
        "perfect_12_of_12_rate": sum(value == 5 for _, value in per_essay) / 400,
    }
    return result, per_essay


def main() -> None:
    need(not RUN_ROOT.exists() and not PARTICIPANT.exists() and not NEW_JUDGE_REPORT.parent.exists(), "evaluation outputs must be fresh")
    RUN_ROOT.mkdir(parents=True); (RUN_ROOT / "logs").mkdir()
    manifest: dict[str, Any] = {
        "schema_version": "mal2026-official-aihub-rationale-evaluation-runner-v1", "status": "waiting_for_training",
        "run_id": RUN_ID, "created_at": now(), "training_run": str(TRAIN_RUN.relative_to(ROOT)),
        "gpu_scope": "GPUs 0-3", "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_evidence_or_predictions",
    }
    atomic_json(RUN_ROOT / "manifest.json", manifest)
    try:
        wait_for_training(); wait_for_gpus_idle(); manifest["status"] = "generating"; atomic_json(RUN_ROOT / "manifest.json", manifest)
        run([str(PYTHON), str(GENERATOR)], RUN_ROOT / "logs/generation.log", "official_aihub_then_api_rationale_generation", "GPUs 0-2")
        generation_dirs = [GEN_ROOT / f"official-aihub-then-api-rationale-generation-v1-ax4-{task}-validation-001" for task in ("content", "organization", "expression")]
        compose_command = [str(PYTHON), str(COMPOSER), "--run-id", "official-aihub-then-api-rationale-ax4-axis-triplet-validation-001", "--score-file", str(SCORE_FILE)]
        for path in generation_dirs:
            compose_command.extend(["--generation-dir", str(path)])
        compose_command.extend(["--output-file", str(PARTICIPANT), "--aggregate-file", str(PARTICIPANT_REPORT), "--expected", "400"])
        run(compose_command, RUN_ROOT / "logs/compose.log", "official_aihub_then_api_participant_composition", "CPU")
        participant_report = json.loads(PARTICIPANT_REPORT.read_text(encoding="utf-8"))
        need(participant_report.get("status") == "completed" and participant_report.get("records") == 400 and participant_report.get("strict_participant_parse_count") == 400 and participant_report.get("score_rationale_model_score_mismatch_count") == 0, "participant composition gate failed")
        judge_command = [str(JUDGE), "full", NEW_JUDGE_ID, "validation", str(PARTICIPANT), "400"]
        run(judge_command, RUN_ROOT / "logs/q4-judge.log", "official_aihub_then_api_Q4_judge", "GPUs 0-3")
        reports = {"no_aihub_axis_triplet": json.loads(OLD_JUDGE_REPORT.read_text(encoding="utf-8")), "aihub_full_then_api_lora_axis_triplet": json.loads(NEW_JUDGE_REPORT.read_text(encoding="utf-8"))}
        for name, value in reports.items():
            need(value.get("status") == "completed" and all(value.get("hard_gates", {}).values()), f"judge aggregate gate failed: {name}")
        winner = max(reports, key=lambda name: (float(reports[name]["macro_mean"]), float(reports[name]["worst_cell_mean"])))
        distributions: dict[str, Any] = {}; essays: dict[str, list[tuple[str, float]]] = {}
        for name, judge_id in (("no_aihub_axis_triplet", OLD_JUDGE_ID), ("aihub_full_then_api_lora_axis_triplet", NEW_JUDGE_ID)):
            distributions[name], essays[name] = judge_histogram(judge_id)
        old, new = essays["no_aihub_axis_triplet"], essays["aihub_full_then_api_lora_axis_triplet"]
        need([x[0] for x in old] == [x[0] for x in new], "paired judge IDs differ")
        paired = {
            "aihub_full_then_api_lora_higher": sum(b[1] > a[1] for a, b in zip(old, new, strict=True)),
            "tie": sum(b[1] == a[1] for a, b in zip(old, new, strict=True)),
            "no_aihub_higher": sum(b[1] < a[1] for a, b in zip(old, new, strict=True)), "paired_essays": 400,
        }
        old_structure = json.loads(OLD_STRUCTURE_REPORT.read_text(encoding="utf-8"))
        score_file_identical = participant_report["score_file_sha256"] == file_sha(SCORE_FILE) == old_structure["score_file_sha256"]
        need(score_file_identical, "comparison score file differs")
        payload = {
            "schema_version": "mal2026-official-aihub-rationale-comparison-v1", "status": "completed", "run_id": RUN_ID,
            "completed_at": now(), "winner_by_frozen_macro_then_worst_cell": winner,
            "methods": {name: {key: value[key] for key in ("macro_mean", "worst_cell_mean", "score_1_or_2_rate", "axis_means", "dimension_means", "cell_means")} for name, value in reports.items()},
            "macro_difference_aihub_minus_no_aihub": reports["aihub_full_then_api_lora_axis_triplet"]["macro_mean"] - reports["no_aihub_axis_triplet"]["macro_mean"],
            "judge_score_distributions": distributions, "paired_essay_macro_comparison": paired,
            "score_file_identical": score_file_identical,
            "score_file_sha256": file_sha(SCORE_FILE), "strict_participant_records": 400, "score_mismatch_count": 0,
            "judge_prompt_kind": "single_frozen_public-spec-aligned_proxy_not_verbatim_organizer_prompt",
            "exact_q4_model_sha256": reports["no_aihub_axis_triplet"]["model_sha256"], "human_or_reference_score_read_or_prompted": False,
            "interpretation_caveat": "validation has been repeatedly exposed and the fixed proxy judge may be saturated; report differences descriptively rather than as unbiased significance",
            "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_evidence_or_predictions",
        }
        atomic_json(RUN_ROOT / "aggregate_aihub_comparison.json", payload)
        manifest.update({"status": "completed", "completed_at": now(), "aggregate_comparison_sha256": file_sha(RUN_ROOT / "aggregate_aihub_comparison.json")}); atomic_json(RUN_ROOT / "manifest.json", manifest)
        ledger("official_aihub_rationale_comparison", "next_stage_complete", str((RUN_ROOT / "aggregate_aihub_comparison.json").relative_to(ROOT)), str(Path(__file__).relative_to(ROOT)), "GPUs 0-3", deviation="same frozen score file and fixed proxy judge; repeated-validation/saturation caveat retained")
        print(json.dumps({"status": "completed", "winner": winner, "macro_difference": payload["macro_difference_aihub_minus_no_aihub"]}, sort_keys=True))
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at": now(), "failure_type": type(exc).__name__, "failure_message": str(exc)}); atomic_json(RUN_ROOT / "manifest.json", manifest); raise


if __name__ == "__main__":
    main()
