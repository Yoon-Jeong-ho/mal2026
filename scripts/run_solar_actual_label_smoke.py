#!/usr/bin/env python3
"""Run the authorized Solar actual-triplet-label augmentation smoke.

This diagnostic keeps all four independently generated candidate families.
The requested axis/score is generation metadata only; the target-blind Solar
verifier's actual three-axis score is the sole pseudo-label.  No full mode is
provided by this runner.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_solar_target_augmentation as base  # noqa: E402
from mal2026.solar_target_augmentation import (  # noqa: E402
    AXES,
    CONFIG_PATH,
    TARGET_SCORES,
    AugmentationTask,
    SolarTargetAugmentationError,
    editor_output_schema,
    file_sha256,
    load_train_rows,
    make_task,
    parse_editor_output,
    parse_fidelity_output,
    parse_verifier_output,
    prompt_config,
    render_editor_messages,
    render_fidelity_messages,
    render_verifier_messages,
    select_smoke_sources,
    validate_actual_label_candidate,
)


OUTPUT_ROOT = ROOT / "outputs/solar-axis-actual-label-v1"
RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_axis_actual_label_v1"
MAX_INFLIGHT = 64
CANDIDATE_FAMILIES = 4
SOURCE_COUNT = 5
TASK_COUNT = SOURCE_COUNT * len(AXES) * len(TARGET_SCORES)
CANDIDATE_COUNT = TASK_COUNT * CANDIDATE_FAMILIES
REPEAT_CONTROL_COUNT = 30
MODAL_LABEL_DRAWS = 5
ACTUAL_EDITOR_MAX_TOKENS = 1600
CONTEXT_RUNTIME_MARGIN = 128


class SolarActualLabelSmokeError(RuntimeError):
    """Raised when the actual-label smoke contract is violated."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SolarActualLabelSmokeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def blind_scores(verifier: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    parsed = parse_verifier_output(json.dumps(verifier, ensure_ascii=False))
    return {axis: int(parsed[axis]["score"]) for axis in AXES}


def audit_editor_context(tasks: Sequence[AugmentationTask]) -> dict[str, Any]:
    """Tokenize every smoke editor request before sending any GPU work."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(base.RUNTIME_MODEL), trust_remote_code=True, local_files_only=True
    )
    lengths: list[int] = []
    violations = 0
    for task in tasks:
        for family_index in range(CANDIDATE_FAMILIES):
            schema = editor_output_schema(task)
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "mal2026_solar_editor",
                    "strict": True,
                    "schema": schema,
                },
            }
            encoded = tokenizer.apply_chat_template(
                render_editor_messages(task, family_index),
                tokenize=True,
                add_generation_prompt=True,
                reasoning_effort="none",
                think_render_option="preserved",
                response_format=response_format,
            )
            input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            length = len(input_ids)
            lengths.append(length)
            if length + ACTUAL_EDITOR_MAX_TOKENS + CONTEXT_RUNTIME_MARGIN > 4096:
                violations += 1
    need(len(lengths) == CANDIDATE_COUNT and violations == 0,
         "actual-label editor context preflight failed")
    return {
        "requests_audited": len(lengths),
        "local_prompt_tokens_min": min(lengths),
        "local_prompt_tokens_max": max(lengths),
        "editor_max_tokens": ACTUAL_EDITOR_MAX_TOKENS,
        "reserved_runtime_token_margin": CONTEXT_RUNTIME_MARGIN,
        "model_context_tokens": 4096,
        "violations": violations,
    }


def movement_metrics(
    task: AugmentationTask,
    actual: Mapping[str, int],
    source_actual: Mapping[str, int],
) -> dict[str, Any]:
    requested = task.target_score
    axis = task.target_axis
    source_target = int(source_actual[axis])
    candidate_target = int(actual[axis])
    source_distance = abs(requested - source_target)
    candidate_distance = abs(requested - candidate_target)
    progress = source_distance - candidate_distance
    non_targets = [item for item in AXES if item != axis]
    non_target_l1 = sum(abs(int(actual[item]) - int(source_actual[item])) for item in non_targets)
    if requested > source_target:
        requested_direction = "up"
        direction_followed = candidate_target > source_target
    elif requested < source_target:
        requested_direction = "down"
        direction_followed = candidate_target < source_target
    else:
        requested_direction = "same_band"
        direction_followed = candidate_target == source_target
    return {
        "source_blind_target_score": source_target,
        "actual_target_score": candidate_target,
        "requested_target_score": requested,
        "requested_direction": requested_direction,
        "direction_followed": direction_followed,
        "target_exact": candidate_target == requested,
        "source_target_distance": source_distance,
        "candidate_target_distance": candidate_distance,
        "target_progress": progress,
        "progress_class": "closer" if progress > 0 else ("same" if progress == 0 else "farther"),
        "non_target_l1_drift": non_target_l1,
        "non_target_exact": non_target_l1 == 0,
    }


def rejection(
    task: AugmentationTask,
    family_index: int,
    stage: str,
    category: str,
    *,
    essay: str | None = None,
    verifier: Mapping[str, Any] | None = None,
    fidelity: Mapping[str, Any] | None = None,
    raw_output: str | None = None,
) -> dict[str, Any]:
    value = base.rejection(
        task,
        family_index + 1,
        stage,
        category,
        essay=essay,
        verifier=verifier,
        fidelity=fidelity,
        raw_output=raw_output,
    )
    value.update({
        "schema_version": "mal2026-solar-actual-label-rejection-v1",
        "candidate_family_index": family_index,
        "candidate_id": f"{task.task_id}::family::{family_index}",
    })
    return value


def generate_candidate(
    endpoint: str,
    task: AugmentationTask,
    family_index: int,
    source_verifier: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Generate one independent family and retain its actual blind score."""
    need(0 <= family_index < CANDIDATE_FAMILIES, "candidate family differs")
    raw_editor: str | None = None
    raw_verifier: str | None = None
    raw_fidelity: str | None = None
    essay: str | None = None
    verifier: dict[str, dict[str, Any]] | None = None
    fidelity: dict[str, bool] | None = None
    try:
        raw_editor = base.request_content(
            endpoint,
            task,
            "editor",
            render_editor_messages(task, family_index),
            editor_output_schema(task),
            min(
                ACTUAL_EDITOR_MAX_TOKENS,
                int(prompt_config()["generation"]["editor"]["max_tokens"]),
            ),
            family_index + 1,
        )
        essay = parse_editor_output(
            raw_editor,
            task.source,
            task.target_axis,
            task.target_score,
            enforce_score_specific_edit_count=False,
        )
    except Exception as exc:
        return None, rejection(
            task, family_index, "editor", base.gate_category(exc), raw_output=raw_editor
        )
    try:
        raw_verifier = base.request_content(
            endpoint,
            task,
            "verifier",
            render_verifier_messages(task.source.prompt, essay),
            base.verifier_output_schema(),
            1000,
            family_index + 1,
        )
        verifier = parse_verifier_output(raw_verifier)
    except Exception as exc:
        return None, rejection(
            task,
            family_index,
            "verifier",
            base.gate_category(exc),
            essay=essay,
            raw_output=raw_verifier,
        )
    try:
        raw_fidelity = base.request_content(
            endpoint,
            task,
            "fidelity",
            render_fidelity_messages(task.source.essay, essay),
            base.fidelity_output_schema(),
            400,
            family_index + 1,
        )
        fidelity = parse_fidelity_output(raw_fidelity)
        validated = validate_actual_label_candidate(task, essay, verifier, fidelity)
    except Exception as exc:
        return None, rejection(
            task,
            family_index,
            "fidelity",
            base.gate_category(exc),
            essay=essay,
            verifier=verifier,
            fidelity=fidelity,
            raw_output=raw_fidelity,
        )

    actual = validated["score"]
    source_actual = blind_scores(source_verifier)
    metrics = movement_metrics(task, actual, source_actual)
    candidate_id = f"{task.task_id}::family::{family_index}"
    record = {
        "schema_version": "mal2026-solar-axis-actual-label-record-v1",
        "candidate_id": candidate_id,
        "task_id": task.task_id,
        "source_id": task.source.identifier,
        "source_document_id": task.source.document_id,
        "source_essay_sha256": sha256(task.source.essay.encode("utf-8")).hexdigest(),
        "candidate_essay_sha256": sha256(essay.encode("utf-8")).hexdigest(),
        "candidate_family_index": family_index,
        "requested_target_axis": task.target_axis,
        "requested_target_score": task.target_score,
        "requested_target_is_label": False,
        "prompt": task.source.prompt,
        "essay": essay,
        "editor_output": json.loads(raw_editor),
        "source_gold_score_for_editor_context_only": task.source.baseline,
        "source_blind_score": source_actual,
        "score": actual,
        "score_provenance": {
            "all_axes": "target_blind_solar_actual_triplet",
            "requested_target": "generation_metadata_only_not_a_label",
            "judge_temperature": 0.0,
        },
        "blind_verifier": verifier,
        "blind_source_verifier": source_verifier,
        "blind_fidelity": fidelity,
        "movement": metrics,
    }
    return record, None


def selection_key(record: Mapping[str, Any]) -> tuple[bytes, int]:
    """Rank families without consulting verifier scores, rationales, or target proximity."""
    task_id = record.get("task_id")
    family_index = record.get("candidate_family_index")
    need(isinstance(task_id, str) and type(family_index) is int,
         "candidate selection provenance differs")
    seed = prompt_config()["generation"]["seed"]
    digest = sha256(f"{seed}\0actual-label-selection\0{task_id}\0{family_index}".encode()).digest()
    return digest, family_index


def select_one_per_task(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        task_id = record.get("task_id")
        need(isinstance(task_id, str), "valid candidate task id differs")
        grouped[task_id].append(record)
    selected: list[dict[str, Any]] = []
    for task_id in sorted(grouped):
        winner = min(grouped[task_id], key=selection_key)
        value = dict(winner)
        value["selected_for_task_diagnostic"] = True
        value["selection_rule"] = (
            "predeclared_sha256_seeded_family_rank_independent_of_scores_rationales_and_targets"
        )
        selected.append(value)
    return selected


def select_repeat_control(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Select a predeclared score-independent 10% control sample."""
    seed = prompt_config()["generation"]["seed"]

    def key(record: Mapping[str, Any]) -> tuple[bytes, str]:
        candidate_id = record.get("candidate_id")
        need(isinstance(candidate_id, str), "repeat-control candidate id differs")
        digest = sha256(f"{seed}\0actual-label-repeat-control\0{candidate_id}".encode()).digest()
        return digest, candidate_id

    return sorted(records, key=key)[:min(REPEAT_CONTROL_COUNT, len(records))]


def repeat_score_candidate(
    endpoint: str,
    task: AugmentationTask,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_id = record.get("candidate_id")
    essay = record.get("essay")
    need(isinstance(candidate_id, str) and isinstance(essay, str),
         "repeat-control candidate differs")
    try:
        raw = base.request_content(
            endpoint,
            task,
            "verifier",
            render_verifier_messages(task.source.prompt, essay),
            base.verifier_output_schema(),
            1000,
            int(record["candidate_family_index"]) + 1,
        )
        repeated = parse_verifier_output(raw)
        repeated_scores = {axis: repeated[axis]["score"] for axis in AXES}
        original_scores = {axis: int(record["score"][axis]) for axis in AXES}
        agreements = {axis: repeated_scores[axis] == original_scores[axis] for axis in AXES}
        return {
            "schema_version": "mal2026-solar-actual-label-repeat-control-v1",
            "candidate_id": candidate_id,
            "status": "scored",
            "original_score": original_scores,
            "repeat_score": repeated_scores,
            "axis_agreement": agreements,
            "exact_triplet_agreement": all(agreements.values()),
            "repeat_blind_verifier": repeated,
        }
    except Exception as exc:
        return {
            "schema_version": "mal2026-solar-actual-label-repeat-control-v1",
            "candidate_id": candidate_id,
            "status": "failed",
            "failure_category": base.gate_category(exc),
        }


def run_repeat_control(
    endpoint: str,
    records: Sequence[Mapping[str, Any]],
    tasks_by_id: Mapping[str, AugmentationTask],
) -> list[dict[str, Any]]:
    sample = select_repeat_control(records)
    values: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(REPEAT_CONTROL_COUNT, max(1, len(sample)))) as pool:
        futures = {}
        for record in sample:
            task_id = record.get("task_id")
            need(isinstance(task_id, str) and task_id in tasks_by_id,
                 "repeat-control task lineage differs")
            futures[pool.submit(repeat_score_candidate, endpoint, tasks_by_id[task_id], record)] = record
        for future in as_completed(futures):
            values.append(future.result())
    values.sort(key=lambda item: item["candidate_id"])
    return values


def modal_triplet(draws: Sequence[Mapping[str, int]]) -> tuple[dict[str, int] | None, int]:
    """Return a unique five-draw majority triplet and its support."""
    if len(draws) != MODAL_LABEL_DRAWS:
        return None, 0
    counts = Counter(tuple(int(draw[axis]) for axis in AXES) for draw in draws)
    triplet, support = counts.most_common(1)[0]
    if support < 3:
        return None, support
    return {axis: triplet[index] for index, axis in enumerate(AXES)}, support


def run_modal_labeling(
    endpoint: str,
    selected: Sequence[Mapping[str, Any]],
    tasks_by_id: Mapping[str, AugmentationTask],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign selected, score-independently chosen candidates a 5-draw modal label.

    The first draw is the blind score obtained before candidate selection.  Four
    additional draws are intentionally issued one request at a time; this keeps
    the stability measurement independent of concurrent scheduler ordering.
    """
    stable: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for record in sorted(selected, key=lambda item: str(item["candidate_id"])):
        task_id = record.get("task_id")
        essay = record.get("essay")
        need(isinstance(task_id, str) and task_id in tasks_by_id and isinstance(essay, str),
             "modal-label candidate lineage differs")
        task = tasks_by_id[task_id]
        draws: list[dict[str, int]] = [
            {axis: int(record["score"][axis]) for axis in AXES}
        ]
        verifier_draws: list[Mapping[str, Any]] = [record["blind_verifier"]]
        failures: list[str] = []
        for draw_index in range(1, MODAL_LABEL_DRAWS):
            try:
                raw = base.request_content(
                    endpoint,
                    task,
                    "verifier",
                    render_verifier_messages(task.source.prompt, essay),
                    base.verifier_output_schema(),
                    1000,
                    int(record["candidate_family_index"]) + 1,
                )
                parsed = parse_verifier_output(raw)
                verifier_draws.append(parsed)
                draws.append({axis: int(parsed[axis]["score"]) for axis in AXES})
            except Exception as exc:
                failures.append(f"draw_{draw_index}:{base.gate_category(exc)}")
        label, support = modal_triplet(draws)
        control = {
            "schema_version": "mal2026-solar-modal-label-control-v1",
            "candidate_id": record["candidate_id"],
            "status": "stable" if label is not None and not failures else "unstable",
            "draws_requested": MODAL_LABEL_DRAWS,
            "draws_scored": len(draws),
            "score_draws": draws,
            "verifier_draws": verifier_draws,
            "modal_score": label,
            "modal_support": support,
            "all_five_exact": len({tuple(draw[axis] for axis in AXES) for draw in draws}) == 1,
            "failures": failures,
        }
        controls.append(control)
        if control["status"] != "stable":
            continue
        value = dict(record)
        value["single_draw_score"] = dict(record["score"])
        value["score"] = label
        value["score_provenance"] = {
            "all_axes": "target_blind_solar_joint_modal_triplet_five_draws",
            "requested_target": "generation_metadata_only_not_a_label",
            "selection": "completed_before_additional_judge_draws_and_score_independent",
            "modal_support": support,
            "draws": MODAL_LABEL_DRAWS,
        }
        value["movement"] = movement_metrics(task, label, record["source_blind_score"])
        value["stable_modal_label"] = True
        stable.append(value)
    return stable, controls


def axis_score_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        axis: {
            str(score): sum(int(record["score"][axis]) == score for record in records)
            for score in TARGET_SCORES
        }
        for axis in AXES
    }


def requested_cell_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        for score in TARGET_SCORES:
            cell = [
                record for record in records
                if record["requested_target_axis"] == axis and
                int(record["requested_target_score"]) == score
            ]
            target_scores = Counter(int(record["score"][axis]) for record in cell)
            values[f"{axis}:{score}"] = {
                "valid": len(cell),
                "actual_target_distribution": {
                    str(value): target_scores[value] for value in TARGET_SCORES
                },
                "target_exact": sum(record["movement"]["target_exact"] for record in cell),
                "direction_followed": sum(
                    record["movement"]["direction_followed"] for record in cell
                ),
                "closer": sum(record["movement"]["progress_class"] == "closer" for record in cell),
                "same": sum(record["movement"]["progress_class"] == "same" for record in cell),
                "farther": sum(record["movement"]["progress_class"] == "farther" for record in cell),
                "non_target_exact": sum(
                    record["movement"]["non_target_exact"] for record in cell
                ),
                "mean_target_abs_error": (
                    round(sum(record["movement"]["candidate_target_distance"] for record in cell) /
                          len(cell), 6) if cell else None
                ),
                "mean_non_target_l1_drift": (
                    round(sum(record["movement"]["non_target_l1_drift"] for record in cell) /
                          len(cell), 6) if cell else None
                ),
            }
    return values


def requested_target_correlations(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Report target-direction signal without using it for candidate selection."""
    result: dict[str, dict[str, Any]] = {}
    xs = [float(score) for score in TARGET_SCORES]
    mean_x = sum(xs) / len(xs)
    for axis in AXES:
        means: list[float] = []
        counts: dict[str, int] = {}
        for requested in TARGET_SCORES:
            cell = [
                int(record["score"][axis]) for record in records
                if record["requested_target_axis"] == axis and
                int(record["requested_target_score"]) == requested
            ]
            counts[str(requested)] = len(cell)
            means.append(sum(cell) / len(cell) if cell else float("nan"))
        if any(math.isnan(value) for value in means):
            correlation = None
        else:
            mean_y = sum(means) / len(means)
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, means))
            denominator = math.sqrt(
                sum((x - mean_x) ** 2 for x in xs) *
                sum((y - mean_y) ** 2 for y in means)
            )
            correlation = round(numerator / denominator, 6) if denominator else 0.0
        result[axis] = {
            "requested_score_cell_counts": counts,
            "actual_axis_mean_by_requested_score": {
                str(score): None if math.isnan(means[index]) else round(means[index], 6)
                for index, score in enumerate(TARGET_SCORES)
            },
            "pearson_requested_vs_actual_cell_mean": correlation,
        }
    return result


def aggregate_result(
    run_id: str,
    records: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    valid_path: Path,
    selected_path: Path,
    rejected_path: Path,
    repeat_control: Sequence[Mapping[str, Any]],
    repeat_control_path: Path,
    modal_selected: Sequence[Mapping[str, Any]],
    modal_selected_path: Path,
    modal_control: Sequence[Mapping[str, Any]],
    modal_control_path: Path,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    task_valid_counts = Counter(record["task_id"] for record in records)
    triple_counts = Counter(
        "/".join(str(int(record["score"][axis])) for axis in AXES) for record in records
    )
    selected_triples = Counter(
        "/".join(str(int(record["score"][axis])) for axis in AXES) for record in selected
    )
    failure_counts = Counter(
        f"{record.get('stage', 'unknown')}:{record.get('category', 'unknown')}"
        for record in rejected
    )
    valid_hashes = [record["candidate_essay_sha256"] for record in records]
    selected_hashes = [record["candidate_essay_sha256"] for record in selected]
    repeat_scored = [item for item in repeat_control if item.get("status") == "scored"]
    repeat_exact = sum(item.get("exact_triplet_agreement") is True for item in repeat_scored)
    repeat_axis_agreement = {
        axis: sum(item.get("axis_agreement", {}).get(axis) is True for item in repeat_scored)
        for axis in AXES
    }
    modal_stable = [item for item in modal_control if item.get("status") == "stable"]
    modal_all_exact = sum(item.get("all_five_exact") is True for item in modal_control)
    modal_support_counts = Counter(str(item.get("modal_support", 0)) for item in modal_control)
    cell_metrics = requested_cell_metrics(records)
    correlations = requested_target_correlations(modal_selected)
    mechanical_yield_ok = len(records) >= math.ceil(CANDIDATE_COUNT * 0.85)
    cell_yield_ok = all(item["valid"] >= 12 for item in cell_metrics.values())
    modal_gate_ok = (
        len(selected) == TASK_COUNT and len(modal_control) == TASK_COUNT and
        len(modal_stable) == TASK_COUNT and len(modal_selected) == TASK_COUNT
    )
    coverage_gate_ok = all(
        sum(count > 0 for count in axis_score_counts(modal_selected)[axis].values()) >= 3
        for axis in AXES
    )
    direction_gate_ok = all(
        item["pearson_requested_vs_actual_cell_mean"] is not None and
        item["pearson_requested_vs_actual_cell_mean"] > 0
        for item in correlations.values()
    )
    automatic_gate_passed = (
        len(task_valid_counts) == TASK_COUNT and mechanical_yield_ok and
        cell_yield_ok and modal_gate_ok and coverage_gate_ok and direction_gate_ok
    )
    return {
        "schema_version": "mal2026-solar-axis-actual-label-smoke-result-v2",
        "status": "completed",
        "run_id": run_id,
        "completed_at": now(),
        "source_records": SOURCE_COUNT,
        "tasks": TASK_COUNT,
        "candidate_families_per_task": CANDIDATE_FAMILIES,
        "candidates_expected": CANDIDATE_COUNT,
        "valid_candidates": len(records),
        "rejected_candidates": len(rejected),
        "tasks_with_valid_candidate": len(task_valid_counts),
        "tasks_without_valid_candidate": TASK_COUNT - len(task_valid_counts),
        "valid_candidates_per_task": {
            str(count): sum(value == count for value in task_valid_counts.values())
            for count in range(1, CANDIDATE_FAMILIES + 1)
        },
        "selected_candidates": len(selected),
        "stable_modal_selected_candidates": len(modal_selected),
        "all_valid_actual_axis_score_counts": axis_score_counts(records),
        "selected_actual_axis_score_counts": axis_score_counts(selected),
        "stable_modal_actual_axis_score_counts": axis_score_counts(modal_selected),
        "all_valid_requested_cell_metrics": cell_metrics,
        "selected_requested_cell_metrics": requested_cell_metrics(selected),
        "stable_modal_requested_cell_metrics": requested_cell_metrics(modal_selected),
        "stable_modal_requested_target_correlations": correlations,
        "all_valid_score_triple_counts": dict(sorted(triple_counts.items())),
        "selected_score_triple_counts": dict(sorted(selected_triples.items())),
        "all_valid_unique_essay_hashes": len(set(valid_hashes)),
        "all_valid_duplicate_essays": len(valid_hashes) - len(set(valid_hashes)),
        "selected_unique_essay_hashes": len(set(selected_hashes)),
        "selected_duplicate_essays": len(selected_hashes) - len(set(selected_hashes)),
        "failure_counts": dict(sorted(failure_counts.items())),
        "repeat_control": {
            "requested": REPEAT_CONTROL_COUNT,
            "attempted": len(repeat_control),
            "scored": len(repeat_scored),
            "failed": len(repeat_control) - len(repeat_scored),
            "exact_triplet_agreement": repeat_exact,
            "axis_agreement": repeat_axis_agreement,
            "selection_uses_scores_or_rationales": False,
            "sha256": file_sha256(repeat_control_path),
        },
        "modal_label_control": {
            "draws_per_candidate": MODAL_LABEL_DRAWS,
            "selection_completed_before_additional_draws": True,
            "selection_uses_scores_or_rationales": False,
            "attempted": len(modal_control),
            "stable_unique_majority": len(modal_stable),
            "all_five_exact": modal_all_exact,
            "modal_support_counts": dict(sorted(modal_support_counts.items())),
            "sha256": file_sha256(modal_control_path),
        },
        "automatic_gates": {
            "all_75_tasks_have_valid_candidate": len(task_valid_counts) == TASK_COUNT,
            "overall_valid_yield_at_least_85_percent": mechanical_yield_ok,
            "every_requested_cell_valid_yield_at_least_60_percent": cell_yield_ok,
            "all_75_score_independent_selections_have_unique_five_draw_majority": modal_gate_ok,
            "at_least_three_actual_score_bins_per_axis": coverage_gate_ok,
            "positive_requested_vs_actual_direction_signal_each_axis": direction_gate_ok,
            "passed": automatic_gate_passed,
            "manual_blind_quality_review_required": True,
        },
        "valid_candidates_sha256": file_sha256(valid_path),
        "selected_candidates_sha256": file_sha256(selected_path),
        "stable_modal_selected_candidates_sha256": file_sha256(modal_selected_path),
        "rejected_candidates_sha256": file_sha256(rejected_path),
        "bindings": dict(bindings),
        "protocol": {
            "requested_target_is_label": False,
            "actual_blind_triplet_is_label": True,
            "all_four_families_generated_independently": True,
            "judge_feedback_passed_to_editor": False,
            "one_per_task_selection_uses_judge_output": False,
            "validation_used": False,
            "full_run_authorized": False,
        },
        "privacy": (
            "aggregate contains no essay, prompt, rationale, identifier, raw output, "
            "or individual prediction row"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--port", type=int, default=19420)
    parser.add_argument("--max-inflight", type=int, default=MAX_INFLIGHT)
    parser.add_argument("--external-endpoint", required=True)
    parser.add_argument("--external-container-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    need(re.fullmatch(r"[a-z0-9][a-z0-9-]{7,95}", args.run_id) is not None, "run ID differs")
    need(args.max_inflight == MAX_INFLIGHT, "concurrency protocol differs")
    need(args.external_endpoint == f"http://127.0.0.1:{args.port}", "external endpoint differs")

    config = prompt_config()
    need(config["execution_gate"]["full_run_authorized"] is False,
         "strict full-run gate must remain disabled during actual-label smoke")
    model_binding = base.verify_model()
    image_binding = base.docker_image_binding()
    rows = select_smoke_sources(load_train_rows(), count=SOURCE_COUNT)
    tasks = [
        make_task(source, axis, score)
        for source in rows for axis in AXES for score in TARGET_SCORES
    ]
    need(len(tasks) == TASK_COUNT and len({task.task_id for task in tasks}) == TASK_COUNT,
         "actual-label smoke task matrix differs")
    editor_context_preflight = audit_editor_context(tasks)

    output = OUTPUT_ROOT / args.run_id
    restricted = RESTRICTED_ROOT / args.run_id
    need(not output.exists() and not restricted.exists(), "run outputs must be fresh")
    output.mkdir(mode=0o700, parents=True)
    restricted.mkdir(mode=0o700, parents=True)
    os.chmod(output, 0o700)
    os.chmod(restricted, 0o700)

    bindings = {
        "prompt_config_sha256": file_sha256(CONFIG_PATH),
        "evaluation_sha256": config["provenance"]["rubric_source_sha256"],
        "train_sha256": config["provenance"]["train_source_sha256"],
        "validation_sha256": config["provenance"]["validation_source_sha256"],
        "model": model_binding,
        "image": image_binding,
        "strict_core_sha256": file_sha256(ROOT / "src/mal2026/solar_target_augmentation.py"),
        "strict_runner_sha256": file_sha256(ROOT / "scripts/run_solar_target_augmentation.py"),
        "actual_label_runner_sha256": file_sha256(Path(__file__).resolve()),
    }
    manifest = {
        "schema_version": "mal2026-solar-axis-actual-label-smoke-manifest-v1",
        "status": "preflight",
        "run_id": args.run_id,
        "created_at": now(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_dirty_at_launch": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
        "command": [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *sys.argv[1:]],
        "gpu_scope": list(base.GPU_SCOPE),
        "gpu_authorization": (
            "repository default GPUs0-3 and user-authorized Solar actual-triplet-label smoke; "
            "GPUs4-7 not queried or used"
        ),
        "scientific_authorization": (
            "user approved continued testing with target-blind actual three-axis labels on "
            "2026-07-30"
        ),
        "source_split": "train_only",
        "validation_used_for_generation_or_selection": False,
        "source_records": SOURCE_COUNT,
        "tasks": TASK_COUNT,
        "candidate_families_per_task": CANDIDATE_FAMILIES,
        "candidates_expected": CANDIDATE_COUNT,
        "editor_max_tokens": ACTUAL_EDITOR_MAX_TOKENS,
        "editor_context_preflight": editor_context_preflight,
        "bindings": bindings,
    }
    atomic_json(output / "manifest.json", manifest)

    try:
        manifest["external_server"] = base.external_server_binding(
            args.external_container_name, args.port
        )
        base.wait_server(None, args.external_endpoint, seconds=60)
        manifest.update({"status": "scoring_sources", "server_ready_at": now()})
        atomic_json(output / "manifest.json", manifest)
        source_verifiers, source_scores_path = base.score_source_rows(
            args.external_endpoint, rows, restricted, args.max_inflight
        )
        manifest.update({
            "blind_source_scores": len(source_verifiers),
            "blind_source_scores_sha256": file_sha256(source_scores_path),
            "status": "generating_candidates",
        })
        atomic_json(output / "manifest.json", manifest)

        records: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        pairs = [
            (task, family_index)
            for task in tasks for family_index in range(CANDIDATE_FAMILIES)
        ]
        with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
            futures = {
                pool.submit(
                    generate_candidate,
                    args.external_endpoint,
                    task,
                    family_index,
                    source_verifiers[task.source.identifier],
                ): (task, family_index)
                for task, family_index in pairs
            }
            for future in as_completed(futures):
                task, family_index = futures[future]
                try:
                    record, failed = future.result()
                except Exception as exc:
                    record, failed = None, rejection(
                        task, family_index, "worker", base.gate_category(exc)
                    )
                if record is not None:
                    records.append(record)
                if failed is not None:
                    rejected.append(failed)

        need(len(records) + len(rejected) == CANDIDATE_COUNT,
             "actual-label candidate population differs")
        records.sort(key=lambda item: item["candidate_id"])
        rejected.sort(key=lambda item: item["candidate_id"])
        selected = select_one_per_task(records)
        tasks_by_id = {task.task_id: task for task in tasks}
        repeat_control = run_repeat_control(
            args.external_endpoint, records, tasks_by_id
        )
        manifest.update({
            "status": "assigning_five_draw_modal_labels",
            "score_independent_selections": len(selected),
        })
        atomic_json(output / "manifest.json", manifest)
        modal_selected, modal_control = run_modal_labeling(
            args.external_endpoint, selected, tasks_by_id
        )
        valid_path = restricted / "valid_candidates.jsonl"
        selected_path = restricted / "selected_candidates.jsonl"
        rejected_path = restricted / "rejected_candidates.jsonl"
        repeat_control_path = restricted / "repeat_control.jsonl"
        modal_selected_path = restricted / "stable_modal_selected_candidates.jsonl"
        modal_control_path = restricted / "modal_label_control.jsonl"
        write_jsonl(valid_path, records)
        write_jsonl(selected_path, selected)
        write_jsonl(rejected_path, rejected)
        write_jsonl(repeat_control_path, repeat_control)
        write_jsonl(modal_selected_path, modal_selected)
        write_jsonl(modal_control_path, modal_control)
        result = aggregate_result(
            args.run_id,
            records,
            selected,
            rejected,
            valid_path,
            selected_path,
            rejected_path,
            repeat_control,
            repeat_control_path,
            modal_selected,
            modal_selected_path,
            modal_control,
            modal_control_path,
            bindings,
        )
        atomic_json(output / "result.json", result)
        manifest.update({
            "status": result["status"],
            "completed_at": now(),
            "result_sha256": file_sha256(output / "result.json"),
        })
        atomic_json(output / "manifest.json", manifest)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "failed_at": now(),
            "failure_category": base.gate_category(exc),
        })
        atomic_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
