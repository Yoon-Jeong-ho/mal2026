#!/usr/bin/env python3
"""Generate, hard-filter, and adaptively label a train-only Solar pilot pool."""
from __future__ import annotations

import argparse
from collections import Counter
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
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_solar_actual_label_smoke as actual  # noqa: E402
import run_solar_target_augmentation as base  # noqa: E402
from mal2026.solar_consensus_pilot import (  # noqa: E402
    JUDGE_DRAWS_INITIAL,
    JUDGE_DRAWS_MAXIMUM,
    modal_label,
    requires_two_more_draws,
    stratified_sources,
    visible_draw_seed,
)
from mal2026.solar_target_augmentation import (  # noqa: E402
    AXES,
    CONFIG_PATH,
    TARGET_SCORES,
    AugmentationTask,
    SourceRow,
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
    validate_actual_label_candidate,
)


OUTPUT_ROOT = ROOT / "outputs/solar-consensus-pilot-v1"
RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_consensus_pilot_v1"
MAX_INFLIGHT = 64
PILOT_SOURCE_COUNT = 20
PILOT_FAMILIES = 8
SMOKE_SOURCE_COUNT = 5
SMOKE_FAMILIES = 2
# The 20-source stratified pilot contains four expression-5 requests whose
# longest prompt is 2,390 tokens.  A 1,570-token cap plus the fixed 128-token
# runtime margin fits the immutable 4,096-token server context (4,088 total).
EDITOR_MAX_TOKENS = 1570
CONTEXT_MARGIN = 128
JUDGE_TEMPERATURE = 0.1
JUDGE_TOP_P = 0.95
JUDGE_MAX_TOKENS = 1000
JUDGE_TRANSPORT_ATTEMPTS = 2


class SolarConsensusRunError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SolarConsensusRunError(message)


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


def mode_matrix(
    mode: str, rows: Sequence[SourceRow],
) -> tuple[list[SourceRow], list[AugmentationTask], int]:
    config_seed = int(prompt_config()["generation"]["seed"])
    if mode == "smoke":
        sources = stratified_sources(rows, SMOKE_SOURCE_COUNT, config_seed)
        tasks = [
            make_task(source, axis, source_index + 1)
            for source_index, source in enumerate(sources)
            for axis in AXES
        ]
        families = SMOKE_FAMILIES
    else:
        sources = stratified_sources(rows, PILOT_SOURCE_COUNT, config_seed)
        tasks = [
            make_task(source, axis, score)
            for source in sources for axis in AXES for score in TARGET_SCORES
        ]
        families = PILOT_FAMILIES
    need(len({source.identifier for source in sources}) == len(sources), "source sample differs")
    need(len({task.task_id for task in tasks}) == len(tasks), "task matrix differs")
    return sources, tasks, families


def source_sampling_summary(sources: Sequence[SourceRow]) -> dict[str, Any]:
    lengths = sorted(len(source.essay.strip()) for source in sources)
    return {
        "records": len(sources),
        "axis_rounded_score_counts": {
            axis: {
                str(score): sum(
                    min(5, max(1, int(math.floor(source.score[index] + 0.5)))) == score
                    for source in sources
                )
                for score in TARGET_SCORES
            }
            for index, axis in enumerate(AXES)
        },
        "essay_character_length": {
            "minimum": lengths[0],
            "median": lengths[len(lengths) // 2],
            "maximum": lengths[-1],
        },
    }


def audit_editor_context(tasks: Sequence[AugmentationTask], families: int) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(base.RUNTIME_MODEL), trust_remote_code=True, local_files_only=True
    )
    lengths: list[int] = []
    violations = 0
    for task in tasks:
        for slot in range(families):
            schema = editor_output_schema(task)
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "mal2026_solar_editor", "strict": True, "schema": schema},
            }
            encoded = tokenizer.apply_chat_template(
                render_editor_messages(task, slot % 4),
                tokenize=True,
                add_generation_prompt=True,
                reasoning_effort="none",
                think_render_option="preserved",
                response_format=response_format,
            )
            input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            length = len(input_ids)
            lengths.append(length)
            violations += length + EDITOR_MAX_TOKENS + CONTEXT_MARGIN > 4096
    need(lengths and violations == 0, "editor context preflight failed")
    return {
        "requests_audited": len(lengths),
        "local_prompt_tokens_min": min(lengths),
        "local_prompt_tokens_max": max(lengths),
        "editor_max_tokens": EDITOR_MAX_TOKENS,
        "reserved_runtime_token_margin": CONTEXT_MARGIN,
        "model_context_tokens": 4096,
        "violations": violations,
    }


def request_judge_draw(
    endpoint: str, prompt: str, essay: str, draw_index: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    messages = render_verifier_messages(prompt, essay)
    schema = base.verifier_output_schema()
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "mal2026_solar_verifier", "strict": True, "schema": schema},
    }
    seed = visible_draw_seed(
        messages, draw_index, int(prompt_config()["generation"]["seed"])
    )
    payload = {
        "model": base.MODEL_ALIAS,
        "messages": messages,
        "temperature": JUDGE_TEMPERATURE,
        "top_p": JUDGE_TOP_P,
        "max_tokens": JUDGE_MAX_TOKENS,
        "seed": seed,
        "response_format": response_format,
        "chat_template_kwargs": {
            "reasoning_effort": "none",
            "think_render_option": "preserved",
            "response_format": response_format,
        },
    }
    last: BaseException | None = None
    for _ in range(JUDGE_TRANSPORT_ATTEMPTS):
        try:
            response = base.http_json(endpoint + "/v1/chat/completions", payload, timeout=900)
            choices = response.get("choices")
            need(isinstance(choices, list) and len(choices) == 1, "judge choices differ")
            choice = choices[0]
            need(isinstance(choice, dict) and choice.get("finish_reason") != "length",
                 "judge output was truncated")
            message = choice.get("message")
            need(isinstance(message, dict) and isinstance(message.get("content"), str),
                 "judge content differs")
            return parse_verifier_output(message["content"]), seed
        except Exception as exc:  # bounded identical transport retry
            last = exc
    assert last is not None
    raise last


def score_values(verifier: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    return {axis: int(verifier[axis]["score"]) for axis in AXES}


def draw_record(
    endpoint: str, prompt: str, essay: str, draw_index: int,
) -> dict[str, Any]:
    try:
        verifier, seed = request_judge_draw(endpoint, prompt, essay, draw_index)
        return {
            "draw_index": draw_index,
            "status": "scored",
            "seed": seed,
            "score": score_values(verifier),
            "verifier": verifier,
        }
    except Exception as exc:
        return {
            "draw_index": draw_index,
            "status": "failed",
            "failure_category": base.gate_category(exc),
        }


def parallel_draws(
    endpoint: str,
    items: Sequence[tuple[str, str, str]],
    draw_indices: Sequence[int],
    max_inflight: int,
) -> dict[str, list[dict[str, Any]]]:
    """Score `(item_id, prompt, essay)` tuples without target metadata."""
    values: dict[str, list[dict[str, Any]]] = {item_id: [] for item_id, _, _ in items}
    with ThreadPoolExecutor(max_workers=max_inflight) as pool:
        futures = {
            pool.submit(draw_record, endpoint, prompt, essay, draw_index): item_id
            for item_id, prompt, essay in items for draw_index in draw_indices
        }
        for future in as_completed(futures):
            values[futures[future]].append(future.result())
    for item in values.values():
        item.sort(key=lambda row: int(row["draw_index"]))
    return values


def scored_only(draws: Sequence[Mapping[str, Any]]) -> list[Mapping[str, int]]:
    return [draw["score"] for draw in draws if draw.get("status") == "scored"]


def adaptive_score_pool(
    endpoint: str,
    items: Sequence[tuple[str, str, str]],
    initial_draw_zero: Mapping[str, Mapping[str, Any]] | None,
    max_inflight: int,
) -> dict[str, dict[str, Any]]:
    """Run exact 3-draw consensus, extending any disagreement to 5 draws."""
    controls: dict[str, dict[str, Any]] = {}
    if initial_draw_zero is None:
        first = parallel_draws(endpoint, items, range(JUDGE_DRAWS_INITIAL), max_inflight)
    else:
        remainder = parallel_draws(endpoint, items, (1, 2), max_inflight)
        first = {
            item_id: [dict(initial_draw_zero[item_id]), *remainder[item_id]]
            for item_id, _, _ in items
        }
    extra_ids: set[str] = set()
    for item_id, draws in first.items():
        scores = scored_only(draws)
        if len(scores) != JUDGE_DRAWS_INITIAL or requires_two_more_draws(scores):
            extra_ids.add(item_id)
    item_by_id = {item_id: (prompt, essay) for item_id, prompt, essay in items}
    extra_items = [(item_id, *item_by_id[item_id]) for item_id in sorted(extra_ids)]
    extra = parallel_draws(endpoint, extra_items, (3, 4), max_inflight) if extra_items else {}
    for item_id, draws in first.items():
        combined = [*draws, *extra.get(item_id, [])]
        scores = scored_only(combined)
        label = None
        support = 0
        distribution: dict[str, int] = {}
        if len(scores) in {JUDGE_DRAWS_INITIAL, JUDGE_DRAWS_MAXIMUM}:
            label, support, distribution = modal_label(scores)
        controls[item_id] = {
            "status": "stable" if label is not None else "unstable",
            "draws_requested": JUDGE_DRAWS_MAXIMUM if item_id in extra_ids else JUDGE_DRAWS_INITIAL,
            "draws_scored": len(scores),
            "draws": combined,
            "modal_score": label,
            "modal_support": support,
            "triplet_distribution": distribution,
            "extended_after_initial_disagreement": item_id in extra_ids,
        }
    return controls


def strict_count_class(
    raw_editor: str, task: AugmentationTask,
) -> tuple[bool, str | None]:
    try:
        parse_editor_output(raw_editor, task.source, task.target_axis, task.target_score)
        return True, None
    except Exception as exc:
        need(str(exc) == "editor substantive sentence edit count differs",
             "relaxed editor passed while a non-count strict gate failed")
        return False, str(exc)


def rejection(
    task: AugmentationTask, slot: int, stage: str, category: str,
    *, raw_output: str | None = None, essay: str | None = None,
    fidelity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = base.rejection(
        task, slot + 1, stage, category,
        raw_output=raw_output, essay=essay, fidelity=fidelity,
    )
    value.update({
        "schema_version": "mal2026-solar-consensus-rejection-v1",
        "candidate_slot": slot,
        "operation_family_index": slot % 4,
        "candidate_id": f"{task.task_id}::slot::{slot}",
    })
    return value


def generate_candidate(
    endpoint: str, task: AugmentationTask, slot: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    raw_editor: str | None = None
    essay: str | None = None
    try:
        raw_editor = base.request_content(
            endpoint,
            task,
            "editor",
            render_editor_messages(task, slot % 4),
            editor_output_schema(task),
            EDITOR_MAX_TOKENS,
            slot + 1,
        )
        essay = parse_editor_output(
            raw_editor,
            task.source,
            task.target_axis,
            task.target_score,
            enforce_score_specific_edit_count=False,
        )
        strict_count_pass, strict_count_failure = strict_count_class(raw_editor, task)
    except Exception as exc:
        return None, rejection(
            task, slot, "editor", base.gate_category(exc), raw_output=raw_editor
        )

    raw_fidelity: str | None = None
    fidelity: dict[str, bool] | None = None
    try:
        raw_fidelity = base.request_content(
            endpoint,
            task,
            "fidelity",
            render_fidelity_messages(task.source.essay, essay),
            base.fidelity_output_schema(),
            400,
            slot + 1,
        )
        fidelity = parse_fidelity_output(raw_fidelity)
        need(all(fidelity[key] is True for key in ("source_based", "topic", "stance", "genre")),
             "source fidelity failed")
        need(fidelity["new_external_facts_added"] is False, "new external facts were added")
    except Exception as exc:
        return None, rejection(
            task, slot, "fidelity", base.gate_category(exc), raw_output=raw_fidelity,
            essay=essay, fidelity=fidelity,
        )

    initial = draw_record(endpoint, task.source.prompt, essay, 0)
    if initial["status"] != "scored":
        return None, rejection(
            task, slot, "verifier", str(initial.get("failure_category", "judge_failure")),
            essay=essay,
        )
    verifier = initial["verifier"]
    try:
        validated = validate_actual_label_candidate(task, essay, verifier, fidelity)
    except Exception as exc:
        return None, rejection(
            task, slot, "validation", base.gate_category(exc), essay=essay, fidelity=fidelity
        )
    candidate_id = f"{task.task_id}::slot::{slot}"
    return {
        "schema_version": "mal2026-solar-consensus-candidate-v1",
        "candidate_id": candidate_id,
        "task_id": task.task_id,
        "source_id": task.source.identifier,
        "source_document_id": task.source.document_id,
        "source_essay_sha256": sha256(task.source.essay.encode()).hexdigest(),
        "candidate_essay_sha256": sha256(essay.encode()).hexdigest(),
        "candidate_slot": slot,
        "operation_family_index": slot % 4,
        "operation_family_replica": slot // 4,
        "requested_target_axis": task.target_axis,
        "requested_target_score": task.target_score,
        "requested_target_is_label": False,
        "prompt": task.source.prompt,
        "essay": essay,
        "editor_output": json.loads(raw_editor),
        "strict_score_specific_edit_count_pass": strict_count_pass,
        "strict_score_specific_edit_count_failure": strict_count_failure,
        "blind_fidelity": fidelity,
        "single_draw_score": validated["score"],
        "initial_judge_draw": initial,
    }, None


def axis_score_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        axis: {
            str(score): sum(int(record["score"][axis]) == score for record in records)
            for score in TARGET_SCORES
        }
        for axis in AXES
    }


def requested_cell_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for axis in AXES:
        for requested in TARGET_SCORES:
            cell = [record for record in records
                    if record["requested_target_axis"] == axis and
                    int(record["requested_target_score"]) == requested]
            distribution = Counter(int(record["score"][axis]) for record in cell)
            result[f"{axis}:{requested}"] = {
                "records": len(cell),
                "actual_target_distribution": {
                    str(score): distribution[score] for score in TARGET_SCORES
                },
                "actual_target_exact_requested": distribution[requested],
            }
    return result


def attach_modal_labels(
    candidates: Sequence[Mapping[str, Any]], controls: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stable: list[dict[str, Any]] = []
    unstable: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        control = controls[candidate_id]
        if control["status"] != "stable":
            unstable.append({
                "schema_version": "mal2026-solar-consensus-unstable-v1",
                "candidate_id": candidate_id,
                "task_id": candidate["task_id"],
                "candidate_essay_sha256": candidate["candidate_essay_sha256"],
                "strict_score_specific_edit_count_pass":
                    candidate["strict_score_specific_edit_count_pass"],
                "judge_control": control,
            })
            continue
        value = dict(candidate)
        value["score"] = control["modal_score"]
        value["score_provenance"] = {
            "all_axes": "target_blind_solar_adaptive_joint_triplet_modal",
            "prompt": "exact_evaluation_txt_system_and_user_sections",
            "requested_target": "generation_metadata_only_not_a_label",
            "draws": control["draws_requested"],
            "modal_support": control["modal_support"],
            "temperature": JUDGE_TEMPERATURE,
            "top_p": JUDGE_TOP_P,
        }
        value["judge_control"] = control
        stable.append(value)
    return stable, unstable


def aggregate(
    run_id: str, mode: str, sources: Sequence[SourceRow], tasks: Sequence[AugmentationTask],
    families: int, valid: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]],
    stable: Sequence[Mapping[str, Any]], unstable: Sequence[Mapping[str, Any]],
    source_controls: Mapping[str, Mapping[str, Any]], paths: Mapping[str, Path],
    bindings: Mapping[str, Any], context_audit: Mapping[str, Any],
) -> dict[str, Any]:
    failures = Counter(f"{row['stage']}:{row['category']}" for row in rejected)
    supports = Counter(str(row["judge_control"]["modal_support"]) for row in stable)
    source_stable = sum(control["status"] == "stable" for control in source_controls.values())
    expected = len(tasks) * families
    return {
        "schema_version": "mal2026-solar-consensus-pilot-result-v1",
        "status": "completed",
        "run_id": run_id,
        "mode": mode,
        "completed_at": now(),
        "sources": len(sources),
        "source_sampling": source_sampling_summary(sources),
        "tasks": len(tasks),
        "candidate_slots_per_task": families,
        "candidate_attempts_expected": expected,
        "candidate_attempts_accounted": len(valid) + len(rejected),
        "hard_filter_valid_before_adaptive_judge": len(valid),
        "hard_filter_rejected": len(rejected),
        "stable_modal_candidates": len(stable),
        "unstable_judge_candidates": len(unstable),
        "strict_edit_count_compatible": sum(
            row["strict_score_specific_edit_count_pass"] is True for row in valid
        ),
        "relaxed_edit_count_only": sum(
            row["strict_score_specific_edit_count_pass"] is False for row in valid
        ),
        "source_modal_labels": {"attempted": len(source_controls), "stable": source_stable},
        "stable_modal_support_counts": dict(sorted(supports.items())),
        "stable_axis_score_counts": axis_score_counts(stable),
        "stable_requested_cell_summary": requested_cell_summary(stable),
        "unique_candidate_essay_hashes": len({row["candidate_essay_sha256"] for row in valid}),
        "duplicate_candidate_essays": len(valid) - len({row["candidate_essay_sha256"] for row in valid}),
        "failure_counts": dict(sorted(failures.items())),
        "context_preflight": dict(context_audit),
        "artifact_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "bindings": dict(bindings),
        "protocol": {
            "train_only": True,
            "validation_used": False,
            "requested_target_is_label": False,
            "official_evaluation_txt_prompt_exact": True,
            "hard_fidelity_before_first_judge": True,
            "adaptive_draws": "3_if_unanimous_else_5_unique_triplet_majority_at_least_3",
            "judge_temperature": JUDGE_TEMPERATURE,
            "judge_top_p": JUDGE_TOP_P,
            "judge_seed_uses_only_visible_prompt_essay_and_draw_index": True,
            "judge_feedback_passed_to_generator": False,
            "judge_score_used_for_candidate_generation_or_selection": False,
            "all_hard_filter_valid_candidates_preserved": True,
            "encoder_consensus_pending": mode == "pilot",
            "full_augmentation_or_training_authorized_by_this_result": False,
        },
        "automatic_checks": {
            "all_attempts_accounted": len(valid) + len(rejected) == expected,
            "all_source_modal_labels_stable": source_stable == len(source_controls),
            "no_exact_duplicate_candidate_essays":
                len(valid) == len({row["candidate_essay_sha256"] for row in valid}),
        },
        "privacy": "aggregate contains no essay, prompt, rationale, identifier, or individual row",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--port", type=int, default=19420)
    parser.add_argument("--max-inflight", type=int, default=MAX_INFLIGHT)
    parser.add_argument("--external-endpoint", required=True)
    parser.add_argument("--external-container-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    need(re.fullmatch(r"[a-z0-9][a-z0-9-]{7,95}", args.run_id) is not None,
         "run ID differs")
    need(args.max_inflight == MAX_INFLIGHT, "concurrency protocol differs")
    need(args.external_endpoint == f"http://127.0.0.1:{args.port}", "endpoint differs")
    config = prompt_config()
    need(config["execution_gate"]["full_run_authorized"] is False,
         "strict full-run gate must remain disabled")
    rows = load_train_rows()
    sources, tasks, families = mode_matrix(args.mode, rows)
    context_audit = audit_editor_context(tasks, families)
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
        "validation_sha256_lineage_only": config["provenance"]["validation_source_sha256"],
        "model": base.verify_model(),
        "image": base.docker_image_binding(),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "helper_sha256": file_sha256(ROOT / "src/mal2026/solar_consensus_pilot.py"),
        "augmentation_core_sha256": file_sha256(ROOT / "src/mal2026/solar_target_augmentation.py"),
    }
    manifest = {
        "schema_version": "mal2026-solar-consensus-pilot-manifest-v1",
        "status": "preflight",
        "run_id": args.run_id,
        "mode": args.mode,
        "created_at": now(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree_dirty_at_launch": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
        "command": [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *sys.argv[1:]],
        "gpu_scope": list(base.GPU_SCOPE),
        "gpu_authorization": "repository default physical GPUs0-3; GPUs4-7 not queried or used",
        "scientific_authorization": (
            "user approved generate-filter-Solar-exact-prompt-encoder-consensus protocol "
            "on 2026-07-30"
        ),
        "source_split": "train_only",
        "validation_used": False,
        "sources": len(sources),
        "tasks": len(tasks),
        "families": families,
        "candidate_attempts": len(tasks) * families,
        "context_preflight": context_audit,
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
        source_items = [(source.identifier, source.prompt, source.essay) for source in sources]
        source_controls = adaptive_score_pool(
            args.external_endpoint, source_items, None, args.max_inflight
        )
        source_controls_path = restricted / "source_judge_controls.jsonl"
        write_jsonl(source_controls_path, [
            {"source_id": source_id, **source_controls[source_id]}
            for source_id in sorted(source_controls)
        ])

        manifest.update({"status": "generating_and_hard_filtering", "source_labels_at": now()})
        atomic_json(output / "manifest.json", manifest)
        pairs = [(task, slot) for task in tasks for slot in range(families)]
        valid: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
            futures = {
                pool.submit(generate_candidate, args.external_endpoint, task, slot): (task, slot)
                for task, slot in pairs
            }
            for future in as_completed(futures):
                task, slot = futures[future]
                try:
                    record, failed = future.result()
                except Exception as exc:
                    record, failed = None, rejection(
                        task, slot, "worker", base.gate_category(exc)
                    )
                if record is not None:
                    valid.append(record)
                if failed is not None:
                    rejected.append(failed)
        need(len(valid) + len(rejected) == len(pairs), "candidate population differs")
        valid.sort(key=lambda row: row["candidate_id"])
        rejected.sort(key=lambda row: row["candidate_id"])
        initial_valid_path = restricted / "hard_filter_valid_initial_draw.jsonl"
        rejected_path = restricted / "hard_filter_rejected.jsonl"
        write_jsonl(initial_valid_path, valid)
        write_jsonl(rejected_path, rejected)

        manifest.update({
            "status": "adaptive_judging",
            "hard_filter_valid": len(valid),
            "hard_filter_rejected": len(rejected),
        })
        atomic_json(output / "manifest.json", manifest)
        candidate_items = [
            (row["candidate_id"], row["prompt"], row["essay"]) for row in valid
        ]
        initial_draws = {row["candidate_id"]: row["initial_judge_draw"] for row in valid}
        controls = adaptive_score_pool(
            args.external_endpoint, candidate_items, initial_draws, args.max_inflight
        )
        judge_controls_path = restricted / "candidate_judge_controls.jsonl"
        write_jsonl(judge_controls_path, [
            {"candidate_id": candidate_id, **controls[candidate_id]}
            for candidate_id in sorted(controls)
        ])
        stable, unstable = attach_modal_labels(valid, controls)
        stable_path = restricted / "stable_modal_candidates.jsonl"
        unstable_path = restricted / "unstable_judge_candidates.jsonl"
        write_jsonl(stable_path, stable)
        write_jsonl(unstable_path, unstable)
        paths = {
            "source_judge_controls": source_controls_path,
            "hard_filter_valid_initial_draw": initial_valid_path,
            "hard_filter_rejected": rejected_path,
            "candidate_judge_controls": judge_controls_path,
            "stable_modal_candidates": stable_path,
            "unstable_judge_candidates": unstable_path,
        }
        result = aggregate(
            args.run_id, args.mode, sources, tasks, families, valid, rejected,
            stable, unstable, source_controls, paths, bindings, context_audit,
        )
        atomic_json(output / "result.json", result)
        manifest.update({
            "status": "completed",
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
