#!/usr/bin/env python3
"""Generate train-only official DPO preferences in three GPU-separable stages.

``rollout`` uses an external vLLM 0.25.1 HTTP server, ``judge`` uses only the
pinned Q4 llama-server, and ``assemble`` excludes ties and writes TRL-ready
conversational pairs.  The staged interface prevents policy and judge servers
from silently overcommitting the authorized GPUs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import sha256_file  # noqa: E402
from mal2026.official_rationale_data import (  # noqa: E402
    OfficialRationaleDataError,
    axes_for_task,
    parse_rationale_output,
    rationale_schema,
)
from mal2026.official_rationale_rl import (  # noqa: E402
    AXES,
    RLSettings,
    completion_text,
    http_json,
    judge_total,
    official_train_rows,
    output_fresh,
    participant,
    q4_score,
    restricted_fresh,
    select_preference,
    select_axis_preference,
    validate_policy_attestation,
    validate_q4_attestation,
    validate_runtime_versions,
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_aliases(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        task, separator, alias = value.partition("=")
        need(separator == "=" and task in {"bundle", *AXES} and bool(alias) and task not in result, "--model must be unique TASK=ALIAS")
        result[task] = alias
    return result


def policy_request(
    endpoint: str,
    alias: str,
    task: str,
    prompt: list[dict[str, str]],
    candidates: int,
    settings: RLSettings,
    seed: int,
) -> tuple[list[dict[str, str]], int, int]:
    axes = axes_for_task(task)
    body = {
        "model": alias,
        "messages": prompt,
        "n": candidates,
        "temperature": settings.policy["sampling_temperature"],
        "top_p": settings.policy["sampling_top_p"],
        "seed": seed,
        "max_tokens": 900 if task == "bundle" else 350,
        "response_format": {"type": "json_schema", "json_schema": {"name": f"official_dpo_{task}", "strict": True, "schema": rationale_schema(axes)}},
    }
    outer = http_json(endpoint, body)
    choices = outer.get("choices")
    need(isinstance(choices, list) and len(choices) == candidates, "policy rollout choice count differs")
    parsed: list[dict[str, str]] = []
    relaxed_control_character_parses = 0
    schema_complete_length_finishes = 0
    for choice in choices:
        need(isinstance(choice, dict), "policy rollout choice differs")
        finish_reason = choice.get("finish_reason")
        need(finish_reason in {"stop", "length"}, "policy rollout finish reason differs")
        content = choice["message"]["content"]
        try:
            parsed.append(parse_rationale_output(content, axes))
        except OfficialRationaleDataError:
            # vLLM's JSON-schema decoder can occasionally serialize a literal
            # control character inside a JSON string.  Python's strict parser
            # rejects that wire representation even though the schema and
            # semantic string are otherwise valid.  ``strict=False`` changes
            # only control-character acceptance; the normal strict rationale
            # shape validator still runs on the decoded object.
            decoded = json.loads(content, strict=False)
            parsed.append(parse_rationale_output(decoded, axes))
            relaxed_control_character_parses += 1
        if finish_reason == "length":
            # A schema-constrained completion may end exactly on the configured
            # token boundary after emitting a complete JSON object.  Accept it
            # only after the unchanged rationale parser has validated the full
            # object; truncated or malformed length finishes still fail closed.
            schema_complete_length_finishes += 1
    return parsed, relaxed_control_character_parses, schema_complete_length_finishes


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    need(path.is_file() and not path.is_symlink(), "input JSONL is unavailable")
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            need(bool(line.strip()), "input JSONL has a blank row")
            raw = json.loads(line)
            need(isinstance(raw, dict), "input JSONL row differs")
            result.append(raw)
    return result


def rollout(args: argparse.Namespace, settings: RLSettings, gate: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    required = {"bundle"} if args.arm == "bundle" else set(AXES)
    need(set(aliases) == required, "policy aliases differ from selected arm")
    validate_policy_attestation(args.policy_attestation, args.policy_endpoint, aliases)
    rows, provenance = official_train_rows("bundle", args.limit)
    prompts_by_task = {"bundle": [row["prompt"] for row in rows]}
    for task in sorted(required - {"bundle"}):
        task_rows, _ = official_train_rows(task, args.limit)
        need([(row["source_id"], row["candidate_number"]) for row in task_rows] == [(row["source_id"], row["candidate_number"]) for row in rows], "task prompt source order differs")
        prompts_by_task[task] = [row["prompt"] for row in task_rows]
    candidate_count = int(settings.policy["candidates_per_prompt"])

    def one(index_row: tuple[int, Mapping[str, Any]]) -> dict[str, Any]:
        index, row = index_row
        generations: list[dict[str, str]] = [dict() for _ in range(candidate_count)]
        relaxed_control_character_parses = 0
        schema_complete_length_finishes = 0
        for task in sorted(required):
            task_prompt = prompts_by_task[task][index]
            outputs, relaxed, complete_length = policy_request(
                args.policy_endpoint,
                aliases[task],
                task,
                task_prompt,
                candidate_count,
                settings,
                int(settings.policy["seed"]) + index,
            )
            relaxed_control_character_parses += relaxed
            schema_complete_length_finishes += complete_length
            for candidate_index, parsed in enumerate(outputs):
                generations[candidate_index].update(parsed)
        need(all(set(value) == set(AXES) for value in generations), "rollout did not produce all axes")
        return {
            "schema_version": "mal2026-official-rationale-rollout-group-v1",
            "split": "train",
            "arm": args.arm,
            "source_id": row["source_id"],
            "candidate_number": row["candidate_number"],
            "source_key": row["source_key"],
            "scores": row["scores"],
            "generations": generations,
            "relaxed_control_character_json_parses": relaxed_control_character_parses,
            "schema_complete_length_finishes": schema_complete_length_finishes,
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
        futures = [pool.submit(one, pair) for pair in enumerate(rows)]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (str(row["source_id"]), int(row["candidate_number"])))
    output = restricted_fresh(args.output)
    write_jsonl(output, results)
    return {
        "schema_version": "mal2026-official-rationale-rollout-aggregate-v1",
        "status": "completed",
        "stage": "rollout",
        "arm": args.arm,
        "groups": len(results),
        "candidates": len(results) * candidate_count,
        "relaxed_control_character_json_parses": sum(int(row["relaxed_control_character_json_parses"]) for row in results),
        "relaxed_parse_semantics": "json.loads(strict=False) only after strict parse failure; strict rationale schema validation still required",
        "schema_complete_length_finishes": sum(int(row["schema_complete_length_finishes"]) for row in results),
        "length_finish_semantics": "accepted only when the unchanged strict rationale schema parser validates the complete JSON object",
        "raw_sha256": sha256_file(output),
        "input_provenance": provenance,
        "contrastive_gate_sha256": gate["directional"]["sha256"],
        "rl_safety_gate_sha256": gate["combined_safety"]["sha256"],
        "policy_attestation_sha256": sha256_file(args.policy_attestation),
        "split": "train",
        "validation_used": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_or_predictions",
    }


def judge(args: argparse.Namespace, settings: RLSettings, gate: Mapping[str, Any]) -> dict[str, Any]:
    validate_q4_attestation(args.judge_attestation, args.judge_endpoint, settings.judge["prompt_sha256"])
    raw = read_jsonl(args.input)
    need(bool(raw) and all(row.get("schema_version") == "mal2026-official-rationale-rollout-group-v1" and row.get("split") == "train" and row.get("arm") == args.arm for row in raw), "rollout input contract differs")
    canonical, _ = official_train_rows("bundle", len(raw))
    lookup = {(row["source_id"], row["candidate_number"]): row for row in canonical}

    def one(item: tuple[int, Mapping[str, Any]]) -> dict[str, Any]:
        row_index, row = item
        source = lookup.get((row["source_id"], row["candidate_number"]))
        need(source is not None and source["source_key"] == row["source_key"] and source["scores"] == row["scores"], "rollout source linkage differs")
        judged: list[dict[str, Any]] = []
        for candidate_index, rationales in enumerate(row["generations"]):
            candidate = participant(row["scores"], rationales)
            output = q4_score(
                args.judge_endpoint[row_index % len(args.judge_endpoint)],
                settings.judge["model_alias"],
                source["prompt_text"],
                source["essay_text"],
                candidate,
                system_prompt=settings.judge_system_prompt(),
            )
            judged.append({"rationales": rationales, "judge_output": output, "judge_total": judge_total(output), "candidate_index": candidate_index})
        return {**row, "schema_version": "mal2026-official-rationale-judged-group-v1", "judged_generations": judged, "generations": None}

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
        futures = [pool.submit(one, pair) for pair in enumerate(raw)]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (str(row["source_id"]), int(row["candidate_number"])))
    output = restricted_fresh(args.output)
    write_jsonl(output, results)
    totals = [item["judge_total"] for row in results for item in row["judged_generations"]]
    return {
        "schema_version": "mal2026-official-rationale-judged-rollout-aggregate-v1",
        "status": "completed",
        "stage": "judge",
        "arm": args.arm,
        "groups": len(results),
        "judgments": len(totals),
        "judge_total_mean": sum(totals) / len(totals),
        "judge_total_population_std": statistics.pstdev(totals) if len(totals) > 1 else 0.0,
        "raw_sha256": sha256_file(output),
        "rollout_sha256": sha256_file(args.input),
        "contrastive_gate_sha256": gate["directional"]["sha256"],
        "rl_safety_gate_sha256": gate["combined_safety"]["sha256"],
        "judge_attestation_sha256": sha256_file(args.judge_attestation),
        "judge_model_sha256": settings.judge["model_sha256"],
        "judge_prompt_sha256": settings.judge["prompt_sha256"],
        "split": "train",
        "validation_used": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_predictions",
    }


def assemble(args: argparse.Namespace, settings: RLSettings, gate: Mapping[str, Any]) -> dict[str, Any]:
    raw = read_jsonl(args.input)
    need(bool(raw) and all(row.get("schema_version") == "mal2026-official-rationale-judged-group-v1" and row.get("split") == "train" and row.get("arm") == args.arm for row in raw), "judged input contract differs")
    source_rows, _ = official_train_rows("bundle", len(raw))
    source_lookup = {(row["source_id"], row["candidate_number"]): row for row in source_rows}
    preferences: list[dict[str, Any]] = []
    ties: Counter[str] = Counter()
    differences: dict[str, list[int]] = {task: [] for task in (("bundle",) if args.arm == "bundle" else AXES)}
    retained: Counter[str] = Counter()
    tasks = ("bundle",) if args.arm == "bundle" else AXES
    task_prompt_lookups: dict[str, dict[tuple[str, int], list[dict[str, str]]]] = {}
    for task in tasks:
        task_rows, _ = official_train_rows(task, len(raw))
        task_prompt_lookups[task] = {(item["source_id"], item["candidate_number"]): item["prompt"] for item in task_rows}
    for row in raw:
        source = source_lookup.get((row["source_id"], row["candidate_number"]))
        need(source is not None and source["source_key"] == row["source_key"] and source["scores"] == row["scores"], "judged source linkage differs")
        scored = [(json.dumps(item["rationales"], ensure_ascii=False, separators=(",", ":")), item["judge_output"]) for item in row["judged_generations"]]
        for task in tasks:
            selected = (
                select_preference(scored, int(settings.reward["preference_minimum_total_difference"]))
                if task == "bundle"
                else select_axis_preference(scored, task, int(settings.reward["preference_minimum_total_difference"]))
            )
            if selected is None:
                ties[task] += 1
                continue
            chosen_all = json.loads(selected["chosen"])
            rejected_all = json.loads(selected["rejected"])
            prompt = task_prompt_lookups[task][(row["source_id"], row["candidate_number"])]
            chosen = completion_text(chosen_all, task)
            rejected = completion_text(rejected_all, task)
            if chosen == rejected:
                ties[task] += 1
                continue
            preference = {
                "schema_version": "mal2026-official-rationale-preference-v1",
                "split": "train",
                "arm": args.arm,
                "task": task,
                "source_key": row["source_key"],
                "candidate_number": row["candidate_number"],
                "score_kind": "frozen_api_emitted_integer_prediction",
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            }
            if task == "bundle":
                preference.update({
                    "chosen_judge_total": selected["chosen_judge_total"],
                    "rejected_judge_total": selected["rejected_judge_total"],
                    "judge_total_difference": selected["judge_total_difference"],
                    "selection_projection": "sum_of_all_12_integer_cells",
                })
                differences[task].append(selected["judge_total_difference"])
            else:
                preference.update({
                    "chosen_axis_judge_total": selected["chosen_axis_judge_total"],
                    "rejected_axis_judge_total": selected["rejected_axis_judge_total"],
                    "axis_judge_total_difference": selected["axis_judge_total_difference"],
                    "selection_projection": f"sum_of_4_integer_cells_for_{task}",
                })
                differences[task].append(selected["axis_judge_total_difference"])
            preferences.append(preference)
            retained[task] += 1
    output = restricted_fresh(args.output)
    write_jsonl(output, preferences)
    group_count = len(raw)
    threshold = float(settings.reward["max_zero_variance_group_fraction"])
    per_task = {
        task: {
            "groups": group_count,
            "preference_rows": int(retained[task]),
            "zero_variance_groups_excluded": int(ties[task]),
            "zero_variance_group_fraction": ties[task] / group_count,
            "passed": ties[task] / group_count <= threshold,
            "observed_difference_min": min(differences[task]) if differences[task] else None,
            "observed_difference_mean": sum(differences[task]) / len(differences[task]) if differences[task] else None,
            "selection_projection": "sum_of_all_12_integer_cells" if task == "bundle" else f"sum_of_4_integer_cells_for_{task}",
        }
        for task in tasks
    }
    need(all(item["preference_rows"] + item["zero_variance_groups_excluded"] == group_count for item in per_task.values()), "per-task preference accounting differs")
    passed = all(item["passed"] for item in per_task.values())
    return {
        "schema_version": "mal2026-official-rationale-preference-aggregate-v1",
        "status": "completed" if passed else "failed_reward_variance_gate",
        "stage": "assemble",
        "arm": args.arm,
        "groups": group_count,
        "preference_rows": len(preferences),
        "tasks_per_retained_group": len(tasks),
        "per_task_reward_variance": per_task,
        "zero_variance_group_fraction": max(item["zero_variance_group_fraction"] for item in per_task.values()),
        "max_zero_variance_group_fraction": settings.reward["max_zero_variance_group_fraction"],
        "reward_variance_gate_passed": passed,
        "minimum_total_difference": settings.reward["preference_minimum_total_difference"],
        "minimum_total_difference_basis": settings.reward["preference_margin_basis"],
        "raw_sha256": sha256_file(output),
        "judged_rollout_sha256": sha256_file(args.input),
        "contrastive_gate_sha256": gate["directional"]["sha256"],
        "rl_safety_gate_sha256": gate["combined_safety"]["sha256"],
        "judge_model_sha256": settings.judge["model_sha256"],
        "judge_prompt_sha256": settings.judge["prompt_sha256"],
        "split": "train",
        "validation_used": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_predictions",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/official_rationale_dpo.v1.json")
    parser.add_argument("--stage", choices=("rollout", "judge", "assemble"))
    parser.add_argument("--arm", choices=("bundle", "axis_triplet"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--aggregate-output", type=Path)
    parser.add_argument("--policy-endpoint")
    parser.add_argument("--policy-attestation", type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--judge-endpoint", action="append", default=[])
    parser.add_argument("--judge-attestation", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    settings = RLSettings.from_json(args.config)
    validate_runtime_versions()
    if args.validate_only:
        print(json.dumps({"status": "validated", "algorithm": settings.algorithm, "gate_ready": settings.gate_evidence()}, sort_keys=True))
        return
    need(args.stage is not None and args.arm is not None and args.output is not None and args.aggregate_output is not None and args.max_inflight >= 1, "stage arguments are incomplete")
    gate = settings.gate_evidence()
    aggregate_path = output_fresh(args.aggregate_output)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "rollout":
        need(args.policy_endpoint and args.policy_attestation, "rollout policy arguments are incomplete")
        report = rollout(args, settings, gate, parse_aliases(args.model))
    elif args.stage == "judge":
        need(args.input and args.judge_endpoint and args.judge_attestation, "judge arguments are incomplete")
        report = judge(args, settings, gate)
    else:
        need(args.input, "assemble input is absent")
        report = assemble(args, settings, gate)
    report.update({"created_at": now(), "config_sha256": sha256_file(args.config)})
    atomic_json(aggregate_path, report)
    print(json.dumps({"status": report["status"], "stage": args.stage, "arm": args.arm}, sort_keys=True))
    if report["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
