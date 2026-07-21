#!/usr/bin/env python3
"""Calibrate score-only rationale judging with deterministic contrastive controls.

The program creates controls only in memory and persists aggregate-safe opaque
observations.  It never writes essays, rationales, source identifiers, raw
prompts, completions, or source writing scores.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import statistics
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
RESTRICTED = ROOT / "data/processed/restricted/openai_rationale_batches"
RUN_PREFIX = "qwen36-native-fp8-rationale-contrastive-v1-"
SCHEMA = "mal2026-rationale-contrastive-validity-v1"
AXES = ("content", "organization", "expression")
CONDITIONS = ("original", "generic", "axis_rotation", "cross_essay")
THRESHOLDS = {"generic": 0.75, "axis_rotation": 0.60, "cross_essay": 0.60}


def load_judge() -> Any:
    spec = importlib.util.spec_from_file_location("mal2026_distribution_judge", ROOT / "scripts" / "score_rationale_distribution_vllm_dp4.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the canonical rationale judge")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JUDGE = load_judge()


def destination(cfg: dict[str, Any], run_id: str) -> Path:
    if not __import__("re").fullmatch(rf"{__import__('re').escape(RUN_PREFIX)}train-20260720-(?:gpu0_smoke|full)-[0-9]{{3}}", run_id):
        raise RuntimeError("contrastive run id is outside the declared lineage")
    return RESTRICTED / cfg["inputs"]["batch_run_id"] / "judge_contrastive_validity_v1" / run_id


def canonical_set_digest(values: set[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8")); digest.update(b"\n")
    return digest.hexdigest()


def generic_rationale() -> dict[str, Any]:
    return {"schema_version": "rationale-only-v1", "content": {"rationale": "이 글의 내용 측면에는 장점과 보완할 부분이 함께 있다."},
            "organization": {"rationale": "이 글의 구성 측면에는 장점과 보완할 부분이 함께 있다."},
            "expression": {"rationale": "이 글의 표현 측면에는 장점과 보완할 부분이 함께 있다."}}


def rotated_rationale(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != "rationale-only-v1":
        raise RuntimeError("rationale-only projection is required for contrastive rotation")
    rotation = {"content": "organization", "organization": "expression", "expression": "content"}
    return {"schema_version": "rationale-only-v1", **{axis: {"rationale": value[rotation[axis]]["rationale"]} for axis in AXES}}


def calibrated_population(cfg: dict[str, Any], source_limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read train essays/candidates; never access the source writing-score field."""
    candidate_path, candidate_manifest, parent_path = JUDGE.validate_candidate_artifact(cfg, "train")
    spec = cfg["inputs"]["train"]
    source_path = ROOT / spec["source_file"]
    if not source_path.is_file() or source_path.is_symlink():
        raise RuntimeError("canonical train source is unavailable")
    sources: dict[str, dict[str, Any]] = {}
    for row in JUDGE.load_jsonl(source_path):
        source_id, essay = str(row.get("id")), row.get("essay")
        # Deliberately do not access row["score"] or any related field.
        if source_id in sources or not isinstance(essay, str):
            raise RuntimeError("canonical train source routing is invalid")
        sources[source_id] = {"sentences": JUDGE.sentence_list(essay), "essay_sha256": hashlib.sha256(essay.encode()).hexdigest()}
    if len(sources) != spec["expected_essays"]:
        raise RuntimeError("canonical train source count changed")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid = 0
    for row in JUDGE.load_jsonl(candidate_path):
        source_id = str(row.get("source_id")); source = sources.get(source_id)
        if (row.get("split") != "train" or source is None or not isinstance(row.get("custom_id"), str) or
                row.get("essay_sha256") != source["essay_sha256"] or not JUDGE.candidate_valid(row.get("rationale"), len(source["sentences"]))):
            invalid += 1; continue
        grouped[source_id].append(row)
    if invalid or len(grouped) != spec["expected_essays"]:
        raise RuntimeError("contrastive population source/candidate validation failed")
    available: list[tuple[str, dict[int, dict[str, Any]]]] = []
    for source_id, rows in grouped.items():
        by_number = {row.get("candidate"): row for row in rows}
        if set(by_number) != {1, 2, 3} or len(by_number) != 3:
            raise RuntimeError("candidate group is incomplete")
        available.append((source_id, by_number))
    available.sort(key=lambda item: JUDGE.opaque(cfg["seed"], "contrastive-validity-v1", item[0]))
    if source_limit not in {1, 80} or source_limit > len(available):
        raise RuntimeError("source limit is not an approved contrastive population")
    selected = available[:source_limit]
    records: list[dict[str, Any]] = []
    selected_ids = {source_id for source_id, _ in selected}
    for index, (source_id, by_number) in enumerate(selected):
        foreign_id, _ = selected[(index + 1) % len(selected)] if len(selected) > 1 else available[source_limit]
        for number in (1, 2, 3):
            raw = by_number[number]
            rationale = JUDGE.project_candidate(raw["rationale"], cfg)
            if rationale.get("schema_version") != "rationale-only-v1":
                raise RuntimeError("contrastive protocol requires rationale-only projection")
            records.append({"base_key": JUDGE.opaque("contrastive-validity-v1", raw["custom_id"]), "candidate_number": number,
                            "sentences": sources[source_id]["sentences"], "foreign_sentences": sources[foreign_id]["sentences"],
                            "rationale": rationale})
    if len(records) != source_limit * 3:
        raise RuntimeError("contrastive population count changed")
    provenance = {"candidate_file_sha256": JUDGE.sha(candidate_path), "candidate_manifest_sha256": JUDGE.sha(RESTRICTED / cfg["inputs"]["batch_run_id"] / spec["candidate_manifest"]),
                  "parent_manifest_sha256": JUDGE.sha(parent_path), "source_file_sha256": JUDGE.sha(source_path),
                  "calibration_source_groups": source_limit, "calibration_candidates": len(records),
                  "calibration_source_id_set_sha256": canonical_set_digest(selected_ids), "invalid_source_or_candidate_rows": invalid,
                  "source_writing_scores_read_or_prompted": False, "validation_source_text_opened": 0}
    return records, provenance


def variants(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for record in records:
        original = copy.deepcopy(record["rationale"])
        by_condition = {
            "original": (record["sentences"], original),
            "generic": (record["sentences"], generic_rationale()),
            "axis_rotation": (record["sentences"], rotated_rationale(original)),
            "cross_essay": (record["foreign_sentences"], original),
        }
        for condition in CONDITIONS:
            sentences, rationale = by_condition[condition]
            values.append({"base_key": record["base_key"], "candidate_number": record["candidate_number"], "condition": condition,
                           "sentences": sentences, "rationale": rationale})
    return values


def task_stream(cfg: dict[str, Any], run_id: str, model: str, values: list[dict[str, Any]], done: set[str]) -> Iterator[dict[str, Any]]:
    prompt_types, seeds = cfg["protocol"]["prompt_types"], cfg["sampling"]["seeds"]
    for value in values:
        for prompt_index, prompt_type in enumerate(prompt_types):
            for seed_index, seed in enumerate(seeds):
                request_key = JUDGE.opaque(run_id, value["base_key"], value["condition"], prompt_index, seed_index)
                if request_key in done:
                    continue
                yield {"opaque_request_key": request_key, "opaque_base_key": value["base_key"], "candidate_number": value["candidate_number"],
                       "condition": value["condition"], "prompt_type_id": prompt_type["id"], "sampling_seed": seed,
                       "response_contract": cfg["protocol"]["response_contract"],
                       "body": JUDGE.request_body(cfg, model, value, list(AXES), prompt_type["layout"], seed, prompt_type["review_emphasis"])}


def validate_server(attestation_path: Path, endpoint: str, cfg: dict[str, Any], phase: str) -> None:
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None or not attestation_path.is_file() or attestation_path.is_symlink():
        raise RuntimeError("contrastive endpoint or attestation is invalid")
    value = json.loads(attestation_path.read_text(encoding="utf-8"))
    expected_gpus, expected_dp = ([0], 1) if phase == "gpu0_smoke" else ([0, 1, 2, 3], 4)
    if (value.get("schema_version") != "mal2026-rationale-contrastive-v1-server-attestation-v1" or value.get("config_sha256") != JUDGE.sha(JUDGE.CONFIG_PATH) or
            value.get("server_host") != "127.0.0.1" or value.get("server_port") != parsed.port or value.get("physical_gpus") != expected_gpus or
            value.get("tensor_parallel_size") != 1 or value.get("data_parallel_size") != expected_dp or value.get("max_model_len") != cfg["runtime"]["context_size"] or
            value.get("max_num_seqs_per_dp_rank") != cfg["runtime"]["max_num_seqs_per_dp_rank"] or value.get("server_process_environment_verified") is not True):
        raise RuntimeError("contrastive server attestation does not bind the declared runtime")


def wilson(successes: int, total: int) -> list[float]:
    if total <= 0: return [None, None]
    z = 1.959963984540054; p = successes / total; denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [round(max(0.0, centre - radius), 6), round(min(1.0, centre + radius), 6)]


def summarize(destination_path: Path, expected_calls: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in (destination_path / "score_observations.jsonl").open(encoding="utf-8") if line.strip()]
    failures = Counter(str(row["failure_category"]) for row in rows if row.get("failure_category"))
    if len(rows) != expected_calls:
        raise RuntimeError("contrastive observations are incomplete")
    keyed: dict[tuple[str, str, str, int], dict[str, dict[str, int]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["opaque_base_key"]), str(row["prompt_type_id"]), int(row["sampling_seed"]), str(row["condition"]))
        group_key, condition = key[:-1], key[-1]
        if condition not in CONDITIONS or condition in keyed[group_key]:
            raise RuntimeError("contrastive observations are malformed")
        if not row.get("scored") or not isinstance(row.get("scores"), dict):
            continue
        keyed[group_key][condition] = row["scores"]
    pair_rows: dict[str, dict[str, list[tuple[str, dict[str, int], dict[str, int]]]]] = {condition: {axis: [] for axis in AXES} for condition in CONDITIONS[1:]}
    per_candidate: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    missing_pairs = 0
    for (base_key, prompt_type_id, seed), conditions in keyed.items():
        original = conditions.get("original")
        for condition in CONDITIONS[1:]:
            negative = conditions.get(condition)
            if original is None or negative is None:
                missing_pairs += 1; continue
            for axis in AXES:
                margin = float(original[axis] - negative[axis])
                pair_rows[condition][axis].append((base_key, original, negative))
                per_candidate[(base_key, condition, axis)].append(margin)
    transforms: dict[str, Any] = {}; validity_passed = not failures and missing_pairs == 0
    for condition in CONDITIONS[1:]:
        axis_result: dict[str, Any] = {}
        for axis in AXES:
            margins = [float(original[axis] - negative[axis]) for _, original, negative in pair_rows[condition][axis]]
            candidate_margins = [statistics.fmean(values) for (base, item_condition, item_axis), values in per_candidate.items() if item_condition == condition and item_axis == axis]
            strict = sum(value > 0 for value in candidate_margins); ties = sum(value == 0 for value in candidate_margins); losses = sum(value < 0 for value in candidate_margins)
            rate = strict / len(candidate_margins) if candidate_margins else 0.0
            passes = rate >= THRESHOLDS[condition]
            validity_passed = validity_passed and passes
            axis_result[axis] = {"paired_observations": len(margins), "observation_mean_margin": round(statistics.fmean(margins), 6) if margins else None,
                                 "candidate_pairs": len(candidate_margins), "candidate_strict_wins": strict, "candidate_ties": ties, "candidate_losses": losses,
                                 "candidate_strict_win_rate": round(rate, 6), "candidate_strict_win_wilson95": wilson(strict, len(candidate_margins)),
                                 "candidate_mean_margin": round(statistics.fmean(candidate_margins), 6) if candidate_margins else None,
                                 "threshold": THRESHOLDS[condition], "passes_predeclared_threshold": passes}
        transforms[condition] = {"axes": axis_result, "all_axes_pass": all(axis_result[axis]["passes_predeclared_threshold"] for axis in AXES)}
    complete = len(rows) == expected_calls
    return {"status": "completed", "counts": {"expected_calls": expected_calls, "observations": len(rows), "scored": sum(bool(row.get("scored")) for row in rows),
            "schema_valid": sum(bool(row.get("schema_valid")) for row in rows), "abstain": sum(bool(row.get("abstain")) for row in rows), "base_candidate_conditions": len(rows) // 50},
            "hard_gates": {"complete_observations": complete, "transport_or_schema_failures": not failures, "matched_original_control_pairs": missing_pairs == 0},
            "failure_categories": dict(sorted(failures.items())), "missing_condition_pairs": missing_pairs, "transforms": transforms,
            "validity_gate": {"passed": validity_passed, "thresholds": THRESHOLDS, "non_promotion_if_failed": True},
            "raw_prompts_or_responses_persisted": False, "selection_artifact_constructed": False}


def prepare(args: argparse.Namespace) -> None:
    cfg = JUDGE.config(); dest = destination(cfg, args.run_id)
    if dest.exists() or dest.is_symlink(): raise FileExistsError("refusing to overwrite contrastive lineage")
    source_limit = 1 if args.phase == "gpu0_smoke" else 80
    records, provenance = calibrated_population(cfg, source_limit); values = variants(records)
    expected = len(values) * len(cfg["protocol"]["prompt_types"]) * len(cfg["sampling"]["seeds"])
    dest.mkdir(mode=0o700, parents=True)
    manifest = {"schema_version": SCHEMA, "status": "prepared", "created_at": JUDGE.now(), "run_id": args.run_id, "phase": args.phase,
                "split": "train_calibration_only", "source_limit": source_limit, "base_candidates": len(records), "conditions": list(CONDITIONS), "expected_calls": expected,
                "samples_per_condition": 50, "provenance": provenance, "config_sha256": JUDGE.sha(JUDGE.CONFIG_PATH), "selection_artifact_constructed": False,
                "calibration_holdout_excluded_from_downstream_training": True, "validation_source_text_opened": 0, "source_writing_scores_read_or_prompted": False}
    JUDGE.atomic_json(dest / "manifest.json", manifest)
    print(json.dumps({"status": "prepared", "phase": args.phase, "base_candidates": len(records), "expected_calls": expected}, sort_keys=True))


def execute(args: argparse.Namespace) -> None:
    cfg = JUDGE.config(); dest = destination(cfg, args.run_id); manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"prepared", "running_interrupted"} or manifest.get("phase") != args.phase:
        raise RuntimeError("contrastive execution does not match a fresh/resumable run")
    validate_server(Path(args.server_attestation), args.endpoint, cfg, args.phase)
    records, provenance = calibrated_population(cfg, int(manifest["source_limit"])); values = variants(records)
    if provenance != manifest.get("provenance"):
        raise RuntimeError("contrastive inputs changed after prepare")
    context = JUDGE.prompt_budget(args.endpoint, cfg, args.model, values)
    observations = dest / "score_observations.jsonl"; done = JUDGE.existing_keys(observations)
    stream = task_stream(cfg, args.run_id, args.model, values, done); expected = int(manifest["expected_calls"])
    if len(done) > expected: raise RuntimeError("contrastive resume file has too many observations")
    inflight = cfg["runtime"]["max_num_seqs_per_dp_rank"] if args.phase == "gpu0_smoke" else cfg["runtime"]["client_max_inflight"]
    pending: set[Any] = set(); submitted = len(done); handle = observations.open("a" if observations.exists() else "x", encoding="utf-8")
    try:
        manifest.update({"status": "running", "started_at": JUDGE.now(), "context_budget": context, "resumed_observations": len(done), "client_max_inflight": inflight})
        JUDGE.atomic_json(manifest_path, manifest)
        with ThreadPoolExecutor(max_workers=inflight) as pool:
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < inflight:
                    try: task = next(stream)
                    except StopIteration: exhausted = True; break
                    pending.add(pool.submit(JUDGE.call, args.endpoint, task)); submitted += 1
                if not pending: continue
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed: handle.write(json.dumps(future.result(), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
    except BaseException:
        manifest.update({"status": "running_interrupted", "interrupted_at": JUDGE.now(), "submitted_or_existing_observations": submitted})
        JUDGE.atomic_json(manifest_path, manifest); raise
    finally:
        handle.close()
    report = summarize(dest, expected)
    report.update({"schema_version": SCHEMA, "created_at": JUDGE.now(), "run_id": args.run_id, "phase": args.phase, "config_sha256": JUDGE.sha(JUDGE.CONFIG_PATH),
                   "model": args.model, "provenance": provenance, "context_budget": context})
    report_path = dest / "aggregate_contrastive_report.json"; JUDGE.atomic_json(report_path, report)
    manifest.update({"status": "executed_completed", "completed_at": JUDGE.now(), "aggregate_report_sha256": JUDGE.sha(report_path),
                     "score_observations_sha256": JUDGE.sha(observations), "validity_gate_passed": report["validity_gate"]["passed"]})
    JUDGE.atomic_json(manifest_path, manifest)
    print(json.dumps({"status": report["status"], "phase": args.phase, "counts": report["counts"], "hard_gates": report["hard_gates"], "validity_gate": report["validity_gate"]}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (("prepare", prepare), ("execute", execute)):
        item = sub.add_parser(name); item.add_argument("--run-id", required=True); item.add_argument("--phase", choices=("gpu0_smoke", "full"), required=True)
        if name == "execute":
            item.add_argument("--endpoint", required=True); item.add_argument("--server-attestation", required=True); item.add_argument("--model", required=True)
        item.set_defaults(function=function)
    args = parser.parse_args(); args.function(args)
