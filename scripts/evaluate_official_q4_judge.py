#!/usr/bin/env python3
"""Pointwise exact-Q4 evaluation of final MAL2026 participant outputs."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import EXPECTED_ESSAYS, SOURCE_SHA256, load_writing_rows, sha256_file  # noqa: E402
from mal2026.official_writing_contract import (  # noqa: E402
    AXES,
    JUDGE_DIMENSIONS,
    judge_json_schema,
    judge_messages,
    parse_judge_output,
    parse_participant_output,
)


MODEL_SHA256 = "b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772"
LLAMA_REVISION = "571d0d540df04f25298d0e159e520d9fc62ed121"
LLAMA_TAG = "b10068"
RESTRICTED_ROOT = ROOT / "data/processed/restricted/official_prompt_alignment_v1/q4_judge"
AGGREGATE_ROOT = ROOT / "outputs/official-prompt-alignment-v1/q4-judge"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_participants(path: Path, expected: int) -> list[dict[str, Any]]:
    need(path.is_file() and not path.is_symlink(), "participant input is unavailable")
    allowed = (ROOT / "data/processed/restricted").resolve()
    need(path.resolve().is_relative_to(allowed), "participant input must remain in restricted data")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            need(bool(line.strip()), "participant input contains a blank row")
            raw = json.loads(line)
            need(isinstance(raw, dict) and set(raw) == {"source_id", "participant_output"}, "participant input row schema differs")
            source_id = raw["source_id"]
            need(isinstance(source_id, str) and source_id not in seen, "participant source ID differs")
            seen.add(source_id)
            rows.append({"source_id": source_id, "participant_output": parse_participant_output(raw["participant_output"])})
    need(len(rows) == expected, "participant population differs")
    return rows


def request_body(model: str, prompt: str, essay: str, participant: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": 1800,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": judge_messages(prompt, essay, participant),
        "response_format": {"type": "json_object", "schema": judge_json_schema()},
    }


def call(endpoint: str, body: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None, int]:
    wire = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    for attempt in range(1, 3):
        request = Request(endpoint.rstrip("/") + "/v1/chat/completions", data=wire, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=300) as response:
                outer = json.loads(response.read().decode("utf-8"))
            choice = outer["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("finish_reason")
            parsed = parse_judge_output(choice["message"]["content"])
            return parsed, None, attempt
        except HTTPError as exc:
            category = "http_429" if exc.code == 429 else ("http_5xx" if exc.code >= 500 else "http_4xx")
        except (URLError, TimeoutError):
            category = "transport"
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            category = "schema_or_finish"
        if category not in {"http_429", "http_5xx", "transport"}:
            break
    return None, category, attempt


def tasks(participants: Sequence[Mapping[str, Any]], model: str, endpoints: Sequence[str], split: str) -> Iterator[dict[str, Any]]:
    writings = {row.identifier: row for row in load_writing_rows(split, include_scores=False)}
    participant_ids = {str(row["source_id"]) for row in participants}
    need(participant_ids <= set(writings), "participant IDs differ from the canonical source")
    for index, row in enumerate(participants):
        writing = writings[str(row["source_id"])]
        yield {
            "source_id": row["source_id"],
            "endpoint": endpoints[index % len(endpoints)],
            "body": request_body(model, writing.prompt, writing.essay, row["participant_output"]),
        }


def aggregate(records: Sequence[Mapping[str, Any]], expected: int) -> dict[str, Any]:
    failures = Counter(str(record["failure_category"]) for record in records if record.get("failure_category"))
    valid = [record for record in records if record.get("judge_output") is not None]
    cells: dict[str, dict[str, list[int]]] = {axis: {dimension: [] for dimension in JUDGE_DIMENSIONS} for axis in AXES}
    low = 0
    total = 0
    for record in valid:
        output = record["judge_output"]
        for axis in AXES:
            for dimension in JUDGE_DIMENSIONS:
                score = int(output[axis][dimension]["score"])
                cells[axis][dimension].append(score)
                total += 1
                low += int(score <= 2)
    cell_means = {
        axis: {dimension: statistics.fmean(cells[axis][dimension]) if cells[axis][dimension] else None for dimension in JUDGE_DIMENSIONS}
        for axis in AXES
    }
    axis_means = {axis: statistics.fmean(cell_means[axis].values()) if all(value is not None for value in cell_means[axis].values()) else None for axis in AXES}
    dimension_means = {
        dimension: statistics.fmean(cell_means[axis][dimension] for axis in AXES) if all(cell_means[axis][dimension] is not None for axis in AXES) else None
        for dimension in JUDGE_DIMENSIONS
    }
    flat = [cell_means[axis][dimension] for axis in AXES for dimension in JUDGE_DIMENSIONS]
    hard_gates = {
        "complete_records": len(records) == expected,
        "all_schema_valid": len(valid) == expected,
        "zero_failures": not failures,
        "twelve_cells_per_record": total == expected * len(AXES) * len(JUDGE_DIMENSIONS),
    }
    return {
        "status": "completed" if all(hard_gates.values()) else "failed_gates",
        "counts": {"expected": expected, "records": len(records), "valid": len(valid), "judge_cells": total},
        "hard_gates": hard_gates,
        "failure_categories": dict(sorted(failures.items())),
        "cell_means": cell_means,
        "axis_means": axis_means,
        "dimension_means": dimension_means,
        "macro_mean": statistics.fmean(flat) if all(value is not None for value in flat) else None,
        "worst_cell_mean": min(flat) if all(value is not None for value in flat) else None,
        "score_1_or_2_rate": low / total if total else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--participant-file", type=Path, required=True)
    parser.add_argument("--model", default="qwen36-35b-a3b-q4_k_m")
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--expected", type=int, default=400)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--max-inflight", type=int, default=16)
    parser.add_argument("--server-attestation", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    need(1 <= args.expected <= EXPECTED_ESSAYS[args.split], "official judge expected count differs")
    need(len(args.endpoint) in {1, 4} and args.max_inflight >= len(args.endpoint), "official judge endpoint/concurrency differs")
    attestation = json.loads(args.server_attestation.read_text(encoding="utf-8"))
    need(attestation.get("schema_version") == "mal2026-official-q4-judge-server-attestation-v1", "official judge attestation schema differs")
    need(attestation.get("model_sha256") == MODEL_SHA256 and attestation.get("llama_revision") == LLAMA_REVISION and attestation.get("llama_tag") == LLAMA_TAG, "official judge runtime provenance differs")
    need(attestation.get("server_endpoints") == args.endpoint, "official judge endpoints differ from attestation")
    participants = load_participants(args.participant_file, args.expected)
    output = RESTRICTED_ROOT / args.run_id
    aggregate_output = AGGREGATE_ROOT / args.run_id
    need(not output.exists() and not aggregate_output.exists(), "official judge output must be fresh")
    output.mkdir(mode=0o700, parents=True)
    aggregate_output.mkdir(parents=True)
    manifest = {
        "schema_version": "mal2026-official-q4-judge-v1",
        "status": "running",
        "run_id": args.run_id,
        "created_at": now(),
        "participant_sha256": sha256_file(args.participant_file),
        "participant_records": len(participants),
        "source_split": args.split,
        "source_sha256": SOURCE_SHA256[args.split],
        "model_sha256": MODEL_SHA256,
        "llama_revision": LLAMA_REVISION,
        "llama_tag": LLAMA_TAG,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "candidate_score_kind": "actual_emitted_integer_prediction",
        "human_or_reference_score_read_or_prompted": False,
        "server_attestation_sha256": sha256_file(args.server_attestation),
    }
    atomic_json(output / "manifest.json", manifest)
    work = list(tasks(participants, args.model, args.endpoint, args.split))
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
        pending: dict[Any, dict[str, Any]] = {}
        iterator = iter(work)
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < args.max_inflight:
                try:
                    task = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(call, task["endpoint"], task["body"])] = task
            if not pending:
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                parsed, failure, attempts = future.result()
                records.append({"source_id": task["source_id"], "judge_output": parsed, "failure_category": failure, "attempts": attempts})
    records.sort(key=lambda row: str(row["source_id"]))
    record_path = output / "judge_records.jsonl"
    with record_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = aggregate(records, args.expected)
    report.update({
        "schema_version": "mal2026-official-q4-judge-aggregate-v1",
        "run_id": args.run_id,
        "judge_records_sha256": sha256_file(record_path),
        "participant_sha256": manifest["participant_sha256"],
        "model_sha256": MODEL_SHA256,
        "llama_revision": LLAMA_REVISION,
        "llama_tag": LLAMA_TAG,
        "temperature": 0.0,
        "seed": 42,
        "candidate_score_kind": "actual_emitted_integer_prediction",
        "human_or_reference_score_read_or_prompted": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_evidence_in_this_report",
    })
    atomic_json(aggregate_output / "aggregate_judge_report.json", report)
    manifest.update({"status": report["status"], "completed_at": now(), "aggregate_report_sha256": sha256_file(aggregate_output / "aggregate_judge_report.json")})
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"status": report["status"], "run_id": args.run_id, "counts": report["counts"], "macro_mean": report["macro_mean"]}, sort_keys=True))
    if report["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
