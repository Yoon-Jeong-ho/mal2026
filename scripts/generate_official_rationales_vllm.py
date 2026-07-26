#!/usr/bin/env python3
"""Generate score-conditioned rationales from one official SFT adapter."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import EXPECTED_ESSAYS, SOURCE_SHA256, load_writing_rows, sha256_file  # noqa: E402
from mal2026.official_rationale_data import axes_for_task, messages, parse_rationale_output, rationale_schema  # noqa: E402


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_scores(path: Path, expected: int) -> dict[str, dict[str, int]]:
    need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to((ROOT / "data/processed/restricted").resolve()), "score input must be restricted")
    result: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            need(isinstance(raw, dict) and set(raw) in ({"source_id", "emitted_integer_prediction"}, {"source_id", "continuous_prediction", "emitted_integer_prediction"}), "score input schema differs")
            source_id, scores = raw["source_id"], raw["emitted_integer_prediction"]
            need(isinstance(source_id, str) and source_id not in result, "score source ID differs")
            need(isinstance(scores, dict) and set(scores) == {"content", "organization", "expression"}, "score axes differ")
            need(all(type(scores[axis]) is int and 1 <= scores[axis] <= 5 for axis in scores), "score must be emitted integers")
            result[source_id] = {axis: int(scores[axis]) for axis in ("content", "organization", "expression")}
    need(len(result) == expected, "score population differs")
    return result


def call(endpoint: str, body: Mapping[str, Any], axes: tuple[str, ...]) -> tuple[dict[str, str] | None, str | None, int]:
    wire = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    for attempt in range(1, 3):
        request = Request(endpoint.rstrip("/") + "/v1/chat/completions", data=wire, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=240) as response:
                outer = json.loads(response.read().decode("utf-8"))
            choice = outer["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("finish_reason")
            content = choice["message"]["content"]
            return parse_rationale_output(content, axes), None, attempt
        except HTTPError as exc:
            category = "http_429" if exc.code == 429 else ("http_5xx" if exc.code >= 500 else "http_4xx")
        except (URLError, TimeoutError):
            category = "transport"
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            category = "schema_or_finish"
        if category not in {"http_429", "http_5xx", "transport"}:
            break
    return None, category, attempt


def request_tasks(args: argparse.Namespace, scores: Mapping[str, Mapping[str, int]]) -> Iterator[dict[str, Any]]:
    axes = axes_for_task(args.task)
    writings = load_writing_rows(args.split, include_scores=False)
    selected = [row for row in writings if row.identifier in scores]
    need(len(selected) == args.expected, "score IDs differ from canonical source")
    schema = rationale_schema(axes)
    for row in selected:
        body = {
            "model": args.model,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "max_tokens": 900 if args.task == "bundle" else 350,
            "messages": messages(row.prompt, row.essay, scores[row.identifier], axes),
            "response_format": {"type": "json_schema", "json_schema": {"name": f"mal2026_rationale_{args.task}", "strict": True, "schema": schema}},
        }
        yield {"source_id": row.identifier, "axes": axes, "body": body}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", choices=("bundle", "content", "organization", "expression"), required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-attestation", type=Path, required=True)
    parser.add_argument("--max-inflight", type=int, default=128)
    args = parser.parse_args()
    need(1 <= args.expected <= EXPECTED_ESSAYS[args.split], "generation population differs")
    output = args.output_dir.resolve()
    need(output.is_relative_to((ROOT / "data/processed/restricted").resolve()) and not output.exists(), "generation output must be a fresh restricted path")
    attestation = json.loads(args.server_attestation.read_text(encoding="utf-8"))
    need(attestation.get("schema_version") == "mal2026-official-rationale-vllm-server-attestation-v1", "generation attestation schema differs")
    legacy_identity = attestation.get("adapter_alias") == args.model and attestation.get("task") == args.task
    grouped_identity = isinstance(attestation.get("adapter_aliases"), dict) and attestation["adapter_aliases"].get(args.model) == args.task
    need(attestation.get("endpoint") == args.endpoint and (legacy_identity or grouped_identity), "generation server identity differs")
    scores = load_scores(args.score_file, args.expected)
    output.mkdir(mode=0o700, parents=True)
    manifest = {
        "schema_version": "mal2026-official-rationale-generation-v1",
        "status": "running",
        "run_id": args.run_id,
        "task": args.task,
        "split": args.split,
        "expected": args.expected,
        "score_file_sha256": sha256_file(args.score_file),
        "source_sha256": SOURCE_SHA256[args.split],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "score_kind": "actual_emitted_integer_prediction",
        "human_or_reference_score_read_or_prompted": False,
        "server_attestation_sha256": sha256_file(args.server_attestation),
        "created_at": now(),
    }
    atomic_json(output / "manifest.json", manifest)
    work = list(request_tasks(args, scores))
    records: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
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
                # Axes are parser metadata and are deliberately kept out of the
                # OpenAI-compatible wire request.
                pending[pool.submit(call, args.endpoint, task["body"], tuple(task["axes"]))] = task
            if not pending:
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                rationales, failure, attempts = future.result()
                if failure:
                    failures[failure] += 1
                records.append({"source_id": task["source_id"], "rationales": rationales, "failure_category": failure, "attempts": attempts})
    records.sort(key=lambda row: str(row["source_id"]))
    record_path = output / "generated_rationales.jsonl"
    with record_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    valid = sum(record["rationales"] is not None for record in records)
    hard_gates = {"complete_records": len(records) == args.expected, "all_parse_valid": valid == args.expected, "zero_failures": not failures}
    report = {
        "schema_version": "mal2026-official-rationale-generation-aggregate-v1",
        "status": "completed" if all(hard_gates.values()) else "failed_gates",
        "run_id": args.run_id,
        "task": args.task,
        "split": args.split,
        "counts": {"expected": args.expected, "records": len(records), "parse_valid": valid},
        "hard_gates": hard_gates,
        "failure_categories": dict(sorted(failures.items())),
        "generated_rationales_sha256": sha256_file(record_path),
        "score_file_sha256": manifest["score_file_sha256"],
        "temperature": 0.0,
        "seed": 42,
        "score_kind": "actual_emitted_integer_prediction",
        "human_or_reference_score_read_or_prompted": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_in_this_report",
    }
    atomic_json(output / "aggregate_generation_report.json", report)
    manifest.update({"status": report["status"], "completed_at": now(), "aggregate_report_sha256": sha256_file(output / "aggregate_generation_report.json")})
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"status": report["status"], "run_id": args.run_id, "task": args.task, "counts": report["counts"]}, sort_keys=True))
    if report["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
