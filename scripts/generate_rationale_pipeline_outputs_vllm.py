#!/usr/bin/env python3
"""Generate score-blind rationale bundles with the frozen pipeline prompt.

The script deliberately loads canonical prompt/essay text without scores.  It
persists individual generations only below the restricted data root and emits
an aggregate-only report suitable for experiment records.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import (  # noqa: E402
    EXPECTED_ESSAYS,
    SOURCE_SHA256,
    load_writing_rows,
    sha256_file,
)
from mal2026.rationale_pipeline_prompts import (  # noqa: E402
    AXES,
    rationale_messages,
    rationale_output,
    round_half_up_score,
    routing,
)


RESTRICTED_PARENT = ROOT / "data/processed/restricted/rationale_pipeline_v1/generation"
AGGREGATE_PARENT = ROOT / "outputs/rationale-pipeline-generation-v1"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def schema() -> dict[str, Any]:
    axis = {
        "type": "object",
        "properties": {"rationale": {"type": "string", "minLength": 1}},
        "required": ["rationale"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {name: axis for name in AXES},
        "required": list(AXES),
        "additionalProperties": False,
    }


def parse_output(value: Any) -> dict[str, str] | None:
    try:
        parsed = rationale_output(value)
    except (TypeError, ValueError):
        return None
    return {axis: str(parsed[axis]["rationale"]).strip() for axis in AXES}


def call(endpoint: str, body: Mapping[str, Any]) -> tuple[dict[str, str] | None, str | None, int]:
    payload = dict(body)
    category = "transport"
    for attempt in range(1, 4):
        wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = Request(
            endpoint.rstrip("/") + "/v1/chat/completions",
            data=wire,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=300) as response:
                choice = json.loads(response.read().decode("utf-8"))["choices"][0]
            if choice.get("finish_reason") == "length" and int(payload.get("max_tokens", 0)) in {1000, 2000}:
                # Capacity-only retry: the frozen prompt, seed, sampling, and
                # schema stay unchanged. No content repair is attempted.
                payload = {**payload, "max_tokens": int(payload["max_tokens"]) + 1000}
                category = "explicit_length_retry"
                continue
            if choice.get("finish_reason") != "stop":
                raise ValueError("finish_reason")
            parsed = parse_output(choice["message"]["content"])
            if parsed is None:
                raise ValueError("schema")
            return parsed, None, attempt
        except HTTPError as exc:
            category = "http_429" if exc.code == 429 else ("http_5xx" if exc.code >= 500 else "http_4xx")
        except (URLError, TimeoutError):
            category = "transport"
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            category = "schema_or_finish"
        if category not in {"http_429", "http_5xx", "transport", "explicit_length_retry"}:
            break
    return None, category, attempt


def tasks(
    split: str,
    expected: int,
    model: str,
    endpoints: Sequence[str],
    tail_multiplicity: bool = False,
    multiplicity_reference: Path | None = None,
    multiplicity_scale: int = 1,
) -> Iterator[dict[str, Any]]:
    # Score blindness is enforced at the loader boundary, not only by prompt text.
    rows = load_writing_rows(split, include_scores=tail_multiplicity)
    if not tail_multiplicity:
        rows = rows[:expected]
    reference_counts: Counter[str] | None = None
    if multiplicity_reference is not None:
        reference_counts = Counter()
        with multiplicity_reference.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    need(isinstance(value, Mapping) and value.get("source_id") is not None, "multiplicity reference row differs")
                    reference_counts[str(value["source_id"])] += 1
        need(reference_counts and multiplicity_scale in {1, 2, 3}, "multiplicity reference contract differs")
    response_schema = schema()
    task_index = 0
    canonical_ids: set[str] = set()
    for row in rows:
        canonical_ids.add(row.identifier)
        # All variants for one source share the exact frozen score-blind
        # prompt. Render it once; only the sampling seed changes.
        messages = rationale_messages(row.prompt, row.essay)
        multiplicity = 1
        if tail_multiplicity:
            need(row.scores is not None, "tail generation scores unavailable")
            bands = {round_half_up_score(value) for value in row.scores.values()}
            multiplicity = 4 if 1 in bands else (2 if bands & {2, 5} else 1)
        elif reference_counts is not None:
            need(row.identifier in reference_counts, "multiplicity reference source coverage differs")
            multiplicity = reference_counts[row.identifier] * multiplicity_scale
        for variant_index in range(multiplicity):
            seed = 2026080704
            temperature, top_p = 0.0, 1.0
            if tail_multiplicity or reference_counts is not None:
                seed = int.from_bytes(sha256(f"2026080802:{row.identifier}:{variant_index}".encode()).digest()[:4], "big") % (2**31 - 1)
                temperature, top_p = 0.7, 0.95
            yield {
                "source_id": row.identifier,
                "variant_index": variant_index,
                "endpoint": endpoints[task_index % len(endpoints)],
                "body": {
                    "model": model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                    "max_tokens": 1000,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": messages,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "mal2026_rationale_pipeline_bundle_v1",
                            "strict": True,
                            "schema": response_schema,
                        },
                    },
                },
            }
            task_index += 1
    if reference_counts is not None:
        need(set(reference_counts) == canonical_ids, "multiplicity reference population differs")
    need(task_index == expected, "generation population differs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--server-attestation", type=Path, required=True)
    parser.add_argument("--max-inflight", type=int, default=128)
    parser.add_argument("--tail-multiplicity", action="store_true")
    parser.add_argument("--multiplicity-reference", type=Path)
    parser.add_argument("--multiplicity-scale", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setproctitle(f"mal2026:rationale-pipeline-generate:{args.run_id}"[:255])
    prompt_routing = routing()
    need(not (args.tail_multiplicity and args.multiplicity_reference is not None), "generation multiplicity modes are mutually exclusive")
    stochastic_multiplicity = args.tail_multiplicity or args.multiplicity_reference is not None
    maximum_expected = 100 * EXPECTED_ESSAYS[args.split] if stochastic_multiplicity else EXPECTED_ESSAYS[args.split]
    need(1 <= args.expected <= maximum_expected, "generation expected count differs")
    need(not args.tail_multiplicity or (args.split == "train" and args.expected > EXPECTED_ESSAYS[args.split]), "tail generation contract differs")
    need(args.multiplicity_reference is None or (args.split == "train" and args.multiplicity_reference.is_file() and args.multiplicity_scale in {1, 2, 3}), "reference multiplicity generation contract differs")
    need(len(args.endpoint) in {1, 4} and args.max_inflight >= len(args.endpoint), "generation endpoint shape differs")
    attestation = json.loads(args.server_attestation.read_text(encoding="utf-8"))
    need(attestation.get("schema_version") == "mal2026-rationale-pipeline-vllm-server-v1", "generation attestation schema differs")
    need(attestation.get("server_endpoints") == args.endpoint, "generation endpoints differ")
    aliases = attestation.get("model_aliases")
    legacy_alias = attestation.get("model_alias")
    need(
        (isinstance(aliases, list) and args.model in aliases)
        or legacy_alias == args.model,
        "generation model alias differs",
    )
    expected_prompt_sha = prompt_routing["rationale_generation_training_evaluation"]["source_file_sha256"]
    need(attestation.get("rationale_prompt_sha256") == expected_prompt_sha, "generation prompt attestation differs")

    restricted = RESTRICTED_PARENT / args.run_id
    aggregate = AGGREGATE_PARENT / args.run_id
    need(not restricted.exists() and not aggregate.exists(), "generation output must be fresh")
    restricted.mkdir(parents=True, mode=0o700)
    aggregate.mkdir(parents=True)
    manifest = {
        "schema_version": "mal2026-rationale-pipeline-generation-v1",
        "status": "running",
        "run_id": args.run_id,
        "created_at": now(),
        "split": args.split,
        "expected": args.expected,
        "source_sha256": SOURCE_SHA256[args.split],
        "model_alias": args.model,
        "max_inflight": args.max_inflight,
        "temperature": 0.7 if stochastic_multiplicity else 0.0,
        "top_p": 0.95 if stochastic_multiplicity else 1.0,
        "seed": "sha256_source_variant" if stochastic_multiplicity else 2026080704,
        "tail_multiplicity": args.tail_multiplicity,
        "reference_multiplicity": args.multiplicity_reference is not None,
        "multiplicity_reference_sha256": sha256_file(args.multiplicity_reference) if args.multiplicity_reference is not None else None,
        "multiplicity_scale": args.multiplicity_scale if args.multiplicity_reference is not None else None,
        "tail_sampling": {"temperature": 0.7, "top_p": 0.95, "seed_scheme": "sha256_source_variant"} if stochastic_multiplicity else None,
        "prompt_sha256": expected_prompt_sha,
        "server_attestation_sha256": sha256_file(args.server_attestation),
        "human_or_reference_score_read_or_prompted": False,
        "score_output": False,
        "average_used": False,
    }
    atomic_json(restricted / "manifest.json", manifest)

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
        pending: dict[Any, dict[str, Any]] = {}
        # Keep at most max_inflight requests materialized; a 3x rationale pool
        # is intentionally large and should not duplicate prompt strings in RAM.
        iterator = tasks(args.split, args.expected, args.model, args.endpoint, args.tail_multiplicity, args.multiplicity_reference, args.multiplicity_scale)
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
                rationales, failure, attempts = future.result()
                records.append({
                    "source_id": task["source_id"],
                    "variant_index": task["variant_index"],
                    "rationales": rationales,
                    "failure_category": failure,
                    "attempts": attempts,
                })
    records.sort(key=lambda row: (str(row["source_id"]), int(row["variant_index"])))
    records_path = restricted / "generated_rationales.jsonl"
    with records_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(records_path, 0o600)

    failures = Counter(str(row["failure_category"]) for row in records if row.get("failure_category"))
    valid = sum(row.get("rationales") is not None for row in records)
    gates = {
        "complete_records": len(records) == args.expected,
        "all_schema_valid": valid == args.expected,
        "zero_failures": not failures,
    }
    report = {
        "schema_version": "mal2026-rationale-pipeline-generation-aggregate-v1",
        "status": "completed" if all(gates.values()) else "failed_gates",
        "run_id": args.run_id,
        "split": args.split,
        "counts": {"expected": args.expected, "records": len(records), "valid": valid},
        "hard_gates": gates,
        "failure_categories": dict(sorted(failures.items())),
        "generated_rationales_sha256": sha256_file(records_path),
        "model_alias": args.model,
        "max_inflight": args.max_inflight,
        "prompt_sha256": expected_prompt_sha,
        "human_or_reference_score_read_or_prompted": False,
        "score_output": False,
        "tail_multiplicity": args.tail_multiplicity,
        "reference_multiplicity": args.multiplicity_reference is not None,
        "multiplicity_reference_sha256": sha256_file(args.multiplicity_reference) if args.multiplicity_reference is not None else None,
        "multiplicity_scale": args.multiplicity_scale if args.multiplicity_reference is not None else None,
        "average_used": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_or_model_weights",
    }
    atomic_json(aggregate / "aggregate.json", report)
    manifest.update({
        "status": report["status"],
        "completed_at": now(),
        "aggregate_sha256": sha256_file(aggregate / "aggregate.json"),
        "generated_rationales_sha256": report["generated_rationales_sha256"],
    })
    atomic_json(restricted / "manifest.json", manifest)
    print(json.dumps({"status": report["status"], "run_id": args.run_id, "counts": report["counts"]}, sort_keys=True), flush=True)
    if report["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
