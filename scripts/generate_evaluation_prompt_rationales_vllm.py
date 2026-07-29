#!/usr/bin/env python3
"""Generate bundled rationales with an evaluation.txt-derived SFT adapter."""
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import SOURCE_SHA256, load_writing_rows, sha256_file  # noqa: E402
from mal2026.evaluation_prompt_matrix import (  # noqa: E402
    RATIONALE_KINDS,
    RATIONALE_SCORE_BLIND,
    RATIONALE_SCORE_CONDITIONED,
    parse_rationale_output,
    prompt_provenance,
    rationale_messages,
    rationale_schema,
)
from mal2026.official_writing_contract import AXES  # noqa: E402


class RationaleQualityError(ValueError):
    pass


QUALITY_RETRY_SUFFIXES = ("""

[품질 게이트 재생성]
직전 생성은 길이 또는 문장 완결성 규칙을 통과하지 못했다. 같은 글을 다시 평가하되 각 영역을 60~420자의 1~4개 완결 문장으로 쓰고, 반복하지 말며, 반드시 마침표·물음표·느낌표 중 하나로 끝내라.""", """

[최종 간결 재생성]
직전 재생성도 완결성 검사를 통과하지 못했다. 각 영역을 핵심 근거만 담은 100~250자의 1~2개 문장으로 새로 작성하라. 열거를 늘이거나 같은 표현을 되풀이하지 말고 반드시 완결된 문장부호로 끝내라.""")
AXIS_FALLBACK_SUFFIXES = ("""

[단일 영역 형식 복구]
세 영역 묶음 중 {axis} 설명만 형식 검사를 통과하지 못했다. 다른 영역은 출력하지 말고 {axis} 기준에 해당하는 구체적 근거만 100~250자의 1~2개 완결 문장으로 다시 작성하라. 반드시 마침표·물음표·느낌표 중 하나로 끝내라.""", """

[단일 영역 2차 형식 복구]
직전 {axis} 단일 영역 응답도 길이나 문장 완결성을 충족하지 못했다. 원문에서 확인되는 {axis} 근거를 중복 없이 100~200자의 한 문장으로만 작성하고, 마지막은 반드시 마침표로 끝내라.""", """

[단일 영역 최종 형식 복구]
{axis} 설명 하나만 출력하라. 해당 영역의 핵심 판단과 원문 근거를 80~160자의 완결된 한국어 한 문장으로 쓰고 반드시 마침표로 끝내라.""")


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
            need(bool(line.strip()), "score input contains a blank row")
            raw = json.loads(line)
            need(isinstance(raw, dict) and "source_id" in raw and "emitted_integer_prediction" in raw, "score input schema differs")
            source_id, scores = raw["source_id"], raw["emitted_integer_prediction"]
            need(isinstance(source_id, str) and source_id not in result, "score source ID differs")
            need(isinstance(scores, dict) and set(scores) == set(AXES), "score axes differ")
            need(all(type(scores[axis]) is int and 1 <= scores[axis] <= 5 for axis in AXES), "score value differs")
            result[source_id] = {axis: int(scores[axis]) for axis in AXES}
    need(len(result) == expected, "score population differs")
    return result


def _quality_valid(value: str) -> bool:
    return 60 <= len(value.strip()) <= 420 and value.rstrip().endswith((".", "?", "!"))


def call(endpoint: str, body: Mapping[str, Any]) -> tuple[dict[str, str] | None, str | None, int, bool]:
    attempts = 0
    category = "transport"
    payload = dict(body)
    last_parsed: dict[str, str] | None = None
    for attempt_index in range(1 + len(QUALITY_RETRY_SUFFIXES)):
        attempts += 1
        wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = Request(endpoint.rstrip("/") + "/v1/chat/completions", data=wire, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=300) as response:
                choice = json.loads(response.read().decode("utf-8"))["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("finish_reason")
            parsed = parse_rationale_output(choice["message"]["content"])
            last_parsed = parsed
            if not all(_quality_valid(parsed[axis]) for axis in AXES):
                raise RationaleQualityError("rationale length or sentence completion gate failed")
            return parsed, None, attempts, False
        except HTTPError as exc:
            category = "http_429" if exc.code == 429 else ("http_5xx" if exc.code >= 500 else "http_4xx")
        except (URLError, TimeoutError):
            category = "transport"
        except RationaleQualityError:
            category = "quality_gate"
            if attempt_index < len(QUALITY_RETRY_SUFFIXES):
                messages = [dict(message) for message in payload["messages"]]
                messages[-1]["content"] += QUALITY_RETRY_SUFFIXES[attempt_index]
                payload = {**payload, "messages": messages}
                continue
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            category = "schema_or_finish"
        if category not in {"http_429", "http_5xx", "transport"}:
            break
    if category == "quality_gate" and last_parsed is not None:
        repaired = dict(last_parsed)
        invalid_axes = [axis for axis in AXES if not _quality_valid(repaired[axis])]
        for axis in invalid_axes:
            axis_category = "axis_quality_or_schema"
            for suffix_index, suffix in enumerate(AXIS_FALLBACK_SUFFIXES, start=1):
                attempts += 1
                messages = [dict(message) for message in body["messages"]]
                messages[-1]["content"] += suffix.format(axis=axis)
                axis_schema = {
                    "type": "object",
                    "properties": {"rationale": {"type": "string", "minLength": 60, "maxLength": 420}},
                    "required": ["rationale"], "additionalProperties": False,
                }
                axis_body = {
                    **body, "messages": messages, "max_tokens": 350,
                    "seed": int(body.get("seed", 42)) + suffix_index,
                    "response_format": {"type": "json_schema", "json_schema": {"name": f"mal2026_{axis}_rationale_repair_{suffix_index}", "strict": True, "schema": axis_schema}},
                }
                wire = json.dumps(axis_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                request = Request(endpoint.rstrip("/") + "/v1/chat/completions", data=wire, headers={"Content-Type": "application/json"}, method="POST")
                try:
                    with urlopen(request, timeout=300) as response:
                        choice = json.loads(response.read().decode("utf-8"))["choices"][0]
                    raw_axis = json.loads(choice["message"]["content"])
                    value = raw_axis["rationale"]
                    if choice.get("finish_reason") != "stop" or set(raw_axis) != {"rationale"} or not isinstance(value, str) or not _quality_valid(value):
                        raise RationaleQualityError("axis fallback quality gate failed")
                    repaired[axis] = value.strip()
                    axis_category = ""
                    break
                except HTTPError as exc:
                    axis_category = "http_429" if exc.code == 429 else ("http_5xx" if exc.code >= 500 else "http_4xx")
                except (URLError, TimeoutError):
                    axis_category = "transport"
                except (RationaleQualityError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                    axis_category = "axis_quality_or_schema"
            if axis_category:
                return None, axis_category, attempts, True
        if all(_quality_valid(repaired[axis]) for axis in AXES):
            return repaired, None, attempts, True
    return None, category, attempts, False


def tasks(
    split: str, expected: int, prompt_kind: str, scores: Mapping[str, Mapping[str, int]] | None,
    endpoint: str, model: str,
) -> Iterator[dict[str, Any]]:
    writings = load_writing_rows(split, include_scores=False)
    need(len(writings) >= expected, "canonical writing population differs")
    selected = writings[:expected]
    if scores is not None:
        by_id = {row.identifier: row for row in selected}
        need(set(scores) == set(by_id), "score IDs differ from canonical writings")
    schema = rationale_schema()
    for row in selected:
        predicted = None if scores is None else scores[row.identifier]
        yield {
            "source_id": row.identifier,
            "body": {
                "model": model, "temperature": 0.0, "top_p": 1.0, "seed": 42,
                "max_tokens": 900, "chat_template_kwargs": {"enable_thinking": False},
                "messages": rationale_messages(row.prompt, row.essay, prompt_kind, predicted),
                "response_format": {"type": "json_schema", "json_schema": {"name": "mal2026_evaluation_rationale_bundle", "strict": True, "schema": schema}},
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt-kind", choices=RATIONALE_KINDS, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--score-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-attestation", type=Path, required=True)
    parser.add_argument("--max-inflight", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    need(args.prompt_kind in RATIONALE_KINDS and args.max_inflight > 0, "generation request differs")
    expected_by_split = {"train": 2000, "validation": 400}
    need(1 <= args.expected <= expected_by_split[args.split], "generation population differs")
    if args.prompt_kind == RATIONALE_SCORE_BLIND:
        need(args.score_file is None, "score-blind generation received a score file")
        scores = None
    else:
        need(args.score_file is not None, "score-conditioned generation lacks a score file")
        scores = load_scores(args.score_file, args.expected)
    attestation = json.loads(args.server_attestation.read_text(encoding="utf-8"))
    need(attestation.get("schema_version") == "mal2026-evaluation-prompt-rationale-server-attestation-v1", "server attestation differs")
    need(attestation.get("endpoint") == args.endpoint and attestation.get("model_alias") == args.model, "server endpoint/alias differs")
    need(attestation.get("prompt_kind") == args.prompt_kind, "server prompt binding differs")
    output = args.output_dir
    need(not output.exists() and output.resolve().is_relative_to((ROOT / "data/processed/restricted").resolve()), "generation output must be fresh and restricted")
    output.mkdir(mode=0o700, parents=True)
    manifest = {
        "schema_version": "mal2026-evaluation-prompt-rationale-generation-v1",
        "status": "running", "run_id": args.run_id, "created_at": now(),
        "prompt_kind": args.prompt_kind, "split": args.split, "expected": args.expected,
        "source_sha256": SOURCE_SHA256[args.split],
        "score_conditioning": args.prompt_kind == RATIONALE_SCORE_CONDITIONED,
        "score_file_sha256": None if args.score_file is None else sha256_file(args.score_file),
        "score_kind": None if args.score_file is None else "score_encoder_actual_emitted_integer_prediction",
        "human_or_reference_score_read_or_prompted": False,
        "server_attestation_sha256": sha256_file(args.server_attestation),
        "rationale_schema_sha256": sha256(json.dumps(rationale_schema(), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "quality_retry_policy_sha256": sha256(json.dumps({"bundle_retries": QUALITY_RETRY_SUFFIXES, "axis_fallback_retries": AXIS_FALLBACK_SUFFIXES}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "quality_retry_maximum": len(QUALITY_RETRY_SUFFIXES),
        "axis_fallback_maximum": len(AXES) * len(AXIS_FALLBACK_SUFFIXES),
        **prompt_provenance(args.prompt_kind),
    }
    atomic_json(output / "manifest.json", manifest)
    work = list(tasks(args.split, args.expected, args.prompt_kind, scores, args.endpoint, args.model))
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
                pending[pool.submit(call, args.endpoint, task["body"])] = task
            if not pending:
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                rationales, failure, attempts, axis_fallback_used = future.result()
                records.append({"source_id": task["source_id"], "rationales": rationales, "failure_category": failure, "attempts": attempts, "axis_fallback_used": axis_fallback_used})
    records.sort(key=lambda row: str(row["source_id"]))
    records_path = output / "generated_rationales.jsonl"
    with records_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    failures = Counter(str(row["failure_category"]) for row in records if row.get("failure_category"))
    valid = sum(row.get("rationales") is not None for row in records)
    report = {
        "schema_version": "mal2026-evaluation-prompt-rationale-generation-aggregate-v1",
        "status": "completed" if len(records) == valid == args.expected and not failures else "failed_gates",
        "run_id": args.run_id, "prompt_kind": args.prompt_kind, "split": args.split,
        "counts": {"expected": args.expected, "records": len(records), "valid": valid},
        "failure_categories": dict(sorted(failures.items())),
        "generated_rationales_sha256": sha256_file(records_path),
        "score_file_sha256": manifest["score_file_sha256"],
        "rationale_schema_sha256": manifest["rationale_schema_sha256"],
        "quality_retry_policy_sha256": manifest["quality_retry_policy_sha256"],
        "quality_retry_maximum": manifest["quality_retry_maximum"],
        "axis_fallback_maximum": manifest["axis_fallback_maximum"],
        "records_requiring_retry": sum(int(row["attempts"] > 1) for row in records),
        "records_requiring_axis_fallback": sum(int(row["axis_fallback_used"]) for row in records),
        "human_or_reference_score_read_or_prompted": False,
        **prompt_provenance(args.prompt_kind),
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_or_predictions",
    }
    atomic_json(output / "aggregate_generation_report.json", report)
    manifest.update({"status": report["status"], "completed_at": now(), "aggregate_report_sha256": sha256_file(output / "aggregate_generation_report.json")})
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"status": report["status"], "counts": report["counts"], "run_id": args.run_id}, sort_keys=True))
    if report["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
