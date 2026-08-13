#!/usr/bin/env python3
"""Test the canonical rationale prompt on a label-balanced restricted train sample.

Raw prompts, essays, identifiers, and model outputs stay under ``data/processed``.
Only aggregate diagnostics without source identifiers or writing text are written
under ``outputs`` and printed to stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = ROOT / "rationale_generation_prompt.txt"
SOURCE_PATH = ROOT / "eval/train.jsonl"
RESTRICTED_ROOT = ROOT / "data/processed/restricted/rationale_prompt_openai_test"
OUTPUT_ROOT = ROOT / "outputs/rationale_prompt_openai_test"
API_ROOT = "https://api.openai.com/v1"
AXES = ("content", "organization", "expression")
MODELS = ("gpt-5.6-luna", "gpt-5.6-terra")
SAMPLE_SIZE = 15
TARGET_PER_BAND = 3
SEED = 20260806
MAX_OUTPUT_TOKENS = 1800

AXIS_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rationale"],
    "properties": {"rationale": {"type": "string"}},
}
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(AXES),
    "properties": {axis: AXIS_RESPONSE_SCHEMA for axis in AXES},
}
SCORE_LEAK_RE = re.compile(
    r"(?<!\d)[1-5]\s*(?:점|등급|단계)|reference_scores?|기준값|주어진\s*점수|"
    r"해당\s*점수|정답\s*라벨|고득점|저득점|최고\s*등급|최하\s*등급",
    re.IGNORECASE,
)
POSITIVE_TERMS = ("명확", "구체", "충실", "자연", "적절", "일관", "효과", "뚜렷", "충분", "다양", "논리적")
NEGATIVE_TERMS = ("부족", "단순", "반복", "어색", "불명확", "미흡", "제한", "약하", "끊", "오류", "드러나지", "제시되지", "모호")
ADVICE_TERMS = (
    "보완해야 한다", "개선해야 한다", "추가해야 한다", "고쳐야 한다",
    "수정해야 한다", "다듬어야 한다", "구체화해야 한다", "명확히 해야 한다",
    "제시할 필요가 있다", "추가할 필요가 있다",
)


class RationaleTestError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(path, 0o600)


def emit(**values: Any) -> None:
    print(json.dumps(values, ensure_ascii=False, sort_keys=True), flush=True)


def round_half_up(value: Any) -> int:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RationaleTestError("non-numeric score") from exc
    if not number.is_finite() or not Decimal("1") <= number <= Decimal("5"):
        raise RationaleTestError("score outside finite [1,5]")
    return int(number.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def checked_prompt_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink() or not resolved.is_relative_to(ROOT.resolve()):
        raise RationaleTestError("prompt file must be a regular repository-local file")
    return resolved


def split_prompt(prompt_path: Path) -> tuple[str, str]:
    value = checked_prompt_path(prompt_path).read_text(encoding="utf-8")
    system_marker = "[시스템 프롬프트]"
    user_marker = "[유저 프롬프트 템플릿]"
    if value.count(system_marker) != 1 or value.count(user_marker) != 1:
        raise RationaleTestError("prompt section markers differ")
    before, template = value.split(user_marker, 1)
    _, system = before.split(system_marker, 1)
    system, template = system.strip(), template.strip()
    required = (
        "{prompt_text_json_string}", "{essay_text_json_string}",
        "{content_score_integer}", "{organization_score_integer}",
        "{expression_score_integer}",
    )
    if any(template.count(field) != 1 for field in required):
        raise RationaleTestError("user template placeholders differ")
    return system, template


def render_user(template: str, row: Mapping[str, Any]) -> str:
    scores = row["integer_scores"]
    replacements = {
        "{prompt_text_json_string}": json.dumps(row["prompt"], ensure_ascii=False),
        "{essay_text_json_string}": json.dumps(row["essay"], ensure_ascii=False),
        "{content_score_integer}": str(scores["content"]),
        "{organization_score_integer}": str(scores["organization"]),
        "{expression_score_integer}": str(scores["expression"]),
    }
    rendered = template
    for source, destination in replacements.items():
        rendered = rendered.replace(source, destination)
    if "{" in rendered and "}" in rendered:
        # Confirm that the embedded portion is a valid JSON object without
        # rejecting braces that may legitimately occur inside essay strings.
        start, end = rendered.find("{"), rendered.rfind("}") + 1
        embedded = json.loads(rendered[start:end])
        if set(embedded) != {"prompt_text", "essay_text", "reference_scores_integer"}:
            raise RationaleTestError("rendered user JSON shape differs")
    return rendered


def load_candidates() -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw: list[dict[str, Any]] = []
    by_text: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    with SOURCE_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            scores = tuple(round_half_up(row["score"][axis]) for axis in AXES)
            text_sha = sha256_bytes((str(row["prompt"]) + "\0" + str(row["essay"])).encode())
            by_text[text_sha].append(scores)
            raw.append({
                "line_number": line_number,
                "source_id": str(row["id"]),
                "document_id": str(row["document_id"]),
                "prompt": str(row["prompt"]),
                "essay": str(row["essay"]),
                "raw_scores": {axis: float(row["score"][axis]) for axis in AXES},
                "integer_scores": dict(zip(AXES, scores)),
                "score_tuple": scores,
                "text_sha256": text_sha,
            })
    conflicts = {key for key, values in by_text.items() if len(set(values)) > 1}
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    for row in raw:
        if row["text_sha256"] in conflicts:
            continue
        if row["text_sha256"] in seen:
            duplicate_count += 1
            continue
        seen.add(row["text_sha256"])
        candidates.append(row)
    return candidates, {
        "source_rows": len(raw),
        "eligible_rows": len(candidates),
        "conflicting_text_groups": len(conflicts),
        "same_label_duplicate_rows_removed": duplicate_count,
    }


def histogram(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        axis: {str(score): sum(row["integer_scores"][axis] == score for row in rows) for score in range(1, 6)}
        for axis in AXES
    }


def objective(indices: Sequence[int], rows: Sequence[Mapping[str, Any]]) -> int:
    counts = histogram([rows[index] for index in indices])
    return sum(abs(counts[axis][str(score)] - TARGET_PER_BAND) for axis in AXES for score in range(1, 6))


def select_balanced(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Find 15 rows with exactly three observations in every axis/score cell."""
    rng = random.Random(SEED)
    universe = list(range(len(rows)))
    best_score = math.inf
    best: list[int] | None = None
    for _restart in range(1000):
        selected = sorted(rng.sample(universe, SAMPLE_SIZE))
        score = objective(selected, rows)
        for _iteration in range(10000):
            old = selected[rng.randrange(len(selected))]
            new = rng.randrange(len(rows))
            if new in selected:
                continue
            proposal = sorted((set(selected) - {old}) | {new})
            proposal_score = objective(proposal, rows)
            if proposal_score < score or (proposal_score == score and rng.random() < 0.0003):
                selected, score = proposal, proposal_score
            if score == 0:
                break
        if score < best_score:
            best_score, best = score, list(selected)
        if score == 0:
            break
    if best is None or best_score != 0:
        raise RationaleTestError(f"exact balanced sample unavailable; objective={best_score}")
    chosen = [dict(rows[index]) for index in best]
    chosen.sort(key=lambda row: (row["score_tuple"], row["text_sha256"]))
    for index, row in enumerate(chosen, 1):
        row["case_key"] = f"case-{index:02d}"
        row.pop("score_tuple", None)
    return chosen


def load_api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return str(os.environ["OPENAI_API_KEY"])
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                if value:
                    return value
    raise RationaleTestError("OPENAI_API_KEY unavailable")


def response_body(model: str, system: str, user: str) -> dict[str, Any]:
    if model not in MODELS:
        raise RationaleTestError("model alias differs")
    return {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [{"type": "input_text", "text": user}]},
        ],
        "text": {"format": {
            "type": "json_schema",
            "name": "mal2026_rationale_generation",
            "strict": True,
            "schema": RESPONSE_SCHEMA,
        }},
        "reasoning": {"effort": "none"},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
    }


def response_text(response: Mapping[str, Any]) -> str:
    for item in response.get("output", []):
        if isinstance(item, Mapping):
            for content in item.get("content", []):
                if isinstance(content, Mapping) and content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return str(content["text"])
    raise RationaleTestError("Responses result has no output_text")


def api_call(payload: Mapping[str, Any], idempotency_key: str) -> dict[str, Any]:
    request = Request(
        API_ROOT + "/responses",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
    )
    request.add_header("Authorization", f"Bearer {load_api_key()}")
    request.add_header("Content-Type", "application/json")
    request.add_header("Idempotency-Key", idempotency_key)
    try:
        with urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RationaleTestError(f"OpenAI response failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RationaleTestError("OpenAI response network failure") from exc
    if not isinstance(result, dict):
        raise RationaleTestError("OpenAI response envelope differs")
    return result


def validate_output(text: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}, ["invalid_json"]
    if not isinstance(value, dict) or set(value) != set(AXES):
        return {}, ["top_level_shape"]
    for axis in AXES:
        part = value.get(axis)
        if not isinstance(part, dict) or set(part) != {"rationale"}:
            errors.append(f"{axis}:shape")
            continue
        rationale = part["rationale"]
        if not isinstance(rationale, str) or len(rationale.strip()) < 12:
            errors.append(f"{axis}:empty_or_short")
        elif SCORE_LEAK_RE.search(rationale):
            errors.append(f"{axis}:score_leak")
    return value, errors


def prepare(run_id: str, prompt_path: Path) -> dict[str, Any]:
    restricted = RESTRICTED_ROOT / run_id
    output = OUTPUT_ROOT / run_id
    if restricted.exists() or output.exists():
        raise RationaleTestError("run directory already exists")
    restricted.mkdir(parents=True, mode=0o700)
    output.mkdir(parents=True, mode=0o700)
    prompt_path = checked_prompt_path(prompt_path)
    system, template = split_prompt(prompt_path)
    candidates, exclusions = load_candidates()
    sample = select_balanced(candidates)
    if histogram(sample) != {axis: {str(score): TARGET_PER_BAND for score in range(1, 6)} for axis in AXES}:
        raise RationaleTestError("sample histogram differs")
    for row in sample:
        rendered = render_user(template, row)
        if row["prompt"] not in rendered or row["essay"] not in rendered:
            raise RationaleTestError("rendered input lost canonical text")
        append_jsonl(restricted / "sample.jsonl", row)
    sample_sha = sha256_file(restricted / "sample.jsonl")
    manifest = {
        "schema_version": "mal2026-rationale-prompt-openai-test-v1",
        "status": "prepared",
        "run_id": run_id,
        "created_at": now(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_split": "train",
        "source_sha256": sha256_file(SOURCE_PATH),
        "prompt_sha256": sha256_file(prompt_path),
        "prompt_repository_path": str(prompt_path.relative_to(ROOT.resolve())),
        "sample_sha256": sample_sha,
        "sample_size": len(sample),
        "sample_histogram": histogram(sample),
        "selection_seed": SEED,
        "selection_uses": "integerized axis labels only; no model outputs",
        "rounding": "Decimal ROUND_HALF_UP",
        "models": list(MODELS),
        "api": {
            "endpoint": "/v1/responses",
            "reasoning_effort": "none",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "structured_output": "strict json_schema",
            "store": False,
            "temperature": "provider default; fixed aliases reject temperature",
        },
        "external_data_transfer_authorization": "user explicitly requested GPT-5.6 Luna and Terra rationale generation test",
        "exclusions": exclusions,
    }
    atomic_json(restricted / "manifest.json", manifest)
    atomic_json(output / "protocol.json", {key: value for key, value in manifest.items() if key not in {"exclusions"}} | {"exclusions": exclusions})
    append_jsonl(restricted / "ledger.jsonl", {"at": now(), "event": "prepared", "sample_sha256": sample_sha})
    return manifest


def load_sample(run_id: str) -> list[dict[str, Any]]:
    path = RESTRICTED_ROOT / run_id / "sample.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def one_request(run_id: str, model: str, row: Mapping[str, Any], system: str, template: str, prompt_sha256: str) -> dict[str, Any]:
    started = time.monotonic()
    payload = response_body(model, system, render_user(template, row))
    idem = sha256_bytes(f"{run_id}\0{model}\0{row['case_key']}\0{prompt_sha256}".encode())
    response = api_call(payload, idem)
    text = response_text(response)
    parsed, errors = validate_output(text)
    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
    return {
        "case_key": row["case_key"],
        "model": model,
        "integer_scores": row["integer_scores"],
        "response_id": response.get("id"),
        "response_status": response.get("status"),
        "output": parsed if parsed else None,
        "raw_output": text,
        "validation_errors": errors,
        "usage": dict(usage),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def run(run_id: str, max_workers: int, prompt_path: Path) -> dict[str, Any]:
    restricted = RESTRICTED_ROOT / run_id
    manifest_path = restricted / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"prepared", "running"}:
        raise RationaleTestError("run status is not resumable")
    prompt_path = checked_prompt_path(prompt_path)
    if sha256_file(SOURCE_PATH) != manifest["source_sha256"] or sha256_file(prompt_path) != manifest["prompt_sha256"]:
        raise RationaleTestError("canonical input hash changed after preparation")
    if str(prompt_path.relative_to(ROOT.resolve())) != manifest.get("prompt_repository_path"):
        raise RationaleTestError("prompt path changed after preparation")
    sample = load_sample(run_id)
    if sha256_file(restricted / "sample.jsonl") != manifest["sample_sha256"] or histogram(sample) != manifest["sample_histogram"]:
        raise RationaleTestError("sample attestation differs")
    system, template = split_prompt(prompt_path)
    manifest["status"] = "running"
    manifest["started_at"] = manifest.get("started_at", now())
    atomic_json(manifest_path, manifest)
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    results_path = restricted / "responses.jsonl"
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result = json.loads(line)
                completed[(result["model"], result["case_key"])] = result

    # Real low-band smoke for both provider aliases, then the remaining calls.
    smoke_row = min(sample, key=lambda row: (sum(row["integer_scores"].values()), row["case_key"]))
    for model in MODELS:
        key = (model, smoke_row["case_key"])
        if key not in completed:
            result = one_request(run_id, model, smoke_row, system, template, manifest["prompt_sha256"])
            append_jsonl(results_path, result)
            completed[key] = result
        if completed[key]["validation_errors"] or completed[key]["response_status"] != "completed":
            raise RationaleTestError(f"real API smoke failed for {model}")
        emit(run_id=run_id, phase="smoke", model=model, status="passed")

    pending = [(model, row) for model in MODELS for row in sample if (model, row["case_key"]) not in completed]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(one_request, run_id, model, row, system, template, manifest["prompt_sha256"]): (model, row["case_key"]) for model, row in pending}
        for future in as_completed(futures):
            result = future.result()
            append_jsonl(results_path, result)
            completed[(result["model"], result["case_key"])] = result
            emit(run_id=run_id, phase="generation", completed=len(completed), total=len(MODELS) * len(sample))
    if len(completed) != len(MODELS) * len(sample):
        raise RationaleTestError("response population differs")
    manifest["status"] = "generated"
    manifest["completed_at"] = now()
    manifest["responses_sha256"] = sha256_file(results_path)
    atomic_json(manifest_path, manifest)
    append_jsonl(restricted / "ledger.jsonl", {"at": now(), "event": "generated", "responses_sha256": manifest["responses_sha256"]})
    return {"run_id": run_id, "status": "generated", "responses": len(completed)}


def mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def quoted_spans(text: str) -> list[str]:
    values = re.findall(r"[“‘\"]([^”’\"]{2,120})[”’\"]", text)
    return [value for value in values if "..." not in value and "…" not in value]


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def summarize(run_id: str) -> dict[str, Any]:
    restricted = RESTRICTED_ROOT / run_id
    output = OUTPUT_ROOT / run_id
    manifest_path = restricted / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"generated", "completed"}:
        raise RationaleTestError("generation is incomplete")
    rows = [json.loads(line) for line in (restricted / "responses.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = {row["case_key"]: row for row in load_sample(run_id)}
    expected = len(MODELS) * SAMPLE_SIZE
    if len(rows) != expected or len({(row["model"], row["case_key"]) for row in rows}) != expected:
        raise RationaleTestError("response population differs")
    models: dict[str, Any] = {}
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        axis_lengths: dict[str, list[int]] = {axis: [] for axis in AXES}
        by_band: dict[str, dict[str, list[dict[str, int]]]] = {axis: {str(score): [] for score in range(1, 6)} for axis in AXES}
        score_leaks = advice_like = 0
        quote_count = grounded_quote_count = 0
        for row in selected:
            source_text = compact_text(samples[row["case_key"]]["prompt"] + "\n" + samples[row["case_key"]]["essay"])
            for axis in AXES:
                rationale = row["output"][axis]["rationale"] if row["output"] else ""
                axis_lengths[axis].append(len(rationale))
                positive = sum(rationale.count(term) for term in POSITIVE_TERMS)
                negative = sum(rationale.count(term) for term in NEGATIVE_TERMS)
                by_band[axis][str(row["integer_scores"][axis])].append({"positive": positive, "negative": negative})
                score_leaks += int(bool(SCORE_LEAK_RE.search(rationale)))
                advice_like += int(any(term in rationale for term in ADVICE_TERMS))
                spans = quoted_spans(rationale)
                quote_count += len(spans)
                grounded_quote_count += sum(compact_text(span) in source_text for span in spans)
        band_diagnostics = {
            axis: {
                score: {
                    "positive_terms_mean": mean([item["positive"] for item in values]),
                    "negative_terms_mean": mean([item["negative"] for item in values]),
                    "polarity_mean": mean([item["positive"] - item["negative"] for item in values]),
                }
                for score, values in bands.items()
            }
            for axis, bands in by_band.items()
        }
        usage = Counter()
        for row in selected:
            usage.update({str(key): int(value) for key, value in row.get("usage", {}).items() if type(value) is int})
        models[model] = {
            "requests": len(selected),
            "completed_responses": sum(row.get("response_status") == "completed" for row in selected),
            "strict_schema_passes": sum(not row["validation_errors"] for row in selected),
            "score_leak_axis_outputs": score_leaks,
            "advice_like_axis_outputs": advice_like,
            "quoted_spans": quote_count,
            "whitespace_normalized_grounded_quoted_spans": grounded_quote_count,
            "grounded_quote_rate": round(grounded_quote_count / quote_count, 4) if quote_count else None,
            "mean_rationale_characters": {axis: mean(values) for axis, values in axis_lengths.items()},
            "mean_elapsed_seconds": mean([float(row["elapsed_seconds"]) for row in selected]),
            "band_lexicon_diagnostic": band_diagnostics,
            "usage": dict(usage),
        }
    aggregate = {
        "schema_version": "mal2026-rationale-prompt-openai-test-aggregate-v1",
        "status": "completed",
        "run_id": run_id,
        "source_split": "train",
        "prompt_sha256": manifest["prompt_sha256"],
        "source_sha256": manifest["source_sha256"],
        "sample_size": SAMPLE_SIZE,
        "sample_histogram": manifest["sample_histogram"],
        "models": models,
        "limitations": [
            "Lexicon polarity is a mechanical diagnostic, not a quality judgment.",
            "This balanced 15-row test estimates prompt behavior, not downstream SFT performance.",
        ],
    }
    atomic_json(output / "aggregate.json", aggregate)
    manifest["status"] = "completed"
    manifest["aggregate_sha256"] = sha256_file(output / "aggregate.json")
    atomic_json(manifest_path, manifest)
    append_jsonl(restricted / "ledger.jsonl", {"at": now(), "event": "completed", "aggregate_sha256": manifest["aggregate_sha256"]})
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "summarize", "all"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_PATH)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,100}", args.run_id):
        raise RationaleTestError("invalid run id")
    if not 1 <= args.max_workers <= 8:
        raise RationaleTestError("max-workers must be in [1,8]")
    setproctitle(f"mal2026:rationale-prompt-openai:{args.command}:{args.run_id}")
    if args.command in {"prepare", "all"}:
        emit(**prepare(args.run_id, args.prompt_file))
    if args.command in {"run", "all"}:
        emit(**run(args.run_id, args.max_workers, args.prompt_file))
    if args.command in {"summarize", "all"}:
        emit(**summarize(args.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
