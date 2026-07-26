#!/usr/bin/env python3
"""Generate three train-only candidates with the official participant prompt.

All prompts, essays, mappings, provider responses, and candidate text stay in
the ignored restricted root.  Stdout and tracked records contain aggregate
counts, usage, hashes, and failure categories only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import sys
import uuid
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_writing_contract import (  # noqa: E402
    OFFICIAL_INFERENCE_SYSTEM_PROMPT,
    participant_json_schema,
    parse_participant_output,
)


API_ROOT = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.6-terra"
SCHEMA_VERSION = "mal2026-official-openai-candidate-v1"
EXPECTED_TRAIN_ROWS = 2000
EXPECTED_CANDIDATES = 3
SOURCE = ROOT / "eval/train.jsonl"
SOURCE_SHA256 = "b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737"
ARTIFACT_ROOT = ROOT / "data/processed/restricted/official_openai_candidates_v1"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def emit(run_id: str, **values: Any) -> None:
    print(json.dumps({"run_id": run_id, **values}, ensure_ascii=False, sort_keys=True))


def run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,100}", run_id):
        raise ValueError("run id contains unsupported characters")
    return ARTIFACT_ROOT / run_id


def manifest_path(run_id: str) -> Path:
    return run_dir(run_id) / "manifest.json"


def read_manifest(run_id: str) -> dict[str, Any]:
    value = json.loads(manifest_path(run_id).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("manifest is not an object")
    return value


def write_manifest(run_id: str, manifest: Mapping[str, Any]) -> None:
    atomic_json(manifest_path(run_id), manifest)


def api_key() -> str:
    value = os.environ.get("OPENAI_API_KEY")
    if value:
        return value
    env = ROOT / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            value = line.split("=", 1)[1].strip().strip("\"'")
            if value:
                return value
    raise RuntimeError("OPENAI_API_KEY is unavailable")


def request_json(method: str, path: str, key: str, payload: Mapping[str, Any] | None = None, headers: Mapping[str, str] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(f"{API_ROOT}{path}", data=body, method=method)
    request.add_header("Authorization", f"Bearer {key}")
    request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urlopen(request, timeout=300) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"OpenAI {method} {path} failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI {method} {path} network failure") from exc
    if not isinstance(value, dict):
        raise RuntimeError("OpenAI response is not an object")
    return value


def request_bytes(path: str, key: str) -> bytes:
    request = Request(f"{API_ROOT}{path}", method="GET")
    request.add_header("Authorization", f"Bearer {key}")
    try:
        with urlopen(request, timeout=600) as response:
            return response.read()
    except HTTPError as exc:
        raise RuntimeError(f"OpenAI GET {path} failed with HTTP {exc.code}") from exc


def upload_file(path: Path, key: str, idempotency_key: str) -> dict[str, Any]:
    boundary = f"----mal2026-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/jsonl"
    body = b"".join([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode(),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
         f"Content-Type: {content_type}\r\n\r\n").encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = Request(f"{API_ROOT}/files", data=body, method="POST")
    request.add_header("Authorization", f"Bearer {key}")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    request.add_header("Idempotency-Key", idempotency_key)
    try:
        with urlopen(request, timeout=600) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"OpenAI file upload failed with HTTP {exc.code}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("id"), str):
        raise RuntimeError("OpenAI file upload envelope differs")
    return value


def stable_key(*parts: str) -> str:
    return sha256("\0".join(parts).encode()).hexdigest()


def response_text(response: Mapping[str, Any]) -> str:
    for item in response.get("output", []):
        if not isinstance(item, Mapping):
            continue
        for content in item.get("content", []):
            if isinstance(content, Mapping) and content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return str(content["text"])
    raise ValueError("response has no output_text")


def load_rows(limit: int | None = None) -> list[dict[str, Any]]:
    if file_sha(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("canonical train checksum changed")
    rows: list[dict[str, Any]] = []
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if set(value) != {"id", "document_id", "prompt_num", "prompt", "essay", "score"}:
                    raise RuntimeError("canonical train row schema changed")
                rows.append(value)
                if limit is not None and len(rows) >= limit:
                    break
    expected = limit if limit is not None else EXPECTED_TRAIN_ROWS
    if len(rows) != expected:
        raise RuntimeError("canonical train row count changed")
    return rows


DIVERSITY = {
    1: "세 평가 영역을 균형 있게 직접 채점하고, 각 rationale에 가장 결정적인 글의 자질을 제시하라.",
    2: "공식 기준은 그대로 적용하되, 주장·문단·표현에서 확인 가능한 구체적 증거와 결함을 특히 엄격히 밝혀라.",
    3: "공식 기준은 그대로 적용하되, 과도한 고득점을 피하고 각 정수 점수를 정당화하는 강점과 약점을 분명히 대비하라.",
}


def user_prompt(row: Mapping[str, Any], candidate: int) -> str:
    if candidate not in DIVERSITY:
        raise ValueError("candidate must be 1, 2, or 3")
    return (
        f"[후보 생성 지침]\n{DIVERSITY[candidate]}\n\n"
        f"[prompt_text]\n{row['prompt']}\n\n[essay_text]\n{row['essay']}"
    )


def response_body(row: Mapping[str, Any], candidate: int, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": OFFICIAL_INFERENCE_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt(row, candidate)}]},
        ],
        "text": {"format": {"type": "json_schema", "name": "mal2026_official_participant_output", "strict": True, "schema": participant_json_schema()}},
        "reasoning": {"effort": "none"},
        "max_output_tokens": 1200,
        "store": False,
    }


def validate_output(response: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]] | None, list[str]]:
    try:
        return parse_participant_output(response_text(response)), []
    except Exception as exc:
        return None, [type(exc).__name__]


def prepare(args: argparse.Namespace) -> None:
    if args.model != DEFAULT_MODEL or args.candidates != EXPECTED_CANDIDATES:
        raise ValueError("official v1 is pinned to gpt-5.6-terra and exactly three candidates")
    destination = run_dir(args.run_id)
    if destination.exists():
        raise FileExistsError("run directory already exists")
    destination.mkdir(mode=0o700, parents=True)
    rows = load_rows()
    salt = secrets.token_bytes(32)
    request_path = destination / "requests.jsonl"
    mapping_path = destination / "source_map.jsonl"
    with request_path.open("x", encoding="utf-8") as requests, mapping_path.open("x", encoding="utf-8") as mappings:
        for row in rows:
            source_id = str(row["id"])
            token = hmac.new(salt, source_id.encode(), sha256).hexdigest()[:24]
            essay_hash = sha256(str(row["essay"]).encode()).hexdigest()
            for candidate in range(1, EXPECTED_CANDIDATES + 1):
                custom_id = f"{args.run_id}:train:{candidate}:{token}"
                requests.write(json.dumps({"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": response_body(row, candidate, args.model)}, ensure_ascii=False) + "\n")
                mappings.write(json.dumps({"custom_id": custom_id, "source_id": source_id, "split": "train", "candidate": candidate, "essay_sha256": essay_hash}, ensure_ascii=False) + "\n")
    expected = EXPECTED_TRAIN_ROWS * EXPECTED_CANDIDATES
    if line_count(request_path) != expected or line_count(mapping_path) != expected:
        raise RuntimeError("prepared request/mapping count differs")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "run_id": args.run_id,
        "created_at": now(),
        "model": args.model,
        "split": "train",
        "train_rows": len(rows),
        "candidates_per_essay": EXPECTED_CANDIDATES,
        "requests": expected,
        "request_file": request_path.name,
        "request_sha256": file_sha(request_path),
        "request_bytes": request_path.stat().st_size,
        "source_map_sha256": file_sha(mapping_path),
        "source_sha256": SOURCE_SHA256,
        "human_or_reference_score_read_or_prompted": False,
        "official_system_prompt_sha256": sha256(OFFICIAL_INFERENCE_SYSTEM_PROMPT.encode()).hexdigest(),
        "candidate_diversity_instructions": {str(key): value for key, value in DIVERSITY.items()},
        "events": [{"at": now(), "event": "prepared"}],
    }
    write_manifest(args.run_id, manifest)
    emit(args.run_id, status="prepared", requests=expected, request_bytes=manifest["request_bytes"], request_sha256=manifest["request_sha256"])


def smoke(args: argparse.Namespace) -> None:
    if args.model != DEFAULT_MODEL or args.candidate not in DIVERSITY:
        raise ValueError("official smoke identity differs")
    row = load_rows(1)[0]
    response = request_json("POST", "/responses", api_key(), response_body(row, args.candidate, args.model))
    parsed, errors = validate_output(response)
    destination = run_dir(args.run_id)
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_json(destination / "smoke_result.json", {"tested_at": now(), "model": args.model, "candidate": args.candidate, "response": response, "schema_valid": not errors, "validation_errors": errors})
    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
    emit(args.run_id, status="smoke_complete", schema_valid=not errors, validation_errors=errors, usage=usage, parsed_axes=sorted(parsed) if parsed else [])
    if errors:
        raise SystemExit(2)


def submit(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.run_id)
    if manifest.get("batch_id"):
        emit(args.run_id, status=manifest["status"], duplicate_submission_prevented=True)
        return
    if manifest.get("status") != "prepared" or manifest.get("requests") != EXPECTED_TRAIN_ROWS * EXPECTED_CANDIDATES:
        raise RuntimeError("only a complete prepared official batch can be submitted")
    request_path = run_dir(args.run_id) / str(manifest["request_file"])
    if file_sha(request_path) != manifest["request_sha256"]:
        raise RuntimeError("prepared request checksum changed")
    key = api_key()
    upload_key = stable_key("mal2026-official-upload", args.run_id, manifest["request_sha256"])
    batch_key = stable_key("mal2026-official-batch", args.run_id, manifest["request_sha256"])
    manifest["events"].append({"at": now(), "event": "submit_intent", "idempotency_key_sha256": sha256(batch_key.encode()).hexdigest()})
    write_manifest(args.run_id, manifest)
    uploaded = upload_file(request_path, key, upload_key)
    manifest["input_file_id"] = uploaded["id"]
    manifest["events"].append({"at": now(), "event": "uploaded"})
    write_manifest(args.run_id, manifest)
    batch = request_json("POST", "/batches", key, {"input_file_id": uploaded["id"], "endpoint": "/v1/responses", "completion_window": "24h", "metadata": {"run_id": args.run_id, "artifact": "mal2026_official_train_candidates_v1"}}, headers={"Idempotency-Key": batch_key})
    manifest.update({"status": "submitted", "batch_id": batch["id"], "submitted_at": now(), "request_counts": batch.get("request_counts")})
    manifest["events"].append({"at": now(), "event": "submitted"})
    write_manifest(args.run_id, manifest)
    emit(args.run_id, status="submitted", request_counts=manifest.get("request_counts"))


def poll(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.run_id)
    if not manifest.get("batch_id"):
        raise RuntimeError("batch is not submitted")
    batch = request_json("GET", f"/batches/{manifest['batch_id']}", api_key())
    manifest.update({"status": batch["status"], "last_polled_at": now(), "request_counts": batch.get("request_counts"), "output_file_id": batch.get("output_file_id"), "error_file_id": batch.get("error_file_id")})
    manifest["events"].append({"at": now(), "event": "polled", "batch_status": batch["status"]})
    write_manifest(args.run_id, manifest)
    emit(args.run_id, status=batch["status"], request_counts=batch.get("request_counts"))


def records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row["custom_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def download(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.run_id)
    if manifest.get("status") == "validated":
        emit(args.run_id, status="validated", duplicate_download_prevented=True, accepted=manifest.get("accepted"))
        return
    # v1 initially inserted ``missing_batch_record: 0`` into a Counter and
    # then used ``not failures`` as the gate.  Counter truthiness is based on
    # key presence, so an otherwise perfect 6,000/6,000 download was marked
    # failed.  Recover that exact integration-only state without downloading
    # again or replacing any preserved row artifact.
    if manifest.get("status") == "failed_validation" and manifest.get("accepted") == manifest.get("expected"):
        counts = manifest.get("failure_categories")
        destination = run_dir(args.run_id)
        required = {
            "output_sha256": destination / "batch_output.jsonl",
            "candidates_sha256": destination / "candidates.train.jsonl",
            "errors_sha256": destination / "errors.jsonl",
        }
        intact = (
            isinstance(counts, dict)
            and counts
            and all(type(value) is int and value == 0 for value in counts.values())
            and all(path.is_file() and file_sha(path) == manifest.get(key) for key, path in required.items())
            and sum(1 for line in required["candidates_sha256"].open(encoding="utf-8") if line.strip()) == int(manifest["expected"])
            and required["errors_sha256"].stat().st_size == 0
        )
        if intact:
            manifest["status"] = "validated"
            manifest["events"].append({"at": now(), "event": "validation_status_repaired", "reason": "zero_count_counter_truthiness"})
            write_manifest(args.run_id, manifest)
            atomic_json(destination / "aggregate_validation.json", {key: manifest[key] for key in ("schema_version", "status", "run_id", "model", "train_rows", "candidates_per_essay", "expected", "accepted", "failure_categories", "source_sha256", "request_sha256", "candidates_sha256", "human_or_reference_score_read_or_prompted")})
            emit(args.run_id, status="validated", validation_status_repaired=True, accepted=manifest.get("accepted"))
            return
    if manifest.get("status") != "completed" or not manifest.get("output_file_id"):
        raise RuntimeError("batch is not complete")
    destination = run_dir(args.run_id)
    raw_path = destination / "batch_output.jsonl"
    raw_path.write_bytes(request_bytes(f"/files/{manifest['output_file_id']}/content", api_key()))
    if manifest.get("error_file_id"):
        (destination / "batch_error.jsonl").write_bytes(request_bytes(f"/files/{manifest['error_file_id']}/content", api_key()))
    mappings = records_by_id(destination / "source_map.jsonl")
    rows = {str(row["id"]): row for row in load_rows()}
    accepted: set[str] = set()
    failures: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    candidate_path = destination / "candidates.train.jsonl"
    with candidate_path.open("x", encoding="utf-8") as output:
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            result = json.loads(line)
            custom_id = result.get("custom_id")
            mapping = mappings.get(custom_id)
            if mapping is None:
                failures["unknown_custom_id"] += 1
                continue
            body = result.get("response", {}).get("body", {})
            parsed, validation_errors = validate_output(body)
            if validation_errors or parsed is None:
                category = validation_errors[0] if validation_errors else "invalid_output"
                failures[category] += 1
                errors.append({"custom_id": custom_id, "failure_category": category})
                continue
            row = rows[mapping["source_id"]]
            if mapping["essay_sha256"] != sha256(str(row["essay"]).encode()).hexdigest():
                failures["essay_linkage"] += 1
                errors.append({"custom_id": custom_id, "failure_category": "essay_linkage"})
                continue
            record = {**mapping, "model": manifest["model"], "schema_version": SCHEMA_VERSION, "participant_output": parsed, "api_response_id": body.get("id")}
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            accepted.add(str(custom_id))
    missing = set(mappings) - accepted - {str(row["custom_id"]) for row in errors}
    if missing:
        failures["missing_batch_record"] += len(missing)
    error_path = destination / "errors.jsonl"
    with error_path.open("x", encoding="utf-8") as output:
        for row in errors:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        for custom_id in sorted(missing):
            output.write(json.dumps({"custom_id": custom_id, "failure_category": "missing_batch_record"}, separators=(",", ":")) + "\n")
    expected = EXPECTED_TRAIN_ROWS * EXPECTED_CANDIDATES
    manifest.update({
        "status": "validated" if len(accepted) == expected and sum(failures.values()) == 0 else "failed_validation",
        "downloaded_at": now(),
        "accepted": len(accepted),
        "expected": expected,
        "failure_categories": dict(sorted(failures.items())),
        "output_sha256": file_sha(raw_path),
        "candidates_sha256": file_sha(candidate_path),
        "errors_sha256": file_sha(error_path),
        "human_or_reference_score_read_or_prompted": False,
    })
    manifest["events"].append({"at": now(), "event": "validated", "status": manifest["status"]})
    write_manifest(args.run_id, manifest)
    atomic_json(destination / "aggregate_validation.json", {key: manifest[key] for key in ("schema_version", "status", "run_id", "model", "train_rows", "candidates_per_essay", "expected", "accepted", "failure_categories", "source_sha256", "request_sha256", "candidates_sha256", "human_or_reference_score_read_or_prompted")})
    emit(args.run_id, status=manifest["status"], accepted=len(accepted), expected=expected, failure_categories=dict(failures))
    if manifest["status"] != "validated":
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-id", required=True)
    common.add_argument("--model", default=DEFAULT_MODEL)
    prepare_parser = commands.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--candidates", type=int, default=EXPECTED_CANDIDATES)
    prepare_parser.set_defaults(func=prepare)
    smoke_parser = commands.add_parser("smoke", parents=[common])
    smoke_parser.add_argument("--candidate", type=int, choices=sorted(DIVERSITY), default=1)
    smoke_parser.set_defaults(func=smoke)
    commands.add_parser("submit", parents=[common]).set_defaults(func=submit)
    commands.add_parser("poll", parents=[common]).set_defaults(func=poll)
    commands.add_parser("download", parents=[common]).set_defaults(func=download)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
