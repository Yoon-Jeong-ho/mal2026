#!/usr/bin/env python3
"""Generate frozen-v3 rationale candidates with tail-aware multiplicity.

All prompts, source mappings, provider responses, and generated rationales are
restricted artifacts.  Stdout and the aggregate report contain counts only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import hmac
import json
import os
from pathlib import Path
import random
import re
import secrets
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Mapping, Sequence

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_openai_rationales as api  # noqa: E402
import test_rationale_generation_prompt_openai as prompt_contract  # noqa: E402


ARTIFACT_ROOT = ROOT / "data/processed/restricted/rationale_v3_tail_batches"
AGGREGATE_ROOT = ROOT / "outputs/rationale-v3-tail-batches"
PROMPT = ROOT / "rationale_generation_prompt_v3.txt"
PROMPT_SHA256 = "b71ee648b9a6707c1e0156681adb9c4d47a3a4a4b751aa2cb90d0bc8808981c6"
BASELINE_SAMPLE = ROOT / "data/processed/restricted/rationale_prompt_openai_test/rationale-prompt-balanced15-gpt56-20260806-001/sample.jsonl"
MODELS = ("gpt-5.6-luna", "gpt-5.6-terra")
AXES = ("content", "organization", "expression")
PILOT_ROWS = 200
PILOT_TAIL_ROWS = 130
SELECTION_SEED = 2026080701
MAX_OUTPUT_TOKENS = 1800
VARIANT_DIRECTIVES = {
    1: "",
    2: "각 영역에서 가장 먼저 떠오르는 관찰을 반복하기보다, 같은 판단을 독립적으로 뒷받침하는 다른 위치나 다른 하위 기준의 근거가 실제로 있으면 그것을 우선하라. 새로운 근거를 발명하지 마라.",
    3: "각 영역에서 글의 앞부분과 뒷부분에 걸친 변화·반복·일관성 중 실제로 확인되는 양상을 우선하여 판단 범위를 설명하라. 해당 양상이 없으면 가장 강한 원문 근거를 사용하라.",
    4: "각 영역에서 한 가지 관찰을 단순히 다시 말하지 말고, 실제로 확인되는 핵심 양상이 해당 영역의 성취나 제한에 미치는 영향을 중심으로 간결하게 설명하라.",
}
RESPONSE_SCHEMA = prompt_contract.RESPONSE_SCHEMA
FOREIGN_SCRIPT_RE = re.compile(r"[\u0400-\u052f\u0600-\u06ff\u0900-\u097f\u3040-\u30ff]")


def now() -> str:
    return api.now()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_dir(run_id: str) -> Path:
    return ARTIFACT_ROOT / run_id


def aggregate_dir(run_id: str) -> Path:
    return AGGREGATE_ROOT / run_id


def manifest_path(run_id: str) -> Path:
    return run_dir(run_id) / "manifest.json"


def read_manifest(run_id: str) -> dict[str, Any]:
    return json.loads(manifest_path(run_id).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def write_manifest(run_id: str, value: dict[str, Any]) -> None:
    write_json(manifest_path(run_id), value)


def emit(run_id: str, **values: Any) -> None:
    print(json.dumps({"run_id": run_id, **values}, ensure_ascii=False, sort_keys=True), flush=True)


def round_half_up(value: Any) -> int:
    number = Decimal(str(value))
    need(number.is_finite() and Decimal("1") <= number <= Decimal("5"), "score outside finite [1,5]")
    return int(number.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def load_rows(split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (ROOT / "eval" / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            scores = {axis: round_half_up(raw["score"][axis]) for axis in AXES}
            rows.append({
                "source_id": str(raw["id"]), "document_id": str(raw["document_id"]),
                "line_number": line_number, "prompt_num": str(raw["prompt_num"]),
                "prompt": str(raw["prompt"]), "essay": str(raw["essay"]),
                "integer_scores": scores,
                "essay_sha256": hashlib.sha256(str(raw["essay"]).encode()).hexdigest(),
            })
    expected = 2000 if split == "train" else 400
    need(len(rows) == expected and len({row["source_id"] for row in rows}) == expected, f"canonical {split} population differs")
    return rows


def target_multiplicity(scores: Mapping[str, int]) -> int:
    values = set(scores.values())
    if 1 in values:
        return 4
    if values & {2, 5}:
        return 2
    return 1


def excluded_pilot_ids() -> set[str]:
    need(BASELINE_SAMPLE.is_file(), "baseline optimization sample unavailable")
    return {
        str(json.loads(line)["source_id"])
        for line in BASELINE_SAMPLE.read_text(encoding="utf-8").splitlines() if line.strip()
    }


def cells(row: Mapping[str, Any], allowed_scores: set[int]) -> list[tuple[str, int]]:
    return [(axis, int(row["integer_scores"][axis])) for axis in AXES if int(row["integer_scores"][axis]) in allowed_scores]


def balanced_greedy(pool: Sequence[dict[str, Any]], count: int, allowed_scores: set[int], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tie = {row["source_id"]: rng.random() for row in pool}
    selected: list[dict[str, Any]] = []
    remaining = list(pool)
    cell_counts: Counter[tuple[str, int]] = Counter()
    prompt_counts: Counter[str] = Counter()
    while len(selected) < count:
        need(bool(remaining), "pilot selection pool exhausted")
        def utility(row: Mapping[str, Any]) -> tuple[float, float, float]:
            relevant = cells(row, allowed_scores)
            coverage = sum(1.0 / (1.0 + cell_counts[cell]) for cell in relevant)
            prompt_balance = 1.0 / (1.0 + prompt_counts[str(row["prompt_num"])])
            return coverage, prompt_balance, tie[str(row["source_id"])]
        chosen = max(remaining, key=utility)
        remaining.remove(chosen)
        selected.append(chosen)
        cell_counts.update(cells(chosen, allowed_scores))
        prompt_counts[str(chosen["prompt_num"])] += 1
    return selected


def pilot_rows() -> list[dict[str, Any]]:
    excluded = excluded_pilot_ids()
    candidates = [row for row in load_rows("train") if row["source_id"] not in excluded]
    any_one = [row for row in candidates if 1 in row["integer_scores"].values()]
    tail = [row for row in candidates if row not in any_one and set(row["integer_scores"].values()) & {2, 5}]
    central = [row for row in candidates if row not in any_one and row not in tail]
    need(len(any_one) < PILOT_TAIL_ROWS, "pilot score-1 population unexpectedly large")
    selected_tail = list(any_one)
    selected_tail.extend(balanced_greedy(tail, PILOT_TAIL_ROWS - len(selected_tail), {2, 5}, SELECTION_SEED))
    selected_ids = {row["source_id"] for row in selected_tail}
    central_pool = [row for row in central if row["source_id"] not in selected_ids]
    selected_central = balanced_greedy(central_pool, PILOT_ROWS - len(selected_tail), {3, 4}, SELECTION_SEED + 1)
    selected = selected_tail + selected_central
    need(len(selected) == PILOT_ROWS and len({row["source_id"] for row in selected}) == PILOT_ROWS, "pilot selection differs")
    selected.sort(key=lambda row: row["line_number"])
    return selected


def selected_rows(scope: str) -> tuple[str, list[dict[str, Any]]]:
    if scope == "pilot":
        return "train", pilot_rows()
    if scope == "train":
        return "train", load_rows("train")
    if scope == "validation":
        return "validation", load_rows("validation")
    raise RuntimeError("unknown generation scope")


def render_user(template: str, row: Mapping[str, Any], variant: int) -> str:
    value = prompt_contract.render_user(template, row)
    directive = VARIANT_DIRECTIVES[variant]
    if directive:
        value += "\n\n[후보 근거 선택 지침]\n" + directive
    return value


def response_body(model: str, system: str, user: str) -> dict[str, Any]:
    need(model in MODELS, "model alias differs")
    return {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [{"type": "input_text", "text": user}]},
        ],
        "text": {"format": {"type": "json_schema", "name": "mal2026_rationale_generation", "strict": True, "schema": RESPONSE_SCHEMA}},
        "reasoning": {"effort": "none"},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
    }


def score_histogram(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        axis: {str(score): sum(int(row["integer_scores"][axis]) == score for row in rows) for score in range(1, 6)}
        for axis in AXES
    }


def prepare(args: argparse.Namespace) -> None:
    need(args.model in MODELS, "unsupported model")
    need(api.sha256_file(PROMPT) == PROMPT_SHA256, "frozen v3 prompt differs")
    destination = run_dir(args.run_id)
    public = aggregate_dir(args.run_id)
    need(not destination.exists() and not public.exists(), "run output must be fresh")
    destination.mkdir(parents=True, mode=0o700)
    public.mkdir(parents=True)
    split, rows = selected_rows(args.scope)
    system, template = prompt_contract.split_prompt(PROMPT)
    salt = secrets.token_bytes(32)
    requests_path = destination / "requests.jsonl"
    source_map_path = destination / "source_map.jsonl"
    request_count = 0
    multiplicities: Counter[int] = Counter()
    with requests_path.open("x", encoding="utf-8") as request_file, source_map_path.open("x", encoding="utf-8") as map_file:
        for row in rows:
            multiplicity = target_multiplicity(row["integer_scores"])
            multiplicities[multiplicity] += 1
            opaque = hmac.new(salt, f"{split}\0{row['source_id']}".encode(), hashlib.sha256).hexdigest()[:24]
            for variant in range(1, multiplicity + 1):
                custom_id = f"{args.model[-1]}-{split[0]}-v{variant}-{opaque}"
                request = {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": response_body(args.model, system, render_user(template, row, variant))}
                mapping = {
                    "custom_id": custom_id, "source_id": row["source_id"], "split": split,
                    "variant": variant, "target_multiplicity": multiplicity,
                    "integer_scores": row["integer_scores"], "essay_sha256": row["essay_sha256"],
                }
                request_file.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
                map_file.write(json.dumps(mapping, ensure_ascii=False, separators=(",", ":")) + "\n")
                request_count += 1
    os.chmod(requests_path, 0o600); os.chmod(source_map_path, 0o600)
    need(api.line_count(requests_path) == api.line_count(source_map_path) == request_count, "request population differs")
    variants_sha = hashlib.sha256(json.dumps(VARIANT_DIRECTIVES, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    manifest = {
        "schema_version": "mal2026-rationale-v3-tail-batch-v1", "status": "prepared",
        "run_id": args.run_id, "created_at": now(), "model": args.model, "scope": args.scope,
        "split": split, "source_rows": len(rows), "requests": request_count,
        "row_multiplicity": {str(key): value for key, value in sorted(multiplicities.items())},
        "score_histogram": score_histogram(rows), "selection_seed": SELECTION_SEED if args.scope == "pilot" else None,
        "pilot_excludes_prompt_optimization_rows": args.scope == "pilot",
        "prompt_file": str(PROMPT.relative_to(ROOT)), "prompt_sha256": PROMPT_SHA256,
        "variant_directives_sha256": variants_sha, "variant_directives": VARIANT_DIRECTIVES,
        "request_file": requests_path.name, "request_sha256": api.sha256_file(requests_path),
        "request_bytes": requests_path.stat().st_size, "source_map_sha256": api.sha256_file(source_map_path),
        "source_file_sha256": api.sha256_file(ROOT / "eval" / f"{split}.jsonl"),
        "api": {"endpoint": "/v1/responses", "reasoning_effort": "none", "max_output_tokens": MAX_OUTPUT_TOKENS, "strict_json_schema": True, "store": False},
        "tail_multiplicity": {"score_1": 4, "score_2": 2, "score_5": 2, "score_3_or_4_only": 1, "row_rule": "maximum across three axes"},
        "external_data_transfer_authorization": "2026-08-07 user approved both Luna and Terra, a train pilot, and full train/validation generation with extra 1/2/5-score rationales",
        "validation_policy": "validation may be generated only after frozen train-pilot gates and is never used for prompt/model selection",
        "events": [{"at": now(), "event": "prepared"}],
    }
    write_manifest(args.run_id, manifest)
    write_json(public / "protocol.json", {key: value for key, value in manifest.items() if key not in {"variant_directives"}})
    emit(args.run_id, status="prepared", model=args.model, scope=args.scope, source_rows=len(rows), requests=request_count, row_multiplicity=manifest["row_multiplicity"])


def smoke(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.run_id)
    need(manifest["status"] == "prepared" and manifest["model"] == args.model, "smoke manifest differs")
    request = json.loads((run_dir(args.run_id) / "requests.jsonl").read_text(encoding="utf-8").splitlines()[0])
    response = api.request_json("POST", "/responses", api.load_env_key(), request["body"])
    text = api.get_text(response)
    parsed, errors = prompt_contract.validate_output(text)
    write_json(run_dir(args.run_id) / "smoke_response.json", {"at": now(), "model": args.model, "response": response, "parsed": parsed, "validation_errors": errors})
    manifest["smoke"] = {"at": now(), "status": "passed" if not errors else "failed", "validation_errors": errors}
    manifest["events"].append({"at": now(), "event": "smoke", "status": manifest["smoke"]["status"]})
    write_manifest(args.run_id, manifest)
    emit(args.run_id, status="smoke_complete", model=args.model, schema_valid=not errors)
    if errors:
        raise SystemExit(2)


def submit(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.run_id)
    need(manifest["model"] == args.model and manifest.get("smoke", {}).get("status") == "passed", "passing real smoke is required")
    if manifest.get("batch_id"):
        emit(args.run_id, status=manifest["status"], duplicate_submission_prevented=True)
        return
    request_path = run_dir(args.run_id) / manifest["request_file"]
    need(api.sha256_file(request_path) == manifest["request_sha256"], "request attestation differs")
    key = api.load_env_key()
    upload_key = api.stable_key("mal2026-v3-tail-upload", args.run_id, manifest["request_sha256"])
    batch_key = api.stable_key("mal2026-v3-tail-batch", args.run_id, manifest["request_sha256"])
    manifest["events"].append({"at": now(), "event": "submit_intent", "idempotency_key_sha256": hashlib.sha256(batch_key.encode()).hexdigest()})
    write_manifest(args.run_id, manifest)
    uploaded = api.upload_file(request_path, key, upload_key)
    manifest["input_file_id"] = uploaded["id"]
    manifest["events"].append({"at": now(), "event": "uploaded"})
    write_manifest(args.run_id, manifest)
    batch = api.request_json("POST", "/batches", key, {
        "input_file_id": uploaded["id"], "endpoint": "/v1/responses", "completion_window": "24h",
        "metadata": {"run_id": args.run_id, "artifact": "mal2026_rationale_v3_tail"},
    }, headers={"Idempotency-Key": batch_key})
    manifest.update({"status": "submitted", "submitted_at": now(), "batch_id": batch["id"], "request_counts": batch.get("request_counts")})
    manifest["events"].append({"at": now(), "event": "submitted"})
    write_manifest(args.run_id, manifest)
    emit(args.run_id, status="submitted", batch_id=batch["id"], request_counts=batch.get("request_counts"))


def poll(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.run_id)
    need(manifest.get("batch_id"), "batch has not been submitted")
    batch = api.request_json("GET", f"/batches/{manifest['batch_id']}", api.load_env_key())
    manifest.update({"status": batch["status"], "last_polled_at": now(), "request_counts": batch.get("request_counts"), "output_file_id": batch.get("output_file_id"), "error_file_id": batch.get("error_file_id")})
    manifest["events"].append({"at": now(), "event": "polled", "batch_status": batch["status"]})
    write_manifest(args.run_id, manifest)
    emit(args.run_id, status=batch["status"], request_counts=batch.get("request_counts"))


def jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {row["custom_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def output_text(body: Mapping[str, Any]) -> str:
    return api.get_text(dict(body))


def download(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.run_id)
    need(manifest.get("status") == "completed" and manifest.get("output_file_id"), "completed batch output unavailable")
    destination = run_dir(args.run_id)
    raw_path = destination / "batch_output.jsonl"
    need(not raw_path.exists(), "batch output already downloaded")
    raw_path.write_bytes(api.request_bytes(f"/files/{manifest['output_file_id']}/content", api.load_env_key()))
    os.chmod(raw_path, 0o600)
    mappings = jsonl_by_id(destination / "source_map.jsonl")
    split, rows = selected_rows(manifest["scope"])
    sources = {row["source_id"]: row for row in rows}
    accepted_path = destination / "candidates.jsonl"
    errors_path = destination / "errors.jsonl"
    accepted: set[str] = set(); errors: list[dict[str, Any]] = []
    with accepted_path.open("x", encoding="utf-8") as output:
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            envelope = json.loads(line); custom_id = envelope.get("custom_id"); mapping = mappings.get(custom_id)
            if mapping is None:
                errors.append({"custom_id": custom_id, "validation_errors": ["unknown_custom_id"]}); continue
            response = envelope.get("response") if isinstance(envelope.get("response"), Mapping) else {}
            body = response.get("body") if response.get("status_code") == 200 and isinstance(response.get("body"), Mapping) else None
            try:
                need(body is not None, "provider_non_200")
                text = output_text(body)
                parsed, validation_errors = prompt_contract.validate_output(text)
                source = sources[mapping["source_id"]]
                source_text = prompt_contract.compact_text(source["prompt"] + "\n" + source["essay"])
                mismatches = [span for axis in AXES for span in prompt_contract.quoted_spans(parsed[axis]["rationale"]) if prompt_contract.compact_text(span) not in source_text] if parsed else []
                if mismatches:
                    validation_errors.append("nonverbatim_quoted_span")
                if parsed and any(FOREIGN_SCRIPT_RE.search(parsed[axis]["rationale"]) for axis in AXES):
                    validation_errors.append("foreign_script")
            except Exception as exc:
                parsed, validation_errors, body = None, [type(exc).__name__], body or {}
            record = {
                **mapping, "model": manifest["model"], "prompt_sha256": PROMPT_SHA256,
                "rationale": parsed, "api_response_id": body.get("id") if isinstance(body, Mapping) else None,
            }
            if validation_errors:
                errors.append({**record, "validation_errors": sorted(set(validation_errors))})
            else:
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                accepted.add(str(custom_id))
    missing = sorted(set(mappings) - accepted - {str(row.get("custom_id")) for row in errors})
    errors.extend({"custom_id": custom_id, "validation_errors": ["missing_batch_record"]} for custom_id in missing)
    with errors_path.open("x", encoding="utf-8") as output:
        for row in errors:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(accepted_path, 0o600); os.chmod(errors_path, 0o600)
    manifest.update({
        "status": "validated", "downloaded_at": now(), "accepted": len(accepted), "rejected_or_missing": len(errors),
        "output_sha256": api.sha256_file(raw_path), "candidates_sha256": api.sha256_file(accepted_path), "errors_sha256": api.sha256_file(errors_path),
    })
    manifest["events"].append({"at": now(), "event": "validated"})
    write_manifest(args.run_id, manifest)
    emit(args.run_id, status="validated", accepted=len(accepted), rejected_or_missing=len(errors))


def retry_direct(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.run_id)
    need(manifest.get("status") == "validated" and not manifest.get("direct_retry"), "one direct retry requires an unretried validated run")
    destination = run_dir(args.run_id)
    errors = [json.loads(line) for line in (destination / "errors.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    allowed_repair_errors = {"nonverbatim_quoted_span", "foreign_script"}
    need(
        errors and all(
            all(error in allowed_repair_errors or error.endswith(":score_leak") for error in row.get("validation_errors", []))
            for row in errors
        ),
        "direct retry is restricted to surface-form violations",
    )
    requests = jsonl_by_id(destination / "requests.jsonl")
    mappings = jsonl_by_id(destination / "source_map.jsonl")
    split, rows = selected_rows(manifest["scope"])
    sources = {row["source_id"]: row for row in rows}
    key = api.load_env_key()

    def call(row: Mapping[str, Any]) -> dict[str, Any]:
        custom_id = str(row["custom_id"])
        request = json.loads(json.dumps(requests[custom_id], ensure_ascii=False))
        user_content = request["body"]["input"][-1]["content"][-1]
        user_content["text"] += "\n\n[형식 재시도]\n이번 응답에서는 직접 인용과 따옴표를 사용하지 말고, 원문 요소를 정확한 한국어로 요약하라. 출력에 점수·등급·단계·정답 라벨을 지칭하는 표현을 쓰지 말고 관찰과 판단만 서술하라. 평가 기준과 판단 강도는 바꾸지 마라."
        idem = api.stable_key("mal2026-v3-tail-direct-retry", args.run_id, custom_id, manifest["request_sha256"])
        response = api.request_json("POST", "/responses", key, request["body"], headers={"Idempotency-Key": idem})
        text = api.get_text(response)
        parsed, validation_errors = prompt_contract.validate_output(text)
        mapping = mappings[custom_id]
        source = sources[mapping["source_id"]]
        source_text = prompt_contract.compact_text(source["prompt"] + "\n" + source["essay"])
        mismatches = [span for axis in AXES for span in prompt_contract.quoted_spans(parsed[axis]["rationale"]) if prompt_contract.compact_text(span) not in source_text] if parsed else []
        if mismatches:
            validation_errors.append("nonverbatim_quoted_span")
        if parsed and any(FOREIGN_SCRIPT_RE.search(parsed[axis]["rationale"]) for axis in AXES):
            validation_errors.append("foreign_script")
        return {
            "custom_id": custom_id, "response": response,
            "candidate": {**mapping, "model": manifest["model"], "prompt_sha256": PROMPT_SHA256, "rationale": parsed, "api_response_id": response.get("id")},
            "validation_errors": sorted(set(validation_errors)),
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(call, row): str(row["custom_id"]) for row in errors}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["custom_id"])
    retry_path = destination / "direct_retry_responses.jsonl"
    with retry_path.open("x", encoding="utf-8") as output:
        for row in results:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(retry_path, 0o600)
    recovered = [row["candidate"] for row in results if not row["validation_errors"]]
    remaining = [{**row["candidate"], "validation_errors": row["validation_errors"], "retry_attempt": 1} for row in results if row["validation_errors"]]
    with (destination / "candidates.jsonl").open("a", encoding="utf-8") as output:
        for row in recovered:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    errors_path = destination / "errors.jsonl"
    temporary = errors_path.with_name(f".{errors_path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as output:
        for row in remaining:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(temporary, 0o600); temporary.replace(errors_path)
    manifest["direct_retry"] = {
        "at": now(), "requests": len(results), "recovered": len(recovered), "remaining": len(remaining),
        "repair": "append no-quotation, Korean-only, and no-score-reference style instruction; scientific score/rubric variables unchanged",
        "response_sha256": api.sha256_file(retry_path),
    }
    manifest["accepted"] = api.line_count(destination / "candidates.jsonl")
    manifest["rejected_or_missing"] = len(remaining)
    manifest["candidates_sha256"] = api.sha256_file(destination / "candidates.jsonl")
    manifest["errors_sha256"] = api.sha256_file(errors_path)
    manifest["events"].append({"at": now(), "event": "direct_retry_completed", "recovered": len(recovered), "remaining": len(remaining)})
    write_manifest(args.run_id, manifest)
    emit(args.run_id, status="direct_retry_completed", requests=len(results), recovered=len(recovered), remaining=len(remaining))


def ngrams(text: str, size: int = 3) -> set[str]:
    compact = prompt_contract.compact_text(text)
    return {compact[index:index + size] for index in range(max(0, len(compact) - size + 1))}


def similarity(left: str, right: str) -> float:
    a, b = ngrams(left), ngrams(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def analyze(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.run_id)
    need(manifest.get("status") == "validated", "validated candidates required")
    candidates = [json.loads(line) for line in (run_dir(args.run_id) / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = [json.loads(line) for line in (run_dir(args.run_id) / "errors.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row["source_id"]].append(row)
    pairwise: list[float] = []
    exact_duplicate_pairs = 0
    for rows in grouped.values():
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                left_text = "\n".join(left["rationale"][axis]["rationale"] for axis in AXES)
                right_text = "\n".join(right["rationale"][axis]["rationale"] for axis in AXES)
                pairwise.append(similarity(left_text, right_text))
                exact_duplicate_pairs += int(prompt_contract.compact_text(left_text) == prompt_contract.compact_text(right_text))
    error_categories = Counter(category for row in errors for category in row.get("validation_errors", []))
    foreign_script_candidates = sum(
        any(FOREIGN_SCRIPT_RE.search(row["rationale"][axis]["rationale"]) for axis in AXES)
        for row in candidates
    )
    report = {
        "schema_version": "mal2026-rationale-v3-tail-batch-aggregate-v1", "status": "completed",
        "run_id": args.run_id, "model": manifest["model"], "scope": manifest["scope"],
        "source_rows": manifest["source_rows"], "requests": manifest["requests"],
        "accepted": len(candidates), "rejected_or_missing": len(errors), "acceptance_rate": len(candidates) / manifest["requests"],
        "error_categories": dict(sorted(error_categories.items())),
        "posthoc_foreign_script_candidate_count": foreign_script_candidates,
        "posthoc_foreign_script_candidates_eligible_for_sft": 0,
        "rows_with_at_least_one_candidate": len(grouped),
        "exact_duplicate_pairs": exact_duplicate_pairs,
        "pairwise_trigram_similarity_mean": sum(pairwise) / len(pairwise) if pairwise else None,
        "pairwise_trigram_similarity_max": max(pairwise) if pairwise else None,
        "prompt_sha256": PROMPT_SHA256, "variant_directives_sha256": manifest["variant_directives_sha256"],
        "score_histogram": manifest["score_histogram"], "row_multiplicity": manifest["row_multiplicity"],
        "privacy": "aggregate_only_no_source_ids_prompts_essays_rationales_or_response_ids",
    }
    write_json(aggregate_dir(args.run_id) / "aggregate.json", report)
    manifest["aggregate_sha256"] = api.sha256_file(aggregate_dir(args.run_id) / "aggregate.json")
    manifest["events"].append({"at": now(), "event": "analyzed"})
    write_manifest(args.run_id, manifest)
    emit(args.run_id, status="analyzed", accepted=len(candidates), rejected_or_missing=len(errors), exact_duplicate_pairs=exact_duplicate_pairs, similarity_mean=report["pairwise_trigram_similarity_mean"])


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-id", required=True)
    common.add_argument("--model", required=True, choices=MODELS)
    prepare_parser = sub.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--scope", required=True, choices=("pilot", "train", "validation"))
    prepare_parser.set_defaults(func=prepare)
    for command, function in (("smoke", smoke), ("submit", submit), ("poll", poll), ("download", download), ("retry-direct", retry_direct), ("analyze", analyze)):
        item = sub.add_parser(command, parents=[common]); item.set_defaults(func=function)
    return parser


def main() -> int:
    args = parser().parse_args()
    setproctitle(f"mal2026:rationale-v3-tail-batch:{args.command}:{args.run_id}"[:255])
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
