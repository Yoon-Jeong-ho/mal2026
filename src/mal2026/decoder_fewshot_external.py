"""External/API extensions of the fixed decoder few-shot validation protocol.

Restricted prompts, validation writings, mappings, and responses stay below
``data/processed/restricted``.  Public artifacts contain aggregate metrics and
provenance only.  The demonstrations and their order exactly replay the
already-prepared train-only decoder protocol.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
import mimetypes
import os
from pathlib import Path
import secrets
import statistics
import subprocess
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

from .api_rationale_data import AXES, EXPECTED_ESSAYS, SOURCE_SHA256, load_writing_rows, sha256_file
from .decoder_fewshot_validation import (
    CONDITIONS,
    RATIONALE_SHA256,
    condition_metrics,
    file_sha256,
    messages_for,
    parse_response,
    response_schema,
    rotate,
    rotation_assignments,
    round_half_up,
    select_shots,
)
from .official_score_prompt import EVALUATION_PROMPT_SHA256


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "mal2026-decoder-fewshot-external-config-v1"
BASE_RUN_ID = "decoder-fewshot-validation-v1-20260731-001"
BASE_RESTRICTED = ROOT / "data/processed/restricted/decoder_fewshot_validation_v1" / BASE_RUN_ID
BASE_PUBLIC = ROOT / "outputs/analysis" / BASE_RUN_ID
RESTRICTED_ROOT = ROOT / "data/processed/restricted/decoder_fewshot_external_v1"
PUBLIC_ROOT = ROOT / "outputs/analysis"
RUNTIME_ROOT = ROOT / "outputs/decoder-fewshot-external-v1"
API_ROOT = "https://api.openai.com/v1"
VALIDATION_ROWS = EXPECTED_ESSAYS["validation"]
EXTERNAL_CONFIG_SHA256 = "e24d56472f660ecab5deadfb206d2d2d877556339cc004998f2296393f840d56"


class ExternalFewshotError(RuntimeError):
    """Fail-closed protocol or integration error."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalFewshotError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_json_fresh(path: Path, value: Mapping[str, Any]) -> str:
    need(not path.exists(), f"fresh output required: {path}")
    _atomic_json(path, value)
    return file_sha256(path)


def _write_jsonl_fresh(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), f"fresh output required: {path}")
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return file_sha256(path)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    need(path.is_file() and not path.is_symlink(), f"JSONL unavailable: {path}")
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                need(isinstance(value, dict), "JSONL row is not an object")
                values.append(value)
    return values


@dataclass(frozen=True)
class SolarSpec:
    model_id: str
    model_path: str
    model_alias: str
    docker_image: str
    docker_image_id: str
    gpu_scope: tuple[int, ...]
    port: int
    max_model_len: int
    max_tokens: int
    retry_max_tokens: int
    max_inflight: int


@dataclass(frozen=True)
class ExternalConfig:
    schema_version: str
    run_id: str
    seed: int
    base_run_id: str
    base_config_sha256: str
    shot_manifest_sha256: str
    score_prompt_sha256: str
    conditions: tuple[str, ...]
    api_models: tuple[str, ...]
    api_max_output_tokens: int
    solar: SolarSpec

    @classmethod
    def from_json(cls, path: Path) -> "ExternalConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        need(isinstance(raw, dict) and set(raw) == {
            "schema_version", "run_id", "seed", "base_run_id", "base_config_sha256",
            "shot_manifest_sha256", "score_prompt_sha256", "conditions", "api_models",
            "api_max_output_tokens", "solar",
        }, "external config schema differs")
        solar_raw = raw.pop("solar")
        conditions = tuple(raw.pop("conditions"))
        api_models = tuple(raw.pop("api_models"))
        solar_raw["gpu_scope"] = tuple(solar_raw["gpu_scope"])
        config = cls(conditions=conditions, api_models=api_models, solar=SolarSpec(**solar_raw), **raw)
        config.validate(path)
        return config

    def validate(self, path: Path) -> None:
        need(self.schema_version == SCHEMA_VERSION and self.run_id == "decoder-fewshot-external-v1-20260731-001", "external run identity differs")
        need(self.seed == 2026073104 and self.base_run_id == BASE_RUN_ID, "base protocol identity differs")
        need(self.conditions == CONDITIONS and self.api_models == ("gpt-5.6-terra", "gpt-5.6-luna"), "external model/condition matrix differs")
        need(self.score_prompt_sha256 == EVALUATION_PROMPT_SHA256, "evaluation prompt binding differs")
        need(self.api_max_output_tokens == 1800, "API output budget differs")
        need(self.solar.gpu_scope == (0, 1, 2, 3), "Solar GPU scope differs")
        need((self.solar.max_model_len, self.solar.max_tokens, self.solar.retry_max_tokens) == (12288, 512, 2048), "Solar generation budget differs")
        need(self.solar.max_inflight == 64 and 1024 <= self.solar.port <= 65535, "Solar serving capacity differs")
        need(Path(self.solar.model_path).is_absolute(), "Solar model path differs")
        need(file_sha256(path) == EXTERNAL_CONFIG_SHA256, "external config checksum differs")


def restricted_dir(config: ExternalConfig) -> Path:
    path = RESTRICTED_ROOT / config.run_id
    need(path.resolve().is_relative_to(RESTRICTED_ROOT.resolve()), "restricted run escaped root")
    return path


def public_dir(config: ExternalConfig) -> Path:
    path = PUBLIC_ROOT / config.run_id
    need(path.resolve().is_relative_to(PUBLIC_ROOT.resolve()), "public run escaped root")
    return path


def runtime_dir(config: ExternalConfig) -> Path:
    path = RUNTIME_ROOT / config.run_id
    need(path.resolve().is_relative_to(RUNTIME_ROOT.resolve()), "runtime run escaped root")
    return path


def _verify_base(config: ExternalConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = ROOT / "configs/decoder_fewshot_validation.v1.json"
    manifest_path = BASE_RESTRICTED / "shot_manifest.json"
    protocol_path = BASE_PUBLIC / "protocol.json"
    need(file_sha256(config_path) == config.base_config_sha256, "base config checksum differs")
    need(file_sha256(manifest_path) == config.shot_manifest_sha256, "shot manifest checksum differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    need(protocol.get("shot_manifest_sha256") == config.shot_manifest_sha256, "base public protocol differs")
    shots = select_shots()
    expected = {condition: [row.source_id for row in shots[condition]] for condition in CONDITIONS}
    actual = {condition: [row["source_id"] for row in manifest["conditions"][condition]] for condition in CONDITIONS}
    need(actual == expected, "train-only shot replay differs")
    return shots, protocol


def request_records(config: ExternalConfig) -> list[dict[str, Any]]:
    need(sha256_file(ROOT / "eval/validation.jsonl") == SOURCE_SHA256["validation"], "validation checksum differs")
    shots, _ = _verify_base(config)
    validation = load_writing_rows("validation", include_scores=True)
    assignments = rotation_assignments([row.identifier for row in validation], config.seed)
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for row in validation:
            need(row.scores is not None, "validation scores unavailable for metrics")
            rotation = assignments[row.identifier]
            records.append({
                "source_id": row.identifier,
                "condition": condition,
                "rotation": rotation,
                "messages": messages_for(rotate(shots[condition], rotation), row.prompt, row.essay),
                "gold_raw": {axis: float(row.scores[axis]) for axis in AXES},
                "gold_integer": {axis: round_half_up(row.scores[axis]) for axis in AXES},
            })
    records.sort(key=lambda row: (row["condition"], row["rotation"], sha256(row["source_id"].encode()).hexdigest()))
    need(len(records) == 2 * VALIDATION_ROWS and len({(row["source_id"], row["condition"]) for row in records}) == len(records), "external request population differs")
    return records


def _responses_input(messages: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "role": row["role"],
            "content": [{
                "type": "output_text" if row["role"] == "assistant" else "input_text",
                "text": row["content"],
            }],
        }
        for row in messages
    ]


def openai_body(model: str, messages: Sequence[Mapping[str, str]], max_output_tokens: int) -> dict[str, Any]:
    need(model in {"gpt-5.6-terra", "gpt-5.6-luna"}, "API model differs")
    return {
        "model": model,
        "input": _responses_input(messages),
        "text": {"format": {"type": "json_schema", "name": "mal2026_decoder_fewshot_score", "strict": True, "schema": response_schema()}},
        "reasoning": {"effort": "none"},
        "max_output_tokens": max_output_tokens,
        "store": False,
    }


def response_text(response: Mapping[str, Any]) -> str:
    for item in response.get("output", []):
        if isinstance(item, Mapping):
            for content in item.get("content", []):
                if isinstance(content, Mapping) and content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return str(content["text"])
    raise ExternalFewshotError("Responses result has no output_text")


def _api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return str(os.environ["OPENAI_API_KEY"])
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            value = line.split("=", 1)[1].strip().strip("\"'")
            if value:
                return value
    raise ExternalFewshotError("OPENAI_API_KEY unavailable")


def _api_json(method: str, path: str, payload: Mapping[str, Any] | None = None, headers: Mapping[str, str] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(API_ROOT + path, data=body, method=method)
    request.add_header("Authorization", f"Bearer {_api_key()}")
    request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ExternalFewshotError(f"OpenAI {method} {path} failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise ExternalFewshotError(f"OpenAI {method} {path} network failure") from exc
    need(isinstance(result, dict), "OpenAI response envelope differs")
    return result


def _upload(path: Path, idempotency_key: str) -> dict[str, Any]:
    boundary = f"----mal2026-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/jsonl"
    body = b"".join([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode(),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
         f"Content-Type: {content_type}\r\n\r\n").encode(), path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = Request(API_ROOT + "/files", data=body, method="POST")
    request.add_header("Authorization", f"Bearer {_api_key()}")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    request.add_header("Idempotency-Key", idempotency_key)
    try:
        with urlopen(request, timeout=900) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ExternalFewshotError(f"OpenAI upload failed with HTTP {exc.code}") from exc
    need(isinstance(result, dict) and isinstance(result.get("id"), str), "OpenAI upload envelope differs")
    return result


def _download(path: str) -> bytes:
    request = Request(API_ROOT + path, method="GET", headers={"Authorization": f"Bearer {_api_key()}"})
    try:
        with urlopen(request, timeout=900) as response:
            return response.read()
    except HTTPError as exc:
        raise ExternalFewshotError(f"OpenAI download failed with HTTP {exc.code}") from exc


def prepare(config: ExternalConfig, config_path: Path) -> dict[str, Any]:
    need(not restricted_dir(config).exists() and not public_dir(config).exists(), "external run already prepared")
    records = request_records(config)
    salt = secrets.token_bytes(32)
    model_artifacts: dict[str, Any] = {}
    for model in config.api_models:
        key = model.replace("gpt-5.6-", "")
        destination = restricted_dir(config) / "api" / key
        destination.mkdir(parents=True, mode=0o700)
        request_path, map_path = destination / "requests.jsonl", destination / "source_map.jsonl"
        requests: list[dict[str, Any]] = []
        mappings: list[dict[str, Any]] = []
        for record in records:
            opaque = hmac.new(salt, f"{model}\0{record['source_id']}\0{record['condition']}".encode(), sha256).hexdigest()[:28]
            custom_id = f"{record['condition'][0]}-{opaque}"
            requests.append({"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": openai_body(model, record["messages"], config.api_max_output_tokens)})
            mappings.append({"custom_id": custom_id, "source_id": record["source_id"], "condition": record["condition"], "rotation": record["rotation"]})
        request_sha = _write_jsonl_fresh(request_path, requests)
        map_sha = _write_jsonl_fresh(map_path, mappings)
        model_artifacts[model] = {"requests": len(requests), "request_sha256": request_sha, "source_map_sha256": map_sha, "request_bytes": request_path.stat().st_size}
        _atomic_json(destination / "manifest.json", {
            "schema_version": "mal2026-decoder-fewshot-api-batch-v1", "status": "prepared",
            "run_id": config.run_id, "model": model, "created_at": now(), **model_artifacts[model],
            "external_transfer_authorization": "user explicitly requested gpt API terra/luna validation testing",
            "validation_labels_in_request": False,
        })
    payload = {
        "schema_version": "mal2026-decoder-fewshot-external-protocol-v1", "status": "prepared",
        "run_id": config.run_id, "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": file_sha256(config_path), "base_run_id": config.base_run_id,
        "base_config_sha256": config.base_config_sha256, "shot_manifest_sha256": config.shot_manifest_sha256,
        "score_prompt_sha256": config.score_prompt_sha256, "rationale_sha256": RATIONALE_SHA256,
        "canonical_source_sha256": SOURCE_SHA256, "conditions": list(CONDITIONS),
        "validation_rows": VALIDATION_ROWS, "requests_per_model": len(records), "api": model_artifacts,
        "solar_model_id": config.solar.model_id, "average_target_used": False,
        "validation_scores_used_for_prompting_or_selection": False,
    }
    _write_json_fresh(public_dir(config) / "protocol.json", payload)
    runtime_dir(config).mkdir(parents=True, exist_ok=True)
    _write_jsonl_fresh(runtime_dir(config) / "ledger.jsonl", [{"event": "prepared", "at": now(), "gpu_scope": list(config.solar.gpu_scope), "external_api_models": list(config.api_models), "config_sha256": file_sha256(config_path)}])
    return payload


def repair_api_assistant_content(config: ExternalConfig) -> dict[str, Any]:
    """Preserve and repair the pre-upload Responses assistant-content type.

    Responses accepts ``output_text`` rather than ``input_text`` for assistant
    demonstrations.  The failed request artifacts remain immutable under an
    attempt-1 name; no batch file had been uploaded before this repair.
    """
    records = request_records(config)
    repaired: dict[str, Any] = {}
    for model in config.api_models:
        destination = restricted_dir(config) / "api" / model.replace("gpt-5.6-", "")
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        request_path = destination / "requests.jsonl"
        failed_path = destination / "requests.attempt1-invalid-assistant-input-text.jsonl"
        need(manifest.get("status") == "prepared" and not manifest.get("batch_id"), "API repair is pre-upload only")
        need(file_sha256(request_path) == manifest["request_sha256"] and not failed_path.exists(), "API attempt-1 binding differs")
        mappings = _jsonl(destination / "source_map.jsonl")
        need(len(mappings) == len(records), "API repair mapping population differs")
        requests: list[dict[str, Any]] = []
        for mapping, record in zip(mappings, records):
            need((mapping["source_id"], mapping["condition"], mapping["rotation"]) == (record["source_id"], record["condition"], record["rotation"]), "API repair order differs")
            requests.append({"custom_id": mapping["custom_id"], "method": "POST", "url": "/v1/responses", "body": openai_body(model, record["messages"], config.api_max_output_tokens)})
        request_path.rename(failed_path)
        repaired_sha = _write_jsonl_fresh(request_path, requests)
        manifest.update({
            "initial_request_sha256": manifest["request_sha256"],
            "initial_request_file": failed_path.name,
            "request_sha256": repaired_sha,
            "request_bytes": request_path.stat().st_size,
            "integration_repair": "assistant demonstration content type input_text -> output_text",
            "scientific_variables_changed": False,
        })
        _atomic_json(manifest_path, manifest)
        repaired[model] = {"initial_request_sha256": manifest["initial_request_sha256"], "request_sha256": repaired_sha, "requests": len(requests)}
    protocol_path = public_dir(config) / "protocol.json"
    attempt_path = public_dir(config) / "protocol.attempt1.json"
    need(protocol_path.is_file() and not attempt_path.exists(), "public API repair binding differs")
    protocol_path.rename(attempt_path)
    protocol = json.loads(attempt_path.read_text(encoding="utf-8"))
    protocol["api"] = {
        model: {
            "requests": repaired[model]["requests"],
            "request_sha256": repaired[model]["request_sha256"],
            "source_map_sha256": file_sha256(restricted_dir(config) / "api" / model.replace("gpt-5.6-", "") / "source_map.jsonl"),
            "request_bytes": (restricted_dir(config) / "api" / model.replace("gpt-5.6-", "") / "requests.jsonl").stat().st_size,
        }
        for model in config.api_models
    }
    protocol["integration_repair"] = {
        "reason": "Responses API rejected input_text for assistant demonstrations before any batch upload",
        "repair": "use output_text only for assistant-role demonstration messages",
        "scientific_variables_changed": False,
    }
    _write_json_fresh(protocol_path, protocol)
    return {"status": "repaired", "models": repaired}


def api_smoke(config: ExternalConfig, model: str) -> dict[str, Any]:
    need(model in config.api_models, "unknown API model")
    records = request_records(config)
    selected = [next(row for row in records if row["condition"] == condition) for condition in CONDITIONS]
    responses: list[dict[str, Any]] = []
    usage = Counter()
    for record in selected:
        response = _api_json("POST", "/responses", openai_body(model, record["messages"], config.api_max_output_tokens))
        parsed = parse_response(response_text(response))
        need(set(parsed) == set(AXES) and response.get("status") == "completed", "API smoke did not complete")
        item_usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
        usage.update({str(key): int(value) for key, value in item_usage.items() if type(value) is int})
        responses.append({"condition": record["condition"], "rotation": record["rotation"], "response": response})
    destination = restricted_dir(config) / "api" / model.replace("gpt-5.6-", "")
    _write_jsonl_fresh(destination / "smoke_responses.jsonl", responses)
    result = {"schema_version": "mal2026-decoder-fewshot-api-smoke-v1", "status": "passed", "run_id": config.run_id, "model": model, "requests": 2, "conditions": list(CONDITIONS), "usage": dict(usage)}
    _write_json_fresh(public_dir(config) / "models" / model / "smoke.json", result)
    return result


def api_submit(config: ExternalConfig, model: str) -> dict[str, Any]:
    key = model.replace("gpt-5.6-", "")
    destination = restricted_dir(config) / "api" / key
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    need(manifest.get("model") == model and manifest.get("status") == "prepared", "API batch is not prepared")
    need((public_dir(config) / "models" / model / "smoke.json").is_file(), "API smoke is unavailable")
    request_path = destination / "requests.jsonl"
    need(file_sha256(request_path) == manifest["request_sha256"], "API request checksum differs")
    upload_key = sha256(f"{config.run_id}\0{model}\0upload\0{manifest['request_sha256']}".encode()).hexdigest()
    batch_key = sha256(f"{config.run_id}\0{model}\0batch\0{manifest['request_sha256']}".encode()).hexdigest()
    uploaded = _upload(request_path, upload_key)
    batch = _api_json("POST", "/batches", {
        "input_file_id": uploaded["id"], "endpoint": "/v1/responses", "completion_window": "24h",
        "metadata": {"run_id": config.run_id, "model": model, "artifact": "mal2026_decoder_fewshot_external_v1"},
    }, headers={"Idempotency-Key": batch_key})
    manifest.update({"status": "submitted", "input_file_id": uploaded["id"], "batch_id": batch["id"], "submitted_at": now(), "request_counts": batch.get("request_counts")})
    _atomic_json(manifest_path, manifest)
    return {"status": "submitted", "model": model, "batch_id": batch["id"], "request_counts": batch.get("request_counts")}


def api_poll(config: ExternalConfig, model: str) -> dict[str, Any]:
    destination = restricted_dir(config) / "api" / model.replace("gpt-5.6-", "")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    need(isinstance(manifest.get("batch_id"), str), "API batch ID unavailable")
    batch = _api_json("GET", f"/batches/{manifest['batch_id']}")
    manifest.update({"status": batch["status"], "last_polled_at": now(), "request_counts": batch.get("request_counts"), "output_file_id": batch.get("output_file_id"), "error_file_id": batch.get("error_file_id")})
    if batch.get("status") == "completed" and isinstance(batch.get("output_file_id"), str):
        output_path = destination / "batch_output.jsonl"
        if not output_path.exists():
            output_path.write_bytes(_download(f"/files/{batch['output_file_id']}/content"))
            os.chmod(output_path, 0o600)
        manifest["output_sha256"] = file_sha256(output_path)
        if isinstance(batch.get("error_file_id"), str):
            error_path = destination / "batch_errors.jsonl"
            if not error_path.exists():
                error_path.write_bytes(_download(f"/files/{batch['error_file_id']}/content"))
                os.chmod(error_path, 0o600)
            manifest["error_sha256"] = file_sha256(error_path)
    _atomic_json(manifest_path, manifest)
    return {"status": batch["status"], "model": model, "request_counts": batch.get("request_counts"), "output_downloaded": (destination / "batch_output.jsonl").exists()}


def _aggregate_rows(config: ExternalConfig, model_key: str, model_id: str, rows: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]) -> dict[str, Any]:
    need(len(rows) == 2 * VALIDATION_ROWS, "prediction population differs")
    metrics: dict[str, Any] = {}
    for condition in CONDITIONS:
        valid = [row for row in rows if row["condition"] == condition and row.get("parse_valid") is True]
        metrics[condition] = condition_metrics(valid, expected_count=len(valid), total_count=VALIDATION_ROWS)
    aggregate = {
        "schema_version": "mal2026-decoder-fewshot-external-model-result-v1", "status": "completed",
        "run_id": config.run_id, "model_key": model_key, "model_id": model_id,
        "validation_rows": VALIDATION_ROWS, "requests": len(rows),
        "parse_failures": sum(row.get("parse_valid") is not True for row in rows),
        "metrics": metrics, **dict(provenance),
    }
    _write_json_fresh(public_dir(config) / "models" / model_key / "aggregate.json", aggregate)
    return aggregate


def api_finalize(config: ExternalConfig, model: str) -> dict[str, Any]:
    key = model.replace("gpt-5.6-", "")
    destination = restricted_dir(config) / "api" / key
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    need(manifest.get("status") == "completed" and (destination / "batch_output.jsonl").is_file(), "API batch is not complete")
    mappings = _jsonl(destination / "source_map.jsonl")
    mapping = {row["custom_id"]: row for row in mappings}
    outputs = _jsonl(destination / "batch_output.jsonl")
    need(len(mapping) == len(outputs) == 2 * VALIDATION_ROWS, "API output population differs")
    validation = {row.identifier: row for row in load_writing_rows("validation", include_scores=True)}
    rows: list[dict[str, Any]] = []
    usage = Counter()
    for output in outputs:
        meta = mapping.get(output.get("custom_id"))
        need(meta is not None and meta["source_id"] in validation, "API mapping differs")
        envelope = output.get("response") if isinstance(output.get("response"), Mapping) else {}
        body = envelope.get("body") if isinstance(envelope.get("body"), Mapping) else {}
        response = body if envelope.get("status_code") == 200 else None
        parsed, text, error = None, None, None
        if response is not None:
            try:
                text = response_text(response)
                parsed = parse_response(text)
            except Exception as exc:
                error = type(exc).__name__ + ":" + str(exc)
            item_usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
            usage.update({str(name): int(value) for name, value in item_usage.items() if type(value) is int})
        else:
            error = "api_non_200_or_missing_body"
        source = validation[meta["source_id"]]
        need(source.scores is not None, "validation scores unavailable for API metrics")
        rows.append({
            "source_id": meta["source_id"], "condition": meta["condition"], "rotation": meta["rotation"],
            "response": text, "response_id": response.get("id") if response else None,
            "response_status": response.get("status") if response else None,
            "parse_valid": parsed is not None, "parse_error": error,
            "prediction": {axis: parsed[axis]["score"] for axis in AXES} if parsed else None,
            "gold_raw": {axis: float(source.scores[axis]) for axis in AXES},
            "gold_integer": {axis: round_half_up(source.scores[axis]) for axis in AXES},
        })
    prediction_path = destination / "predictions.jsonl"
    prediction_sha = _write_jsonl_fresh(prediction_path, rows)
    return _aggregate_rows(config, model, model, rows, {
        "provider": "OpenAI Responses Batch API", "config_sha256": EXTERNAL_CONFIG_SHA256,
        "request_sha256": manifest.get("request_sha256"),
        "prediction_sha256": prediction_sha, "batch_id": manifest["batch_id"], "usage": dict(usage),
        "temperature": "provider_default_unsupported_by_fixed_api_model", "reasoning_effort": "none",
        "external_data_transfer_authorized_by_user": True,
    })


def _response_format() -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "mal2026_decoder_fewshot_score", "strict": True, "schema": response_schema()}}


def _solar_payload(config: ExternalConfig, messages: Sequence[Mapping[str, str]], max_tokens: int) -> dict[str, Any]:
    response_format = _response_format()
    return {
        "model": config.solar.model_alias, "messages": list(messages), "temperature": 0.0, "top_p": 1.0,
        "max_tokens": max_tokens, "seed": config.seed, "response_format": response_format,
        "chat_template_kwargs": {"reasoning_effort": "none", "think_render_option": "preserved", "response_format": response_format},
    }


def _http_json(url: str, payload: Mapping[str, Any] | None = None, timeout: int = 900) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="GET" if data is None else "POST")
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    need(isinstance(value, dict), "Solar HTTP response differs")
    return value


def _assert_gpus_idle(scope: Sequence[int]) -> None:
    raw = subprocess.check_output(["nvidia-smi", "-i", ",".join(map(str, scope)), "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    rows = [tuple(int(part.strip()) for part in line.split(",")) for line in raw.splitlines() if line.strip()]
    need([row[0] for row in rows] == list(scope), "Solar GPU inventory differs")
    need(all(memory == 0 and utilization == 0 for _, memory, utilization in rows), "Solar GPU scope is not idle")


def _solar_preflight(config: ExternalConfig, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model_path = Path(config.solar.model_path)
    need(model_path.is_dir() and not model_path.is_symlink(), "Solar model unavailable")
    bindings = {
        "config_sha256": file_sha256(model_path / "config.json"),
        "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        "model_code_sha256": file_sha256(model_path / "modeling_solar_open2.py"),
        "chat_template_sha256": file_sha256(model_path / "chat_template.jinja"),
    }
    need(bindings == {
        "config_sha256": "039c9fe98844aa026aba4260692c1869a3bd2eae385d06f865714b816928a7b5",
        "weight_index_sha256": "255b0cb9e82b5f564290bdd1c52734e2f9809d74ee80b056fdf3e3c601df1ae7",
        "model_code_sha256": "b6ea8bfbbf66588ec47e6b7fa683a7ca75c328546c331a7a51015f7bb0563ed1",
        "chat_template_sha256": "111eec19d6dd69146a4f29a084ea50a356aa907e83029e0d8d1c9dec883679c0",
    }, "Solar model binding differs")
    image_raw = json.loads(subprocess.check_output(["docker", "image", "inspect", config.solar.docker_image], text=True))
    need(len(image_raw) == 1 and image_raw[0].get("Id") == config.solar.docker_image_id, "Solar image binding differs")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    lengths: list[int] = []
    response_format = _response_format()
    for record in records:
        encoded = tokenizer.apply_chat_template(record["messages"], tokenize=True, add_generation_prompt=True, reasoning_effort="none", think_render_option="preserved", response_format=response_format)
        tokens = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        lengths.append(len(tokens))
    need(max(lengths) + config.solar.retry_max_tokens <= config.solar.max_model_len, "Solar context preflight failed")
    return {**bindings, "requests_audited": len(lengths), "prompt_tokens_min": min(lengths), "prompt_tokens_max": max(lengths)}


def _solar_server_command(config: ExternalConfig, container: str) -> list[str]:
    cache = RUNTIME_ROOT / "_solar_vllm_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return [
        "docker", "run", "--rm", "--name", container, "--gpus", '"device=0,1,2,3"', "--ipc=host", "--network=host",
        "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1", "--env", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0",
        "--mount", f"type=bind,src={cache},dst=/root/.cache/vllm",
        "--mount", f"type=bind,src={config.solar.model_path},dst=/models/Solar-Open2-250B-Nota-INT4,readonly",
        config.solar.docker_image, "/models/Solar-Open2-250B-Nota-INT4", "--served-model-name", config.solar.model_alias,
        "--host", "127.0.0.1", "--port", str(config.solar.port), "--tensor-parallel-size", "4", "--trust-remote-code",
        "--enable-expert-parallel", "--moe-backend", "triton", "--default-chat-template-kwargs", '{"think_render_option":"preserved"}',
        "--reasoning-parser", "solar_open2", "--logits-processors", "vllm.v1.sample.logits_processor.solar_open2:SolarOpen2TemplateLogitsProcessor",
        "--max-model-len", str(config.solar.max_model_len), "--gpu-memory-utilization", "0.90", "--max-num-seqs", "64",
        "--max-num-batched-tokens", "32768", "--enable-prefix-caching", "--enable-chunked-prefill",
    ]


def _wait_solar(process: subprocess.Popen[str], endpoint: str) -> None:
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        need(process.poll() is None, f"Solar server exited during startup: {process.returncode}")
        try:
            models = _http_json(endpoint + "/v1/models", timeout=5)
            if any(row.get("id") == "solar-open2-250b-nota-int4" for row in models.get("data", []) if isinstance(row, dict)):
                return
        except Exception:
            pass
        time.sleep(2)
    raise ExternalFewshotError("Solar server readiness timed out")


def _solar_request(config: ExternalConfig, endpoint: str, record: Mapping[str, Any]) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for max_tokens in (config.solar.max_tokens, config.solar.retry_max_tokens):
        last: BaseException | None = None
        response: dict[str, Any] | None = None
        for _ in range(3):
            try:
                response = _http_json(endpoint + "/v1/chat/completions", _solar_payload(config, record["messages"], max_tokens), timeout=900)
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                last = exc
        if response is None:
            assert last is not None
            raise last
        choices = response.get("choices")
        need(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict), "Solar choices differ")
        choice = choices[0]
        message = choice.get("message")
        text = message.get("content") if isinstance(message, Mapping) else None
        need(isinstance(text, str), "Solar content differs")
        parsed, error = None, None
        try:
            parsed = parse_response(text)
        except Exception as exc:
            error = type(exc).__name__ + ":" + str(exc)
        usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
        attempt = {"max_tokens": max_tokens, "finish_reason": choice.get("finish_reason"), "response": text, "parse_valid": parsed is not None, "parse_error": error, "usage": dict(usage)}
        attempts.append(attempt)
        if parsed is not None and choice.get("finish_reason") != "length":
            return {**record, "response": text, "parse_valid": True, "parse_error": None, "prediction": {axis: parsed[axis]["score"] for axis in AXES}, "finish_reason": choice.get("finish_reason"), "usage": dict(usage), "attempts": attempts}
        if choice.get("finish_reason") != "length":
            break
    last_attempt = attempts[-1]
    return {**record, "response": last_attempt["response"], "parse_valid": False, "parse_error": last_attempt["parse_error"], "prediction": None, "finish_reason": last_attempt["finish_reason"], "usage": last_attempt["usage"], "attempts": attempts}


def solar_run(config: ExternalConfig) -> dict[str, Any]:
    records = request_records(config)
    _assert_gpus_idle(config.solar.gpu_scope)
    preflight = _solar_preflight(config, records)
    container = f"mal2026-decoder-fewshot-solar-{config.solar.port}"
    endpoint = f"http://127.0.0.1:{config.solar.port}"
    log_path = runtime_dir(config) / "solar-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(_solar_server_command(config, container), stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            _wait_solar(process, endpoint)
            smoke_records = [next(row for row in records if row["condition"] == condition) for condition in CONDITIONS]
            smoke = [_solar_request(config, endpoint, row) for row in smoke_records]
            need(all(row["parse_valid"] for row in smoke), "Solar real smoke failed")
            _write_jsonl_fresh(restricted_dir(config) / "solar" / "smoke_predictions.jsonl", smoke)
            _write_json_fresh(public_dir(config) / "models" / "solar-open2-int4" / "smoke.json", {
                "schema_version": "mal2026-decoder-fewshot-solar-smoke-v1", "status": "passed", "requests": 2,
                "conditions": list(CONDITIONS), "preflight": preflight, "gpu_scope": list(config.solar.gpu_scope),
            })
            rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=config.solar.max_inflight) as pool:
                futures = {pool.submit(_solar_request, config, endpoint, row): index for index, row in enumerate(records)}
                resolved: dict[int, dict[str, Any]] = {}
                for future in as_completed(futures):
                    resolved[futures[future]] = future.result()
                rows = [resolved[index] for index in range(len(records))]
        finally:
            subprocess.run(["docker", "stop", "--time", "30", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=30)
    _assert_gpus_idle(config.solar.gpu_scope)
    prediction_path = restricted_dir(config) / "solar" / "predictions.jsonl"
    prediction_sha = _write_jsonl_fresh(prediction_path, rows)
    usage = Counter()
    for row in rows:
        usage.update({str(name): int(value) for name, value in row.get("usage", {}).items() if type(value) is int})
    return _aggregate_rows(config, "solar-open2-int4", config.solar.model_id, rows, {
        "provider": "local official Solar vLLM Docker", "docker_image": config.solar.docker_image,
        "docker_image_id": config.solar.docker_image_id, "gpu_scope": list(config.solar.gpu_scope),
        "prediction_sha256": prediction_sha, "usage": dict(usage), "preflight": preflight,
        "temperature": 0.0, "seed": config.seed, "reasoning_effort": "none",
    })


def aggregate(config: ExternalConfig) -> dict[str, Any]:
    keys = ["solar-open2-int4", *config.api_models]
    models: dict[str, Any] = {}
    for key in keys:
        path = public_dir(config) / "models" / key / "aggregate.json"
        need(path.is_file(), f"external model aggregate unavailable: {key}")
        models[key] = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "schema_version": "mal2026-decoder-fewshot-external-aggregate-v1", "status": "completed",
        "run_id": config.run_id, "conditions": list(CONDITIONS), "validation_rows": VALIDATION_ROWS,
        "models": {key: {"model_id": row["model_id"], "parse_failures": row["parse_failures"], "metrics": row["metrics"]} for key, row in models.items()},
        "average_target_used": False,
    }
    _write_json_fresh(public_dir(config) / "aggregate.json", result)
    return result
