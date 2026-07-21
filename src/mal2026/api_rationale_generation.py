"""Batched vLLM-server generation of restricted rationale-only artifacts."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .api_rationale_data import (
    AXES, RESTRICTED_ROOT, ROOT, APIRationaleContractError, aggregate_input_provenance,
    axes_for_task, decoder_messages, parse_rationale_output, load_writing_rows, sha256_file,
)
from .api_rationale_sft import RUN_ROOT, SUPPORTED_MODELS


OUTPUT_ROOT = RESTRICTED_ROOT / "decoder_generation_v1"


class APIRationaleGenerationError(APIRationaleContractError):
    """Raised for server, adapter, parsing, or private-output contract violations."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise APIRationaleGenerationError(message)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class APIRationaleGenerationConfig:
    schema_version: str
    run_id: str
    base_key: str
    model_id: str
    model_revision: str
    model_path: str
    task: str
    adapter_path: str
    adapter_alias: str
    source: str
    restricted_output_dir: str
    max_new_tokens: int
    max_model_len: int
    client_max_inflight: int

    @classmethod
    def from_json(cls, path: Path) -> "APIRationaleGenerationConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        _need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "generation config has unknown or missing fields")
        value = cls(**raw); value.validate(); return value

    def validate(self) -> None:
        _need(self.schema_version == "mal2026-api-rationale-generation-v1", "unexpected generation config schema")
        _need(self.base_key in SUPPORTED_MODELS and SUPPORTED_MODELS[self.base_key] == (self.model_id, self.model_revision), "generation base identity differs")
        axes_for_task(self.task)
        _need(self.source in {"train", "validation"}, "generation source differs")
        path = Path(self.model_path); adapter = Path(self.adapter_path); output = Path(self.restricted_output_dir)
        _need(path.is_absolute() and path.is_dir() and not path.is_symlink() and path.name.endswith(self.model_revision), "generation model snapshot differs")
        _need(adapter.is_absolute() and adapter.is_dir() and adapter.parent.parent == RUN_ROOT.resolve() and adapter.name == "adapter", "generation adapter path differs")
        _need(output.is_absolute() and output.parent == OUTPUT_ROOT.resolve() and not output.exists(), "generation output must be a fresh restricted direct child")
        # The first Phi bundled validation attempt established that 384 output
        # tokens can terminate a JSON-schema constrained response before it
        # closes; a uniform 512-token -002 replay reached the same one-row
        # deterministic length gate.  Both failed artifacts are preserved.
        # Bundle lineage -003 therefore retains 512 tokens and adds the
        # schema-level per-axis character bound below.  A subsequent Phi
        # content-only -001 job exposed the same finish=length category under
        # its 192-token budget, so axis lineage -002 adds that same bound
        # uniformly for all three bases/axes.  All failed/superseded outputs
        # remain preserved under their original ignored directories.
        expected_suffix = "003" if self.task == "bundle" else "002"
        expected_tokens = 512 if self.task == "bundle" else 192
        _need(self.max_model_len == 3072 and self.max_new_tokens == expected_tokens, "generation token budget differs")
        _need(self.client_max_inflight == 256, "generation client concurrency differs")
        expected = f"api-rationale-generation-v1-{self.base_key}-{self.task}-{self.source}-{expected_suffix}"
        _need(self.run_id == expected and output.name == expected, "generation lineage does not bind base/task/source")
        _need(self.adapter_alias == f"api-rationale-{self.base_key}-{self.task}", "adapter alias does not bind base/task")


def _adapter_completion(config: APIRationaleGenerationConfig) -> Mapping[str, Any]:
    completion = Path(config.adapter_path).parent / "training_complete.json"
    try:
        value = json.loads(completion.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise APIRationaleGenerationError("adapter lacks completed training provenance") from exc
    _need(isinstance(value, dict) and value.get("status") == "completed", "adapter provenance is incomplete")
    for key, expected in (("base_key", config.base_key), ("model_id", config.model_id), ("model_revision", config.model_revision), ("task", config.task)):
        _need(value.get(key) == expected, "adapter provenance differs from generation config")
    return value


def _response_schema(axes: tuple[str, ...], *, rationale_character_limit: int | None = None) -> dict[str, Any]:
    properties: dict[str, Any] = {"schema_version": {"type": "string", "const": "rationale-only-v1"}}
    for axis in axes:
        rationale: dict[str, Any] = {"type": "string", "minLength": 1}
        if rationale_character_limit is not None:
            _need(rationale_character_limit >= 1, "rationale character limit must be positive")
            rationale["maxLength"] = rationale_character_limit
        properties[axis] = {"type": "object", "properties": {"rationale": rationale}, "required": ["rationale"], "additionalProperties": False}
    return {"type": "object", "properties": properties, "required": ["schema_version", *axes], "additionalProperties": False}


def _server_attestation(config: APIRationaleGenerationConfig, endpoint: str, path: Path) -> Mapping[str, Any]:
    parsed = urlparse(endpoint)
    _need(parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port is not None, "generation endpoint must be localhost HTTP")
    _need(path.is_file() and not path.is_symlink(), "generation server attestation missing")
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise APIRationaleGenerationError("generation server attestation unreadable") from exc
    expected = {
        "schema_version": "mal2026-api-rationale-generation-v1-server-attestation-v1", "server_host": "127.0.0.1",
        "server_port": parsed.port, "physical_gpus": [0, 1, 2, 3], "tensor_parallel_size": 4,
        "max_model_len": config.max_model_len, "server_process_environment_verified": True,
    }
    _need(isinstance(value, dict) and all(value.get(key) == expected_value for key, expected_value in expected.items()), "generation server attestation differs")
    _need(value.get("model_id") == config.model_id and value.get("model_revision") == config.model_revision and value.get("adapter_path") == str(Path(config.adapter_path).resolve()) and value.get("adapter_alias") == config.adapter_alias, "generation server model/adapter attestation differs")
    return value


def _request(endpoint: str, body: dict[str, Any]) -> tuple[dict[str, str] | None, str | None]:
    wire = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    try:
        with urlopen(Request(endpoint + "/v1/chat/completions", data=wire, headers={"Content-Type": "application/json"}, method="POST"), timeout=180) as response:
            outer = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return None, "http_429" if exc.code == 429 else ("http_5xx" if 500 <= exc.code <= 599 else "http_4xx")
    except (URLError, TimeoutError):
        return None, "connection_or_timeout"
    except json.JSONDecodeError:
        return None, "outer_json"
    if not isinstance(outer, dict) or not isinstance(outer.get("choices"), list) or len(outer["choices"]) != 1:
        return None, "envelope"
    choice = outer["choices"][0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop" or not isinstance(choice.get("message"), dict):
        return None, "finish"
    return choice["message"].get("content"), None


def _call(endpoint: str, task: Mapping[str, Any]) -> dict[str, Any]:
    categories: list[str] = []
    for attempt in range(1, 3):
        content, failure = _request(endpoint, task["body"])
        if failure is None:
            parsed = parse_rationale_output(content, task["axes"])
            return {"source_id": task["source_id"], "rationale": parsed, "parse_valid": parsed is not None, "failure_category": None if parsed is not None else "rationale_schema", "attempts": attempt}
        categories.append(failure)
        if failure not in {"http_429", "http_5xx", "connection_or_timeout"} or attempt == 2:
            return {"source_id": task["source_id"], "rationale": None, "parse_valid": False, "failure_category": failure, "attempts": attempt}
        time.sleep(0.2 * attempt)
    raise AssertionError("unreachable retry path")


def _tasks(config: APIRationaleGenerationConfig) -> Iterator[dict[str, Any]]:
    axes = axes_for_task(config.task)
    # The 192-character bound is deliberately an output-schema control, not a
    # score/reward signal.  It exceeds every completed Phi rationale observed
    # before this repair (bundled maxima 145/114/119 and content-only maximum
    # 144 characters), while forcing an otherwise deterministic runaway field
    # to terminate and close its JSON object.
    schema = _response_schema(axes, rationale_character_limit=192)
    for row in load_writing_rows(config.source, include_scores=False):
        yield {
            "source_id": row.identifier, "axes": axes,
            "body": {"model": config.adapter_alias, "temperature": 0.0, "top_p": 1.0, "max_tokens": config.max_new_tokens,
                     "messages": decoder_messages(row, axes),
                     "response_format": {"type": "json_schema", "json_schema": {"name": "mal2026_rationale_only_v1", "strict": True, "schema": schema}}},
        }


def run_api_rationale_generation(config: APIRationaleGenerationConfig, endpoint: str, server_attestation: Path) -> dict[str, Any]:
    """Generate one all-source private rationale artifact; never save prompts/completions."""
    config.validate(); completion = _adapter_completion(config); attestation = _server_attestation(config, endpoint, server_attestation)
    tasks = list(_tasks(config)); output = Path(config.restricted_output_dir)
    output.mkdir(mode=0o700, parents=True)
    manifest = {
        "schema_version": config.schema_version, "status": "running", "run_id": config.run_id, "source": config.source,
        "base_key": config.base_key, "task": config.task, "expected_records": len(tasks), "config": asdict(config),
        "adapter_training_completion_sha256": _sha(Path(config.adapter_path).parent / "training_complete.json"),
        "server_attestation_sha256": _sha(server_attestation), "input_provenance": aggregate_input_provenance(),
        "source_writing_scores_read_or_prompted": False, "candidate_scores_read_or_prompted": False,
        "raw_prompts_or_model_completions_persisted": False,
        "rationale_character_limit_per_axis": 192,
    }
    _atomic_json(output / "manifest.json", manifest)
    records_path = output / "generated_rationales.jsonl"; failures: dict[str, int] = {}
    pending: set[Any] = set(); iterator = iter(tasks); exhausted = False
    with records_path.open("x", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=config.client_max_inflight) as pool:
        while pending or not exhausted:
            while not exhausted and len(pending) < config.client_max_inflight:
                try: task = next(iterator)
                except StopIteration: exhausted = True; break
                pending.add(pool.submit(_call, endpoint, task))
            if not pending: continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                item = future.result(); category = item.get("failure_category")
                if category: failures[str(category)] = failures.get(str(category), 0) + 1
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
    count = sum(1 for _ in records_path.open(encoding="utf-8"))
    _need(count == len(tasks), "generation record count is incomplete")
    valid = count - sum(failures.values())
    report = {
        "status": "completed" if valid == count else "failed_gates", "run_id": config.run_id, "source": config.source,
        "base_key": config.base_key, "task": config.task,
        "counts": {"expected": len(tasks), "observations": count, "parse_valid": valid},
        "hard_gates": {"complete_records": count == len(tasks), "all_outputs_parse_valid": valid == count, "zero_transport_or_schema_failures": not failures},
        "failure_categories": dict(sorted(failures.items())), "generated_rationales_sha256": _sha(records_path),
        "adapter_training_completion_sha256": manifest["adapter_training_completion_sha256"], "server_attestation_sha256": manifest["server_attestation_sha256"],
        "model_attested": bool(attestation), "training_provenance_bound": bool(completion),
        "source_writing_scores_read_or_prompted": False, "candidate_scores_read_or_prompted": False,
        "raw_prompts_or_model_completions_persisted": False,
        "rationale_character_limit_per_axis": 192,
    }
    _atomic_json(output / "aggregate_generation_report.json", report)
    manifest["status"] = "completed" if all(report["hard_gates"].values()) else "failed_gates"; manifest["completed_at"] = _now(); manifest["aggregate_report_sha256"] = _sha(output / "aggregate_generation_report.json")
    _atomic_json(output / "manifest.json", manifest)
    if not all(report["hard_gates"].values()):
        raise APIRationaleGenerationError("generation hard gate failed; preserve restricted artifact and stop")
    return report
