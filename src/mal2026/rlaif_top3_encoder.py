"""Private top-three RLAIF rationale generation and three-axis regression.

This is deliberately a separate protocol from ``api_score_regression``.  It
binds exactly the three complete RLAIF bundle adapters selected from v8 and
never combines their outputs.  Restricted essays and generated rationales are
only read or written beneath the ignored restricted root; public artifacts
contain aggregate counts, checksums, and metrics only.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .api_rationale_data import (
    AXES,
    APIRationaleContractError,
    EXPECTED_ESSAYS,
    ROOT,
    SOURCE_SHA256,
    decoder_messages,
    load_writing_rows,
    parse_rationale_output,
    sha256_file,
)


RLAIF_FINAL_SUMMARY = ROOT / "outputs" / "aggregate-reports" / "rlaif-grpo-prompt-ensemble-v8-20260722-023.final-summary.json"
RLAIF_OUTPUT_ROOT = ROOT / "outputs" / "rlaif-grpo-prompt-ensemble-v8"
RESTRICTED_ROOT = ROOT / "data" / "processed" / "restricted" / "rlaif_top3_encoder_v1"
GENERATION_ROOT = RESTRICTED_ROOT / "rationales"
TRAINING_ROOT = ROOT / "outputs" / "rlaif-top3-score-regression-v1"
EVALUATION_ROOT = ROOT / "outputs" / "rlaif-top3-score-regression-evals-v1"

ENCODER = {
    "backbone_key": "qwen25_7b",
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    "model_path": ROOT / "outputs" / "model-cache" / "Qwen--Qwen2.5-7B-Instruct-a09a35458c702b33eeacc393d103063234e8bc28",
    "pooling": "last_nonpad",
}

# These are the three highest frozen-Qwen macro scores among independent,
# complete three-axis bundle adapters.  ``all5`` is its RLAIF reward arm, not a
# generation ensemble.
SELECTIONS: dict[str, dict[str, Any]] = {
    "rank1_midm2_random1": {
        "rank": 1,
        "base_key": "midm2_base",
        "arm": "random1",
        "frozen_macro": 4.189100,
        "model_id": "K-intelligence/Midm-2.0-Base-Instruct",
        "model_revision": "35479c5fc9a18a5db7cc6dbadcf1db68db7beab0",
        "model_path": ROOT / "outputs" / "model-cache" / "K-intelligence--Midm-2.0-Base-Instruct-35479c5fc9a18a5db7cc6dbadcf1db68db7beab0",
        "training_dir": RLAIF_OUTPUT_ROOT / "rlaif-grpo-prompt-ensemble-v8-midm2_base-bundle-random1-full-022",
        "training_run_id": "rlaif-grpo-prompt-ensemble-v8-midm2_base-bundle-random1-full-022",
    },
    "rank2_ax4_random1": {
        "rank": 2,
        "base_key": "ax4_light",
        "arm": "random1",
        "frozen_macro": 4.187033,
        "model_id": "skt/A.X-4.0-Light",
        "model_revision": "ba21c20ea1b31ded1ec3e2fb432335077dc4be98",
        "model_path": ROOT / "outputs" / "model-cache" / "skt--A.X-4.0-Light-ba21c20ea1b31ded1ec3e2fb432335077dc4be98",
        "training_dir": RLAIF_OUTPUT_ROOT / "rlaif-grpo-prompt-ensemble-v8-ax4_light-bundle-random1-full-023",
        "training_run_id": "rlaif-grpo-prompt-ensemble-v8-ax4_light-bundle-random1-full-023",
    },
    "rank3_ax4_all5": {
        "rank": 3,
        "base_key": "ax4_light",
        "arm": "all5",
        "frozen_macro": 4.184067,
        "model_id": "skt/A.X-4.0-Light",
        "model_revision": "ba21c20ea1b31ded1ec3e2fb432335077dc4be98",
        "model_path": ROOT / "outputs" / "model-cache" / "skt--A.X-4.0-Light-ba21c20ea1b31ded1ec3e2fb432335077dc4be98",
        "training_dir": RLAIF_OUTPUT_ROOT / "rlaif-grpo-prompt-ensemble-v8-ax4_light-bundle-all5-full-023",
        "training_run_id": "rlaif-grpo-prompt-ensemble-v8-ax4_light-bundle-all5-full-023",
    },
}


class RLAIFTop3EncoderError(APIRationaleContractError):
    """Raised before a top-three generation/regression contract is violated."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RLAIFTop3EncoderError(message)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RLAIFTop3EncoderError(f"{description} is unreadable") from exc
    _need(isinstance(value, dict), f"{description} must be an object")
    return value


def _selection(source_key: str, selected_rank: int) -> Mapping[str, Any]:
    value = SELECTIONS.get(source_key)
    _need(value is not None and value["rank"] == selected_rank, "top-three source identity differs")
    return value


def selected_sources() -> tuple[str, str, str]:
    return tuple(key for key, _ in sorted(SELECTIONS.items(), key=lambda item: item[1]["rank"]))  # type: ignore[return-value]


def _summary_binding() -> str:
    _need(RLAIF_FINAL_SUMMARY.is_file() and not RLAIF_FINAL_SUMMARY.is_symlink(), "RLAIF v8 final summary is unavailable")
    summary = _read_json(RLAIF_FINAL_SUMMARY, "RLAIF v8 final summary")
    _need(summary.get("schema_version") == "mal2026-rlaif-grpo-prompt-ensemble-v8-final-summary-v1" and summary.get("status") == "completed", "RLAIF v8 final summary is incomplete")
    arms = summary.get("arms")
    _need(isinstance(arms, list), "RLAIF v8 summary arms are invalid")
    for expected in SELECTIONS.values():
        matching = [arm for arm in arms if isinstance(arm, dict) and arm.get("base_key") == expected["base_key"] and arm.get("task") == "bundle" and arm.get("arm") == expected["arm"]]
        _need(len(matching) == 1 and matching[0].get("status") == "completed", "selected RLAIF bundle arm is absent")
        _need(float(matching[0].get("macro_mean")) == expected["frozen_macro"], "selected frozen macro differs")
    return _sha(RLAIF_FINAL_SUMMARY)


def _adapter_completion(source_key: str, selected_rank: int) -> tuple[Mapping[str, Any], Path]:
    selection = _selection(source_key, selected_rank)
    training_dir = Path(selection["training_dir"])
    completion_path = training_dir / "training_complete.json"
    adapter = training_dir / "adapter"
    _need(training_dir.is_dir() and adapter.is_dir() and completion_path.is_file(), "selected RLAIF adapter is unavailable")
    completion = _read_json(completion_path, "selected RLAIF training completion")
    config = completion.get("config")
    _need(isinstance(config, dict), "selected RLAIF completion lacks config")
    expected = {
        "status": "completed",
        "run_id": selection["training_run_id"],
        "base_key": selection["base_key"],
        "arm": selection["arm"],
        "task": "bundle",
        "model_id": selection["model_id"],
        "model_revision": selection["model_revision"],
    }
    _need(all(completion.get(key) == value for key, value in expected.items()), "selected RLAIF completion identity differs")
    _need(config.get("output_dir") == str(training_dir.resolve()) and config.get("phase") == "full", "selected RLAIF completion lineage differs")
    _need(Path(selection["model_path"]).is_dir() and Path(selection["model_path"]).name.endswith(str(selection["model_revision"])), "selected RLAIF base snapshot is unavailable")
    return completion, adapter


def generation_dir(source_key: str, source: str, phase: str) -> Path:
    return GENERATION_ROOT / f"rlaif-top3-rationale-generation-v1-{source_key}-{source}-{phase}-001"


@dataclass(frozen=True)
class RLAIFTop3GenerationConfig:
    schema_version: str
    run_id: str
    source_key: str
    selected_rank: int
    source: str
    phase: str
    model_id: str
    model_revision: str
    model_path: str
    rlaif_adapter_path: str
    rlaif_training_completion_path: str
    restricted_output_dir: str
    max_new_tokens: int
    max_model_len: int
    client_max_inflight: int
    record_limit: int

    @classmethod
    def from_json(cls, path: Path) -> "RLAIFTop3GenerationConfig":
        raw = _read_json(path, "top-three generation config")
        _need(set(raw) == set(cls.__dataclass_fields__), "top-three generation config has unknown or missing fields")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self, *, require_fresh_output: bool = True) -> None:
        _need(self.schema_version == "mal2026-rlaif-top3-rationale-generation-v1", "top-three generation schema differs")
        selection = _selection(self.source_key, self.selected_rank)
        _need(self.source in {"train", "validation"} and self.phase in {"gpu0_preflight", "full"}, "top-three generation source/phase differs")
        _need((self.model_id, self.model_revision, self.model_path) == (selection["model_id"], selection["model_revision"], str(Path(selection["model_path"]).resolve())), "top-three decoder snapshot differs")
        training_dir = Path(selection["training_dir"]).resolve()
        _need(self.rlaif_adapter_path == str((training_dir / "adapter").resolve()) and self.rlaif_training_completion_path == str((training_dir / "training_complete.json").resolve()), "top-three adapter provenance differs")
        output = Path(self.restricted_output_dir)
        _need(output.is_absolute() and output.parent == GENERATION_ROOT.resolve() and (not output.exists() if require_fresh_output else output.is_dir()), "top-three restricted output root differs")
        expected_id = f"rlaif-top3-rationale-generation-v1-{self.source_key}-{self.source}-{self.phase}-001"
        _need(self.run_id == expected_id and output.name == expected_id, "top-three generation lineage differs")
        _need(self.max_new_tokens == 512 and self.max_model_len == 3072 and self.client_max_inflight == 256, "top-three generation decoding contract differs")
        expected_records = EXPECTED_ESSAYS[self.source] if self.phase == "full" else 1
        _need(self.record_limit == expected_records and (self.phase == "full" or self.source == "train"), "top-three generation record limit differs")


def generation_config(source_key: str, source: str, phase: str) -> dict[str, Any]:
    selection = SELECTIONS[source_key]
    training_dir = Path(selection["training_dir"])
    run_id = f"rlaif-top3-rationale-generation-v1-{source_key}-{source}-{phase}-001"
    return {
        "schema_version": "mal2026-rlaif-top3-rationale-generation-v1",
        "run_id": run_id,
        "source_key": source_key,
        "selected_rank": selection["rank"],
        "source": source,
        "phase": phase,
        "model_id": selection["model_id"],
        "model_revision": selection["model_revision"],
        "model_path": str(Path(selection["model_path"]).resolve()),
        "rlaif_adapter_path": str((training_dir / "adapter").resolve()),
        "rlaif_training_completion_path": str((training_dir / "training_complete.json").resolve()),
        "restricted_output_dir": str(generation_dir(source_key, source, phase).resolve()),
        "max_new_tokens": 512,
        "max_model_len": 3072,
        "client_max_inflight": 256,
        "record_limit": EXPECTED_ESSAYS[source] if phase == "full" else 1,
    }


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "const": "rationale-only-v1"},
            **{
                axis: {
                    "type": "object",
                    "properties": {"rationale": {"type": "string", "minLength": 1, "maxLength": 192}},
                    "required": ["rationale"],
                    "additionalProperties": False,
                }
                for axis in AXES
            },
        },
        "required": ["schema_version", *AXES],
        "additionalProperties": False,
    }


def _generation_tasks(config: RLAIFTop3GenerationConfig) -> Iterator[dict[str, Any]]:
    schema = _response_schema()
    rows = load_writing_rows(config.source, include_scores=False)
    for row in rows[:config.record_limit]:
        yield {
            "source_id": row.identifier,
            "body": {
                "model": f"rlaif-top3-{config.source_key}",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": config.max_new_tokens,
                "messages": decoder_messages(row, AXES),
                "response_format": {"type": "json_schema", "json_schema": {"name": "mal2026_rlaif_top3_rationale_v1", "strict": True, "schema": schema}},
            },
        }


def _generation_attestation(config: RLAIFTop3GenerationConfig, endpoint: str, path: Path) -> Mapping[str, Any]:
    parsed = urlparse(endpoint)
    _need(parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port is not None, "generation endpoint must be localhost HTTP")
    value = _read_json(path, "top-three generation server attestation")
    expected = {
        "schema_version": "mal2026-rlaif-top3-rationale-generation-server-attestation-v1",
        "server_host": "127.0.0.1",
        "server_port": parsed.port,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "model_path": config.model_path,
        "rlaif_adapter_path": config.rlaif_adapter_path,
        "adapter_alias": f"rlaif-top3-{config.source_key}",
        "max_model_len": config.max_model_len,
        "server_process_environment_verified": True,
    }
    _need(all(value.get(key) == expected_value for key, expected_value in expected.items()), "generation server attestation differs")
    expected_tp = 4 if config.phase == "full" else 1
    expected_gpus = [0, 1, 2, 3] if config.phase == "full" else [0]
    _need(value.get("tensor_parallel_size") == expected_tp and value.get("physical_gpus") == expected_gpus, "generation server topology differs")
    return value


def _request(endpoint: str, body: Mapping[str, Any]) -> tuple[str | None, str | None]:
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
    content = choice["message"].get("content")
    return (content, None) if isinstance(content, str) else (None, "content")


def _generate_one(endpoint: str, task: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for attempt in range(1, 3):
        content, failure = _request(endpoint, task["body"])
        if failure is None:
            parsed = parse_rationale_output(content, AXES)
            return {"source_id": task["source_id"], "rationale": parsed, "parse_valid": parsed is not None, "failure_category": None if parsed is not None else "rationale_schema", "attempts": attempt}
        failures.append(failure)
        if failure not in {"http_429", "http_5xx", "connection_or_timeout"} or attempt == 2:
            return {"source_id": task["source_id"], "rationale": None, "parse_valid": False, "failure_category": failure, "attempts": attempt}
        time.sleep(0.2 * attempt)
    raise AssertionError("unreachable generation retry")


def run_rationale_generation(config: RLAIFTop3GenerationConfig, endpoint: str, server_attestation: Path) -> dict[str, Any]:
    """Generate one independent source; no score field is loaded or prompted."""
    config.validate()
    summary_sha = _summary_binding()
    completion, adapter = _adapter_completion(config.source_key, config.selected_rank)
    attestation = _generation_attestation(config, endpoint, server_attestation)
    tasks = list(_generation_tasks(config))
    _need(len(tasks) == config.record_limit, "top-three generation task count differs")
    output = Path(config.restricted_output_dir)
    output.mkdir(mode=0o700, parents=True)
    manifest = {
        "schema_version": config.schema_version,
        "status": "running",
        "run_id": config.run_id,
        "config": asdict(config),
        "expected_records": len(tasks),
        "rlaif_v8_final_summary_sha256": summary_sha,
        "rlaif_training_completion_sha256": _sha(Path(config.rlaif_training_completion_path)),
        "rlaif_adapter_sha256": _sha(adapter / "adapter_model.safetensors"),
        "server_attestation_sha256": _sha(server_attestation),
        "source_writing_scores_read_or_prompted": False,
        "raw_prompts_or_model_completions_persisted_outside_restricted_root": False,
        "rationale_character_limit_per_axis": 192,
    }
    _atomic_json(output / "manifest.json", manifest)
    records = output / "generated_rationales.jsonl"
    failures: dict[str, int] = {}
    pending: set[Any] = set()
    iterator = iter(tasks)
    exhausted = False
    with records.open("x", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=config.client_max_inflight) as pool:
        while pending or not exhausted:
            while not exhausted and len(pending) < config.client_max_inflight:
                try:
                    task = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(pool.submit(_generate_one, endpoint, task))
            if not pending:
                continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                item = future.result()
                category = item["failure_category"]
                if category is not None:
                    failures[str(category)] = failures.get(str(category), 0) + 1
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
    count = sum(1 for _ in records.open(encoding="utf-8"))
    valid = count - sum(failures.values())
    report = {
        "status": "completed" if valid == len(tasks) else "failed_gates",
        "run_id": config.run_id,
        "source_key": config.source_key,
        "selected_rank": config.selected_rank,
        "source": config.source,
        "phase": config.phase,
        "counts": {"expected": len(tasks), "observations": count, "parse_valid": valid},
        "hard_gates": {"complete_records": count == len(tasks), "all_outputs_parse_valid": valid == len(tasks), "zero_transport_or_schema_failures": not failures},
        "failure_categories": dict(sorted(failures.items())),
        "generated_rationales_sha256": _sha(records),
        "rlaif_v8_final_summary_sha256": summary_sha,
        "rlaif_training_completion_sha256": manifest["rlaif_training_completion_sha256"],
        "rlaif_adapter_sha256": manifest["rlaif_adapter_sha256"],
        "server_attestation_sha256": manifest["server_attestation_sha256"],
        "model_attested": bool(attestation),
        "training_provenance_bound": bool(completion),
        "source_writing_scores_read_or_prompted": False,
        "raw_prompts_or_model_completions_persisted_outside_restricted_root": False,
    }
    _atomic_json(output / "aggregate_generation_report.json", report)
    manifest["status"] = "completed" if all(report["hard_gates"].values()) else "failed_gates"
    manifest["aggregate_report_sha256"] = _sha(output / "aggregate_generation_report.json")
    _atomic_json(output / "manifest.json", manifest)
    _need(all(report["hard_gates"].values()), "top-three rationale generation hard gate failed")
    return report


def _load_generated_rationales(path: Path, source_key: str, source: str, phase: str, record_limit: int) -> dict[str, dict[str, str]]:
    root = path.resolve()
    _need(root.parent == GENERATION_ROOT.resolve() and root.is_dir() and not root.is_symlink(), "top-three rationale root differs")
    manifest = _read_json(root / "manifest.json", "top-three rationale manifest")
    report = _read_json(root / "aggregate_generation_report.json", "top-three rationale aggregate")
    config = manifest.get("config")
    expected = generation_config(source_key, source, phase)
    _need(manifest.get("status") == "completed" and report.get("status") == "completed" and config == expected, "top-three rationale provenance differs")
    _need(report.get("counts") == {"expected": record_limit, "observations": record_limit, "parse_valid": record_limit} and all(report.get("hard_gates", {}).values()), "top-three rationale aggregate gate failed")
    records = root / "generated_rationales.jsonl"
    _need(records.is_file() and report.get("generated_rationales_sha256") == _sha(records), "top-three rationale checksum differs")
    result: dict[str, dict[str, str]] = {}
    with records.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            _need(isinstance(raw, dict) and set(raw) == {"attempts", "failure_category", "parse_valid", "rationale", "source_id"}, "top-three rationale row schema differs")
            identifier = raw.get("source_id")
            rationale = raw.get("rationale")
            _need(isinstance(identifier, str) and identifier not in result and raw.get("parse_valid") is True and raw.get("failure_category") is None, "top-three rationale row is invalid")
            _need(isinstance(rationale, dict) and set(rationale) == set(AXES), "top-three rationale axes differ")
            values: dict[str, str] = {}
            for axis in AXES:
                value = rationale[axis]
                _need(isinstance(value, str) and bool(value.strip()), "top-three rationale text is invalid")
                values[axis] = value
            result[identifier] = values
    _need(len(result) == record_limit, "top-three rationale count differs")
    return result


def _finite(value: Any, name: str) -> float:
    _need(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be numeric")
    parsed = float(value)
    _need(math.isfinite(parsed), f"{name} must be finite")
    return parsed


def _rationale_text(rationales: Mapping[str, str]) -> str:
    _need(set(rationales) == set(AXES), "score-regression rationale must cover exactly three axes")
    return json.dumps({axis: {"rationale": rationales[axis]} for axis in AXES}, ensure_ascii=False, separators=(",", ":"))


def _input_text(prompt: str, essay: str, rationales: Mapping[str, str]) -> str:
    return f"<writing_prompt>\n{prompt}\n</writing_prompt>\n<student_essay>\n{essay}\n</student_essay>\n<evaluation_rationales>\n{_rationale_text(rationales)}\n</evaluation_rationales>"


def _labels(row: Any) -> list[float]:
    _need(row.scores is not None and set(row.scores) == set(AXES), "three-axis labels are unavailable")
    return [_finite(row.scores[axis], f"label.{axis}") for axis in AXES]


def training_dir(source_key: str, phase: str) -> Path:
    return TRAINING_ROOT / f"rlaif-top3-score-regression-v1-{source_key}-{phase}-001"


@dataclass(frozen=True)
class RLAIFTop3RegressionConfig:
    schema_version: str
    run_id: str
    source_key: str
    selected_rank: int
    phase: str
    backbone_key: str
    model_id: str
    model_revision: str
    model_path: str
    decoder_train_generation_dir: str
    decoder_validation_generation_dir: str | None
    output_dir: str
    score_fields: tuple[str, str, str]
    seed: int
    max_length: int
    learning_rate: float
    num_train_epochs: float
    max_steps: int
    train_record_limit: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    logging_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    training_dtype: str

    @classmethod
    def from_json(cls, path: Path) -> "RLAIFTop3RegressionConfig":
        raw = _read_json(path, "top-three regression config")
        _need(isinstance(raw.get("score_fields"), list), "top-three score fields must be a JSON list")
        raw["score_fields"] = tuple(raw["score_fields"])
        _need(set(raw) == set(cls.__dataclass_fields__), "top-three regression config has unknown or missing fields")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self, *, require_fresh_output: bool = True) -> None:
        _need(self.schema_version == "mal2026-rlaif-top3-score-regression-v1", "top-three regression schema differs")
        _selection(self.source_key, self.selected_rank)
        _need(self.phase in {"gpu0_preflight", "full"}, "top-three regression phase differs")
        _need((self.backbone_key, self.model_id, self.model_revision, self.model_path) == (ENCODER["backbone_key"], ENCODER["model_id"], ENCODER["model_revision"], str(Path(ENCODER["model_path"]).resolve())), "top-three encoder backbone differs")
        model, output = Path(self.model_path), Path(self.output_dir)
        _need(model.is_dir() and not model.is_symlink() and model.name.endswith(self.model_revision), "top-three encoder snapshot is unavailable")
        _need(output.is_absolute() and output.parent == TRAINING_ROOT.resolve() and (not output.exists() if require_fresh_output else output.is_dir()), "top-three training output root differs")
        _need(self.run_id == f"rlaif-top3-score-regression-v1-{self.source_key}-{self.phase}-001" and output.name == self.run_id, "top-three regression lineage differs")
        _need(self.score_fields == AXES, "only content, organization, and expression are permitted score targets")
        _need(self.seed == 2026072501 and self.max_length == 3072 and self.learning_rate == 2e-5 and self.logging_steps > 0, "top-three regression optimization contract differs")
        _need((self.lora_r, self.lora_alpha, self.lora_dropout, self.training_dtype) == (16, 32, 0.05, "float32"), "top-three regression LoRA/numeric contract differs")
        expected_train = generation_dir(self.source_key, "train", self.phase)
        _need(self.decoder_train_generation_dir == str(expected_train.resolve()), "top-three train rationale lineage differs")
        if self.phase == "full":
            _need(self.decoder_validation_generation_dir == str(generation_dir(self.source_key, "validation", "full").resolve()), "top-three validation rationale lineage differs")
            _need((self.num_train_epochs, self.max_steps, self.train_record_limit, self.per_device_train_batch_size, self.gradient_accumulation_steps) == (12.0, -1, 2000, 2, 8), "top-three full training schedule differs")
        else:
            _need(self.decoder_validation_generation_dir is None and (self.num_train_epochs, self.max_steps, self.train_record_limit, self.per_device_train_batch_size, self.gradient_accumulation_steps) == (1.0, 1, 1, 1, 1), "top-three GPU0 preflight schedule differs")


def regression_config(source_key: str, phase: str) -> dict[str, Any]:
    selection = SELECTIONS[source_key]
    full = phase == "full"
    run_id = f"rlaif-top3-score-regression-v1-{source_key}-{phase}-001"
    return {
        "schema_version": "mal2026-rlaif-top3-score-regression-v1",
        "run_id": run_id,
        "source_key": source_key,
        "selected_rank": selection["rank"],
        "phase": phase,
        "backbone_key": ENCODER["backbone_key"],
        "model_id": ENCODER["model_id"],
        "model_revision": ENCODER["model_revision"],
        "model_path": str(Path(ENCODER["model_path"]).resolve()),
        "decoder_train_generation_dir": str(generation_dir(source_key, "train", phase).resolve()),
        "decoder_validation_generation_dir": str(generation_dir(source_key, "validation", "full").resolve()) if full else None,
        "output_dir": str(training_dir(source_key, phase).resolve()),
        "score_fields": list(AXES),
        "seed": 2026072501,
        "max_length": 3072,
        "learning_rate": 2e-5,
        "num_train_epochs": 12.0 if full else 1.0,
        "max_steps": -1 if full else 1,
        "train_record_limit": 2000 if full else 1,
        "per_device_train_batch_size": 2 if full else 1,
        "gradient_accumulation_steps": 8 if full else 1,
        "logging_steps": 5,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "training_dtype": "float32",
    }


def _training_examples(config: RLAIFTop3RegressionConfig) -> list[dict[str, Any]]:
    generated = _load_generated_rationales(Path(config.decoder_train_generation_dir), config.source_key, "train", config.phase, config.train_record_limit)
    rows = load_writing_rows("train", include_scores=True)[:config.train_record_limit]
    examples = [{"text": _input_text(row.prompt, row.essay, generated[row.identifier]), "labels": _labels(row)} for row in rows]
    _need(len(examples) == config.train_record_limit, "top-three training input count differs")
    return examples


def _validation_examples(config: RLAIFTop3RegressionConfig) -> list[dict[str, Any]]:
    _need(config.phase == "full" and config.decoder_validation_generation_dir is not None, "validation requires a full top-three training source")
    generated = _load_generated_rationales(Path(config.decoder_validation_generation_dir), config.source_key, "validation", "full", EXPECTED_ESSAYS["validation"])
    rows = load_writing_rows("validation", include_scores=True)
    examples = [{"source_id": row.identifier, "text": _input_text(row.prompt, row.essay, generated[row.identifier]), "labels": _labels(row)} for row in rows]
    _need(len(examples) == EXPECTED_ESSAYS["validation"], "top-three validation input count differs")
    return examples


def _tokenize_examples(examples: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int, *, include_source: bool) -> Any:
    from datasets import Dataset
    payload: dict[str, Any] = {"text": [item["text"] for item in examples], "labels": [item["labels"] for item in examples]}
    if include_source:
        payload["source_id"] = [item["source_id"] for item in examples]
    dataset = Dataset.from_dict(payload)
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])


def _build_model(config: RLAIFTop3RegressionConfig) -> tuple[Any, list[str]]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError("top-three score regression requires .venv-standard") from exc
    base = AutoModel.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    leaves = {name.rsplit(".", maxsplit=1)[-1] for name, _ in base.named_modules()}
    _need(set(targets) <= leaves, "Qwen encoder lacks reviewed LoRA targets")
    peft = get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=targets, bias="none"))
    hidden = getattr(peft.config, "hidden_size", None)
    _need(type(hidden) is int and hidden > 0, "Qwen encoder lacks hidden size")

    class Regressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = peft
            self.regression_head = nn.Linear(hidden, len(AXES))

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None, **_: Any) -> Mapping[str, Any]:
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
            positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
            index = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
            _need(bool((index >= 0).all().item()), "all encoder sequences require a nonpad token")
            embedding = output[torch.arange(output.shape[0], device=output.device), index]
            logits = 1.0 + 4.0 * torch.sigmoid(self.regression_head(functional.normalize(embedding.float(), p=2, dim=-1)))
            result: dict[str, Any] = {"logits": logits}
            if labels is not None:
                _need(tuple(labels.shape[-1:]) == (len(AXES),), "three-axis regression label dimension differs")
                result["loss"] = functional.mse_loss(logits, labels.float())
            return result

    return Regressor(), targets


def _collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        labels = [feature.pop("labels") for feature in features]
        for feature in features:
            feature.pop("source_id", None)
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch
    return collate


def _finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {str(key): _finite(value, f"Trainer metric {key}") for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
    _need("train_loss" in result, "Trainer did not emit train_loss")
    return result


def run_score_regression(config: RLAIFTop3RegressionConfig) -> dict[str, Any]:
    config.validate()
    try:
        import torch
        from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError("top-three score regression requires .venv-standard") from exc
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        _need(tokenizer.eos_token is not None, "encoder tokenizer lacks pad/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    examples = _training_examples(config)
    dataset = _tokenize_examples(examples, tokenizer, config.max_length, include_source=False)
    model, targets = _build_model(config)
    args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_id,
        do_train=True,
        do_eval=False,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=False,
        tf32=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        max_grad_norm=0.1,
        optim="adamw_torch",
        logging_nan_inf_filter=False,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=_collator(tokenizer))
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None
    failed = False
    if trainer.is_world_process_zero():
        try:
            metrics = _finite_metrics(trained.metrics)
            _need(all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters() if parameter.requires_grad), "one or more trainable encoder parameters are non-finite")
            final = Path(config.output_dir) / "final_model"
            trainer.save_model(str(final))
            tokenizer.save_pretrained(str(final))
            state = final / "model.safetensors"
            _need(state.is_file(), "Trainer did not save a safetensors model state")
            payload = {
                "status": "completed",
                "run_id": config.run_id,
                "source_key": config.source_key,
                "selected_rank": config.selected_rank,
                "backbone_key": config.backbone_key,
                "model_id": config.model_id,
                "model_revision": config.model_revision,
                "score_fields": list(AXES),
                "train_records": len(examples),
                "global_step": int(trainer.state.global_step),
                "train_metrics": metrics,
                "lora_targets": targets,
                "model_state_sha256": _sha(state),
                "input_provenance": {"source_sha256": dict(SOURCE_SHA256), "rlaif_v8_final_summary_sha256": _summary_binding()},
                "config": asdict(config),
                "source_writing_scores_read_or_prompted": True,
                "score_targets": list(AXES),
                "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
            }
            completion = Path(config.output_dir) / "training_complete.json"
            _need(not completion.exists(), "top-three training completion already exists")
            _atomic_json(completion, payload)
        except Exception:
            failed = True
    state_payload: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state_payload, src=0)
    if state_payload[0]:
        raise RLAIFTop3EncoderError("rank-zero top-three training persistence/health gate failed")
    _need(isinstance(state_payload[1], dict) and state_payload[1].get("status") == "completed", "top-three training completion was not published")
    trainer.accelerator.wait_for_everyone()
    return state_payload[1]


def _average_rank(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in range(start, end):
            ranks[indexed[index][0]] = rank
        start = end
    return ranks


def _spearman(truth: Sequence[float], prediction: Sequence[float]) -> float:
    _need(len(truth) == len(prediction) and len(truth) >= 2, "Spearman needs aligned nontrivial vectors")
    left, right = _average_rank(truth), _average_rank(prediction)
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    denominator = math.sqrt(sum((value - left_mean) ** 2 for value in left) * sum((value - right_mean) ** 2 for value in right))
    _need(denominator > 0, "Spearman is undefined for constant ranks")
    return sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True)) / denominator


def three_axis_metrics(truth: Sequence[Sequence[float]], predictions: Sequence[Sequence[float]]) -> dict[str, Any]:
    _need(len(truth) == len(predictions) and len(truth) > 0, "metric vectors must align")
    result: dict[str, Any] = {}
    rmses: list[float] = []
    correlations: list[float] = []
    for index, axis in enumerate(AXES):
        observed = [float(row[index]) for row in truth]
        predicted = [float(row[index]) for row in predictions]
        _need(all(math.isfinite(value) for value in observed + predicted), "non-finite encoder prediction")
        rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(observed, predicted, strict=True)) / len(observed))
        correlation = _spearman(observed, predicted)
        result[axis] = {"rmse": rmse, "spearman": correlation}
        rmses.append(rmse)
        correlations.append(correlation)
    # This aggregates three evaluation axes only; it is not a fourth label or
    # predicted target.
    result["three_axis_macro_rmse"] = sum(rmses) / len(rmses)
    result["three_axis_macro_spearman"] = sum(correlations) / len(correlations)
    return result


def evaluation_dir(source_key: str) -> Path:
    return EVALUATION_ROOT / f"rlaif-top3-score-regression-eval-v1-{source_key}-validation-001"


@dataclass(frozen=True)
class RLAIFTop3RegressionEvalConfig:
    schema_version: str
    run_id: str
    source_key: str
    selected_rank: int
    training_metadata_path: str
    output_dir: str
    per_device_eval_batch_size: int

    @classmethod
    def from_json(cls, path: Path) -> "RLAIFTop3RegressionEvalConfig":
        raw = _read_json(path, "top-three evaluation config")
        _need(set(raw) == set(cls.__dataclass_fields__), "top-three evaluation config has unknown or missing fields")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self) -> None:
        _selection(self.source_key, self.selected_rank)
        _need(self.schema_version == "mal2026-rlaif-top3-score-regression-eval-v1", "top-three evaluation schema differs")
        metadata, output = Path(self.training_metadata_path), Path(self.output_dir)
        _need(metadata.is_absolute() and metadata == (training_dir(self.source_key, "full") / "training_complete.json").resolve(), "top-three evaluation training metadata differs")
        _need(output.is_absolute() and output.parent == EVALUATION_ROOT.resolve() and not output.exists(), "top-three evaluation output must be fresh")
        _need(self.run_id == f"rlaif-top3-score-regression-eval-v1-{self.source_key}-validation-001" and output.name == self.run_id and self.per_device_eval_batch_size == 4, "top-three evaluation identity differs")


def evaluation_config(source_key: str) -> dict[str, Any]:
    selection = SELECTIONS[source_key]
    run_id = f"rlaif-top3-score-regression-eval-v1-{source_key}-validation-001"
    return {
        "schema_version": "mal2026-rlaif-top3-score-regression-eval-v1",
        "run_id": run_id,
        "source_key": source_key,
        "selected_rank": selection["rank"],
        "training_metadata_path": str((training_dir(source_key, "full") / "training_complete.json").resolve()),
        "output_dir": str(evaluation_dir(source_key).resolve()),
        "per_device_eval_batch_size": 4,
    }


def _load_training(config: RLAIFTop3RegressionEvalConfig) -> tuple[Mapping[str, Any], RLAIFTop3RegressionConfig, Path]:
    metadata = _read_json(Path(config.training_metadata_path), "top-three training metadata")
    _need(metadata.get("status") == "completed" and metadata.get("source_key") == config.source_key and metadata.get("selected_rank") == config.selected_rank, "top-three training metadata is incomplete")
    raw = metadata.get("config")
    _need(isinstance(raw, dict) and isinstance(raw.get("score_fields"), list), "top-three training metadata lacks config")
    raw["score_fields"] = tuple(raw["score_fields"])
    train = RLAIFTop3RegressionConfig(**raw)
    train.validate(require_fresh_output=False)
    _need(train.phase == "full" and metadata.get("score_fields") == list(AXES), "top-three training target provenance differs")
    state = Path(train.output_dir) / "final_model" / "model.safetensors"
    _need(state.is_file() and _sha(state) == metadata.get("model_state_sha256"), "top-three trained state checksum differs")
    return metadata, train, state


def run_score_regression_evaluation(config: RLAIFTop3RegressionEvalConfig) -> dict[str, Any]:
    config.validate()
    metadata, train, state = _load_training(config)
    try:
        import numpy as np
        import torch
        from safetensors.torch import load_model
        from transformers import AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError("top-three score-regression evaluation requires .venv-standard") from exc
    tokenizer = AutoTokenizer.from_pretrained(train.model_path, revision=train.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = _validation_examples(train)
    dataset = _tokenize_examples(examples, tokenizer, train.max_length, include_source=True)
    model, _ = _build_model(train)
    missing, unexpected = load_model(model, str(state), strict=False)
    _need(not missing and not unexpected, "saved top-three model state differs from rebuilt model")
    output = Path(config.output_dir)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(output), do_train=False, do_eval=False, per_device_eval_batch_size=config.per_device_eval_batch_size, bf16=False, tf32=True, report_to=[], remove_unused_columns=False), data_collator=_collator(tokenizer))
    prediction = trainer.predict(dataset).predictions
    values = prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
    _need(len(values) == len(examples), "top-three prediction count differs")
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    labels: dict[str, list[float]] = {}
    for item, vector in zip(examples, values, strict=True):
        grouped[item["source_id"]].append([_finite(value, "prediction") for value in vector])
        labels[item["source_id"]] = [_finite(value, "label") for value in item["labels"]]
    _need(len(grouped) == EXPECTED_ESSAYS["validation"] and all(len(vectors) == 1 for vectors in grouped.values()) and set(grouped) == set(labels), "top-three validation must have exactly one prediction per essay")
    identifiers = sorted(grouped)
    metrics = three_axis_metrics([labels[identifier] for identifier in identifiers], [grouped[identifier][0] for identifier in identifiers])
    payload = {
        "status": "completed",
        "run_id": config.run_id,
        "training_run_id": metadata.get("run_id"),
        "source_key": config.source_key,
        "selected_rank": config.selected_rank,
        "backbone_key": train.backbone_key,
        "score_fields": list(AXES),
        "metrics": metrics,
        "validation": {"unique_essays": len(grouped), "input_records": len(examples), "predictions_per_essay": 1, "rationale_sources_combined": 0},
        "model_state_sha256": _sha(state),
        "config": asdict(config),
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
    }
    trainer.accelerator.wait_for_everyone()
    failed = False
    if trainer.is_world_process_zero():
        try:
            _need(output.is_dir() and not (output / "aggregate_metrics.json").exists(), "top-three evaluation output was reused")
            _atomic_json(output / "aggregate_metrics.json", payload)
        except Exception:
            failed = True
    state_payload: list[Any] = [failed, payload if trainer.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state_payload, src=0)
    if state_payload[0]:
        raise RLAIFTop3EncoderError("rank-zero top-three evaluation persistence failed")
    _need(isinstance(state_payload[1], dict) and state_payload[1].get("status") == "completed", "top-three evaluation completion was not published")
    trainer.accelerator.wait_for_everyone()
    return state_payload[1]
