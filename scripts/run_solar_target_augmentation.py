#!/usr/bin/env python3
"""Run source-grounded Solar Open2 INT4 axis/target-score augmentation.

The smoke and full populations are intentionally separate.  A full run is
accepted only after a complete smoke result and an explicit aggregate review
attestation for the same prompt/model/image bindings.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_rl_servers import assert_gpus_idle  # noqa: E402
from mal2026.solar_target_augmentation import (  # noqa: E402
    AXES,
    CONFIG_PATH,
    TARGET_SCORES,
    AugmentationTask,
    SolarTargetAugmentationError,
    build_tasks,
    editor_output_schema,
    file_sha256,
    load_train_rows,
    make_task,
    parse_editor_output,
    parse_fidelity_output,
    parse_verifier_output,
    prompt_config,
    render_editor_messages,
    render_fidelity_messages,
    render_verifier_messages,
    select_smoke_sources,
    validate_candidate,
)


RUNTIME_MODEL = Path("/dataset/large-models/nota-ai/Solar-Open2-250B-Nota-INT4")
DOCKER_IMAGE = "upstage/vllm-solar-open2:latest"
DOCKER_IMAGE_ID = "sha256:b34b7dd42d986bafb8dcfb285758de2fbfd55e7cf28a1b5ba6b858e6b2b7c8c3"
DOCKER_REPO_DIGEST = "upstage/vllm-solar-open2@sha256:b34b7dd42d986bafb8dcfb285758de2fbfd55e7cf28a1b5ba6b858e6b2b7c8c3"
CONTAINER_MODEL = "/models/Solar-Open2-250B-Nota-INT4"
MODEL_ALIAS = "solar-open2-250b-nota-int4"
OUTPUT_ROOT = ROOT / "outputs/solar-axis-target-v2"
RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_axis_target_v2"
VLLM_CACHE_DIR = OUTPUT_ROOT / "_vllm_cache"
GPU_SCOPE = (0, 1, 2, 3)
MAX_INFLIGHT = 64
RETRIES = 4
MODEL_BINDINGS = {
    "runtime_config_sha256": "039c9fe98844aa026aba4260692c1869a3bd2eae385d06f865714b816928a7b5",
    "runtime_weight_index_sha256": "255b0cb9e82b5f564290bdd1c52734e2f9809d74ee80b056fdf3e3c601df1ae7",
    "runtime_model_code_sha256": "b6ea8bfbbf66588ec47e6b7fa683a7ca75c328546c331a7a51015f7bb0563ed1",
    "runtime_config_code_sha256": "a71b8084bc2db40c01cc38edbba045177aba4f2f840121c198a8eb5fdb7e3279",
    "runtime_chat_template_sha256": "111eec19d6dd69146a4f29a084ea50a356aa907e83029e0d8d1c9dec883679c0",
}
EXPECTED_WEIGHT_BYTES = 142_924_057_368


class SolarTargetRunError(RuntimeError):
    """Raised when the executable augmentation protocol is violated."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SolarTargetRunError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SolarTargetRunError(f"invalid JSON artifact: {path}") from exc
    need(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def validate_execution_gate(mode: str, config: Mapping[str, Any]) -> None:
    """Fail closed before full generation when the scientific protocol is unresolved."""
    need(mode in {"smoke", "full"}, "execution mode differs")
    if mode == "full":
        gate = config.get("execution_gate")
        need(isinstance(gate, Mapping) and gate.get("full_run_authorized") is True,
             "full run is not scientifically authorized")


def verify_model() -> dict[str, Any]:
    required = {
        "runtime_config_sha256": RUNTIME_MODEL / "config.json",
        "runtime_weight_index_sha256": RUNTIME_MODEL / "model.safetensors.index.json",
        "runtime_model_code_sha256": RUNTIME_MODEL / "modeling_solar_open2.py",
        "runtime_config_code_sha256": RUNTIME_MODEL / "configuration_solar_open2.py",
        "runtime_chat_template_sha256": RUNTIME_MODEL / "chat_template.jinja",
    }
    for key, path in required.items():
        need(path.is_file() and not path.is_symlink(), f"Solar model binding unavailable: {key}")
        need(file_sha256(path) == MODEL_BINDINGS[key], f"Solar model binding differs: {key}")
    shards = sorted(RUNTIME_MODEL.glob("model-*.safetensors"))
    need(len(shards) == 27 and all(path.is_file() and not path.is_symlink() for path in shards),
         "Solar shard inventory differs")
    weight_bytes = sum(path.stat().st_size for path in shards)
    need(weight_bytes == EXPECTED_WEIGHT_BYTES, "Solar weight byte count differs")
    return {**MODEL_BINDINGS, "runtime_shards": 27, "runtime_weight_bytes": weight_bytes}


def docker_image_binding() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            ["docker", "image", "inspect", DOCKER_IMAGE], text=True, stderr=subprocess.DEVNULL
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SolarTargetRunError(f"official Solar image is not local: {DOCKER_IMAGE}") from exc
    value = json.loads(raw)
    need(isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict),
         "Docker image inspection differs")
    item = value[0]
    digests = item.get("RepoDigests")
    need(item.get("Id") == DOCKER_IMAGE_ID, "Solar Docker image ID differs")
    need(isinstance(digests, list) and DOCKER_REPO_DIGEST in digests, "Solar Docker digest differs")
    return {
        "docker_image": DOCKER_IMAGE,
        "docker_image_id": DOCKER_IMAGE_ID,
        "docker_repo_digest": DOCKER_REPO_DIGEST,
        "docker_image_size_bytes": item.get("Size"),
    }


def server_command(port: int, container_name: str | None = None) -> list[str]:
    name = container_name or f"mal2026-solar-target-{port}"
    return [
        "docker", "run", "--rm", "--name", name, "--gpus", '"device=0,1,2,3"',
        "--ipc=host", "--network=host",
        "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
        "--env", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0",
        "--mount", f"type=bind,src={VLLM_CACHE_DIR},dst=/root/.cache/vllm",
        "--mount", f"type=bind,src={RUNTIME_MODEL},dst={CONTAINER_MODEL},readonly",
        DOCKER_IMAGE, CONTAINER_MODEL,
        "--served-model-name", MODEL_ALIAS,
        "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", "4",
        "--trust-remote-code",
        "--enable-expert-parallel",
        "--moe-backend", "triton",
        "--default-chat-template-kwargs", '{"think_render_option":"preserved"}',
        "--reasoning-parser", "solar_open2",
        "--logits-processors",
        "vllm.v1.sample.logits_processor.solar_open2:SolarOpen2TemplateLogitsProcessor",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "64",
        "--max-num-batched-tokens", "32768",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
    ]


def http_json(url: str, payload: Mapping[str, Any] | None = None, timeout: int = 900) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    need(isinstance(value, dict), "Solar HTTP response is not an object")
    return value


def wait_server(process: subprocess.Popen[str] | None, endpoint: str, seconds: int = 2400) -> None:
    deadline = time.monotonic() + seconds
    last = "not_ready"
    while time.monotonic() < deadline:
        if process is not None:
            need(process.poll() is None, f"Solar server exited during startup: {process.returncode}")
        try:
            models = http_json(endpoint + "/v1/models", timeout=5)
            data = models.get("data")
            if isinstance(data, list) and any(
                isinstance(item, dict) and item.get("id") == MODEL_ALIAS for item in data
            ):
                return
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, SolarTargetRunError) as exc:
            last = type(exc).__name__
        time.sleep(2)
    raise SolarTargetRunError(f"Solar server readiness timed out: {last}")


def external_server_binding(container_name: str, port: int) -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            ["docker", "container", "inspect", container_name], text=True, stderr=subprocess.DEVNULL
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SolarTargetRunError("declared external Solar container is unavailable") from exc
    values = json.loads(raw)
    need(isinstance(values, list) and len(values) == 1 and isinstance(values[0], dict),
         "external Solar container inspection differs")
    item = values[0]
    state = item.get("State", {})
    config = item.get("Config", {})
    need(state.get("Running") is True, "external Solar container is not running")
    need(item.get("Image") == DOCKER_IMAGE_ID and config.get("Image") == DOCKER_IMAGE,
         "external Solar container image differs")
    mounts = item.get("Mounts")
    need(isinstance(mounts, list) and any(
        mount.get("Source") == str(RUNTIME_MODEL) and
        mount.get("Destination") == CONTAINER_MODEL and
        mount.get("RW") is False
        for mount in mounts if isinstance(mount, dict)
    ), "external Solar model mount differs")
    need(any(
        mount.get("Source") == str(VLLM_CACHE_DIR) and
        mount.get("Destination") == "/root/.cache/vllm" and
        mount.get("RW") is True
        for mount in mounts if isinstance(mount, dict)
    ), "external Solar cache mount differs")
    expected_command = server_command(port, container_name)
    expected_container_command = expected_command[expected_command.index(DOCKER_IMAGE) + 1:]
    need(config.get("Cmd") == expected_container_command,
         "external Solar container command differs")
    host_config = item.get("HostConfig", {})
    need(host_config.get("NetworkMode") == "host" and
         host_config.get("IpcMode") == "host" and
         host_config.get("AutoRemove") is True,
         "external Solar host isolation settings differ")
    environment = config.get("Env")
    required_environment = {
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0",
    }
    need(isinstance(environment, list) and required_environment.issubset(environment),
         "external Solar offline environment differs")
    device_requests = host_config.get("DeviceRequests")
    need(isinstance(device_requests, list) and len(device_requests) == 1 and
         device_requests[0].get("DeviceIDs") == ["0", "1", "2", "3"] and
         device_requests[0].get("Capabilities") == [["gpu"]],
         "external Solar GPU device request differs")
    return {
        "container_name": container_name,
        "container_id": item.get("Id"),
        "container_started_at": state.get("StartedAt"),
        "container_image_id": item.get("Image"),
        "network_mode": host_config.get("NetworkMode"),
        "ipc_mode": host_config.get("IpcMode"),
        "offline_environment": sorted(required_environment),
        "cache_mount": str(VLLM_CACHE_DIR),
    }


def verifier_output_schema() -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["score", "rationale"],
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "rationale": {"type": "string", "minLength": 1},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(AXES),
        "properties": {axis: item for axis in AXES},
    }


def fidelity_output_schema() -> dict[str, Any]:
    names = ["source_based", "topic", "stance", "genre", "new_external_facts_added"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": names,
        "properties": {name: {"type": "boolean"} for name in names},
    }


def stable_seed(task: AugmentationTask, stage: str, attempt: int) -> int:
    config_seed = int(prompt_config()["generation"]["seed"])
    digest = sha256(f"{task.task_id}\0{stage}\0{attempt}".encode("utf-8")).digest()
    return (config_seed + int.from_bytes(digest[:4], "big")) % 2_147_483_647


def stable_blind_seed(stage: str, messages: Sequence[Mapping[str, str]]) -> int:
    """Bind blind evaluator randomness only to its visible request content."""
    need(stage in {"verifier", "fidelity"}, "blind seed stage differs")
    config_seed = int(prompt_config()["generation"]["seed"])
    visible = json.dumps(list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = sha256(f"{stage}\0{visible}".encode("utf-8")).digest()
    return (config_seed + int.from_bytes(digest[:4], "big")) % 2_147_483_647


def request_content(
    endpoint: str,
    task: AugmentationTask,
    stage: str,
    messages: Sequence[Mapping[str, str]],
    schema: Mapping[str, Any],
    max_tokens: int,
    attempt: int,
) -> str:
    config = prompt_config()["generation"][stage]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": f"mal2026_solar_{stage}",
            "strict": True,
            "schema": schema,
        },
    }
    payload = {
        "model": MODEL_ALIAS,
        "messages": list(messages),
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": min(max_tokens, int(config["max_tokens"])),
        "seed": (
            stable_seed(task, stage, attempt)
            if stage == "editor" else stable_blind_seed(stage, messages)
        ),
        "response_format": response_format,
        "chat_template_kwargs": {
            "reasoning_effort": config["reasoning_effort"],
            "think_render_option": "preserved",
            "response_format": response_format,
        },
    }
    response = http_json(endpoint + "/v1/chat/completions", payload, timeout=900)
    choices = response.get("choices")
    need(isinstance(choices, list) and len(choices) == 1, f"Solar {stage} choices differ")
    choice = choices[0]
    need(isinstance(choice, dict) and choice.get("finish_reason") != "length",
         f"Solar {stage} output was truncated")
    message = choice.get("message")
    need(isinstance(message, dict) and isinstance(message.get("content"), str),
         f"Solar {stage} content differs")
    return message["content"]


def gate_category(exc: BaseException) -> str:
    message = str(exc)
    if "non-target verifier score" in message:
        return "non_target_score"
    if "target verifier score" in message:
        return "target_score"
    if "source fidelity" in message:
        return "source_fidelity"
    if "external facts" in message:
        return "external_facts"
    if "length ratio" in message or "too short" in message:
        return "length"
    if "exact source copy" in message:
        return "exact_copy"
    if "substantive source change" in message:
        return "near_copy"
    if "evaluation metadata" in message:
        return "metadata_leak"
    if "organization sentence inventory" in message:
        return "organization_inventory"
    if "content paragraph scaffold" in message or "content sentence scaffold" in message or \
            "content lexical scaffold" in message:
        return "content_scaffold"
    if "expression paragraph scaffold" in message or "expression sentence scaffold" in message or \
            "expression lexical scaffold" in message or "expression numeric token" in message:
        return "expression_scaffold"
    if isinstance(exc, (HTTPError, URLError, TimeoutError)):
        return "transport"
    if "editor" in message or "organization plan" in message or \
            "organization sentence order" in message or \
            "organization paragraph break plan" in message or \
            "organization connector plan" in message:
        return "editor_schema"
    if "verifier" in message:
        return "verifier_schema"
    if "fidelity" in message:
        return "fidelity_schema"
    return type(exc).__name__


def rejection(
    task: AugmentationTask,
    attempt: int,
    stage: str,
    category: str,
    essay: str | None = None,
    verifier: Mapping[str, Any] | None = None,
    fidelity: Mapping[str, Any] | None = None,
    raw_output: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "mal2026-solar-target-rejection-v2",
        "task_id": task.task_id,
        "source_id": task.source.identifier,
        "source_essay_sha256": sha256(task.source.essay.encode("utf-8")).hexdigest(),
        "target_axis": task.target_axis,
        "target_score": task.target_score,
        "attempt": attempt,
        "stage": stage,
        "category": category,
    }
    if essay is not None:
        value["augmented_essay"] = essay
    if verifier is not None:
        value["blind_verifier"] = verifier
    if fidelity is not None:
        value["blind_fidelity"] = fidelity
    if raw_output is not None:
        value["raw_output"] = raw_output
    return value


def label_scores(task: AugmentationTask) -> dict[str, float | int]:
    values: dict[str, float | int] = dict(task.source.baseline)
    values[task.target_axis] = task.target_score
    return values


def generate_task(
    endpoint: str,
    task: AugmentationTask,
    source_verifier: Mapping[str, Mapping[str, Any]],
    retries: int = RETRIES,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    for attempt in range(1, retries + 1):
        raw_editor: str | None = None
        raw_verifier: str | None = None
        raw_fidelity: str | None = None
        essay: str | None = None
        verifier: dict[str, dict[str, Any]] | None = None
        fidelity: dict[str, bool] | None = None
        try:
            raw_editor = request_content(
                endpoint, task, "editor", render_editor_messages(task, attempt - 1),
                editor_output_schema(task), int(prompt_config()["generation"]["editor"]["max_tokens"]), attempt,
            )
            essay = parse_editor_output(
                raw_editor, task.source, task.target_axis, task.target_score
            )
        except Exception as exc:
            category = gate_category(exc)
            rejected.append(rejection(task, attempt, "editor", category, raw_output=raw_editor))
            continue
        try:
            raw_verifier = request_content(
                endpoint, task, "verifier", render_verifier_messages(task.source.prompt, essay),
                verifier_output_schema(), 1000, attempt,
            )
            verifier = parse_verifier_output(raw_verifier)
        except Exception as exc:
            rejected.append(rejection(task, attempt, "verifier", gate_category(exc), essay=essay,
                                        raw_output=raw_verifier))
            continue
        scores = {axis: verifier[axis]["score"] for axis in AXES}
        source_scores = {axis: source_verifier[axis]["score"] for axis in AXES}
        score_category: str | None = None
        if scores[task.target_axis] != task.target_score:
            score_category = "target_score"
        changed_non_targets = [
            axis for axis in AXES
            if axis != task.target_axis and scores[axis] != source_scores[axis]
        ]
        if score_category is None and changed_non_targets:
            score_category = "non_target_score"
        if score_category is not None:
            rejected.append(rejection(
                task, attempt, "quality_gate", score_category, essay=essay, verifier=verifier
            ))
            continue
        try:
            raw_fidelity = request_content(
                endpoint, task, "fidelity",
                render_fidelity_messages(task.source.essay, essay), fidelity_output_schema(), 400, attempt,
            )
            fidelity = parse_fidelity_output(raw_fidelity)
        except Exception as exc:
            rejected.append(rejection(task, attempt, "fidelity", gate_category(exc), essay=essay,
                                        verifier=verifier, raw_output=raw_fidelity))
            continue
        try:
            validate_candidate(task, essay, verifier, source_verifier, fidelity)
        except SolarTargetAugmentationError as exc:
            category = gate_category(exc)
            rejected.append(rejection(task, attempt, "quality_gate", category, essay, verifier, fidelity))
            continue
        source_target_score = source_verifier[task.target_axis]["score"]
        movement_delta = task.target_score - source_target_score
        record = {
            "schema_version": "mal2026-solar-axis-target-record-v2",
            "source_id": task.source.identifier,
            "source_document_id": task.source.document_id,
            "source_essay_sha256": sha256(task.source.essay.encode("utf-8")).hexdigest(),
            "augmented_id": task.task_id,
            "target_axis": task.target_axis,
            "target_score": task.target_score,
            "source_blind_target_score": source_target_score,
            "movement": "same" if movement_delta == 0 else ("up" if movement_delta > 0 else "down"),
            "movement_magnitude": abs(movement_delta),
            "operation_family_index": attempt - 1,
            "prompt": task.source.prompt,
            "essay": essay,
            "editor_output": json.loads(raw_editor),
            "source_baseline_score": task.source.baseline,
            "score": label_scores(task),
            "score_provenance": {
                "target_axis": "requested_integer_target",
                "non_target_axes": "canonical_train_gold",
                "preservation_gate": "target_blind_solar_source_rescore_integer_equality",
            },
            "blind_verifier": verifier,
            "blind_source_verifier": source_verifier,
            "blind_fidelity": fidelity,
            "attempts": attempt,
        }
        return record, rejected
    return None, rejected


def score_source_rows(
    endpoint: str,
    source_rows: Sequence[Any],
    restricted: Path,
    max_inflight: int,
) -> tuple[dict[str, dict[str, dict[str, Any]]], Path]:
    """Score each immutable source once with the same target-blind verifier."""
    path = restricted / "blind_source_scores.jsonl"

    def one(source: Any) -> tuple[str, dict[str, Any]]:
        task = make_task(source, "content", 3)
        raw = request_content(
            endpoint,
            task,
            "verifier",
            render_verifier_messages(source.prompt, source.essay),
            verifier_output_schema(),
            1000,
            0,
        )
        parsed = parse_verifier_output(raw)
        return source.identifier, {
            "schema_version": "mal2026-solar-blind-source-score-v1",
            "source_id": source.identifier,
            "source_essay_sha256": sha256(source.essay.encode("utf-8")).hexdigest(),
            "blind_source_verifier": parsed,
        }

    values: list[tuple[str, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=min(max_inflight, len(source_rows))) as pool:
        futures = [pool.submit(one, source) for source in source_rows]
        for future in as_completed(futures):
            values.append(future.result())
    need(len(values) == len(source_rows) and len({key for key, _ in values}) == len(source_rows),
         "blind source score population differs")
    values.sort(key=lambda item: item[0])
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        for _, value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {key: value["blind_source_verifier"] for key, value in values}, path


def validate_smoke_approval(result_path: Path, review_path: Path, bindings: Mapping[str, Any]) -> dict[str, Any]:
    result = read_json(result_path)
    review = read_json(review_path)
    need(result.get("schema_version") == "mal2026-solar-axis-target-result-v2", "smoke result schema differs")
    need(result.get("status") == "completed" and result.get("mode") == "smoke", "smoke is incomplete")
    need(result.get("source_records") == 5 and result.get("blind_source_records") == 5,
         "smoke source population differs")
    need(result.get("records") == 75 and result.get("records_expected") == 75 and
         result.get("variants_per_source") == 15,
         "smoke population did not fully pass")
    need(result.get("axis_counts") == {axis: 25 for axis in AXES},
         "smoke axis population differs")
    expected_targets = {axis: {str(score): 5 for score in TARGET_SCORES} for axis in AXES}
    need(result.get("target_score_counts") == expected_targets,
         "smoke axis-target matrix differs")
    need(result.get("failures") == {}, "smoke has terminal failures")
    need(result.get("binding_sha256") == canonical_sha(bindings), "smoke bindings differ")
    augmented_path_value = result.get("augmented_train_path")
    need(isinstance(augmented_path_value, str), "smoke augmented artifact path differs")
    augmented_path = Path(augmented_path_value)
    need(augmented_path.is_file() and not augmented_path.is_symlink() and
         augmented_path.resolve().is_relative_to(RESTRICTED_ROOT.resolve()),
         "smoke augmented artifact location differs")
    augmented_sha = file_sha256(augmented_path)
    need(result.get("augmented_train_sha256") == augmented_sha,
         "smoke augmented artifact digest differs")
    rows = [json.loads(line) for line in augmented_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    need(len(rows) == 75 and len({row.get("augmented_id") for row in rows}) == 75,
         "smoke augmented artifact row population differs")
    sources = {row.get("source_id") for row in rows}
    need(len(sources) == 5 and all(
        sum(row.get("source_id") == source for row in rows) == 15 for source in sources
    ), "smoke augmented artifact source matrix differs")
    need(all(sum(row.get("target_axis") == axis and row.get("target_score") == score
                     for row in rows) == 5
             for axis in AXES for score in TARGET_SCORES),
         "smoke augmented artifact axis-target matrix differs")
    need(review.get("schema_version") == "mal2026-solar-smoke-review-v1", "smoke review schema differs")
    need(review.get("status") == "approved" and review.get("all_records_reviewed") is True,
         "smoke review is not approved")
    need(review.get("result_sha256") == file_sha256(result_path), "smoke review result binding differs")
    need(review.get("augmented_train_sha256") == augmented_sha and
         review.get("reviewed_record_count") == 75 and
         review.get("reviewed_cell_counts") == expected_targets and
         review.get("unresolved_findings") == 0,
         "smoke review coverage differs")
    reviewers = review.get("reviewers")
    need(isinstance(reviewers, list) and "lead" in reviewers and len(reviewers) >= 2,
         "smoke review lacks lead and subagent evidence")
    return {
        "smoke_result_path": str(result_path.resolve()),
        "smoke_result_sha256": file_sha256(result_path),
        "smoke_review_path": str(review_path.resolve()),
        "smoke_review_sha256": file_sha256(review_path),
        "smoke_augmented_train_sha256": augmented_sha,
    }


def run_population(
    endpoint: str,
    tasks: Sequence[AugmentationTask],
    restricted: Path,
    max_inflight: int,
    retries: int,
    source_verifiers: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], Counter[str], Counter[int], int]:
    accepted_path = restricted / "accepted.partial.jsonl"
    rejected_path = restricted / "rejected.partial.jsonl"
    records: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    attempts: Counter[int] = Counter()
    rejected_attempts = 0
    lock = threading.Lock()

    # Run one randomly selected contract synchronously as the smallest real
    # output diagnostic, then preserve a complete 75-cell smoke matrix even if
    # that first contract fails so prompt iteration has full failure evidence.
    first, first_rejections = generate_task(
        endpoint, tasks[0], source_verifiers[tasks[0].source.identifier], retries
    )
    with rejected_path.open("x", encoding="utf-8") as rejected_handle:
        os.chmod(rejected_path, 0o600)
        for item in first_rejections:
            rejected_handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            rejected_attempts += 1
        rejected_handle.flush()
        with accepted_path.open("x", encoding="utf-8") as accepted_handle:
            os.chmod(accepted_path, 0o600)
            if first is None:
                category = first_rejections[-1]["category"] if first_rejections else "unknown"
                failures[category] += 1
            else:
                accepted_handle.write(json.dumps(first, ensure_ascii=False, separators=(",", ":")) + "\n")
                accepted_handle.flush()
                records.append(first)
                attempts[first["attempts"]] += 1

            def work(task: AugmentationTask) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
                return generate_task(
                    endpoint, task, source_verifiers[task.source.identifier], retries
                )

            with ThreadPoolExecutor(max_workers=max_inflight) as pool:
                futures = {pool.submit(work, task): task for task in tasks[1:]}
                for index, future in enumerate(as_completed(futures), 1):
                    task = futures[future]
                    try:
                        record, rejected = future.result()
                    except Exception as exc:
                        record, rejected = None, [rejection(task, retries, "worker", gate_category(exc))]
                    with lock:
                        for item in rejected:
                            rejected_handle.write(
                                json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                            )
                            rejected_attempts += 1
                        if record is None:
                            category = rejected[-1]["category"] if rejected else "unknown"
                            failures[category] += 1
                        else:
                            accepted_handle.write(
                                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                            )
                            records.append(record)
                            attempts[record["attempts"]] += 1
                        if index % 25 == 0:
                            accepted_handle.flush()
                            rejected_handle.flush()
    return records, failures, attempts, rejected_attempts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--port", type=int, default=19420)
    parser.add_argument("--max-inflight", type=int, default=MAX_INFLIGHT)
    parser.add_argument("--retries", type=int, default=RETRIES)
    parser.add_argument("--smoke-result", type=Path)
    parser.add_argument("--smoke-review", type=Path)
    parser.add_argument("--external-endpoint")
    parser.add_argument("--external-container-name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    need(re.fullmatch(r"[a-z0-9][a-z0-9-]{7,95}", args.run_id) is not None, "run ID differs")
    need(args.max_inflight == MAX_INFLIGHT and args.retries == RETRIES, "concurrency/retry protocol differs")
    need(1024 <= args.port <= 65535, "port differs")
    if args.external_endpoint is None:
        need(args.external_container_name is None, "external container without endpoint")
    else:
        need(args.external_endpoint == f"http://127.0.0.1:{args.port}", "external endpoint differs")
        need(isinstance(args.external_container_name, str) and bool(args.external_container_name),
             "external container name is required")

    config = prompt_config()
    validate_execution_gate(args.mode, config)
    model_binding = verify_model()
    image_binding = docker_image_binding()
    rows = load_train_rows()
    if args.mode == "smoke":
        need(args.smoke_result is None and args.smoke_review is None, "smoke approval inputs are unexpected")
        source_rows = select_smoke_sources(rows, count=int(config["smoke"]["minimum_sources"]))
        tasks = [
            task for source in source_rows
            for task in (
                AugmentationTask(f"{source.identifier}::solar-target::{axis}::{score}", source, axis, score)
                for axis in AXES for score in TARGET_SCORES
            )
        ]
    else:
        need(args.smoke_result is not None and args.smoke_review is not None,
             "full mode requires smoke result and review")
        source_rows = rows
        tasks = build_tasks(rows)

    bindings = {
        "prompt_config_sha256": file_sha256(CONFIG_PATH),
        "evaluation_sha256": config["provenance"]["rubric_source_sha256"],
        "train_sha256": config["provenance"]["train_source_sha256"],
        "validation_sha256": config["provenance"]["validation_source_sha256"],
        "model": model_binding,
        "image": image_binding,
        "implementation": {
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "core_sha256": file_sha256(ROOT / "src/mal2026/solar_target_augmentation.py"),
        },
    }
    smoke_approval: dict[str, Any] | None = None
    if args.mode == "full":
        smoke_approval = validate_smoke_approval(args.smoke_result, args.smoke_review, bindings)

    output = OUTPUT_ROOT / args.run_id
    restricted = RESTRICTED_ROOT / args.run_id
    need(not output.exists() and not restricted.exists(), "run outputs must be fresh")
    output.mkdir(mode=0o700, parents=True)
    restricted.mkdir(mode=0o700, parents=True)
    os.chmod(restricted, 0o700)
    VLLM_CACHE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    command = server_command(args.port, args.external_container_name)
    manifest_path = output / "manifest.json"
    log_path = output / "vllm-server.log"
    manifest: dict[str, Any] = {
        "schema_version": "mal2026-solar-axis-target-run-v2",
        "status": "preflight",
        "mode": args.mode,
        "run_id": args.run_id,
        "created_at": now(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree_dirty_at_launch": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
        "command": [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *sys.argv[1:]],
        "server_command": command,
        "server_mode": "external" if args.external_endpoint else "managed",
        "gpu_scope": list(GPU_SCOPE),
        "gpu_authorization": "repository default GPUs0-3 plus explicit user Solar augmentation authorization; GPUs4-7 not queried or used",
        "source_split": "train_only",
        "validation_used_for_generation_or_selection": False,
        "average_read_emitted_or_used": False,
        "source_records": len(source_rows),
        "records_expected": len(tasks),
        "variants_per_source": 15,
        "bindings": bindings,
        "binding_sha256": canonical_sha(bindings),
        "smoke_approval": smoke_approval,
    }
    atomic_json(manifest_path, manifest)

    process: subprocess.Popen[str] | None = None
    log_handle = None
    endpoint = args.external_endpoint or f"http://127.0.0.1:{args.port}"
    try:
        if args.external_endpoint:
            manifest["external_server"] = external_server_binding(args.external_container_name, args.port)
            manifest.update({"status": "server_connecting", "server_connected_at": now()})
            atomic_json(manifest_path, manifest)
            wait_server(None, endpoint, seconds=60)
        else:
            assert_gpus_idle(GPU_SCOPE)
            log_handle = log_path.open("x", encoding="utf-8")
            manifest.update({"status": "server_starting", "server_started_at": now()})
            atomic_json(manifest_path, manifest)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                },
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            wait_server(process, endpoint)
        manifest.update({"status": "generating", "server_ready_at": now()})
        atomic_json(manifest_path, manifest)
        source_verifiers, source_score_path = score_source_rows(
            endpoint, source_rows, restricted, args.max_inflight
        )
        manifest.update({
            "blind_source_scores": len(source_verifiers),
            "blind_source_scores_sha256": file_sha256(source_score_path),
        })
        atomic_json(manifest_path, manifest)
        records, failures, attempts, rejected_attempts = run_population(
            endpoint, tasks, restricted, args.max_inflight, args.retries, source_verifiers
        )
        if failures or len(records) != len(tasks):
            failed_result = {
                "schema_version": "mal2026-solar-axis-target-result-v2",
                "status": "failed",
                "mode": args.mode,
                "run_id": args.run_id,
                "failed_at": now(),
                "source_records": len(source_rows),
                "records": len(records),
                "records_expected": len(tasks),
                "attempt_histogram": {str(key): attempts[key] for key in range(1, args.retries + 1)},
                "rejected_attempts": rejected_attempts,
                "blind_source_records": len(source_verifiers),
                "failures": dict(failures),
                "accepted_partial_records": len(records),
                "accepted_partial_sha256": file_sha256(restricted / "accepted.partial.jsonl"),
                "rejected_partial_records": rejected_attempts,
                "rejected_partial_sha256": file_sha256(restricted / "rejected.partial.jsonl"),
                "binding_sha256": canonical_sha(bindings),
                "privacy": "aggregate failure result contains no essay, prompt, rationale, identifier, prediction row, or individual score",
            }
            atomic_json(output / "result.failed.json", failed_result)
        need(not failures and len(records) == len(tasks),
             f"Solar target generation gates failed: valid={len(records)} failures={dict(failures)}")
        records.sort(key=lambda item: (
            item["source_id"], AXES.index(item["target_axis"]), int(item["target_score"])
        ))
        expected_ids = {task.task_id for task in tasks}
        need(len({record["augmented_id"] for record in records}) == len(tasks), "accepted IDs are not unique")
        need({record["augmented_id"] for record in records} == expected_ids, "accepted population differs")
        final_path = restricted / "augmented.train.jsonl"
        with final_path.open("x", encoding="utf-8") as handle:
            os.chmod(final_path, 0o600)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

        axis_counts = {axis: sum(record["target_axis"] == axis for record in records) for axis in AXES}
        target_counts = {
            axis: {
                str(score): sum(
                    record["target_axis"] == axis and record["target_score"] == score for record in records
                ) for score in TARGET_SCORES
            } for axis in AXES
        }
        result = {
            "schema_version": "mal2026-solar-axis-target-result-v2",
            "status": "completed",
            "mode": args.mode,
            "run_id": args.run_id,
            "completed_at": now(),
            "source_records": len(source_rows),
            "records": len(records),
            "records_expected": len(tasks),
            "variants_per_source": 15,
            "axis_counts": axis_counts,
            "target_score_counts": target_counts,
            "attempt_histogram": {str(key): attempts[key] for key in range(1, args.retries + 1)},
            "rejected_attempts": rejected_attempts,
            "blind_source_records": len(source_verifiers),
            "failures": dict(failures),
            "binding_sha256": canonical_sha(bindings),
            "augmented_train_path": str(final_path.resolve()),
            "augmented_train_sha256": file_sha256(final_path),
            "privacy": "aggregate result contains no essay, prompt, rationale, identifier, prediction row, or individual score",
        }
        result_path = output / "result.json"
        atomic_json(result_path, result)
        manifest.update({"status": "completed", "completed_at": now(), "result_sha256": file_sha256(result_path)})
        atomic_json(manifest_path, manifest)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at": now(), "failure_category": gate_category(exc)})
        atomic_json(manifest_path, manifest)
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    main()
