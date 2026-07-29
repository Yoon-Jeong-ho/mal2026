#!/usr/bin/env python3
"""Generate 6,000 train-only axis-degraded essays with local Solar Open 2 INT4."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_rl_servers import assert_gpus_idle  # noqa: E402
from mal2026.solar_axis_augmentation import (  # noqa: E402
    AXES,
    CONFIG_PATH,
    SourceRow,
    file_sha256,
    load_train_rows,
    output_schema,
    parse_output,
    prompt_config,
    render_messages,
    task_count,
)


RUN_ID = "solar-open2-axis-degradation-train-v1-20260729-004"
RUNTIME_MODEL = Path("/dataset/large-models/nota-ai/Solar-Open2-250B-Nota-INT4")
BASE_MODEL = Path("/dataset/large-models/upstage/Solar-Open2-250B")
DOCKER_IMAGE = "upstage/vllm-solar-open2:latest"
CONTAINER_MODEL = "/models/Solar-Open2-250B-Nota-INT4"
OUTPUT_ROOT = ROOT / "outputs/solar-axis-degradation-v1"
RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_axis_degradation_v1"
QWEN_RESULT = ROOT / "outputs/rationale-aware-encoder-v1/rationale-aware-qwen3-embedding-8b-aihub-mal-v1-20260729-002/result.json"
KURE_RESULT = ROOT / "outputs/rationale-aware-encoder-v1/rationale-aware-kure-v1-aihub-mal-v1-20260729-002/result.json"
MODEL_ALIAS = "solar-open2-250b-nota-int4"
MODEL_BINDINGS = {
    "base_config_sha256": "fb6428ba165af1ace1d98f9170f6bafce061347593a94bd16b4b8aa3d6fe09f9",
    "runtime_config_sha256": "039c9fe98844aa026aba4260692c1869a3bd2eae385d06f865714b816928a7b5",
    "runtime_weight_index_sha256": "255b0cb9e82b5f564290bdd1c52734e2f9809d74ee80b056fdf3e3c601df1ae7",
    "runtime_model_code_sha256": "b6ea8bfbbf66588ec47e6b7fa683a7ca75c328546c331a7a51015f7bb0563ed1",
    "runtime_config_code_sha256": "a71b8084bc2db40c01cc38edbba045177aba4f2f840121c198a8eb5fdb7e3279",
    "runtime_chat_template_sha256": "111eec19d6dd69146a4f29a084ea50a356aa907e83029e0d8d1c9dec883679c0",
}


class SolarRunError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SolarRunError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_sha(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"JSON object differs: {path}")
    return value


def check_gate() -> dict[str, Any]:
    need(QWEN_RESULT.is_file() and KURE_RESULT.is_file(), "both rationale-aware encoder results are required")
    values: dict[str, Any] = {}
    for name, path in (("qwen3_embedding_8b", QWEN_RESULT), ("kure_v1", KURE_RESULT)):
        result = read_json(path)
        metric = result.get("canonical_validation", {}).get("aligned_bundle_metrics", {}).get("macro_continuous_rmse")
        need(type(metric) in {int, float} and float(metric) > 0.5, "Solar gate did not trigger")
        values[name] = {"result_sha256": file_sha256(path), "macro_continuous_rmse": float(metric)}
    return values


def verify_model() -> dict[str, Any]:
    required = {
        "base_config_sha256": BASE_MODEL / "config.json",
        "runtime_config_sha256": RUNTIME_MODEL / "config.json",
        "runtime_weight_index_sha256": RUNTIME_MODEL / "model.safetensors.index.json",
        "runtime_model_code_sha256": RUNTIME_MODEL / "modeling_solar_open2.py",
        "runtime_config_code_sha256": RUNTIME_MODEL / "configuration_solar_open2.py",
        "runtime_chat_template_sha256": RUNTIME_MODEL / "chat_template.jinja",
    }
    for key, path in required.items():
        need(path.is_file() and not path.is_symlink() and file_sha256(path) == MODEL_BINDINGS[key], f"Solar binding differs: {key}")
    shards = sorted(RUNTIME_MODEL.glob("model-*.safetensors"))
    need(len(shards) == 27 and all(path.is_file() and not path.is_symlink() for path in shards), "Solar shard inventory differs")
    return {**MODEL_BINDINGS, "runtime_shards": len(shards), "runtime_weight_bytes": sum(path.stat().st_size for path in shards)}


def docker_image_binding() -> dict[str, str]:
    try:
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", DOCKER_IMAGE],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SolarRunError(
            f"approved official Solar Docker image is not local: {DOCKER_IMAGE}"
        ) from exc
    need(image_id.startswith("sha256:") and len(image_id) == 71, "Solar Docker image ID differs")
    return {"docker_image": DOCKER_IMAGE, "docker_image_id": image_id}


def server_command(port: int) -> list[str]:
    return [
        "docker", "run", "--rm", "--gpus", '"device=0,1,2,3"', "--ipc=host", "--network=host",
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
        "--logits-processors", "vllm.v1.sample.logits_processor.solar_open2:SolarOpen2TemplateLogitsProcessor",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "64",
        "--max-num-batched-tokens", "32768",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
    ]


def http_json(url: str, payload: Mapping[str, Any] | None = None, timeout: int = 600) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="GET" if data is None else "POST")
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    need(isinstance(value, dict), "Solar HTTP response differs")
    return value


def wait_server(process: subprocess.Popen[str], endpoint: str, seconds: int = 1800) -> None:
    deadline = time.monotonic() + seconds
    last = "not ready"
    while time.monotonic() < deadline:
        code = process.poll()
        need(code is None, f"Solar server exited during startup: {code}")
        try:
            models = http_json(endpoint + "/v1/models", timeout=5)
            data = models.get("data")
            if isinstance(data, list) and any(item.get("id") == MODEL_ALIAS for item in data if isinstance(item, dict)):
                return
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, SolarRunError) as exc:
            last = type(exc).__name__
        time.sleep(2)
    raise SolarRunError(f"Solar server readiness timed out: {last}")


def request_one(endpoint: str, row: SourceRow, axis: str, attempt: int) -> dict[str, Any]:
    config = prompt_config()
    payload = {
        "model": MODEL_ALIAS,
        "messages": render_messages(row, axis),
        "temperature": config["generation"]["temperature"],
        "top_p": config["generation"]["top_p"],
        "max_tokens": config["generation"]["max_tokens"],
        "seed": config["generation"]["seed"] + attempt,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "mal2026_axis_degradation", "strict": True, "schema": output_schema()},
        },
        "chat_template_kwargs": {"reasoning_effort": "none", "think_render_option": "preserved"},
    }
    response = http_json(endpoint + "/v1/chat/completions", payload, timeout=900)
    choices = response.get("choices")
    need(isinstance(choices, list) and len(choices) == 1, "Solar choices differ")
    message = choices[0].get("message")
    need(isinstance(message, dict) and isinstance(message.get("content"), str), "Solar content differs")
    parsed = parse_output(message["content"], row, axis)
    return {
        "source_id": row.identifier,
        "target_axis": axis,
        "augmented_id": f"{row.identifier}::solar-degrade::{axis}",
        "prompt": row.prompt,
        "essay": parsed["augmented_essay"],
        "score": parsed["score"],
        "attempts": attempt,
    }


def generate_with_retries(endpoint: str, row: SourceRow, axis: str, retries: int) -> tuple[dict[str, Any] | None, str | None]:
    last: str | None = None
    for attempt in range(1, retries + 1):
        try:
            return request_one(endpoint, row, axis, attempt), None
        except Exception as exc:  # preserve only aggregate failure categories
            last = type(exc).__name__
            time.sleep(min(attempt, 3))
    return None, last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--port", type=int, default=19420)
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    need(args.run_id == RUN_ID, "Solar run identity differs")
    need(args.max_inflight == 64 and args.retries == 3, "Solar concurrency/retry protocol differs")

    gate = check_gate()
    model_binding = verify_model()
    model_binding.update(docker_image_binding())
    rows = load_train_rows()
    need(task_count(rows) == 6000, "Solar task population differs")
    output = OUTPUT_ROOT / args.run_id
    restricted = RESTRICTED_ROOT / args.run_id
    need(not output.exists() and not restricted.exists(), "Solar outputs must be fresh")
    output.mkdir(mode=0o700, parents=True)
    restricted.mkdir(mode=0o700, parents=True)
    log_path = output / "vllm-server.log"
    manifest_path = output / "manifest.json"
    command = server_command(args.port)
    manifest: dict[str, Any] = {
        "schema_version": "mal2026-solar-axis-degradation-run-v1",
        "status": "running", "run_id": args.run_id, "created_at": now(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *sys.argv[1:]],
        "server_command": command,
        "gpu_scope": [0, 1, 2, 3],
        "gpu_authorization": "repository default GPUs0-3 and explicit user request; GPUs4-7 not queried or used",
        "trigger_gate": gate,
        "model_binding": model_binding,
        "prompt_config_sha256": file_sha256(CONFIG_PATH),
        "records_expected": 6000,
        "source_split": "train_only",
        "validation_used_for_generation_or_selection": False,
        "average_read_or_used": False,
    }
    atomic_json(manifest_path, manifest)

    assert_gpus_idle((0, 1, 2, 3))
    endpoint = f"http://127.0.0.1:{args.port}"
    process: subprocess.Popen[str] | None = None
    log_handle = log_path.open("x", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
            stdout=log_handle, stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        wait_server(process, endpoint)
        tasks = [(row, axis) for row in rows for axis in AXES]
        records: list[dict[str, Any]] = []
        failures: dict[str, int] = {}

        # Smallest real gate on the same loaded TP4 server; pass continues directly.
        smoke, failure = generate_with_retries(endpoint, tasks[0][0], tasks[0][1], args.retries)
        need(smoke is not None and failure is None, f"Solar one-row smoke failed: {failure}")
        records.append(smoke)

        lock = threading.Lock()
        progress_path = restricted / "generated.partial.jsonl"
        with progress_path.open("x", encoding="utf-8") as progress:
            progress.write(json.dumps(smoke, ensure_ascii=False, separators=(",", ":")) + "\n")
            progress.flush()

            def work(item: tuple[SourceRow, str]) -> tuple[dict[str, Any] | None, str | None]:
                return generate_with_retries(endpoint, item[0], item[1], args.retries)

            with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
                futures = [pool.submit(work, task) for task in tasks[1:]]
                for index, future in enumerate(as_completed(futures), 1):
                    record, category = future.result()
                    if record is None:
                        failures[category or "unknown"] = failures.get(category or "unknown", 0) + 1
                    else:
                        records.append(record)
                        with lock:
                            progress.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if index % 100 == 0:
                        with lock:
                            progress.flush()
        need(not failures and len(records) == 6000, f"Solar generation hard gates failed: valid={len(records)} failures={failures}")
        records.sort(key=lambda item: (item["source_id"], AXES.index(item["target_axis"])))
        final_path = restricted / "augmented.train.jsonl"
        with final_path.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        summary = {
            "schema_version": "mal2026-solar-axis-degradation-result-v1",
            "status": "completed", "run_id": args.run_id, "completed_at": now(),
            "records": len(records), "source_records": len(rows), "variants_per_source": 3,
            "axis_counts": {axis: sum(record["target_axis"] == axis for record in records) for axis in AXES},
            "attempt_histogram": {str(attempt): sum(record["attempts"] == attempt for record in records) for attempt in range(1, args.retries + 1)},
            "failures": failures,
            "augmented_train_path": str(final_path.resolve()),
            "augmented_train_sha256": file_sha256(final_path),
            "input_contract_sha256": canonical_sha({"train_sha256": "b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737", "prompt_sha256": file_sha256(CONFIG_PATH)}),
            "privacy": "aggregate report contains no essay, rationale, score, prediction, or identifier rows",
        }
        result_path = output / "result.json"
        atomic_json(result_path, summary)
        manifest.update({"status": "completed", "completed_at": now(), "result_sha256": file_sha256(result_path)})
        atomic_json(manifest_path, manifest)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    except BaseException as exc:
        manifest.update({"status": "failed", "failed_at": now(), "failure_category": type(exc).__name__})
        atomic_json(manifest_path, manifest)
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        log_handle.close()


if __name__ == "__main__":
    main()
