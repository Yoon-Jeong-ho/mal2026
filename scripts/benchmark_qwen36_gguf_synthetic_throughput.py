#!/usr/bin/env python3
"""Synthetic-only llama.cpp throughput benchmark with strict JSON/repeat gates.

The runner never opens project datasets or writes prompts/responses.  It uses
one physical GPU selected by CUDA_VISIBLE_DEVICES and stores aggregate-only
provenance, timing, schema, and repeatability evidence in a new run directory.
"""
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
import time
from typing import Any
from urllib.request import Request, urlopen


PARALLEL_LEVELS = (1, 4, 8)
PROMPT = (
    "This is a synthetic systems benchmark. Return only the specified JSON. "
    "No student, candidate, or source data is present."
)
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "status", "slots"],
    "properties": {
        "schema_version": {"type": "string", "const": "mal2026-synthetic-throughput-v1"},
        "status": {"type": "string", "const": "ok"},
        "slots": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "integer", "minimum": 1, "maximum": 3},
        },
    },
}


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def request_payload() -> bytes:
    payload = {
        "model": "pinned-gguf",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 2026071903,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {"name": "synthetic_throughput", "strict": True, "schema": SCHEMA}},
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"schema_version", "status", "slots"}
        and value["schema_version"] == "mal2026-synthetic-throughput-v1"
        and value["status"] == "ok"
        and value["slots"] == [1, 2, 3]
    )


def call(server: str, payload: bytes) -> tuple[str, float]:
    start = time.monotonic()
    request = Request(server + "/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=180) as response:
        raw = json.loads(response.read().decode("utf-8"))
    elapsed = time.monotonic() - start
    content = raw["choices"][0]["message"]["content"]
    value = json.loads(content)
    need(valid(value), "synthetic response violates fixed schema/value gate")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")), elapsed


def await_health(server: str, seconds: int = 180) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(server + "/health", timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("llama-server health endpoint did not become ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--llama-repo", type=Path, required=True)
    parser.add_argument("--llama-revision", required=True)
    parser.add_argument("--llama-tag", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True, choices=(1,))
    parser.add_argument("--port", type=int, default=18091)
    parser.add_argument("--requests-per-level", type=int, default=16)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.physical_gpu), "CUDA_VISIBLE_DEVICES must be exactly the assigned physical GPU")
    need(args.requests_per_level >= 8 and args.requests_per_level % 8 == 0, "requests-per-level must be a multiple of 8")
    need(not args.run_dir.exists(), "benchmark run directory already exists")
    need(args.model.is_file() and args.llama_server.is_file() and os.access(args.llama_server, os.X_OK), "pinned model/server is unavailable")
    need(digest(args.model) == args.model_sha256, "pinned GGUF SHA-256 gate failed")
    revision = subprocess.check_output(["git", "-C", str(args.llama_repo), "rev-parse", "HEAD"], text=True).strip()
    tag = subprocess.check_output(["git", "-C", str(args.llama_repo), "describe", "--tags", "--exact-match"], text=True).strip()
    need(revision == args.llama_revision and tag == args.llama_tag, "pinned llama.cpp revision/tag gate failed")
    args.run_dir.mkdir(parents=True)
    payload = request_payload()
    server = f"http://127.0.0.1:{args.port}"
    aggregate: dict[str, Any] = {"schema_version": "mal2026-gguf-synthetic-throughput-v1", "started_at": now(), "physical_gpu": args.physical_gpu, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "prompt_sha256": sha256(payload).hexdigest(), "model_sha256": args.model_sha256, "llama_revision": revision, "llama_tag": tag, "parallel_levels": {}, "data_access": False, "raw_prompts_or_responses_persisted": False}
    process: subprocess.Popen[bytes] | None = None
    try:
        for parallel in PARALLEL_LEVELS:
            process = subprocess.Popen([str(args.llama_server), "--model", str(args.model), "--host", "127.0.0.1", "--port", str(args.port), "--n-gpu-layers", "99", "--parallel", str(parallel), "--ctx-size", "4096", "--no-webui", "--reasoning", "off"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ.copy())
            await_health(server)
            # One warm-up is deliberately excluded from throughput and repeat gates.
            call(server, payload)
            started = time.monotonic()
            responses: list[tuple[str, float]] = []
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                pending = [pool.submit(call, server, payload) for _ in range(args.requests_per_level)]
                for future in as_completed(pending):
                    responses.append(future.result())
            elapsed = time.monotonic() - started
            canonical = [item[0] for item in responses]
            latencies = [item[1] for item in responses]
            repeatable = len(set(canonical)) == 1
            need(repeatable, f"parallel={parallel} exact-repeatability gate failed")
            aggregate["parallel_levels"][str(parallel)] = {"request_count": len(responses), "schema_valid_count": len(responses), "exact_repeatability": True, "wall_seconds": round(elapsed, 6), "requests_per_second": round(len(responses) / elapsed, 6), "mean_latency_seconds": round(sum(latencies) / len(latencies), 6), "max_latency_seconds": round(max(latencies), 6), "canonical_response_sha256": sha256(canonical[0].encode("utf-8")).hexdigest()}
            process.terminate(); process.wait(timeout=30); process = None
        aggregate["status"] = "passed"
    except Exception as exc:
        aggregate["status"] = "failed"
        aggregate["failure_class"] = type(exc).__name__
        aggregate["failure_message"] = str(exc)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=30)
        aggregate["finished_at"] = now()
        (args.run_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
