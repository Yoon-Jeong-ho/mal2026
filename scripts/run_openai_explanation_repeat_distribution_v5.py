#!/usr/bin/env python3
"""V5 train-only runner: aggregate-only, context-budgeted, and fail closed."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("MAL2026_REPEAT_CONFIG", ROOT / "configs/openai_explanation_repeat_distribution.v5.pilot.json"))
SCHEMA = "mal2026-openai-explanation-repeat-distribution-v5"
BASE_SPEC = importlib.util.spec_from_file_location("repeat_v4_base", ROOT / "scripts/run_openai_explanation_repeat_distribution_v4.py")
assert BASE_SPEC and BASE_SPEC.loader
BASE = importlib.util.module_from_spec(BASE_SPEC); BASE_SPEC.loader.exec_module(BASE)
BASE_PAYLOAD_LAYOUT = BASE.payload_layout
PREFLIGHT_SPEC = importlib.util.spec_from_file_location("repeat_v5_wire", ROOT / "scripts/preflight_openai_repeat_v5_synthetic.py")
assert PREFLIGHT_SPEC and PREFLIGHT_SPEC.loader
WIRE = importlib.util.module_from_spec(PREFLIGHT_SPEC); PREFLIGHT_SPEC.loader.exec_module(WIRE)


def config(sample_essays: int | None = None) -> dict[str, Any]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != SCHEMA or cfg["selection"].get("split") != "train":
        raise RuntimeError("unexpected v5 train-only configuration")
    runtime, request, retry = cfg["runtime"], cfg["request"], cfg["retry"]
    if runtime.get("physical_gpus") != [4, 5, 6, 7] or runtime.get("parallel_requests_per_server") != 1 or runtime.get("context_size") != 4096 or runtime.get("context_safety_margin") != 256:
        raise RuntimeError("v5 requires four GPU4-7 one-slot 4096-token servers")
    if request != {"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 192, "top_p": 1.0} or retry.get("max_attempts") != 2:
        raise RuntimeError("v5 request/retry contract changed")
    if sample_essays is not None:
        if not 1 <= sample_essays <= cfg["selection"]["max_essays"]:
            raise RuntimeError("requested sample size is outside the capped train-only pilot")
        cfg = copy.deepcopy(cfg); cfg["selection"]["max_essays"] = sample_essays
    return cfg


def run_dir(run_id: str) -> Path:
    if not run_id.startswith("openai-repeat-v5-20260720-") or not run_id[-3:].isdigit():
        raise RuntimeError("run id does not bind the versioned v5 lineage")
    return BASE.RESTRICTED / BASE.BATCH / "judge_runs" / run_id


def body(server_model: str, prompt: str, temperature: float, seed: int) -> dict[str, Any]:
    BASE.SCHEMA = SCHEMA
    cfg = config()
    return {"model": server_model, "temperature": temperature, "top_p": cfg["request"]["top_p"], "seed": seed,
            "max_tokens": cfg["request"]["max_tokens"], "chat_template_kwargs": cfg["request"]["chat_template_kwargs"],
            "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object", "schema": BASE.score_schema()}}


def payload_layout(scores: dict[str, float], sentences: list[str], candidate: dict[str, Any], order: list[str], layout: str) -> str:
    return BASE_PAYLOAD_LAYOUT(scores, sentences, candidate, order, layout) + "\n\nResponse contract: return verdict scored if and only if all three hard_gates are true. Return verdict abstain if and only if at least one hard_gate is false. Always emit all three integer scores; abstain scores are ignored."


def prepare(args: argparse.Namespace) -> None:
    cfg = config(args.sample_essays)
    BASE.SCHEMA = SCHEMA; BASE.CONFIG_PATH = CONFIG_PATH; BASE.run_dir = run_dir; BASE.config = lambda: cfg; BASE.body = body; BASE.payload_layout = payload_layout
    BASE.prepare(args)
    manifest_path = run_dir(args.run_id) / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"schema_version": SCHEMA, "execution_mode": args.execution_mode, "requested_sample_essays": args.sample_essays, "max_tokens": cfg["request"]["max_tokens"], "context_safety_margin": cfg["runtime"]["context_safety_margin"], "retry_max_attempts": cfg["retry"]["max_attempts"]})
    BASE.atomic_json(manifest_path, manifest)


def validate_servers(args: argparse.Namespace, destination: Path, cfg: dict[str, Any]) -> dict[int, str]:
    values: dict[int, str] = {}
    for item in args.server:
        gpu_text, url = item.split("=", 1); gpu = int(gpu_text)
        if gpu not in (4, 5, 6, 7) or not url.startswith("http://127.0.0.1:"):
            raise RuntimeError("server mapping is outside v5 localhost GPU scope")
        values[gpu] = url
    attestation = json.loads((destination / "server_attestation.json").read_text(encoding="utf-8"))
    if set(values) != {4, 5, 6, 7} or attestation.get("config_sha256") != BASE.sha256(CONFIG_PATH) or attestation.get("physical_gpus") != [4, 5, 6, 7] or attestation.get("parallel_requests_per_server") != 1 or attestation.get("slot_context") != 4096:
        raise RuntimeError("server attestation failed")
    return values


def token_count(server: str, content: str) -> int:
    request = Request(server + "/tokenize", data=json.dumps({"content": content, "parse_special": False}, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=60) as response:
        tokens = json.loads(response.read().decode("utf-8")).get("tokens")
    if not isinstance(tokens, list) or not all(type(item) is int for item in tokens):
        raise RuntimeError("server tokenizer response is invalid")
    return len(tokens)


def call(server: str, request_body: dict[str, Any]) -> dict[str, Any]:
    result = WIRE.call(server, request_body)
    failure = result["failure"]
    return {"scores": result["scores"], "schema_valid": failure is None, "evidence_valid": failure is None and result["scores"] is not None,
            "abstain": failure is None and result["scores"] is None, "transport_or_schema_failure": failure is not None,
            "failure_category": failure, "attempts": result["attempts"]}


def execute(args: argparse.Namespace) -> None:
    cfg = config(); destination = run_dir(args.run_id); manifest_path = destination / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared" or (destination / "pilot_response_observations.jsonl").exists():
        raise RuntimeError("execution requires a newly prepared, untouched v5 run")
    servers = validate_servers(args, destination, cfg); requests = BASE.load_jsonl(destination / "pilot_requests.jsonl")
    token_counts = [token_count(servers[int(item["gpu"])], item["body"]["messages"][0]["content"]) for item in requests]
    if not token_counts or max(token_counts) + cfg["request"]["max_tokens"] + cfg["runtime"]["context_safety_margin"] > cfg["runtime"]["context_size"]:
        raise RuntimeError("prepared request exceeds attested per-slot context budget")
    grouped: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in (4, 5, 6, 7)}
    for item in requests: grouped[int(item["gpu"])].append(item)
    def process(gpu: int) -> list[dict[str, Any]]:
        return [{"opaque_request_key": item["opaque_request_key"], **call(servers[gpu], item["body"])} for item in grouped[gpu]]
    responses: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(process, gpu) for gpu in (4, 5, 6, 7)]
        for future in as_completed(futures): responses.extend(future.result())
    observations = destination / "pilot_response_observations.jsonl"
    with observations.open("x", encoding="utf-8") as handle:
        for item in responses: handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    watchdog = json.loads((destination / "watchdog_final.json").read_text(encoding="utf-8"))
    BASE.SCHEMA = SCHEMA; metrics = BASE.aggregate(requests, responses, cfg, int(watchdog.get("fault_count", -1)))
    metrics.update({"failure_categories": dict(sorted(Counter(item["failure_category"] for item in responses if item["failure_category"]).items())), "attempt_count": sum(item["attempts"] for item in responses), "context_budget": {"min_prompt_tokens": min(token_counts), "max_prompt_tokens": max(token_counts), "slot_context": 4096, "max_tokens": 192, "safety_margin": 256}})
    gate_values = BASE.gates(metrics, int(manifest["sample_essays"]), cfg)
    gate_values["context_budget"] = max(token_counts) + 192 + 256 <= 4096
    report = {"schema_version": SCHEMA, "created_at": BASE.now(), "status": "passed" if all(gate_values.values()) else "failed_gates", "metrics": metrics, "hard_gates": gate_values, "comparison": BASE.comparison(metrics, gate_values), "raw_payloads_restricted": True, "selection_artifact_constructed": False, "config_sha256": BASE.sha256(CONFIG_PATH), "request_sha256": BASE.sha256(destination / "pilot_requests.jsonl"), "response_observations_sha256": BASE.sha256(observations)}
    report_path = destination / "aggregate_pilot_report.json"; BASE.atomic_json(report_path, report)
    manifest.update({"status": "executed_passed_gates" if all(gate_values.values()) else "executed_failed_gates", "executed_at": BASE.now(), "aggregate_report_sha256": BASE.sha256(report_path), "pilot_passed_hard_gates": all(gate_values.values()), "selection_artifact_constructed": False})
    BASE.atomic_json(manifest_path, manifest)
    print(json.dumps({"status": report["status"], "sample_essays": manifest["sample_essays"], "hard_gates": gate_values, "selection_artifact_constructed": False}, sort_keys=True))
    if report["status"] != "passed": raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare"); prepare_parser.add_argument("--run-id", required=True); prepare_parser.add_argument("--server-model", default="qwen36-35b-a3b-q4_k_m"); prepare_parser.add_argument("--sample-essays", type=int, required=True); prepare_parser.add_argument("--execution-mode", choices=("smoke", "pilot"), required=True); prepare_parser.set_defaults(func=prepare)
    execute_parser = sub.add_parser("execute"); execute_parser.add_argument("--run-id", required=True); execute_parser.add_argument("--server", action="append", required=True); execute_parser.set_defaults(func=execute)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__": main()
