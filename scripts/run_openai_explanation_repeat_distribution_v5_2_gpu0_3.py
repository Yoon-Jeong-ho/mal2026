#!/usr/bin/env python3
"""Migrated v5.2 runner for project-owned GPUs 0--3 only.

It retains v5.2's parser, byte-stable retry, controls, and train-only request
builder.  A one-GPU smoke records the cross-GPU gate as not evaluated; that
gate remains mandatory whenever the selected topology contains multiple GPUs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/openai_explanation_repeat_distribution.v5_2.gpu0_3.json"
SCHEMA = "mal2026-openai-explanation-repeat-distribution-v5_2-gpu0_3"
SPEC = importlib.util.spec_from_file_location("repeat_v5_2", ROOT / "scripts/run_openai_explanation_repeat_distribution_v5_2.py")
assert SPEC and SPEC.loader
WRAPPER = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(WRAPPER)
V5 = WRAPPER.V5
V5.CONFIG_PATH = CONFIG_PATH
V5.SCHEMA = SCHEMA
V5.WIRE.SCHEMA = SCHEMA


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir(run_id: str) -> Path:
    if not run_id.startswith("openai-repeat-v5_2-gpu0-3-20260720-") or not run_id[-3:].isdigit():
        raise RuntimeError("run id does not bind the v5.2 GPU0--3 lineage")
    return V5.BASE.RESTRICTED / V5.BASE.BATCH / "judge_runs" / run_id


def config(sample_essays: int | None, selected_gpus: list[int]) -> dict[str, Any]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8")); runtime = cfg["runtime"]
    if cfg.get("schema_version") != SCHEMA or cfg["selection"].get("split") != "train":
        raise RuntimeError("unexpected v5.2 GPU0--3 configuration")
    allowed = runtime.get("physical_gpus")
    if allowed != [0, 1, 2, 3] or not selected_gpus or len(set(selected_gpus)) != len(selected_gpus) or any(gpu not in allowed for gpu in selected_gpus):
        raise RuntimeError("GPU topology is outside project-owned GPUs 0--3")
    parallel, total, slot = runtime.get("parallel_requests_per_server"), runtime.get("context_size"), runtime.get("slot_context_size")
    if parallel != 4 or total != 16384 or slot != 4096 or total // parallel != slot or runtime.get("context_safety_margin") != 256:
        raise RuntimeError("v5.2 context/parallel remediation contract changed")
    if cfg["request"] != {"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 192, "top_p": 1.0} or cfg["retry"].get("max_attempts") != 2:
        raise RuntimeError("v5.2 request/retry contract changed")
    if sample_essays is not None and not 1 <= sample_essays <= cfg["selection"]["max_essays"]:
        raise RuntimeError("requested sample is outside train-only v5.2 envelope")
    result = copy.deepcopy(cfg)
    result["runtime"]["physical_gpus"] = list(selected_gpus)
    if sample_essays is not None:
        result["selection"]["max_essays"] = sample_essays
    return result


def prepare(args: argparse.Namespace) -> None:
    cfg = config(args.sample_essays, args.gpus)
    # The inherited builder stripes over four positions.  Repeating GPU0 keeps
    # the same fixed repeat assignment for the explicitly authorized fallback.
    build_cfg = copy.deepcopy(cfg)
    if len(args.gpus) == 1:
        build_cfg["runtime"]["physical_gpus"] = args.gpus * 4
    original_config = V5.config
    V5.CONFIG_PATH = CONFIG_PATH; V5.SCHEMA = SCHEMA; V5.WIRE.SCHEMA = SCHEMA; V5.run_dir = run_dir
    V5.config = lambda sample_essays=None: build_cfg
    try:
        V5.prepare(args)
    finally:
        V5.config = original_config
    manifest_path = run_dir(args.run_id) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"schema_version": SCHEMA, "selected_physical_gpus": args.gpus,
                     "parallel_requests_per_server": 4, "slot_context_size": 4096,
                     "expected_real_calls": args.sample_essays * 3 * 10,
                     "repeat_schedule": "five deterministic plus five dispersion repeats per isolated candidate",
                     "gpu_ownership": "project-owned GPUs 0--3 only; GPUs 4--7 never queried or used"})
    V5.BASE.atomic_json(manifest_path, manifest)


def validate_servers(values: list[str], destination: Path, selected_gpus: list[int]) -> dict[int, str]:
    servers: dict[int, str] = {}
    for item in values:
        gpu_text, url = item.split("=", 1); gpu = int(gpu_text); parsed = urlparse(url)
        if gpu not in selected_gpus or parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None:
            raise RuntimeError("server mapping is not an allowed localhost selected GPU")
        servers[gpu] = url
    attestation = json.loads((destination / "server_attestation.json").read_text(encoding="utf-8"))
    if set(servers) != set(selected_gpus) or attestation.get("config_sha256") != sha(CONFIG_PATH) or attestation.get("physical_gpus") != selected_gpus or attestation.get("parallel_requests_per_server") != 4 or attestation.get("slot_context") != 4096 or attestation.get("watchdog_faults") != 0:
        raise RuntimeError("server attestation failed")
    return servers


def token_count(server: str, content: str) -> int:
    request = Request(server + "/tokenize", data=json.dumps({"content": content, "parse_special": False}, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=60) as response:
        tokens = json.loads(response.read().decode("utf-8")).get("tokens")
    if not isinstance(tokens, list) or not all(type(token) is int for token in tokens):
        raise RuntimeError("server tokenizer response is invalid")
    return len(tokens)


def execute(args: argparse.Namespace) -> None:
    cfg = config(None, args.gpus); destination = run_dir(args.run_id); manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observations = destination / "pilot_response_observations.jsonl"
    if manifest.get("status") != "prepared" or observations.exists() or manifest.get("selected_physical_gpus") != args.gpus:
        raise RuntimeError("execution requires a newly prepared, topology-matched run")
    servers = validate_servers(args.server, destination, args.gpus)
    requests = V5.BASE.load_jsonl(destination / "pilot_requests.jsonl")
    if any(int(item["gpu"]) not in servers for item in requests):
        raise RuntimeError("prepared request escaped selected GPU topology")
    counts = [token_count(servers[int(item["gpu"])], item["body"]["messages"][0]["content"]) for item in requests]
    if not counts or max(counts) + 192 + 256 > 4096:
        raise RuntimeError("prepared request exceeds the immutable per-slot context budget")
    def call(item: dict[str, Any]) -> dict[str, Any]:
        return {"opaque_request_key": item["opaque_request_key"], **V5.call(servers[int(item["gpu"])], item["body"])}
    grouped: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in args.gpus}
    for item in requests:
        grouped[int(item["gpu"])].append(item)
    def process_endpoint(gpu: int) -> list[dict[str, Any]]:
        # Independent endpoint queues cap admission at the attested four slots.
        with ThreadPoolExecutor(max_workers=4) as endpoint_pool:
            futures = [endpoint_pool.submit(call, item) for item in grouped[gpu]]
            return [future.result() for future in as_completed(futures)]
    responses: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as endpoints:
        futures = [endpoints.submit(process_endpoint, gpu) for gpu in args.gpus]
        for future in as_completed(futures):
            responses.extend(future.result())
    with observations.open("x", encoding="utf-8") as handle:
        for item in responses:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    watchdog = json.loads((destination / "watchdog_final.json").read_text(encoding="utf-8"))
    V5.BASE.SCHEMA = SCHEMA
    metrics = V5.BASE.aggregate(requests, responses, cfg, int(watchdog.get("fault_count", -1)))
    metrics.update({"failure_categories": dict(sorted(Counter(item["failure_category"] for item in responses if item["failure_category"]).items())),
                    "attempt_count": sum(item["attempts"] for item in responses),
                    "context_budget": {"min_prompt_tokens": min(counts), "max_prompt_tokens": max(counts), "slot_context": 4096, "max_tokens": 192, "safety_margin": 256}})
    gates = V5.BASE.gates(metrics, int(manifest["sample_essays"]), cfg)
    gates["context_budget"] = max(counts) + 192 + 256 <= 4096
    topology_gate = "evaluated" if len(args.gpus) > 1 else "not_evaluated_single_gpu"
    if len(args.gpus) == 1:
        # Cross-GPU stability is not executable on a GPU0-only phase and is
        # therefore omitted rather than treated as a passing observation.  It
        # remains a required configured gate for every multi-GPU full run.
        gates.pop("cross_gpu_agreement", None)
    passed = all(gates.values())
    report = {"schema_version": SCHEMA, "created_at": V5.BASE.now(), "status": "passed" if passed else "failed_gates",
              "metrics": metrics, "hard_gates": gates, "cross_gpu_gate": topology_gate,
              "comparison": V5.BASE.comparison(metrics, gates), "raw_payloads_restricted": True,
              "selection_artifact_constructed": False, "config_sha256": sha(CONFIG_PATH),
              "request_sha256": sha(destination / "pilot_requests.jsonl"), "response_observations_sha256": sha(observations),
              "selected_physical_gpus": args.gpus, "parallel_requests_per_server": 4, "slot_context_size": 4096,
              "gpu_ownership": "project-owned GPUs 0--3 only; GPUs 4--7 never queried or used"}
    report_path = destination / "aggregate_pilot_report.json"; V5.BASE.atomic_json(report_path, report)
    manifest.update({"status": "executed_passed_gates" if passed else "executed_failed_gates", "executed_at": V5.BASE.now(),
                     "aggregate_report_sha256": sha(report_path), "pilot_passed_hard_gates": passed,
                     "selection_artifact_constructed": False})
    V5.BASE.atomic_json(manifest_path, manifest)
    print(json.dumps({"status": report["status"], "sample_essays": manifest["sample_essays"], "hard_gates": gates, "cross_gpu_gate": topology_gate, "selection_artifact_constructed": False}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (("prepare", prepare), ("execute", execute)):
        item = sub.add_parser(name); item.add_argument("--run-id", required=True); item.add_argument("--gpus", nargs="+", type=int, required=True)
        if name == "prepare":
            item.add_argument("--sample-essays", type=int, required=True); item.add_argument("--execution-mode", choices=("smoke", "full"), required=True); item.add_argument("--server-model", default="qwen36-35b-a3b-q4_k_m")
        else:
            item.add_argument("--server", action="append", required=True)
        item.set_defaults(func=function)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
