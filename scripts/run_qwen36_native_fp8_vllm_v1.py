#!/usr/bin/env python3
"""Native-FP8 runtime adapter that reuses the frozen v5.2 request semantics."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen36_native_fp8_vllm.v1.json"
SCHEMA = "mal2026-qwen36-native-fp8-vllm-v1"
SPEC = importlib.util.spec_from_file_location("v52_semantic_request_builder", ROOT / "scripts/run_openai_explanation_repeat_distribution_v5.py")
V5 = importlib.util.module_from_spec(SPEC); assert SPEC and SPEC.loader; SPEC.loader.exec_module(V5)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def config(sample: int | None = None) -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    r = cfg["runtime"]
    if cfg.get("schema_version") != SCHEMA or cfg["selection"].get("split") != "train" or r.get("physical_gpus") != [0, 1, 2, 3] or r.get("topology") != "four-independent-single-gpu-workers" or r.get("context_size") != 4096:
        raise RuntimeError("native-FP8 immutable topology/config changed")
    if cfg["request"] != {"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 192, "top_p": 1.0} or cfg["retry"].get("max_attempts") != 2 or cfg["protocol"].get("selection_artifact_permitted") is not False:
        raise RuntimeError("inherited request or selection contract changed")
    protocol = cfg["protocol"]
    if (protocol.get("controls") != {"duplicate_identity": True, "invalid_evidence": True, "padded_verbosity": True, "repeats": 5} or
            protocol.get("prompt_layouts") != ["rubric_then_essay", "essay_then_rubric", "rubric_compact", "essay_compact", "interleaved"] or
            protocol.get("rubric_permutations") != [["content", "organization", "expression"], ["content", "expression", "organization"], ["organization", "content", "expression"], ["organization", "expression", "content"], ["expression", "content", "organization"]]):
        raise RuntimeError("native-FP8 config is missing the inherited prompt/control protocol")
    if sample is not None:
        if sample not in (3, 2000): raise RuntimeError("only the authorized 3-essay smoke or 2,000-essay full run is allowed")
        cfg = copy.deepcopy(cfg); cfg["selection"]["max_essays"] = sample
    return cfg


def run_dir(run_id: str) -> Path:
    if not run_id.startswith("native-fp8-vllm-20260720-") or not run_id[-3:].isdigit(): raise RuntimeError("run id is outside native-FP8 lineage")
    return V5.BASE.RESTRICTED / V5.BASE.BATCH / "judge_runs" / run_id


def prepare(args: argparse.Namespace) -> None:
    cfg = config(args.sample_essays)
    V5.CONFIG_PATH = CONFIG; V5.SCHEMA = SCHEMA; V5.WIRE.SCHEMA = SCHEMA; V5.run_dir = run_dir
    original = V5.config; V5.config = lambda sample_essays=None: cfg
    try: V5.prepare(args)
    finally: V5.config = original
    path = run_dir(args.run_id) / "manifest.json"; value = json.loads(path.read_text(encoding="utf-8"))
    value.update({"schema_version": SCHEMA, "runtime_engine": "vllm", "runtime_topology": "four-independent-single-gpu-workers", "selected_physical_gpus": args.gpus, "max_model_len": 4096, "max_tokens": 192, "selection_artifact_constructed": False})
    V5.BASE.atomic_json(path, value)


def validate_servers(items: list[str], destination: Path, gpus: list[int]) -> dict[int, str]:
    servers = {}
    for item in items:
        gpu_s, url = item.split("=", 1); gpu = int(gpu_s); parsed = urlparse(url)
        if gpu not in gpus or parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None: raise RuntimeError("endpoint is outside the attested localhost GPU scope")
        servers[gpu] = url
    attest = json.loads((destination / "server_attestation.json").read_text(encoding="utf-8"))
    if set(servers) != set(gpus) or attest.get("config_sha256") != sha(CONFIG) or attest.get("physical_gpus") != gpus or attest.get("tensor_parallel_size") != 1 or attest.get("max_model_len") != 4096 or attest.get("watchdog_faults") != 0:
        raise RuntimeError("vLLM server attestation failed")
    return servers


def token_count(endpoint: str, content: str) -> int:
    req = Request(endpoint + "/tokenize", data=json.dumps({"prompt": content}, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=60) as response: value = json.loads(response.read().decode("utf-8"))
    tokens = value.get("tokens")
    if not isinstance(tokens, list) or not all(type(x) is int for x in tokens): raise RuntimeError("vLLM tokenizer response is invalid")
    return len(tokens)


def vllm_grammar_schema(schema: dict) -> dict:
    """Remove only vLLM 0.25.1 xgrammar's unsupported multi-clause allOf.

    The full conditional relation remains enforced by the inherited
    fail-closed response normalizer; this projection is only for token-level
    JSON-schema grammar construction in native vLLM.
    """
    projected = copy.deepcopy(schema)
    constraints = projected.pop("allOf", None)
    if constraints is None:
        return projected
    if (not isinstance(constraints, list) or len(constraints) != 2 or
            any(not isinstance(item, dict) for item in constraints)):
        raise RuntimeError("frozen verdict-to-hard-gates constraint changed unexpectedly")
    return projected


def vllm_json_schema_body(request_body: dict) -> dict:
    """Apply the one native-vLLM response-envelope compatibility change.

    The inherited train-only builder uses the llama.cpp-compatible
    ``json_object`` plus ``schema`` form.  vLLM 0.25.1 accepts that request
    but deliberately constrains only JSON syntax, silently ignoring that
    sibling schema.  Its OpenAI-compatible ``json_schema`` form nests the
    same schema under a named object.  vLLM's grammar receives the shape-only
    projection because xgrammar reports incomplete support for the inherited
    multi-clause conditional ``allOf``; the unchanged normalizer still checks
    that condition after decoding.  Prompts, rubric, sampling, no-thinking
    setting, and every other schema constraint are untouched.
    """
    request = copy.deepcopy(request_body)
    legacy = request.pop("response_format", None)
    if (not isinstance(legacy, dict) or legacy.get("type") != "json_object" or
            set(legacy) != {"type", "schema"} or not isinstance(legacy["schema"], dict)):
        raise RuntimeError("frozen inherited response schema is unavailable for vLLM translation")
    properties = legacy["schema"].get("properties")
    if (not isinstance(properties, dict) or
            properties.get("schema_version", {}).get("const") != SCHEMA):
        raise RuntimeError("inherited response schema is not bound to the native-FP8 lineage")
    request["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": SCHEMA, "strict": True,
                        "schema": vllm_grammar_schema(legacy["schema"])},
    }
    return request


def execute(args: argparse.Namespace) -> None:
    V5.CONFIG_PATH = CONFIG; V5.SCHEMA = SCHEMA; V5.WIRE.SCHEMA = SCHEMA
    cfg = config(); dest = run_dir(args.run_id); manifest_path = dest / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared" or (dest / "pilot_response_observations.jsonl").exists() or manifest.get("selected_physical_gpus") != args.gpus: raise RuntimeError("execution requires a fresh, topology-matched prepared run")
    servers = validate_servers(args.server, dest, args.gpus); requests = V5.BASE.load_jsonl(dest / "pilot_requests.jsonl")
    counts = [token_count(servers[int(x["gpu"])], x["body"]["messages"][0]["content"]) for x in requests]
    if not counts or max(counts) + 192 > 4096: raise RuntimeError("real train-only request exceeds vLLM context contract")
    def call(item: dict) -> dict:
        result = V5.WIRE.call(servers[int(item["gpu"])], vllm_json_schema_body(item["body"]))
        failure = result["failure"]
        return {"opaque_request_key": item["opaque_request_key"], "scores": result["scores"], "schema_valid": failure is None, "evidence_valid": failure is None and result["scores"] is not None, "abstain": failure is None and result["scores"] is None, "transport_or_schema_failure": failure is not None, "failure_category": failure, "attempts": result["attempts"]}
    grouped = {gpu: [] for gpu in args.gpus}
    for item in requests: grouped[int(item["gpu"])].append(item)
    def endpoint_work(gpu: int) -> list[dict]:
        with ThreadPoolExecutor(max_workers=4) as pool: return [future.result() for future in as_completed([pool.submit(call, item) for item in grouped[gpu]])]
    responses = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        for future in as_completed([pool.submit(endpoint_work, gpu) for gpu in args.gpus]): responses.extend(future.result())
    observations = dest / "pilot_response_observations.jsonl"
    with observations.open("x", encoding="utf-8") as f:
        for item in responses: f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    watchdog = json.loads((dest / "watchdog_final.json").read_text(encoding="utf-8")); V5.BASE.SCHEMA = SCHEMA
    metrics = V5.BASE.aggregate(requests, responses, cfg, int(watchdog.get("fault_count", -1)))
    metrics.update({"failure_categories": dict(sorted(Counter(x["failure_category"] for x in responses if x["failure_category"]).items())), "attempt_count": sum(x["attempts"] for x in responses), "context_budget": {"min_prompt_tokens": min(counts), "max_prompt_tokens": max(counts), "max_model_len": 4096, "max_tokens": 192}})
    gates = V5.BASE.gates(metrics, int(manifest["sample_essays"]), cfg); gates["context_budget"] = max(counts) + 192 <= 4096
    passed = all(gates.values()); report = {"schema_version": SCHEMA, "status": "passed" if passed else "failed_gates", "metrics": metrics, "hard_gates": gates, "raw_payloads_restricted": True, "selection_artifact_constructed": False, "config_sha256": sha(CONFIG), "request_sha256": sha(dest / "pilot_requests.jsonl"), "response_observations_sha256": sha(observations), "selected_physical_gpus": args.gpus}
    report_path = dest / "aggregate_pilot_report.json"; V5.BASE.atomic_json(report_path, report)
    manifest.update({"status": "executed_passed_gates" if passed else "executed_failed_gates", "aggregate_report_sha256": sha(report_path), "pilot_passed_hard_gates": passed, "selection_artifact_constructed": False}); V5.BASE.atomic_json(manifest_path, manifest)
    print(json.dumps({"status": report["status"], "sample_essays": manifest["sample_essays"], "hard_gates": gates}, sort_keys=True))
    if not passed: raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--run-id", required=True); prep.add_argument("--gpus", nargs="+", type=int, required=True); prep.add_argument("--sample-essays", type=int, required=True); prep.add_argument("--execution-mode", choices=("smoke", "full"), required=True); prep.add_argument("--server-model", required=True); prep.set_defaults(func=prepare)
    exe = sub.add_parser("execute"); exe.add_argument("--run-id", required=True); exe.add_argument("--gpus", nargs="+", type=int, required=True); exe.add_argument("--server", action="append", required=True); exe.set_defaults(func=execute)
    a = p.parse_args(); a.func(a)


if __name__ == "__main__": main()
