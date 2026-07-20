#!/usr/bin/env python3
"""Data-free, <=20-call-per-worker OpenAI/vLLM schema and non-thinking gate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen36_native_fp8_vllm.v1.json"
SPEC = importlib.util.spec_from_file_location("inherited_v52_wire", ROOT / "scripts/preflight_openai_repeat_v5_synthetic.py")
WIRE = importlib.util.module_from_spec(SPEC); assert SPEC and SPEC.loader; SPEC.loader.exec_module(WIRE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vllm_grammar_schema(schema: dict) -> dict:
    """Project only vLLM 0.25.1's unsupported conditional constraint.

    The synthetic replay showed xgrammar's explicit warning that its support
    for a multi-clause ``allOf`` is incomplete.  The three conditional clauses
    are already fail-closed in ``WIRE.normalize`` after decoding, so retain the
    complete object shape and every scalar bound in the grammar while removing
    exactly the unsupported grammar-only encoding of that semantic relation.
    """
    projected = copy.deepcopy(schema)
    constraints = projected.pop("allOf", None)
    if constraints is None:
        return projected
    if (not isinstance(constraints, list) or len(constraints) != 2 or
            any(not isinstance(item, dict) for item in constraints)):
        raise RuntimeError("frozen verdict-to-hard-gates constraint changed unexpectedly")
    return projected


def vllm_json_schema_body(text: str, schema_name: str) -> dict:
    """Translate the frozen OpenAI/llama JSON wire shape for vLLM.

    vLLM 0.25.1 treats ``response_format.type=json_object`` only as a
    request for *some* JSON object; the sibling ``schema`` field used by the
    llama.cpp/OpenAI-compatible endpoint is not consumed.  Its documented
    schema-constrained form is ``json_schema`` with a named nested schema.
    The native grammar projection omits only its unsupported conditional
    ``allOf`` encoding; the inherited normalizer still enforces that exact
    verdict-to-hard-gates relation after decoding.  Prompts, sampling fields,
    and all other schema bounds stay unchanged.
    """
    request = copy.deepcopy(WIRE.body(text))
    legacy = request.pop("response_format", None)
    if (not isinstance(legacy, dict) or legacy.get("type") != "json_object" or
            set(legacy) != {"type", "schema"} or not isinstance(legacy["schema"], dict)):
        raise RuntimeError("frozen inherited response schema is unavailable for vLLM translation")
    request["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "strict": True,
                        "schema": vllm_grammar_schema(legacy["schema"])},
    }
    return request


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--endpoint", required=True); p.add_argument("--model", required=True)
    p.add_argument("--gpu", type=int, required=True); p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=CONFIG); a = p.parse_args()
    cfg_path = a.config.resolve()
    if not cfg_path.is_file() or cfg_path.is_symlink() or cfg_path.parent != CONFIG.parent.resolve():
        raise SystemExit("synthetic worker config is outside the canonical config directory")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")); runtime = cfg["runtime"]
    if a.gpu not in runtime["physical_gpus"] or not a.endpoint.startswith("http://127.0.0.1:"):
        raise SystemExit("synthetic worker is outside the project GPU/localhost scope")
    if cfg["request"] != {"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 192, "top_p": 1.0} or runtime["context_size"] != 4096:
        raise SystemExit("native synthetic request/context contract changed")
    if a.run_dir.exists(): raise SystemExit("refusing to overwrite synthetic aggregate")
    a.run_dir.mkdir(parents=True)
    WIRE.SCHEMA = cfg["schema_version"]; WIRE.MAX_TOKENS = 192; WIRE.RETRY_ATTEMPTS = 2
    jobs = [(kind, vllm_json_schema_body(WIRE.prompt(kind), cfg["schema_version"]))
            for kind in ("boundary", "duplicate", "padded", "invalid") for _ in range(5)]
    # This is a throughput gate for one continuously-batched vLLM endpoint,
    # not a latency benchmark for 20 serial HTTP calls.  Keep the immutable
    # 5-repeat control set, but submit its independent calls together up to
    # the endpoint's configured sequence capacity.
    client_concurrency = cfg["protocol"].get("client_concurrency", runtime.get("max_num_seqs", 4))
    server_capacity = runtime.get("max_num_seqs", client_concurrency)
    if (type(client_concurrency) is not int or type(server_capacity) is not int or
            not 1 <= client_concurrency <= server_capacity):
        raise SystemExit("synthetic client concurrency exceeds the configured vLLM capacity")
    def invoke(job: tuple[str, dict]) -> tuple[str, dict, float]:
        kind, body = job
        body["model"] = a.model
        started = time.monotonic()
        result = WIRE.call(a.endpoint, body)
        return kind, result, time.monotonic() - started

    results = defaultdict(list); elapsed = []
    wall_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(client_concurrency, len(jobs))) as pool:
        # executor.map preserves immutable control ordering while allowing
        # vLLM's continuous batcher to schedule all independent requests.
        for kind, result, duration in pool.map(invoke, jobs):
            results[kind].append(result); elapsed.append(duration)
    wall_elapsed = time.monotonic() - wall_started
    failures = Counter(item["failure"] for rows in results.values() for item in rows if item["failure"])
    valid = [item["scores"] for kind, rows in results.items() if kind in {"boundary", "duplicate", "padded"} for item in rows if item["failure"] is None]
    invalid = results["invalid"]; duplicate = [item["scores"] for item in results["duplicate"] if item["failure"] is None]
    padded = [item["scores"] for item in results["padded"] if item["failure"] is None]
    base = [sum(x.values()) for x in duplicate if x is not None]; pad = [sum(x.values()) for x in padded if x is not None]
    repeat = {kind: len(rows) == 5 and all(x["failure"] is None for x in rows) and len({json.dumps(x["scores"], sort_keys=True) for x in rows}) == 1 for kind, rows in results.items()}
    aggregate = {"calls": len(jobs), "schema_valid_calls": sum(item["failure"] is None for rows in results.values() for item in rows),
                 "failure_categories": dict(sorted(failures.items())), "attempt_count": sum(item["attempts"] for rows in results.values() for item in rows),
                 "no_thinking_placement": "passed" if not failures.get("reasoning_present") else "failed",
                 "repeat_agreement": repeat, "required_rubric_fields_parsed": len(valid) == 15,
                 "invalid_control_abstain": all(x["failure"] is None and x["scores"] is None for x in invalid),
                 "duplicate_identity_agreement": len({json.dumps(x, sort_keys=True) for x in duplicate}) == 1,
                 "padded_verbosity_non_improvement": bool(base and pad and max(pad) <= min(base)),
                 "client_concurrency": min(client_concurrency, len(jobs)),
                 "latency_seconds": {"count": len(elapsed), "total": round(sum(elapsed), 6), "mean": round(sum(elapsed) / len(elapsed), 6), "max": round(max(elapsed), 6), "wall_total": round(wall_elapsed, 6)},
                 "throughput_requests_per_second": round(len(elapsed) / wall_elapsed, 6) if wall_elapsed else 0.0}
    gates = {"zero_transport_or_schema_failures": aggregate["calls"] == aggregate["schema_valid_calls"] and not aggregate["failure_categories"],
             "no_thinking_placement": aggregate["no_thinking_placement"] == "passed", "all_required_rubric_fields_parse": aggregate["required_rubric_fields_parsed"],
             "deterministic_repeat_agreement": all(repeat.values()), "invalid_control": aggregate["invalid_control_abstain"],
             "duplicate_control": aggregate["duplicate_identity_agreement"], "padded_control": aggregate["padded_verbosity_non_improvement"]}
    report = {"schema_version": cfg["schema_version"], "status": "passed" if all(gates.values()) else "failed_gates", "data_access": False,
              "raw_prompts_or_responses_persisted": False, "physical_gpu": a.gpu, "context_size": 4096, "max_tokens": 192,
              "aggregate": aggregate, "hard_gates": gates, "config_sha256": sha(cfg_path), "inherited_wire_sha256": sha(ROOT / "scripts/preflight_openai_repeat_v5_synthetic.py")}
    (a.run_dir / "aggregate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "calls": 20, "gpu": a.gpu, "aggregate": str(a.run_dir / "aggregate.json")}, sort_keys=True))
    if report["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__": main()
