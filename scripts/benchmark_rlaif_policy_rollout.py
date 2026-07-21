#!/usr/bin/env python3
"""Aggregate-only live vLLM rollout-batch gate for RLAIF policy settings.

This does not generate training data, start a trainer, or call the Qwen
reward model.  It proves that the exact score-blind policy prompt, JSON schema,
adapter, sample count, and sampler settings complete as one full GRPO rollout
batch before a long continuation is authorized.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from typing import Any
from urllib.request import Request, urlopen

from mal2026.api_rationale_data import axes_for_task
from mal2026.rlaif_grpo import (
    RLAIFRunConfig,
    RLAIFSettings,
    _policy_response_schema,
    canonical_completion_text,
    train_examples,
)


def _post(endpoint: str, route: str, body: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    wire = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with urlopen(Request(endpoint + route, data=wire, headers={"Content-Type": "application/json"}, method="POST"), timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
    # vLLM's dynamic LoRA load/unload endpoints acknowledge a successful
    # in-place mutation with a non-JSON or empty 2xx body depending on the
    # server patch version.  Chat completions must remain JSON envelopes.
    if route in {"/v1/load_lora_adapter", "/v1/unload_lora_adapter"}:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("vLLM response envelope is not an object")
    return value


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-prompts", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    args = parser.parse_args()
    if args.source_prompts < 1 or args.timeout_seconds < 1 or args.max_wall_seconds <= 0:
        raise RuntimeError("benchmark bounds are invalid")
    endpoint = args.endpoint.rstrip("/")
    if not args.adapter.is_dir() or not (args.adapter / "adapter_config.json").is_file():
        raise RuntimeError("policy adapter is unavailable")
    run = RLAIFRunConfig.from_json(args.run_config)
    settings = RLAIFSettings.from_json()
    examples, provenance = train_examples(settings, run)
    selected = examples[: args.source_prompts]
    if len(selected) != args.source_prompts:
        raise RuntimeError("requested source-prompt count exceeds train population")
    axes = axes_for_task(run.task)
    mode = str(settings.policy.get("rollout_structured_output_mode", "json_schema"))
    if mode == "json_schema":
        schema = _policy_response_schema(
            axes,
            int(settings.reward["field_character_limit"]),
            enforce_character_limit=bool(settings.policy.get("rollout_json_schema_enforces_field_limit", True)),
        )
        response_format: dict[str, Any] = {"type": "json_schema", "json_schema": {"name": "mal2026_rationale_only_v1", "strict": True, "schema": schema}}
    elif mode == "json_object":
        response_format = {"type": "json_object"}
    else:
        raise RuntimeError("unsupported policy structured output mode")
    alias = f"rlaif_policy_batch_gate_{run.run_id.replace('-', '_')}"
    _post(endpoint, "/v1/load_lora_adapter", {"lora_name": alias, "lora_path": str(args.adapter.resolve()), "load_inplace": True}, args.timeout_seconds)
    durations: list[float] = []
    outputs: list[list[str]] = []
    try:
        def one(index: int) -> tuple[float, list[str]]:
            item = selected[index]
            started = time.monotonic()
            outer = _post(endpoint, "/v1/chat/completions", {
                "model": alias,
                "messages": item["prompt"],
                "n": int(settings.policy["num_generations"]),
                "temperature": float(settings.policy["sampling_temperature"]),
                "top_p": float(settings.policy["sampling_top_p"]),
                "max_tokens": int(settings.policy["max_completion_tokens"]),
                "seed": int(settings.policy["seed"]) + index,
                "response_format": response_format,
            }, args.timeout_seconds)
            choices = outer.get("choices")
            if not isinstance(choices, list) or len(choices) != int(settings.policy["num_generations"]):
                raise RuntimeError("policy batch response choice count differs")
            values: list[str] = []
            for choice in choices:
                if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
                    raise RuntimeError("policy batch response did not stop")
                message = choice.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str):
                    raise RuntimeError("policy batch response content is missing")
                values.append(content)
            return time.monotonic() - started, values

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            for duration, values in pool.map(one, range(len(selected))):
                durations.append(duration)
                outputs.append(values)
        wall = time.monotonic() - started
    finally:
        _post(endpoint, "/v1/unload_lora_adapter", {"lora_name": alias}, args.timeout_seconds)
    flattened = [item for group in outputs for item in group]
    valid = sum(canonical_completion_text(text, axes, int(settings.reward["field_character_limit"])) is not None for text in flattened)
    payload = {
        "schema_version": "mal2026-rlaif-policy-rollout-batch-gate-v1",
        "status": "passed" if valid == len(flattened) and wall <= args.max_wall_seconds else "failed_gate",
        "run_id": run.run_id,
        "source_prompts": len(selected),
        "num_generations": int(settings.policy["num_generations"]),
        "policy_completions": len(flattened),
        "parse_valid": valid,
        "parse_valid_rate": round(valid / len(flattened), 6),
        "structured_output_mode": mode,
        "structured_json_schema_field_max_length_enforced": bool(settings.policy.get("rollout_json_schema_enforces_field_limit", True)),
        "batch_wall_seconds": round(wall, 3),
        "request_latency_seconds": {"p50": round(_percentile(durations, 0.5), 3), "p95": round(_percentile(durations, 0.95), 3), "max": round(max(durations), 3)},
        "batch_wall_limit_seconds": args.max_wall_seconds,
        "input_provenance": provenance,
        "source_writing_scores_read_or_prompted": False,
        "candidate_scores_read_or_prompted": False,
        "raw_prompts_or_completions_persisted": False,
    }
    args.report.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "passed":
        raise SystemExit("full rollout batch gate failed")


if __name__ == "__main__":
    main()
