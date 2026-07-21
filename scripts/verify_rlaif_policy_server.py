#!/usr/bin/env python3
"""Synthetic-only health gate for vLLM dynamic LoRA plus strict JSON output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def post(endpoint: str, route: str, body: dict[str, object]) -> dict[str, object] | str:
    wire = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with urlopen(Request(endpoint + route, data=wire, headers={"Content-Type": "application/json"}, method="POST"), timeout=180) as response:
        value = response.read().decode("utf-8")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if not isinstance(parsed, dict):
        raise RuntimeError("policy-server response envelope is not an object")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    if not args.adapter.is_dir() or not (args.adapter / "adapter_config.json").is_file():
        raise RuntimeError("policy-server source adapter is unavailable")
    alias = "rlaif_policy_synthetic_gate"
    post(endpoint, "/v1/load_lora_adapter", {"lora_name": alias, "lora_path": str(args.adapter.resolve()), "load_inplace": True})
    try:
        schema = {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"], "additionalProperties": False}
        result = post(endpoint, "/v1/chat/completions", {"model": alias, "messages": [{"role": "user", "content": "Return only the requested JSON object."}], "temperature": 0.0, "top_p": 1.0, "max_tokens": 32, "seed": 2026072209, "response_format": {"type": "json_schema", "json_schema": {"name": "synthetic_policy_server", "strict": True, "schema": schema}}})
        choices = result.get("choices") if isinstance(result, dict) else None
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise RuntimeError("policy-server synthetic completion envelope differs")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if choices[0].get("finish_reason") != "stop" or not isinstance(content, str):
            raise RuntimeError("policy-server synthetic completion did not stop")
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or set(parsed) != {"result"} or not isinstance(parsed["result"], str):
            raise RuntimeError("policy-server synthetic JSON schema differs")
    finally:
        post(endpoint, "/v1/unload_lora_adapter", {"lora_name": alias})
    print(json.dumps({"status": "passed", "dynamic_lora": True, "structured_json": True, "raw_writing_or_completion_persisted": False}, sort_keys=True))


if __name__ == "__main__":
    main()
