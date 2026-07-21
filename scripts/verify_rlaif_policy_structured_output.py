#!/usr/bin/env python3
"""Aggregate-only GPU preflight for the exact RLAIF policy JSON contract.

This is deliberately not a generation-data producer.  It samples the existing
SFT policy with vLLM's JSON-schema constrained decoding and writes only the
parse/count evidence required to decide whether online GRPO can proceed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mal2026.rlaif_grpo import (
    RLAIFRunConfig,
    RLAIFSettings,
    canonical_completion_text,
    train_examples,
)


def response_schema(axes: tuple[str, ...], character_limit: int) -> dict[str, object]:
    rationale = {"type": "string", "minLength": 1, "maxLength": character_limit}
    axis = {
        "type": "object",
        "properties": {"rationale": rationale},
        "required": ["rationale"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {"const": "rationale-only-v1"},
            **{name: axis for name in axes},
        },
        "required": ["schema_version", *axes],
        "additionalProperties": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run = RLAIFRunConfig.from_json(args.config)
    settings = RLAIFSettings.from_json()
    examples, _ = train_examples(settings, run)

    # Imports are local to the GPU stage, so module-level validation remains CPU-only.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    model_path = Path(run.source_adapter).parent / "training_complete.json"
    source = json.loads(model_path.read_text(encoding="utf-8"))
    base_path = Path(str(source["config"]["model_path"]))
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    prompts = [
        tokenizer.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
        for item in examples
    ]
    llm = LLM(
        model=str(base_path),
        dtype="bfloat16",
        seed=run.seed,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=3072,
    )
    from mal2026.api_rationale_data import axes_for_task

    axes = axes_for_task(run.task)
    schema = response_schema(tuple(axes), int(settings.reward["field_character_limit"]))
    params = SamplingParams(
        n=int(settings.policy["num_generations"]),
        temperature=float(settings.policy["sampling_temperature"]),
        top_p=float(settings.policy["sampling_top_p"]),
        top_k=0,
        max_tokens=int(settings.policy["max_completion_tokens"]),
        structured_outputs=StructuredOutputsParams(json=schema, disable_additional_properties=True),
    )
    outputs = llm.generate(
        prompts,
        sampling_params=params,
        lora_request=LoRARequest("sft", 1, str(Path(run.source_adapter))),
        use_tqdm=False,
    )
    total = 0
    valid = 0
    for request in outputs:
        for output in request.outputs:
            total += 1
            valid += canonical_completion_text(output.text, axes, int(settings.reward["field_character_limit"])) is not None
    payload = {
        "schema_version": "mal2026-rlaif-vllm-structured-policy-preflight-v1",
        "status": "passed" if valid == total else "failed_gate",
        "run_id": run.run_id,
        "policy_completions": total,
        "parse_valid": valid,
        "parse_invalid": total - valid,
        "parse_valid_rate": round(valid / total, 6) if total else None,
        "generation_backend": "vllm_structured_outputs_json_schema",
        "source_writing_scores_read_or_prompted": False,
        "candidate_scores_read_or_prompted": False,
        "raw_prompts_or_completions_persisted": False,
    }
    args.report.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if valid != total:
        raise SystemExit("strict structured-policy parse gate failed")


if __name__ == "__main__":
    main()
