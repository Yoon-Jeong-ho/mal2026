#!/usr/bin/env python3
"""Generate the final score-blind student-rationale train/validation handoff."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mal2026.api_rationale_data import sha256_file  # noqa: E402
from mal2026.rationale_pipeline_prompts import routing  # noqa: E402
from run_rationale_pipeline_sft_evaluation import (  # noqa: E402
    GENERATION_AGGREGATE,
    GENERATION_RESTRICTED,
    GEN_PORTS,
    GPUS,
    launch_vllm,
    run_generation_on_server,
    stop_owned,
    wait_released,
)


RESTRICTED_PARENT = ROOT / "data/processed/restricted/rationale_pipeline_v1/final_student"
OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-final-generation-v1"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--training-completion", type=Path, required=True)
    args = parser.parse_args()
    setproctitle(f"mal2026:rationale-final-generation:{args.candidate_key}"[:255])
    need((args.base_model / "config.json").is_file() and (args.adapter / "adapter_model.safetensors").is_file(), "final rationale model artifact unavailable")
    completion = json.loads(args.training_completion.read_text(encoding="utf-8"))
    score_blind = completion.get("human_or_reference_score_read_or_prompted") is False
    if completion.get("schema_version") in {
        "mal2026-rationale-pipeline-dpo-complete-v1",
        "mal2026-rationale-pipeline-grpo-complete-v1",
    }:
        score_blind = completion.get("scores_in_policy_prompt") is False and completion.get("validation_used") is False
    need(completion.get("status") == "completed" and score_blind, "final rationale model is not score-blind completed training")
    prompt = routing()["rationale_generation_training_evaluation"]
    restricted = RESTRICTED_PARENT / args.run_id
    output = OUTPUT_PARENT / args.run_id
    need(not restricted.exists() and not output.exists(), "final rationale generation output must be fresh")
    restricted.mkdir(parents=True, mode=0o700); output.mkdir(parents=True)
    candidate = {
        "key": args.candidate_key,
        "base_model_path": str(args.base_model.resolve()),
        "adapter_path": str(args.adapter.resolve()),
        "training_completion_path": str(args.training_completion.resolve()),
    }

    processes = []
    try:
        # Two of 2,000 first-pass train generations reached the prior 2,000-token
        # capacity.  Keep prompt/sampling/schema frozen and allow only a larger
        # context/capacity retry for the final handoff.
        processes, endpoints, attestation, aliases = launch_vllm([candidate], GPUS, GEN_PORTS, output / "runtime", max_model_len=6144)
        alias = aliases[args.candidate_key]
        # One real train row is the smallest meaningful schema/prompt preflight;
        # keep the same loaded replicas for the declared full generation.
        smoke = run_generation_on_server(args.run_id, candidate, "train", 1, endpoints, attestation, alias, output / "telemetry-smoke.jsonl", GPUS)
        train = run_generation_on_server(args.run_id, candidate, "train", 2000, endpoints, attestation, alias, output / "telemetry-train.jsonl", GPUS)
        validation = run_generation_on_server(args.run_id, candidate, "validation", 400, endpoints, attestation, alias, output / "telemetry-validation.jsonl", GPUS)
    finally:
        if processes:
            stop_owned(processes); wait_released(GPUS)

    run_prefix = f"{args.run_id}-{args.candidate_key}"
    report_paths = {
        "smoke": GENERATION_AGGREGATE / f"{run_prefix}-train1" / "aggregate.json",
        "train": GENERATION_AGGREGATE / f"{run_prefix}-train2000" / "aggregate.json",
        "validation": GENERATION_AGGREGATE / f"{run_prefix}-validation400" / "aggregate.json",
    }
    for key, path in report_paths.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        expected = {"smoke": 1, "train": 2000, "validation": 400}[key]
        need(report.get("status") == "completed" and report.get("counts", {}).get("valid") == expected, f"final rationale {key} report differs")
    handoff = {
        "schema_version": "mal2026-rationale-pipeline-final-student-handoff-v1",
        "status": "completed", "run_id": args.run_id, "completed_at": now(),
        "candidate_key": args.candidate_key,
        "base_model_config_sha256": sha256_file(args.base_model / "config.json"),
        "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "training_completion_sha256": sha256_file(args.training_completion),
        "rationale_prompt_sha256": prompt["source_file_sha256"],
        "train_path": str(train.resolve()), "train_sha256": sha256_file(train), "train_records": 2000,
        "validation_path": str(validation.resolve()), "validation_sha256": sha256_file(validation), "validation_records": 400,
        "smoke_sha256": sha256_file(smoke),
        "score_conditioning": False, "human_or_reference_score_read_or_prompted": False,
        "score_output": False, "average_used": False, "gpu_scope": list(GPUS),
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights",
    }
    atomic_json(output / "aggregate.json", handoff)
    atomic_json(restricted / "manifest.json", {
        "schema_version": "mal2026-rationale-pipeline-final-student-manifest-v1",
        "status": "completed", "run_id": args.run_id,
        "train_path": handoff["train_path"], "train_sha256": handoff["train_sha256"],
        "validation_path": handoff["validation_path"], "validation_sha256": handoff["validation_sha256"],
        "aggregate_sha256": sha256_file(output / "aggregate.json"),
    })
    print(json.dumps({"status": "completed", "run_id": args.run_id, "train": 2000, "validation": 400}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
