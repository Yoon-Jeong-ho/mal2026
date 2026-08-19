#!/usr/bin/env python3
"""Generate score-blind student rationale variants with frozen tail multiplicity."""
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
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from mal2026.api_rationale_data import sha256_file  # noqa: E402
from mal2026.rationale_pipeline_prompts import routing  # noqa: E402
from run_rationale_pipeline_sft_evaluation import (  # noqa: E402
    GENERATION_AGGREGATE, GENERATION_RESTRICTED, GEN_PORTS, GPUS,
    launch_vllm, run_generation_on_server, stop_owned, wait_released,
)


OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-final-tail-generation-v1"
EXPECTED_TAIL = 2856


def need(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--candidate-key", required=True); parser.add_argument("--base-model", type=Path, required=True); parser.add_argument("--adapter", type=Path, required=True); parser.add_argument("--training-completion", type=Path, required=True); args = parser.parse_args()
    setproctitle(f"mal2026:rationale-final-tail-generation:{args.candidate_key}"[:255])
    need((args.base_model / "config.json").is_file() and (args.adapter / "adapter_model.safetensors").is_file(), "tail rationale model artifact unavailable")
    completion = json.loads(args.training_completion.read_text(encoding="utf-8")); need(completion.get("status") == "completed" and completion.get("human_or_reference_score_read_or_prompted") is False, "tail rationale model is not score-blind SFT")
    output = OUTPUT_PARENT / args.run_id; need(not output.exists(), "tail rationale output must be fresh"); output.mkdir(parents=True)
    candidate = {"key": args.candidate_key, "base_model_path": str(args.base_model.resolve()), "adapter_path": str(args.adapter.resolve()), "training_completion_path": str(args.training_completion.resolve())}
    processes = []
    try:
        processes, endpoints, attestation, aliases = launch_vllm([candidate], GPUS, GEN_PORTS, output / "runtime", max_model_len=6144); alias = aliases[args.candidate_key]
        smoke = run_generation_on_server(args.run_id, candidate, "train", 1, endpoints, attestation, alias, output / "telemetry-smoke.jsonl", GPUS)
        generated = run_generation_on_server(args.run_id, candidate, "train", EXPECTED_TAIL, endpoints, attestation, alias, output / "telemetry-tail.jsonl", GPUS, tail_multiplicity=True)
    finally:
        if processes: stop_owned(processes); wait_released(GPUS)
    prefix = f"{args.run_id}-{args.candidate_key}"
    smoke_report = GENERATION_AGGREGATE / f"{prefix}-train1" / "aggregate.json"
    tail_report = GENERATION_AGGREGATE / f"{prefix}-train{EXPECTED_TAIL}" / "aggregate.json"
    for path, expected, tail in ((smoke_report, 1, False), (tail_report, EXPECTED_TAIL, True)):
        report = json.loads(path.read_text(encoding="utf-8")); need(report.get("status") == "completed" and report.get("counts", {}).get("valid") == expected and report.get("tail_multiplicity") is tail, "tail rationale generation report differs")
    payload = {"schema_version": "mal2026-rationale-pipeline-final-tail-generation-v1", "status": "completed", "run_id": args.run_id, "completed_at": now(), "candidate_key": args.candidate_key, "records": EXPECTED_TAIL, "source_records": 2000, "multiplicity_source_counts": {"1": 1214, "2": 751, "4": 35}, "score_conditioning": False, "scores_used_only_for_offline_sampling_multiplicity": True, "sampling": {"temperature": 0.7, "top_p": 0.95, "seed_scheme": "sha256_source_variant"}, "train_path": str(generated.resolve()), "train_sha256": sha256_file(generated), "generation_report_path": str(tail_report.resolve()), "generation_report_sha256": sha256_file(tail_report), "smoke_sha256": sha256_file(smoke), "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"), "training_completion_sha256": sha256_file(args.training_completion), "rationale_prompt_sha256": routing()["rationale_generation_training_evaluation"]["source_file_sha256"], "gpu_scope": list(GPUS), "average_used": False, "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights"}
    atomic_json(output / "aggregate.json", payload); print(json.dumps({"status": "completed", "run_id": args.run_id, "records": EXPECTED_TAIL}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
