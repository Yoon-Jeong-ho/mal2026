#!/usr/bin/env python3
"""Measure train-only exact-Q4 reward variance before score-blind GRPO.

The policy never receives a score.  Canonical per-axis scores are attached
only after generation, when composing the exact ``llm_as_judge.txt`` request.
Row-level samples and judgments remain under the restricted data root.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mal2026.api_rationale_data import load_writing_rows, sha256_file  # noqa: E402
from mal2026.official_rationale_rl import judge_total, q4_score  # noqa: E402
from mal2026.rationale_pipeline_prompts import (  # noqa: E402
    AXES,
    judge_participant,
    rationale_messages,
    rationale_output,
    routing,
)
from generate_rationale_pipeline_outputs_vllm import schema  # noqa: E402
from run_rationale_pipeline_sft_evaluation import (  # noqa: E402
    GPUS,
    GEN_PORTS,
    Q4_PORTS,
    launch_q4,
    launch_vllm,
    stop_owned,
    wait_released,
)


RESTRICTED_PARENT = ROOT / "data/processed/restricted/rationale_pipeline_v1/grpo_variance"
OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-grpo-variance-v1"
JUDGE_PROMPT = ROOT / "llm_as_judge.txt"
EXPECTED_JUDGE_SHA = "91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d"
POLICY_SEED = 2026080705


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_json(endpoint: str, body: Mapping[str, Any]) -> dict[str, Any]:
    wire = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    error: Exception | None = None
    for _ in range(3):
        try:
            request = Request(endpoint.rstrip("/") + "/v1/chat/completions", data=wire, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=600) as response:
                value = json.loads(response.read().decode())
            need(isinstance(value, dict), "GRPO variance rollout envelope differs")
            return value
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            error = exc
    raise RuntimeError("GRPO variance rollout transport failed") from error


def sample_one(endpoint: str, alias: str, source_id: str, prompt: str, essay: str, count: int) -> dict[str, Any]:
    source_seed = (POLICY_SEED ^ int.from_bytes(sha256(source_id.encode()).digest()[:4], "big")) % (2**31 - 1)
    body = {
        "model": alias,
        "temperature": 0.7,
        "top_p": 0.95,
        "seed": source_seed,
        "n": count,
        "max_tokens": 2000,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": rationale_messages(prompt, essay),
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "mal2026_grpo_variance_bundle_v1", "strict": True, "schema": schema()},
        },
    }
    outer = request_json(endpoint, body)
    choices = outer.get("choices")
    need(isinstance(choices, list) and len(choices) == count, "GRPO variance completion count differs")
    samples: list[dict[str, str]] = []
    finishes: list[str] = []
    for choice in choices:
        need(isinstance(choice, dict), "GRPO variance choice differs")
        finish = str(choice.get("finish_reason")); finishes.append(finish)
        need(finish == "stop", "GRPO variance completion did not stop")
        message = choice.get("message")
        need(isinstance(message, dict), "GRPO variance message differs")
        parsed = rationale_output(message.get("content"))
        samples.append({axis: str(parsed[axis]["rationale"]) for axis in AXES})
    return {"source_id": source_id, "rationales": samples, "finish_reasons": finishes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--training-completion", type=Path, required=True)
    parser.add_argument("--sources", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()
    setproctitle(f"mal2026:rationale-grpo-variance:{args.run_id}"[:255])
    need(args.sources == 128 and args.group_size == 4, "GRPO variance sample contract differs")
    need(sha256_file(JUDGE_PROMPT) == EXPECTED_JUDGE_SHA, "exact judge prompt differs")
    need((args.base_model / "config.json").is_file() and (args.adapter / "adapter_model.safetensors").is_file(), "GRPO variance policy artifact unavailable")
    completion = json.loads(args.training_completion.read_text(encoding="utf-8"))
    score_blind = completion.get("human_or_reference_score_read_or_prompted") is False
    if completion.get("schema_version") == "mal2026-rationale-pipeline-dpo-complete-v1":
        score_blind = completion.get("scores_in_policy_prompt") is False and completion.get("validation_used") is False
    need(completion.get("status") == "completed" and score_blind, "GRPO variance policy completion differs")
    routing()

    restricted = RESTRICTED_PARENT / args.run_id
    output = OUTPUT_PARENT / args.run_id
    need(not restricted.exists() and not output.exists(), "GRPO variance output must be fresh")
    restricted.mkdir(parents=True, mode=0o700); output.mkdir(parents=True)
    writings = load_writing_rows("train", include_scores=True)
    # A deterministic train-distribution sample; no validation row is loaded.
    selected = sorted(writings, key=lambda row: sha256(f"{POLICY_SEED}:{row.identifier}".encode()).hexdigest())[:args.sources]
    need(len(selected) == args.sources and all(row.scores is not None for row in selected), "GRPO variance train sample differs")

    candidate = {
        "key": "warmstart-policy",
        "base_model_path": str(args.base_model.resolve()),
        "adapter_path": str(args.adapter.resolve()),
        "training_completion_path": str(args.training_completion.resolve()),
    }
    policy_processes = []
    rollout_rows: list[dict[str, Any]] = []
    try:
        policy_processes, endpoints, _, aliases = launch_vllm([candidate], GPUS, GEN_PORTS, output / "runtime/policy")
        alias = aliases["warmstart-policy"]
        with ThreadPoolExecutor(max_workers=32 * len(endpoints)) as pool:
            futures = {
                pool.submit(sample_one, endpoints[index % len(endpoints)], alias, row.identifier, row.prompt, row.essay, args.group_size): row.identifier
                for index, row in enumerate(selected)
            }
            for future in as_completed(futures):
                rollout_rows.append(future.result())
    finally:
        if policy_processes:
            stop_owned(policy_processes); wait_released(GPUS)
    rollout_rows.sort(key=lambda row: str(row["source_id"]))
    need(len(rollout_rows) == args.sources and all(len(row["rationales"]) == args.group_size for row in rollout_rows), "GRPO variance rollout population differs")

    by_id = {row.identifier: row for row in selected}
    judge_processes = []
    judged: list[dict[str, Any]] = []
    system_prompt = JUDGE_PROMPT.read_text(encoding="utf-8")
    try:
        judge_processes, endpoints, _ = launch_q4(GPUS, Q4_PORTS, output / "runtime/judge")
        tasks: list[tuple[str, int, str, str, Mapping[str, Any]]] = []
        for row in rollout_rows:
            source = by_id[str(row["source_id"])]
            assert source.scores is not None
            for sample_index, rationales in enumerate(row["rationales"]):
                tasks.append((source.identifier, sample_index, source.prompt, source.essay, judge_participant(source.scores, rationales)))
        with ThreadPoolExecutor(max_workers=4 * len(endpoints)) as pool:
            futures = {
                pool.submit(q4_score, endpoints[index % len(endpoints)], "qwen36-35b-a3b-q4_k_m", prompt, essay, participant, system_prompt=system_prompt): (source_id, sample_index)
                for index, (source_id, sample_index, prompt, essay, participant) in enumerate(tasks)
            }
            for future in as_completed(futures):
                source_id, sample_index = futures[future]
                judge = future.result()
                judged.append({"source_id": source_id, "sample_index": sample_index, "judge_output": judge, "reward": judge_total(judge) / 12.0})
    finally:
        if judge_processes:
            stop_owned(judge_processes); wait_released(GPUS)
    judged.sort(key=lambda row: (str(row["source_id"]), int(row["sample_index"])))
    need(len(judged) == args.sources * args.group_size, "GRPO variance judge population differs")

    rollout_path = restricted / "rollouts.train.jsonl"
    judge_path = restricted / "judgments.train.jsonl"
    with rollout_path.open("x", encoding="utf-8") as handle:
        for row in rollout_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with judge_path.open("x", encoding="utf-8") as handle:
        for row in judged:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(rollout_path, 0o600); os.chmod(judge_path, 0o600)

    grouped: dict[str, list[float]] = {}
    distribution: Counter[str] = Counter()
    for row in judged:
        grouped.setdefault(str(row["source_id"]), []).append(float(row["reward"]))
        distribution[f"{float(row['reward']):.6f}"] += 1
    zero_variance = sum(math.isclose(statistics.pstdev(values), 0.0, abs_tol=1e-12) for values in grouped.values())
    zero_fraction = zero_variance / len(grouped)
    reward_values = [value for values in grouped.values() for value in values]
    gates = {
        "all_rollouts_parse_valid": len(rollout_rows) == args.sources,
        "all_exact_q4_judgments_valid": len(judged) == args.sources * args.group_size,
        "zero_variance_group_fraction_lte_0_8": zero_fraction <= 0.8,
    }
    report = {
        "schema_version": "mal2026-rationale-pipeline-grpo-variance-v1",
        "status": "passed" if all(gates.values()) else "failed_gates",
        "run_id": args.run_id, "completed_at": now(), "split": "train",
        "sources": args.sources, "group_size": args.group_size, "completions": len(judged),
        "sampling": {"temperature": 0.7, "top_p": 0.95, "seed_basis": POLICY_SEED},
        "reward_projection": "exact_Q4_sum_of_12_integer_cells_divided_by_12",
        "reward_mean": statistics.fmean(reward_values),
        "reward_std": statistics.pstdev(reward_values),
        "zero_variance_groups": zero_variance,
        "zero_variance_group_fraction": zero_fraction,
        "threshold_frozen_before_results": 0.8,
        "reward_distribution": dict(sorted(distribution.items())),
        "hard_gates": gates,
        "base_model_config_sha256": sha256_file(args.base_model / "config.json"),
        "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "training_completion_sha256": sha256_file(args.training_completion),
        "rationale_prompt_sha256": routing()["rationale_generation_training_evaluation"]["source_file_sha256"],
        "judge_prompt_sha256": EXPECTED_JUDGE_SHA,
        "scores_in_policy_prompt": False, "canonical_scores_attached_only_to_judge": True,
        "validation_used": False, "average_used": False,
        "rollout_sha256": sha256_file(rollout_path), "judgment_sha256": sha256_file(judge_path),
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights",
    }
    atomic_json(output / "aggregate.json", report)
    atomic_json(restricted / "manifest.json", {
        "schema_version": "mal2026-rationale-pipeline-grpo-variance-manifest-v1",
        "status": report["status"], "run_id": args.run_id,
        "rollout_sha256": report["rollout_sha256"], "judgment_sha256": report["judgment_sha256"],
        "aggregate_sha256": sha256_file(output / "aggregate.json"),
    })
    print(json.dumps({"status": report["status"], "zero_variance_group_fraction": zero_fraction, "reward_mean": report["reward_mean"]}, sort_keys=True), flush=True)
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
