#!/usr/bin/env python3
"""Evaluate the joint score+rationale decoder on canonical validation."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from mal2026.api_rationale_data import load_writing_rows, sha256_file  # noqa: E402
from mal2026.official_rationale_rl import q4_score  # noqa: E402
from mal2026.official_writing_contract import JUDGE_DIMENSIONS  # noqa: E402
from mal2026.rationale_pipeline_prompts import AXES, round_half_up_score  # noqa: E402
from run_rationale_pipeline_sft_evaluation import (  # noqa: E402
    GEN_PORTS,
    GPUS,
    PYTHON,
    Q4_PORTS,
    VLLM_WRAPPER,
    launch_q4,
    require_idle,
    stop_owned,
    wait_health,
    wait_released,
)
from train_rationale_pipeline_joint_decoder import EVALUATION_PROMPT, EVALUATION_PROMPT_SHA, messages  # noqa: E402


RESTRICTED_PARENT = ROOT / "data/processed/restricted/rationale_pipeline_v1/joint_decoder_evaluation"
OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-joint-decoder-evaluation-v1"
JUDGE_PROMPT = ROOT / "llm_as_judge.txt"
JUDGE_PROMPT_SHA = "91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d"


def need(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(message)


def now() -> str: return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp"); temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(path)


def schema(max_rationale_chars: int | None = None) -> dict[str, Any]:
    rationale: dict[str, Any] = {"type": "string", "minLength": 1}
    if max_rationale_chars is not None:
        need(max_rationale_chars >= 1, "joint decoder rationale bound differs")
        rationale["maxLength"] = max_rationale_chars
    axis = {"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}, "rationale": rationale}, "required": ["score", "rationale"], "additionalProperties": False}
    return {"type": "object", "properties": {name: axis for name in AXES}, "required": list(AXES), "additionalProperties": False}


def parse(value: Any) -> dict[str, dict[str, Any]]:
    try: raw = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc: raise RuntimeError("joint decoder output is not JSON") from exc
    need(isinstance(raw, dict) and set(raw) == set(AXES), "joint decoder output axes differ")
    result = {}
    for axis in AXES:
        part = raw[axis]; need(isinstance(part, dict) and set(part) == {"score", "rationale"}, "joint decoder axis fields differ")
        need(type(part["score"]) is int and 1 <= part["score"] <= 5 and isinstance(part["rationale"], str) and part["rationale"].strip(), "joint decoder axis value differs")
        result[axis] = {"score": part["score"], "rationale": part["rationale"].strip()}
    return result


def request(endpoint: str, body: Mapping[str, Any]) -> dict[str, Any]:
    wire = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(); error: Exception | None = None
    for _ in range(3):
        try:
            req = Request(endpoint.rstrip("/") + "/v1/chat/completions", data=wire, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=600) as response: return json.loads(response.read().decode())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc: error = exc
    raise RuntimeError("joint decoder HTTP request failed") from error


def generate(endpoint: str, alias: str, source_id: str, prompt: str, essay: str) -> dict[str, Any]:
    body = {"model": alias, "temperature": 0.0, "top_p": 1.0, "seed": 2026080708, "max_tokens": 2000, "chat_template_kwargs": {"enable_thinking": False}, "messages": messages(prompt, essay), "response_format": {"type": "json_schema", "json_schema": {"name": "mal2026_joint_score_rationale_v1", "strict": True, "schema": schema()}}}
    outer = request(endpoint, body); choice = outer["choices"][0]
    initial_finish_reason = choice.get("finish_reason")
    recovery_attempts: list[dict[str, Any]] = []
    if initial_finish_reason != "stop":
        # The frozen decoder occasionally enters a deterministic rationale-string
        # repetition loop.  A 400-row diagnostic reproduced 10/400 length stops;
        # a narrowly scoped retry with 1.05 repetition penalty and a 512-character
        # rationale schema stopped and parsed all 10/10.  The bound is above the
        # largest training-target rationale (322 characters), so this is an
        # integration recovery rather than a target-distribution truncation.
        need(initial_finish_reason == "length", "joint decoder unexpected finish reason differs")
        for repetition_penalty in (1.05, 1.10, 1.15):
            retry = dict(body)
            retry["repetition_penalty"] = repetition_penalty
            retry["response_format"] = {"type": "json_schema", "json_schema": {"name": "mal2026_joint_score_rationale_recovery_v1", "strict": True, "schema": schema(max_rationale_chars=512)}}
            outer = request(endpoint, retry); choice = outer["choices"][0]
            recovery_attempts.append({"repetition_penalty": repetition_penalty, "finish_reason": choice.get("finish_reason")})
            if choice.get("finish_reason") == "stop":
                break
    need(choice.get("finish_reason") == "stop", "joint decoder finish reason differs after bounded repetition recovery")
    return {"source_id": source_id, "participant_output": parse(choice["message"]["content"]), "generation_recovery": {"initial_finish_reason": initial_finish_reason, "attempts": recovery_attempts}}


def launch_policy(base: Path, adapter: Path, runtime: Path) -> tuple[list[subprocess.Popen[str]], list[str], str]:
    require_idle(GPUS); runtime.mkdir(parents=True); alias = "mal2026-joint-decoder"; processes = []; endpoints = []
    for gpu, port in zip(GPUS, GEN_PORTS, strict=True):
        endpoint = f"http://127.0.0.1:{port}"; endpoints.append(endpoint); log = (runtime / f"vllm-gpu{gpu}.log").open("x", encoding="utf-8")
        command = [str(PYTHON), str(VLLM_WRAPPER), f"mal2026:vllm:joint-decoder:gpu{gpu}", "serve", str(base), "--served-model-name", "mal2026-joint-base", "--host", "127.0.0.1", "--port", str(port), "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--max-model-len", "4096", "--max-num-seqs", "128", "--max-num-batched-tokens", "32768", "--gpu-memory-utilization", "0.82", "--generation-config", "vllm", "--enable-prefix-caching", "--enable-lora", "--max-loras", "1", "--max-cpu-loras", "1", "--max-lora-rank", "32", "--lora-modules", f"{alias}={adapter}"]
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(ROOT / "src"), "PATH": f"{ROOT / '.venv-standard/bin'}:{os.environ.get('PATH', '')}"}, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True); log.close(); processes.append(process)
    try: wait_health(processes, endpoints)
    except Exception: stop_owned(processes); wait_released(GPUS); raise
    return processes, endpoints, alias


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--base-model", type=Path, required=True); parser.add_argument("--adapter", type=Path, required=True); parser.add_argument("--training-completion", type=Path, required=True); args = parser.parse_args()
    setproctitle(f"mal2026:joint-decoder-evaluation:{args.run_id}"[:255])
    need(sha256_file(EVALUATION_PROMPT) == EVALUATION_PROMPT_SHA and sha256_file(JUDGE_PROMPT) == JUDGE_PROMPT_SHA, "joint decoder evaluation prompt differs")
    completion = json.loads(args.training_completion.read_text(encoding="utf-8")); need(completion.get("status") == "completed" and completion.get("mode") == "full" and completion.get("average_used") is False, "joint decoder training completion differs")
    need((args.base_model / "config.json").is_file() and (args.adapter / "adapter_model.safetensors").is_file(), "joint decoder artifact unavailable")
    restricted = RESTRICTED_PARENT / args.run_id; output = OUTPUT_PARENT / args.run_id; need(not restricted.exists() and not output.exists(), "joint decoder evaluation output must be fresh"); restricted.mkdir(parents=True, mode=0o700); output.mkdir(parents=True)
    writings = load_writing_rows("validation", include_scores=True); need(len(writings) == 400, "joint decoder validation population differs")
    processes = []; generated = []
    try:
        processes, endpoints, alias = launch_policy(args.base_model, args.adapter, output / "runtime/policy")
        # One real-row check runs on the loaded servers immediately before the
        # full batch; it does not require a second model startup.
        smoke = generate(endpoints[0], alias, writings[0].identifier, writings[0].prompt, writings[0].essay); need(smoke["participant_output"], "joint decoder smoke failed")
        with ThreadPoolExecutor(max_workers=128) as pool:
            futures = {pool.submit(generate, endpoints[index % 4], alias, row.identifier, row.prompt, row.essay): row.identifier for index, row in enumerate(writings)}
            for future in as_completed(futures): generated.append(future.result())
    finally:
        if processes: stop_owned(processes); wait_released(GPUS)
    generated.sort(key=lambda row: row["source_id"]); need(len(generated) == 400, "joint decoder generation population differs")
    generation_path = restricted / "predictions.validation.jsonl"
    with generation_path.open("x", encoding="utf-8") as handle:
        for row in generated: handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(generation_path, 0o600)

    by_id = {row.identifier: row for row in writings}; squared = {axis: [] for axis in AXES}; accuracies = {axis: [] for axis in AXES}; distribution = {axis: Counter() for axis in AXES}
    for row in generated:
        writing = by_id[row["source_id"]]; assert writing.scores is not None
        for axis in AXES:
            predicted = int(row["participant_output"][axis]["score"]); gold = round_half_up_score(writing.scores[axis]); squared[axis].append((predicted - gold) ** 2); accuracies[axis].append(predicted == gold); distribution[axis][predicted] += 1
    score_metrics = {axis: {"integer_rmse": math.sqrt(statistics.fmean(squared[axis])), "integer_accuracy": statistics.fmean(accuracies[axis]), "prediction_distribution": {str(score): distribution[axis][score] for score in range(1, 6)}} for axis in AXES}
    score_metrics["macro_integer_rmse"] = statistics.fmean(score_metrics[axis]["integer_rmse"] for axis in AXES); score_metrics["overall_integer_rmse"] = math.sqrt(statistics.fmean(value for axis in AXES for value in squared[axis]))

    judges = []; judged = []; system_prompt = JUDGE_PROMPT.read_text(encoding="utf-8")
    try:
        judges, endpoints, _ = launch_q4(GPUS, Q4_PORTS, output / "runtime/judge")
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(q4_score, endpoints[index % 4], "qwen36-35b-a3b-q4_k_m", by_id[row["source_id"]].prompt, by_id[row["source_id"]].essay, row["participant_output"], system_prompt=system_prompt): row["source_id"] for index, row in enumerate(generated)}
            for future in as_completed(futures): judged.append({"source_id": futures[future], "judge_output": future.result()})
    finally:
        if judges: stop_owned(judges); wait_released(GPUS)
    judged.sort(key=lambda row: row["source_id"]); need(len(judged) == 400, "joint decoder judge population differs")
    judge_path = restricted / "judge.validation.jsonl"
    with judge_path.open("x", encoding="utf-8") as handle:
        for row in judged: handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(judge_path, 0o600)
    cells = {(axis, dimension): [] for axis in AXES for dimension in JUDGE_DIMENSIONS}
    for row in judged:
        for axis in AXES:
            for dimension in JUDGE_DIMENSIONS: cells[(axis, dimension)].append(int(row["judge_output"][axis][dimension]["score"]))
    cell_means = {f"{axis}.{dimension}": statistics.fmean(values) for (axis, dimension), values in cells.items()}; judge_macro = statistics.fmean(cell_means.values()); judge_worst = min(cell_means.values())
    recovery_count = sum(bool(row["generation_recovery"]["attempts"]) for row in generated)
    recovery_penalty_counts = Counter(str(attempt["repetition_penalty"]) for row in generated for attempt in row["generation_recovery"]["attempts"] if attempt["finish_reason"] == "stop")
    report = {"schema_version": "mal2026-rationale-pipeline-joint-decoder-evaluation-v1", "status": "completed", "run_id": args.run_id, "completed_at": now(), "gpu_scope": list(GPUS), "records": 400, "score_metrics": score_metrics, "judge": {"prompt_sha256": JUDGE_PROMPT_SHA, "macro_mean": judge_macro, "worst_cell_mean": judge_worst, "cell_means": cell_means, "candidate_scores": "joint_decoder_own_predictions"}, "evaluation_prompt_sha256": EVALUATION_PROMPT_SHA, "generation_temperature": 0.0, "generation_recovery": {"bounded_repetition_retry_count": recovery_count, "primary_finish_stop_count": len(generated) - recovery_count, "successful_retry_penalty_counts": dict(sorted(recovery_penalty_counts.items())), "retry_repetition_penalties": [1.05, 1.10, 1.15], "retry_rationale_max_length_chars": 512, "diagnostic_evidence": {"primary_length_stops": "10/400", "repetition_penalty_1.05_success": "10/10 in isolated exact-row smoke", "repetition_penalty_1.10_success": "10/10 in isolated exact-row smoke"}}, "average_used": False, "training_completion_sha256": sha256_file(args.training_completion), "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"), "prediction_sha256": sha256_file(generation_path), "judge_sha256": sha256_file(judge_path), "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights"}
    atomic_json(output / "aggregate.json", report); print(json.dumps({"status": "completed", "macro_integer_rmse": score_metrics["macro_integer_rmse"], "judge_macro": judge_macro}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
