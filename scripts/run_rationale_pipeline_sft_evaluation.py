#!/usr/bin/env python3
"""Generate and exact-Q4-evaluate score-blind rationale SFT candidates."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import load_writing_rows, sha256_file  # noqa: E402
from mal2026.official_writing_contract import JUDGE_DIMENSIONS, parse_participant_output  # noqa: E402
from mal2026.rationale_pipeline_prompts import AXES, judge_participant, routing  # noqa: E402


PYTHON = ROOT / ".venv-standard/bin/python"
VLLM_WRAPPER = ROOT / "scripts/run_named_vllm.py"
GENERATOR = ROOT / "scripts/generate_rationale_pipeline_outputs_vllm.py"
Q4_EVALUATOR = ROOT / "scripts/evaluate_official_q4_rationale_fidelity.py"
JUDGE_PROMPT = ROOT / "llm_as_judge.txt"
Q4_SERVER = ROOT / "outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server"
Q4_REPO = ROOT / "outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/llama.cpp"
Q4_MODEL = ROOT / "outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
Q4_MODEL_SHA = "b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772"
Q4_SERVER_SHA = "bcb90718e997c836ead03c6808877c14dfc82926745cc7639f3e9628f53ad250"
Q4_REVISION = "571d0d540df04f25298d0e159e520d9fc62ed121"
Q4_TAG = "b10068"
Q4_PROMPT_SHA = "91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d"
RESTRICTED_PARENT = ROOT / "data/processed/restricted/rationale_pipeline_v1/evaluation"
OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-evaluation-v1"
GENERATION_RESTRICTED = ROOT / "data/processed/restricted/rationale_pipeline_v1/generation"
GENERATION_AGGREGATE = ROOT / "outputs/rationale-pipeline-generation-v1"
Q4_RESTRICTED = ROOT / "data/processed/restricted/official_rationale_fidelity_v1/q4_judge"
Q4_AGGREGATE = ROOT / "outputs/official-rationale-fidelity-v1/q4-judge"
GPUS = (0, 1, 2, 3)
GEN_PORTS = (19600, 19601, 19602, 19603)
Q4_PORTS = (19610, 19611, 19612, 19613)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gpu_state(gpus: Sequence[int]) -> list[dict[str, int]]:
    raw = subprocess.check_output([
        "nvidia-smi", f"--id={','.join(map(str, gpus))}",
        "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True)
    result = []
    for line in raw.splitlines():
        index, memory, utilization = (int(part.strip()) for part in line.split(","))
        result.append({"index": index, "memory_used_mib": memory, "utilization_percent": utilization})
    need(tuple(row["index"] for row in result) == tuple(gpus), "GPU inventory differs")
    return result


def gpu_compute_processes(gpus: Sequence[int]) -> list[str]:
    result = subprocess.run([
        "nvidia-smi", f"--id={','.join(map(str, gpus))}",
        "--query-compute-apps=pid,process_name", "--format=csv,noheader",
    ], text=True, capture_output=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def require_idle(gpus: Sequence[int]) -> list[dict[str, int]]:
    processes = gpu_compute_processes(gpus)
    state = gpu_state(gpus)
    need(not processes and all(row["utilization_percent"] == 0 and row["memory_used_mib"] <= 16 for row in state), "authorized GPU scope is not idle; existing processes were not altered")
    return state


def wait_released(gpus: Sequence[int], timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not gpu_compute_processes(gpus):
            state = gpu_state(gpus)
            if all(row["utilization_percent"] == 0 and row["memory_used_mib"] <= 16 for row in state):
                return
        time.sleep(1)
    raise RuntimeError("owned server allocations did not release")


def wait_health(processes: Sequence[subprocess.Popen[str]], endpoints: Sequence[str], timeout: int = 900) -> None:
    deadline = time.time() + timeout
    pending = set(endpoints)
    while pending and time.time() < deadline:
        for process in processes:
            need(process.poll() is None, "server exited before health gate")
        for endpoint in tuple(pending):
            try:
                with urlopen(endpoint + "/health", timeout=2) as response:
                    if response.status == 200:
                        pending.remove(endpoint)
            except Exception:
                pass
        if pending:
            time.sleep(2)
    need(not pending, f"server health timeout: {sorted(pending)}")


def stop_owned(processes: Sequence[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.time() + 60
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def telemetry(stop: threading.Event, destination: Path, gpus: Sequence[int]) -> None:
    while not stop.is_set():
        try:
            with destination.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"at": now(), "gpus": gpu_state(gpus)}, separators=(",", ":")) + "\n")
        except Exception:
            pass
        stop.wait(1)


def run_with_telemetry(command: Sequence[str], destination: Path, gpus: Sequence[int], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    stop = threading.Event()
    thread = threading.Thread(target=telemetry, args=(stop, destination, gpus), daemon=True)
    thread.start()
    try:
        return subprocess.run(command, text=True, **kwargs)
    finally:
        stop.set(); thread.join(timeout=5)


def candidate_hash(candidate: Mapping[str, Any]) -> str:
    base = Path(candidate["base_model_path"])
    adapter = Path(candidate["adapter_path"])
    completion = Path(candidate["training_completion_path"])
    value = {
        "key": candidate["key"],
        "base_model_path": str(base.resolve()),
        "base_config_sha256": sha256_file(base / "config.json"),
        "adapter_path": str(adapter.resolve()),
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "training_completion_path": str(completion.resolve()),
        "training_completion_sha256": sha256_file(completion),
    }
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    need(value.get("schema_version") == "mal2026-rationale-pipeline-sft-evaluation-v1", "evaluation config schema differs")
    need(value.get("gpu_scope") == list(GPUS) and isinstance(value.get("user_authorization"), str) and value["user_authorization"].strip(), "evaluation GPU authorization differs")
    candidates = value.get("candidates")
    need(isinstance(candidates, list) and len(candidates) >= 1, "evaluation candidates differ")
    keys: set[str] = set()
    for candidate in candidates:
        need(set(candidate) == {"key", "base_model_path", "adapter_path", "training_completion_path"}, "candidate fields differ")
        need(candidate["key"] not in keys, "candidate key duplicated"); keys.add(candidate["key"])
        base, adapter, completion = (Path(candidate[name]) for name in ("base_model_path", "adapter_path", "training_completion_path"))
        need((base / "config.json").is_file() and (adapter / "adapter_config.json").is_file() and (adapter / "adapter_model.safetensors").is_file() and completion.is_file(), f"candidate artifact unavailable: {candidate['key']}")
        complete = json.loads(completion.read_text(encoding="utf-8"))
        score_blind = complete.get("human_or_reference_score_read_or_prompted") is False
        if complete.get("schema_version") in {
            "mal2026-rationale-pipeline-dpo-complete-v1",
            "mal2026-rationale-pipeline-grpo-complete-v1",
        }:
            score_blind = complete.get("scores_in_policy_prompt") is False and complete.get("validation_used") is False
        need(complete.get("status") == "completed" and score_blind, f"candidate training completion differs: {candidate['key']}")
    need(sha256_file(JUDGE_PROMPT) == Q4_PROMPT_SHA and sha256_file(Q4_MODEL) == Q4_MODEL_SHA and sha256_file(Q4_SERVER) == Q4_SERVER_SHA, "exact judge artifact differs")
    need(subprocess.check_output(["git", "-C", str(Q4_REPO), "rev-parse", "HEAD"], text=True).strip() == Q4_REVISION, "llama revision differs")
    need(subprocess.check_output(["git", "-C", str(Q4_REPO), "describe", "--tags", "--exact-match"], text=True).strip() == Q4_TAG, "llama tag differs")
    routing()
    return value


def compatible_groups(candidates: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(Path(candidate["base_model_path"]).resolve()), []).append(candidate)
    return [sorted(group, key=lambda value: str(value["key"])) for _, group in sorted(grouped.items())]


def launch_vllm(candidates: Sequence[Mapping[str, Any]], gpus: Sequence[int], ports: Sequence[int], runtime: Path, *, max_model_len: int = 4096) -> tuple[list[subprocess.Popen[str]], list[str], Path, dict[str, str]]:
    need(bool(candidates) and len({str(Path(c["base_model_path"]).resolve()) for c in candidates}) == 1, "vLLM candidate group differs")
    need(max_model_len in {4096, 6144}, "vLLM max model length differs")
    require_idle(gpus)
    runtime.mkdir(parents=True)
    base_model = str(candidates[0]["base_model_path"])
    aliases = {str(candidate["key"]): f"mal2026-rationale-{candidate['key']}" for candidate in candidates}
    processes: list[subprocess.Popen[str]] = []
    endpoints: list[str] = []
    for gpu, port in zip(gpus, ports, strict=True):
        endpoint = f"http://127.0.0.1:{port}"; endpoints.append(endpoint)
        log = (runtime / f"vllm-gpu{gpu}.log").open("x", encoding="utf-8")
        command = [
            str(PYTHON), str(VLLM_WRAPPER), f"mal2026:vllm:rationale-eval:group-{len(candidates)}:gpu{gpu}",
            "serve", base_model, "--served-model-name", "mal2026-rationale-evaluation-base",
            "--host", "127.0.0.1", "--port", str(port), "--tensor-parallel-size", "1",
            "--dtype", "bfloat16", "--max-model-len", str(max_model_len), "--max-num-seqs", "128",
            "--max-num-batched-tokens", "32768", "--gpu-memory-utilization", "0.82",
            "--generation-config", "vllm", "--enable-prefix-caching", "--enable-lora",
            "--max-loras", "1", "--max-cpu-loras", str(len(candidates)), "--max-lora-rank", "32",
            "--lora-modules", *[f"{aliases[str(candidate['key'])]}={candidate['adapter_path']}" for candidate in candidates],
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": str(ROOT / "src"),
                # vLLM/FlashInfer may JIT a sampling kernel at first startup.
                # The pinned environment already contains ninja; expose that
                # existing binary rather than installing anything.
                "PATH": f"{ROOT / '.venv-standard/bin'}:{os.environ.get('PATH', '')}",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        log.close(); processes.append(process)
    try:
        wait_health(processes, endpoints)
    except Exception:
        stop_owned(processes); wait_released(gpus); raise
    prompt_sha = routing()["rationale_generation_training_evaluation"]["source_file_sha256"]
    attestation = runtime / "server_attestation.json"
    atomic_json(attestation, {
        "schema_version": "mal2026-rationale-pipeline-vllm-server-v1",
        "created_at": now(), "physical_gpus": list(gpus), "server_endpoints": endpoints,
        "server_pids": [process.pid for process in processes], "model_aliases": sorted(aliases.values()),
        "base_model_config_sha256": sha256_file(Path(base_model) / "config.json"),
        "adapters": {
            str(candidate["key"]): {
                "alias": aliases[str(candidate["key"])],
                "adapter_config_sha256": sha256_file(Path(candidate["adapter_path"]) / "adapter_config.json"),
                "adapter_model_sha256": sha256_file(Path(candidate["adapter_path"]) / "adapter_model.safetensors"),
            }
            for candidate in candidates
        },
        "rationale_prompt_sha256": prompt_sha, "human_or_reference_score_read_or_prompted": False,
        "max_model_len": max_model_len,
    })
    return processes, endpoints, attestation, aliases


def launch_q4(gpus: Sequence[int], ports: Sequence[int], runtime: Path) -> tuple[list[subprocess.Popen[str]], list[str], Path]:
    require_idle(gpus)
    runtime.mkdir(parents=True)
    processes: list[subprocess.Popen[str]] = []; endpoints: list[str] = []
    for gpu, port in zip(gpus, ports, strict=True):
        endpoint = f"http://127.0.0.1:{port}"; endpoints.append(endpoint)
        log = (runtime / f"llama-q4-gpu{gpu}.log").open("x", encoding="utf-8")
        command = [str(Q4_SERVER), "--model", str(Q4_MODEL), "--host", "127.0.0.1", "--port", str(port), "--n-gpu-layers", "99", "--parallel", "4", "--ctx-size", "32768", "--batch-size", "2048", "--ubatch-size", "512", "--no-webui", "--reasoning", "off"]
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        log.close(); processes.append(process)
    try:
        wait_health(processes, endpoints, timeout=300)
    except Exception:
        stop_owned(processes); wait_released(gpus); raise
    attestation = runtime / "server_attestation.json"
    atomic_json(attestation, {
        "schema_version": "mal2026-official-q4-judge-server-attestation-v1", "created_at": now(),
        "physical_gpus": list(gpus), "server_endpoints": endpoints, "server_pids": [p.pid for p in processes],
        "parallel_per_server": 4, "context_per_slot": 8192, "model_sha256": Q4_MODEL_SHA,
        "llama_server_sha256": Q4_SERVER_SHA, "llama_revision": Q4_REVISION, "llama_tag": Q4_TAG,
        "judge_prompt_sha256": Q4_PROMPT_SHA,
    })
    return processes, endpoints, attestation


def run_generation_on_server(campaign: str, candidate: Mapping[str, Any], split: str, expected: int, endpoints: Sequence[str], attestation: Path, alias: str, telemetry_path: Path, gpus: Sequence[int], *, tail_multiplicity: bool = False, multiplicity_reference: Path | None = None, multiplicity_scale: int = 1) -> Path:
    run_id = f"{campaign}-{candidate['key']}-{split}{expected}"
    # Match each vLLM replica's declared --max-num-seqs=128.  The previous
    # 32/replica client cap left the four GPUs at roughly 20--30% utilization.
    command = [str(PYTHON), str(GENERATOR), "--run-id", run_id, "--split", split, "--expected", str(expected), "--model", alias, "--server-attestation", str(attestation), "--max-inflight", str(128 * len(gpus))]
    for endpoint in endpoints:
        command.extend(("--endpoint", endpoint))
    if tail_multiplicity:
        command.append("--tail-multiplicity")
    if multiplicity_reference is not None:
        command.extend(("--multiplicity-reference", str(multiplicity_reference), "--multiplicity-scale", str(multiplicity_scale)))
    result = run_with_telemetry(command, telemetry_path, gpus, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    need(result.returncode == 0, f"generation failed: {run_id}")
    records = GENERATION_RESTRICTED / run_id / "generated_rationales.jsonl"
    report = json.loads((GENERATION_AGGREGATE / run_id / "aggregate.json").read_text(encoding="utf-8"))
    need(report.get("status") == "completed" and report.get("counts", {}).get("valid") == expected and report.get("generated_rationales_sha256") == sha256_file(records), "generation completion differs")
    return records


def participant_file(campaign: str, candidate: Mapping[str, Any], split: str, generated: Path, destination: Path) -> Path:
    rows = jsonl(generated)
    writings = {row.identifier: row for row in load_writing_rows(split, include_scores=True)}
    need(len(rows) and len(rows) <= len(writings), "participant population differs")
    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for row in rows:
            source_id = str(row["source_id"]); need(source_id in writings and row.get("rationales") is not None, "participant source differs")
            writing = writings[source_id]; need(writing.scores is not None, "canonical scores unavailable")
            participant = parse_participant_output(judge_participant(writing.scores, row["rationales"]))
            handle.write(json.dumps({"source_id": source_id, "participant_output": participant}, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(destination, 0o600)
    return destination


def run_q4(run_id: str, participant: Path, split: str, expected: int, endpoints: Sequence[str], attestation: Path, telemetry_path: Path, gpus: Sequence[int]) -> Path:
    command = [str(PYTHON), str(Q4_EVALUATOR), "--run-id", run_id, "--participant-file", str(participant), "--expected", str(expected), "--split", split, "--max-inflight", str(4 * len(endpoints)), "--server-attestation", str(attestation), "--system-prompt-file", str(JUDGE_PROMPT)]
    for endpoint in endpoints:
        command.extend(("--endpoint", endpoint))
    result = run_with_telemetry(command, telemetry_path, gpus, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    report = Q4_AGGREGATE / run_id / "aggregate_judge_report.json"
    need(result.returncode == 0 and report.is_file(), f"exact Q4 judge failed: {run_id}")
    value = json.loads(report.read_text(encoding="utf-8"))
    need(value.get("status") == "completed" and value.get("counts", {}).get("valid") == expected, f"exact Q4 judge gates failed: {run_id}")
    return report


def band_metrics(participant: Path, judge_records: Path) -> dict[str, Any]:
    participants = {row["source_id"]: row["participant_output"] for row in jsonl(participant)}
    records = jsonl(judge_records)
    values = {axis: {str(score): [] for score in range(1, 6)} for axis in AXES}
    dimension_values = {name: [] for name in JUDGE_DIMENSIONS}
    distribution: Counter[int] = Counter()
    for record in records:
        output = record.get("judge_output"); source_id = record["source_id"]
        need(output is not None and source_id in participants, "judge record linkage differs")
        for axis in AXES:
            band = str(participants[source_id][axis]["score"])
            for dimension in JUDGE_DIMENSIONS:
                score = int(output[axis][dimension]["score"])
                values[axis][band].append(score); dimension_values[dimension].append(score); distribution[score] += 1
    axis_band = {axis: {band: (statistics.fmean(scores) if scores else None) for band, scores in bands.items()} for axis, bands in values.items()}
    band = {str(score): (statistics.fmean(value for axis in AXES for value in values[axis][str(score)]) if any(values[axis][str(score)] for axis in AXES) else None) for score in range(1, 6)}
    return {
        "axis_reference_band_macro_means": axis_band,
        "reference_band_macro_means": band,
        "dimension_means": {name: statistics.fmean(scores) for name, scores in dimension_values.items()},
        "judge_score_distribution": {str(score): distribution[score] for score in range(1, 6)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    campaign = str(config["campaign"])
    setproctitle(f"mal2026:rationale-sft-evaluation:{campaign}"[:255])
    restricted = RESTRICTED_PARENT / campaign; output = OUTPUT_PARENT / campaign
    need(not restricted.exists() and not output.exists(), "evaluation campaign output must be fresh")
    restricted.mkdir(parents=True, mode=0o700); output.mkdir(parents=True)
    atomic_json(restricted / "manifest.json", {
        "schema_version": "mal2026-rationale-pipeline-sft-evaluation-campaign-v1", "status": "running",
        "campaign": campaign, "created_at": now(), "gpu_scope_authorized": list(GPUS),
        "user_authorization": config["user_authorization"], "config_sha256": sha256_file(args.config),
        "candidates": [{"key": c["key"], "identity_sha256": candidate_hash(c)} for c in config["candidates"]],
        "generation_score_blind": True, "judge_uses_canonical_half_up_scores": True, "average_used": False,
    })

    smoke_participants: dict[str, Path] = {}; validation_participants: dict[str, Path] = {}
    # Smallest meaningful real-row generation preflight for each distinct model/adapter.
    groups = compatible_groups(config["candidates"])
    for group_index, group in enumerate(groups, 1):
        processes: list[subprocess.Popen[str]] = []
        try:
            processes, endpoints, attestation, aliases = launch_vllm(group, (0,), (GEN_PORTS[0],), output / "runtime" / f"smoke-generation-group{group_index}")
            for candidate in group:
                key = str(candidate["key"])
                generated = run_generation_on_server(campaign, candidate, "train", 1, endpoints, attestation, aliases[key], output / f"telemetry-smoke-generation-{key}.jsonl", (0,))
                smoke_participants[key] = participant_file(campaign, candidate, "train", generated, restricted / "participants" / f"{key}.train-smoke1.jsonl")
        finally:
            if processes:
                stop_owned(processes); wait_released((0,))

    # One participant from each arm validates the exact judge composition before the full judge.
    q4_processes: list[subprocess.Popen[str]] = []
    try:
        q4_processes, endpoints, attestation = launch_q4((0,), (Q4_PORTS[0],), output / "runtime" / "smoke-q4")
        for candidate in config["candidates"]:
            run_q4(f"{campaign}-{candidate['key']}-train-smoke1", smoke_participants[candidate["key"]], "train", 1, endpoints, attestation, output / f"telemetry-smoke-q4-{candidate['key']}.jsonl", (0,))
    finally:
        if q4_processes:
            stop_owned(q4_processes); wait_released((0,))

    # Four independent single-GPU vLLM replicas maximize throughput for the
    # model, which fits comfortably on each authorized 80 GiB device.
    for group_index, group in enumerate(groups, 1):
        processes = []
        try:
            processes, endpoints, attestation, aliases = launch_vllm(group, GPUS, GEN_PORTS, output / "runtime" / f"full-generation-group{group_index}")
            for candidate in group:
                key = str(candidate["key"])
                generated = run_generation_on_server(campaign, candidate, "validation", 400, endpoints, attestation, aliases[key], output / f"telemetry-full-generation-{key}.jsonl", GPUS)
                validation_participants[key] = participant_file(campaign, candidate, "validation", generated, restricted / "participants" / f"{key}.validation400.jsonl")
        finally:
            if processes:
                stop_owned(processes); wait_released(GPUS)

    q4_processes = []
    evaluations: dict[str, Any] = {}
    try:
        q4_processes, endpoints, attestation = launch_q4(GPUS, Q4_PORTS, output / "runtime" / "full-q4")
        for candidate in config["candidates"]:
            key = candidate["key"]; run_id = f"{campaign}-{key}-validation400"
            report_path = run_q4(run_id, validation_participants[key], "validation", 400, endpoints, attestation, output / f"telemetry-full-q4-{key}.jsonl", GPUS)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            stratified = band_metrics(validation_participants[key], Q4_RESTRICTED / run_id / "judge_records.jsonl")
            evaluations[key] = {
                "candidate_identity_sha256": candidate_hash(candidate),
                "generation_sha256": sha256_file(GENERATION_RESTRICTED / f"{campaign}-{key}-validation400" / "generated_rationales.jsonl"),
                "participant_sha256": sha256_file(validation_participants[key]),
                "judge_report_sha256": sha256_file(report_path),
                "macro_mean": report["macro_mean"], "worst_cell_mean": report["worst_cell_mean"],
                "axis_means": report["axis_means"], "dimension_means": report["dimension_means"],
                **stratified,
            }
    finally:
        if q4_processes:
            stop_owned(q4_processes); wait_released(GPUS)

    summary = {
        "schema_version": "mal2026-rationale-pipeline-sft-evaluation-aggregate-v1", "status": "completed",
        "campaign": campaign, "completed_at": now(), "gpu_scope_authorized": list(GPUS),
        "rationale_prompt_sha256": routing()["rationale_generation_training_evaluation"]["source_file_sha256"],
        "judge_prompt_sha256": Q4_PROMPT_SHA, "judge_model_sha256": Q4_MODEL_SHA,
        "generation_score_blind": True, "judge_uses_canonical_half_up_scores": True,
        "deployment_like_score_prediction_evaluation": False, "average_used": False,
        "evaluations": evaluations,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights",
    }
    atomic_json(output / "aggregate.json", summary)
    manifest = json.loads((restricted / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"status": "completed", "completed_at": now(), "aggregate_sha256": sha256_file(output / "aggregate.json")})
    atomic_json(restricted / "manifest.json", manifest)
    print(json.dumps({"status": "completed", "campaign": campaign, "macro_means": {key: value["macro_mean"] for key, value in evaluations.items()}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
