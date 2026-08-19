#!/usr/bin/env python3
"""Judge the balanced Luna/Terra rationale sample with exact Qwen3.6 Q4.

The script prepares restricted participant files, runs one real GPU4 smoke,
then uses four independent GPU4--7 llama.cpp replicas for the 30 full judge
requests.  Public output is aggregate-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from mal2026.official_writing_contract import AXES, JUDGE_DIMENSIONS, parse_participant_output  # noqa: E402


DEFAULT_CAMPAIGN = "rationale-prompt-balanced15-q4-judge-20260807-001"
DEFAULT_SOURCE_RUN = "rationale-prompt-balanced15-gpt56-20260806-001"
CAMPAIGN = DEFAULT_CAMPAIGN
SOURCE_RUN = DEFAULT_SOURCE_RUN
SOURCE_ROOT = ROOT / "data/processed/restricted/rationale_prompt_openai_test" / SOURCE_RUN
PRIVATE_ROOT = ROOT / "data/processed/restricted/rationale_prompt_q4_judge_v1" / CAMPAIGN
PARTICIPANTS = PRIVATE_ROOT / "participants"
MANIFEST = PRIVATE_ROOT / "manifest.json"
OUT = ROOT / "outputs/rationale-prompt-q4-judge-v1" / CAMPAIGN
RUNTIME = OUT / "runtime"
LEDGER = OUT / "ledger.jsonl"
PROMPT = ROOT / "llm_as_judge.txt"
PYTHON = ROOT / ".venv-standard/bin/python"
EVALUATOR = ROOT / "scripts/evaluate_official_q4_rationale_fidelity.py"
SERVER = ROOT / "outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server"
LLAMA_REPO = ROOT / "outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/llama.cpp"
MODEL = ROOT / "outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
MODEL_SHA = "b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772"
SERVER_SHA = "bcb90718e997c836ead03c6808877c14dfc82926745cc7639f3e9628f53ad250"
PROMPT_SHA = "91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d"
LLAMA_REVISION = "571d0d540df04f25298d0e159e520d9fc62ed121"
LLAMA_TAG = "b10068"
MODELS = ("gpt-5.6-luna", "gpt-5.6-terra")
GPUS = (4, 5, 6, 7)
PORTS = (19500, 19501, 19502, 19503)
SMOKE_GPU = (4,)
SMOKE_PORT = (19504,)
EXPECTED = 15
IDLE_DRIVER_MEMORY_MIB_MAX = 8
GPU_AUTHORIZATION = "2026-08-07: user explicitly requested judging the generated rationales on GPUs 4-7"


def configure(campaign: str, source_run: str, gpu_scope: Sequence[int], gpu_authorization: str) -> None:
    global CAMPAIGN, SOURCE_RUN, SOURCE_ROOT, PRIVATE_ROOT, PARTICIPANTS, MANIFEST, OUT, RUNTIME, LEDGER
    global GPUS, SMOKE_GPU, GPU_AUTHORIZATION
    for value, label in ((campaign, "campaign"), (source_run, "source run")):
        need(bool(value) and all(character.islower() or character.isdigit() or character in ".-_" for character in value), f"invalid {label}")
    CAMPAIGN = campaign
    SOURCE_RUN = source_run
    need(len(gpu_scope) == 4 and len(set(gpu_scope)) == 4 and all(0 <= gpu <= 7 for gpu in gpu_scope), "GPU scope must contain four distinct GPUs in [0,7]")
    need(bool(gpu_authorization.strip()), "GPU authorization text is required")
    GPUS = tuple(gpu_scope)
    SMOKE_GPU = (GPUS[0],)
    GPU_AUTHORIZATION = gpu_authorization.strip()
    SOURCE_ROOT = ROOT / "data/processed/restricted/rationale_prompt_openai_test" / SOURCE_RUN
    PRIVATE_ROOT = ROOT / "data/processed/restricted/rationale_prompt_q4_judge_v1" / CAMPAIGN
    PARTICIPANTS = PRIVATE_ROOT / "participants"
    MANIFEST = PRIVATE_ROOT / "manifest.json"
    OUT = ROOT / "outputs/rationale-prompt-q4-judge-v1" / CAMPAIGN
    RUNTIME = OUT / "runtime"
    LEDGER = OUT / "ledger.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def append_ledger(event: str, **values: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": now(), "event": event, **values}, ensure_ascii=False, separators=(",", ":")) + "\n")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gpu_state(gpus: Sequence[int]) -> list[dict[str, int]]:
    raw = subprocess.check_output([
        "nvidia-smi", f"--id={','.join(map(str, gpus))}",
        "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in raw.splitlines():
        index, used, util = (int(part.strip()) for part in line.split(","))
        rows.append({"index": index, "memory_used_mib": used, "utilization_percent": util})
    need(tuple(row["index"] for row in rows) == tuple(gpus), "GPU inventory differs")
    return rows


def require_idle(gpus: Sequence[int]) -> list[dict[str, int]]:
    rows = gpu_state(gpus)
    need(
        all(
            row["memory_used_mib"] <= IDLE_DRIVER_MEMORY_MIB_MAX
            and row["utilization_percent"] == 0
            for row in rows
        ),
        "authorized GPU scope is not idle; refusing to conflict",
    )
    return rows


def wait_own_servers_released(gpus: Sequence[int], timeout: int = 90) -> list[dict[str, int]]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = gpu_state(gpus)
        if all(
            row["memory_used_mib"] <= IDLE_DRIVER_MEMORY_MIB_MAX
            and row["utilization_percent"] == 0
            for row in rows
        ):
            return rows
        time.sleep(1)
    raise RuntimeError("own server GPU allocations did not release before timeout")


def prepare() -> dict[str, Any]:
    need(not PRIVATE_ROOT.exists() and not OUT.exists(), "campaign output already exists")
    sample_path = SOURCE_ROOT / "sample.jsonl"
    responses_path = SOURCE_ROOT / "responses.jsonl"
    source_manifest_path = SOURCE_ROOT / "manifest.json"
    need(sample_path.is_file() and responses_path.is_file() and source_manifest_path.is_file(), "balanced generation artifacts unavailable")
    sample = jsonl(sample_path)
    responses = jsonl(responses_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    need(source_manifest.get("status") == "completed" and len(sample) == EXPECTED and len(responses) == EXPECTED * len(MODELS), "balanced generation population differs")
    prompt_sha = source_manifest.get("prompt_sha256")
    need(isinstance(prompt_sha, str) and len(prompt_sha) == 64, "rationale generation prompt provenance differs")
    need(sha256_file(PROMPT) == PROMPT_SHA and sha256_file(SERVER) == SERVER_SHA, "judge prompt or server binary differs")
    need(sha256_file(MODEL) == MODEL_SHA, "judge model differs")
    revision = subprocess.check_output(["git", "-C", str(LLAMA_REPO), "rev-parse", "HEAD"], text=True).strip()
    tag = subprocess.check_output(["git", "-C", str(LLAMA_REPO), "describe", "--tags", "--exact-match"], text=True).strip()
    need((revision, tag) == (LLAMA_REVISION, LLAMA_TAG), "llama.cpp provenance differs")
    samples = {row["case_key"]: row for row in sample}
    need(len(samples) == EXPECTED, "balanced sample keys differ")
    PRIVATE_ROOT.mkdir(parents=True, mode=0o700)
    PARTICIPANTS.mkdir(mode=0o700)
    participant_inventory: list[dict[str, Any]] = []
    smoke_row: dict[str, Any] | None = None
    for model in MODELS:
        model_rows = sorted((row for row in responses if row["model"] == model), key=lambda row: row["case_key"])
        need(len(model_rows) == EXPECTED and all(not row["validation_errors"] for row in model_rows), f"generation output differs: {model}")
        destination = PARTICIPANTS / f"{model}.train.jsonl"
        with destination.open("x", encoding="utf-8") as handle:
            for row in model_rows:
                source = samples[row["case_key"]]
                participant = parse_participant_output({
                    axis: {"score": int(row["integer_scores"][axis]), "rationale": row["output"][axis]["rationale"]}
                    for axis in AXES
                })
                value = {"source_id": source["source_id"], "participant_output": participant}
                handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
                if model == MODELS[0] and row["case_key"] == "case-01":
                    smoke_row = value
        os.chmod(destination, 0o600)
        participant_inventory.append({"model": model, "records": EXPECTED, "path": str(destination.resolve()), "sha256": sha256_file(destination)})
    need(smoke_row is not None, "low-band smoke participant unavailable")
    smoke_path = PARTICIPANTS / "gpt-5.6-luna.case-01.train-smoke1.jsonl"
    smoke_path.write_text(json.dumps(smoke_row, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.chmod(smoke_path, 0o600)
    manifest = {
        "schema_version": "mal2026-balanced-rationale-q4-judge-campaign-v1",
        "status": "prepared",
        "campaign": CAMPAIGN,
        "created_at": now(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_run": SOURCE_RUN,
        "source_sample_sha256": sha256_file(sample_path),
        "source_responses_sha256": sha256_file(responses_path),
        "source_generation_prompt_sha256": source_manifest["prompt_sha256"],
        "judge_prompt_sha256": PROMPT_SHA,
        "judge_model": "Qwen3.6-35B-A3B Q4_K_M GGUF",
        "judge_model_sha256": MODEL_SHA,
        "llama_revision": LLAMA_REVISION,
        "llama_tag": LLAMA_TAG,
        "gpu_scope_authorized": list(GPUS),
        "user_authorization": GPU_AUTHORIZATION,
        "gpu_scope_authorization_history": [{"at": now(), "gpu_scope": list(GPUS), "authorization": GPU_AUTHORIZATION}],
        "score_track": "same Decimal ROUND_HALF_UP human-reference integer supplied to rationale generation, inserted as predicted_score",
        "human_or_reference_score_read_or_prompted": True,
        "deployment_like_emitted_score_evaluation": False,
        "sample_size_per_model": EXPECTED,
        "models": participant_inventory,
        "smoke": {"records": 1, "path": str(smoke_path.resolve()), "sha256": sha256_file(smoke_path)},
    }
    atomic_json(MANIFEST, manifest)
    OUT.mkdir(parents=True)
    append_ledger(
        "prepared",
        user_authorization=manifest["user_authorization"],
        gpu_scope=list(GPUS),
        source_responses_sha256=manifest["source_responses_sha256"],
        judge_prompt_sha256=PROMPT_SHA,
        setproctitle=f"mal2026:balanced-rationale-q4-judge:{CAMPAIGN}",
    )
    return manifest


def launch(
    gpus: Sequence[int],
    ports: Sequence[int],
    phase: str,
    *,
    parallel_per_server: int = 4,
) -> tuple[list[subprocess.Popen[str]], Path]:
    need(len(gpus) == len(ports), "server launch shape differs")
    need(parallel_per_server in {1, 4}, "server parallelism differs")
    phase_root = RUNTIME / phase
    need(not phase_root.exists(), "runtime phase must be fresh")
    logs = phase_root / "logs"
    logs.mkdir(parents=True)
    processes: list[subprocess.Popen[str]] = []
    for gpu, port in zip(gpus, ports, strict=True):
        log = (logs / f"llama-server-gpu{gpu}.log").open("x", encoding="utf-8")
        process = subprocess.Popen([
            str(SERVER), "--model", str(MODEL), "--host", "127.0.0.1", "--port", str(port),
            "--n-gpu-layers", "99", "--parallel", str(parallel_per_server), "--ctx-size", "32768",
            "--batch-size", "2048", "--ubatch-size", "512", "--no-webui", "--reasoning", "off",
        ], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        log.close()
        processes.append(process)
    for process, port in zip(processes, ports, strict=True):
        healthy = False
        for _ in range(300):
            if process.poll() is not None:
                break
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    healthy = response.status == 200
            except Exception:
                pass
            if healthy:
                break
            time.sleep(1)
        need(healthy, f"judge server health timeout on port {port}")
    attestation_path = phase_root / "server_attestation.json"
    attestation = {
        "schema_version": "mal2026-official-q4-judge-server-attestation-v1",
        "created_at": now(),
        "physical_gpus": list(gpus),
        "server_endpoints": [f"http://127.0.0.1:{port}" for port in ports],
        "server_pids": [process.pid for process in processes],
        "parallel_per_server": parallel_per_server,
        "context_per_slot": 32768 // parallel_per_server,
        "model_sha256": MODEL_SHA,
        "llama_server_sha256": SERVER_SHA,
        "llama_revision": LLAMA_REVISION,
        "llama_tag": LLAMA_TAG,
        "judge_prompt_sha256": PROMPT_SHA,
        "phase": phase,
    }
    atomic_json(attestation_path, attestation)
    append_ledger("servers_ready", phase=phase, gpus=list(gpus), ports=list(ports), pids=attestation["server_pids"])
    return processes, attestation_path


def stop(processes: Sequence[subprocess.Popen[str]], phase: str) -> None:
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.time() + 30
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    append_ledger("servers_stopped", phase=phase, returncodes=[process.returncode for process in processes])


def evaluator_paths(run_id: str) -> tuple[Path, Path]:
    restricted = ROOT / "data/processed/restricted/official_rationale_fidelity_v1/q4_judge" / run_id
    aggregate = ROOT / "outputs/official-rationale-fidelity-v1/q4-judge" / run_id / "aggregate_judge_report.json"
    return restricted, aggregate


def evaluate(run_id: str, participant_file: Path, expected: int, endpoints: Sequence[str], attestation: Path, *, split: str = "train") -> dict[str, Any]:
    need(split in {"train", "validation"}, "judge split differs")
    restricted, aggregate = evaluator_paths(run_id)
    need(not restricted.exists() and not aggregate.parent.exists(), f"judge output must be fresh: {run_id}")
    command = [
        str(PYTHON), str(EVALUATOR), "--run-id", run_id, "--split", split,
        "--participant-file", str(participant_file), "--expected", str(expected),
        "--max-inflight", str(4 * len(endpoints)), "--server-attestation", str(attestation),
        "--system-prompt-file", str(PROMPT),
    ]
    for endpoint in endpoints:
        command.extend(("--endpoint", endpoint))
    log_path = OUT / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    append_ledger("judge_started", run_id=run_id, split=split, expected=expected, participant_sha256=sha256_file(participant_file))
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, stdout=log, stderr=subprocess.STDOUT, text=True)
    need(result.returncode == 0, f"judge failed: {run_id}; inspect {log_path}")
    report = json.loads(aggregate.read_text(encoding="utf-8"))
    need(report.get("status") == "completed", f"judge gates failed: {run_id}")
    append_ledger("judge_completed", run_id=run_id, macro_mean=report["macro_mean"], worst_cell_mean=report["worst_cell_mean"])
    return report


def telemetry(stop_event: threading.Event, phase: str) -> None:
    path = RUNTIME / phase / "gpu_telemetry.jsonl"
    while not stop_event.is_set():
        try:
            rows = gpu_state(GPUS)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"at": now(), "gpus": rows}, separators=(",", ":")) + "\n")
        except Exception as exc:
            append_ledger("telemetry_error", phase=phase, error=type(exc).__name__)
        stop_event.wait(0.5)


def stratified_metrics(participant_file: Path, judge_records: Path) -> dict[str, Any]:
    participants = {row["source_id"]: row["participant_output"] for row in jsonl(participant_file)}
    records = jsonl(judge_records)
    values: dict[str, dict[str, list[int]]] = {axis: {str(score): [] for score in range(1, 6)} for axis in AXES}
    dimensions: dict[str, list[int]] = {dimension: [] for dimension in JUDGE_DIMENSIONS}
    distribution: Counter[int] = Counter()
    for record in records:
        output = record.get("judge_output")
        need(output is not None and record["source_id"] in participants, "stratified judge input differs")
        participant = participants[record["source_id"]]
        for axis in AXES:
            band = str(participant[axis]["score"])
            for dimension in JUDGE_DIMENSIONS:
                score = int(output[axis][dimension]["score"])
                values[axis][band].append(score)
                dimensions[dimension].append(score)
                distribution[score] += 1
    axis_band_mean = {
        axis: {band: statistics.fmean(scores) for band, scores in bands.items()}
        for axis, bands in values.items()
    }
    band_mean = {
        str(band): statistics.fmean(score for axis in AXES for score in values[axis][str(band)])
        for band in range(1, 6)
    }
    return {
        "axis_reference_band_means": axis_band_mean,
        "reference_band_macro_means": band_mean,
        "judge_score_distribution": {str(score): distribution[score] for score in range(1, 6)},
        "dimension_means_recomputed": {dimension: statistics.fmean(scores) for dimension, scores in dimensions.items()},
    }


def telemetry_summary(path: Path) -> dict[str, Any]:
    rows = jsonl(path)
    return {
        "samples": len(rows),
        "per_gpu_peak_memory_mib": {
            str(gpu): max((item["memory_used_mib"] for row in rows for item in row["gpus"] if item["index"] == gpu), default=None)
            for gpu in GPUS
        },
        "per_gpu_peak_utilization_percent": {
            str(gpu): max((item["utilization_percent"] for row in rows for item in row["gpus"] if item["index"] == gpu), default=None)
            for gpu in GPUS
        },
    }


def run() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    need(manifest.get("status") == "prepared", "campaign manifest differs")
    if manifest.get("gpu_scope_authorized") != list(GPUS):
        previous_scope = manifest.get("gpu_scope_authorized")
        history = manifest.get("gpu_scope_authorization_history")
        if not isinstance(history, list):
            history = [{"at": manifest.get("created_at"), "gpu_scope": previous_scope, "authorization": manifest.get("user_authorization")}]
        history.append({"at": now(), "gpu_scope": list(GPUS), "authorization": GPU_AUTHORIZATION})
        manifest["gpu_scope_authorization_history"] = history
        manifest["gpu_scope_authorized"] = list(GPUS)
        manifest["user_authorization"] = GPU_AUTHORIZATION
        atomic_json(MANIFEST, manifest)
        append_ledger("gpu_scope_reauthorized", previous_scope=previous_scope, gpu_scope=list(GPUS), user_authorization=GPU_AUTHORIZATION)
    initial = require_idle(GPUS)
    append_ledger("authorized_preflight", gpu_scope=list(GPUS), gpu_state=initial, user_authorization=manifest["user_authorization"])

    smoke_processes: list[subprocess.Popen[str]] = []
    try:
        smoke_phase = f"smoke-gpu{SMOKE_GPU[0]}"
        smoke_processes, smoke_attestation = launch(SMOKE_GPU, SMOKE_PORT, smoke_phase)
        evaluate(f"{CAMPAIGN}-smoke-train1", Path(manifest["smoke"]["path"]), 1, [f"http://127.0.0.1:{SMOKE_PORT[0]}"], smoke_attestation)
    finally:
        if smoke_processes:
            stop(smoke_processes, smoke_phase)
    after_smoke = wait_own_servers_released(GPUS)
    append_ledger("smoke_passed_full_preflight", gpu_scope=list(GPUS), gpu_state=after_smoke)

    full_processes: list[subprocess.Popen[str]] = []
    reports: dict[str, Any] = {}
    phase = "full-gpu" + "-".join(map(str, GPUS))
    monitor_stop = threading.Event()
    monitor: threading.Thread | None = None
    try:
        full_processes, full_attestation = launch(GPUS, PORTS, phase)
        monitor = threading.Thread(target=telemetry, args=(monitor_stop, phase), daemon=True)
        monitor.start()
        endpoints = [f"http://127.0.0.1:{port}" for port in PORTS]
        for item in manifest["models"]:
            key = item["model"].replace("gpt-5.6-", "")
            run_id = f"{CAMPAIGN}-{key}"
            reports[item["model"]] = evaluate(run_id, Path(item["path"]), EXPECTED, endpoints, full_attestation)
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=5)
        if full_processes:
            stop(full_processes, phase)
    final_gpu_state = wait_own_servers_released(GPUS)
    models: dict[str, Any] = {}
    for item in manifest["models"]:
        model = item["model"]
        key = model.replace("gpt-5.6-", "")
        restricted, _ = evaluator_paths(f"{CAMPAIGN}-{key}")
        report = reports[model]
        models[model] = {
            "records": EXPECTED,
            "judge_cells": report["counts"]["judge_cells"],
            "macro_mean": report["macro_mean"],
            "worst_cell_mean": report["worst_cell_mean"],
            "axis_means": report["axis_means"],
            "dimension_means": report["dimension_means"],
            "cell_means": report["cell_means"],
            "score_1_or_2_rate": report["score_1_or_2_rate"],
            **stratified_metrics(Path(item["path"]), restricted / "judge_records.jsonl"),
        }
    summary = {
        "schema_version": "mal2026-balanced-rationale-q4-judge-summary-v1",
        "status": "completed",
        "campaign": CAMPAIGN,
        "completed_at": now(),
        "gpu_scope": list(GPUS),
        "judge_model": manifest["judge_model"],
        "judge_model_sha256": MODEL_SHA,
        "judge_prompt_sha256": PROMPT_SHA,
        "source_generation_prompt_sha256": manifest["source_generation_prompt_sha256"],
        "sample_size_per_model": EXPECTED,
        "score_track": manifest["score_track"],
        "human_or_reference_score_read_or_prompted": True,
        "deployment_like_emitted_score_evaluation": False,
        "models": models,
        "telemetry": telemetry_summary(RUNTIME / phase / "gpu_telemetry.jsonl"),
        "final_gpu_state": final_gpu_state,
        "caveat": "Judge scores measure rationale fidelity conditional on the supplied integerized human/reference score; they do not evaluate score prediction or downstream RMSE.",
        "privacy": "aggregate_only_no_source_ids_prompts_essays_rationales_or_judge_evidence",
    }
    atomic_json(OUT / "aggregate_summary.json", summary)
    manifest["status"] = "completed"
    manifest["completed_at"] = now()
    manifest["aggregate_summary_sha256"] = sha256_file(OUT / "aggregate_summary.json")
    atomic_json(MANIFEST, manifest)
    append_ledger("campaign_completed", aggregate_summary_sha256=manifest["aggregate_summary_sha256"], final_gpu_state=final_gpu_state)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "all"))
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--gpu-scope", default="4,5,6,7")
    parser.add_argument("--gpu-authorization", default="2026-08-07: user explicitly requested judging the generated rationales on GPUs 4-7")
    args = parser.parse_args()
    try:
        gpu_scope = tuple(int(value.strip()) for value in args.gpu_scope.split(","))
    except ValueError as exc:
        raise RuntimeError("GPU scope must be comma-separated integers") from exc
    configure(args.campaign, args.source_run, gpu_scope, args.gpu_authorization)
    setproctitle(f"mal2026:balanced-rationale-q4-judge:{args.command}:{CAMPAIGN}"[:255])
    if args.command in {"prepare", "all"}:
        manifest = prepare()
        print(json.dumps({"status": manifest["status"], "campaign": CAMPAIGN, "gpu_scope": manifest["gpu_scope_authorized"]}, sort_keys=True), flush=True)
    if args.command in {"run", "all"}:
        summary = run()
        print(json.dumps({"status": summary["status"], "campaign": CAMPAIGN, "models": {model: row["macro_mean"] for model, row in summary["models"].items()}}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
