#!/usr/bin/env python3
"""Run one smoke then the full historical rationale-fidelity matrix on GPUs 4--7."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "official-rationale-fidelity-q4-v1-20260806-001"
PRIVATE = ROOT / "data/processed/restricted/official_rationale_fidelity_v1" / CAMPAIGN / "manifest.json"
OUT = ROOT / "outputs/official-rationale-fidelity-v1" / CAMPAIGN
RUNTIME = OUT / "runtime"
LEDGER = OUT / "ledger.jsonl"
PROMPT = ROOT / "llm_as_judge.txt"
PYTHON = ROOT / ".venv-standard/bin/python"
EVALUATOR = ROOT / "scripts/evaluate_official_q4_rationale_fidelity.py"
SERVER = ROOT / "outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server"
LLAMA_REPO = ROOT / "outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/llama.cpp"
MODEL = ROOT / "outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
MODEL_SHA = "b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772"
REVISION = "571d0d540df04f25298d0e159e520d9fc62ed121"
TAG = "b10068"
PROMPT_SHA = "91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d"
FULL_GPUS = (4, 5, 6, 7)
FULL_PORTS = (19400, 19401, 19402, 19403)
SMOKE_GPU = (4,)
SMOKE_PORT = (19404,)


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


def append_ledger(event: str, **values: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": now(), "event": event, **values}, ensure_ascii=False, separators=(",", ":")) + "\n")


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
    need(all(row["memory_used_mib"] == 0 and row["utilization_percent"] == 0 for row in rows), "authorized GPU scope is not idle; refusing to conflict")
    return rows


def wait_until_idle(gpus: Sequence[int], poll_seconds: int = 15, stable_seconds: int = 300) -> list[dict[str, int]]:
    """Wait read-only for a stable idle window; never alter an observed process."""
    announced = False
    idle_since: float | None = None
    while True:
        rows = gpu_state(gpus)
        if all(row["memory_used_mib"] == 0 and row["utilization_percent"] == 0 for row in rows):
            idle_since = idle_since or time.monotonic()
            if time.monotonic() - idle_since >= stable_seconds:
                append_ledger(
                    "gpu_scope_stably_idle",
                    gpu_scope=list(gpus), gpu_state=rows, stable_seconds=stable_seconds,
                    action="read_only_observation",
                )
                return rows
        else:
            idle_since = None
        if idle_since is None and not announced:
            append_ledger(
                "gpu_conflict_waiting_read_only",
                gpu_scope=list(gpus),
                gpu_state=rows,
                required_continuous_idle_seconds=stable_seconds,
                action="wait_only_never_terminate_displace_or_modify_observed_processes",
            )
            announced = True
        time.sleep(poll_seconds)


def verify_inputs() -> dict[str, Any]:
    need(PRIVATE.is_file() and PYTHON.is_file() and SERVER.is_file() and os.access(SERVER, os.X_OK) and MODEL.is_file() and PROMPT.is_file(), "batch prerequisites unavailable")
    manifest = json.loads(PRIVATE.read_text(encoding="utf-8"))
    need(manifest.get("status") == "prepared" and manifest.get("gpu_scope_authorized") == list(FULL_GPUS), "campaign manifest differs")
    need(len(manifest.get("participants", [])) == 40 and manifest.get("human_or_reference_score_read") is True, "campaign participant inventory differs")
    need(sha256_file(MODEL) == MODEL_SHA and sha256_file(PROMPT) == PROMPT_SHA, "pinned model or prompt checksum differs")
    revision = subprocess.check_output(["git", "-C", str(LLAMA_REPO), "rev-parse", "HEAD"], text=True).strip()
    tag = subprocess.check_output(["git", "-C", str(LLAMA_REPO), "describe", "--tags", "--exact-match"], text=True).strip()
    need(revision == REVISION and tag == TAG, "pinned llama.cpp revision differs")
    return manifest


def launch(gpus: Sequence[int], ports: Sequence[int], phase: str) -> tuple[list[subprocess.Popen[str]], Path]:
    need(len(gpus) == len(ports), "server launch shape differs")
    phase_root = RUNTIME / phase
    need(not phase_root.exists(), f"fresh runtime directory required: {phase}")
    logs = phase_root / "logs"
    logs.mkdir(parents=True)
    processes: list[subprocess.Popen[str]] = []
    for gpu, port in zip(gpus, ports, strict=True):
        log = (logs / f"llama-server-gpu{gpu}.log").open("x", encoding="utf-8")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        process = subprocess.Popen([
            str(SERVER), "--model", str(MODEL), "--host", "127.0.0.1", "--port", str(port),
            "--n-gpu-layers", "99", "--parallel", "4", "--ctx-size", "32768",
            "--batch-size", "2048", "--ubatch-size", "512", "--no-webui", "--reasoning", "off",
        ], env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
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
        "parallel_per_server": 4,
        "context_per_slot": 8192,
        "model_sha256": MODEL_SHA,
        "llama_server_sha256": sha256_file(SERVER),
        "llama_revision": REVISION,
        "llama_tag": TAG,
        "judge_prompt_sha256": PROMPT_SHA,
        "phase": phase,
    }
    attestation_path.write_text(json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_ledger("servers_ready", phase=phase, gpus=list(gpus), ports=list(ports), pids=attestation["server_pids"])
    return processes, attestation_path


def next_phase(prefix: str) -> str:
    for attempt in range(1, 100):
        value = f"{prefix}-attempt{attempt}"
        if not (RUNTIME / value).exists():
            return value
    raise RuntimeError(f"runtime attempt namespace exhausted: {prefix}")


def stop(processes: Sequence[subprocess.Popen[str]], phase: str) -> None:
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.time() + 30
    for process in processes:
        remaining = max(0.0, deadline - time.time())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    append_ledger("servers_stopped", phase=phase, returncodes=[process.returncode for process in processes])


def evaluate(run_id: str, split: str, participant_file: str, expected: int, endpoints: Sequence[str], attestation: Path, log_path: Path) -> None:
    restricted_result = ROOT / "data/processed/restricted/official_rationale_fidelity_v1/q4_judge" / run_id
    aggregate_result = ROOT / "outputs/official-rationale-fidelity-v1/q4-judge" / run_id / "aggregate_judge_report.json"
    if restricted_result.is_dir() and aggregate_result.is_file():
        manifest = json.loads((restricted_result / "manifest.json").read_text(encoding="utf-8"))
        report = json.loads(aggregate_result.read_text(encoding="utf-8"))
        need(manifest.get("status") == "completed" and report.get("status") == "completed", f"cannot resume incomplete run: {run_id}")
        append_ledger("judge_skipped_completed", run_id=run_id)
        return
    need(not restricted_result.exists() and not aggregate_result.parent.exists(), f"stale partial judge output blocks run: {run_id}")
    command = [
        str(PYTHON), str(EVALUATOR), "--run-id", run_id, "--split", split,
        "--participant-file", participant_file, "--expected", str(expected),
        "--max-inflight", str(4 * len(endpoints)), "--server-attestation", str(attestation),
        "--system-prompt-file", str(PROMPT),
    ]
    for endpoint in endpoints:
        command.extend(("--endpoint", endpoint))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    append_ledger("judge_started", run_id=run_id, split=split, expected=expected)
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, stdout=log, stderr=subprocess.STDOUT, text=True)
    need(result.returncode == 0, f"judge failed: {run_id}; inspect {log_path}")
    report = json.loads(aggregate_result.read_text(encoding="utf-8"))
    append_ledger("judge_completed", run_id=run_id, macro_mean=report["macro_mean"], worst_cell_mean=report["worst_cell_mean"])


def completed_run_id(base_run_id: str) -> str | None:
    candidates = [base_run_id, *(f"{base_run_id}-retry{index}" for index in range(1, 100))]
    for run_id in candidates:
        restricted = ROOT / "data/processed/restricted/official_rationale_fidelity_v1/q4_judge" / run_id
        report_path = ROOT / "outputs/official-rationale-fidelity-v1/q4-judge" / run_id / "aggregate_judge_report.json"
        if restricted.is_dir() and report_path.is_file():
            manifest = json.loads((restricted / "manifest.json").read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "completed" and report.get("status") == "completed":
                return run_id
    return None


def next_run_id(base_run_id: str) -> str:
    if completed_run_id(base_run_id) is not None:
        return completed_run_id(base_run_id) or base_run_id
    candidates = [base_run_id, *(f"{base_run_id}-retry{index}" for index in range(1, 100))]
    for run_id in candidates:
        restricted = ROOT / "data/processed/restricted/official_rationale_fidelity_v1/q4_judge" / run_id
        aggregate = ROOT / "outputs/official-rationale-fidelity-v1/q4-judge" / run_id
        if not restricted.exists() and not aggregate.exists():
            return run_id
    raise RuntimeError(f"judge retry namespace exhausted: {base_run_id}")


def aggregate(manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for item in manifest["participants"]:
        base_run_id = f"{CAMPAIGN}-{item['key']}"
        run_id = completed_run_id(base_run_id)
        need(run_id is not None, f"completed judge run unavailable: {base_run_id}")
        path = ROOT / "outputs/official-rationale-fidelity-v1/q4-judge" / run_id / "aggregate_judge_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        need(report.get("status") == "completed" and report.get("judge_system_prompt_sha256") == PROMPT_SHA, f"aggregate input differs: {run_id}")
        rows.append({
            "key": item["key"], "family": item["family"], "macro_mean": report["macro_mean"],
            "worst_cell_mean": report["worst_cell_mean"], "score_1_or_2_rate": report["score_1_or_2_rate"],
            "axis_means": report["axis_means"], "dimension_means": report["dimension_means"],
            "aggregate_report_sha256": sha256_file(path),
        })
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    summary = {
        "schema_version": "mal2026-official-rationale-fidelity-summary-v1",
        "status": "completed",
        "campaign": CAMPAIGN,
        "completed_at": now(),
        "gpu_scope": list(FULL_GPUS),
        "judge_prompt_sha256": PROMPT_SHA,
        "model_sha256": MODEL_SHA,
        "score_track": "canonical_human_reference_integerized_half_up_for_rationale_fidelity_only",
        "human_or_reference_score_read_or_prompted": True,
        "deployment_like_emitted_score_evaluation": False,
        "counts": {"participants": len(rows), "rows_per_participant": 400, "judge_requests": len(rows) * 400},
        "family_macro_means": {family: sum(float(row["macro_mean"]) for row in values) / len(values) for family, values in sorted(families.items())},
        "best_participant": max(rows, key=lambda row: float(row["macro_mean"])),
        "worst_participant": min(rows, key=lambda row: float(row["macro_mean"])),
        "participants": sorted(rows, key=lambda row: float(row["macro_mean"]), reverse=True),
        "caveat": "This isolates rationale fidelity to the integerized human/reference target. It is not an emitted-score end-to-end estimate and validation was previously exposed.",
        "privacy": "aggregate_only_no_ids_prompts_essays_rationales_evidence_scores_or_predictions",
    }
    path = OUT / "aggregate_summary.json"
    need(not path.exists(), "campaign aggregate summary must be fresh")
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    setproctitle("mal2026:official-rationale-fidelity-q4-gpu4-7")
    manifest = verify_inputs()
    initial = wait_until_idle(FULL_GPUS)
    append_ledger(
        "authorized_preflight",
        named_task=CAMPAIGN,
        user_authorization="2026-08-06 user explicitly requested GPUs 4-7 for OpenAI and trained-rationale exact official judge evaluation",
        gpu_scope=list(FULL_GPUS),
        gpu_state=initial,
        setproctitle="mal2026:official-rationale-fidelity-q4-gpu4-7",
    )

    smoke_base = f"{CAMPAIGN}-smoke-train1"
    if completed_run_id(smoke_base) is None:
        smoke_processes: list[subprocess.Popen[str]] = []
        smoke_phase = next_phase("smoke-gpu4")
        try:
            smoke_processes, smoke_attestation = launch(SMOKE_GPU, SMOKE_PORT, smoke_phase)
            smoke = manifest["smoke"]
            smoke_run_id = next_run_id(smoke_base)
            evaluate(
                smoke_run_id, "train", smoke["participant_file"], 1,
                [f"http://127.0.0.1:{SMOKE_PORT[0]}"], smoke_attestation, OUT / f"logs/{smoke_run_id}.log",
            )
        finally:
            if smoke_processes:
                stop(smoke_processes, smoke_phase)
    else:
        append_ledger("smoke_reused_completed", run_id=completed_run_id(smoke_base))

    after_smoke = wait_until_idle(FULL_GPUS)
    append_ledger("full_preflight_passed", gpu_scope=list(FULL_GPUS), gpu_state=after_smoke, smoke_run=f"{CAMPAIGN}-smoke-train1")
    full_processes: list[subprocess.Popen[str]] = []
    full_phase = next_phase("full-gpu4-7")
    try:
        full_processes, full_attestation = launch(FULL_GPUS, FULL_PORTS, full_phase)
        endpoints = [f"http://127.0.0.1:{port}" for port in FULL_PORTS]
        total = len(manifest["participants"])
        for index, item in enumerate(manifest["participants"], start=1):
            base_run_id = f"{CAMPAIGN}-{item['key']}"
            completed = completed_run_id(base_run_id)
            if completed is not None:
                append_ledger("judge_skipped_completed", run_id=completed)
                print(json.dumps({"progress": f"{index}/{total}", "run_id": completed, "reused": True}), flush=True)
                continue
            run_id = next_run_id(base_run_id)
            evaluate(run_id, "validation", item["participant_file"], 400, endpoints, full_attestation, OUT / f"logs/{index:02d}-{item['key']}-{run_id.rsplit('-', 1)[-1]}.log")
            print(json.dumps({"progress": f"{index}/{total}", "run_id": run_id}), flush=True)
    finally:
        if full_processes:
            stop(full_processes, full_phase)
    summary = aggregate(manifest)
    append_ledger("campaign_completed", summary_sha256=sha256_file(OUT / "aggregate_summary.json"), best=summary["best_participant"]["key"], best_macro_mean=summary["best_participant"]["macro_mean"])
    print(json.dumps({"status": "completed", "participants": len(summary["participants"]), "best": summary["best_participant"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
