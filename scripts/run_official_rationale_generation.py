#!/usr/bin/env python3
"""Serve four official-rationale adapters and generate validation outputs.

The four independent vLLM replicas deliberately use one physical GPU each:
bundle on GPU0 and content/organization/expression on GPUs1/2/3.  All servers
are started once.  A real train-row bundle request is the generation smoke;
after that gate passes, the four 400-row validation clients run concurrently.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-standard/bin/python"
VLLM = ROOT / ".venv-standard/bin/vllm"
MODEL_PATH = ROOT / "outputs/model-cache/skt--A.X-4.0-Light-ba21c20ea1b31ded1ec3e2fb432335077dc4be98"
MODEL_ID = "skt/A.X-4.0-Light"
MODEL_REVISION = "ba21c20ea1b31ded1ec3e2fb432335077dc4be98"
VLLM_VERSION = "0.25.1"
TASKS = ("bundle", "content", "organization", "expression")
GPUS = (0, 1, 2, 3)
PORTS = (19200, 19201, 19202, 19203)
SFT_ROOT = ROOT / "outputs/official-rationale-sft-v1"
RUN_SUFFIX = "004"
RUN_ID = f"official-rationale-generation-v1-ax4-four-structures-validation-{RUN_SUFFIX}"
RUN_ROOT = ROOT / "outputs/official-prompt-alignment-v1/rationale-generation-runtime" / RUN_ID
RESTRICTED_ROOT = ROOT / "data/processed/restricted/official_prompt_alignment_v1/rationale_generation"
SCORE_FILE = ROOT / "data/processed/restricted/official_prompt_alignment_v1/score_predictions/official-score-essay-only-full-20260727-002/essay_only_epoch_04.jsonl"
API_CANDIDATES = ROOT / "data/processed/restricted/official_openai_candidates_v1/official-openai-candidates-v1-train3-20260727-001/candidates.train.jsonl"


class RunnerError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def adapter_path(task: str) -> Path:
    return SFT_ROOT / f"official-rationale-sft-v1-ax4-{task}-full-001/adapter"


def training_completion(task: str) -> Path:
    return SFT_ROOT / f"official-rationale-sft-v1-ax4-{task}-full-001/training_complete.json"


def verify_inputs() -> None:
    need(PYTHON.is_file() and VLLM.is_file() and MODEL_PATH.is_dir(), "local vLLM/model prerequisite is unavailable")
    need(importlib.metadata.version("vllm") == VLLM_VERSION, "vLLM version differs")
    need(SCORE_FILE.is_file() and API_CANDIDATES.is_file(), "restricted score/candidate prerequisite is unavailable")
    for task in TASKS:
        complete_path = training_completion(task)
        need(complete_path.is_file() and adapter_path(task).is_dir(), f"completed adapter is unavailable: {task}")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        need(
            complete.get("status") == "completed"
            and complete.get("task") == task
            and complete.get("model_id") == MODEL_ID
            and complete.get("model_revision") == MODEL_REVISION
            and complete.get("train_records") == 6000,
            f"adapter completion provenance differs: {task}",
        )
    need(not RUN_ROOT.exists(), "generation runtime output must be fresh")
    outputs = [RESTRICTED_ROOT / f"official-rationale-generation-v1-ax4-{task}-validation-{RUN_SUFFIX}" for task in TASKS]
    outputs.append(RESTRICTED_ROOT / f"official-rationale-generation-v1-ax4-bundle-train-smoke-{RUN_SUFFIX}")
    need(not any(path.exists() for path in outputs), "generation data output must be fresh")


def check_gpus_idle() -> None:
    for gpu in GPUS:
        value = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        index, memory, utilization = (part.strip() for part in value.split(","))
        need((int(index), int(memory), int(utilization)) == (gpu, 0, 0), f"GPU {gpu} is not idle")


def make_smoke_scores() -> Path:
    destination = RESTRICTED_ROOT / "official-api-candidate-train-smoke-scores-001.jsonl"
    with API_CANDIDATES.open(encoding="utf-8") as handle:
        row = json.loads(next(line for line in handle if line.strip()))
    participant = row["participant_output"]
    axes = ("content", "organization", "expression")
    scores = {axis: participant[axis]["score"] for axis in axes}
    need(all(type(scores[axis]) is int and 1 <= scores[axis] <= 5 for axis in axes), "smoke API scores differ")
    expected = {"source_id": row["source_id"], "emitted_integer_prediction": scores}
    if destination.exists():
        existing = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines() if line.strip()]
        need(existing == [expected], "preserved smoke score file differs")
        return destination
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return destination


def server_command(task: str, port: int) -> list[str]:
    alias = f"official-rationale-ax4-{task}"
    return [
        str(VLLM), "serve", str(MODEL_PATH),
        "--served-model-name", MODEL_ID,
        "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", "1",
        "--attention-backend", "FLASH_ATTN",
        "--max-model-len", "4096",
        "--max-num-seqs", "128",
        "--max-num-batched-tokens", "16384",
        "--gpu-memory-utilization", "0.90",
        "--disable-custom-all-reduce",
        "--enable-lora", "--max-loras", "1", "--max-lora-rank", "32",
        "--lora-modules", f"{alias}={adapter_path(task)}",
        "--generation-config", "vllm",
        "--enable-prefix-caching",
        "--no-enable-flashinfer-autotune",
        # The preserved -001 run failed in Inductor autotuning, -002 failed in
        # CUDA graph capture, and -003 isolated a 65,536-token dummy profile
        # GEMM failure even without either.  Disable the first two setup paths
        # and bound the prefill profile to 16,384 tokens while retaining vLLM
        # scheduling, continuous batching, FlashAttention and prefix caching.
        "--compilation-config", '{"mode":0,"cudagraph_mode":"NONE","pass_config":{"fuse_allreduce_rms":false}}',
    ]


def start_servers() -> tuple[list[subprocess.Popen[str]], list[Any], list[Path]]:
    processes: list[subprocess.Popen[str]] = []
    handles: list[Any] = []
    attestations: list[Path] = []
    try:
        # Initialize one replica at a time.  The failed -001 lineage started
        # four Inductor autotuners against a shared cache simultaneously.
        for task, gpu, port in zip(TASKS, GPUS, PORTS, strict=True):
            log = RUN_ROOT / "logs" / f"vllm-{task}-gpu{gpu}.log"
            handle = log.open("x", encoding="utf-8")
            (RUN_ROOT / "cache" / task).mkdir(mode=0o700, parents=True)
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "MAL2026_RESERVED_PHYSICAL_GPUS": str(gpu),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "VLLM_CACHE_ROOT": str((RUN_ROOT / "cache" / task / "vllm").resolve()),
                "TORCHINDUCTOR_CACHE_DIR": str((RUN_ROOT / "cache" / task / "torchinductor").resolve()),
            }
            process = subprocess.Popen(
                server_command(task, port), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
            processes.append(process)
            handles.append(handle)
            attestation = RUN_ROOT / "attestations" / f"{task}.json"
            attestations.append(attestation)
            for _ in range(360):
                need(process.poll() is None, f"vLLM server exited before health gate: {task}")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(1)
            else:
                raise RunnerError(f"vLLM server health timeout: {task}")
            environment = Path(f"/proc/{process.pid}/environ").read_bytes().split(b"\0")
            visible = next(item.split(b"=", 1)[1].decode() for item in environment if item.startswith(b"CUDA_VISIBLE_DEVICES="))
            need(visible == str(gpu), f"vLLM CUDA binding differs: {task}")
            atomic_json(attestation, {
                "schema_version": "mal2026-official-rationale-vllm-server-attestation-v1",
                "created_at": now(),
                "endpoint": f"http://127.0.0.1:{port}",
                "physical_gpu": gpu,
                "tensor_parallel_size": 1,
                "max_model_len": 4096,
                "max_num_seqs": 128,
                "max_num_batched_tokens": 16384,
                "enforce_eager": False,
                "compilation_mode": 0,
                "cudagraph_mode": "NONE",
                "effective_eager_equivalent": True,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "vllm_version": VLLM_VERSION,
                "model_config_sha256": file_sha(MODEL_PATH / "config.json"),
                "adapter_alias": f"official-rationale-ax4-{task}",
                "adapter_path": str(adapter_path(task).resolve()),
                "adapter_training_completion_sha256": file_sha(training_completion(task)),
                "task": task,
                "server_pid": process.pid,
                "server_environment_verified": True,
            })
        return processes, handles, attestations
    except Exception:
        stop_servers(processes, handles)
        raise


def generation_command(task: str, score_file: Path, split: str, expected: int, output: Path, attestation: Path) -> list[str]:
    run_id = output.name
    index = TASKS.index(task)
    return [
        str(PYTHON), str(ROOT / "scripts/generate_official_rationales_vllm.py"),
        "--run-id", run_id,
        "--task", task,
        "--split", split,
        "--expected", str(expected),
        "--score-file", str(score_file),
        "--output-dir", str(output),
        "--endpoint", f"http://127.0.0.1:{PORTS[index]}",
        "--model", f"official-rationale-ax4-{task}",
        "--server-attestation", str(attestation),
        "--max-inflight", "128",
    ]


def run_client(command: list[str], log: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    need(completed.returncode == 0, f"generation client failed: {log.name}")


def stop_servers(processes: list[subprocess.Popen[str]], handles: list[Any]) -> None:
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
    for handle in handles:
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    need(args.run_id == RUN_ID, "generation run ID differs from the frozen task card")
    verify_inputs()
    check_gpus_idle()
    RUN_ROOT.mkdir(mode=0o700, parents=True)
    (RUN_ROOT / "logs").mkdir(mode=0o700)
    (RUN_ROOT / "attestations").mkdir(mode=0o700)
    manifest: dict[str, Any] = {
        "schema_version": "mal2026-official-rationale-generation-runner-v1",
        "status": "running",
        "run_id": RUN_ID,
        "created_at": now(),
        "physical_gpus": list(GPUS),
        "tasks": list(TASKS),
        "score_file_sha256": file_sha(SCORE_FILE),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "vllm_version": VLLM_VERSION,
        "candidate_score_kind": "actual_emitted_integer_prediction",
        "human_or_reference_score_read_or_prompted": False,
    }
    atomic_json(RUN_ROOT / "manifest.json", manifest)
    smoke_scores = make_smoke_scores()
    processes: list[subprocess.Popen[str]] = []
    handles: list[Any] = []
    try:
        processes, handles, attestations = start_servers()
        smoke_output = RESTRICTED_ROOT / f"official-rationale-generation-v1-ax4-bundle-train-smoke-{RUN_SUFFIX}"
        run_client(
            generation_command("bundle", smoke_scores, "train", 1, smoke_output, attestations[0]),
            RUN_ROOT / "logs/client-bundle-train-smoke.log",
        )
        smoke_report = json.loads((smoke_output / "aggregate_generation_report.json").read_text(encoding="utf-8"))
        need(smoke_report.get("status") == "completed" and all(smoke_report.get("hard_gates", {}).values()), "generation smoke gate failed")
        clients: list[subprocess.Popen[str]] = []
        client_handles: list[Any] = []
        outputs: dict[str, Path] = {}
        for task, attestation in zip(TASKS, attestations, strict=True):
            output = RESTRICTED_ROOT / f"official-rationale-generation-v1-ax4-{task}-validation-{RUN_SUFFIX}"
            log = RUN_ROOT / "logs" / f"client-{task}-validation.log"
            handle = log.open("x", encoding="utf-8")
            command = generation_command(task, SCORE_FILE, "validation", 400, output, attestation)
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, stdout=handle, stderr=subprocess.STDOUT, text=True)
            clients.append(process)
            client_handles.append(handle)
            outputs[task] = output
        failures: dict[str, int] = {}
        for task, process, handle in zip(TASKS, clients, client_handles, strict=True):
            code = process.wait()
            handle.close()
            if code:
                failures[task] = code
        need(not failures, f"validation generation clients failed: {failures}")
        reports: dict[str, Any] = {}
        for task, output in outputs.items():
            report_path = output / "aggregate_generation_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            need(report.get("status") == "completed" and all(report.get("hard_gates", {}).values()), f"generation aggregate gate failed: {task}")
            reports[task] = {"counts": report["counts"], "aggregate_report_sha256": file_sha(report_path)}
        manifest.update({"status": "completed", "completed_at": now(), "smoke_status": "completed", "validation_reports": reports})
        atomic_json(RUN_ROOT / "manifest.json", manifest)
        print(json.dumps({"status": "completed", "run_id": RUN_ID, "validation_reports": reports}, sort_keys=True))
    except Exception:
        manifest.update({"status": "failed", "failed_at": now()})
        atomic_json(RUN_ROOT / "manifest.json", manifest)
        raise
    finally:
        stop_servers(processes, handles)


if __name__ == "__main__":
    main()
