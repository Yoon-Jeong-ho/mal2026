#!/usr/bin/env python3
"""Generate validation rationales from the AI-Hub-full -> API-LoRA axis models."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-standard/bin/python"
VLLM = ROOT / ".venv-standard/bin/vllm"
MODEL_PATH = ROOT / "outputs/official-aihub-rationale-full-sft-v1/official-aihub-rationale-full-sft-v1-ax4-axis_triplet-full-002/final_model"
FULL_COMPLETION = MODEL_PATH.parent / "training_complete.json"
SERVED_BASE = "official-aihub-full-rationale-ax4-axis-triplet"
VLLM_VERSION = "0.25.1"
TASKS = ("content", "organization", "expression")
GPUS = (0, 1, 2)
PORTS = (19300, 19301, 19302)
LORA_ROOT = ROOT / "outputs/official-aihub-then-api-rationale-lora-v1"
RUN_SUFFIX = "001"
RUN_ID = f"official-aihub-then-api-rationale-generation-v1-ax4-axis-triplet-validation-{RUN_SUFFIX}"
RUN_ROOT = ROOT / "outputs/official-prompt-alignment-v1/rationale-generation-runtime" / RUN_ID
RESTRICTED_ROOT = ROOT / "data/processed/restricted/official_prompt_alignment_v1/rationale_generation"
SCORE_FILE = ROOT / "data/processed/restricted/official_prompt_alignment_v1/score_predictions/official-score-essay-only-full-20260727-002/essay_only_epoch_04.jsonl"
SMOKE_SCORES = RESTRICTED_ROOT / "official-api-candidate-train-smoke-scores-001.jsonl"


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
    return LORA_ROOT / f"official-aihub-then-api-rationale-lora-v1-ax4-axis_triplet-{task}-full-001/adapter"


def training_completion(task: str) -> Path:
    return LORA_ROOT / f"official-aihub-then-api-rationale-lora-v1-ax4-axis_triplet-{task}-full-001/training_complete.json"


def output_path(task: str, split: str) -> Path:
    return RESTRICTED_ROOT / f"official-aihub-then-api-rationale-generation-v1-ax4-{task}-{split}-{RUN_SUFFIX}"


def verify_inputs() -> None:
    need(PYTHON.is_file() and VLLM.is_file() and MODEL_PATH.is_dir(), "local vLLM/full model prerequisite is unavailable")
    need(importlib.metadata.version("vllm") == VLLM_VERSION, "vLLM version differs")
    need(SCORE_FILE.is_file() and SMOKE_SCORES.is_file(), "restricted score prerequisite is unavailable")
    full = json.loads(FULL_COMPLETION.read_text(encoding="utf-8"))
    need(full.get("status") == "completed" and full.get("structure") == "axis_triplet" and full.get("training_kind") == "full_parameter" and full.get("train_records") == 48030, "full model provenance differs")
    for task in TASKS:
        complete_path = training_completion(task)
        need(complete_path.is_file() and adapter_path(task).is_dir(), f"completed continuation adapter is unavailable: {task}")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        need(complete.get("status") == "completed" and complete.get("structure") == "axis_triplet" and complete.get("task") == task and complete.get("training_kind") == "lora_after_aihub_full_parameter" and complete.get("train_records") == 6000, f"continuation provenance differs: {task}")
    need(not RUN_ROOT.exists(), "generation runtime output must be fresh")
    outputs = [output_path(task, "validation") for task in TASKS] + [output_path("content", "train-smoke")]
    need(not any(path.exists() for path in outputs), "generation data output must be fresh")


def check_gpus_idle() -> None:
    for gpu in GPUS:
        value = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
        index, memory, utilization = (part.strip() for part in value.split(","))
        need((int(index), int(memory), int(utilization)) == (gpu, 0, 0), f"GPU {gpu} is not idle")


def server_command(task: str, port: int) -> list[str]:
    alias = f"official-aihub-then-api-rationale-ax4-{task}"
    return [
        str(VLLM), "serve", str(MODEL_PATH), "--served-model-name", SERVED_BASE,
        "--host", "127.0.0.1", "--port", str(port), "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--attention-backend", "FLASH_ATTN", "--max-model-len", "4096",
        "--max-num-seqs", "128", "--max-num-batched-tokens", "16384",
        "--gpu-memory-utilization", "0.90", "--disable-custom-all-reduce",
        "--enable-lora", "--max-loras", "1", "--max-lora-rank", "32",
        "--lora-modules", f"{alias}={adapter_path(task)}", "--generation-config", "vllm",
        "--enable-prefix-caching", "--no-enable-flashinfer-autotune",
        "--compilation-config", '{"mode":0,"cudagraph_mode":"NONE","pass_config":{"fuse_allreduce_rms":false}}',
    ]


def stop_servers(processes: Sequence[subprocess.Popen[str]], handles: Sequence[Any]) -> None:
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


def start_servers() -> tuple[list[subprocess.Popen[str]], list[Any], list[Path]]:
    processes: list[subprocess.Popen[str]] = []
    handles: list[Any] = []
    attestations: list[Path] = []
    try:
        for task, gpu, port in zip(TASKS, GPUS, PORTS, strict=True):
            log = RUN_ROOT / "logs" / f"vllm-{task}-gpu{gpu}.log"
            handle = log.open("x", encoding="utf-8")
            (RUN_ROOT / "cache" / task).mkdir(mode=0o700, parents=True)
            env = {
                **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MAL2026_RESERVED_PHYSICAL_GPUS": str(gpu),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "VLLM_CACHE_ROOT": str((RUN_ROOT / "cache" / task / "vllm").resolve()),
                "TORCHINDUCTOR_CACHE_DIR": str((RUN_ROOT / "cache" / task / "torchinductor").resolve()),
            }
            process = subprocess.Popen(server_command(task, port), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            processes.append(process); handles.append(handle)
            attestation = RUN_ROOT / "attestations" / f"{task}.json"; attestations.append(attestation)
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
                "schema_version": "mal2026-official-rationale-vllm-server-attestation-v1", "created_at": now(),
                "endpoint": f"http://127.0.0.1:{port}", "physical_gpu": gpu, "tensor_parallel_size": 1,
                "max_model_len": 4096, "max_num_seqs": 128, "max_num_batched_tokens": 16384,
                "enforce_eager": False, "compilation_mode": 0, "cudagraph_mode": "NONE", "effective_eager_equivalent": True,
                "inference_dtype": "bfloat16",
                "model_id": SERVED_BASE, "model_revision": file_sha(FULL_COMPLETION), "vllm_version": VLLM_VERSION,
                "model_config_sha256": file_sha(MODEL_PATH / "config.json"), "adapter_alias": f"official-aihub-then-api-rationale-ax4-{task}",
                "adapter_path": str(adapter_path(task).resolve()), "adapter_training_completion_sha256": file_sha(training_completion(task)),
                "task": task, "server_pid": process.pid, "server_environment_verified": True,
            })
        return processes, handles, attestations
    except Exception:
        stop_servers(processes, handles)
        raise


def generation_command(task: str, score_file: Path, split: str, expected: int, output: Path, attestation: Path) -> list[str]:
    index = TASKS.index(task)
    return [
        str(PYTHON), str(ROOT / "scripts/generate_official_rationales_vllm.py"), "--run-id", output.name,
        "--task", task, "--split", split, "--expected", str(expected), "--score-file", str(score_file),
        "--output-dir", str(output), "--endpoint", f"http://127.0.0.1:{PORTS[index]}",
        "--model", f"official-aihub-then-api-rationale-ax4-{task}", "--server-attestation", str(attestation), "--max-inflight", "128",
    ]


def run_client(command: Sequence[str], log: Path) -> None:
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, stdout=handle, stderr=subprocess.STDOUT, text=True)
    need(completed.returncode == 0, f"generation client failed: {log.name}")


def main() -> None:
    verify_inputs(); check_gpus_idle()
    RUN_ROOT.mkdir(mode=0o700, parents=True); (RUN_ROOT / "logs").mkdir(mode=0o700); (RUN_ROOT / "attestations").mkdir(mode=0o700)
    manifest: dict[str, Any] = {
        "schema_version": "mal2026-official-rationale-generation-runner-v1", "status": "running", "run_id": RUN_ID,
        "created_at": now(), "physical_gpus": list(GPUS), "tasks": list(TASKS), "score_file_sha256": file_sha(SCORE_FILE),
        "model_id": SERVED_BASE, "full_model_completion_sha256": file_sha(FULL_COMPLETION), "vllm_version": VLLM_VERSION,
        "candidate_score_kind": "actual_emitted_integer_prediction", "human_or_reference_score_read_or_prompted": False,
    }
    atomic_json(RUN_ROOT / "manifest.json", manifest)
    processes: list[subprocess.Popen[str]] = []; handles: list[Any] = []
    try:
        processes, handles, attestations = start_servers()
        smoke = output_path("content", "train-smoke")
        run_client(generation_command("content", SMOKE_SCORES, "train", 1, smoke, attestations[0]), RUN_ROOT / "logs/client-content-train-smoke.log")
        smoke_report = json.loads((smoke / "aggregate_generation_report.json").read_text(encoding="utf-8"))
        need(smoke_report.get("status") == "completed" and all(smoke_report.get("hard_gates", {}).values()), "generation smoke gate failed")
        clients: list[tuple[str, subprocess.Popen[str], Any, Path]] = []
        for task, attestation in zip(TASKS, attestations, strict=True):
            output = output_path(task, "validation"); log = RUN_ROOT / "logs" / f"client-{task}-validation.log"; handle = log.open("x", encoding="utf-8")
            process = subprocess.Popen(generation_command(task, SCORE_FILE, "validation", 400, output, attestation), cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, stdout=handle, stderr=subprocess.STDOUT, text=True)
            clients.append((task, process, handle, output))
        failures: dict[str, int] = {}
        for task, process, handle, _ in clients:
            code = process.wait(); handle.close()
            if code: failures[task] = code
        need(not failures, f"validation generation clients failed: {failures}")
        reports: dict[str, Any] = {}
        for task, _, _, output in clients:
            report_path = output / "aggregate_generation_report.json"; report = json.loads(report_path.read_text(encoding="utf-8"))
            need(report.get("status") == "completed" and all(report.get("hard_gates", {}).values()), f"generation aggregate gate failed: {task}")
            reports[task] = {"counts": report["counts"], "aggregate_report_sha256": file_sha(report_path)}
        manifest.update({"status": "completed", "completed_at": now(), "smoke_status": "completed", "validation_reports": reports})
        atomic_json(RUN_ROOT / "manifest.json", manifest)
        print(json.dumps({"status": "completed", "run_id": RUN_ID, "validation_reports": reports}, sort_keys=True))
    except Exception:
        manifest.update({"status": "failed", "failed_at": now()}); atomic_json(RUN_ROOT / "manifest.json", manifest); raise
    finally:
        stop_servers(processes, handles)


if __name__ == "__main__":
    main()
