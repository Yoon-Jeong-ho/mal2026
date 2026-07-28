"""Owned local server lifecycle for the official rationale RL experiment."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Any, Iterator, Mapping, Sequence
from urllib.request import urlopen
from uuid import uuid4

from .api_rationale_data import ROOT, sha256_file
from .official_rationale_rl import (
    JUDGE_PROMPT_SHA256,
    LLAMA_REVISION,
    LLAMA_TAG,
    MODEL_ID,
    MODEL_PATH,
    MODEL_REVISION,
    Q4_MODEL_SHA256,
    VLLM_VERSION,
)


PYTHON = ROOT / ".venv-standard/bin/python"
VLLM = ROOT / ".venv-standard/bin/vllm"
LLAMA_SERVER = ROOT / "outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server"
LLAMA_REPO = ROOT / "outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/llama.cpp"
Q4_MODEL = ROOT / "outputs/model-cache/Qwen--Qwen3.6-35B-A3B-GGUF-5c2410d71524f4f72b023ce8daf7a80528226d5f/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"


class OfficialRLServerError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialRLServerError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_compute_pids(gpu: int) -> tuple[int, ...]:
    need(gpu in {0, 1, 2, 3}, "GPU is outside the authorized scope")
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        cwd=ROOT, text=True, capture_output=True,
    )
    need(result.returncode == 0, f"GPU{gpu} process query failed")
    values = tuple(int(line.strip()) for line in result.stdout.splitlines() if line.strip() and line.strip() != "[N/A]")
    return values


def assert_gpus_idle(gpus: Sequence[int]) -> None:
    chosen = tuple(gpus)
    need(bool(chosen) and len(set(chosen)) == len(chosen) and set(chosen) <= {0, 1, 2, 3}, "GPU scope differs")
    conflicts = {gpu: gpu_compute_pids(gpu) for gpu in chosen}
    need(not any(conflicts.values()), f"pre-existing GPU compute processes block launch: {conflicts}")


def assert_ports_free(ports: Sequence[int]) -> None:
    need(bool(ports) and len(set(ports)) == len(ports), "server ports differ")
    for port in ports:
        need(1024 <= port <= 65535, "server port differs")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                handle.bind(("127.0.0.1", port))
            except OSError as exc:
                raise OfficialRLServerError(f"server port is occupied: {port}") from exc


def verify_server_prerequisites() -> None:
    need(PYTHON.is_file() and VLLM.is_file() and MODEL_PATH.is_dir(), "vLLM/model prerequisite is unavailable")
    need(importlib.metadata.version("vllm") == VLLM_VERSION, "vLLM version differs")
    need(LLAMA_SERVER.is_file() and os.access(LLAMA_SERVER, os.X_OK) and Q4_MODEL.is_file(), "Q4 runtime prerequisite is unavailable")
    need(sha256_file(Q4_MODEL) == Q4_MODEL_SHA256, "Q4 model digest differs")
    revision = subprocess.check_output(["git", "-C", str(LLAMA_REPO), "rev-parse", "HEAD"], text=True).strip()
    tag = subprocess.check_output(["git", "-C", str(LLAMA_REPO), "describe", "--tags", "--exact-match"], text=True).strip()
    need(revision == LLAMA_REVISION and tag == LLAMA_TAG, "llama.cpp revision differs")


class OwnedProcesses:
    """Tracks only Popen objects created by this context; never adopts PIDs."""

    def __init__(self, token: str):
        self.token = token
        self.items: list[tuple[subprocess.Popen[str], Any]] = []

    def add(self, process: subprocess.Popen[str], handle: Any) -> None:
        self.items.append((process, handle))

    def stop(self) -> None:
        for process, _ in self.items:
            if process.poll() is None:
                environment = Path(f"/proc/{process.pid}/environ")
                if not environment.is_file():
                    process.poll()
                    need(process.returncode is not None, "refusing to stop a process without current-run ownership proof")
                    continue
                need(f"MAL2026_OFFICIAL_RL_SERVER_TOKEN={self.token}".encode() in environment.read_bytes().split(b"\0"), "refusing to stop a process without current-run ownership proof")
                os.killpg(process.pid, signal.SIGTERM)
        for process, _ in self.items:
            if process.poll() is None:
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    environment = Path(f"/proc/{process.pid}/environ")
                    need(environment.is_file() and f"MAL2026_OFFICIAL_RL_SERVER_TOKEN={self.token}".encode() in environment.read_bytes().split(b"\0"), "refusing forced stop without ownership proof")
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
        for _, handle in self.items:
            handle.close()


def wait_health(process: subprocess.Popen[str], endpoint: str, label: str) -> None:
    for _ in range(480):
        need(process.poll() is None, f"server exited before health gate: {label}")
        try:
            with urlopen(endpoint + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise OfficialRLServerError(f"server health timeout: {label}")


def vllm_policy_command(
    *, gpus: Sequence[int], port: int, adapters: Mapping[str, Path], aliases: Mapping[str, str],
    max_num_seqs: int, max_num_batched_tokens: int, dynamic_updates: bool,
    max_model_len: int = 4096,
    model_path: Path = MODEL_PATH, model_id: str = MODEL_ID,
) -> list[str]:
    command = [
        str(VLLM), "serve", str(model_path), "--served-model-name", model_id,
        "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", str(len(tuple(gpus))), "--attention-backend", "FLASH_ATTN",
        "--max-model-len", str(max_model_len), "--max-num-seqs", str(max_num_seqs),
        "--max-num-batched-tokens", str(max_num_batched_tokens), "--gpu-memory-utilization", "0.90",
        "--disable-custom-all-reduce", "--enable-lora", "--max-loras", str(len(adapters)), "--max-lora-rank", "32",
        "--generation-config", "vllm", "--enable-prefix-caching", "--no-enable-flashinfer-autotune",
        "--compilation-config", '{"pass_config":{"fuse_allreduce_rms":false}}',
    ]
    if not dynamic_updates:
        command += ["--lora-modules", *[f"{aliases[task]}={adapters[task]}" for task in sorted(adapters)]]
    return command


def q4_server_command(port: int) -> list[str]:
    return [
        str(LLAMA_SERVER), "--model", str(Q4_MODEL), "--host", "127.0.0.1", "--port", str(port),
        "--n-gpu-layers", "99", "--parallel", "4", "--ctx-size", "32768",
        "--batch-size", "2048", "--ubatch-size", "512", "--no-webui", "--reasoning", "off",
    ]


@contextmanager
def vllm_policy_server(
    *,
    runtime_root: Path,
    label: str,
    gpus: Sequence[int],
    port: int,
    adapters: Mapping[str, Path],
    aliases: Mapping[str, str],
    max_num_seqs: int,
    max_num_batched_tokens: int,
    dynamic_updates: bool,
    max_model_len: int = 4096,
    model_path: Path = MODEL_PATH,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> Iterator[tuple[str, Path]]:
    verify_server_prerequisites()
    chosen = tuple(gpus)
    need(set(adapters) == set(aliases) and 1 <= len(adapters) <= 4, "policy adapter declarations differ")
    need(model_path.is_dir() and not model_path.is_symlink() and (model_path / "config.json").is_file(), "policy base model snapshot is unavailable")
    for task, adapter in adapters.items():
        need(adapter.is_dir() and not adapter.is_symlink() and (adapter / "adapter_config.json").is_file(), f"policy adapter unavailable: {task}")
    assert_gpus_idle(chosen)
    assert_ports_free((port,))
    token = uuid4().hex
    owner = OwnedProcesses(token)
    endpoint = f"http://127.0.0.1:{port}"
    log = runtime_root / "logs" / f"vllm-{label}.log"
    attestation = runtime_root / "attestations" / f"vllm-{label}.json"
    need(not log.exists() and not attestation.exists(), "policy server artifacts must be fresh")
    log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    attestation.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    command = vllm_policy_command(gpus=chosen, port=port, adapters=adapters, aliases=aliases,
                                   max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                                   dynamic_updates=dynamic_updates, max_model_len=max_model_len,
                                   model_path=model_path, model_id=model_id)
    visible = ",".join(str(gpu) for gpu in chosen)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": visible,
        "MAL2026_RESERVED_PHYSICAL_GPUS": visible,
        "MAL2026_OFFICIAL_RL_SERVER_TOKEN": token,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1" if dynamic_updates else "0",
    }
    handle = log.open("x", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    owner.add(process, handle)
    try:
        wait_health(process, endpoint, label)
        process_environment = Path(f"/proc/{process.pid}/environ").read_bytes().split(b"\0")
        need(f"CUDA_VISIBLE_DEVICES={visible}".encode() in process_environment and f"MAL2026_OFFICIAL_RL_SERVER_TOKEN={token}".encode() in process_environment, "policy server environment differs")
        atomic_json(attestation, {
            "schema_version": "mal2026-official-rl-policy-server-attestation-v1",
            "created_at": now(), "endpoint": endpoint, "physical_gpus": list(chosen),
            "tensor_parallel_size": len(chosen), "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs, "max_num_batched_tokens": max_num_batched_tokens,
            "gpu_memory_utilization": 0.9, "enforce_eager": False,
            "compatibility_workaround": "disable_custom_all_reduce_and_fuse_allreduce_rms_only; compilation_and_cuda_graphs_remain_enabled",
            "vllm_version": VLLM_VERSION, "model_id": model_id, "model_revision": model_revision,
            "model_path": str(model_path.resolve()), "model_config_sha256": sha256_file(model_path / "config.json"),
            "adapter_aliases": dict(aliases),
            "adapter_paths": {task: str(path.resolve()) for task, path in adapters.items()},
            "adapter_config_sha256": {task: sha256_file(path / "adapter_config.json") for task, path in adapters.items()},
            "dynamic_lora": True, "dynamic_runtime_updates": dynamic_updates,
            "structured_outputs_json_schema": True, "train_split_only": True,
            "judge_prompt_sha256": JUDGE_PROMPT_SHA256,
            "server_pid": process.pid, "server_process_environment_verified": True,
        })
        yield endpoint, attestation
    finally:
        owner.stop()


@contextmanager
def q4_judge_servers(
    *,
    runtime_root: Path,
    label: str,
    gpus: Sequence[int],
    ports: Sequence[int],
    judge_prompt_sha256: str = JUDGE_PROMPT_SHA256,
) -> Iterator[tuple[list[str], Path]]:
    verify_server_prerequisites()
    chosen, chosen_ports = tuple(gpus), tuple(ports)
    need(len(chosen) == len(chosen_ports) and bool(chosen), "Q4 topology differs")
    assert_gpus_idle(chosen)
    assert_ports_free(chosen_ports)
    token = uuid4().hex
    owner = OwnedProcesses(token)
    endpoints: list[str] = []
    pids: list[int] = []
    attestation = runtime_root / "attestations" / f"q4-{label}.json"
    need(not attestation.exists(), "Q4 attestation must be fresh")
    attestation.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        for gpu, port in zip(chosen, chosen_ports, strict=True):
            endpoint = f"http://127.0.0.1:{port}"
            log = runtime_root / "logs" / f"q4-{label}-gpu{gpu}.log"
            need(not log.exists(), "Q4 server log must be fresh")
            log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            command = q4_server_command(port)
            environment = {
                **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MAL2026_RESERVED_PHYSICAL_GPUS": str(gpu),
                "MAL2026_OFFICIAL_RL_SERVER_TOKEN": token,
            }
            handle = log.open("x", encoding="utf-8")
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            owner.add(process, handle)
            wait_health(process, endpoint, f"{label}-gpu{gpu}")
            process_environment = Path(f"/proc/{process.pid}/environ").read_bytes().split(b"\0")
            need(f"CUDA_VISIBLE_DEVICES={gpu}".encode() in process_environment and f"MAL2026_OFFICIAL_RL_SERVER_TOKEN={token}".encode() in process_environment, "Q4 server environment differs")
            endpoints.append(endpoint)
            pids.append(process.pid)
        atomic_json(attestation, {
            "schema_version": "mal2026-official-q4-judge-server-attestation-v1",
            "created_at": now(), "physical_gpus": list(chosen), "server_endpoints": endpoints,
            "server_pids": pids, "parallel_per_server": 4, "context_per_slot": 8192,
            "batch_size": 2048, "ubatch_size": 512,
            "model_sha256": Q4_MODEL_SHA256, "judge_prompt_sha256": judge_prompt_sha256,
            "llama_server_sha256": sha256_file(LLAMA_SERVER),
            "llama_revision": LLAMA_REVISION, "llama_tag": LLAMA_TAG,
            "server_process_environment_verified": True,
        })
        yield endpoints, attestation
    finally:
        owner.stop()
