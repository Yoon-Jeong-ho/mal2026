#!/usr/bin/env python3
"""Sequential, reproducible runner for the declared RLAIF prompt-ensemble study.

All runtime configs, server logs, adapters, completions, and score observations
live under ignored roots.  This runner writes only aggregate JSON metadata.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv-standard" / "bin" / "python"
VLLM = ROOT / ".venv-standard" / "bin" / "vllm"
CONFIG = ROOT / os.environ.get("MAL2026_RLAIF_CONFIG", "configs/rlaif_grpo_prompt_ensemble.v1.json")
_BOOTSTRAP_CONFIG = json.loads(CONFIG.read_text(encoding="utf-8"))
STUDY_ID = str(_BOOTSTRAP_CONFIG["schema_version"]).removeprefix("mal2026-")
RUNTIME_ID = os.environ.get("MAL2026_RLAIF_RUNTIME_ID", "20260722-001")
if not __import__("re").fullmatch(r"20260722-[0-9]{3}", RUNTIME_ID):
    raise RuntimeError("MAL2026_RLAIF_RUNTIME_ID must be a fresh YYYYMMDD-NNN lineage")
RUN_ROOT = ROOT / "outputs" / STUDY_ID / RUNTIME_ID
ARM_ROOT = ROOT / "outputs" / STUDY_ID
RESTRICTED = ROOT / "data" / "processed" / "restricted" / "openai_rationale_batches" / "openai-rationale-terra-full-20260719-001"
EVALUATION_ROOT = RESTRICTED / f"rlaif_grpo_{STUDY_ID.rsplit('-', 1)[-1]}"
GEN_ROOT = RESTRICTED / "decoder_generation_v1"
JUDGE_ROOT = RESTRICTED / "decoder_judge_v1"
AGGREGATE_ROOT = ROOT / "outputs" / "aggregate-reports"

MODELS = {
    "ax4_light": {"id": "skt/A.X-4.0-Light", "revision": "ba21c20ea1b31ded1ec3e2fb432335077dc4be98", "path": ROOT / "outputs" / "model-cache" / "skt--A.X-4.0-Light-ba21c20ea1b31ded1ec3e2fb432335077dc4be98"},
    "phi4_mini": {"id": "microsoft/Phi-4-mini-instruct", "revision": "cfbefacb99257ffa30c83adab238a50856ac3083", "path": ROOT / "outputs" / "model-cache" / "microsoft--Phi-4-mini-instruct-cfbefacb99257ffa30c83adab238a50856ac3083"},
    "midm2_base": {"id": "K-intelligence/Midm-2.0-Base-Instruct", "revision": "35479c5fc9a18a5db7cc6dbadcf1db68db7beab0", "path": ROOT / "outputs" / "model-cache" / "K-intelligence--Midm-2.0-Base-Instruct-35479c5fc9a18a5db7cc6dbadcf1db68db7beab0"},
}
TASKS = ("bundle", "content", "organization", "expression")
ARMS = ("all5", "random1")
STRUCTURED_SCHEMA_SUFFIXES = ("-v2", "-v3", "-v4", "-v5", "-v6", "-v7", "-v8")
PILOT_SCHEMA_SUFFIXES = ("-v3", "-v4", "-v5", "-v6", "-v7", "-v8")
TP2_SCHEMA_SUFFIXES = ("-v6", "-v7", "-v8")


class RunnerError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"unreadable aggregate artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"aggregate artifact is not an object: {path}")
    return value


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def config() -> dict[str, Any]:
    value = read_json(CONFIG)
    ensure(value.get("schema_version") == _BOOTSTRAP_CONFIG.get("schema_version") and value.get("run_id_prefix") == _BOOTSTRAP_CONFIG.get("run_id_prefix") and sha(ROOT / value["fixed_v6_config"]) == value.get("fixed_v6_config_sha256"), "declared RLAIF config or fixed-v6 binding differs")
    return value


def fixed_v6_enforce_eager(cfg: Mapping[str, Any]) -> bool:
    """Read the already hash-bound frozen-v6 runtime toggle exactly once."""
    template = read_json(ROOT / str(cfg["fixed_v6_config"]))
    runtime = template.get("runtime")
    ensure(isinstance(runtime, dict) and isinstance(runtime.get("enforce_eager"), bool), "fixed-v6 eager-mode setting is unavailable")
    return bool(runtime["enforce_eager"])


def ledger(event: Mapping[str, Any]) -> None:
    RUN_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (RUN_ROOT / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": now(), **event}, ensure_ascii=False, sort_keys=True) + "\n")


def write_config(name: str, value: Mapping[str, Any]) -> Path:
    path = RUN_ROOT / "configs" / f"{name}.json"; path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    ensure(not path.exists(), f"refusing to overwrite runtime config {path}")
    atomic_json(path, value)
    return path


def base_env(gpus: str) -> dict[str, str]:
    return {"CUDA_VISIBLE_DEVICES": gpus, "MAL2026_RESERVED_PHYSICAL_GPUS": gpus, "PYTHONPATH": str(ROOT / "src"), "VLLM_USE_FLASHINFER_SAMPLER": "0"}


def run_stage(name: str, command: list[str], env: Mapping[str, str]) -> None:
    logs = RUN_ROOT / "logs"; logs.mkdir(mode=0o700, parents=True, exist_ok=True); log = logs / f"{name}.log"
    ensure(not log.exists(), f"refusing to overwrite stage log {name}")
    complete = os.environ.copy(); complete.update(env); complete["PYTHONPATH"] = str(ROOT / "src")
    runtime_env = {key: complete[key] for key in ("PYTORCH_CUDA_ALLOC_CONF",) if key in complete}
    ledger({"stage": name, "event": "start", "command": command, "resource_scope": complete.get("CUDA_VISIBLE_DEVICES", "none"), "execution_env": runtime_env})
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, env=complete, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        ledger({"stage": name, "event": "failed", "exit_code": result.returncode, "log": str(log.relative_to(ROOT))})
        raise RunnerError(f"stage {name} failed; preserved {log}")
    ledger({"stage": name, "event": "completed", "exit_code": 0, "log": str(log.relative_to(ROOT))})


def check_idle(gpus: list[int], settle_seconds: int = 90) -> None:
    ensure(all(gpu in {0, 1, 2, 3} for gpu in gpus), "only project GPUs 0--3 are permitted")
    deadline = time.monotonic() + settle_seconds
    while True:
        status: dict[int, tuple[int, int]] = {}
        for gpu in gpus:
            value = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
            index, memory, utilization = [item.strip() for item in value.split(",")]
            ensure(index == str(gpu), "nvidia-smi returned an unexpected GPU")
            status[gpu] = (int(memory), int(utilization))
        if all(memory == 0 and utilization == 0 for memory, utilization in status.values()):
            return
        if time.monotonic() >= deadline:
            raise RunnerError(f"GPUs are not idle within the declared resource scope: {status}")
        time.sleep(2)


@contextmanager
def reward_server(*, gpus: list[int], data_parallel_size: int, port: int, label: str) -> Iterator[Path]:
    cfg = config(); check_idle(gpus)
    qwen_path = ROOT / cfg["judge"]["model_path"]
    ensure(qwen_path.is_dir() and not qwen_path.is_symlink(), "declared Qwen judge snapshot is unavailable")
    logs = RUN_ROOT / "logs"; logs.mkdir(mode=0o700, parents=True, exist_ok=True); log = logs / f"server-reward-{label}.log"; ensure(not log.exists(), "reward-server log already exists")
    compile_config = '{"pass_config":{"fuse_allreduce_rms":false}}'
    enforce_eager = fixed_v6_enforce_eager(cfg)
    command = [str(VLLM), "serve", str(qwen_path), "--served-model-name", cfg["judge"]["model_id"], "--host", "127.0.0.1", "--port", str(port),
               "--tensor-parallel-size", "1", "--data-parallel-size", str(data_parallel_size), "--attention-backend", "FLASH_ATTN", "--max-model-len", "4096",
               "--max-num-seqs", "192", "--max-num-batched-tokens", "65536", "--gpu-memory-utilization", "0.9", "--disable-custom-all-reduce",
               "--gdn-prefill-backend", "triton", "--generation-config", "vllm", "--enable-prefix-caching", "--no-enable-flashinfer-autotune",
               "--compilation-config", compile_config]
    if enforce_eager:
        command.append("--enforce-eager")
    environment = {**os.environ, **base_env(",".join(map(str, gpus))), "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "0"}
    ledger({"stage": f"reward-server-{label}", "event": "start", "command": command, "resource_scope": gpus, "execution_env": {"VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "0"}})
    handle = log.open("x", encoding="utf-8"); process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    attestation = RUN_ROOT / "attestations" / f"reward-{label}.json"; attestation.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        for _ in range(360):
            if process.poll() is not None:
                raise RunnerError(f"reward server {label} exited before health gate")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RunnerError(f"reward server {label} health timeout")
        environ = Path(f"/proc/{process.pid}/environ").read_bytes(); expected_visible = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))}".encode()
        ensure(expected_visible in environ, "reward server CUDA visibility differs")
        atomic_json(attestation, {"schema_version": "mal2026-rlaif-grpo-reward-server-attestation-v1", "server_host": "127.0.0.1", "server_port": port,
                                  "physical_gpus": gpus, "model_id": cfg["judge"]["model_id"], "model_revision": cfg["judge"]["model_revision"],
                                  "tensor_parallel_size": 1, "data_parallel_size": data_parallel_size, "max_model_len": 4096, "max_num_seqs_per_rank": 192,
                                  "enforce_eager": enforce_eager, "fixed_v6_config_sha256": cfg["fixed_v6_config_sha256"], "server_process_environment_verified": True})
        run_stage(f"reward-schema-health-{label}", [str(PY), "scripts/verify_rlaif_reward_server.py", "--endpoint", f"http://127.0.0.1:{port}"], base_env(",".join(map(str, gpus))))
        ledger({"stage": f"reward-server-{label}", "event": "health_and_score_schema_pass", "resource_scope": gpus, "raw_response_persisted": False})
        yield attestation
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=30)
        handle.close(); ledger({"stage": f"reward-server-{label}", "event": "stopped", "resource_scope": gpus})


@contextmanager
def policy_rollout_server(*, base_key: str, task: str, port: int, label: str) -> Iterator[Path]:
    """Declared-topology vLLM policy server for structured rollout requests.

    Its LoRA is intentionally loaded through the local dynamic endpoint by the
    trainer immediately before each rollout batch, so the server never owns a
    stale static policy adapter.
    """
    cfg = config(); ensure(cfg["schema_version"].endswith(STRUCTURED_SCHEMA_SUFFIXES), "policy rollout server is only valid for a structured-rollout config")
    gpus = list(cfg["runtime"]["full_rollout_gpus"]); tensor_parallel_size = int(cfg["runtime"]["rollout_tensor_parallel_size"])
    ensure(gpus == list(range(len(gpus))) and len(gpus) == tensor_parallel_size, "policy rollout GPU/TP topology differs"); check_idle(gpus)
    model = MODELS[base_key]; logs = RUN_ROOT / "logs"; logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    log = logs / f"server-policy-rollout-{label}.log"; ensure(not log.exists(), "policy rollout server log already exists")
    compile_config = '{"pass_config":{"fuse_allreduce_rms":false}}'
    command = [str(VLLM), "serve", str(model["path"]), "--served-model-name", model["id"], "--host", "127.0.0.1", "--port", str(port),
               "--tensor-parallel-size", str(tensor_parallel_size), "--attention-backend", "FLASH_ATTN", "--max-model-len", str(cfg["runtime"]["rollout_max_model_len"]),
               "--max-num-seqs", str(cfg["runtime"]["rollout_max_num_seqs"]), "--max-num-batched-tokens", "65536", "--gpu-memory-utilization", str(cfg["runtime"]["gpu_memory_utilization"]),
               "--disable-custom-all-reduce", "--enable-lora", "--max-loras", "1", "--max-lora-rank", "32", "--generation-config", "vllm",
               "--enable-prefix-caching", "--no-enable-flashinfer-autotune", "--compilation-config", compile_config]
    visible = ",".join(map(str, gpus))
    environment = {**os.environ, **base_env(visible), "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"}
    ledger({"stage": f"policy-rollout-server-{label}", "event": "start", "command": command, "resource_scope": gpus,
            "execution_env": {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"}})
    handle = log.open("x", encoding="utf-8"); process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    attestation = RUN_ROOT / "attestations" / f"policy-rollout-{label}.json"; attestation.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        for _ in range(360):
            if process.poll() is not None:
                raise RunnerError("policy rollout server exited before health gate")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RunnerError("policy rollout server health timeout")
        environ = Path(f"/proc/{process.pid}/environ").read_bytes(); ensure(f"CUDA_VISIBLE_DEVICES={visible}".encode() in environ, "policy rollout CUDA visibility differs")
        atomic_json(attestation, {"schema_version": "mal2026-rlaif-grpo-vllm-policy-server-attestation-v1", "server_host": "127.0.0.1", "server_port": port,
                                  "physical_gpus": gpus, "tensor_parallel_size": tensor_parallel_size, "max_model_len": cfg["runtime"]["rollout_max_model_len"], "max_num_seqs": cfg["runtime"]["rollout_max_num_seqs"],
                                  "model_id": model["id"], "model_revision": model["revision"], "dynamic_lora": True, "structured_outputs_json_schema": True,
                                  "enforce_eager": False, "server_process_environment_verified": True})
        run_stage(f"policy-rollout-synthetic-{label}", [str(PY), "scripts/verify_rlaif_policy_server.py", "--endpoint", f"http://127.0.0.1:{port}", "--adapter", str(sft_adapter(base_key, task))], base_env(visible))
        ledger({"stage": f"policy-rollout-server-{label}", "event": "health_dynamic_lora_structured_json_pass", "resource_scope": gpus, "raw_response_persisted": False})
        yield attestation
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=30)
        handle.close(); ledger({"stage": f"policy-rollout-server-{label}", "event": "stopped", "resource_scope": gpus})


@contextmanager
def generation_server(*, base_key: str, adapter: Path, alias: str, port: int, label: str) -> Iterator[Path]:
    check_idle([0, 1, 2, 3]); model = MODELS[base_key]; logs = RUN_ROOT / "logs"; logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    log = logs / f"server-generation-{label}.log"; ensure(not log.exists(), "generation-server log already exists")
    compile_config = '{"pass_config":{"fuse_allreduce_rms":false}}'
    command = [str(VLLM), "serve", str(model["path"]), "--served-model-name", model["id"], "--host", "127.0.0.1", "--port", str(port), "--tensor-parallel-size", "4",
               "--attention-backend", "FLASH_ATTN", "--max-model-len", "3072", "--gpu-memory-utilization", "0.9", "--disable-custom-all-reduce", "--enable-lora", "--max-lora-rank", "32",
               "--lora-modules", f"{alias}={adapter}", "--generation-config", "vllm", "--enable-prefix-caching", "--no-enable-flashinfer-autotune", "--compilation-config", compile_config]
    environment = {**os.environ, **base_env("0,1,2,3")}
    ledger({"stage": f"generation-server-{label}", "event": "start", "command": command, "resource_scope": [0, 1, 2, 3]})
    handle = log.open("x", encoding="utf-8"); process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    attestation = RUN_ROOT / "attestations" / f"generation-{label}.json"; attestation.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        for _ in range(360):
            if process.poll() is not None:
                raise RunnerError("generation server exited before health gate")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RunnerError("generation server health timeout")
        environ = Path(f"/proc/{process.pid}/environ").read_bytes(); ensure(b"CUDA_VISIBLE_DEVICES=0,1,2,3" in environ, "generation server CUDA visibility differs")
        atomic_json(attestation, {"schema_version": "mal2026-rlaif-grpo-generation-server-attestation-v1", "server_host": "127.0.0.1", "server_port": port, "physical_gpus": [0, 1, 2, 3],
                                  "tensor_parallel_size": 4, "max_model_len": 3072, "model_id": model["id"], "model_revision": model["revision"], "adapter_path": str(adapter.resolve()), "adapter_alias": alias,
                                  "server_process_environment_verified": True})
        yield attestation
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=30)
        handle.close(); ledger({"stage": f"generation-server-{label}", "event": "stopped", "resource_scope": [0, 1, 2, 3]})


@contextmanager
def frozen_v6_server(*, port: int, label: str) -> Iterator[Path]:
    cfg = config(); check_idle([0, 1, 2, 3]); qwen_path = ROOT / cfg["judge"]["model_path"]
    logs = RUN_ROOT / "logs"; logs.mkdir(mode=0o700, parents=True, exist_ok=True); log = logs / f"server-v6-{label}.log"; ensure(not log.exists(), "v6-server log already exists")
    compile_config = '{"pass_config":{"fuse_allreduce_rms":false}}'
    enforce_eager = fixed_v6_enforce_eager(cfg)
    command = [str(VLLM), "serve", str(qwen_path), "--served-model-name", cfg["judge"]["model_id"], "--host", "127.0.0.1", "--port", str(port), "--tensor-parallel-size", "1", "--data-parallel-size", "4",
               "--attention-backend", "FLASH_ATTN", "--max-model-len", "4096", "--max-num-seqs", "192", "--max-num-batched-tokens", "65536", "--gpu-memory-utilization", "0.9", "--disable-custom-all-reduce",
               "--gdn-prefill-backend", "triton", "--generation-config", "vllm", "--enable-prefix-caching", "--no-enable-flashinfer-autotune", "--compilation-config", compile_config]
    if enforce_eager:
        command.append("--enforce-eager")
    environment = {**os.environ, **base_env("0,1,2,3"), "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "0"}
    ledger({"stage": f"v6-server-{label}", "event": "start", "command": command, "resource_scope": [0, 1, 2, 3], "execution_env": {"VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "0"}})
    handle = log.open("x", encoding="utf-8"); process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    attestation = RUN_ROOT / "attestations" / f"v6-{label}.json"; attestation.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        for _ in range(360):
            if process.poll() is not None:
                raise RunnerError("frozen-v6 judge server exited before health gate")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RunnerError("frozen-v6 judge server health timeout")
        environ = Path(f"/proc/{process.pid}/environ").read_bytes(); ensure(b"CUDA_VISIBLE_DEVICES=0,1,2,3" in environ, "v6 server CUDA visibility differs")
        atomic_json(attestation, {"schema_version": "mal2026-rlaif-grpo-v6-evaluation-server-attestation-v1", "server_host": "127.0.0.1", "server_port": port, "physical_gpus": [0, 1, 2, 3], "tensor_parallel_size": 1,
                                  "data_parallel_size": 4, "max_model_len": 4096, "max_num_seqs_per_rank": 192, "enforce_eager": enforce_eager, "fixed_v6_config_sha256": cfg["fixed_v6_config_sha256"], "server_process_environment_verified": True})
        yield attestation
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=30)
        handle.close(); ledger({"stage": f"v6-server-{label}", "event": "stopped", "resource_scope": [0, 1, 2, 3]})


def sft_adapter(base_key: str, task: str) -> Path:
    value = ROOT / "outputs" / "api-rationale-sft-v1" / f"api-rationale-sft-v1-{base_key}-{task}-001" / "adapter"
    ensure(value.is_dir() and (value / "adapter_config.json").is_file(), "completed source SFT adapter is unavailable")
    return value


def arm_id(base_key: str, task: str, arm: str, phase: str) -> str:
    # A preserved failed preflight must never block a fresh same-protocol
    # launcher from writing its own ignored adapter/runtime directory.
    attempt = RUNTIME_ID.rsplit("-", 1)[1]
    return f"{config()['run_id_prefix']}{base_key}-{task}-{arm}-{phase}-{attempt}"


def arm_dir(base_key: str, task: str, arm: str, phase: str) -> Path:
    return ARM_ROOT / arm_id(base_key, task, arm, phase)


def training_config(base_key: str, task: str, arm: str, phase: str, endpoint: str, attestation: Path) -> dict[str, Any]:
    cfg = config(); run_id = arm_id(base_key, task, arm, phase)
    if phase == "full":
        values = {"train_limit": 1920, "max_steps": 480, "per_device_train_batch_size": cfg["policy"]["per_device_train_batch_size_full"], "generation_batch_size": 64, "steps_per_generation": 4}
    elif phase == "pilot":
        ensure(cfg["schema_version"].endswith(PILOT_SCHEMA_SUFFIXES), "pilot is only authorized by v3--v7")
        values = {"train_limit": 320, "max_steps": 80, "per_device_train_batch_size": cfg["policy"]["per_device_train_batch_size_full"], "generation_batch_size": 64, "steps_per_generation": 4}
    else:
        values = {"train_limit": 4, "max_steps": 1, "per_device_train_batch_size": 4, "generation_batch_size": 4, "steps_per_generation": 1}
    return {"schema_version": "mal2026-rlaif-grpo-run-v1", "run_id": run_id, "base_key": base_key, "task": task, "arm": arm, "phase": phase, "output_dir": str(arm_dir(base_key, task, arm, phase).resolve()),
            "source_adapter": str(sft_adapter(base_key, task).resolve()), "reward_endpoint": endpoint, "reward_server_attestation": str(attestation.resolve()), "seed": cfg["policy"]["seed"], **values}


def baseline_paths(base_key: str, task: str) -> tuple[str, Path, Path]:
    kind = "bundle" if task == "bundle" else "axis_triplet"
    generation_suffix = "bundle-validation-003" if task == "bundle" else "axis_triplet-validation-002"
    generation = GEN_ROOT / f"api-rationale-generation-v1-{base_key}-{generation_suffix}"
    judge = JUDGE_ROOT / f"api-rationale-judge-v1-{base_key}-{kind}-validation-002"
    ensure((generation / "aggregate_generation_report.json").is_file() and (judge / "aggregate_judge_report.json").is_file(), "frozen SFT baseline artifacts are unavailable")
    return kind, generation, judge


def evaluation_config(base_key: str, task: str, arm: str, phase: str = "full") -> dict[str, Any]:
    ensure(phase in {"full", "pilot"} and (phase == "full" or config()["schema_version"].endswith(PILOT_SCHEMA_SUFFIXES)), "evaluation phase is not authorized")
    kind, generation, judge = baseline_paths(base_key, task)
    suffix = "validation-001" if phase == "full" else "pilot-validation-001"
    run_id = f"{config()['run_id_prefix']}{base_key}-{task}-{arm}-{suffix}"; training = arm_dir(base_key, task, arm, phase)
    schema_version = f"mal2026-rlaif-grpo-evaluation-{STUDY_ID.rsplit('-', 1)[-1]}"
    return {"schema_version": schema_version, "run_id": run_id, "base_key": base_key, "task": task, "arm": arm, "rl_phase": phase, "rl_training_dir": str(training.resolve()),
            "output_dir": str((EVALUATION_ROOT / run_id).resolve()), "generation_adapter_alias": f"rlaif_{base_key}_{task}_{arm}", "baseline_kind": kind,
            "baseline_generation_dir": str(generation.resolve()), "baseline_judge_dir": str(judge.resolve()), "deterministic_max_new_tokens": 512 if task == "bundle" else 192, "character_limit": 192}


def verify_training(base_key: str, task: str, arm: str, phase: str) -> dict[str, Any]:
    path = arm_dir(base_key, task, arm, phase) / "training_complete.json"; ensure(path.is_file(), f"RLAIF completion missing: {path}")
    value = read_json(path); ensure(value.get("status") == "completed" and all(value.get(key) == expected for key, expected in (("base_key", base_key), ("task", task), ("arm", arm), ("phase", phase))), "RLAIF completion provenance differs")
    ensure((arm_dir(base_key, task, arm, phase) / "adapter" / "adapter_config.json").is_file(), "RLAIF adapter is missing")
    return value


def train_arm(base_key: str, task: str, arm: str, phase: str, endpoint: str, attestation: Path, policy_gpus: str, *, rollout_endpoint: str | None = None, rollout_attestation: Path | None = None) -> dict[str, Any]:
    output = arm_dir(base_key, task, arm, phase); ensure(not output.exists(), f"refusing to overwrite RLAIF arm {output}")
    runtime = write_config(f"train-{base_key}-{task}-{arm}-{phase}", training_config(base_key, task, arm, phase, endpoint, attestation))
    world_size = len([gpu for gpu in policy_gpus.split(",") if gpu])
    command = [str(PY), "scripts/train_rlaif_grpo.py", "--config", str(runtime)] if phase == "gpu0_actual" or world_size == 1 else [str(PY), "-m", "torch.distributed.run", "--nproc_per_node", str(world_size), "scripts/train_rlaif_grpo.py", "--config", str(runtime)]
    environment = base_env(policy_gpus)
    if config()["schema_version"].endswith(STRUCTURED_SCHEMA_SUFFIXES):
        ensure(rollout_endpoint is not None and rollout_attestation is not None, "structured-rollout training requires a vLLM rollout server")
        environment.update({"MAL2026_RLAIF_CONFIG": str(CONFIG.relative_to(ROOT)), "MAL2026_RLAIF_ROLLOUT_ENDPOINT": rollout_endpoint,
                            "MAL2026_RLAIF_ROLLOUT_ATTESTATION": str(rollout_attestation.resolve()), "MAL2026_RLAIF_ROLLOUT_SYNC_DIR": str((output / "rollout_sync").resolve())})
    allocator_conf = config()["runtime"].get("policy_training_cuda_alloc_conf")
    if allocator_conf is not None:
        ensure(config()["schema_version"].endswith(("-v7", "-v8")) and allocator_conf == "expandable_segments:True", "unrecognized policy-training allocator configuration")
        environment["PYTORCH_CUDA_ALLOC_CONF"] = str(allocator_conf)
    run_stage(f"train-{base_key}-{task}-{arm}-{phase}", command, environment)
    completed = verify_training(base_key, task, arm, phase); ledger({"stage": f"train-{base_key}-{task}-{arm}-{phase}", "event": "hard_gates_pass", "resource_scope": policy_gpus, "evidence_ref": str((output / "training_complete.json").relative_to(ROOT)), "decision": "continue"})
    return completed


def evaluate_arm(base_key: str, task: str, arm: str, port_generation: int, port_judge: int, *, phase: str = "full") -> dict[str, Any]:
    completed = verify_training(base_key, task, arm, phase); runtime = write_config(f"eval-{base_key}-{task}-{arm}-{phase}", evaluation_config(base_key, task, arm, phase)); output = Path(read_json(runtime)["output_dir"])
    ensure(not output.exists(), "refusing to overwrite RLAIF evaluation output")
    alias = f"rlaif_{base_key}_{task}_{arm}"; adapter = arm_dir(base_key, task, arm, phase) / "adapter"
    with generation_server(base_key=base_key, adapter=adapter, alias=alias, port=port_generation, label=f"{base_key}-{task}-{arm}") as attestation:
        run_stage(f"generate-{base_key}-{task}-{arm}", [str(PY), "scripts/evaluate_rlaif_grpo.py", "generate", "--config", str(runtime), "--endpoint", f"http://127.0.0.1:{port_generation}", "--server-attestation", str(attestation)], base_env("0,1,2,3"))
    with frozen_v6_server(port=port_judge, label=f"{base_key}-{task}-{arm}") as attestation:
        run_stage(f"judge-{base_key}-{task}-{arm}", [str(PY), "scripts/evaluate_rlaif_grpo.py", "judge", "--config", str(runtime), "--endpoint", f"http://127.0.0.1:{port_judge}", "--server-attestation", str(attestation)], base_env("0,1,2,3"))
    report = read_json(output / "aggregate_judge_report.json"); ensure(report.get("status") == "completed" and all(report.get("hard_gates", {}).values()), "post-RL frozen-v6 judge gate failed")
    ledger({"stage": f"evaluate-{base_key}-{task}-{arm}-{phase}", "event": "next_stage_complete", "resource_scope": [0, 1, 2, 3], "evidence_ref": str((output / "aggregate_judge_report.json").relative_to(ROOT)), "decision": "continue"})
    return report


def run_gpu0_preflight() -> None:
    # GPU0 first: standalone Qwen health, then a constrained policy rollout on
    # GPU0, with its actual one-update trainer and reward server disjoint.
    with reward_server(gpus=[0], data_parallel_size=1, port=18510, label="gpu0-health"):
        pass
    schema = config()["schema_version"]
    if schema.endswith(TP2_SCHEMA_SUFFIXES):
        # v6 reserves a TP=2 policy rollout on GPUs 0--1, one policy-training
        # rank on GPU 2, and the reward model on GPU 3.  The ordinary v2--v5
        # preflight topology is not valid for this placement, so bind the
        # actual-update gate to the same topology used by the later full arm.
        policy_gpus = ",".join(map(str, config()["runtime"]["full_policy_gpus"]))
        reward_gpus = list(config()["runtime"]["full_reward_gpus"])
        with policy_rollout_server(base_key="midm2_base", task="bundle", port=18512, label="midm-gpu0-two-arm") as rollout_attestation:
            with reward_server(gpus=reward_gpus, data_parallel_size=1, port=18511, label="midm-gpu0-two-arm") as attestation:
                for arm in ARMS:
                    train_arm("midm2_base", "bundle", arm, "gpu0_actual", "http://127.0.0.1:18511", attestation, policy_gpus,
                              rollout_endpoint="http://127.0.0.1:18512", rollout_attestation=rollout_attestation)
    elif schema.endswith(("-v2", "-v3", "-v4", "-v5")):
        # The two arms begin from the same source adapter, so retain both
        # servers across their two one-update gates and reload only the LoRA.
        with policy_rollout_server(base_key="midm2_base", task="bundle", port=18512, label="midm-gpu0-two-arm") as rollout_attestation:
            with reward_server(gpus=[1], data_parallel_size=1, port=18511, label="midm-gpu0-two-arm") as attestation:
                for arm in ARMS:
                    train_arm("midm2_base", "bundle", arm, "gpu0_actual", "http://127.0.0.1:18511", attestation, "2", rollout_endpoint="http://127.0.0.1:18512", rollout_attestation=rollout_attestation)
    else:
        for arm in ARMS:
            with reward_server(gpus=[1], data_parallel_size=1, port=18511, label=f"midm-gpu0-{arm}") as attestation:
                train_arm("midm2_base", "bundle", arm, "gpu0_actual", "http://127.0.0.1:18511", attestation, "0")
    scope = [0, 1, 2, 3] if schema.endswith(TP2_SCHEMA_SUFFIXES) else ([0, 1, 2] if schema.endswith(("-v2", "-v3", "-v4", "-v5")) else [0, 1])
    ledger({"stage": "gpu0_actual_preflight", "event": "smoke_pass", "resource_scope": scope, "gpu_scope_authorization": "current-user explicit GPUs 0-3 authorization", "decision": "continue_to_midm_full"})


def run_full_arm(base_key: str, task: str, arm: str, port: int) -> dict[str, Any]:
    schema = config()["schema_version"]
    if schema.endswith(TP2_SCHEMA_SUFFIXES):
        policy_gpus = ",".join(map(str, config()["runtime"]["full_policy_gpus"]))
        reward_gpus = list(config()["runtime"]["full_reward_gpus"])
        with policy_rollout_server(base_key=base_key, task=task, port=port + 50, label=f"{base_key}-{task}-{arm}") as rollout_attestation:
            with reward_server(gpus=reward_gpus, data_parallel_size=1, port=port, label=f"{base_key}-{task}-{arm}") as attestation:
                train_arm(base_key, task, arm, "full", f"http://127.0.0.1:{port}", attestation, policy_gpus,
                          rollout_endpoint=f"http://127.0.0.1:{port + 50}", rollout_attestation=rollout_attestation)
    elif schema.endswith(("-v2", "-v3", "-v4", "-v5")):
        with policy_rollout_server(base_key=base_key, task=task, port=port + 50, label=f"{base_key}-{task}-{arm}") as rollout_attestation:
            with reward_server(gpus=[3], data_parallel_size=1, port=port, label=f"{base_key}-{task}-{arm}") as attestation:
                train_arm(base_key, task, arm, "full", f"http://127.0.0.1:{port}", attestation, "1,2", rollout_endpoint=f"http://127.0.0.1:{port + 50}", rollout_attestation=rollout_attestation)
    else:
        with reward_server(gpus=[0, 3], data_parallel_size=2, port=port, label=f"{base_key}-{task}-{arm}") as attestation:
            train_arm(base_key, task, arm, "full", f"http://127.0.0.1:{port}", attestation, "1,2")
    return evaluate_arm(base_key, task, arm, port + 100, port + 200)


def run_full_task_arms(base_key: str, task: str, port: int) -> None:
    """Train both reward-estimator arms while reusing costly vLLM servers."""
    schema = config()["schema_version"]
    if schema.endswith(TP2_SCHEMA_SUFFIXES):
        policy_gpus = ",".join(map(str, config()["runtime"]["full_policy_gpus"]))
        reward_gpus = list(config()["runtime"]["full_reward_gpus"])
        with policy_rollout_server(base_key=base_key, task=task, port=port + 50, label=f"{base_key}-{task}-two-arm") as rollout_attestation:
            with reward_server(gpus=reward_gpus, data_parallel_size=1, port=port, label=f"{base_key}-{task}-two-arm") as attestation:
                for arm in ARMS:
                    train_arm(base_key, task, arm, "full", f"http://127.0.0.1:{port}", attestation, policy_gpus,
                              rollout_endpoint=f"http://127.0.0.1:{port + 50}", rollout_attestation=rollout_attestation)
        for index, arm in enumerate(ARMS):
            evaluate_arm(base_key, task, arm, port + 100 + index * 10, port + 200 + index * 10)
    elif schema.endswith(("-v2", "-v3", "-v4", "-v5")):
        with policy_rollout_server(base_key=base_key, task=task, port=port + 50, label=f"{base_key}-{task}-two-arm") as rollout_attestation:
            with reward_server(gpus=[3], data_parallel_size=1, port=port, label=f"{base_key}-{task}-two-arm") as attestation:
                for arm in ARMS:
                    train_arm(base_key, task, arm, "full", f"http://127.0.0.1:{port}", attestation, "1,2", rollout_endpoint=f"http://127.0.0.1:{port + 50}", rollout_attestation=rollout_attestation)
        for index, arm in enumerate(ARMS):
            evaluate_arm(base_key, task, arm, port + 100 + index * 10, port + 200 + index * 10)
    else:
        for index, arm in enumerate(ARMS):
            run_full_arm(base_key, task, arm, port + index * 10)


def require_gpu0_preflight() -> None:
    for arm in ARMS:
        verify_training("midm2_base", "bundle", arm, "gpu0_actual")


def run_midm() -> None:
    require_gpu0_preflight()
    run_full_task_arms("midm2_base", "bundle", 18520)
    ledger({"stage": "midm_bundle_two_arm_evaluation", "event": "next_stage_complete", "resource_scope": [0, 1, 2, 3], "decision": "continue_to_remaining_models"})


def run_midm_pilot() -> None:
    """Run the fixed 320-group, two-arm v3--v7 decision experiment before full scale."""
    ensure(config()["schema_version"].endswith(PILOT_SCHEMA_SUFFIXES), "Midm pilot is only authorized by v3--v7")
    if config()["schema_version"].endswith(("-v4", "-v5", "-v6", "-v7", "-v8")):
        # v3's single 64-choice gate passed but exact DDP rollouts exhibited
        # long structured-decoding tails.  Every post-v3 repair must therefore
        # prove its changed grammar on one fresh full batch *before* any policy update. Keeping
        # the subsequently required Qwen server open avoids a second costly
        # load and lets the two actual gates and both arms use the exact same
        # server topology.
        version = config()["schema_version"].rsplit("-", 1)[-1]
        policy_gpus = ",".join(map(str, config()["runtime"]["full_policy_gpus"]))
        rollout_gpus = ",".join(map(str, config()["runtime"]["full_rollout_gpus"]))
        with policy_rollout_server(base_key="midm2_base", task="bundle", port=18570, label=f"midm2_base-bundle-{version}-pilot-two-arm") as rollout_attestation:
            with reward_server(gpus=[3], data_parallel_size=1, port=18520, label=f"midm2_base-bundle-{version}-pilot-two-arm") as attestation:
                # The aggregate-only gate needs at least 16 source prompts;
                # bind it to the declared pilot population but do not create
                # or train its output directory.
                gate_runtime = write_config(f"policy-batch-gate-midm2_base-bundle-{version}", training_config("midm2_base", "bundle", "all5", "pilot", "http://127.0.0.1:18520", attestation))
                gate_report = RUN_ROOT / "aggregate" / f"midm2_bundle_{version}_full_batch_gate.json"
                run_stage(f"policy-batch-gate-midm2_base-bundle-{version}", [str(PY), "scripts/benchmark_rlaif_policy_rollout.py", "--run-config", str(gate_runtime), "--endpoint", "http://127.0.0.1:18570", "--adapter", str(sft_adapter("midm2_base", "bundle")), "--report", str(gate_report), "--source-prompts", "16", "--max-wall-seconds", "240"], base_env(rollout_gpus))
                gate = read_json(gate_report)
                ensure(gate.get("status") == "passed" and gate.get("policy_completions") == 64 and gate.get("parse_valid") == 64 and gate.get("structured_json_schema_field_max_length_enforced") is False, "post-v3 full policy batch gate failed")
                ledger({"stage": f"policy-batch-gate-midm2_base-bundle-{version}", "event": "hard_gates_pass", "resource_scope": list(config()["runtime"]["full_rollout_gpus"]), "evidence_ref": str(gate_report.relative_to(ROOT)), "decision": "continue_to_one_update_gates"})
                for arm in ARMS:
                    train_arm("midm2_base", "bundle", arm, "gpu0_actual", "http://127.0.0.1:18520", attestation, policy_gpus, rollout_endpoint="http://127.0.0.1:18570", rollout_attestation=rollout_attestation)
                for arm in ARMS:
                    train_arm("midm2_base", "bundle", arm, "pilot", "http://127.0.0.1:18520", attestation, policy_gpus, rollout_endpoint="http://127.0.0.1:18570", rollout_attestation=rollout_attestation)
        for index, arm in enumerate(ARMS):
            evaluate_arm("midm2_base", "bundle", arm, 18620 + index * 10, 18720 + index * 10, phase="pilot")
        ledger({"stage": "midm_bundle_two_arm_pilot_evaluation", "event": "next_stage_complete", "resource_scope": [0, 1, 2, 3], "decision": "compare_to_frozen_sft_before_full_promotion"})
        return
    run_gpu0_preflight()
    with policy_rollout_server(base_key="midm2_base", task="bundle", port=18570, label="midm2_base-bundle-pilot-two-arm") as rollout_attestation:
        with reward_server(gpus=[3], data_parallel_size=1, port=18520, label="midm2_base-bundle-pilot-two-arm") as attestation:
            for arm in ARMS:
                train_arm("midm2_base", "bundle", arm, "pilot", "http://127.0.0.1:18520", attestation, "1,2", rollout_endpoint="http://127.0.0.1:18570", rollout_attestation=rollout_attestation)
    for index, arm in enumerate(ARMS):
        evaluate_arm("midm2_base", "bundle", arm, 18620 + index * 10, 18720 + index * 10, phase="pilot")
    ledger({"stage": "midm_bundle_two_arm_pilot_evaluation", "event": "next_stage_complete", "resource_scope": [0, 1, 2, 3], "decision": "compare_to_frozen_sft_before_full_promotion"})


def require_midm() -> None:
    for arm in ARMS:
        path = EVALUATION_ROOT / f"{config()['run_id_prefix']}midm2_base-bundle-{arm}-validation-001" / "aggregate_judge_report.json"
        value = read_json(path); ensure(value.get("status") == "completed" and all(value.get("hard_gates", {}).values()), "Midm full evaluation is not complete")


def completed_task_evaluations(base_key: str, task: str) -> bool:
    """Return true only for a fully frozen-v6-evaluated two-arm task.

    A later fresh runtime lineage may safely resume after a preserved failed
    arm.  Evaluation identities do not include the runtime suffix, so verified
    already-complete pairs must be reused rather than regenerated or
    overwritten.  Incomplete pairs intentionally return false and are trained
    in the fresh lineage.
    """
    cfg = config()
    for arm in ARMS:
        run_id = f"{cfg['run_id_prefix']}{base_key}-{task}-{arm}-validation-001"
        root = EVALUATION_ROOT / run_id
        judge_path, generation_path, manifest_path = root / "aggregate_judge_report.json", root / "aggregate_generation_report.json", root / "manifest.json"
        if not all(path.is_file() for path in (judge_path, generation_path, manifest_path)):
            return False
        value, generation, manifest = read_json(judge_path), read_json(generation_path), read_json(manifest_path)
        expected_judge_counts = {"expected_calls": 20000, "observations": 20000, "scored": 20000, "schema_valid": 20000, "abstain": 0, "generated_candidates": 400}
        expected_generation_counts = {"expected": 400, "observations": 400, "parse_valid": 400}
        privacy_keys = ("source_writing_scores_read_or_prompted", "candidate_scores_read_or_prompted", "raw_prompts_or_completions_tracked")
        ensure(value.get("status") == "completed" and value.get("run_id") == run_id and value.get("base_key") == base_key and value.get("task") == task and value.get("arm") == arm,
               "existing frozen-v6 evaluation identity differs")
        ensure(value.get("fixed_v6_config_sha256") == cfg["fixed_v6_config_sha256"] and value.get("counts") == expected_judge_counts and value.get("failure_categories") == {} and all(value.get("hard_gates", {}).values()),
               "existing frozen-v6 evaluation aggregate gate differs")
        ensure(all(value.get(key) is False for key in privacy_keys), "existing frozen-v6 evaluation privacy contract differs")
        ensure(generation.get("status") == "completed" and generation.get("run_id") == run_id and generation.get("base_key") == base_key and generation.get("task") == task and generation.get("arm") == arm,
               "existing post-RL generation identity differs")
        ensure(generation.get("counts") == expected_generation_counts and generation.get("failure_categories") == {} and all(generation.get("hard_gates", {}).values()) and all(generation.get(key) is False for key in privacy_keys),
               "existing post-RL generation aggregate gate differs")
        ensure(value.get("generation_report_sha256") == sha(generation_path) and manifest.get("aggregate_generation_report_sha256") == sha(generation_path) and manifest.get("aggregate_judge_report_sha256") == sha(judge_path),
               "existing evaluation artifact hashes differ")
        manifest_config = manifest.get("config")
        ensure(isinstance(manifest_config, dict) and manifest.get("status") == "completed" and manifest.get("run_id") == run_id and manifest.get("rlaif_config_sha256") == sha(CONFIG) and
               manifest_config.get("run_id") == run_id and manifest_config.get("base_key") == base_key and manifest_config.get("task") == task and manifest_config.get("arm") == arm and
               manifest_config.get("rl_phase") == "full" and manifest_config.get("output_dir") == str(root.resolve()) and manifest_config.get("deterministic_max_new_tokens") == (512 if task == "bundle" else 192) and manifest_config.get("character_limit") == 192,
               "existing evaluation manifest provenance differs")
        training_dir = Path(str(manifest_config.get("rl_training_dir", ""))).resolve()
        training_path = training_dir / "training_complete.json"
        ensure(training_dir.parent == ARM_ROOT.resolve() and training_dir.is_dir() and not training_dir.is_symlink() and training_path.is_file() and (training_dir / "adapter" / "adapter_config.json").is_file(),
               "existing RLAIF training provenance is unavailable")
        training = read_json(training_path)
        training_sha = sha(training_path)
        ensure(training.get("status") == "completed" and all(training.get(key) == expected for key, expected in (("base_key", base_key), ("task", task), ("arm", arm), ("phase", "full"))) and
               training.get("rlaif_config_sha256") == sha(CONFIG) and generation.get("rlaif_training_complete_sha256") == training_sha and manifest.get("rlaif_training_complete_sha256") == training_sha,
               "existing RLAIF training/evaluation binding differs")
    return True


def run_remaining() -> None:
    require_midm(); index = 0
    for base_key in ("ax4_light", "phi4_mini", "midm2_base"):
        for task in TASKS:
            if (base_key, task) == ("midm2_base", "bundle"):
                continue
            port = 18600 + index * 20
            if completed_task_evaluations(base_key, task):
                ledger({"stage": f"resume-{base_key}-{task}", "event": "verified_existing_two_arm_evaluation", "resource_scope": "none",
                        "decision": "skip_completed_task", "evidence_ref": str((EVALUATION_ROOT / f"{config()['run_id_prefix']}{base_key}-{task}-all5-validation-001" / "aggregate_judge_report.json").relative_to(ROOT))})
            else:
                run_full_task_arms(base_key, task, port)
            index += 1
    final_summary()


def final_summary() -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for base_key in ("midm2_base", "ax4_light", "phi4_mini"):
        for task in TASKS:
            for arm in ARMS:
                run_id = f"{config()['run_id_prefix']}{base_key}-{task}-{arm}-validation-001"; report = read_json(EVALUATION_ROOT / run_id / "aggregate_judge_report.json")
                values.append({"base_key": base_key, "task": task, "arm": arm, "status": report.get("status"), "axis_means": report.get("axis_means"), "macro_mean": report.get("macro_mean"), "primary_requested_axes": report.get("primary_requested_axes"), "primary_paired_delta": report.get("primary_paired_delta"), "hard_gates": report.get("hard_gates")})
    destination = AGGREGATE_ROOT / f"{STUDY_ID}-{RUNTIME_ID}.final-summary.json"; ensure(not destination.exists(), "RLAIF final aggregate already exists")
    result = {"schema_version": f"mal2026-{STUDY_ID}-final-summary-v1", "status": "completed" if all(item["status"] == "completed" and all(item["hard_gates"].values()) for item in values) else "failed_gates", "rlaif_config_sha256": sha(CONFIG), "fixed_v6_config_sha256": config()["fixed_v6_config_sha256"], "arms": values, "raw_prompts_or_completions_tracked": False}
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True); atomic_json(destination, result)
    ledger({"stage": "aggregate", "event": "next_stage_complete", "evidence_ref": str(destination.relative_to(ROOT)), "resource_scope": "none", "decision": "complete"})
    return result


def initialize(mode: str) -> None:
    cfg = config(); RUN_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True); manifest = RUN_ROOT / "runner_manifest.json"
    if not manifest.exists():
        atomic_json(manifest, {"schema_version": "mal2026-rlaif-grpo-prompt-ensemble-v1-runner-v1", "status": "running", "runtime_id": RUNTIME_ID, "created_at": now(), "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "rlaif_config_sha256": sha(CONFIG), "fixed_v6_config_sha256": cfg["fixed_v6_config_sha256"], "physical_gpus": [0, 1, 2, 3], "raw_prompts_or_completions_tracked": False})
    else:
        value = read_json(manifest); ensure(value.get("rlaif_config_sha256") == sha(CONFIG) and value.get("fixed_v6_config_sha256") == cfg["fixed_v6_config_sha256"], "runner manifest config binding differs")
    ledger({"stage": "runner", "event": "start", "mode": mode, "resource_scope": [0, 1, 2, 3], "gpu_scope_authorization": "current-user explicit GPUs 0-3 authorization", "decision": "continue"})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("gpu0-preflight", "midm-pilot", "midm", "remaining", "all", "aggregate")); args = parser.parse_args(); initialize(args.mode)
    if args.mode == "gpu0-preflight":
        run_gpu0_preflight()
    elif args.mode == "midm-pilot":
        run_midm_pilot()
    elif args.mode == "midm":
        run_midm()
    elif args.mode == "remaining":
        run_remaining()
    elif args.mode == "all":
        run_gpu0_preflight(); run_midm(); run_remaining()
    else:
        final_summary()


if __name__ == "__main__":
    main()
