#!/usr/bin/env python3
"""Durable sequential runner for the user-authorized API-rationale matrix.

All detailed data/configuration files and logs are beneath ignored outputs.
The runner records only aggregate status, metrics, checksums, and commands in
its ledger; restricted writing/rationale text never enters tracked files.
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
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv-standard" / "bin" / "python"
VLLM = ROOT / ".venv-standard" / "bin" / "vllm"
RUN_BASE = ROOT / "outputs" / "api-rationale-sft-score-regression-v1" / "20260721-001"
RUN_ROOT = RUN_BASE
SFT_ROOT = ROOT / "outputs" / "api-rationale-sft-v1"
GEN_ROOT = ROOT / "data" / "processed" / "restricted" / "openai_rationale_batches" / "openai-rationale-terra-full-20260719-001" / "decoder_generation_v1"
JUDGE_ROOT = ROOT / "data" / "processed" / "restricted" / "openai_rationale_batches" / "openai-rationale-terra-full-20260719-001" / "decoder_judge_v1"
REG_ROOT = ROOT / "outputs" / "api-score-regression-v1"
REG_EVAL_ROOT = ROOT / "outputs" / "api-score-regression-evals-v1"
AGGREGATE_ROOT = ROOT / "outputs" / "aggregate-reports"
V6_CONFIG = ROOT / "configs" / "qwen36_native_fp8_vllm_rationale_only_score5x10.v6.json"
QWEN_JUDGE_PATH = ROOT / "outputs" / "model-cache" / "Qwen--Qwen3.6-35B-A3B-FP8-native-fp8-v1"
QWEN_JUDGE_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
MODELS = {
    "ax4_light": {"id": "skt/A.X-4.0-Light", "revision": "ba21c20ea1b31ded1ec3e2fb432335077dc4be98", "path": ROOT / "outputs" / "model-cache" / "skt--A.X-4.0-Light-ba21c20ea1b31ded1ec3e2fb432335077dc4be98"},
    "phi4_mini": {"id": "microsoft/Phi-4-mini-instruct", "revision": "cfbefacb99257ffa30c83adab238a50856ac3083", "path": ROOT / "outputs" / "model-cache" / "microsoft--Phi-4-mini-instruct-cfbefacb99257ffa30c83adab238a50856ac3083"},
    "midm2_base": {"id": "K-intelligence/Midm-2.0-Base-Instruct", "revision": "35479c5fc9a18a5db7cc6dbadcf1db68db7beab0", "path": ROOT / "outputs" / "model-cache" / "K-intelligence--Midm-2.0-Base-Instruct-35479c5fc9a18a5db7cc6dbadcf1db68db7beab0"},
}
ENCODERS = {
    "qwen25_7b": {"id": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28", "path": ROOT / "outputs" / "model-cache" / "Qwen--Qwen2.5-7B-Instruct-a09a35458c702b33eeacc393d103063234e8bc28"},
    "kure_v1": {"id": "nlpai-lab/KURE-v1", "revision": "d14c8a9423946e268a0c9952fecf3a7aabd73bd9", "path": ROOT / "outputs" / "model-cache" / "nlpai-lab--KURE-v1-d14c8a9423946e268a0c9952fecf3a7aabd73bd9"},
}
TASKS = ("bundle", "content", "organization", "expression")


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
    partial = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise RunnerError(f"unreadable aggregate artifact: {path}") from exc
    if not isinstance(value, dict): raise RunnerError(f"aggregate artifact is not an object: {path}")
    return value


def ensure(condition: bool, message: str) -> None:
    if not condition: raise RunnerError(message)


def ledger(event: Mapping[str, Any]) -> None:
    RUN_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (RUN_ROOT / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": now(), **event}, ensure_ascii=False, sort_keys=True) + "\n")


def write_config(name: str, value: Mapping[str, Any]) -> Path:
    path = RUN_ROOT / "configs" / f"{name}.json"; path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    ensure(not path.exists(), f"refusing to overwrite runtime config {path}")
    atomic_json(path, value); return path


def run_stage(name: str, command: list[str], env: Mapping[str, str] | None = None) -> None:
    logs = RUN_ROOT / "logs"; logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    log = logs / f"{name}.log"; ensure(not log.exists(), f"refusing to overwrite stage log {name}")
    complete_env = os.environ.copy(); complete_env["PYTHONPATH"] = str(ROOT / "src")
    if env: complete_env.update(env)
    ledger({"stage": name, "event": "start", "command": command, "resource_scope": complete_env.get("CUDA_VISIBLE_DEVICES", "none")})
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, env=complete_env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        ledger({"stage": name, "event": "failed", "exit_code": result.returncode, "log": str(log.relative_to(ROOT))})
        raise RunnerError(f"stage {name} failed; preserved log {log}")
    ledger({"stage": name, "event": "completed", "exit_code": 0, "log": str(log.relative_to(ROOT))})


def check_idle(gpus: list[int], *, settle_seconds: int = 90) -> None:
    """Wait briefly for an owned vLLM server to release CUDA allocations.

    This does not permit overlap: a device that remains busy beyond the
    bounded teardown window is still a resource-boundary failure.
    """
    for gpu in gpus:
        ensure(gpu in {0, 1, 2, 3}, "only project GPUs 0--3 are permitted")
    deadline = time.monotonic() + settle_seconds
    latest: dict[int, tuple[int, int, int]] = {}
    while True:
        for gpu in gpus:
            line = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], text=True).strip()
            index, memory, util, temp = [part.strip() for part in line.split(",")]
            ensure(index == str(gpu), f"nvidia-smi returned a different GPU for GPU {gpu}")
            latest[gpu] = (int(memory), int(util), int(temp))
        if all(memory == 0 and util == 0 and temp <= 80 for memory, util, temp in latest.values()):
            return
        if time.monotonic() >= deadline:
            raise RunnerError(f"GPUs are not idle/cool after {settle_seconds}s teardown window: {latest}")
        time.sleep(2)


def base_env(gpus: str) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": gpus,
        "MAL2026_RESERVED_PHYSICAL_GPUS": gpus,
        "PYTHONPATH": str(ROOT / "src"),
        # FlashInfer's top-k/top-p sampler also JIT-builds through ninja in
        # this fixed vLLM build.  The documented false setting keeps native
        # PyTorch sampling while the model still runs through vLLM.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }


def sft_output(base: str, task: str, phase: str) -> Path:
    suffix = {"full": "", "gpu0_smoke": "-gpu0_smoke", "numeric_recovery": "-numeric_recovery"}.get(phase)
    ensure(suffix is not None, "unknown SFT phase")
    return SFT_ROOT / f"api-rationale-sft-v1-{base}-{task}{suffix}-001"


def sft_config(base: str, task: str, phase: str) -> dict[str, Any]:
    model = MODELS[base]
    settings = {"full": (f"api-rationale-sft-v1-{base}-{task}-001", 6000, -1), "gpu0_smoke": (f"api-rationale-sft-v1-{base}-{task}-gpu0_smoke-001", 1, 1), "numeric_recovery": (f"api-rationale-sft-v1-{base}-{task}-numeric_recovery-001", 6000, 5)}
    ensure(phase in settings, "unknown SFT phase")
    run_id, train_limit, max_steps = settings[phase]
    return {"schema_version": "mal2026-api-rationale-sft-v1", "run_id": run_id, "base_key": base, "model_id": model["id"], "model_revision": model["revision"], "model_path": str(model["path"].resolve()),
            "task": task, "phase": phase, "train_limit": train_limit, "max_steps": max_steps, "output_dir": str(sft_output(base, task, phase).resolve()),
            "seed": 2026072108, "max_length": 3072, "learning_rate": 2e-5, "num_train_epochs": 2.0, "per_device_train_batch_size": 2, "gradient_accumulation_steps": 8,
            "logging_steps": 5, "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05, "training_dtype": "float32", "trust_remote_code": False}


def verify_sft(base: str, task: str, phase: str) -> dict[str, Any]:
    value = read_json(sft_output(base, task, phase) / "training_complete.json")
    ensure(value.get("status") == "completed" and value.get("base_key") == base and value.get("task") == task and value.get("phase") == phase, "SFT completion provenance differs")
    ensure(value.get("config") == sft_config(base, task, phase), "SFT completion config differs from the immutable matrix")
    precision = value.get("adapter_precision", {})
    ensure(value["config"].get("training_dtype") == "float32" and precision.get("trainable_adapter_dtype") == "float32", "SFT numeric-recovery precision provenance differs")
    ensure(all(math_isfinite(metric) for metric in value.get("train_metrics", {}).values()), "SFT metric is non-finite")
    return value


def math_isfinite(value: Any) -> bool:
    import math
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


@contextmanager
def generation_server(base: str, task: str, source: str, port: int) -> Iterator[Path]:
    model = MODELS[base]; adapter = sft_output(base, task, "full") / "adapter"; alias = f"api-rationale-{base}-{task}"
    check_idle([0, 1, 2, 3]); log = RUN_ROOT / "logs" / f"server-generation-{base}-{task}-{source}.log"; log.parent.mkdir(mode=0o700, parents=True, exist_ok=True); ensure(not log.exists(), "generation server log already exists")
    # vLLM 0.25 selects optional FlashInfer TP fusion/custom-allreduce paths
    # under TP=4.  They JIT-build a FlashInfer extension through ``ninja``,
    # intentionally unavailable in this fixed environment.  Fall back only
    # for TP collectives to NCCL and disable the affected fusion/autotune;
    # ordinary vLLM compilation, CUDA graphs, TP, prefix caching, and LoRA
    # serving remain enabled.
    safe_compile = '{"pass_config":{"fuse_allreduce_rms":false}}'
    command = [str(VLLM), "serve", str(model["path"]), "--served-model-name", model["id"], "--host", "127.0.0.1", "--port", str(port), "--tensor-parallel-size", "4", "--attention-backend", "FLASH_ATTN", "--max-model-len", "3072", "--gpu-memory-utilization", "0.9", "--disable-custom-all-reduce", "--enable-lora", "--max-lora-rank", "32", "--lora-modules", f"{alias}={adapter}", "--generation-config", "vllm", "--enable-prefix-caching", "--no-enable-flashinfer-autotune", "--compilation-config", safe_compile]
    ledger({"stage": f"server-generation-{base}-{task}-{source}", "event": "start", "command": command, "resource_scope": "GPUs 0-3"})
    handle = log.open("x", encoding="utf-8"); process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **base_env("0,1,2,3")}, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    attestation = RUN_ROOT / "attestations" / f"generation-{base}-{task}.json"; attestation.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        for _ in range(360):
            if process.poll() is not None: raise RunnerError(f"generation server exited before health gate: {base}/{task}")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.status == 200: break
            except Exception: time.sleep(1)
        else: raise RunnerError(f"generation server health timeout: {base}/{task}")
        environ = Path(f"/proc/{process.pid}/environ").read_bytes(); ensure(b"CUDA_VISIBLE_DEVICES=0,1,2,3" in environ, "generation server CUDA visibility differs")
        atomic_json(attestation, {"schema_version": "mal2026-api-rationale-generation-v1-server-attestation-v1", "server_host": "127.0.0.1", "server_port": port, "physical_gpus": [0, 1, 2, 3], "tensor_parallel_size": 4, "max_model_len": 3072, "model_id": model["id"], "model_revision": model["revision"], "adapter_path": str(adapter.resolve()), "adapter_alias": alias, "server_process_environment_verified": True})
        yield attestation
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=60)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=30)
        handle.close(); ledger({"stage": f"server-generation-{base}-{task}-{source}", "event": "stopped", "resource_scope": "GPUs 0-3"})


def generation_config(base: str, task: str, source: str) -> dict[str, Any]:
    model = MODELS[base]
    # See APIRationaleGenerationConfig: both Phi bundle -001 (384 tokens) and
    # equal-budget -002 (512 tokens) preserved the same deterministic one-row
    # length failure.  Bundle -003 adds maxLength=192 while retaining 512
    # tokens.  Phi content-only -001 then showed the same category under its
    # 192-token budget, so every axis uses the same maxLength guard in -002.
    suffix = "003" if task == "bundle" else "002"
    run_id = f"api-rationale-generation-v1-{base}-{task}-{source}-{suffix}"
    return {"schema_version": "mal2026-api-rationale-generation-v1", "run_id": run_id, "base_key": base, "model_id": model["id"], "model_revision": model["revision"], "model_path": str(model["path"].resolve()),
            "task": task, "adapter_path": str((sft_output(base, task, "full") / "adapter").resolve()), "adapter_alias": f"api-rationale-{base}-{task}", "source": source,
            "restricted_output_dir": str((GEN_ROOT / run_id).resolve()), "max_new_tokens": 512 if task == "bundle" else 192, "max_model_len": 3072, "client_max_inflight": 256}


def verify_generation(base: str, task: str, source: str) -> dict[str, Any]:
    config = generation_config(base, task, source)
    output = Path(config["restricted_output_dir"])
    report = read_json(output / "aggregate_generation_report.json")
    manifest = read_json(output / "manifest.json")
    expected = 2000 if source == "train" else 400
    counts = report.get("counts", {})
    ensure(report.get("status") == "completed" and report.get("run_id") == config["run_id"], "generation completion provenance differs")
    ensure(report.get("base_key") == base and report.get("task") == task and report.get("source") == source, "generation completion binding differs")
    ensure(manifest.get("status") == "completed" and manifest.get("config") == config, "generation config provenance differs")
    ensure(counts == {"expected": expected, "observations": expected, "parse_valid": expected} and all(report.get("hard_gates", {}).values()), "generation aggregate gate failed")
    ensure(report.get("adapter_training_completion_sha256") == sha(sft_output(base, task, "full") / "training_complete.json"), "generation adapter provenance differs")
    ensure(report.get("candidate_scores_read_or_prompted") is False and report.get("source_writing_scores_read_or_prompted") is False, "generation score exclusion differs")
    return report


def run_generation(base: str, task: str, source: str, port: int, *, reuse_verified: bool = False) -> dict[str, Any]:
    config = generation_config(base, task, source); path = write_config(f"generation-{base}-{task}-{source}", config)
    existing = Path(config["restricted_output_dir"]) / "aggregate_generation_report.json"
    if reuse_verified and existing.is_file():
        report = verify_generation(base, task, source)
        ledger({"stage": f"generation-{base}-{task}-{source}", "event": "reused_verified_completed", "counts": report.get("counts"), "hard_gates": report.get("hard_gates"), "completion_sha256": sha(existing)})
        return report
    with generation_server(base, task, source, port) as attestation:
        run_stage(f"generation-{base}-{task}-{source}", [str(PY), "scripts/generate_api_rationales_vllm.py", "--config", str(path), "--endpoint", f"http://127.0.0.1:{port}", "--server-attestation", str(attestation)], base_env("0,1,2,3"))
    report = verify_generation(base, task, source)
    ledger({"stage": f"generation-{base}-{task}-{source}", "event": "aggregate", "counts": report.get("counts"), "hard_gates": report.get("hard_gates")}); return report


def merge_axis_triplet(base: str, *, reuse_verified: bool = False) -> dict[str, Any]:
    run_id = f"api-rationale-generation-v1-{base}-axis_triplet-validation-002"; config = {"schema_version": "mal2026-api-rationale-merge-v1", "run_id": run_id, "base_key": base, "source": "validation",
        "content_generation_dir": str((GEN_ROOT / f"api-rationale-generation-v1-{base}-content-validation-002").resolve()), "organization_generation_dir": str((GEN_ROOT / f"api-rationale-generation-v1-{base}-organization-validation-002").resolve()), "expression_generation_dir": str((GEN_ROOT / f"api-rationale-generation-v1-{base}-expression-validation-002").resolve()), "output_dir": str((GEN_ROOT / run_id).resolve())}
    path=write_config(f"merge-{base}",config); output=Path(config["output_dir"]); existing=output/"aggregate_generation_report.json"
    def verify() -> dict[str, Any]:
        report=read_json(existing);manifest=read_json(output/"manifest.json")
        ensure(report.get("status")=="completed" and report.get("run_id")==run_id and report.get("base_key")==base and report.get("source")=="validation" and report.get("task")=="axis_triplet", "axis-triplet merge provenance differs")
        ensure(report.get("counts")=={"expected":400,"observations":400,"parse_valid":400} and all(report.get("hard_gates",{}).values()),"axis-triplet merge gate failed")
        ensure(manifest.get("status")=="completed" and manifest.get("config")==config, "axis-triplet merge config differs")
        expected_reports={axis:sha(Path(config[f"{axis}_generation_dir"])/"aggregate_generation_report.json") for axis in ("content","organization","expression")}
        ensure(report.get("source_axis_generation_report_sha256")==expected_reports,"axis-triplet source provenance differs")
        return report
    if reuse_verified and existing.is_file():
        report=verify();ledger({"stage":f"merge-{base}","event":"reused_verified_completed","counts":report.get("counts"),"hard_gates":report.get("hard_gates"),"completion_sha256":sha(existing)});return report
    run_stage(f"merge-{base}",[str(PY),"scripts/merge_api_rationale_axis_triplet.py","--config",str(path)],base_env(""))
    report=verify();ledger({"stage":f"merge-{base}","event":"aggregate","counts":report.get("counts"),"hard_gates":report.get("hard_gates")});return report


@contextmanager
def qwen_judge_server(port: int) -> Iterator[Path]:
    check_idle([0, 1, 2, 3]); log=RUN_ROOT/"logs"/"server-qwen-v6-judge.log";log.parent.mkdir(mode=0o700,parents=True,exist_ok=True);ensure(not log.exists(),"Qwen judge server log already exists")
    safe_compile='{"pass_config":{"fuse_allreduce_rms":false}}'
    # vLLM 0.25.1's DP=4 CUDA-graph warmup invokes an unavailable local ninja
    # compiler for this FP8 Qwen snapshot.  Eager mode retains DP=4 batching,
    # FlashAttention, prefix caching and the exact judge protocol, while
    # bypassing only that failed graph-capture setup path.
    command=[str(VLLM),"serve",str(QWEN_JUDGE_PATH),"--served-model-name",QWEN_JUDGE_ID,"--host","127.0.0.1","--port",str(port),"--tensor-parallel-size","1","--data-parallel-size","4","--attention-backend","FLASH_ATTN","--max-model-len","4096","--max-num-seqs","192","--max-num-batched-tokens","65536","--gpu-memory-utilization","0.9","--disable-custom-all-reduce","--gdn-prefill-backend","triton","--generation-config","vllm","--enable-prefix-caching","--no-enable-flashinfer-autotune","--enforce-eager","--compilation-config",safe_compile]
    # vLLM documents this switch for the Hopper FP8 block-scale FlashInfer
    # GEMM.  Its default can JIT through unavailable ninja; the DeepGEMM
    # fallback supports this exact FP8 model without changing judge labels.
    judge_env={**os.environ,**base_env("0,1,2,3"),"VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER":"0"}
    ledger({"stage":"server-qwen-v6-judge","event":"start","command":command,"resource_scope":"GPUs 0-3","execution_env":{"VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER":"0"}});handle=log.open("x",encoding="utf-8");process=subprocess.Popen(command,cwd=ROOT,env=judge_env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    attestation=RUN_ROOT/"attestations"/"qwen-v6-judge.json";attestation.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    try:
        for _ in range(360):
            if process.poll() is not None: raise RunnerError("Qwen judge server exited before health gate")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as response:
                    if response.status==200:break
            except Exception: time.sleep(1)
        else: raise RunnerError("Qwen judge server health timeout")
        # This synthetic one-token request verifies the FP8 linear path before
        # opening a restricted rationale.  Its response is deliberately not
        # stored; it is a transport/setup gate, not a judge observation.
        request=Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps({"model":QWEN_JUDGE_ID,"messages":[{"role":"user","content":"Reply with one token."}],"max_tokens":1,"temperature":0,"seed":0},ensure_ascii=False).encode(),headers={"Content-Type":"application/json"},method="POST")
        try:
            with urlopen(request,timeout=120) as response:
                probe=json.loads(response.read().decode())
            ensure(isinstance(probe,dict) and isinstance(probe.get("choices"),list) and probe["choices"],"Qwen judge synthetic inference response is invalid")
        except Exception as exc:
            raise RunnerError("Qwen judge synthetic inference gate failed") from exc
        ledger({"stage":"server-qwen-v6-judge","event":"health_and_synthetic_inference_pass","resource_scope":"GPUs 0-3","raw_response_persisted":False})
        environ=Path(f"/proc/{process.pid}/environ").read_bytes();ensure(b"CUDA_VISIBLE_DEVICES=0,1,2,3" in environ,"Qwen judge CUDA visibility differs")
        atomic_json(attestation,{"schema_version":"mal2026-native-fp8-vllm-distribution100-server-attestation-v1","server_host":"127.0.0.1","server_port":port,"physical_gpus":[0,1,2,3],"tensor_parallel_size":1,"data_parallel_size":4,"max_model_len":4096,"max_num_seqs_per_dp_rank":192,"enforce_eager":True,"config_sha256":sha(V6_CONFIG),"server_process_environment_verified":True})
        yield attestation
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=30)
        handle.close();ledger({"stage":"server-qwen-v6-judge","event":"stopped","resource_scope":"GPUs 0-3"})


def judge_config(base: str, kind: str) -> dict[str, Any]:
    run_id=f"api-rationale-judge-v1-{base}-{kind}-validation-002"
    generation_suffix="003" if kind=="bundle" else "002"
    generation=f"api-rationale-generation-v1-{base}-{kind}-validation-{generation_suffix}"
    return {"schema_version":"mal2026-api-rationale-judge-v1","run_id":run_id,"base_key":base,"system_kind":kind,"generation_dir":str((GEN_ROOT/generation).resolve()),"output_dir":str((JUDGE_ROOT/run_id).resolve()),"source":"validation","client_max_inflight":768}


def run_judges() -> list[dict[str, Any]]:
    values=[]
    with qwen_judge_server(18450) as attestation:
        for base in MODELS:
            for kind in ("bundle","axis_triplet"):
                config=judge_config(base,kind);path=write_config(f"judge-{base}-{kind}",config)
                output=Path(config["output_dir"]);report_path=output/"aggregate_judge_report.json"
                if report_path.is_file():
                    report=read_json(report_path);manifest=read_json(output/"manifest.json")
                    ensure(manifest.get("status")=="completed" and manifest.get("config")==config,"completed judge provenance differs")
                    ensure(report.get("status")=="completed" and report.get("run_id")==config["run_id"] and report.get("base_key")==base and report.get("system_kind")==kind and report.get("counts")=={"expected_calls":20000,"observations":20000,"scored":20000,"schema_valid":20000,"abstain":0,"generated_candidates":400} and all(report.get("hard_gates",{}).values()),"completed judge gate differs")
                    ledger({"stage":f"judge-{base}-{kind}","event":"reused_verified_completed","macro_mean":report.get("macro_mean"),"axis_means":report.get("axis_means"),"report_sha256":sha(report_path)})
                else:
                    run_stage(f"judge-{base}-{kind}",[str(PY),"scripts/judge_api_rationales_v6.py","--config",str(path),"--endpoint","http://127.0.0.1:18450","--server-attestation",str(attestation),"--model",QWEN_JUDGE_ID],{**base_env("0,1,2,3"),"MAL2026_DIST100_CONFIG":str(V6_CONFIG),"MAL2026_DIST100_SCHEMA":"mal2026-qwen36-native-fp8-vllm-rationale-only-score5x10-v6"})
                    report=read_json(report_path);ensure(report.get("status")=="completed" and all(report.get("hard_gates",{}).values()),"decoder judge aggregate gate failed")
                    ledger({"stage":f"judge-{base}-{kind}","event":"aggregate","macro_mean":report.get("macro_mean"),"axis_means":report.get("axis_means"),"delta_from_api_baseline":report.get("delta_from_api_baseline")})
                values.append(report)
    return values


def select_decoder(reports: list[dict[str, Any]]) -> dict[str, Any]:
    candidates=[item for item in reports if item.get("system_kind")=="bundle"]
    ensure(len(candidates)==3,"three bundled decoder judge reports are required")
    def key(item: Mapping[str,Any]) -> tuple[float,float,str]:
        ranges=item.get("prompt_type_axis_ranges",{});stability=sum(float(ranges[axis]) for axis in ("content","organization","expression"))
        return (-float(item["macro_mean"]),stability,str(item["base_key"]))
    selected=sorted(candidates,key=key)[0]
    result={"schema_version":"mal2026-api-rationale-decoder-selection-v1","status":"completed","selection_rule":"maximum bundled validation macro judge mean; ties lower summed prompt-axis range then lexical base key","selected_base_key":selected["base_key"],"selected_system_kind":"bundle","selected_macro_mean":selected["macro_mean"],"candidate_reports":{item["base_key"]:{"macro_mean":item["macro_mean"],"prompt_type_axis_ranges":item["prompt_type_axis_ranges"],"report_sha256":sha(Path(judge_config(str(item["base_key"]),"bundle")["output_dir"])/"aggregate_judge_report.json")} for item in candidates},"validation_selection_user_authorized":True,"downstream_validation_results_descriptive_not_new_untouched_test":True}
    path=AGGREGATE_ROOT/"api-rationale-sft-score-regression-v1-20260721-001.decoder-selection.json";AGGREGATE_ROOT.mkdir(mode=0o700,parents=True,exist_ok=True)
    if path.is_file():
        existing=read_json(path)
        ensure(existing==result,"existing decoder selection provenance differs")
        ledger({"stage":"decoder-selection","event":"reused_verified_completed","selected_base_key":result["selected_base_key"],"selected_macro_mean":result["selected_macro_mean"],"aggregate":str(path.relative_to(ROOT))});return existing
    atomic_json(path,result);ledger({"stage":"decoder-selection","event":"completed","selected_base_key":result["selected_base_key"],"selected_macro_mean":result["selected_macro_mean"],"aggregate":str(path.relative_to(ROOT))});return result


def regression_config(backbone: str, condition: str, selected_base: str) -> dict[str, Any]:
    model=ENCODERS[backbone];suffix="003" if backbone=="qwen25_7b" else "004";run_id=f"api-score-regression-v1-{backbone}-{condition}-{suffix}";decoder=None
    if condition=="decoder_rationale":decoder=str((GEN_ROOT/f"api-rationale-generation-v1-{selected_base}-bundle-train-003").resolve())
    return {"schema_version":"mal2026-api-score-regression-v1","run_id":run_id,"backbone_key":backbone,"model_id":model["id"],"model_revision":model["revision"],"model_path":str(model["path"].resolve()),"input_condition":condition,"decoder_generation_dir":decoder,"output_dir":str((REG_ROOT/run_id).resolve()),"seed":2026072108,"max_length":3072,"learning_rate":2e-5 if backbone=="qwen25_7b" else 1e-4,"num_train_epochs":6.0 if condition=="api_rationale" else 12.0,"per_device_train_batch_size":2 if backbone=="qwen25_7b" else 8,"gradient_accumulation_steps":8 if backbone=="qwen25_7b" else 2,"logging_steps":5,"lora_r":16,"lora_alpha":32,"lora_dropout":0.05,"training_dtype":"float32"}


def run_regressions(selected_base: str) -> list[dict[str, Any]]:
    result=[]
    for backbone in ENCODERS:
        for condition in ("direct","api_rationale","decoder_rationale"):
            config=regression_config(backbone,condition,selected_base);path=write_config(f"regression-{backbone}-{condition}",config);check_idle([0,1,2,3])
            complete_path=Path(config["output_dir"])/"training_complete.json"
            if complete_path.is_file():
                complete=read_json(complete_path)
                ensure(complete.get("status")=="completed" and complete.get("config")==config and all(math_isfinite(x) for x in complete.get("train_metrics",{}).values()),"completed score-regression train provenance differs")
                ledger({"stage":f"regression-train-{backbone}-{condition}","event":"reused_verified_completed","global_step":complete.get("global_step"),"train_metrics":complete.get("train_metrics"),"completion_sha256":sha(complete_path)})
            else:
                run_stage(f"regression-train-{backbone}-{condition}",[str(PY),"-m","torch.distributed.run","--nproc_per_node=4","scripts/train_api_score_regression.py","--config",str(path)],base_env("0,1,2,3"))
                complete=read_json(complete_path);ensure(complete.get("status")=="completed" and all(math_isfinite(x) for x in complete.get("train_metrics",{}).values()),"score-regression train gate failed")
            suffix="003" if backbone=="qwen25_7b" else "004";eval_id=f"api-score-regression-eval-v1-{backbone}-{condition}-validation-{suffix}";evaluation={"schema_version":"mal2026-api-score-regression-eval-v1","run_id":eval_id,"training_metadata_path":str((Path(config["output_dir"])/"training_complete.json").resolve()),"output_dir":str((REG_EVAL_ROOT/eval_id).resolve()),"per_device_eval_batch_size":4 if backbone=="qwen25_7b" else 16}
            eval_path=write_config(f"regression-eval-{backbone}-{condition}",evaluation);check_idle([0,1,2,3])
            report_path=Path(evaluation["output_dir"])/"aggregate_metrics.json"
            if report_path.is_file():
                report=read_json(report_path);ensure(report.get("status")=="completed" and report.get("config")==evaluation,"completed score-regression evaluation provenance differs")
                ledger({"stage":f"regression-eval-{backbone}-{condition}","event":"reused_verified_completed","metrics":report.get("metrics"),"validation":report.get("validation"),"report_sha256":sha(report_path)})
            else:
                run_stage(f"regression-eval-{backbone}-{condition}",[str(PY),"-m","torch.distributed.run","--nproc_per_node=4","scripts/evaluate_api_score_regression.py","--config",str(eval_path)],base_env("0,1,2,3"))
                report=read_json(report_path);ensure(report.get("status")=="completed","score-regression evaluation gate failed")
            for value in report.get("metrics",{}).values():
                if isinstance(value,dict): ensure(all(math_isfinite(metric) for metric in value.values()),"non-finite score regression metric")
                elif isinstance(value,(int,float)): ensure(math_isfinite(value),"non-finite score regression macro metric")
            ledger({"stage":f"regression-eval-{backbone}-{condition}","event":"aggregate","metrics":report.get("metrics"),"validation":report.get("validation")});result.append(report)
    return result


def final_summary(decoder_reports: list[dict[str, Any]], selection: Mapping[str, Any], regression_reports: list[dict[str, Any]]) -> None:
    output=AGGREGATE_ROOT/"api-rationale-sft-score-regression-v1-20260721-001.final-summary.json";ensure(not output.exists(),"final aggregate summary already exists")
    payload={"schema_version":"mal2026-api-rationale-sft-score-regression-v1-summary-v1","status":"completed","decoder_systems":[{"base_key":item["base_key"],"system_kind":item["system_kind"],"macro_mean":item["macro_mean"],"axis_means":item["axis_means"],"delta_from_api_baseline":item["delta_from_api_baseline"],"prompt_type_axis_ranges":item["prompt_type_axis_ranges"]} for item in decoder_reports],"decoder_selection":selection,"score_regression":[{"training_run_id":item["training_run_id"],"backbone_key":item["backbone_key"],"input_condition":item["input_condition"],"metrics":item["metrics"],"validation":item["validation"]} for item in regression_reports],"privacy":"aggregate_only_no_rows_prompts_essays_rationales_ids_candidate_scores_or_predictions_persisted","git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    atomic_json(output,payload);ledger({"stage":"final-summary","event":"completed","aggregate":str(output.relative_to(ROOT))})


def run_smoke() -> None:
    ensure(PY.is_file() and VLLM.is_file(),"project runtime is unavailable");check_idle([0])
    config=sft_config("phi4_mini","bundle","gpu0_smoke");path=write_config("gpu0-smoke-phi4-bundle",config)
    run_stage("gpu0-smoke-phi4-bundle",[str(PY),"-m","torch.distributed.run","--nproc_per_node=1","scripts/train_api_rationale_sft.py","--config",str(path)],base_env("0"))
    complete=verify_sft("phi4_mini","bundle","gpu0_smoke");ledger({"stage":"gpu0-smoke-phi4-bundle","event":"aggregate","global_step":complete.get("global_step"),"train_metrics":complete.get("train_metrics")})


def run_full(*, reuse_verified_sft: bool = False) -> None:
    smoke=read_json(sft_output("phi4_mini","bundle","gpu0_smoke")/"training_complete.json");ensure(smoke.get("status")=="completed" and smoke.get("phase")=="gpu0_smoke","GPU0 SFT preflight is required before full")
    ensure(PY.is_file() and VLLM.is_file() and V6_CONFIG.is_file() and QWEN_JUDGE_PATH.is_dir(),"required project runtime/model is unavailable")
    for base in MODELS:
        for task in TASKS:
            config=sft_config(base,task,"full");write_config(f"sft-{base}-{task}",config)
            existing=sft_output(base,task,"full")/"training_complete.json"
            if reuse_verified_sft and existing.is_file():
                # A prior full SFT may be reused only after its completion
                # record proves exact fixed-matrix and float32-recovery
                # provenance.  This never overwrites a completed adapter.
                complete=verify_sft(base,task,"full")
                ledger({"stage":f"sft-{base}-{task}","event":"reused_verified_completed","global_step":complete.get("global_step"),"train_records":complete.get("train_records"),"train_metrics":complete.get("train_metrics"),"completion_sha256":sha(existing)})
            else:
                path=RUN_ROOT/"configs"/f"sft-{base}-{task}.json";check_idle([0,1,2,3])
                run_stage(f"sft-{base}-{task}",[str(PY),"-m","torch.distributed.run","--nproc_per_node=4","scripts/train_api_rationale_sft.py","--config",str(path)],base_env("0,1,2,3"));complete=verify_sft(base,task,"full")
                ledger({"stage":f"sft-{base}-{task}","event":"aggregate","global_step":complete.get("global_step"),"train_records":complete.get("train_records"),"train_metrics":complete.get("train_metrics")})
            run_generation(base,task,"validation",18400,reuse_verified=reuse_verified_sft)
        merge_axis_triplet(base,reuse_verified=reuse_verified_sft)
    reports=run_judges();selection=select_decoder(reports);selected=str(selection["selected_base_key"])
    run_generation(selected,"bundle","train",18420,reuse_verified=reuse_verified_sft)
    regressions=run_regressions(selected);final_summary(reports,selection,regressions)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("gpu0-smoke","full","full-resume"));args=parser.parse_args()
    global RUN_ROOT
    RUN_ROOT = RUN_BASE / {"gpu0-smoke":"gpu0-smoke","full":"full","full-resume":"full-resume-015"}[args.mode]
    # The launcher may redirect its own stderr/stdout to ``runner.log`` before
    # this process starts, which necessarily creates the runtime directory.
    # That file contains no stage state, so permit precisely that empty-wrapper
    # case while refusing to reuse any actual run artifact.
    if RUN_ROOT.exists():
        entries = {entry.name for entry in RUN_ROOT.iterdir()}
        if entries - {"runner.log"}:
            raise RunnerError(f"runtime root already exists; preserve its ledger and outputs: {RUN_ROOT}")
    else:
        RUN_ROOT.mkdir(mode=0o700,parents=True)
    manifest={"schema_version":"mal2026-api-rationale-sft-score-regression-v1-runner-v1","status":"running","mode":args.mode,"git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"physical_gpus":[0] if args.mode=="gpu0-smoke" else [0,1,2,3],"seed":2026072108,"runtime_root":str(RUN_ROOT),"resume_of":str((RUN_BASE/"full-resume-014").resolve()) if args.mode=="full-resume" else None,"reuse_policy":"only exact completed full SFT, generation, axis-merge, judge, decoder-selection, and score-regression artifacts with fixed provenance; bundle -003 and axis -002 use the documented schema-character-bound closure repair; the Qwen DP=4 judge keeps the fixed model/topology/batching/protocol and disables only FlashInfer block-scale FP8 GEMM after its preserved ninja JIT failure; Qwen score regressors reuse fresh -003 float32 artifacts, while KURE uses fresh -004 float32 artifacts with DDP unused-parameter discovery enabled after the preserved direct -003 reduction failure" if args.mode=="full-resume" else "none","scripts":{"runner":sha(Path(__file__)),"sft":sha(ROOT/"scripts"/"train_api_rationale_sft.py"),"generation":sha(ROOT/"scripts"/"generate_api_rationales_vllm.py"),"judge":sha(ROOT/"scripts"/"judge_api_rationales_v6.py"),"regression":sha(ROOT/"scripts"/"train_api_score_regression.py")}}
    atomic_json(RUN_ROOT/"manifest.json",manifest)
    try:
        if args.mode=="gpu0-smoke":run_smoke()
        else:run_full(reuse_verified_sft=args.mode=="full-resume")
    except Exception:
        manifest["status"]="failed";manifest["failed_at"]=now();atomic_json(RUN_ROOT/"manifest.json",manifest);raise
    manifest["status"]="completed";manifest["completed_at"]=now();atomic_json(RUN_ROOT/"manifest.json",manifest)


if __name__=="__main__":main()
