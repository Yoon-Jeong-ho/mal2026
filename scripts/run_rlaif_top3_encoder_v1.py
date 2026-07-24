#!/usr/bin/env python3
"""Durably run the selected-RLAIF-rationale / Qwen2.5 encoder protocol.

The runner owns sequence, private-output paths, and aggregate-only ledger
records.  ``gpu0-preflight`` is one actual decoder request and one actual
encoder update.  ``full`` then runs the three independent sources on GPUs
0--3: first their train/validation generations, then their three separate
encoder train/validation pairs.  ``full-resume`` never overwrites artifacts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping
from urllib.request import urlopen

from mal2026.rlaif_top3_encoder import (
    EVALUATION_ROOT,
    GENERATION_ROOT,
    RLAIFTop3EncoderError,
    SELECTIONS,
    TRAINING_ROOT,
    evaluation_config,
    generation_config,
    regression_config,
    selected_sources,
)


ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv-standard" / "bin" / "python"
VLLM = ROOT / ".venv-standard" / "bin" / "vllm"
RUN_BASE = ROOT / "outputs" / "rlaif-top3-encoder-v1" / "20260725-001"
AGGREGATE_ROOT = ROOT / "outputs" / "aggregate-reports"
RUN_ROOT = RUN_BASE


class RunnerError(RuntimeError):
    """A durable stage or its declared promotion gate failed."""


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{description} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"{description} is not an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def ledger(event: Mapping[str, Any]) -> None:
    RUN_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (RUN_ROOT / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": now(), **event}, ensure_ascii=False, sort_keys=True) + "\n")


def write_config(name: str, value: Mapping[str, Any]) -> Path:
    path = RUN_ROOT / "configs" / f"{name}.json"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    ensure(not path.exists(), f"refusing to overwrite runtime config: {path}")
    atomic_json(path, value)
    return path


def base_env(gpus: str) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": gpus,
        "MAL2026_RESERVED_PHYSICAL_GPUS": gpus,
        "PYTHONPATH": str(ROOT / "src"),
        # The fixed vLLM package would otherwise build its optional sampler
        # extension through unavailable ninja.  This leaves the vLLM engine,
        # TP, FlashAttention, structured output, and prefix cache enabled.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }


def run_stage(name: str, command: list[str], env: Mapping[str, str]) -> None:
    logs = RUN_ROOT / "logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    log = logs / f"{name}.log"
    ensure(not log.exists(), f"refusing to overwrite stage log: {name}")
    full_env = os.environ.copy()
    full_env.update(env)
    ledger({"stage": name, "event": "start", "command": command, "resource_scope": env.get("CUDA_VISIBLE_DEVICES", "none")})
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, env=full_env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        ledger({"stage": name, "event": "failed", "exit_code": result.returncode, "log": str(log.relative_to(ROOT)), "resource_scope": env.get("CUDA_VISIBLE_DEVICES", "none")})
        raise RunnerError(f"stage {name} failed; preserved log: {log}")
    ledger({"stage": name, "event": "completed", "exit_code": 0, "log": str(log.relative_to(ROOT)), "resource_scope": env.get("CUDA_VISIBLE_DEVICES", "none")})


def check_idle(gpus: list[int], *, settle_seconds: int = 90) -> None:
    """Check only the project-owned GPU IDs and never displace a process."""
    ensure(gpus and all(gpu in {0, 1, 2, 3} for gpu in gpus), "only project GPUs 0--3 may be checked or used")
    deadline = time.monotonic() + settle_seconds
    latest: dict[int, tuple[int, int]] = {}
    while True:
        for gpu in gpus:
            line = subprocess.check_output(
                ["nvidia-smi", f"--id={gpu}", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
            index, memory, utilization = [part.strip() for part in line.split(",")]
            ensure(index == str(gpu), f"GPU query returned a different device for GPU {gpu}")
            latest[gpu] = (int(memory), int(utilization))
        if all(memory == 0 and utilization == 0 for memory, utilization in latest.values()):
            return
        if time.monotonic() >= deadline:
            raise RunnerError(f"project GPUs are not idle after bounded wait: {latest}")
        time.sleep(2)


def verify_preflight_requirement() -> None:
    path = TRAINING_ROOT / "rlaif-top3-score-regression-v1-rank1_midm2_random1-gpu0_preflight-001" / "training_complete.json"
    value = read_json(path, "GPU0 encoder preflight completion")
    expected = regression_config("rank1_midm2_random1", "gpu0_preflight")
    ensure(value.get("status") == "completed" and value.get("config") == expected and value.get("train_records") == 1 and value.get("score_fields") == ["content", "organization", "expression"], "GPU0 encoder preflight provenance differs")


def verify_generation(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(config["restricted_output_dir"]))
    report = read_json(root / "aggregate_generation_report.json", "top-three generation report")
    manifest = read_json(root / "manifest.json", "top-three generation manifest")
    expected_count = int(config["record_limit"])
    ensure(manifest.get("status") == "completed" and manifest.get("config") == config, "top-three generation manifest provenance differs")
    ensure(report.get("status") == "completed" and report.get("run_id") == config["run_id"] and report.get("source_key") == config["source_key"] and report.get("selected_rank") == config["selected_rank"] and report.get("source") == config["source"] and report.get("phase") == config["phase"], "top-three generation identity differs")
    ensure(report.get("counts") == {"expected": expected_count, "observations": expected_count, "parse_valid": expected_count} and all(report.get("hard_gates", {}).values()), "top-three generation hard gate failed")
    ensure(report.get("source_writing_scores_read_or_prompted") is False, "top-three generation read writing scores")
    return report


def verify_training(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(config["output_dir"]))
    completion = read_json(root / "training_complete.json", "top-three training completion")
    state = root / "final_model" / "model.safetensors"
    ensure(completion.get("status") == "completed" and completion.get("config") == config and completion.get("source_key") == config["source_key"] and completion.get("selected_rank") == config["selected_rank"], "top-three training provenance differs")
    ensure(completion.get("score_fields") == ["content", "organization", "expression"] and completion.get("score_targets") == ["content", "organization", "expression"], "top-three training targets differ")
    ensure(completion.get("train_records") == config["train_record_limit"] and isinstance(completion.get("global_step"), int) and completion["global_step"] > 0, "top-three training completion count differs")
    ensure(state.is_file() and completion.get("model_state_sha256") == sha(state), "top-three training state checksum differs")
    metrics = completion.get("train_metrics", {})
    ensure(isinstance(metrics, dict) and metrics and all(finite(value) for value in metrics.values()), "top-three training metrics are non-finite")
    return completion


def verify_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(config["output_dir"]))
    report = read_json(root / "aggregate_metrics.json", "top-three evaluation report")
    expected_keys = {"content", "organization", "expression", "three_axis_macro_rmse", "three_axis_macro_spearman"}
    ensure(report.get("status") == "completed" and report.get("config") == config and report.get("source_key") == config["source_key"] and report.get("selected_rank") == config["selected_rank"], "top-three evaluation provenance differs")
    ensure(report.get("score_fields") == ["content", "organization", "expression"] and set(report.get("metrics", {})) == expected_keys, "top-three evaluation target/metric fields differ")
    for key, value in report["metrics"].items():
        if isinstance(value, dict):
            ensure(set(value) == {"rmse", "spearman"} and all(finite(metric) for metric in value.values()), f"top-three {key} metrics are non-finite")
        else:
            ensure(finite(value), f"top-three {key} metric is non-finite")
    ensure(report.get("validation") == {"unique_essays": 400, "input_records": 400, "predictions_per_essay": 1, "rationale_sources_combined": 0}, "top-three evaluation validation contract differs")
    return report


@contextmanager
def generation_server(config: Mapping[str, Any], port: int) -> Iterator[Path]:
    phase = str(config["phase"])
    gpus = [0, 1, 2, 3] if phase == "full" else [0]
    gpu_text = ",".join(str(gpu) for gpu in gpus)
    check_idle(gpus)
    selection = SELECTIONS[str(config["source_key"])]
    tensor_parallel = 4 if phase == "full" else 1
    alias = f"rlaif-top3-{config['source_key']}"
    logs = RUN_ROOT / "logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    log = logs / f"server-generation-{config['source_key']}-{config['source']}-{phase}.log"
    ensure(not log.exists(), f"refusing to overwrite vLLM server log: {log}")
    safe_compile = '{"pass_config":{"fuse_allreduce_rms":false}}'
    command = [
        str(VLLM), "serve", str(selection["model_path"]), "--served-model-name", str(config["model_id"]), "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel), "--attention-backend", "FLASH_ATTN", "--max-model-len", str(config["max_model_len"]),
        "--gpu-memory-utilization", "0.9", "--disable-custom-all-reduce", "--enable-lora", "--max-lora-rank", "32",
        "--lora-modules", f"{alias}={config['rlaif_adapter_path']}", "--generation-config", "vllm", "--enable-prefix-caching",
        "--no-enable-flashinfer-autotune", "--compilation-config", safe_compile,
    ]
    ledger({"stage": f"server-generation-{config['source_key']}-{config['source']}-{phase}", "event": "start", "command": command, "resource_scope": f"GPUs {gpu_text}"})
    handle = log.open("x", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **base_env(gpu_text)}, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    attestation = RUN_ROOT / "attestations" / f"generation-{config['source_key']}-{config['source']}-{phase}.json"
    attestation.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        for _ in range(360):
            if process.poll() is not None:
                raise RunnerError(f"vLLM generation server exited before health gate: {config['source_key']}")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RunnerError(f"vLLM generation server health timeout: {config['source_key']}")
        environment = Path(f"/proc/{process.pid}/environ").read_bytes()
        ensure(f"CUDA_VISIBLE_DEVICES={gpu_text}".encode() in environment, "vLLM CUDA visibility differs")
        atomic_json(attestation, {
            "schema_version": "mal2026-rlaif-top3-rationale-generation-server-attestation-v1",
            "server_host": "127.0.0.1", "server_port": port, "physical_gpus": gpus, "tensor_parallel_size": tensor_parallel,
            "model_id": config["model_id"], "model_revision": config["model_revision"], "model_path": config["model_path"],
            "rlaif_adapter_path": config["rlaif_adapter_path"], "adapter_alias": alias, "max_model_len": config["max_model_len"],
            "server_process_environment_verified": True,
        })
        ledger({"stage": f"server-generation-{config['source_key']}-{config['source']}-{phase}", "event": "health_pass", "resource_scope": f"GPUs {gpu_text}"})
        yield attestation
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        handle.close()
        ledger({"stage": f"server-generation-{config['source_key']}-{config['source']}-{phase}", "event": "stopped", "resource_scope": f"GPUs {gpu_text}"})


def run_generation(config: Mapping[str, Any], config_path: Path, port: int, *, reuse: bool) -> dict[str, Any]:
    output = Path(str(config["restricted_output_dir"]))
    existing = output / "aggregate_generation_report.json"
    stage = f"generation-{config['source_key']}-{config['source']}-{config['phase']}"
    if existing.is_file():
        ensure(reuse, f"refusing to overwrite completed/partial generation artifact: {output}")
        report = verify_generation(config)
        ledger({"stage": stage, "event": "reused_verified_completed", "counts": report["counts"], "hard_gates": report["hard_gates"], "report_sha256": sha(existing)})
        return report
    ensure(not output.exists(), f"refusing to overwrite partial generation artifact: {output}")
    with generation_server(config, port) as attestation:
        gpus = "0,1,2,3" if config["phase"] == "full" else "0"
        run_stage(stage, [str(PY), "scripts/generate_rlaif_top3_rationales.py", "--config", str(config_path), "--endpoint", f"http://127.0.0.1:{port}", "--server-attestation", str(attestation)], base_env(gpus))
    report = verify_generation(config)
    ledger({"stage": stage, "event": "aggregate", "counts": report["counts"], "hard_gates": report["hard_gates"], "report_sha256": sha(existing)})
    return report


def run_training(config: Mapping[str, Any], config_path: Path, *, reuse: bool) -> dict[str, Any]:
    output = Path(str(config["output_dir"]))
    completion = output / "training_complete.json"
    stage = f"encoder-train-{config['source_key']}-{config['phase']}"
    if completion.is_file():
        ensure(reuse, f"refusing to overwrite completed/partial training artifact: {output}")
        result = verify_training(config)
        ledger({"stage": stage, "event": "reused_verified_completed", "global_step": result["global_step"], "train_metrics": result["train_metrics"], "completion_sha256": sha(completion)})
        return result
    ensure(not output.exists(), f"refusing to overwrite partial training artifact: {output}")
    gpus = "0,1,2,3" if config["phase"] == "full" else "0"
    check_idle([0, 1, 2, 3] if config["phase"] == "full" else [0])
    nproc = "4" if config["phase"] == "full" else "1"
    run_stage(stage, [str(PY), "-m", "torch.distributed.run", "--nproc_per_node", nproc, "scripts/train_rlaif_top3_score_regression.py", "--config", str(config_path)], base_env(gpus))
    result = verify_training(config)
    ledger({"stage": stage, "event": "aggregate", "global_step": result["global_step"], "train_metrics": result["train_metrics"], "completion_sha256": sha(completion)})
    return result


def run_evaluation(config: Mapping[str, Any], config_path: Path, *, reuse: bool) -> dict[str, Any]:
    output = Path(str(config["output_dir"]))
    report_path = output / "aggregate_metrics.json"
    stage = f"encoder-eval-{config['source_key']}-validation"
    if report_path.is_file():
        ensure(reuse, f"refusing to overwrite completed/partial evaluation artifact: {output}")
        result = verify_evaluation(config)
        ledger({"stage": stage, "event": "reused_verified_completed", "metrics": result["metrics"], "report_sha256": sha(report_path)})
        return result
    ensure(not output.exists(), f"refusing to overwrite partial evaluation artifact: {output}")
    check_idle([0, 1, 2, 3])
    run_stage(stage, [str(PY), "-m", "torch.distributed.run", "--nproc_per_node=4", "scripts/evaluate_rlaif_top3_score_regression.py", "--config", str(config_path)], base_env("0,1,2,3"))
    result = verify_evaluation(config)
    ledger({"stage": stage, "event": "aggregate", "metrics": result["metrics"], "report_sha256": sha(report_path)})
    return result


def run_preflight() -> None:
    ensure(PY.is_file() and VLLM.is_file(), "project runtime is unavailable")
    source = "rank1_midm2_random1"
    generation = generation_config(source, "train", "gpu0_preflight")
    generation_path = write_config("generation-rank1-gpu0-preflight", generation)
    run_generation(generation, generation_path, 18700, reuse=False)
    training = regression_config(source, "gpu0_preflight")
    training_path = write_config("encoder-rank1-gpu0-preflight", training)
    run_training(training, training_path, reuse=False)
    verify_preflight_requirement()
    ledger({"stage": "gpu0-preflight", "event": "smoke_pass", "evidence_ref": str((Path(training["output_dir"]) / "training_complete.json").relative_to(ROOT)), "command_ref": "generation+one-update", "resource_scope": "GPU0", "gpu_scope_authorization": "default", "decision": "continue"})


def final_summary(reports: Mapping[str, Mapping[str, Any]], *, reuse: bool) -> None:
    result_path = AGGREGATE_ROOT / "rlaif-top3-encoder-v1-20260725-001.final-summary.json"
    entries = []
    for source in selected_sources():
        report = reports[source]
        entries.append({
            "source_key": source,
            "selected_rank": SELECTIONS[source]["rank"],
            "rlaif_bundle_reward_arm": SELECTIONS[source]["arm"],
            "rlaif_frozen_macro": SELECTIONS[source]["frozen_macro"],
            "encoder_backbone": report["backbone_key"],
            "score_fields": report["score_fields"],
            "metrics": report["metrics"],
            "validation": report["validation"],
        })
    best = min(entries, key=lambda item: (float(item["metrics"]["three_axis_macro_rmse"]), -float(item["metrics"]["three_axis_macro_spearman"]), str(item["source_key"])))
    payload = {
        "schema_version": "mal2026-rlaif-top3-encoder-v1-final-summary-v1",
        "status": "completed",
        "selection_rule": "exactly the three highest complete RLAIF v8 bundle adapters by frozen macro; no axis-only adapter, decoder ensemble, rationale-source merge, or fourth score target",
        "encoder_backbone": "Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28",
        "score_fields": ["content", "organization", "expression"],
        "encoder_runs": entries,
        "best_by_three_axis_macro_rmse_then_spearman": {"source_key": best["source_key"], "selected_rank": best["selected_rank"], "metrics": best["metrics"]},
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    AGGREGATE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if result_path.exists():
        ensure(reuse and read_json(result_path, "existing final summary") == payload, "existing final summary provenance differs")
    else:
        atomic_json(result_path, payload)
    ledger({"stage": "final-summary", "event": "completed", "evidence_ref": str(result_path.relative_to(ROOT)), "best_source_key": best["source_key"], "best_three_axis_macro_rmse": best["metrics"]["three_axis_macro_rmse"], "resource_scope": "none", "gpu_scope_authorization": "default", "decision": "complete"})


def run_full(*, reuse: bool) -> None:
    verify_preflight_requirement()
    ensure(PY.is_file() and VLLM.is_file(), "project runtime is unavailable")
    # Generate each independent training/validation rationale source before
    # any encoder is trained.  No source is concatenated or averaged.
    for offset, source in enumerate(selected_sources()):
        train = generation_config(source, "train", "full")
        validation = generation_config(source, "validation", "full")
        train_path = write_config(f"generation-{source}-train-full", train)
        validation_path = write_config(f"generation-{source}-validation-full", validation)
        train_exists = (Path(train["restricted_output_dir"]) / "aggregate_generation_report.json").is_file()
        validation_exists = (Path(validation["restricted_output_dir"]) / "aggregate_generation_report.json").is_file()
        if train_exists and validation_exists:
            ensure(reuse, "full generation artifacts already exist outside resume mode")
            run_generation(train, train_path, 18710 + offset, reuse=True)
            run_generation(validation, validation_path, 18710 + offset, reuse=True)
        else:
            # One model start serves both split-specific calls for an adapter.
            # If a prior interrupted artifact exists, resume verifies it and
            # executes only the missing split without overwriting anything.
            pending: list[tuple[Mapping[str, Any], Path]] = []
            for config, path in ((train, train_path), (validation, validation_path)):
                output = Path(config["restricted_output_dir"])
                if (output / "aggregate_generation_report.json").is_file():
                    ensure(reuse, f"generation artifact already exists outside resume mode: {output}")
                    report = verify_generation(config)
                    ledger({"stage": f"generation-{config['source_key']}-{config['source']}-full", "event": "reused_verified_completed", "counts": report["counts"], "hard_gates": report["hard_gates"]})
                    continue
                ensure(not output.exists(), f"partial generation artifact requires a preserved failure review: {output}")
                pending.append((config, path))
            if pending:
                server_config = pending[0][0]
                with generation_server(server_config, 18710 + offset) as attestation:
                    for config, path in pending:
                        run_stage(f"generation-{config['source_key']}-{config['source']}-full", [str(PY), "scripts/generate_rlaif_top3_rationales.py", "--config", str(path), "--endpoint", f"http://127.0.0.1:{18710 + offset}", "--server-attestation", str(attestation)], base_env("0,1,2,3"))
                        report = verify_generation(config)
                        ledger({"stage": f"generation-{config['source_key']}-{config['source']}-full", "event": "aggregate", "counts": report["counts"], "hard_gates": report["hard_gates"]})
    reports: dict[str, Mapping[str, Any]] = {}
    for source in selected_sources():
        training = regression_config(source, "full")
        training_path = write_config(f"encoder-{source}-full", training)
        run_training(training, training_path, reuse=reuse)
        evaluation = evaluation_config(source)
        evaluation_path = write_config(f"encoder-eval-{source}-validation", evaluation)
        reports[source] = run_evaluation(evaluation, evaluation_path, reuse=reuse)
    final_summary(reports, reuse=reuse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("gpu0-preflight", "full", "full-resume"))
    args = parser.parse_args()
    global RUN_ROOT
    name = {"gpu0-preflight": "gpu0-preflight", "full": "full", "full-resume": "full-resume-001"}[args.mode]
    RUN_ROOT = RUN_BASE / name
    if RUN_ROOT.exists():
        entries = {entry.name for entry in RUN_ROOT.iterdir()}
        ensure(not (entries - {"runner.log"}), f"runtime root already contains preserved state: {RUN_ROOT}")
    else:
        RUN_ROOT.mkdir(mode=0o700, parents=True)
    manifest = {
        "schema_version": "mal2026-rlaif-top3-encoder-v1-runner-v1",
        "status": "running",
        "mode": args.mode,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "physical_gpus": [0] if args.mode == "gpu0-preflight" else [0, 1, 2, 3],
        "gpu_scope_authorization": "default project GPUs 0--3; GPU0 first preflight",
        "selected_sources": [{"source_key": source, "rank": SELECTIONS[source]["rank"], "arm": SELECTIONS[source]["arm"], "frozen_macro": SELECTIONS[source]["frozen_macro"]} for source in selected_sources()],
        "task_card": {
            "deliverable": "three independent RLAIF-generated rationale datasets and three Qwen2.5 encoder validations",
            "completion_predicate": "all generation hard gates, all three finite encoder trainings, and all three 400-essay validations complete",
            "permitted_inputs": "canonical train/validation writings, selected RLAIF adapters, local Qwen2.5 snapshot",
            "outputs": "ignored restricted rationales and ignored aggregate/checkpoint roots only",
            "test_ladder": "GPU0 one-request+one-update preflight, then TP4 generation and DDP4 train/evaluation",
            "escalation": "stop on a generation/schema/training/evaluation gate or project-GPU conflict",
        },
        "scripts": {name: sha(path) for name, path in {
            "runner": Path(__file__),
            "protocol": ROOT / "src" / "mal2026" / "rlaif_top3_encoder.py",
            "generation": ROOT / "scripts" / "generate_rlaif_top3_rationales.py",
            "training": ROOT / "scripts" / "train_rlaif_top3_score_regression.py",
            "evaluation": ROOT / "scripts" / "evaluate_rlaif_top3_score_regression.py",
        }.items()},
    }
    atomic_json(RUN_ROOT / "manifest.json", manifest)
    try:
        if args.mode == "gpu0-preflight":
            run_preflight()
        else:
            run_full(reuse=args.mode == "full-resume")
    except Exception:
        manifest["status"] = "failed"
        manifest["failed_at"] = now()
        atomic_json(RUN_ROOT / "manifest.json", manifest)
        raise
    manifest["status"] = "completed"
    manifest["completed_at"] = now()
    atomic_json(RUN_ROOT / "manifest.json", manifest)


if __name__ == "__main__":
    main()
