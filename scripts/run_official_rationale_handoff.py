#!/usr/bin/env python3
"""Select the final rationale model and generate bootstrap-conditioned data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_rationale_handoff import (  # noqa: E402
    AXES, HandoffConfig, combine_rationales, convert_bootstrap_scores, file_sha256, need,
)


PYTHON = ROOT / ".venv-standard" / "bin" / "python"
GENERATOR = ROOT / "scripts" / "generate_official_rationales_vllm.py"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    need(not path.exists(), f"refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rewrite_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_processes(gpus: Sequence[int]) -> list[str]:
    result = subprocess.run(["nvidia-smi", f"--id={','.join(map(str, gpus))}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True, capture_output=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def aliases(candidate: Mapping[str, Any]) -> dict[str, str]:
    return {task: f"mal2026-final-rationale-{task}" for task in candidate["adapters"]}


def server_command(config: HandoffConfig, candidate: Mapping[str, Any], tensor_parallel: int) -> list[str]:
    vllm = config["vllm"]
    command = [
        vllm["executable"], "serve", candidate["model_path"], "--served-model-name", candidate["model_id"],
        "--host", vllm["host"], "--port", str(vllm["port"]), "--tensor-parallel-size", str(tensor_parallel),
        "--dtype", "bfloat16", "--max-model-len", str(vllm["max_model_len"]),
        "--max-num-seqs", str(vllm["max_num_seqs"]), "--max-num-batched-tokens", str(vllm["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(vllm["gpu_memory_utilization"]), "--generation-config", "vllm",
        "--enable-prefix-caching", "--enable-lora", "--max-loras", str(len(candidate["adapters"])),
        "--max-lora-rank", str(vllm["max_lora_rank"]), "--lora-modules",
        *[f"{aliases(candidate)[task]}={adapter['path']}" for task, adapter in sorted(candidate["adapters"].items())],
        "--compilation-config", '{"mode":0,"cudagraph_mode":"NONE"}',
    ]
    return command


def start_server(config: HandoffConfig, candidate: Mapping[str, Any], gpus: Sequence[int], log: Path) -> tuple[subprocess.Popen[str], Any]:
    need(not gpu_processes(gpus), f"GPU boundary conflict on {list(gpus)}; existing processes were not altered")
    endpoint = f"http://{config['vllm']['host']}:{config['vllm']['port']}"
    try:
        urlopen(endpoint + "/health", timeout=1)
    except Exception:
        pass
    else:
        raise RuntimeError("vLLM port is already serving another process")
    handle = log.open("x", encoding="utf-8")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    process = subprocess.Popen(server_command(config, candidate, len(gpus)), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        for _ in range(360):
            need(process.poll() is None, "vLLM server exited before health gate")
            try:
                with urlopen(endpoint + "/health", timeout=2) as response:
                    if response.status == 200:
                        return process, handle
            except Exception:
                time.sleep(1)
        raise RuntimeError("vLLM health timeout")
    except Exception:
        stop_server(process, handle)
        raise


def stop_server(process: subprocess.Popen[str] | None, handle: Any | None) -> None:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=30)
    if handle is not None:
        handle.close()


def attestation(config: HandoffConfig, candidate: Mapping[str, Any], task: str, endpoint: str, gpu_scope: Sequence[int], selection_sha: str, path: Path) -> None:
    adapter = candidate["adapters"][task]
    write_new_json(path, {
        "schema_version": "mal2026-official-rationale-vllm-server-attestation-v1", "created_at": now(),
        "endpoint": endpoint, "physical_gpu": gpu_scope[0], "physical_gpus": list(gpu_scope), "tensor_parallel_size": len(gpu_scope),
        "max_model_len": config["vllm"]["max_model_len"], "max_num_seqs": config["vllm"]["max_num_seqs"], "max_num_batched_tokens": config["vllm"]["max_num_batched_tokens"],
        "enforce_eager": False, "compilation_mode": 0, "cudagraph_mode": "NONE", "effective_eager_equivalent": True,
        "inference_dtype": "bfloat16", "model_id": candidate["model_id"], "model_revision": candidate["model_revision"],
        "model_config_sha256": candidate["model_config_sha256"], "adapter_alias": aliases(candidate)[task], "adapter_path": str(Path(adapter["path"]).resolve()),
        "adapter_training_completion_sha256": adapter["training_completion_sha256"], "task": task,
        "winner_selection_sha256": selection_sha, "server_environment_verified": True,
    })


def generation_command(config: HandoffConfig, task: str, split: str, expected: int, score: Path, output: Path, attest: Path) -> list[str]:
    endpoint = f"http://{config['vllm']['host']}:{config['vllm']['port']}"
    return [str(PYTHON), str(GENERATOR), "--run-id", output.name, "--task", task, "--split", split, "--expected", str(expected), "--score-file", str(score), "--output-dir", str(output), "--endpoint", endpoint, "--model", f"mal2026-final-rationale-{task}", "--server-attestation", str(attest), "--max-inflight", str(config["generation"]["max_inflight"])]


def run_client(command: Sequence[str], log: Path) -> None:
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(list(command), cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, stdout=handle, stderr=subprocess.STDOUT)
    need(result.returncode == 0, f"generation client failed: {log.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = HandoffConfig.from_json(args.config)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "gpu_started": False, "candidate_count": len(config.candidates), "required_methods": ["official_sft", "aihub_sft", "dpo", "grpo"], "gpu0_real_row_smoke": True, "full_gpu_scope": [0, 1, 2, 3], "populations": {"train": 2000, "validation": 400}}, sort_keys=True))
        return
    resolved = config.validate_dependencies()
    if args.validate_only:
        print(json.dumps({"status": "validated", "gpu_started": False, "winner": resolved["winner"]}, sort_keys=True))
        return

    candidate, bootstrap = resolved["candidate"], resolved["bootstrap"]
    runtime = Path(config["runtime_output_root"]); restricted = Path(config["restricted_output_root"]); selection_path = Path(config["selection_output_path"])
    need(PYTHON.is_file() and GENERATOR.is_file() and importlib.metadata.version("vllm") == config["vllm"]["version"], "vLLM runtime differs")
    need(not runtime.exists() and not restricted.exists() and not selection_path.exists(), "handoff outputs must be fresh")
    runtime.mkdir(parents=True); (runtime / "logs").mkdir(); (runtime / "attestations").mkdir(); restricted.mkdir(mode=0o700, parents=True)
    selection = {
        "schema_version": "mal2026-official-rationale-final-selection-v1", "status": "completed", "run_id": config["run_id"],
        "winner": resolved["winner"], "candidate_identity": candidate_identity_for_output(candidate), "evaluated_candidates": resolved["evaluated"],
        "selection_rule": "highest fixed-Q4 macro mean, then highest worst cell, then strict parse rate, then candidate key",
        "selection_source": "single fixed public-spec proxy judge only",
        "repeated_validation_caveat": "the same canonical validation population has been inspected repeatedly; ranking is comparative and not an untouched generalization estimate",
        "judge": {key: config["judge"][key] for key in ("contract_sha256", "model_sha256", "directional_gate_sha256", "injection_gate_sha256", "prompt_kind", "repeats_per_validation_record")},
    }
    write_new_json(selection_path, selection); selection_sha = file_sha256(selection_path)
    manifest_path = runtime / "manifest.json"
    manifest = {"schema_version": "mal2026-official-rationale-handoff-run-v1", "status": "running", "run_id": config["run_id"], "started_at": now(), "winner_key": candidate["key"], "winner_selection_sha256": selection_sha, "gpu_scope": {"smoke": [0], "full": [0, 1, 2, 3]}, "human_or_reference_score_read_or_prompted": False}
    write_new_json(manifest_path, manifest)

    selected_scores = bootstrap["selected_score_files"]
    adapted: dict[str, Path] = {}
    for split, count in (("train", 2000), ("validation", 400)):
        adapted[split] = restricted / f"bootstrap-scores.{split}.jsonl"
        convert_bootstrap_scores(Path(selected_scores[f"{split}_path"]), adapted[split], count, split)
    smoke_score = restricted / "bootstrap-scores.train-smoke.jsonl"
    with adapted["train"].open(encoding="utf-8") as source, smoke_score.open("x", encoding="utf-8") as target:
        target.write(next(source))

    tasks = ("bundle",) if candidate["structure"] == "bundle" else AXES
    process: subprocess.Popen[str] | None = None; handle: Any | None = None
    try:
        endpoint = f"http://{config['vllm']['host']}:{config['vllm']['port']}"
        process, handle = start_server(config, candidate, [0], runtime / "logs" / "vllm-gpu0-smoke.log")
        smoke_inputs: dict[str, Path] = {}
        for task in tasks:
            attest = runtime / "attestations" / f"smoke-{task}.json"; attestation(config, candidate, task, endpoint, [0], selection_sha, attest)
            output = restricted / "smoke" / task; smoke_inputs[task] = output / "generated_rationales.jsonl"
            run_client(generation_command(config, task, "train", 1, smoke_score, output, attest), runtime / "logs" / f"smoke-{task}.log")
        combine_rationales(smoke_inputs, restricted / "rationales.train-smoke.jsonl", 1, candidate["structure"])
        stop_server(process, handle); process = None; handle = None

        process, handle = start_server(config, candidate, [0, 1, 2, 3], runtime / "logs" / "vllm-gpu0-3-full.log")
        generated: dict[str, dict[str, Path]] = {"train": {}, "validation": {}}
        clients: list[tuple[subprocess.Popen[Any], Any, Path]] = []
        for task in tasks:
            attest = runtime / "attestations" / f"full-{task}.json"; attestation(config, candidate, task, endpoint, [0, 1, 2, 3], selection_sha, attest)
            for split, count in (("train", 2000), ("validation", 400)):
                output = restricted / "raw" / split / task; generated[split][task] = output / "generated_rationales.jsonl"
                log = runtime / "logs" / f"client-{split}-{task}.log"; log_handle = log.open("x", encoding="utf-8")
                client = subprocess.Popen(generation_command(config, task, split, count, adapted[split], output, attest), cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, stdout=log_handle, stderr=subprocess.STDOUT)
                clients.append((client, log_handle, log))
        for client, log_handle, log in clients:
            code = client.wait(); log_handle.close(); need(code == 0, f"full generation failed: {log.name}")
        rationale_sha = {}
        for split, count in (("train", 2000), ("validation", 400)):
            final = restricted / f"rationales.{split}.jsonl"
            rationale_sha[split] = combine_rationales(generated[split], final, count, candidate["structure"])

        final_manifest = {
            "schema_version": "mal2026-official-rationale-score-matrix-handoff-v1", "status": "completed", "rationale_key": candidate["key"],
            "bootstrap_selection_sha256": config["bootstrap_selection_sha256"], "bootstrap_selected_result_sha256": bootstrap["selected_result_sha256"],
            "score_train_sha256": selected_scores["train_sha256"], "score_validation_sha256": selected_scores["validation_sha256"],
            "rationale_train_sha256": rationale_sha["train"], "rationale_validation_sha256": rationale_sha["validation"],
            "winner_selection_sha256": selection_sha, "winner_candidate_identity_sha256": resolved["winner"]["candidate_identity_sha256"],
            "winner_evaluation_sha256": resolved["winner"]["evaluation_sha256"], "structure": candidate["structure"],
            "model_config_sha256": candidate["model_config_sha256"],
            "model_id": candidate["model_id"], "model_revision": candidate["model_revision"], "model_binding_sha256": candidate["model_binding_sha256"],
            "adapter_bindings": {task: {key: value[key] for key in ("adapter_config_sha256", "adapter_model_sha256", "training_completion_sha256")} for task, value in candidate["adapters"].items()},
            "judge_contract_sha256": config["judge"]["contract_sha256"], "judge_model_sha256": config["judge"]["model_sha256"],
            "directional_gate_sha256": config["judge"]["directional_gate_sha256"], "injection_gate_sha256": config["judge"]["injection_gate_sha256"],
            "score_kind": "bootstrap_model_actual_emitted_integer_prediction", "human_or_reference_score_read_or_prompted": False,
        }
        handoff_manifest = restricted / "aggregate_handoff_manifest.json"
        write_new_json(handoff_manifest, final_manifest)
        manifest.update({"status": "completed", "completed_at": now(), "restricted_handoff_manifest_path": str(handoff_manifest.resolve()), "restricted_handoff_manifest_sha256": file_sha256(handoff_manifest), "rationale_train_sha256": rationale_sha["train"], "rationale_validation_sha256": rationale_sha["validation"]})
        rewrite_json(manifest_path, manifest)
    except Exception as exc:
        manifest.update({"status": "failed", "failed_at": now(), "failure": f"{type(exc).__name__}: {exc}"}); rewrite_json(manifest_path, manifest); raise
    finally:
        stop_server(process, handle)


def candidate_identity_for_output(candidate: Mapping[str, Any]) -> dict[str, Any]:
    from mal2026.official_rationale_handoff import candidate_identity
    return candidate_identity(candidate)


if __name__ == "__main__":
    main()
