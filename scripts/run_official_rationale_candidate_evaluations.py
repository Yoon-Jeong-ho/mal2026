#!/usr/bin/env python3
"""Generate and evaluate every declared final-rationale candidate.

Actual execution performs a GPU0 real train-row generation/judge smoke before
full validation generation. Compatible candidates share one vLLM base-model
server. The exact pinned Q4 judge then runs as four llama.cpp replicas on GPUs
0--3; its ten calls per participant keep temperature 0 and seed 42 unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
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

from mal2026.official_rationale_candidate_evaluation import (  # noqa: E402
    aggregate_deterministic_repeats, compose_participants, evaluation_payload, load_emitted_scores,
)
from mal2026.official_rationale_handoff import (  # noqa: E402
    AXES, HandoffConfig, candidate_identity_sha256, combine_rationales,
    convert_bootstrap_scores, file_sha256, iter_jsonl, need, read_json,
    validate_training_completion,
)

PYTHON = ROOT / ".venv-standard/bin/python"
GENERATOR = ROOT / "scripts/generate_official_rationales_vllm.py"
JUDGE = ROOT / "scripts/evaluate_official_q4_judge.py"
LLAMA_SERVER = ROOT / "outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/build-cuda/bin/llama-server"
LLAMA_REPO = ROOT / "outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/llama.cpp"
LLAMA_REVISION = "571d0d540df04f25298d0e159e520d9fc62ed121"
LLAMA_TAG = "b10068"
JUDGE_RESTRICTED = ROOT / "data/processed/restricted/official_prompt_alignment_v1/q4_judge"
JUDGE_AGGREGATE = ROOT / "outputs/official-prompt-alignment-v1/q4-judge"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_processes(gpus: Sequence[int]) -> list[str]:
    result = subprocess.run(["nvidia-smi", f"--id={','.join(map(str, gpus))}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True, capture_output=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def stop_processes(processes: Sequence[tuple[subprocess.Popen[Any], Any]]) -> None:
    for process, _ in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for process, handle in processes:
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        handle.close()


def wait_health(endpoints: Sequence[str], timeout: int = 900) -> None:
    deadline = time.time() + timeout
    pending = set(endpoints)
    while pending and time.time() < deadline:
        for endpoint in tuple(pending):
            try:
                with urlopen(endpoint + "/health", timeout=3) as response:
                    if response.status == 200:
                        pending.remove(endpoint)
            except Exception:
                pass
        if pending: time.sleep(2)
    need(not pending, f"server health timeout: {sorted(pending)}")


def candidate_tasks(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    return ("bundle",) if candidate["structure"] == "bundle" else AXES


def candidate_namespace(candidate: Mapping[str, Any]) -> str:
    return f"{candidate['key']}-{candidate_identity_sha256(candidate)[:12]}"


def fresh_numbered_path(path: Path) -> Path:
    if not path.exists(): return path
    for attempt in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-attempt-{attempt:03d}{path.suffix}")
        if not candidate.exists(): return candidate
    raise RuntimeError("runtime attempt namespace exhausted")


def alias(candidate: Mapping[str, Any], task: str) -> str:
    return f"mal2026-{candidate['key'].replace('_', '-')}-{task}"


def compatible_groups(candidates: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate["model_path"], candidate["model_id"], candidate["model_revision"])].append(candidate)
    return [sorted(group, key=lambda item: item["key"]) for _, group in sorted(grouped.items())]


def validate_candidate_artifacts(config: HandoffConfig) -> dict[str, Any]:
    """Validate every dependency needed to produce evaluations, excluding evaluations themselves."""
    raw = config.raw
    need("REQUIRED_" not in json.dumps({k: v for k, v in raw.items() if k != "candidates"}), "unresolved non-candidate placeholder")
    bootstrap_path = Path(raw["bootstrap_selection_path"])
    need(bootstrap_path.is_file() and file_sha256(bootstrap_path) == raw["bootstrap_selection_sha256"], "bootstrap selection SHA differs")
    bootstrap = read_json(bootstrap_path, "bootstrap selection")
    need(bootstrap.get("status") == "stage_a_completed" and bootstrap.get("canonical_validation_used_for_selection") is False, "bootstrap selection contract differs")
    scores = bootstrap.get("selected_score_files", {})
    for split, expected in (("train", 2000), ("validation", 400)):
        path = Path(scores.get(f"{split}_path", ""))
        need(path.is_file() and file_sha256(path) == scores.get(f"{split}_sha256") and scores.get(f"{split}_records") == expected, f"bootstrap {split} score differs")
    judge = raw["judge"]
    for key in ("contract", "model", "directional_gate", "injection_gate"):
        path = Path(judge[f"{key}_path"]); need(path.is_file() and file_sha256(path) == judge[f"{key}_sha256"], f"judge {key} differs")
    directional = read_json(Path(judge["directional_gate_path"]), "directional gate")
    injection = read_json(Path(judge["injection_gate_path"]), "injection gate")
    need(directional.get("schema_version") == "mal2026-official-proxy-judge-contrastive-gate-v1" and directional.get("status") == "passed" and directional.get("rl_with_this_proxy_judge_allowed") is True, "directional judge gate has not passed")
    need(injection.get("schema_version") == "mal2026-official-proxy-judge-rl-safety-gate-v1" and injection.get("status") == "passed" and injection.get("directional_contrastive_gate_passed") is True and injection.get("prompt_injection_gate_passed") is True and injection.get("rl_allowed") is True, "combined injection judge gate has not passed")
    for candidate in config.candidates:
        need("REQUIRED_" not in json.dumps({k: v for k, v in candidate.items() if not k.startswith("evaluation_")}), f"candidate artifact placeholder remains: {candidate['key']}")
        model = Path(candidate["model_path"]); binding = Path(candidate["model_binding_path"])
        need(model.is_dir() and file_sha256(model / "config.json") == candidate["model_config_sha256"], f"candidate model differs: {candidate['key']}")
        need(binding.is_file() and file_sha256(binding) == candidate["model_binding_sha256"], f"candidate model binding differs: {candidate['key']}")
        for task, adapter in candidate["adapters"].items():
            root, completion = Path(adapter["path"]), Path(adapter["training_completion_path"])
            need(file_sha256(root / "adapter_config.json") == adapter["adapter_config_sha256"] and file_sha256(root / "adapter_model.safetensors") == adapter["adapter_model_sha256"], f"candidate adapter differs: {candidate['key']}/{task}")
            need(file_sha256(completion) == adapter["training_completion_sha256"], f"candidate completion differs: {candidate['key']}/{task}")
            validate_training_completion(candidate, task, read_json(completion, "training completion"))
    return {"bootstrap": bootstrap, "score_files": scores}


def start_vllm(config: HandoffConfig, group: Sequence[Mapping[str, Any]], gpus: Sequence[int], log: Path) -> tuple[list[tuple[subprocess.Popen[Any], Any]], str, dict[str, str]]:
    need(not gpu_processes(gpus), f"GPU conflict on {list(gpus)}; existing processes were not altered")
    base = group[0]; endpoint = f"http://{config['vllm']['host']}:{config['vllm']['port']}"
    aliases = {alias(candidate, task): task for candidate in group for task in candidate_tasks(candidate)}
    modules = [f"{alias(candidate, task)}={candidate['adapters'][task]['path']}" for candidate in group for task in candidate_tasks(candidate)]
    command = [
        config["vllm"]["executable"], "serve", base["model_path"], "--served-model-name", base["model_id"],
        "--host", config["vllm"]["host"], "--port", str(config["vllm"]["port"]), "--tensor-parallel-size", str(len(gpus)),
        "--dtype", "bfloat16", "--max-model-len", str(config["vllm"]["max_model_len"]),
        "--max-num-seqs", str(config["vllm"]["max_num_seqs"]), "--max-num-batched-tokens", str(config["vllm"]["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(config["vllm"]["gpu_memory_utilization"]), "--generation-config", "vllm",
        "--enable-prefix-caching", "--enable-lora", "--max-loras", "1", "--max-cpu-loras", str(len(modules)),
        "--max-lora-rank", str(config["vllm"]["max_lora_rank"]), "--lora-modules", *modules,
    ]
    if len(gpus) > 1: command.append("--fully-sharded-loras")
    log = fresh_numbered_path(log); log.parent.mkdir(parents=True, exist_ok=True); handle = log.open("x", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}, stdout=handle, stderr=subprocess.STDOUT)
    try:
        wait_health([endpoint]); need(process.poll() is None, "vLLM exited during startup")
    except Exception:
        stop_processes([(process, handle)]); raise
    return [(process, handle)], endpoint, aliases


def generation_attestation(path: Path, endpoint: str, aliases: Mapping[str, str], group: Sequence[Mapping[str, Any]], gpus: Sequence[int]) -> None:
    atomic_json(path, {
        "schema_version": "mal2026-official-rationale-vllm-server-attestation-v1", "endpoint": endpoint,
        "adapter_aliases": dict(sorted(aliases.items())), "physical_gpus": list(gpus),
        "compatible_base_group": [candidate_identity_sha256(candidate) for candidate in group],
        "human_or_reference_score_read_or_prompted": False,
    })


def completed_generation(root: Path, expected: int, score_sha256: str) -> Path | None:
    report, rows = root / "aggregate_generation_report.json", root / "generated_rationales.jsonl"
    if report.is_file() and rows.is_file():
        value = read_json(report, "generation report")
        if value.get("status") == "completed" and value.get("counts", {}).get("records") == expected and value.get("generated_rationales_sha256") == file_sha256(rows) and value.get("score_file_sha256") == score_sha256:
            return rows
    return None


def run_generation(config: HandoffConfig, candidate: Mapping[str, Any], task: str, split: str, expected: int, scores: Path, endpoint: str, attestation: Path, root: Path) -> Path:
    stem = root / candidate_namespace(candidate) / split / task; score_sha = file_sha256(scores)
    for attempt in range(1, 1000):
        output = stem / f"attempt-{attempt:03d}"
        completed = completed_generation(output, expected, score_sha)
        if completed is not None: return completed
        if output.exists(): continue
        command = [str(PYTHON), str(GENERATOR), "--run-id", f"{candidate['key']}-{split}-{task}-{attempt:03d}", "--task", task, "--split", split, "--expected", str(expected), "--score-file", str(scores), "--output-dir", str(output), "--endpoint", endpoint, "--model", alias(candidate, task), "--server-attestation", str(attestation), "--max-inflight", str(config["generation"]["max_inflight"])]
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        if result.returncode == 0:
            completed = completed_generation(output, expected, score_sha); need(completed is not None, "generation completion is invalid"); return completed
    raise RuntimeError("generation attempt namespace exhausted")


def combine_candidate(candidate: Mapping[str, Any], generated: Mapping[str, Path], root: Path, expected: int, split: str) -> Path:
    for attempt in range(1, 1000):
        output = root / candidate_namespace(candidate) / split / f"combined-attempt-{attempt:03d}.jsonl"
        if output.is_file():
            try:
                rows = list(iter_jsonl(output))
                if len(rows) == expected and all(set(row) == {"source_id", "rationales"} and isinstance(row["rationales"], dict) and set(row["rationales"]) == set(AXES) for row in rows): return output
            except Exception: continue
        combine_rationales(generated, output, expected, candidate["structure"]); return output
    raise RuntimeError("combined rationale attempt namespace exhausted")


def compose_candidate(candidate: Mapping[str, Any], score: Path, rationale: Path, root: Path, expected: int, split: str) -> Path:
    emitted = load_emitted_scores(score, expected)
    for attempt in range(1, 1000):
        output = root / candidate_namespace(candidate) / split / f"participants-attempt-{attempt:03d}.jsonl"
        if output.is_file():
            try:
                rows = list(iter_jsonl(output))
                if len(rows) == expected and {row["source_id"] for row in rows} == set(emitted) and all(all(row["participant_output"][axis]["score"] == emitted[row["source_id"]][axis] for axis in AXES) for row in rows): return output
            except Exception: continue
        compose_participants(score, rationale, output, expected); return output
    raise RuntimeError("participant attempt namespace exhausted")


def start_q4(config: HandoffConfig, gpus: Sequence[int], runtime: Path) -> tuple[list[tuple[subprocess.Popen[Any], Any]], list[str], Path]:
    need(not gpu_processes(gpus), f"GPU conflict on {list(gpus)}; existing processes were not altered")
    judge = config["judge"]
    need(LLAMA_SERVER.is_file() and os.access(LLAMA_SERVER, os.X_OK), "pinned llama-server unavailable")
    need(subprocess.check_output(["git", "-C", str(LLAMA_REPO), "rev-parse", "HEAD"], text=True).strip() == LLAMA_REVISION, "llama.cpp revision differs")
    need(subprocess.check_output(["git", "-C", str(LLAMA_REPO), "describe", "--tags", "--exact-match"], text=True).strip() == LLAMA_TAG, "llama.cpp tag differs")
    runtime = fresh_numbered_path(runtime); processes: list[tuple[subprocess.Popen[Any], Any]] = []; endpoints: list[str] = []
    for gpu in gpus:
        port = 19100 + gpu; endpoint = f"http://127.0.0.1:{port}"; endpoints.append(endpoint)
        log = runtime / "logs" / f"llama-q4-gpu{gpu}.log"; log.parent.mkdir(parents=True, exist_ok=True); handle = log.open("x", encoding="utf-8")
        command = [str(LLAMA_SERVER), "--model", judge["model_path"], "--host", "127.0.0.1", "--port", str(port), "--n-gpu-layers", "99", "--parallel", "4", "--ctx-size", "32768", "--batch-size", "2048", "--ubatch-size", "512", "--no-webui", "--reasoning", "off"]
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=handle, stderr=subprocess.STDOUT); processes.append((process, handle))
    try: wait_health(endpoints, timeout=300)
    except Exception: stop_processes(processes); raise
    attestation = runtime / "q4-server-attestation.json"
    atomic_json(attestation, {"schema_version": "mal2026-official-q4-judge-server-attestation-v1", "created_at": now(), "physical_gpus": list(gpus), "server_endpoints": endpoints, "parallel_per_server": 4, "context_per_slot": 8192, "model_sha256": judge["model_sha256"], "llama_revision": LLAMA_REVISION, "llama_tag": LLAMA_TAG})
    return processes, endpoints, attestation


def complete_repeat(stem: str, expected: int, participant_sha256: str) -> tuple[Path, Path] | None:
    report = JUDGE_AGGREGATE / stem / "aggregate_judge_report.json"; records = JUDGE_RESTRICTED / stem / "judge_records.jsonl"
    if report.is_file() and records.is_file():
        value = read_json(report, "judge report")
        if value.get("status") == "completed" and value.get("counts", {}).get("records") == expected and value.get("judge_records_sha256") == file_sha256(records) and value.get("participant_sha256") == participant_sha256: return report, records
    return None


def run_judge(candidate: Mapping[str, Any], participant: Path, split: str, expected: int, repeat: int, endpoints: Sequence[str], attestation: Path) -> tuple[Path, Path]:
    participant_sha = file_sha256(participant)
    base = f"official-rationale-candidate-{candidate_namespace(candidate)}-{split}-repeat-{repeat:02d}"
    for attempt in range(1, 1000):
        run_id = f"{base}-attempt-{attempt:03d}"
        completed = complete_repeat(run_id, expected, participant_sha)
        if completed is not None: return completed
        if (JUDGE_AGGREGATE / run_id).exists() or (JUDGE_RESTRICTED / run_id).exists(): continue
        command = [str(PYTHON), str(JUDGE), "--run-id", run_id, "--participant-file", str(participant), "--expected", str(expected), "--split", split, "--max-inflight", str(4 * len(endpoints)), "--server-attestation", str(attestation)]
        for endpoint in endpoints: command += ["--endpoint", endpoint]
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        if result.returncode == 0:
            completed = complete_repeat(run_id, expected, participant_sha); need(completed is not None, "judge completion is invalid"); return completed
    raise RuntimeError("judge attempt namespace exhausted")


def adapted_score(score_path: Path, root: Path, expected: int, split: str) -> Path:
    output = root / f"bootstrap-{split}-emitted-integers.jsonl"
    if output.is_file():
        actual = load_emitted_scores(output, expected)
        source_rows = list(iter_jsonl(score_path))
        need(len(source_rows) == expected, "bootstrap source population differs")
        projected = {row["source_id"]: row["scores"] for row in source_rows if set(row) == {"source_id", "split", "arm", "scores"} and row["split"] == split}
        need(actual == projected, "adapted bootstrap score binding differs")
        return output
    convert_bootstrap_scores(score_path, output, expected, split); return output


def first_score(source: Path, output: Path) -> Path:
    with source.open(encoding="utf-8") as handle:
        line = handle.readline()
    need(bool(line.strip()), "smoke score row is unavailable")
    if output.is_file():
        need(output.read_text(encoding="utf-8") == line, "smoke score binding differs")
        return output
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.write_text(line, encoding="utf-8")
    return output


def execute(config: HandoffConfig, dependencies: Mapping[str, Any]) -> None:
    restricted = Path(config["restricted_output_root"]).parent / "candidate_evaluations" / config["run_id"]
    runtime = Path(config["runtime_output_root"]).parent / "candidate_evaluations" / config["run_id"]
    aggregate = ROOT / "outputs/official-rationale-candidate-evaluations-v1" / config["run_id"]
    restricted.mkdir(mode=0o700, parents=True, exist_ok=True); runtime.mkdir(parents=True, exist_ok=True); aggregate.mkdir(parents=True, exist_ok=True)
    pending: list[Mapping[str, Any]] = []
    for candidate in config.candidates:
        expected_output = aggregate / candidate["key"] / "evaluation.json"
        need(Path(candidate["evaluation_path"]).resolve() == expected_output.resolve(), f"candidate evaluation output path differs: {candidate['key']}")
        if expected_output.is_file():
            existing = read_json(expected_output, f"candidate evaluation {candidate['key']}")
            need(existing.get("schema_version") == "mal2026-official-rationale-candidate-evaluation-v1" and existing.get("status") == "completed" and existing.get("candidate_identity_sha256") == candidate_identity_sha256(candidate) and existing.get("bootstrap_validation_score_sha256") == dependencies["score_files"]["validation_sha256"], f"existing candidate evaluation differs: {candidate['key']}")
        else:
            pending.append(candidate)
    if not pending: return
    scores = {split: adapted_score(Path(dependencies["score_files"][f"{split}_path"]), restricted, count, split) for split, count in (("train", 2000), ("validation", 400))}
    smoke_score = first_score(scores["train"], restricted / "bootstrap-train-smoke-emitted-integer.jsonl")
    groups = compatible_groups(pending)
    participants: dict[str, dict[str, Any]] = defaultdict(dict)
    # Real-row smoke generation, grouped by compatible base.
    for group_index, group in enumerate(groups, 1):
        processes, endpoint, aliases = start_vllm(config, group, [0], runtime / f"generation-group-{group_index}-smoke.log")
        try:
            attest = fresh_numbered_path(runtime / f"generation-group-{group_index}-smoke-attestation.json"); generation_attestation(attest, endpoint, aliases, group, [0])
            for candidate in group:
                raw = {task: run_generation(config, candidate, task, "train", 1, smoke_score, endpoint, attest, restricted / "generation-smoke") for task in candidate_tasks(candidate)}
                rationales = combine_candidate(candidate, raw, restricted / "generation-smoke", 1, "train")
                participants[candidate["key"]]["smoke"] = compose_candidate(candidate, smoke_score, rationales, restricted / "generation-smoke", 1, "train")
        finally: stop_processes(processes)
    q4, endpoints, attest = start_q4(config, [0], runtime / "judge-smoke")
    try:
        for candidate in pending: run_judge(candidate, participants[candidate["key"]]["smoke"], "train", 1, 0, endpoints, attest)
    finally: stop_processes(q4)
    # Full validation generation, again sharing compatible bases.
    for group_index, group in enumerate(groups, 1):
        processes, endpoint, aliases = start_vllm(config, group, [0, 1, 2, 3], runtime / f"generation-group-{group_index}-full.log")
        try:
            attest = fresh_numbered_path(runtime / f"generation-group-{group_index}-full-attestation.json"); generation_attestation(attest, endpoint, aliases, group, [0, 1, 2, 3])
            for candidate in group:
                raw = {task: run_generation(config, candidate, task, "validation", 400, scores["validation"], endpoint, attest, restricted / "generation-full") for task in candidate_tasks(candidate)}
                rationales = combine_candidate(candidate, raw, restricted / "generation-full", 400, "validation")
                participants[candidate["key"]]["rationale"] = rationales
                participants[candidate["key"]]["validation"] = compose_candidate(candidate, scores["validation"], rationales, restricted / "generation-full", 400, "validation")
                participants[candidate["key"]]["generation_reports"] = {
                    task: {
                        "aggregate_report_sha256": file_sha256(path.parent / "aggregate_generation_report.json"),
                        "server_attestation_sha256": read_json(path.parent / "manifest.json", "generation manifest")["server_attestation_sha256"],
                    } for task, path in raw.items()
                }
        finally: stop_processes(processes)
    q4, endpoints, attest = start_q4(config, [0, 1, 2, 3], runtime / "judge-full")
    try:
        for candidate in pending:
            output = aggregate / candidate["key"] / "evaluation.json"
            if output.is_file(): continue
            repeated = [run_judge(candidate, participants[candidate["key"]]["validation"], "validation", 400, repeat, endpoints, attest) for repeat in range(1, 11)]
            summary = aggregate_deterministic_repeats(repeated)
            payload = evaluation_payload(candidate, config["judge"], dependencies["score_files"]["validation_sha256"], file_sha256(participants[candidate["key"]]["rationale"]), file_sha256(participants[candidate["key"]]["validation"]), participants[candidate["key"]]["generation_reports"], summary)
            atomic_json(output, payload)
    finally: stop_processes(q4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/official_rationale_handoff.v1.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(); config = HandoffConfig.from_json(args.config)
    plan = {"candidate_count": len(config.candidates), "candidate_policies": {candidate["key"]: {"origin_classification": candidate["origin_classification"], "historical_method": candidate["historical_method"], "historical_source_sha256": candidate["historical_source_sha256"], "final_winner_eligible": candidate["final_winner_eligible"], "ranking_caveat": candidate["ranking_caveat"]} for candidate in config.candidates}, "compatible_base_groups": [[item["key"] for item in group] for group in compatible_groups(config.candidates)], "gpu0_real_row_smoke": True, "full_generation_gpu_scope": [0, 1, 2, 3], "judge_replicas": 4, "validation_records_per_candidate": 400, "deterministic_repeats_per_record": 10, "independent_samples": False, "temperature": 0.0, "seed": 42, "gpu_started": False}
    if args.dry_run: print(json.dumps(plan, sort_keys=True)); return
    dependencies = validate_candidate_artifacts(config)
    if args.validate_only: print(json.dumps({**plan, "status": "validated"}, sort_keys=True)); return
    execute(config, dependencies)


if __name__ == "__main__":
    main()
