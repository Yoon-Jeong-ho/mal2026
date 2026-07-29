#!/usr/bin/env python3
"""Smoke then TP4 generation for one evaluation.txt rationale prompt arm."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import sha256_file  # noqa: E402
from mal2026.evaluation_prompt_matrix import (  # noqa: E402
    RATIONALE_KINDS,
    RATIONALE_SCORE_BLIND,
    RATIONALE_SCORE_CONDITIONED,
    prompt_provenance,
    rationale_schema,
)
from mal2026.evaluation_prompt_rationale_sft import MODEL_PATH  # noqa: E402
from mal2026.official_rl_servers import assert_gpus_idle, vllm_policy_server  # noqa: E402
from mal2026.official_writing_contract import AXES  # noqa: E402


CLIENT = ROOT / "scripts/generate_evaluation_prompt_rationales_vllm.py"
RESTRICTED_BASE = ROOT / "data/processed/restricted/evaluation_prompt_rationale_v2"
OUTPUT_BASE = ROOT / "outputs/evaluation-prompt-rationale-generation-v2"
ADAPTER_ROOT = ROOT / "outputs/evaluation-prompt-rationale-sft-v2"
ALIASES = {
    RATIONALE_SCORE_BLIND: "mal2026-evaluation-rationale-score-blind",
    RATIONALE_SCORE_CONDITIONED: "mal2026-evaluation-rationale-score-conditioned",
}
RUNS = {
    RATIONALE_SCORE_BLIND: "evaluation-prompt-rationale-generation-v2-score-blind-20260729-004",
    RATIONALE_SCORE_CONDITIONED: "evaluation-prompt-rationale-generation-v2-score-conditioned-20260729-002",
}
SFT_RUNS = {
    RATIONALE_SCORE_BLIND: "evaluation-prompt-rationale-sft-v2-ax4-score-blind-20260729-001",
    RATIONALE_SCORE_CONDITIONED: "evaluation-prompt-rationale-sft-v2-ax4-score-conditioned-20260729-001",
}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def attestation(path: Path, endpoint: str, alias: str, prompt_kind: str, base: Path, adapter: Path) -> Path:
    need(not path.exists(), "generation attestation must be fresh")
    atomic_json(path, {
        "schema_version": "mal2026-evaluation-prompt-rationale-server-attestation-v1",
        "created_at": now(), "endpoint": endpoint, "model_alias": alias,
        "prompt_kind": prompt_kind, "base_server_attestation_sha256": sha256_file(base),
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
    })
    return path


def command(
    *, run_id: str, prompt_kind: str, split: str, expected: int, score_file: Path | None,
    output: Path, endpoint: str, alias: str, server_attestation: Path, max_inflight: int,
) -> list[str]:
    value = [
        str(Path(sys.executable).resolve()), str(CLIENT), "--run-id", run_id,
        "--prompt-kind", prompt_kind, "--split", split, "--expected", str(expected),
        "--output-dir", str(output), "--endpoint", endpoint, "--model", alias,
        "--server-attestation", str(server_attestation), "--max-inflight", str(max_inflight),
    ]
    if score_file is not None:
        value.extend(["--score-file", str(score_file)])
    return value


def run_command(value: Sequence[str], log: Path) -> None:
    need(not log.exists(), "generation log must be fresh")
    log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(list(value), cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, stdout=handle, stderr=subprocess.STDOUT, text=True)
    need(completed.returncode == 0, f"rationale generation failed: {log}")


def wait_release(gpus: Sequence[int], seconds: int = 120) -> None:
    deadline = time.monotonic() + seconds
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            assert_gpus_idle(gpus)
            return
        except Exception as exc:
            last = exc
            time.sleep(1)
    raise RuntimeError(f"owned GPU contexts did not release: {last}")


def one_score(source: Path, output: Path) -> Path:
    need(source.is_file() and not output.exists(), "smoke score request differs")
    with source.open(encoding="utf-8") as handle:
        line = handle.readline()
    need(bool(line.strip()), "score source is empty")
    output.write_text(line, encoding="utf-8")
    return output


def normalize(source: Path, output: Path, expected: int) -> str:
    need(source.is_file() and not output.exists(), "rationale normalization request differs")
    seen: set[str] = set()
    with source.open(encoding="utf-8") as input_handle, output.open("x", encoding="utf-8") as output_handle:
        for line in input_handle:
            raw = json.loads(line)
            source_id, rationales = raw.get("source_id"), raw.get("rationales")
            need(isinstance(source_id, str) and source_id not in seen, "rationale source ID differs")
            need(isinstance(rationales, dict) and set(rationales) == set(AXES), "rationale axes differ")
            need(all(
                isinstance(rationales[axis], str)
                and 60 <= len(rationales[axis].strip()) <= 420
                and rationales[axis].rstrip().endswith((".", "?", "!"))
                for axis in AXES
            ), "rationale length or sentence completion gate differs")
            seen.add(source_id)
            output_handle.write(json.dumps({"source_id": source_id, "rationales": {axis: rationales[axis].strip() for axis in AXES}}, ensure_ascii=False, separators=(",", ":")) + "\n")
    need(len(seen) == expected, "rationale population differs")
    return sha256_file(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-kind", choices=RATIONALE_KINDS, required=True)
    parser.add_argument("--score-train", type=Path)
    parser.add_argument("--score-validation", type=Path)
    parser.add_argument("--score-source-result", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_kind = args.prompt_kind
    conditioned = prompt_kind == RATIONALE_SCORE_CONDITIONED
    if conditioned:
        need(args.score_train is not None and args.score_validation is not None and args.score_source_result is not None, "conditioned arm lacks scores or their result attestation")
    else:
        need(args.score_train is None and args.score_validation is None and args.score_source_result is None, "blind arm received score lineage")
    score_source: dict[str, Any] | None = None
    if conditioned:
        assert args.score_train is not None and args.score_validation is not None and args.score_source_result is not None
        need(args.score_source_result.is_file() and not args.score_source_result.is_symlink(), "score source result is unavailable")
        raw_source = json.loads(args.score_source_result.read_text(encoding="utf-8"))
        need(isinstance(raw_source, dict) and raw_source.get("status") == "completed" and raw_source.get("mode") == "full", "score source result is incomplete")
        need(raw_source.get("input_kind") == "direct" and raw_source.get("average_read") is False and raw_source.get("average_target_used") is False, "conditioned rationale scores are not from the direct three-axis arm")
        predictions = raw_source.get("prediction_outputs")
        need(isinstance(predictions, dict) and predictions.get("human_or_reference_score_prompted") is False, "score source prediction lineage differs")
        need(Path(predictions.get("train_path", "")).resolve() == args.score_train.resolve() and Path(predictions.get("validation_path", "")).resolve() == args.score_validation.resolve(), "score source paths differ")
        need(predictions.get("train_sha256") == sha256_file(args.score_train) and predictions.get("validation_sha256") == sha256_file(args.score_validation), "score source hashes differ")
        score_source = {
            "run_id": raw_source.get("run_id"), "model_key": raw_source.get("model_key"),
            "result_sha256": sha256_file(args.score_source_result),
        }
        need(all(isinstance(value, str) and value for value in score_source.values()), "score source identity differs")
    run_id, alias = RUNS[prompt_kind], ALIASES[prompt_kind]
    adapter = ADAPTER_ROOT / SFT_RUNS[prompt_kind] / "adapter"
    completion = adapter.parent / "training_complete.json"
    need(adapter.is_dir() and completion.is_file() and not adapter.is_symlink(), "completed rationale SFT adapter is unavailable")
    completion_value = json.loads(completion.read_text(encoding="utf-8"))
    need(completion_value.get("status") == "completed" and completion_value.get("prompt_kind") == prompt_kind and completion_value.get("mode") == "full", "rationale SFT completion differs")
    restricted, output = RESTRICTED_BASE / run_id, OUTPUT_BASE / run_id
    need(not restricted.exists() and not output.exists(), "rationale generation outputs must be fresh")
    restricted.mkdir(mode=0o700, parents=True)
    output.mkdir(mode=0o700, parents=True)
    (output / "logs").mkdir(mode=0o700)
    (output / "attestations").mkdir(mode=0o700)
    manifest = {
        "schema_version": "mal2026-evaluation-prompt-rationale-handoff-run-v1",
        "status": "running", "run_id": run_id, "created_at": now(),
        "prompt_kind": prompt_kind, "score_conditioning": conditioned,
        "gpu_scope": [0, 1, 2, 3], "smoke_gpu_scope": [0], "tensor_parallel_size": 4,
        "sft_completion_sha256": sha256_file(completion),
        "score_source": score_source,
        "human_or_reference_score_read_or_prompted": False,
        "rationale_schema_sha256": sha256(json.dumps(rationale_schema(), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        **prompt_provenance(prompt_kind),
    }
    atomic_json(output / "manifest.json", manifest)
    scores = {"train": args.score_train, "validation": args.score_validation}
    smoke_score = one_score(args.score_train, restricted / "scores.train-smoke.jsonl") if conditioned and args.score_train is not None else None

    assert_gpus_idle((0,))
    with vllm_policy_server(
        runtime_root=output, label="evaluation-rationale-smoke", gpus=(0,), port=19350,
        adapters={"bundle": adapter}, aliases={"bundle": alias}, max_num_seqs=16,
        max_num_batched_tokens=8192, dynamic_updates=False, max_model_len=4096, data_split="train",
    ) as (endpoint, base_attestation):
        bound = attestation(output / "attestations/smoke.json", endpoint, alias, prompt_kind, base_attestation, adapter)
        run_command(command(
            run_id=f"{run_id}-smoke", prompt_kind=prompt_kind, split="train", expected=1,
            score_file=smoke_score, output=restricted / "raw-smoke", endpoint=endpoint,
            alias=alias, server_attestation=bound, max_inflight=1,
        ), output / "logs/smoke.log")
    wait_release((0,))

    assert_gpus_idle((0, 1, 2, 3))
    raw_dirs = {split: restricted / "raw" / split for split in ("train", "validation")}
    with vllm_policy_server(
        runtime_root=output, label="evaluation-rationale-full", gpus=(0, 1, 2, 3), port=19351,
        adapters={"bundle": adapter}, aliases={"bundle": alias}, max_num_seqs=256,
        max_num_batched_tokens=32768, dynamic_updates=False, max_model_len=4096,
        data_split="train_and_validation",
    ) as (endpoint, base_attestation):
        bounds = {split: attestation(output / f"attestations/full-{split}.json", endpoint, alias, prompt_kind, base_attestation, adapter) for split in ("train", "validation")}
        processes: list[tuple[str, subprocess.Popen[str], Any, Path]] = []
        for split, expected in (("train", 2000), ("validation", 400)):
            log = output / "logs" / f"full-{split}.log"
            handle = log.open("x", encoding="utf-8")
            process = subprocess.Popen(
                command(run_id=f"{run_id}-{split}", prompt_kind=prompt_kind, split=split, expected=expected,
                        score_file=scores[split], output=raw_dirs[split], endpoint=endpoint, alias=alias,
                        server_attestation=bounds[split], max_inflight=128),
                cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
            processes.append((split, process, handle, log))
        failures: dict[str, int] = {}
        for split, process, handle, _ in processes:
            code = process.wait(); handle.close()
            if code != 0:
                failures[split] = code
        need(not failures, f"full rationale generation failed: {failures}")

    raw_reports = {
        split: json.loads((raw_dirs[split] / "aggregate_generation_report.json").read_text(encoding="utf-8"))
        for split in ("train", "validation")
    }
    need(all(
        report.get("status") == "completed"
        and report.get("counts", {}).get("expected") == report.get("counts", {}).get("valid")
        and not report.get("failure_categories")
        for report in raw_reports.values()
    ), "rationale raw generation gates differ")
    retry_policy_shas = {report.get("quality_retry_policy_sha256") for report in raw_reports.values()}
    need(len(retry_policy_shas) == 1 and all(isinstance(value, str) and len(value) == 64 for value in retry_policy_shas), "rationale retry policy differs")
    retry_policy_sha = next(iter(retry_policy_shas))
    retry_counts = {split: int(raw_reports[split]["records_requiring_retry"]) for split in raw_reports}
    axis_fallback_counts = {split: int(raw_reports[split]["records_requiring_axis_fallback"]) for split in raw_reports}

    rationale_paths = {split: restricted / f"rationales.{split}.jsonl" for split in ("train", "validation")}
    rationale_hashes = {
        "train": normalize(raw_dirs["train"] / "generated_rationales.jsonl", rationale_paths["train"], 2000),
        "validation": normalize(raw_dirs["validation"] / "generated_rationales.jsonl", rationale_paths["validation"], 400),
    }
    rationale_key = f"ax4_{'score_conditioned' if conditioned else 'score_blind'}_evaluation_txt_sft_v2"
    handoff = {
        "schema_version": "mal2026-evaluation-prompt-rationale-handoff-v1",
        "status": "completed", "run_id": run_id, "rationale_key": rationale_key,
        "structure": "bundle", "prompt_kind": prompt_kind,
        "score_conditioning": conditioned,
        "score_train_sha256": None if args.score_train is None else sha256_file(args.score_train),
        "score_validation_sha256": None if args.score_validation is None else sha256_file(args.score_validation),
        "score_kind": None if not conditioned else "score_encoder_actual_emitted_integer_prediction",
        "score_source": score_source,
        "rationale_train_sha256": rationale_hashes["train"],
        "rationale_validation_sha256": rationale_hashes["validation"],
        "sft_completion_sha256": sha256_file(completion),
        "human_or_reference_score_read_or_prompted": False,
        "rationale_schema_sha256": manifest["rationale_schema_sha256"],
        "quality_retry_policy_sha256": retry_policy_sha,
        "records_requiring_retry": retry_counts,
        "records_requiring_axis_fallback": axis_fallback_counts,
        **prompt_provenance(prompt_kind),
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_or_predictions",
    }
    handoff_path = restricted / "aggregate_handoff_manifest.json"
    atomic_json(handoff_path, handoff)
    manifest.update({
        "status": "completed", "completed_at": now(), "rationale_key": rationale_key,
        "records_requiring_retry": retry_counts,
        "records_requiring_axis_fallback": axis_fallback_counts,
        "quality_retry_policy_sha256": retry_policy_sha,
        "rationale_train_path": str(rationale_paths["train"].resolve()), "rationale_train_sha256": rationale_hashes["train"],
        "rationale_validation_path": str(rationale_paths["validation"].resolve()), "rationale_validation_sha256": rationale_hashes["validation"],
        "handoff_manifest_path": str(handoff_path.resolve()), "handoff_manifest_sha256": sha256_file(handoff_path),
    })
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "completed", "run_id": run_id, "rationale_key": rationale_key}, sort_keys=True))


if __name__ == "__main__":
    main()
