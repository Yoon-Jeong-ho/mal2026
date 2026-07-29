#!/usr/bin/env python3
"""Generate the bundle-DPO train/validation rationale handoff for encoders."""
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
from mal2026.official_rationale_handoff import combine_rationales  # noqa: E402
from mal2026.official_rationale_rl import MODEL_ID, MODEL_REVISION, MODEL_PATH  # noqa: E402
from mal2026.official_rl_servers import (  # noqa: E402
    PYTHON,
    assert_gpus_idle,
    vllm_policy_server,
)


RUN_ID = "official-rationale-dpo-selected-handoff-exact-bundle-20260729-021"
BOOTSTRAP = ROOT / "outputs/official-score-matrix-v1/bootstrap_selection.json"
SELECTION = ROOT / (
    "outputs/official-rationale-rl-v1/evaluation/"
    "official-rationale-dpo-bundle-validation-exact-judge-20260729-020/"
    "aggregate_bundle_dpo_validation_comparison.json"
)
DPO_ADAPTER = ROOT / (
    "outputs/official-rationale-rl-v1/orchestration/"
    "official-rationale-rl-experiment-v1-exact-judge-20260729-019/models/"
    "dpo-official-bundle-ddp4-full-user-aligned-001/adapter"
)
DPO_COMPLETION = DPO_ADAPTER.parent / "training_complete.json"
GENERATOR = ROOT / "scripts/generate_official_rationales_vllm.py"
RESTRICTED_BASE = ROOT / "data/processed/restricted/official_prompt_alignment_v1/final_rationale_handoff"
OUTPUT_BASE = ROOT / "outputs/official-rationale-handoff-v1"
ALIAS = "mal2026-selected-dpo-bundle"
AXES = ("content", "organization", "expression")


class HandoffError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise HandoffError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def canonical_sha(value: Any) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def convert_scores(source: Path, destination: Path, split: str, expected: int) -> str:
    need(not destination.exists(), "adapted score output must be fresh")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    seen: set[str] = set()
    with source.open(encoding="utf-8") as input_handle, destination.open("x", encoding="utf-8") as output_handle:
        for line in input_handle:
            raw = json.loads(line)
            need(
                isinstance(raw, dict)
                and set(raw) == {"source_id", "split", "arm", "scores"}
                and raw["split"] == split,
                "bootstrap score row differs",
            )
            source_id, scores = raw["source_id"], raw["scores"]
            need(isinstance(source_id, str) and source_id not in seen, "bootstrap score ID differs")
            need(
                isinstance(scores, dict) and set(scores) == set(AXES)
                and all(type(scores[axis]) is int and 1 <= scores[axis] <= 5 for axis in AXES),
                "bootstrap emitted score differs",
            )
            seen.add(source_id)
            output_handle.write(json.dumps(
                {"source_id": source_id, "emitted_integer_prediction": {axis: scores[axis] for axis in AXES}},
                ensure_ascii=False, separators=(",", ":"),
            ) + "\n")
    need(len(seen) == expected, "bootstrap score population differs")
    return sha256_file(destination)


def generator_attestation(destination: Path, endpoint: str, policy_attestation: Path, split: str) -> Path:
    need(not destination.exists(), "generator attestation must be fresh")
    atomic_json(destination, {
        "schema_version": "mal2026-official-rationale-vllm-server-attestation-v1",
        "created_at": now(),
        "endpoint": endpoint,
        "adapter_aliases": {ALIAS: "bundle"},
        "base_server_attestation_sha256": sha256_file(policy_attestation),
        "data_split": split,
        "train_split_only": split == "train",
    })
    return destination


def generation_command(
    *, run_id: str, split: str, expected: int, score_file: Path,
    output_dir: Path, endpoint: str, attestation: Path, max_inflight: int,
) -> list[str]:
    return [
        str(PYTHON), str(GENERATOR), "--run-id", run_id,
        "--task", "bundle", "--split", split, "--expected", str(expected),
        "--score-file", str(score_file), "--output-dir", str(output_dir),
        "--endpoint", endpoint, "--model", ALIAS,
        "--server-attestation", str(attestation), "--max-inflight", str(max_inflight),
    ]


def run_command(command: Sequence[str], log: Path) -> None:
    need(not log.exists(), f"log must be fresh: {log}")
    log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command), cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
    need(completed.returncode == 0, f"generation failed: {log}")


def wait_gpu_release(gpus: Sequence[int], seconds: int = 120) -> None:
    deadline = time.monotonic() + seconds
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            assert_gpus_idle(gpus)
            return
        except Exception as exc:
            last = exc
            time.sleep(1)
    raise HandoffError(f"owned GPU contexts did not release: {last}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    run_id = args.run_id
    need(run_id == RUN_ID, "selected handoff run identity differs")

    restricted = RESTRICTED_BASE / run_id
    output = OUTPUT_BASE / run_id
    need(not restricted.exists() and not output.exists(), "handoff outputs must be fresh")
    required = (
        BOOTSTRAP, SELECTION, DPO_COMPLETION, DPO_ADAPTER / "adapter_config.json",
        DPO_ADAPTER / "adapter_model.safetensors", MODEL_PATH / "config.json", GENERATOR,
    )
    need(all(path.is_file() and not path.is_symlink() for path in required), "selected handoff prerequisite is unavailable")

    bootstrap = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    completion = json.loads(DPO_COMPLETION.read_text(encoding="utf-8"))
    need(
        bootstrap.get("status") == "stage_a_completed"
        and bootstrap.get("selection_source") == "train_internal_dev_only"
        and bootstrap.get("canonical_validation_used_for_selection") is False,
        "bootstrap selection lineage differs",
    )
    need(
        selection.get("status") == "completed"
        and selection.get("contract") == "single_bundled_three_axis_participant_json"
        and selection.get("axis_triplet_used_for_training_or_selection") is False
        and selection.get("selection", {}).get("selected") == "dpo",
        "bundle-DPO selection differs",
    )
    need(
        completion.get("status") == "completed"
        and completion.get("task") == "bundle"
        and completion.get("split") == "train"
        and completion.get("validation_used_for_preferences_or_training") is False,
        "DPO completion lineage differs",
    )
    selected_scores = bootstrap.get("selected_score_files")
    need(isinstance(selected_scores, dict), "selected bootstrap scores are unavailable")
    for split, expected in (("train", 2000), ("validation", 400)):
        source = Path(str(selected_scores.get(f"{split}_path", "")))
        need(
            source.is_file() and source.resolve().is_relative_to((ROOT / "data/processed/restricted").resolve())
            and sha256_file(source) == selected_scores.get(f"{split}_sha256")
            and selected_scores.get(f"{split}_records") == expected,
            f"selected {split} score binding differs",
        )

    restricted.mkdir(mode=0o700, parents=True)
    output.mkdir(mode=0o700, parents=True)
    for name in ("logs", "attestations"):
        (output / name).mkdir(mode=0o700)
    manifest_path = output / "manifest.json"
    manifest = {
        "schema_version": "mal2026-selected-dpo-rationale-handoff-run-v1",
        "status": "running", "run_id": run_id, "created_at": now(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *sys.argv[1:]],
        "gpu_scope": [0, 1, 2, 3],
        "gpu_authorization": "repository default GPUs0-3 and explicit user request; GPUs4-7 not queried or used",
        "structure": "bundle", "axis_triplet_used": False,
        "bootstrap_selection_sha256": sha256_file(BOOTSTRAP),
        "winner_selection_sha256": sha256_file(SELECTION),
        "dpo_training_completion_sha256": sha256_file(DPO_COMPLETION),
        "human_or_reference_score_read_or_prompted": False,
    }
    atomic_json(manifest_path, manifest)

    adapted: dict[str, Path] = {}
    adapted_sha: dict[str, str] = {}
    for split, expected in (("train", 2000), ("validation", 400)):
        adapted[split] = restricted / f"bootstrap-scores.{split}.jsonl"
        adapted_sha[split] = convert_scores(
            Path(selected_scores[f"{split}_path"]), adapted[split], split, expected
        )
    smoke_scores = restricted / "bootstrap-scores.train-smoke.jsonl"
    with adapted["train"].open(encoding="utf-8") as source:
        smoke_scores.write_text(source.readline(), encoding="utf-8")

    # Required smallest real preflight: one train row on physical GPU0.
    assert_gpus_idle((0,))
    with vllm_policy_server(
        runtime_root=output, label="selected-dpo-smoke", gpus=(0,), port=19340,
        adapters={"bundle": DPO_ADAPTER}, aliases={"bundle": ALIAS},
        max_num_seqs=16, max_num_batched_tokens=8192, dynamic_updates=False,
        max_model_len=4096, data_split="train",
    ) as (endpoint, policy_attestation):
        attestation = generator_attestation(
            output / "attestations/generator-smoke.json", endpoint, policy_attestation, "train"
        )
        smoke_dir = restricted / "raw-smoke"
        run_command(generation_command(
            run_id=f"{run_id}-smoke", split="train", expected=1,
            score_file=smoke_scores, output_dir=smoke_dir, endpoint=endpoint,
            attestation=attestation, max_inflight=1,
        ), output / "logs/generation-smoke.log")
        smoke_report = json.loads((smoke_dir / "aggregate_generation_report.json").read_text(encoding="utf-8"))
        need(smoke_report.get("status") == "completed", "selected DPO generation smoke failed")
    wait_gpu_release((0,))

    # Full generation uses one TP4 server; train and validation clients feed it
    # concurrently so all 256 scheduler slots can remain occupied.
    assert_gpus_idle((0, 1, 2, 3))
    raw_dirs = {
        "train": restricted / "raw" / "train" / "bundle",
        "validation": restricted / "raw" / "validation" / "bundle",
    }
    with vllm_policy_server(
        runtime_root=output, label="selected-dpo-full", gpus=(0, 1, 2, 3), port=19341,
        adapters={"bundle": DPO_ADAPTER}, aliases={"bundle": ALIAS},
        max_num_seqs=256, max_num_batched_tokens=32768, dynamic_updates=False,
        max_model_len=4096, data_split="train_and_validation",
    ) as (endpoint, policy_attestation):
        attestations = {
            split: generator_attestation(
                output / f"attestations/generator-full-{split}.json",
                endpoint, policy_attestation, split,
            )
            for split in ("train", "validation")
        }
        processes: list[tuple[str, subprocess.Popen[str], Any, Path]] = []
        for split, expected in (("train", 2000), ("validation", 400)):
            log = output / "logs" / f"generation-full-{split}.log"
            handle = log.open("x", encoding="utf-8")
            process = subprocess.Popen(
                generation_command(
                    run_id=f"{run_id}-{split}", split=split, expected=expected,
                    score_file=adapted[split], output_dir=raw_dirs[split], endpoint=endpoint,
                    attestation=attestations[split], max_inflight=128,
                ),
                cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
            processes.append((split, process, handle, log))
        failures: dict[str, int] = {}
        for split, process, handle, _ in processes:
            code = process.wait(); handle.close()
            if code != 0:
                failures[split] = code
        need(not failures, f"full rationale generation failed: {failures}")

    rationale_paths: dict[str, Path] = {}
    rationale_sha: dict[str, str] = {}
    for split, expected in (("train", 2000), ("validation", 400)):
        report = json.loads((raw_dirs[split] / "aggregate_generation_report.json").read_text(encoding="utf-8"))
        need(report.get("status") == "completed", f"{split} generation hard gates failed")
        rationale_paths[split] = restricted / f"rationales.{split}.jsonl"
        rationale_sha[split] = combine_rationales(
            {"bundle": raw_dirs[split] / "generated_rationales.jsonl"},
            rationale_paths[split], expected, "bundle",
        )

    adapter_bindings = {
        "bundle": {
            "adapter_config_sha256": sha256_file(DPO_ADAPTER / "adapter_config.json"),
            "adapter_model_sha256": sha256_file(DPO_ADAPTER / "adapter_model.safetensors"),
            "training_completion_sha256": sha256_file(DPO_COMPLETION),
        }
    }
    candidate_identity = {
        "key": "official_bundle_dpo_exact_judge_20260729_019",
        "method": "dpo", "structure": "bundle",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "adapter_bindings": adapter_bindings,
        "selection_sha256": sha256_file(SELECTION),
    }
    handoff = {
        "schema_version": "mal2026-official-rationale-score-matrix-handoff-v1",
        "status": "completed",
        "rationale_key": candidate_identity["key"],
        "bootstrap_selection_sha256": sha256_file(BOOTSTRAP),
        "bootstrap_selected_result_sha256": bootstrap["selected_result_sha256"],
        "score_train_sha256": selected_scores["train_sha256"],
        "score_validation_sha256": selected_scores["validation_sha256"],
        "adapted_score_train_sha256": adapted_sha["train"],
        "adapted_score_validation_sha256": adapted_sha["validation"],
        "rationale_train_sha256": rationale_sha["train"],
        "rationale_validation_sha256": rationale_sha["validation"],
        "winner_selection_sha256": sha256_file(SELECTION),
        "winner_candidate_identity_sha256": canonical_sha(candidate_identity),
        "winner_evaluation_sha256": sha256_file(SELECTION),
        "structure": "bundle",
        "axis_triplet_used_for_training_or_selection": False,
        "model_config_sha256": candidate_identity["model_config_sha256"],
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_binding_sha256": candidate_identity["model_config_sha256"],
        "adapter_bindings": adapter_bindings,
        "judge_contract_sha256": sha256_file(ROOT / "src/mal2026/official_writing_contract.py"),
        "judge_model_sha256": selection["input_sha256"].get("judge_model", "b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772"),
        "judge_prompt_sha256": selection["judge_prompt_sha256"],
        "directional_gate_sha256": completion["contrastive_gate_sha256"],
        "injection_gate_sha256": completion["rl_safety_gate_sha256"],
        "score_kind": "bootstrap_model_actual_emitted_integer_prediction",
        "human_or_reference_score_read_or_prompted": False,
        "generation": {"temperature": 0.0, "top_p": 1.0, "seed": 42},
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_or_predictions",
    }
    # The comparison report currently records the judge digest at top level;
    # bind the immutable model digest from the DPO completion as the authority.
    handoff["judge_model_sha256"] = completion["judge_model_sha256"]
    handoff_path = restricted / "aggregate_handoff_manifest.json"
    atomic_json(handoff_path, handoff)
    manifest.update({
        "status": "completed", "completed_at": now(),
        "rationale_train_path": str(rationale_paths["train"].resolve()),
        "rationale_train_sha256": rationale_sha["train"],
        "rationale_validation_path": str(rationale_paths["validation"].resolve()),
        "rationale_validation_sha256": rationale_sha["validation"],
        "handoff_manifest_path": str(handoff_path.resolve()),
        "handoff_manifest_sha256": sha256_file(handoff_path),
    })
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "status": "completed", "run_id": run_id,
        "records": {"train": 2000, "validation": 400},
        "rationale_sha256": rationale_sha,
        "handoff_manifest_sha256": sha256_file(handoff_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
