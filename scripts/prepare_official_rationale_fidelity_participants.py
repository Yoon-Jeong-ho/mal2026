#!/usr/bin/env python3
"""Prepare restricted target-score/rationale participants for exact Q4 judging.

This is a retrospective rationale-fidelity audit.  Historical rationale-only
generators did not emit a score, so their rationales are paired with the
canonical human/reference score (integerized with the public output rule).
That track must not be confused with the deployment-like emitted-score judge.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
os.sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import (  # noqa: E402
    CANDIDATE_PATH,
    SOURCE_SHA256,
    load_candidates,
    load_writing_rows,
    sha256_file,
)
from mal2026.official_writing_contract import AXES, integerize_scores, parse_participant_output  # noqa: E402


CAMPAIGN = "official-rationale-fidelity-q4-v1-20260806-001"
RESTRICTED_PARENT = ROOT / "data/processed/restricted/official_rationale_fidelity_v1"
AGGREGATE_PARENT = ROOT / "outputs/official-rationale-fidelity-v1"

INITIAL_SFT_RUNS = (
    "api-rationale-generation-v1-ax4_light-axis_triplet-validation-002",
    "api-rationale-generation-v1-ax4_light-bundle-validation-003",
    "api-rationale-generation-v1-midm2_base-axis_triplet-validation-002",
    "api-rationale-generation-v1-midm2_base-bundle-validation-003",
    "api-rationale-generation-v1-phi4_mini-axis_triplet-validation-002",
    "api-rationale-generation-v1-phi4_mini-bundle-validation-003",
)
RLAIF_V7_RUNS = (
    "rlaif-grpo-prompt-ensemble-v7-ax4_light-bundle-all5-validation-001",
    "rlaif-grpo-prompt-ensemble-v7-ax4_light-bundle-random1-validation-001",
    "rlaif-grpo-prompt-ensemble-v7-midm2_base-bundle-all5-validation-001",
    "rlaif-grpo-prompt-ensemble-v7-midm2_base-bundle-random1-validation-001",
)
RLAIF_V8_RUNS = tuple(
    f"rlaif-grpo-prompt-ensemble-v8-{base}-{task}-{arm}-validation-001"
    for base in ("ax4_light", "midm2_base", "phi4_mini")
    for task in ("bundle", "content", "organization", "expression")
    for arm in ("all5", "random1")
)
TOP3_RUNS = (
    "rlaif-top3-rationale-generation-v1-rank1_midm2_random1-validation-full-001",
    "rlaif-top3-rationale-generation-v1-rank2_ax4_random1-validation-full-001",
    "rlaif-top3-rationale-generation-v1-rank3_ax4_all5-validation-full-001",
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def key_for(run_id: str) -> str:
    return run_id.removeprefix("api-rationale-generation-v1-").removeprefix("rlaif-grpo-prompt-ensemble-").removeprefix("rlaif-top3-rationale-generation-v1-")


def rationale_paths() -> list[tuple[str, str, Path]]:
    api = ROOT / "data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001"
    top3 = ROOT / "data/processed/restricted/rlaif_top3_encoder_v1/rationales"
    result: list[tuple[str, str, Path]] = []
    for run_id in INITIAL_SFT_RUNS:
        result.append(("initial_decoder_sft", key_for(run_id), api / "decoder_generation_v1" / run_id / "generated_rationales.jsonl"))
    for run_id in RLAIF_V7_RUNS:
        result.append(("legacy_rlaif_v7", key_for(run_id), api / "rlaif_grpo_v7" / run_id / "generated_rationales.jsonl"))
    for run_id in RLAIF_V8_RUNS:
        result.append(("legacy_rlaif_v8", key_for(run_id), api / "rlaif_grpo_v8" / run_id / "generated_rationales.jsonl"))
    for run_id in TOP3_RUNS:
        result.append(("top3_regeneration", key_for(run_id), top3 / run_id / "generated_rationales.jsonl"))
    return result


def text_rationales(raw: Mapping[str, Any]) -> dict[str, str]:
    value = raw.get("rationale", raw.get("rationales"))
    need(isinstance(value, Mapping), "historical rationale object differs")
    result: dict[str, str] = {}
    for axis in AXES:
        part = value.get(axis)
        if isinstance(part, str):
            rationale = part
        else:
            need(isinstance(part, Mapping), f"historical {axis} rationale shape differs")
            rationale = part.get("rationale")
        need(isinstance(rationale, str) and bool(rationale.strip()), f"historical {axis} rationale is blank")
        result[axis] = rationale.strip()
    return result


def read_generated(path: Path) -> dict[str, dict[str, str]]:
    need(path.is_file() and not path.is_symlink(), f"historical rationale input unavailable: {path}")
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            need(bool(line.strip()), "historical rationale file contains a blank row")
            raw = json.loads(line)
            source_id = raw.get("source_id")
            need(isinstance(source_id, str) and source_id not in rows, "historical source ID differs")
            rows[source_id] = text_rationales(raw)
    need(len(rows) == 400, f"historical rationale population differs: {path}")
    return rows


def participant(scores: Mapping[str, float], rationales: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    integer = integerize_scores(scores)
    value = {axis: {"score": integer[axis], "rationale": rationales[axis]} for axis in AXES}
    return parse_participant_output(value)


def write_participants(path: Path, rows: list[tuple[str, dict[str, dict[str, Any]]]]) -> str:
    seen: set[str] = set()
    with path.open("x", encoding="utf-8") as handle:
        for source_id, output in rows:
            need(source_id not in seen, "participant source ID is duplicated")
            seen.add(source_id)
            handle.write(json.dumps({"source_id": source_id, "participant_output": output}, ensure_ascii=False, separators=(",", ":")) + "\n")
    return sha256_file(path)


def source_inventory() -> list[tuple[str, str, Path]]:
    paths = rationale_paths()
    need(len(paths) == 37 and len({key for _, key, _ in paths}) == 37, "historical source inventory differs")
    return paths


def check_inputs() -> dict[str, Any]:
    paths = source_inventory()
    for _, _, path in paths:
        read_generated(path)
    validation_candidates = load_candidates("validation")
    need(len(validation_candidates) == 1200, "OpenAI validation candidate population differs")
    return {
        "campaign": CAMPAIGN,
        "historical_rationale_sources": len(paths),
        "openai_candidate_sources": 3,
        "total_full_participants": len(paths) + 3,
        "rows_per_full_participant": 400,
        "total_full_judge_requests": (len(paths) + 3) * 400,
        "source_sha256": SOURCE_SHA256,
        "openai_candidate_sha256": {split: sha256_file(path) for split, path in CANDIDATE_PATH.items()},
    }


def prepare() -> dict[str, Any]:
    check = check_inputs()
    destination = RESTRICTED_PARENT / CAMPAIGN
    aggregate_destination = AGGREGATE_PARENT / CAMPAIGN
    need(not destination.exists() and not aggregate_destination.exists(), "fidelity participant output must be fresh")
    participants_dir = destination / "participants"
    participants_dir.mkdir(mode=0o700, parents=True)
    aggregate_destination.mkdir(parents=True)

    validation = load_writing_rows("validation", include_scores=True)
    by_id = {row.identifier: row for row in validation}
    need(all(row.scores is not None for row in validation), "validation reference scores unavailable")
    inventory: list[dict[str, Any]] = []

    candidates = load_candidates("validation", writings=validation)
    for candidate_number in (1, 2, 3):
        selected = [row for row in candidates if row.candidate_number == candidate_number]
        need(len(selected) == 400, "OpenAI candidate-number population differs")
        rows = [
            (row.source_id, participant(by_id[row.source_id].scores or {}, row.diagnoses))
            for row in selected
        ]
        key = f"openai_terra_candidate_{candidate_number}"
        path = participants_dir / f"{key}.validation.jsonl"
        digest = write_participants(path, rows)
        inventory.append({
            "key": key,
            "family": "openai_terra_initial",
            "participant_file": str(path.resolve()),
            "participant_sha256": digest,
            "rationale_source_file": str(CANDIDATE_PATH["validation"].resolve()),
            "rationale_source_sha256": sha256_file(CANDIDATE_PATH["validation"]),
            "records": 400,
        })

    for family, key, rationale_path in source_inventory():
        rationales = read_generated(rationale_path)
        need(set(rationales) == set(by_id), f"historical rationale IDs differ: {key}")
        rows = [
            (row.identifier, participant(row.scores or {}, rationales[row.identifier]))
            for row in validation
        ]
        path = participants_dir / f"{key}.validation.jsonl"
        digest = write_participants(path, rows)
        inventory.append({
            "key": key,
            "family": family,
            "participant_file": str(path.resolve()),
            "participant_sha256": digest,
            "rationale_source_file": str(rationale_path.resolve()),
            "rationale_source_sha256": sha256_file(rationale_path),
            "records": 400,
        })

    need(len(inventory) == 40 and len({row["participant_sha256"] for row in inventory}) == 40, "full participant inventory differs")

    train = load_writing_rows("train", include_scores=True)
    train_candidates = load_candidates("train", writings=train)
    smoke_candidate = next(row for row in train_candidates if row.candidate_number == 1)
    train_by_id = {row.identifier: row for row in train}
    smoke_path = participants_dir / "openai_terra_candidate_1.train-smoke1.jsonl"
    smoke_sha = write_participants(smoke_path, [(
        smoke_candidate.source_id,
        participant(train_by_id[smoke_candidate.source_id].scores or {}, smoke_candidate.diagnoses),
    )])

    private_manifest = {
        "schema_version": "mal2026-official-rationale-fidelity-participants-v1",
        "status": "prepared",
        "campaign": CAMPAIGN,
        "created_at": now(),
        "score_track": "canonical_human_reference_integerized_half_up_for_rationale_fidelity_only",
        "human_or_reference_score_read": True,
        "deployment_like_emitted_score_evaluation": False,
        "gpu_scope_authorized": [4, 5, 6, 7],
        "user_authorization": "2026-08-06: evaluate OpenAI rationales and not-yet-exact-judged trained rationale models on GPUs 4-7 with llm_as_judge.txt",
        "judge_prompt_file": str((ROOT / "llm_as_judge.txt").resolve()),
        "judge_prompt_sha256": sha256((ROOT / "llm_as_judge.txt").read_bytes()).hexdigest(),
        "source_sha256": SOURCE_SHA256,
        "smoke": {"split": "train", "records": 1, "participant_file": str(smoke_path.resolve()), "participant_sha256": smoke_sha},
        "participants": inventory,
    }
    atomic_json(destination / "manifest.json", private_manifest)
    aggregate_inventory = {
        "schema_version": "mal2026-official-rationale-fidelity-inventory-v1",
        "status": "prepared",
        "campaign": CAMPAIGN,
        "created_at": private_manifest["created_at"],
        "score_track": private_manifest["score_track"],
        "human_or_reference_score_read": True,
        "deployment_like_emitted_score_evaluation": False,
        "gpu_scope_authorized": [4, 5, 6, 7],
        "judge_prompt_sha256": private_manifest["judge_prompt_sha256"],
        "counts": {
            "participants": len(inventory),
            "rows_per_participant": 400,
            "judge_requests": len(inventory) * 400,
            "families": {family: sum(row["family"] == family for row in inventory) for family in sorted({row["family"] for row in inventory})},
        },
        "participant_provenance": [
            {key: row[key] for key in ("key", "family", "participant_sha256", "rationale_source_sha256", "records")}
            for row in inventory
        ],
        "privacy": "aggregate_only_no_ids_prompts_essays_rationales_scores_or_predictions",
    }
    atomic_json(aggregate_destination / "inventory.json", aggregate_inventory)
    return aggregate_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    setproctitle("mal2026:prepare-official-rationale-fidelity")
    args = parse_args()
    value = check_inputs() if args.check_only else prepare()
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
