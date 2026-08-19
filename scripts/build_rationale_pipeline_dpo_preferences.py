#!/usr/bin/env python3
"""Build train-only DPO pairs from exact-Q4-scored teacher rationales."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import load_writing_rows, sha256_file  # noqa: E402
from mal2026.rationale_pipeline_prompts import AXES, rationale_messages, rationale_output, routing  # noqa: E402


MERGED_PARENT = ROOT / "data/processed/restricted/rationale_v3_tail_sft_merged"
RESTRICTED_PARENT = ROOT / "data/processed/restricted/rationale_pipeline_v1/dpo_preferences"
AGGREGATE_PARENT = ROOT / "outputs/rationale-pipeline-dpo-preferences-v1"
DIMENSIONS = ("score_rationale_consistency", "groundedness", "specificity", "domain_match")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def judge_total(value: Mapping[str, Any]) -> int:
    need(set(value) == set(AXES), "judge axes differ")
    scores: list[int] = []
    for axis in AXES:
        need(set(value[axis]) == set(DIMENSIONS), "judge dimensions differ")
        for dimension in DIMENSIONS:
            cell = value[axis][dimension]
            score = cell.get("score") if isinstance(cell, Mapping) else cell
            need(type(score) is int and 1 <= score <= 5, "judge cell score differs")
            scores.append(score)
    need(len(scores) == 12, "judge cell count differs")
    return sum(scores)


def stable_pick(source_id: str, candidates: list[dict[str, Any]], role: str) -> dict[str, Any]:
    need(candidates, "preference candidate set is empty")
    return min(candidates, key=lambda row: sha256(f"2026080704:{role}:{source_id}:{row['candidate_key']}".encode()).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--merged-run-id", default="rationale-v3-tail-sft-merged-20260807-001")
    parser.add_argument("--minimum-margin", type=int, default=1)
    args = parser.parse_args()
    setproctitle(f"mal2026:rationale-dpo-preferences:{args.run_id}"[:255])
    need(args.minimum_margin == 1, "preference margin is frozen to the smallest exact non-tie")
    prompt_routing = routing()
    root = MERGED_PARENT / args.merged_run_id
    manifest_path = root / "manifest.json"
    target_path = root / "sft_targets.train.quality_filtered.jsonl"
    provenance_path = root / "provenance.train.jsonl"
    need(manifest_path.is_file() and target_path.is_file() and provenance_path.is_file(), "merged rationale handoff unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    need(manifest.get("status") == "completed", "merged handoff is incomplete")
    for path in (target_path, provenance_path):
        item = manifest["files"][path.name]
        need(item["sha256"] == sha256_file(path), f"merged file checksum differs: {path.name}")

    restricted = RESTRICTED_PARENT / args.run_id
    aggregate = AGGREGATE_PARENT / args.run_id
    need(not restricted.exists() and not aggregate.exists(), "DPO preference output must be fresh")
    restricted.mkdir(parents=True, mode=0o700); aggregate.mkdir(parents=True)

    targets = {row["candidate_key"]: row for row in read_jsonl(target_path)}
    need(len(targets) == int(manifest["files"][target_path.name]["records"]), "DPO target population differs")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    quality_rows = 0
    for row in read_jsonl(provenance_path):
        if row.get("quality_filtered_included") is not True:
            continue
        candidate_key = str(row["candidate_key"])
        need(candidate_key in targets and targets[candidate_key]["source_id"] == row["source_id"], "DPO target/provenance linkage differs")
        total = judge_total(row["judge_scores"])
        grouped[str(row["source_id"])].append({
            "candidate_key": candidate_key,
            "total": total,
            "teacher_model": str(row["teacher_model"]),
            "variant": int(row["variant"]),
            "rationale": targets[candidate_key]["rationale"],
        })
        quality_rows += 1
    need(quality_rows == len(targets) and len(grouped) == 2000, "DPO grouped population differs")
    writings = {row.identifier: row for row in load_writing_rows("train", include_scores=False)}
    need(set(grouped) == set(writings), "DPO sources differ from canonical train")

    pairs: list[dict[str, Any]] = []
    excluded_ties = 0
    margins: Counter[int] = Counter()
    teacher_pairs: Counter[str] = Counter()
    candidate_counts: Counter[int] = Counter()
    for source_id in sorted(grouped):
        candidates = grouped[source_id]
        candidate_counts[len(candidates)] += 1
        minimum = min(row["total"] for row in candidates)
        maximum = max(row["total"] for row in candidates)
        margin = maximum - minimum; margins[margin] += 1
        if margin < args.minimum_margin:
            excluded_ties += 1
            continue
        chosen = stable_pick(source_id, [row for row in candidates if row["total"] == maximum], "chosen")
        rejected = stable_pick(source_id, [row for row in candidates if row["total"] == minimum], "rejected")
        need(chosen["candidate_key"] != rejected["candidate_key"] and chosen["total"] > rejected["total"], "DPO preference direction differs")
        writing = writings[source_id]
        chosen_target = json.dumps(rationale_output(chosen["rationale"]), ensure_ascii=False, separators=(",", ":"))
        rejected_target = json.dumps(rationale_output(rejected["rationale"]), ensure_ascii=False, separators=(",", ":"))
        pairs.append({
            "prompt": rationale_messages(writing.prompt, writing.essay),
            "chosen": [{"role": "assistant", "content": chosen_target}],
            "rejected": [{"role": "assistant", "content": rejected_target}],
            "metadata": {
                "source_id": source_id,
                "chosen_candidate_key": chosen["candidate_key"],
                "rejected_candidate_key": rejected["candidate_key"],
                "chosen_judge_total": chosen["total"],
                "rejected_judge_total": rejected["total"],
                "judge_total_margin": margin,
                "chosen_teacher": chosen["teacher_model"],
                "rejected_teacher": rejected["teacher_model"],
            },
        })
        teacher_pairs[f"{chosen['teacher_model']}->{rejected['teacher_model']}"] += 1
    need(len(pairs) + excluded_ties == 2000 and pairs, "DPO preference accounting differs")
    output_path = restricted / "preferences.train.jsonl"
    with output_path.open("x", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(output_path, 0o600)
    nonzero_margins = [row["metadata"]["judge_total_margin"] for row in pairs]
    report = {
        "schema_version": "mal2026-rationale-pipeline-dpo-preferences-aggregate-v1",
        "status": "completed", "run_id": args.run_id, "created_at": now(),
        "split": "train", "source_records": len(grouped), "quality_candidates": quality_rows,
        "preference_pairs": len(pairs), "excluded_exact_ties": excluded_ties,
        "minimum_total_margin": args.minimum_margin,
        "margin_policy": "smallest observable non-tie in exact deterministic 12-cell integer sum",
        "margin_distribution_all_sources": {str(key): value for key, value in sorted(margins.items())},
        "selected_margin_mean": statistics.fmean(nonzero_margins),
        "candidate_count_distribution": {str(key): value for key, value in sorted(candidate_counts.items())},
        "teacher_direction_distribution": dict(sorted(teacher_pairs.items())),
        "preferences_sha256": sha256_file(output_path),
        "merged_manifest_sha256": sha256_file(manifest_path),
        "merged_targets_sha256": sha256_file(target_path),
        "merged_provenance_sha256": sha256_file(provenance_path),
        "rationale_prompt_sha256": prompt_routing["rationale_generation_training_evaluation"]["source_file_sha256"],
        "judge_prompt_sha256": prompt_routing["rationale_reward_and_quality_judge"]["source_file_sha256"],
        "judge_total_definition": "sum of content/organization/expression x consistency/groundedness/specificity/domain_match",
        "scores_in_policy_prompt": False, "validation_used": False, "average_used": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_model_weights",
    }
    atomic_json(aggregate / "aggregate.json", report)
    atomic_json(restricted / "manifest.json", {
        "schema_version": "mal2026-rationale-pipeline-dpo-preferences-manifest-v1", "status": "completed",
        "run_id": args.run_id, "preferences_sha256": report["preferences_sha256"],
        "aggregate_sha256": sha256_file(aggregate / "aggregate.json"), "records": len(pairs),
    })
    print(json.dumps({"status": "completed", "run_id": args.run_id, "pairs": len(pairs), "ties": excluded_ties}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
