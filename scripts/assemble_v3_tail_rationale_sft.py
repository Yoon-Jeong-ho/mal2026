#!/usr/bin/env python3
"""Assemble score-free SFT targets from the frozen-v3 two-teacher campaign.

Raw candidate text, identifiers, labels, and per-candidate judge scores remain in
the restricted tree.  Public output is aggregate-only.  Judge consistency is
diagnostic: it is deliberately not a deletion rule because the Q4 judge can
re-grade the supplied human/reference label, especially at score 5.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_v3_tail_rationale_batch as generation  # noqa: E402
import run_balanced_rationale_q4_judge as judge_base  # noqa: E402
from mal2026.official_writing_contract import AXES, JUDGE_DIMENSIONS  # noqa: E402


GENERATION_ROOT = ROOT / "data/processed/restricted/rationale_v3_tail_batches"
JUDGE_ROOT = ROOT / "data/processed/restricted/official_rationale_fidelity_v1/q4_judge"
PRIVATE_ROOT = ROOT / "data/processed/restricted/rationale_v3_tail_sft"
OUTPUT_ROOT = ROOT / "outputs/rationale-v3-tail-sft"
EXPECTED_PROMPT_SHA256 = generation.PROMPT_SHA256
EXPECTED_JUDGE_PROMPT_SHA256 = "91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d"
TEACHERS = generation.MODELS
NON_CONSISTENCY_DIMENSIONS = tuple(d for d in JUDGE_DIMENSIONS if d != "score_rationale_consistency")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(path, 0o600)


def normalized_target(rationale: Mapping[str, Any]) -> str:
    return re.sub(r"\s+", "", json.dumps(rationale, ensure_ascii=False, sort_keys=True))


def candidate_key(row: Mapping[str, Any]) -> str:
    value = "\0".join((str(row["model"]), str(row["split"]), str(row["variant"]), str(row["source_id"])))
    return hashlib.sha256(value.encode()).hexdigest()


def judge_inventory_item(
    judge_manifest: Mapping[str, Any], model: str, split: str, variant: int
) -> Mapping[str, Any]:
    short = model.removeprefix("gpt-5.6-")
    key = f"{short}-{split}-v{variant}"
    items = [item for item in judge_manifest["participants"] if item["key"] == key]
    need(len(items) == 1, f"judge inventory differs: {key}")
    return items[0]


def load_judges(
    campaign: str,
    model: str,
    split: str,
    variant: int,
    judge_manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    short = model.removeprefix("gpt-5.6-")
    item = judge_inventory_item(judge_manifest, model, split, variant)
    judge_run_id = str(item.get("judge_run_id") or f"{campaign}-{short}-{split}-v{variant}")
    path = JUDGE_ROOT / judge_run_id / "judge_records.jsonl"
    need(path.is_file(), f"judge records unavailable: {short}-{split}-v{variant}")
    rows = jsonl(path)
    need(all(row.get("judge_output") is not None for row in rows), "judge output is incomplete")
    result = {str(row["source_id"]): row["judge_output"] for row in rows}
    need(len(result) == len(rows), "judge source IDs differ")
    return result


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    private = PRIVATE_ROOT / args.run_id
    public = OUTPUT_ROOT / args.run_id
    need(not private.exists() and not public.exists(), "assembly output must be fresh")
    judge_manifest_path = ROOT / "data/processed/restricted/rationale_v3_tail_q4_judge" / args.judge_campaign / "manifest.json"
    judge_summary_path = ROOT / "outputs/rationale-v3-tail-q4-judge" / args.judge_campaign / "aggregate_summary.json"
    need(judge_manifest_path.is_file() and judge_summary_path.is_file(), "completed judge campaign is required")
    judge_manifest = json.loads(judge_manifest_path.read_text(encoding="utf-8"))
    judge_summary = json.loads(judge_summary_path.read_text(encoding="utf-8"))
    need(judge_manifest.get("status") == "completed" and judge_summary.get("status") == "completed", "judge campaign is incomplete")
    need(judge_summary.get("judge_prompt_sha256") == EXPECTED_JUDGE_PROMPT_SHA256, "judge prompt differs")

    private.mkdir(parents=True, mode=0o700)
    public.mkdir(parents=True)
    candidates: list[dict[str, Any]] = []
    generation_inputs: list[dict[str, Any]] = []
    for run_id in args.generation_run:
        root = GENERATION_ROOT / run_id
        manifest_path = root / "manifest.json"
        candidate_path = root / "candidates.jsonl"
        need(manifest_path.is_file() and candidate_path.is_file(), f"generation run unavailable: {run_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = jsonl(candidate_path)
        need(manifest.get("status") == "validated" and manifest.get("prompt_sha256") == EXPECTED_PROMPT_SHA256, "generation provenance differs")
        need(len(rows) == int(manifest["accepted"]), "accepted generation population differs")
        candidates.extend(rows)
        generation_inputs.append({
            "run_id": run_id, "model": manifest["model"], "split": manifest["split"],
            "requests": manifest["requests"], "accepted": manifest["accepted"],
            "rejected": manifest["rejected_or_missing"], "candidate_sha256": sha256_file(candidate_path),
            "manifest_sha256": sha256_file(manifest_path),
        })
    need({str(row["model"]) for row in candidates} == set(TEACHERS), "both teachers are required")
    need({str(row["split"]) for row in candidates} == {"train", "validation"}, "train and validation are required")
    keys = [candidate_key(row) for row in candidates]
    need(len(keys) == len(set(keys)), "candidate keys differ")

    judge_cache: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = {}
    target_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    provenance_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_seen: dict[tuple[str, str, str], str] = {}
    cross_candidate_exact_duplicates = 0
    excluded_unjudgeable = 0
    severe_count = review_count = clean_count = 0
    teacher_counts: Counter[tuple[str, str]] = Counter()
    band_candidate_counts: Counter[tuple[str, str, int]] = Counter()
    quality_filtered_teacher_counts: Counter[tuple[str, str]] = Counter()
    coverage: dict[tuple[str, str], set[str]] = defaultdict(set)
    quality_coverage: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in sorted(candidates, key=lambda value: (value["split"], value["source_id"], value["model"], int(value["variant"]))):
        model, split, variant = str(row["model"]), str(row["split"]), int(row["variant"])
        cache_key = (model, split, variant)
        inventory_item = judge_inventory_item(judge_manifest, model, split, variant)
        if str(row["source_id"]) in {
            str(value) for value in inventory_item.get("excluded_source_ids", [])
        }:
            excluded_unjudgeable += 1
            continue
        if cache_key not in judge_cache:
            judge_cache[cache_key] = load_judges(
                args.judge_campaign, model, split, variant, judge_manifest
            )
        output = judge_cache[cache_key].get(str(row["source_id"]))
        need(output is not None, "candidate has no matching judge record")
        scores = {
            axis: {dimension: int(output[axis][dimension]["score"]) for dimension in JUDGE_DIMENSIONS}
            for axis in AXES
        }
        non_consistency = [scores[axis][dimension] for axis in AXES for dimension in NON_CONSISTENCY_DIMENSIONS]
        consistency = [scores[axis]["score_rationale_consistency"] for axis in AXES]
        severe = min(non_consistency) <= 2
        review = not severe and (min(non_consistency) == 3 or min(consistency) <= 2)
        tier = "severe_non_consistency_issue" if severe else "review" if review else "clean"
        severe_count += int(severe); review_count += int(review); clean_count += int(tier == "clean")
        key = candidate_key(row)
        target = {axis: {"rationale": str(row["rationale"][axis]["rationale"])} for axis in AXES}
        parsed, errors = generation.prompt_contract.validate_output(json.dumps(target, ensure_ascii=False))
        need(not errors and parsed == target, "accepted target no longer passes score-free output validation")
        need(not any(generation.FOREIGN_SCRIPT_RE.search(target[axis]["rationale"]) for axis in AXES), "accepted target contains foreign script")
        target_row = {"candidate_key": key, "source_id": str(row["source_id"]), "rationale": target}
        target_rows[f"{split}.valid"].append(target_row)
        teacher_counts[(split, model)] += 1
        coverage[(split, model)].add(str(row["source_id"]))
        for axis in AXES:
            band_candidate_counts[(split, axis, int(row["integer_scores"][axis]))] += 1

        fingerprint_key = (split, str(row["source_id"]), normalized_target(target))
        duplicate_of = exact_seen.get(fingerprint_key)
        if duplicate_of is None:
            exact_seen[fingerprint_key] = key
        else:
            cross_candidate_exact_duplicates += 1
        quality_included = not severe and duplicate_of is None
        if quality_included:
            target_rows[f"{split}.quality_filtered"].append(target_row)
            quality_filtered_teacher_counts[(split, model)] += 1
            quality_coverage[(split, model)].add(str(row["source_id"]))
        provenance_rows[split].append({
            "candidate_key": key, "source_id": str(row["source_id"]), "teacher_model": model,
            "variant": variant, "target_multiplicity": int(row["target_multiplicity"]),
            "integer_scores": {axis: int(row["integer_scores"][axis]) for axis in AXES},
            "judge_scores": scores, "quality_tier": tier,
            "quality_filtered_included": quality_included, "exact_duplicate_of": duplicate_of,
            "essay_sha256": row["essay_sha256"], "generation_prompt_sha256": row["prompt_sha256"],
        })

    files: dict[str, dict[str, Any]] = {}
    for name, rows in sorted(target_rows.items()):
        path = private / f"sft_targets.{name}.jsonl"
        write_jsonl(path, rows)
        files[path.name] = {"records": len(rows), "sha256": sha256_file(path), "contains_scores": False}
    for split, rows in sorted(provenance_rows.items()):
        path = private / f"provenance.{split}.jsonl"
        write_jsonl(path, rows)
        files[path.name] = {"records": len(rows), "sha256": sha256_file(path), "contains_scores": True}

    summary = {
        "schema_version": "mal2026-rationale-v3-tail-sft-aggregate-v1", "status": "completed",
        "run_id": args.run_id, "judge_campaign": args.judge_campaign,
        "generation_inputs": generation_inputs, "candidate_records": len(candidates),
        "excluded_unjudgeable_candidates": excluded_unjudgeable,
        "judged_candidate_records": len(candidates) - excluded_unjudgeable,
        "mechanically_valid_targets": {split: len(target_rows[f"{split}.valid"]) for split in ("train", "validation")},
        "quality_filtered_targets": {split: len(target_rows[f"{split}.quality_filtered"]) for split in ("train", "validation")},
        "quality_tiers": {"clean": clean_count, "review": review_count, "severe_non_consistency_issue": severe_count},
        "teacher_candidate_counts": {split: {model: teacher_counts[(split, model)] for model in TEACHERS} for split in ("train", "validation")},
        "quality_filtered_teacher_counts": {split: {model: quality_filtered_teacher_counts[(split, model)] for model in TEACHERS} for split in ("train", "validation")},
        "source_coverage": {split: {model: len(coverage[(split, model)]) for model in TEACHERS} for split in ("train", "validation")},
        "quality_filtered_source_coverage": {split: {model: len(quality_coverage[(split, model)]) for model in TEACHERS} for split in ("train", "validation")},
        "candidate_axis_band_counts": {
            split: {axis: {str(band): band_candidate_counts[(split, axis, band)] for band in range(1, 6)} for axis in AXES}
            for split in ("train", "validation")
        },
        "cross_candidate_exact_duplicates_removed_from_quality_filtered": cross_candidate_exact_duplicates,
        "selection_policy": {
            "valid": "all mechanically valid score-free candidates from both frozen teachers",
            "quality_filtered": "valid minus exact within-source target duplicates and any candidate with domain_match, groundedness, or specificity at most 2 on any axis",
            "consistency": "diagnostic only; never an automatic deletion rule because the judge can re-grade supplied human/reference score 5",
            "unjudgeable": "exclude only candidates explicitly recorded by the restricted campaign manifest after repeated exact-judge schema/finish failure; never impute a judge score",
        },
        "target_contract": "target files contain source_id plus three rationales only; integer scores and judge metadata are isolated in restricted provenance files",
        "files": files,
        "privacy": "aggregate_only_no_source_ids_prompts_essays_rationales_or_judge_evidence",
    }
    write_json(public / "aggregate.json", summary)
    manifest = {
        "schema_version": "mal2026-rationale-v3-tail-sft-manifest-v1", "status": "completed",
        "created_at": generation.now(), "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "run_id": args.run_id, "judge_campaign": args.judge_campaign,
        "judge_manifest_sha256": sha256_file(judge_manifest_path), "judge_summary_sha256": sha256_file(judge_summary_path),
        "generation_inputs": generation_inputs, "files": files,
        "aggregate_sha256": sha256_file(public / "aggregate.json"),
        "privacy": "restricted targets/provenance; public aggregate only",
    }
    write_json(private / "manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--judge-campaign", required=True)
    parser.add_argument("--generation-run", action="append", required=True)
    args = parser.parse_args()
    need(bool(re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,120}", args.run_id)), "invalid run ID")
    setproctitle(f"mal2026:assemble-v3-tail-sft:{args.run_id}"[:255])
    result = assemble(args)
    print(json.dumps({
        "status": result["status"], "run_id": args.run_id,
        "mechanically_valid_targets": result["mechanically_valid_targets"],
        "quality_filtered_targets": result["quality_filtered_targets"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
