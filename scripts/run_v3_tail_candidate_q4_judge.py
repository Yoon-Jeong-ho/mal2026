#!/usr/bin/env python3
"""Judge every accepted frozen-v3 tail candidate with exact Qwen3.6 Q4."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
from typing import Any, Mapping, Sequence

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_balanced_rationale_q4_judge as base  # noqa: E402
import evaluate_official_q4_judge as q4_aggregate  # noqa: E402
from mal2026.official_writing_contract import AXES, JUDGE_DIMENSIONS, parse_participant_output  # noqa: E402


PRIVATE_PARENT = ROOT / "data/processed/restricted/rationale_v3_tail_q4_judge"
OUTPUT_PARENT = ROOT / "outputs/rationale-v3-tail-q4-judge"
GENERATION_PARENT = ROOT / "data/processed/restricted/rationale_v3_tail_batches"
PROMPT_SHA256 = "91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d"
MODEL_SHA256 = "b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772"
LOW_JUDGE_SCORE_RATE_MAX = 0.005


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def parse_gpus(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise RuntimeError("GPU scope must be comma-separated integers") from exc
    need(len(result) == 4 and len(set(result)) == 4 and all(0 <= gpu <= 7 for gpu in result), "GPU scope must contain four distinct GPUs")
    return result


def paths(campaign: str) -> tuple[Path, Path, Path, Path]:
    private = PRIVATE_PARENT / campaign
    output = OUTPUT_PARENT / campaign
    return private, output, private / "manifest.json", private / "participants"


def configure_base(campaign: str, output: Path, gpus: Sequence[int], authorization: str) -> None:
    base.CAMPAIGN = campaign
    base.OUT = output
    base.RUNTIME = output / "runtime"
    base.LEDGER = output / "ledger.jsonl"
    base.GPUS = tuple(gpus)
    base.SMOKE_GPU = (gpus[0],)
    base.GPU_AUTHORIZATION = authorization


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    private, output, manifest_path, participants_dir = paths(args.campaign)
    need(not private.exists() and not output.exists(), "judge campaign output must be fresh")
    need(sha256_file(ROOT / "llm_as_judge.txt") == PROMPT_SHA256, "judge prompt differs")
    need(sha256_file(base.MODEL) == MODEL_SHA256, "judge model differs")
    gpus = parse_gpus(args.gpu_scope)
    private.mkdir(parents=True, mode=0o700); participants_dir.mkdir(mode=0o700); output.mkdir(parents=True)
    inventory: list[dict[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    smoke_value: dict[str, Any] | None = None
    for generation_run in args.generation_run:
        generation_root = GENERATION_PARENT / generation_run
        generation_manifest_path = generation_root / "manifest.json"
        candidates_path = generation_root / "candidates.jsonl"
        need(generation_manifest_path.is_file() and candidates_path.is_file(), f"generation run unavailable: {generation_run}")
        generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
        candidates = jsonl(candidates_path)
        rejected = int(generation_manifest.get("rejected_or_missing", 0))
        requested = int(generation_manifest.get("requests", 0))
        need(
            generation_manifest.get("status") == "validated"
            and len(candidates) == generation_manifest.get("accepted")
            and len(candidates) + rejected == requested
            and len(candidates) / requested >= 0.99,
            f"generation gates incomplete: {generation_run}",
        )
        need(generation_manifest.get("prompt_sha256") == "b71ee648b9a6707c1e0156681adb9c4d47a3a4a4b751aa2cb90d0bc8808981c6", "generation prompt differs")
        model = str(generation_manifest["model"])
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            grouped[(str(row["split"]), int(row["variant"]))].append(row)
        for (split, variant), rows in sorted(grouped.items()):
            rows.sort(key=lambda row: str(row["source_id"]))
            need(len({row["source_id"] for row in rows}) == len(rows), "variant participant IDs differ")
            key = f"{model.replace('gpt-5.6-', '')}-{split}-v{variant}"
            participant_path = participants_dir / f"{key}.jsonl"
            with participant_path.open("x", encoding="utf-8") as handle:
                for row in rows:
                    participant = parse_participant_output({
                        axis: {"score": int(row["integer_scores"][axis]), "rationale": row["rationale"][axis]["rationale"]}
                        for axis in AXES
                    })
                    value = {"source_id": row["source_id"], "participant_output": participant}
                    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if smoke_value is None:
                        smoke_value = value
            os.chmod(participant_path, 0o600)
            inventory.append({
                "key": key, "model": model, "split": split, "variant": variant,
                "records": len(rows), "participant_file": str(participant_path.resolve()),
                "participant_sha256": sha256_file(participant_path), "generation_run": generation_run,
            })
        source_runs.append({
            "run_id": generation_run, "model": model, "scope": generation_manifest["scope"],
            "records": len(candidates), "rejected_surface_failures": rejected,
            "requests": requested, "acceptance_rate": len(candidates) / requested,
            "candidates_sha256": sha256_file(candidates_path),
            "generation_manifest_sha256": sha256_file(generation_manifest_path),
        })
    need(smoke_value is not None and inventory, "judge participant inventory is empty")
    smoke_path = participants_dir / "smoke1.jsonl"
    smoke_path.write_text(json.dumps(smoke_value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.chmod(smoke_path, 0o600)
    manifest = {
        "schema_version": "mal2026-rationale-v3-tail-q4-judge-campaign-v1", "status": "prepared",
        "campaign": args.campaign, "created_at": base.now(),
        "git_sha": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "generation_runs": source_runs, "participants": inventory,
        "participant_groups": len(inventory), "candidate_records": sum(item["records"] for item in inventory),
        "smoke": {"split": inventory[0]["split"], "path": str(smoke_path.resolve()), "sha256": sha256_file(smoke_path)},
        "judge_prompt_sha256": PROMPT_SHA256, "judge_model_sha256": MODEL_SHA256,
        "gpu_scope_authorized": list(gpus), "user_authorization": args.gpu_authorization,
        "human_or_reference_score_read_or_prompted": True,
        "deployment_like_emitted_score_evaluation": False,
        "privacy": "restricted participants; aggregate output contains no source IDs, prompts, essays, rationales, or judge evidence",
    }
    write_json(manifest_path, manifest)
    configure_base(args.campaign, output, gpus, args.gpu_authorization)
    base.append_ledger("prepared", gpu_scope=list(gpus), user_authorization=args.gpu_authorization, participant_groups=len(inventory), candidate_records=manifest["candidate_records"])
    return manifest


def combined_metrics(manifest: Mapping[str, Any]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    inventory_by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in manifest["participants"]:
        inventory_by_model[str(item["model"])].append(item)
    for model, items in sorted(inventory_by_model.items()):
        all_scores: list[int] = []
        axis_scores: dict[str, list[int]] = {axis: [] for axis in AXES}
        dimension_scores: dict[str, list[int]] = {dimension: [] for dimension in JUDGE_DIMENSIONS}
        band_scores: dict[str, list[int]] = {str(score): [] for score in range(1, 6)}
        candidate_minima: list[int] = []
        for item in items:
            participant = {row["source_id"]: row["participant_output"] for row in jsonl(Path(str(item["participant_file"])))}
            judge_run_id = str(item.get("judge_run_id") or f"{manifest['campaign']}-{item['key']}")
            restricted, _ = base.evaluator_paths(judge_run_id)
            records = jsonl(restricted / "judge_records.jsonl")
            need(len(records) == int(item["records"]), "judge record population differs")
            for record in records:
                output = record.get("judge_output")
                need(output is not None and record["source_id"] in participant, "judge record is invalid")
                current: list[int] = []
                for axis in AXES:
                    reference_band = str(participant[record["source_id"]][axis]["score"])
                    for dimension in JUDGE_DIMENSIONS:
                        score = int(output[axis][dimension]["score"])
                        all_scores.append(score); current.append(score)
                        axis_scores[axis].append(score); dimension_scores[dimension].append(score); band_scores[reference_band].append(score)
                candidate_minima.append(min(current))
        distribution = Counter(all_scores)
        low_judge_score_rate = (distribution[1] + distribution[2]) / len(all_scores)
        metrics = {
            "candidate_records": len(candidate_minima), "judge_cells": len(all_scores),
            "macro_mean": statistics.fmean(all_scores), "worst_candidate_cell_mean": statistics.fmean(candidate_minima),
            "candidates_with_any_score_below_4": sum(value < 4 for value in candidate_minima),
            "candidates_with_all_cells_5": sum(value == 5 for value in candidate_minima),
            "judge_score_distribution": {str(score): distribution[score] for score in range(1, 6)},
            "judge_score_1_or_2_rate": low_judge_score_rate,
            "axis_means": {axis: statistics.fmean(values) for axis, values in axis_scores.items()},
            "dimension_means": {dimension: statistics.fmean(values) for dimension, values in dimension_scores.items()},
            "reference_band_macro_means": {band: statistics.fmean(values) if values else None for band, values in band_scores.items()},
        }
        gates = {
            "macro_at_least_4_90": metrics["macro_mean"] >= 4.90,
            "consistency_at_least_4_85": metrics["dimension_means"]["score_rationale_consistency"] >= 4.85,
            "groundedness_at_least_4_85": metrics["dimension_means"]["groundedness"] >= 4.85,
            "every_reference_band_at_least_4_80": all(value is not None and value >= 4.80 for value in metrics["reference_band_macro_means"].values()),
            "judge_score_1_or_2_rate_at_most_0_005": low_judge_score_rate <= LOW_JUDGE_SCORE_RATE_MAX,
        }
        by_model[model] = {**metrics, "gates": gates, "passed": all(gates.values())}
    return by_model


def completed_judge(run_id: str, expected: int) -> bool:
    restricted, aggregate = base.evaluator_paths(run_id)
    if not (restricted / "judge_records.jsonl").is_file() or not aggregate.is_file():
        return False
    report = json.loads(aggregate.read_text(encoding="utf-8"))
    return (
        report.get("status") == "completed"
        and int(report.get("counts", {}).get("records", -1)) == expected
        and int(report.get("counts", {}).get("valid", -1)) == expected
    )


def repair_one_schema(args: argparse.Namespace) -> dict[str, Any]:
    """Repair one deterministic schema/finish failure without rerunning valid rows."""
    _, output, manifest_path, participants_dir = paths(args.campaign)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gpus = parse_gpus(args.gpu_scope)
    need(
        manifest.get("status") == "prepared"
        and manifest.get("gpu_scope_authorized") == list(gpus),
        "repair campaign manifest differs",
    )
    need(manifest.get("user_authorization") == args.gpu_authorization, "GPU authorization differs")
    items = [item for item in manifest["participants"] if item["key"] == args.participant_key]
    need(len(items) == 1, "repair participant key differs")
    item = items[0]
    need(not item.get("judge_run_id"), "participant already has a recovered judge run")
    original_id = f"{args.campaign}-{item['key']}"
    original_restricted, original_aggregate = base.evaluator_paths(original_id)
    need(
        (original_restricted / "judge_records.jsonl").is_file() and original_aggregate.is_file(),
        "failed judge output is unavailable",
    )
    original_report = json.loads(original_aggregate.read_text(encoding="utf-8"))
    need(
        original_report.get("status") == "failed_gates"
        and original_report.get("failure_categories") == {"schema_or_finish": 1}
        and int(original_report.get("counts", {}).get("records", -1)) == int(item["records"]),
        "repair is restricted to exactly one schema/finish failure",
    )
    records = jsonl(original_restricted / "judge_records.jsonl")
    invalid = [row for row in records if row.get("judge_output") is None]
    need(
        len(invalid) == 1 and invalid[0].get("failure_category") == "schema_or_finish",
        "failed record identity differs",
    )
    source_id = str(invalid[0]["source_id"])
    participants = {
        str(row["source_id"]): row for row in jsonl(Path(str(item["participant_file"])))
    }
    need(source_id in participants, "failed participant is unavailable")
    repair_participant = participants_dir / f"repair-{item['key']}-schema1.jsonl"
    expected_repair_bytes = (
        json.dumps(participants[source_id], ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if repair_participant.exists():
        need(repair_participant.read_bytes() == expected_repair_bytes, "repair participant differs")
    else:
        repair_participant.write_bytes(expected_repair_bytes)
        os.chmod(repair_participant, 0o600)

    configure_base(args.campaign, output, gpus, args.gpu_authorization)
    initial = base.require_idle(gpus)
    base.append_ledger(
        "schema_repair_preflight",
        participant_key=item["key"],
        failed_records=1,
        gpu_scope=list(gpus),
        gpu_state=initial,
    )
    processes = []
    repair_index = 1
    while base.evaluator_paths(f"{original_id}-schema-retry{repair_index}")[0].exists():
        repair_index += 1
    repair_run_id = f"{original_id}-schema-retry{repair_index}"
    phase = f"repair-schema{repair_index}-gpu{gpus[0]}"
    try:
        processes, attestation = base.launch(
            (gpus[0],), base.SMOKE_PORT, phase, parallel_per_server=1
        )
        base.evaluate(
            repair_run_id,
            repair_participant,
            1,
            [f"http://127.0.0.1:{base.SMOKE_PORT[0]}"],
            attestation,
            split=str(item["split"]),
        )
    finally:
        if processes:
            base.stop(processes, phase)
    base.wait_own_servers_released(gpus)

    repair_restricted, _ = base.evaluator_paths(repair_run_id)
    repair_records = jsonl(repair_restricted / "judge_records.jsonl")
    need(
        len(repair_records) == 1 and repair_records[0].get("judge_output") is not None,
        "schema repair did not produce one valid record",
    )
    replacement = {
        **repair_records[0],
        "recovery": {
            "kind": f"schema_retry{repair_index}",
            "original_run_id": original_id,
            "repair_run_id": repair_run_id,
        },
    }
    merged = [
        replacement if str(row["source_id"]) == source_id else row for row in records
    ]
    need(
        len(merged) == int(item["records"])
        and sum(row.get("judge_output") is not None for row in merged) == len(merged),
        "recovered population differs",
    )

    recovered_id = f"{original_id}-recovered1"
    recovered_restricted, recovered_aggregate = base.evaluator_paths(recovered_id)
    need(
        not recovered_restricted.exists() and not recovered_aggregate.parent.exists(),
        "recovered judge output must be fresh",
    )
    recovered_restricted.mkdir(parents=True, mode=0o700)
    recovered_aggregate.parent.mkdir(parents=True)
    record_path = recovered_restricted / "judge_records.jsonl"
    with record_path.open("x", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(record_path, 0o600)
    report = q4_aggregate.aggregate(merged, int(item["records"]))
    report.update({
        "schema_version": "mal2026-official-q4-rationale-fidelity-aggregate-v1",
        "run_id": recovered_id,
        "judge_records_sha256": sha256_file(record_path),
        "participant_sha256": item["participant_sha256"],
        "model_sha256": MODEL_SHA256,
        "llama_revision": base.LLAMA_REVISION,
        "llama_tag": base.LLAMA_TAG,
        "temperature": 0.0,
        "seed": 42,
        "candidate_score_kind": "canonical_human_reference_integerized_half_up_for_rationale_fidelity_only",
        "human_or_reference_score_read_or_prompted": True,
        "deployment_like_emitted_score_evaluation": False,
        "judge_system_prompt_kind": "repository_file_exact_bytes",
        "judge_system_prompt_sha256": PROMPT_SHA256,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_evidence_scores_or_predictions_in_this_report",
        "recovery": {
            "kind": "single_schema_or_finish_retry",
            "original_run_id": original_id,
            "original_judge_records_sha256": sha256_file(
                original_restricted / "judge_records.jsonl"
            ),
            "repair_run_id": repair_run_id,
            "valid_original_records_preserved": len(merged) - 1,
            "repaired_records": 1,
        },
    })
    need(report.get("status") == "completed", "recovered judge gates did not pass")
    write_json(recovered_aggregate, report)
    original_manifest = json.loads(
        (original_restricted / "manifest.json").read_text(encoding="utf-8")
    )
    recovered_manifest = {
        **original_manifest,
        "status": "completed",
        "run_id": recovered_id,
        "completed_at": base.now(),
        "aggregate_report_sha256": sha256_file(recovered_aggregate),
        "recovery": report["recovery"],
    }
    write_json(recovered_restricted / "manifest.json", recovered_manifest)
    item["judge_run_id"] = recovered_id
    manifest.setdefault("integration_recoveries", []).append(report["recovery"])
    write_json(manifest_path, manifest)
    base.append_ledger(
        "schema_repair_completed",
        participant_key=item["key"],
        recovered_run_id=recovered_id,
        preserved=len(merged) - 1,
        repaired=1,
    )
    return {
        "status": "completed",
        "campaign": args.campaign,
        "all_models_passed": None,
        "recovered_run_id": recovered_id,
    }


def exclude_one_schema(args: argparse.Namespace) -> dict[str, Any]:
    """Exclude one pathologically non-terminating judge candidate after retries."""
    _, output, manifest_path, participants_dir = paths(args.campaign)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gpus = parse_gpus(args.gpu_scope)
    need(
        manifest.get("status") == "prepared"
        and manifest.get("gpu_scope_authorized") == list(gpus),
        "exclusion campaign manifest differs",
    )
    need(manifest.get("user_authorization") == args.gpu_authorization, "GPU authorization differs")
    items = [item for item in manifest["participants"] if item["key"] == args.participant_key]
    need(len(items) == 1 and not items[0].get("judge_run_id"), "exclusion participant key differs")
    item = items[0]
    original_id = f"{args.campaign}-{item['key']}"
    original_restricted, original_aggregate = base.evaluator_paths(original_id)
    report = json.loads(original_aggregate.read_text(encoding="utf-8"))
    records = jsonl(original_restricted / "judge_records.jsonl")
    invalid = [row for row in records if row.get("judge_output") is None]
    need(
        report.get("failure_categories") == {"schema_or_finish": 1}
        and len(invalid) == 1
        and len(records) == int(item["records"]),
        "exclusion is restricted to exactly one schema/finish failure",
    )
    retry_reports = []
    for index in range(1, 5):
        _, retry_aggregate = base.evaluator_paths(f"{original_id}-schema-retry{index}")
        need(retry_aggregate.is_file(), "four schema retries are required before exclusion")
        value = json.loads(retry_aggregate.read_text(encoding="utf-8"))
        need(
            value.get("status") == "failed_gates"
            and value.get("failure_categories") == {"schema_or_finish": 1},
            "schema retry evidence differs",
        )
        retry_reports.append({
            "run_id": value["run_id"],
            "aggregate_sha256": sha256_file(retry_aggregate),
        })
    source_id = str(invalid[0]["source_id"])
    participant_rows = jsonl(Path(str(item["participant_file"])))
    filtered_participants = [
        row for row in participant_rows if str(row["source_id"]) != source_id
    ]
    valid_records = [row for row in records if row.get("judge_output") is not None]
    need(
        len(filtered_participants) == len(valid_records) == int(item["records"]) - 1,
        "exclusion population differs",
    )
    filtered_path = participants_dir / f"{item['key']}-schema-excluded1.jsonl"
    need(not filtered_path.exists(), "filtered participant output must be fresh")
    with filtered_path.open("x", encoding="utf-8") as handle:
        for row in filtered_participants:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(filtered_path, 0o600)

    recovered_id = f"{original_id}-schema-excluded1"
    recovered_restricted, recovered_aggregate = base.evaluator_paths(recovered_id)
    need(
        not recovered_restricted.exists() and not recovered_aggregate.parent.exists(),
        "excluded judge output must be fresh",
    )
    recovered_restricted.mkdir(parents=True, mode=0o700)
    recovered_aggregate.parent.mkdir(parents=True)
    record_path = recovered_restricted / "judge_records.jsonl"
    with record_path.open("x", encoding="utf-8") as handle:
        for row in valid_records:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(record_path, 0o600)
    recovered_report = q4_aggregate.aggregate(valid_records, len(valid_records))
    recovery = {
        "kind": "exclude_single_nonterminating_schema_candidate_after_four_retries",
        "original_run_id": original_id,
        "original_records": len(records),
        "preserved_valid_records": len(valid_records),
        "excluded_records": 1,
        "retry_evidence": retry_reports,
        "scientific_variables_changed": False,
        "reason": "exact prompt/model/seed repeatedly failed schema/finish at increasing token caps; no score or rationale was imputed",
    }
    recovered_report.update({
        "schema_version": "mal2026-official-q4-rationale-fidelity-aggregate-v1",
        "run_id": recovered_id,
        "judge_records_sha256": sha256_file(record_path),
        "participant_sha256": sha256_file(filtered_path),
        "model_sha256": MODEL_SHA256,
        "llama_revision": base.LLAMA_REVISION,
        "llama_tag": base.LLAMA_TAG,
        "temperature": 0.0,
        "seed": 42,
        "candidate_score_kind": "canonical_human_reference_integerized_half_up_for_rationale_fidelity_only",
        "human_or_reference_score_read_or_prompted": True,
        "deployment_like_emitted_score_evaluation": False,
        "judge_system_prompt_kind": "repository_file_exact_bytes",
        "judge_system_prompt_sha256": PROMPT_SHA256,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_evidence_scores_or_predictions_in_this_report",
        "recovery": recovery,
    })
    need(recovered_report.get("status") == "completed", "excluded judge aggregate did not pass")
    write_json(recovered_aggregate, recovered_report)
    original_manifest = json.loads(
        (original_restricted / "manifest.json").read_text(encoding="utf-8")
    )
    write_json(recovered_restricted / "manifest.json", {
        **original_manifest,
        "status": "completed",
        "run_id": recovered_id,
        "participant_sha256": sha256_file(filtered_path),
        "participant_records": len(filtered_participants),
        "completed_at": base.now(),
        "aggregate_report_sha256": sha256_file(recovered_aggregate),
        "recovery": recovery,
    })
    item.update({
        "original_records": int(item["records"]),
        "original_participant_file": item["participant_file"],
        "original_participant_sha256": item["participant_sha256"],
        "records": len(filtered_participants),
        "participant_file": str(filtered_path.resolve()),
        "participant_sha256": sha256_file(filtered_path),
        "judge_run_id": recovered_id,
        "excluded_source_ids": [source_id],
        "excluded_unjudgeable_records": 1,
    })
    manifest["candidate_records"] = int(manifest["candidate_records"]) - 1
    manifest.setdefault("integration_recoveries", []).append(recovery)
    write_json(manifest_path, manifest)
    configure_base(args.campaign, output, gpus, args.gpu_authorization)
    base.append_ledger(
        "nonterminating_schema_candidate_excluded",
        participant_key=item["key"],
        preserved=len(valid_records),
        excluded=1,
        recovered_run_id=recovered_id,
    )
    return {
        "status": "completed",
        "campaign": args.campaign,
        "all_models_passed": None,
        "recovered_run_id": recovered_id,
    }


def build_summary(
    manifest: Mapping[str, Any],
    gpus: Sequence[int],
    telemetry: Mapping[str, Any],
    final_gpu_state: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    models = combined_metrics(manifest)
    return {
        "schema_version": "mal2026-rationale-v3-tail-q4-judge-summary-v2", "status": "completed",
        "campaign": manifest["campaign"], "completed_at": base.now(), "gpu_scope": list(gpus),
        "judge_prompt_sha256": PROMPT_SHA256, "judge_model_sha256": MODEL_SHA256,
        "candidate_records": manifest["candidate_records"], "participant_groups": manifest["participant_groups"],
        "models": models, "all_models_passed": all(row["passed"] for row in models.values()),
        "telemetry": telemetry, "final_gpu_state": list(final_gpu_state),
        "gate_interpretation": {
            "judge_score_1_or_2_rate_max": LOW_JUDGE_SCORE_RATE_MAX,
            "rationale": "The preregistered wording allowed score 1/2 cells to be absent or very rare; this operationalizes very rare as at most 0.5% of judge cells.",
        },
        "caveat": "Scores measure conditional rationale fidelity to supplied integerized human/reference scores, not emitted-score prediction. Low consistency cells can reflect judge re-grading, especially for human/reference score 5, and are diagnostic rather than an automatic tail-candidate deletion rule.",
        "privacy": "aggregate_only_no_source_ids_prompts_essays_rationales_or_judge_evidence",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    private, output, manifest_path, _ = paths(args.campaign)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gpus = parse_gpus(args.gpu_scope)
    need(manifest.get("status") == "prepared" and manifest.get("gpu_scope_authorized") == list(gpus), "judge campaign manifest differs")
    need(manifest.get("user_authorization") == args.gpu_authorization, "GPU authorization differs")
    configure_base(args.campaign, output, gpus, args.gpu_authorization)
    initial = base.require_idle(gpus)
    base.append_ledger("authorized_preflight", gpu_scope=list(gpus), gpu_state=initial)
    smoke_processes = []
    smoke_phase = f"smoke-gpu{gpus[0]}"
    smoke_run_id = f"{args.campaign}-smoke1"
    if not completed_judge(smoke_run_id, 1):
        try:
            smoke_processes, attestation = base.launch((gpus[0],), base.SMOKE_PORT, smoke_phase)
            smoke = manifest["smoke"]
            base.evaluate(
                smoke_run_id, Path(str(smoke["path"])), 1,
                [f"http://127.0.0.1:{base.SMOKE_PORT[0]}"], attestation,
                split=str(smoke["split"]),
            )
        finally:
            if smoke_processes:
                base.stop(smoke_processes, smoke_phase)
    else:
        base.append_ledger("smoke_reused", run_id=smoke_run_id)
    base.wait_own_servers_released(gpus)
    full_processes = []
    attempt = len(manifest.get("run_attempts", [])) + 1
    phase = "full-gpu" + "-".join(map(str, gpus)) + f"-attempt{attempt}"
    manifest.setdefault("run_attempts", []).append({
        "attempt": attempt, "started_at": base.now(), "phase": phase,
    })
    write_json(manifest_path, manifest)
    monitor_stop = threading.Event(); monitor = None
    try:
        full_processes, attestation = base.launch(gpus, base.PORTS, phase)
        monitor = threading.Thread(target=base.telemetry, args=(monitor_stop, phase), daemon=True); monitor.start()
        endpoints = [f"http://127.0.0.1:{port}" for port in base.PORTS]
        for index, item in enumerate(manifest["participants"], 1):
            run_id = str(item.get("judge_run_id") or f"{args.campaign}-{item['key']}")
            if completed_judge(run_id, int(item["records"])):
                print(json.dumps({
                    "progress": f"{index}/{len(manifest['participants'])}",
                    "run_id": run_id,
                    "reused": True,
                }), flush=True)
                continue
            base.evaluate(
                run_id, Path(str(item["participant_file"])), int(item["records"]), endpoints, attestation,
                split=str(item["split"]),
            )
            print(json.dumps({"progress": f"{index}/{len(manifest['participants'])}", "run_id": run_id}), flush=True)
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=5)
        if full_processes:
            base.stop(full_processes, phase)
    final_gpu_state = base.wait_own_servers_released(gpus)
    summary = build_summary(
        manifest, gpus, base.telemetry_summary(base.RUNTIME / phase / "gpu_telemetry.jsonl"), final_gpu_state,
    )
    write_json(output / "aggregate_summary.json", summary)
    manifest["status"] = "completed"; manifest["completed_at"] = base.now(); manifest["aggregate_sha256"] = sha256_file(output / "aggregate_summary.json")
    write_json(manifest_path, manifest)
    base.append_ledger("campaign_completed", all_models_passed=summary["all_models_passed"], aggregate_sha256=manifest["aggregate_sha256"])
    return summary


def reaggregate(args: argparse.Namespace) -> dict[str, Any]:
    """Recompute aggregate gates from immutable completed judge records."""
    _, output, manifest_path, _ = paths(args.campaign)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gpus = parse_gpus(args.gpu_scope)
    need(manifest.get("status") == "completed", "completed judge campaign required")
    need(manifest.get("gpu_scope_authorized") == list(gpus), "judge campaign GPU scope differs")
    need(manifest.get("user_authorization") == args.gpu_authorization, "GPU authorization differs")
    old_summary_path = output / "aggregate_summary.json"
    old_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
    summary = build_summary(
        manifest, gpus, old_summary.get("telemetry", {}), old_summary.get("final_gpu_state", []),
    )
    summary["reaggregated_from_schema_version"] = old_summary.get("schema_version")
    write_json(old_summary_path, summary)
    manifest["aggregate_sha256"] = sha256_file(old_summary_path)
    manifest["reaggregated_at"] = base.now()
    write_json(manifest_path, manifest)
    configure_base(args.campaign, output, gpus, args.gpu_authorization)
    base.append_ledger(
        "campaign_reaggregated",
        all_models_passed=summary["all_models_passed"],
        low_judge_score_rate_max=LOW_JUDGE_SCORE_RATE_MAX,
        aggregate_sha256=manifest["aggregate_sha256"],
    )
    return summary


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--campaign", required=True)
    common.add_argument("--gpu-scope", default="0,1,2,3")
    common.add_argument("--gpu-authorization", required=True)
    prepare_parser = sub.add_parser("prepare", parents=[common]); prepare_parser.add_argument("--generation-run", action="append", required=True); prepare_parser.set_defaults(func=prepare)
    run_parser = sub.add_parser("run", parents=[common]); run_parser.set_defaults(func=run)
    repair_parser = sub.add_parser("repair-one-schema", parents=[common])
    repair_parser.add_argument("--participant-key", required=True)
    repair_parser.set_defaults(func=repair_one_schema)
    exclusion_parser = sub.add_parser("exclude-one-schema", parents=[common])
    exclusion_parser.add_argument("--participant-key", required=True)
    exclusion_parser.set_defaults(func=exclude_one_schema)
    reaggregate_parser = sub.add_parser("reaggregate", parents=[common]); reaggregate_parser.set_defaults(func=reaggregate)
    return parser


def main() -> int:
    args = parser().parse_args()
    setproctitle(f"mal2026:rationale-v3-tail-q4:{args.command}:{args.campaign}"[:255])
    result = args.func(args)
    print(json.dumps({"status": result["status"], "campaign": args.campaign, "all_models_passed": result.get("all_models_passed")}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
