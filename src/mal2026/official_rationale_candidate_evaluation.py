"""Contracts for final rationale-candidate generation and repeated Q4 evaluation.

The ten judge calls deliberately preserve temperature 0 and seed 42.  They
are repeated deterministic executions, not independent samples.  Row-bearing
participant, rationale, and judge artifacts must remain below the restricted
data root; only aggregate evaluation payloads are suitable for ``outputs/``.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from .official_rationale_handoff import (
    AXES, ROOT, candidate_identity_sha256, file_sha256,
    iter_jsonl, need, read_json, validate_training_completion,
)
from .official_writing_contract import JUDGE_DIMENSIONS, parse_judge_output, parse_participant_output


EVALUATION_SCHEMA = "mal2026-official-rationale-candidate-evaluation-v1"
BINDING_SCHEMA = "mal2026-official-rationale-candidate-bindings-v1"
REPEAT_INTERPRETATION = (
    "ten deterministic repeated executions at temperature=0 and seed=42; "
    "not independent samples; zero variance means runtime repeat agreement, not calibrated certainty"
)


def _restricted(path: Path) -> None:
    need(path.resolve().is_relative_to((ROOT / "data" / "processed" / "restricted").resolve()), "row artifact must remain restricted")


def load_emitted_scores(path: Path, expected: int) -> dict[str, dict[str, int]]:
    _restricted(path)
    result: dict[str, dict[str, int]] = {}
    for raw in iter_jsonl(path):
        need(set(raw) in ({"source_id", "emitted_integer_prediction"}, {"source_id", "continuous_prediction", "emitted_integer_prediction"}), "emitted score schema differs")
        source_id, scores = raw["source_id"], raw["emitted_integer_prediction"]
        need(isinstance(source_id, str) and source_id not in result, "emitted score source ID differs")
        need(isinstance(scores, dict) and set(scores) == set(AXES), "emitted score axes differ")
        need(all(type(scores[axis]) is int and 1 <= scores[axis] <= 5 for axis in AXES), "emitted score is not an official integer")
        result[source_id] = {axis: int(scores[axis]) for axis in AXES}
    need(len(result) == expected, "emitted score population differs")
    return result


def compose_participants(score_path: Path, rationale_path: Path, output: Path, expected: int) -> str:
    """Compose exact participant JSON while copying emitted scores unchanged."""
    _restricted(score_path); _restricted(rationale_path); _restricted(output)
    need(not output.exists(), "participant output already exists")
    scores = load_emitted_scores(score_path, expected)
    rationales: dict[str, dict[str, str]] = {}
    for raw in iter_jsonl(rationale_path):
        need(set(raw) == {"source_id", "rationales"}, "combined rationale schema differs")
        source_id, values = raw["source_id"], raw["rationales"]
        need(isinstance(source_id, str) and source_id not in rationales and isinstance(values, dict) and set(values) == set(AXES), "combined rationale identity differs")
        need(all(isinstance(values[axis], str) and values[axis].strip() for axis in AXES), "combined rationale is blank")
        rationales[source_id] = {axis: values[axis].strip() for axis in AXES}
    need(set(rationales) == set(scores) and len(scores) == expected, "score/rationale populations differ")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for source_id in sorted(scores):
            participant = {axis: {"score": scores[source_id][axis], "rationale": rationales[source_id][axis]} for axis in AXES}
            parsed = parse_participant_output(participant)
            need(all(parsed[axis]["score"] == scores[source_id][axis] for axis in AXES), "participant score mismatch")
            handle.write(json.dumps({"source_id": source_id, "participant_output": parsed}, ensure_ascii=False, separators=(",", ":")) + "\n")
    return file_sha256(output)


def _valid_repeat(report_path: Path, record_path: Path, expected: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = read_json(report_path, "judge repeat report")
    need(report.get("schema_version") == "mal2026-official-q4-judge-aggregate-v1", "judge repeat schema differs")
    need(report.get("status") == "completed" and report.get("judge_records_sha256") == file_sha256(record_path), "judge repeat is incomplete")
    need(report.get("temperature") == 0.0 and report.get("seed") == 42, "judge deterministic decode contract differs")
    need(report.get("human_or_reference_score_read_or_prompted") is False, "judge received a human/reference score")
    need(report.get("counts", {}).get("expected") == expected, "judge repeat population differs")
    records = list(iter_jsonl(record_path))
    need(len(records) == expected, "judge repeat record count differs")
    return report, records


def aggregate_deterministic_repeats(
    report_and_record_paths: Sequence[tuple[Path, Path]], expected: int = 400, repeats: int = 10,
) -> dict[str, Any]:
    need(len(report_and_record_paths) == repeats == 10, "judge repeat count differs")
    by_cell: dict[tuple[str, str, str], list[int]] = {}
    valid = 0
    repeat_summaries: list[dict[str, Any]] = []
    reference_ids: set[str] | None = None
    for repeat_index, (report_path, record_path) in enumerate(report_and_record_paths, 1):
        report, records = _valid_repeat(report_path, record_path, expected)
        repeat_valid = 0
        seen: set[str] = set()
        for record in records:
            source_id = record.get("source_id")
            need(isinstance(source_id, str) and source_id not in seen, "judge repeat source ID differs")
            seen.add(source_id)
            output = record.get("judge_output")
            if output is None:
                continue
            parsed = parse_judge_output(output); repeat_valid += 1; valid += 1
            for axis in AXES:
                for dimension in JUDGE_DIMENSIONS:
                    by_cell.setdefault((source_id, axis, dimension), []).append(int(parsed[axis][dimension]["score"]))
        if reference_ids is None: reference_ids = seen
        need(seen == reference_ids, "judge repeat populations differ")
        repeat_summaries.append({
            "repeat": repeat_index, "report_sha256": file_sha256(report_path),
            "records_sha256": file_sha256(record_path), "valid": repeat_valid,
            "macro_mean": report.get("macro_mean"), "worst_cell": report.get("worst_cell_mean"),
        })
    total_expected = expected * repeats
    cell_groups: dict[tuple[str, str], list[int]] = {(axis, dim): [] for axis in AXES for dim in JUDGE_DIMENSIONS}
    for (_, axis, dimension), values in by_cell.items():
        cell_groups[(axis, dimension)].extend(values)
    cell_means = {f"{axis}.{dimension}": statistics.fmean(values) if values else None for (axis, dimension), values in cell_groups.items()}
    complete_sequences = [values for values in by_cell.values() if len(values) == repeats]
    variances = [statistics.pvariance(values) for values in complete_sequences]
    parse_rate = valid / total_expected
    need(all(value is not None and math.isfinite(float(value)) for value in cell_means.values()), "judge cell aggregate is incomplete")
    return {
        "metrics": {
            "macro_mean": statistics.fmean(float(value) for value in cell_means.values()),
            "worst_cell": min(float(value) for value in cell_means.values()),
            "strict_parse_rate": parse_rate,
        },
        "repeat_diagnostics": {
            "interpretation": REPEAT_INTERPRETATION,
            "complete_record_cells": len(complete_sequences),
            "exact_agreement_rate": sum(len(set(values)) == 1 for values in complete_sequences) / len(complete_sequences) if complete_sequences else 0.0,
            "mean_population_variance": statistics.fmean(variances) if variances else None,
            "nonzero_variance_rate": sum(value > 0 for value in variances) / len(variances) if variances else 0.0,
            "zero_variance_is_not_independence_evidence": True,
        },
        "repeat_aggregates": repeat_summaries,
        "valid_calls": valid,
        "expected_calls": total_expected,
        "cell_means": cell_means,
    }


def evaluation_payload(
    candidate: Mapping[str, Any], judge: Mapping[str, Any], bootstrap_validation_score_sha256: str,
    rationale_sha256: str, participant_sha256: str, generation_reports: Mapping[str, Mapping[str, str]],
    repeated: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": EVALUATION_SCHEMA, "status": "completed", "candidate_key": candidate["key"],
        "candidate_identity_sha256": candidate_identity_sha256(candidate),
        "origin_classification": candidate["origin_classification"], "historical_method": candidate["historical_method"],
        "historical_source_sha256": candidate["historical_source_sha256"],
        "final_winner_eligible": candidate["final_winner_eligible"], "ranking_caveat": candidate["ranking_caveat"],
        "judge_contract_sha256": judge["contract_sha256"], "judge_model_sha256": judge["model_sha256"],
        "judge_prompt_kind": judge["prompt_kind"], "validation_records": 400, "repeats_per_record": 10,
        "decode_contract": {"temperature": 0.0, "top_p": 1.0, "seed": 42, "repeats_are_independent": False},
        "bootstrap_validation_score_sha256": bootstrap_validation_score_sha256,
        "generation_candidate_identity_sha256": candidate_identity_sha256(candidate),
        "generation_reports": {task: dict(value) for task, value in sorted(generation_reports.items())},
        "candidate_rationale_sha256": rationale_sha256, "participant_sha256": participant_sha256,
        "score_mismatches": 0, "human_or_reference_score_read_or_prompted": False,
        "validation_use": "descriptive_repeated_evaluation_only_not_training_reward_or_protocol_retuning",
        **dict(repeated),
    }


def resolve_handoff(template: Mapping[str, Any], bindings: Mapping[str, Any], require_evaluations: bool = True) -> dict[str, Any]:
    """Replace all runtime placeholders from explicit, completed artifact bindings."""
    need(bindings.get("schema_version") == BINDING_SCHEMA, "candidate binding schema differs")
    result = deepcopy(dict(template))
    bootstrap = Path(result["bootstrap_selection_path"])
    need(bootstrap.is_file(), "bootstrap selection is unavailable")
    result["bootstrap_selection_sha256"] = file_sha256(bootstrap)
    for key in ("directional_gate", "injection_gate"):
        path = Path(result["judge"][f"{key}_path"]); need(path.is_file(), f"{key} is unavailable")
        result["judge"][f"{key}_sha256"] = file_sha256(path)
    declared = {candidate["key"]: candidate for candidate in result["candidates"]}
    supplied = bindings.get("candidates")
    need(isinstance(supplied, dict) and set(supplied) == set(declared), "candidate binding coverage differs")
    for key, candidate in declared.items():
        binding = supplied[key]
        need(isinstance(binding, dict) and set(binding) == {"model_path", "model_binding_path", "adapters", "evaluation_path"}, f"candidate binding fields differ: {key}")
        model = Path(binding["model_path"]); model_binding = Path(binding["model_binding_path"])
        need(model.is_dir() and (model / "config.json").is_file() and model_binding.is_file(), f"candidate model binding unavailable: {key}")
        candidate["model_path"] = str(model.resolve()); candidate["model_config_sha256"] = file_sha256(model / "config.json")
        candidate["model_binding_path"] = str(model_binding.resolve()); candidate["model_binding_sha256"] = file_sha256(model_binding)
        tasks = {"bundle"} if candidate["structure"] == "bundle" else set(AXES)
        need(isinstance(binding["adapters"], dict) and set(binding["adapters"]) == tasks, f"candidate adapter coverage differs: {key}")
        for task in tasks:
            completion = Path(binding["adapters"][task]); value = read_json(completion, f"training completion {key}/{task}")
            validate_training_completion(candidate, task, value)
            adapter = completion.parent / "adapter"
            need((adapter / "adapter_config.json").is_file() and (adapter / "adapter_model.safetensors").is_file(), f"adapter export unavailable: {key}/{task}")
            candidate["adapters"][task] = {
                "path": str(adapter.resolve()), "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
                "adapter_model_sha256": file_sha256(adapter / "adapter_model.safetensors"),
                "training_completion_path": str(completion.resolve()), "training_completion_sha256": file_sha256(completion),
            }
            if value.get("model_id") is not None:
                need(value.get("model_revision") is not None, f"model revision absent: {key}/{task}")
                if str(candidate["model_id"]).startswith("REQUIRED_"):
                    candidate["model_id"] = value["model_id"]; candidate["model_revision"] = value["model_revision"]
                need(candidate["model_id"] == value["model_id"] and candidate["model_revision"] == value["model_revision"], f"candidate model identity differs: {key}/{task}")
            if value.get("model_path") is not None:
                need(Path(value["model_path"]).resolve() == model.resolve(), f"candidate completion base model differs: {key}/{task}")
        evaluation = Path(binding["evaluation_path"])
        candidate["evaluation_path"] = str(evaluation.resolve())
        if require_evaluations:
            need(evaluation.is_file(), f"candidate evaluation unavailable: {key}")
            candidate["evaluation_sha256"] = file_sha256(evaluation)
        else:
            candidate["evaluation_sha256"] = "PENDING_EVALUATION_SHA256"
    return result
