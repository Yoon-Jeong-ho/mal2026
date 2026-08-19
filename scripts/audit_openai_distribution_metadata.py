#!/usr/bin/env python3
"""Create a versioned, aggregate-only audit of scoring and API populations.

The program reads restricted rows only long enough to calculate numeric scores,
character lengths, and opaque task-group cardinalities.  It never writes or
prints essays, prompts, feedback, identifiers, candidate content, or API
credentials.  `average` is always derived from the three component scores;
there is no average model target in this audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from math import log2, sqrt
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/aihub_human_feedback_v1.json"
EVAL_TRAIN = ROOT / "eval/train.jsonl"
EVAL_VALIDATION = ROOT / "eval/validation.jsonl"
RESTRICTED_BATCHES = ROOT / "data/processed/restricted/openai_rationale_batches"
UTILIZATION_CONFIG = ROOT / "configs/openai_distribution_utilization.v1.json"
SCORE_AXES = ("content", "organization", "expression")
SCORE_BANDS = ("[1,2)", "[2,3)", "[3,4)", "[4,5]")
LENGTH_BINS = ((0, 200), (200, 400), (400, 800), (800, 1200), (1200, None))
REPORT_SCHEMA = "mal2026-openai-distribution-audit-v1"


class AuditError(ValueError):
    pass


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def score_band(value: Decimal) -> str:
    if not Decimal("1") <= value <= Decimal("5"):
        raise AuditError("component score is outside [1, 5]")
    return SCORE_BANDS[min(3, int(value) - 1)]


def length_bin(length: int) -> str:
    if length < 0:
        raise AuditError("negative text length")
    for lower, upper in LENGTH_BINS:
        if upper is None or length < upper:
            return f"[{lower},inf)" if upper is None else f"[{lower},{upper})"
    raise AssertionError("unreachable length bin")


@dataclass
class Population:
    name: str
    rows: int = 0
    score_axes: dict[str, Counter[str]] = field(default_factory=lambda: {axis: Counter() for axis in SCORE_AXES})
    score_joint: Counter[str] = field(default_factory=Counter)
    average_bands: Counter[str] = field(default_factory=Counter)
    lengths: Counter[str] = field(default_factory=Counter)
    task_groups: Counter[str] = field(default_factory=Counter)
    score_sums: dict[str, Decimal] = field(default_factory=lambda: {axis: Decimal(0) for axis in (*SCORE_AXES, "average")})

    def add(self, *, score: dict[str, Decimal], essay_length: int, task_group: str) -> None:
        average = sum((score[axis] for axis in SCORE_AXES), Decimal(0)) / Decimal(3)
        bands = []
        for axis in SCORE_AXES:
            band = score_band(score[axis])
            self.score_axes[axis][band] += 1
            self.score_sums[axis] += score[axis]
            bands.append(band)
        self.score_joint["|".join(bands)] += 1
        self.average_bands[score_band(average)] += 1
        self.score_sums["average"] += average
        self.lengths[length_bin(essay_length)] += 1
        self.task_groups[task_group] += 1
        self.rows += 1


def decimal_score(value: Any, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise AuditError(f"{field_name} must be numeric")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AuditError(f"{field_name} must be numeric") from exc
    if not parsed.is_finite() or not Decimal("1") <= parsed <= Decimal("5"):
        raise AuditError(f"{field_name} is outside [1, 5]")
    return parsed


def opaque_task_group(row: dict[str, Any]) -> str:
    """Use task metadata without serializing task text or identifiers."""
    prompt_num = row.get("prompt_num")
    if prompt_num is not None:
        return "prompt_num:" + sha256(str(prompt_num).encode()).hexdigest()
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise AuditError("row has no usable task/prompt metadata")
    normalized = " ".join(prompt.split())
    return "prompt_hash:" + sha256(normalized.encode()).hexdigest()


def read_population(name: str, paths: Iterable[Path]) -> Population:
    population = Population(name)
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise AuditError(f"required population file is unavailable: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise AuditError(f"blank row in {path.name}:{line_number}")
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("score"), dict) or not isinstance(row.get("essay"), str):
                    raise AuditError(f"invalid restricted row schema in {path.name}:{line_number}")
                score = {axis: decimal_score(row["score"].get(axis), f"score.{axis}") for axis in SCORE_AXES}
                population.add(score=score, essay_length=len(row["essay"]), task_group=opaque_task_group(row))
    if not population.rows:
        raise AuditError(f"population {name} is empty")
    return population


def proportions(counts: Counter[str], universe: Iterable[str], total: int) -> dict[str, float]:
    return {key: counts[key] / total for key in universe}


def divergence(left: Counter[str], right: Counter[str], *, left_total: int, right_total: int) -> dict[str, Any]:
    universe = sorted(set(left) | set(right))
    p, q = proportions(left, universe, left_total), proportions(right, universe, right_total)
    tvd = 0.5 * sum(abs(p[key] - q[key]) for key in universe)
    midpoint = {key: (p[key] + q[key]) / 2 for key in universe}
    def kl(values: dict[str, float], base: dict[str, float]) -> float:
        return sum(value * log2(value / base[key]) for key, value in values.items() if value)
    jsd = 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)
    overlap = sum(min(p[key], q[key]) for key in universe)
    hellinger = sqrt(0.5 * sum((sqrt(p[key]) - sqrt(q[key])) ** 2 for key in universe))
    zero_left = sum(1 for key in universe if left[key] == 0)
    zero_right = sum(1 for key in universe if right[key] == 0)
    return {
        "cell_count": len(universe), "total_variation_distance": tvd,
        "jensen_shannon_divergence_base2": jsd, "hellinger_distance": hellinger,
        "overlap_min_mass": overlap, "max_absolute_share_delta": max((abs(p[key] - q[key]) for key in universe), default=0.0),
        "zero_cells_left": zero_left, "zero_cells_right": zero_right,
        "sparse_cells_min_count_lt_20": sum(1 for key in universe if min(left[key], right[key]) < 20),
    }


def rendered_counts(counts: Counter[str], keys: Iterable[str]) -> dict[str, int]:
    return {key: int(counts[key]) for key in keys}


def population_report(population: Population) -> dict[str, Any]:
    return {
        "record_count": population.rows,
        "mean_scores": {axis: float(population.score_sums[axis] / population.rows) for axis in (*SCORE_AXES, "average")},
        "score_band_counts": {axis: rendered_counts(population.score_axes[axis], SCORE_BANDS) for axis in SCORE_AXES},
        "external_average_band_counts": rendered_counts(population.average_bands, SCORE_BANDS),
        "joint_score_stratum_counts": rendered_counts(population.score_joint, ("|".join(parts) for parts in __import__("itertools").product(SCORE_BANDS, repeat=3))),
        "essay_length_bin_counts": rendered_counts(population.lengths, (length_bin(lower) for lower, _ in LENGTH_BINS)),
        "task_prompt_metadata": {"opaque_group_count": len(population.task_groups), "available_for_all_rows": sum(population.task_groups.values()) == population.rows},
    }


def comparisons(populations: dict[str, Population], pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for left_name, right_name in pairs:
        left, right = populations[left_name], populations[right_name]
        result[f"{left_name}_to_{right_name}"] = {
            "content": divergence(left.score_axes["content"], right.score_axes["content"], left_total=left.rows, right_total=right.rows),
            "organization": divergence(left.score_axes["organization"], right.score_axes["organization"], left_total=left.rows, right_total=right.rows),
            "expression": divergence(left.score_axes["expression"], right.score_axes["expression"], left_total=left.rows, right_total=right.rows),
            "external_average": divergence(left.average_bands, right.average_bands, left_total=left.rows, right_total=right.rows),
            "joint_score_strata": divergence(left.score_joint, right.score_joint, left_total=left.rows, right_total=right.rows),
            "essay_length": divergence(left.lengths, right.lengths, left_total=left.rows, right_total=right.rows),
            "task_prompt_metadata": divergence(left.task_groups, right.task_groups, left_total=left.rows, right_total=right.rows),
        }
    return result


def api_candidate_coverage() -> dict[str, Any]:
    manifests = sorted(RESTRICTED_BATCHES.glob("*/manifest.json"))
    if not manifests:
        return {"status": "not_found"}
    # A report can make one aggregate claim only when there is one validated batch.
    manifest_path = manifests[-1]
    aggregate_path = manifest_path.with_name("validation_aggregate.json")
    if not aggregate_path.is_file():
        return {"status": "missing_validation_aggregate"}
    manifest, aggregate = json.loads(manifest_path.read_text()), json.loads(aggregate_path.read_text())
    splits = manifest.get("splits", {})
    per_essay = manifest.get("candidates_per_essay")
    if not isinstance(splits, dict) or not isinstance(per_essay, int):
        raise AuditError("API candidate manifest has invalid aggregate split metadata")
    expected = sum(splits.values()) * per_essay
    qc_fields = ("candidate_duplicate_records", "candidate_unknown_records", "candidate_schema_or_grounding_invalid_records", "mapping_duplicate_records", "missing_records", "rejected_records")
    judge_manifests = list(manifest_path.parent.glob("judge_runs/*/manifest.json"))
    judge_v3_passed = any(
        str(json.loads(path.read_text(encoding="utf-8")).get("schema_version", "")).startswith("qwen36-gguf-judge-v3")
        and json.loads(path.read_text(encoding="utf-8")).get("pilot_passed_hard_gates") is True
        for path in judge_manifests
    )
    return {
        "status": "aggregate_manifest_available", "expected_candidate_count": expected,
        "accepted_candidate_count": aggregate.get("accepted_records"), "candidate_availability_rate": aggregate.get("accepted_records", 0) / expected,
        "source_split_counts": {key: int(value) for key, value in splits.items()}, "candidates_per_source": per_essay,
        "strict_quality_control_status": aggregate.get("status"),
        "quality_control_failure_counts": {key: aggregate.get(key) for key in qc_fields},
        "restricted_train_only_artifact_present": any(manifest_path.parent.glob("derived/*/candidates.train.manifest.json")),
        "judge_v3_passed": judge_v3_passed,
        "selection_or_sft_authorized": False,
        "note": "availability/QC is not a selection decision and does not prove membership beyond recorded aggregate lineage",
    }


def build_report() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    required = ("refit_train", "selection_train", "selection_dev")
    if any(key not in files for key in required):
        raise AuditError("canonical manifest lacks a prepared split")
    prepared_paths = {key: ROOT / "data/processed/aihub_human_feedback_v1" / files[key]["filename"] for key in required}
    api_train = read_population("api_train", (EVAL_TRAIN,))
    api_validation = read_population("api_validation", (EVAL_VALIDATION,))
    api_source = read_population("api_source_union", (EVAL_TRAIN, EVAL_VALIDATION))
    prepared = {"prepared_source": read_population("prepared_source", (prepared_paths["refit_train"],)), "prepared_train": read_population("prepared_train", (prepared_paths["selection_train"],)), "prepared_dev": read_population("prepared_dev", (prepared_paths["selection_dev"],))}
    api = {"api_source_union": api_source, "api_train": api_train, "api_validation": api_validation}
    return {
        "schema_version": REPORT_SCHEMA, "status": "completed", "privacy": "aggregate_only_no_essays_explanations_prompts_source_ids_or_api_keys",
        "fixed_binning": {"component_score_bands": list(SCORE_BANDS), "joint_score_strata": "cartesian_product_of_three_component_bands", "essay_character_bins": ["[0,200)", "[200,400)", "[400,800)", "[800,1200)", "[1200,inf)"], "average_policy": "computed_outside_score_model_from_unrounded_content_organization_expression"},
        "confidence_count_guardrails": {"publish_slice_rate_or_metric_min_n": 20, "selection_calibration_base_min_n": 30, "selection_calibration_accepted_min_n": 10, "sparse_cell_definition": "count_lt_20", "zero_support_selection": "prohibited"},
        "populations": {"api": {key: population_report(value) for key, value in api.items()}, "prepared": {key: population_report(value) for key, value in prepared.items()}},
        "divergence_diagnostics": {"definition": "TVD=0.5*L1; JSD uses base-2 midpoint and zero terms contribute zero; overlap=min-mass; task groups are only opaque in-memory keys", "api": comparisons(api, (("api_source_union", "api_train"), ("api_source_union", "api_validation"), ("api_train", "api_validation"))), "prepared": comparisons(prepared, (("prepared_source", "prepared_train"), ("prepared_source", "prepared_dev"), ("prepared_train", "prepared_dev")))},
        "api_candidate_availability_and_qc": api_candidate_coverage(),
        "provenance": {"audit_script_sha256": digest(Path(__file__)), "utilization_config_sha256": digest(UTILIZATION_CONFIG), "canonical_manifest_sha256": digest(MANIFEST), "eval_train_sha256": digest(EVAL_TRAIN), "eval_validation_sha256": digest(EVAL_VALIDATION), "api_source_union_is_train_plus_validation": True, "cross_population_comparisons": "not_reported_without_a_deterministic_provenance_join"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/reports/openai_distribution_audit_v1.json")
    args = parser.parse_args()
    if args.output.resolve().parent != (ROOT / "data/reports").resolve():
        raise SystemExit("output must be a direct file under data/reports")
    report = build_report()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "api_train_records": report["populations"]["api"]["api_train"]["record_count"], "api_validation_records": report["populations"]["api"]["api_validation"]["record_count"], "selection_or_sft_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
