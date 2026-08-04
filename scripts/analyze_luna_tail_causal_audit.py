#!/usr/bin/env python3
"""Create case-clustered, aggregate-only analysis of the Luna tail audit."""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from setproctitle import getproctitle, setproctitle

from run_luna_tail_causal_audit import AXES, CAUSES, file_sha, jsonl_rows, parse_batch_record, response_text


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "luna-r0-tail-causal-audit-v1-20260804-002"
RESTRICTED = ROOT / "data/processed/restricted/luna_tail_causal_audit_v1" / RUN_ID
PUBLIC = ROOT / "outputs/luna-tail-causal-audit-v1" / RUN_ID
OOF = ROOT / "data/processed/restricted/r0_exact_oof_v1/r0-exact-oof-20260731-002/merged/oof_predictions.jsonl"
TRAIN = ROOT / "eval/train.jsonl"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    need(not path.exists(), f"refusing to overwrite {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def average(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rmse(values: Sequence[float]) -> float | None:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else None


def quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    need(bool(ordered), "empty quantiles")
    def q(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper: return float(ordered[lower])
        return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))
    return {"min": float(ordered[0]), "q25": q(.25), "median": q(.5), "q75": q(.75), "max": float(ordered[-1]), "mean": float(sum(ordered) / len(ordered))}


def wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0: return None
    z = 1.959963984540054
    p = successes / total; d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / d
    return [max(0.0, center - half), min(1.0, center + half)]


def load_results() -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    manifest = json.loads((RESTRICTED / "manifest.json").read_text(encoding="utf-8"))
    need(manifest.get("status") == "validated", "audit is not validated")
    mappings = {row["custom_id"]: row for row in jsonl_rows(RESTRICTED / "source_map.jsonl")}
    retries = {row["custom_id"]: row["response_body"] for row in jsonl_rows(RESTRICTED / "retry_outputs.jsonl")}
    parsed = []
    for row in jsonl_rows(RESTRICTED / "batch_output.jsonl"):
        custom_id = row["custom_id"]
        result = json.loads(response_text(retries[custom_id])) if custom_id in retries else parse_batch_record(row)
        parsed.append((mappings[custom_id], result))
    need(len(parsed) == len(mappings) == 6016, "analysis coverage differs")
    return parsed, manifest


def baseline_summary() -> tuple[dict[str, Any], dict[tuple[str, str], list[dict[str, Any]]]]:
    sources = {str(row["id"]): row for row in jsonl_rows(TRAIN)}
    rows = jsonl_rows(OOF); need(len(sources) == len(rows) == 2000, "baseline population differs")
    by_prompt_low: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    result: dict[str, Any] = {}
    for axis in AXES:
        confusion = Counter(); predictions = Counter(); primary = []
        for row in rows:
            gold = int(row["half_up_integer_prediction"][axis] * 0 + round(float(row["reference_score"][axis]) + 1e-12))
            # The canonical data uses .05 increments; explicit half-up is encoded by the OOF reference projection.
            from decimal import Decimal, ROUND_HALF_UP
            gold = int(Decimal(str(row["reference_score"][axis])).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            pred = int(row["half_up_integer_prediction"][axis])
            confusion[(gold, pred)] += 1; predictions[pred] += 1
            if gold in (1, 2):
                item = {"source_id": str(row["source_id"]), "prompt_num": str(sources[str(row["source_id"])]["prompt_num"]), "gold": gold, "pred": pred, "gold_raw": float(row["reference_score"][axis]), "pred_raw": float(row["continuous_prediction"][axis])}
                by_prompt_low[(axis, item["prompt_num"])].append(item)
                if pred in (3, 4): primary.append(item)
        low_total = sum(value for (gold, _), value in confusion.items() if gold in (1, 2))
        low_central = sum(value for (gold, pred), value in confusion.items() if gold in (1, 2) and pred in (3, 4))
        high_total = sum(value for (gold, _), value in confusion.items() if gold == 5)
        high_central = sum(value for (gold, pred), value in confusion.items() if gold == 5 and pred in (3, 4))
        result[axis] = {"gold_distribution": {str(score): sum(value for (gold, _), value in confusion.items() if gold == score) for score in range(1, 6)}, "prediction_distribution": {str(score): predictions[score] for score in range(1, 6)}, "low_1_2_to_3_4": {"count": low_central, "denominator": low_total, "rate": low_central / low_total}, "score5_to_3_4": {"count": high_central, "denominator": high_total, "rate": high_central / high_total}, "primary_gold_raw": quantiles([row["gold_raw"] for row in primary]), "primary_r0_raw": quantiles([row["pred_raw"] for row in primary]), "primary_mean_signed_error": average([row["pred_raw"] - row["gold_raw"] for row in primary])}
    return result, by_prompt_low


def main() -> None:
    setproctitle("mal2026:luna-tail-audit:case-analysis")
    need(getproctitle() == "mal2026:luna-tail-audit:case-analysis", "process title differs")
    parsed, manifest = load_results()
    baseline, by_prompt_low = baseline_summary()
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for mapping, result in parsed:
        grouped[(mapping["source_id"], mapping["axis"], mapping["stratum"], mapping["condition"])].append(result)
    blind_case: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for key, values in grouped.items():
        source_id, axis, stratum, condition = key
        if condition not in ("canonical_blind", "operational_blind"): continue
        scores = [int(value["predicted_score"]) for value in values]
        need(len(scores) == 3, "blind repetition count differs")
        blind_case[key] = {"score": int(median(scores)), "repeat_exact": len(set(scores)) == 1, "spread": max(scores) - min(scores)}
    case_summaries: dict[str, Any] = {}
    for stratum in ("primary_low_to_central", "low_control", "center_control", "high_to_central"):
        for axis in AXES:
            mappings = {}
            for mapping, _ in parsed:
                if mapping["stratum"] == stratum and mapping["axis"] == axis:
                    mappings[mapping["source_id"]] = mapping
            for condition in ("canonical_blind", "operational_blind"):
                cases = [(mapping, blind_case[(source_id, axis, stratum, condition)]) for source_id, mapping in mappings.items()]
                scores = [case["score"] for _, case in cases]; low = sum(score in (1, 2) for score in scores); central = sum(score in (3, 4) for score in scores)
                case_summaries[f"{stratum}:{axis}:{condition}"] = {"cases": len(cases), "median_score_distribution": {str(score): scores.count(score) for score in range(1, 6)}, "mean_median_score": average(scores), "rmse_to_raw_human_score": rmse([case["score"] - mapping["gold_raw"] for mapping, case in cases]), "human_band_agreement": average([int(case["score"] == mapping["gold_band"]) for mapping, case in cases]), "r0_band_agreement": average([int(case["score"] == mapping["pred_band"]) for mapping, case in cases]), "low_rate": low / len(cases), "low_rate_wilson95": wilson(low, len(cases)), "central_rate": central / len(cases), "central_rate_wilson95": wilson(central, len(cases)), "repeat_exact_agreement_rate": average([int(case["repeat_exact"]) for _, case in cases]), "mean_repeat_spread": average([case["spread"] for _, case in cases])}
            canonical = [blind_case[(source_id, axis, stratum, "canonical_blind")]["score"] for source_id in mappings]
            operational = [blind_case[(source_id, axis, stratum, "operational_blind")]["score"] for source_id in mappings]
            central_to_low = sum(c in (3, 4) and o in (1, 2) for c, o in zip(canonical, operational, strict=True))
            case_summaries[f"{stratum}:{axis}:paired_prompt_effect"] = {"cases": len(canonical), "operational_minus_canonical_mean": average([o - c for c, o in zip(canonical, operational, strict=True)]), "operational_lower_rate": average([int(o < c) for c, o in zip(canonical, operational, strict=True)]), "same_rate": average([int(o == c) for c, o in zip(canonical, operational, strict=True)]), "operational_higher_rate": average([int(o > c) for c, o in zip(canonical, operational, strict=True)]), "canonical_central_to_operational_low": central_to_low, "canonical_central_to_operational_low_rate": central_to_low / len(canonical)}
    causal_primary: dict[str, Any] = {}
    rationale_names = ("r0_input", "official_blind", "official_conditioned", "final_dpo", "shuffled_r0_input")
    for axis in AXES:
        cases = [values for (source_id, item_axis, stratum, condition), values in grouped.items() if item_axis == axis and stratum == "primary_low_to_central" and condition == "revealed_causal"]
        need(all(len(values) == 2 for values in cases), "causal repetition count differs")
        cause_both = {cause: sum(all(value["causes"][cause] for value in values) for values in cases) for cause in CAUSES}
        cause_either = {cause: sum(any(value["causes"][cause] for value in values) for values in cases) for cause in CAUSES}
        causal_primary[axis] = {"cases": len(cases), "primary_cause_exact_repeat_agreement_rate": average([int(values[0]["primary_cause"] == values[1]["primary_cause"]) for values in cases]), "primary_cause_observation_votes": dict(Counter(value["primary_cause"] for values in cases for value in values)), "cause_both_repeats_rate": {cause: cause_both[cause] / len(cases) for cause in CAUSES}, "cause_either_repeat_rate": {cause: cause_either[cause] / len(cases) for cause in CAUSES}, "preferred_score_repeat_agreement_rate": average([int(values[0]["preferred_score"] == values[1]["preferred_score"]) for values in cases]), "rationale": {name: {"mean_faithfulness": average([value["rationales"][name]["faithfulness"] for values in cases for value in values]), "mean_specificity": average([value["rationales"][name]["specificity"] for values in cases for value in values]), "positive_polarity_rate": average([int(value["rationales"][name]["polarity"] == "positive") for values in cases for value in values]), "supports_human_rate": average([int(value["rationales"][name]["supports_human"]) for values in cases for value in values]), "supports_r0_rate": average([int(value["rationales"][name]["supports_r0"]) for values in cases for value in values])} for name in rationale_names}}
    prompt_summary: dict[str, Any] = {}
    for axis in AXES:
        for prompt_num in sorted({prompt for item_axis, prompt in by_prompt_low if item_axis == axis}):
            low_rows = by_prompt_low[(axis, prompt_num)]; errors = [row for row in low_rows if row["pred"] in (3, 4)]
            selected_ids = {row["source_id"] for row in errors}
            canonical = [case["score"] for (source_id, item_axis, stratum, condition), case in blind_case.items() if source_id in selected_ids and item_axis == axis and stratum == "primary_low_to_central" and condition == "canonical_blind"]
            operational = [case["score"] for (source_id, item_axis, stratum, condition), case in blind_case.items() if source_id in selected_ids and item_axis == axis and stratum == "primary_low_to_central" and condition == "operational_blind"]
            prompt_summary[f"{axis}:{prompt_num}"] = {"gold_low_cases": len(low_rows), "r0_low_to_central": len(errors), "r0_low_to_central_rate": len(errors) / len(low_rows), "luna_canonical_low_rate_on_r0_errors": average([int(score in (1, 2)) for score in canonical]), "luna_operational_low_rate_on_r0_errors": average([int(score in (1, 2)) for score in operational])}
    result = {"schema_version": "mal2026-luna-tail-causal-case-analysis-v1", "status": "completed", "run_id": RUN_ID, "batch_id": manifest["batch_id"], "requests": manifest["requests"], "valid_observations": 6016, "axis_cases": manifest["population"]["selected_axis_cases"], "source_hashes": {"exact_r0_oof": file_sha(OOF), "canonical_train": file_sha(TRAIN), "batch_output": manifest["output_sha256"], "retry_output": manifest["retry_output_sha256"]}, "baseline": baseline, "case_summaries": case_summaries, "causal_primary": causal_primary, "prompt_summary": prompt_summary, "interpretation_limits": ["Luna is a model auditor, not an independent human ground truth.", "Blind conditions support score-disagreement diagnosis; revealed-causal responses may anchor on the displayed human/reference score.", "Repeated Luna calls on one case are clustered, not independent observations.", "The audit estimates causes and prompt sensitivity; it does not authorize training or relabeling."], "validation_rows_loaded": False, "average_target_used": False, "privacy": "aggregate_only_no_essay_rationale_prompt_text_identifier_or_row_prediction"}
    PUBLIC.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = PUBLIC / "case_analysis.json"; atomic_json(path, result)
    print(json.dumps({"status": "completed", "path": str(path.relative_to(ROOT)), "sha256": file_sha(path), "axis_cases": result["axis_cases"], "observations": result["valid_observations"]}))


if __name__ == "__main__":
    main()
