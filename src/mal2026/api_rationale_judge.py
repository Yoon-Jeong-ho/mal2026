"""Score generated all-axis rationales with the fixed v6 pointwise judge."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterator, Mapping

from .api_rationale_data import (
    AXES, RESTRICTED_ROOT, ROOT, APIRationaleContractError, aggregate_input_provenance,
    load_generated_rationales, load_writing_rows, sha256_file,
)
from .api_rationale_sft import SUPPORTED_MODELS


JUDGE_OUTPUT_ROOT = RESTRICTED_ROOT / "decoder_judge_v1"
API_BASELINE_REPORT = RESTRICTED_ROOT / "frozen_validation_judge_runs_rationale_only_score5x10_v6" / "qwen36-native-fp8-rationale-only-score5x10-v6-validation-20260720-full-001" / "aggregate_score_report.json"


class APIRationaleJudgeError(APIRationaleContractError):
    """Raised when generated-rationale judging is not bound to the fixed v6 protocol."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise APIRationaleJudgeError(message)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _load_judge() -> Any:
    target = ROOT / "scripts" / "score_rationale_distribution_vllm_dp4.py"
    spec = importlib.util.spec_from_file_location("mal2026_fixed_v6_judge", target)
    _need(spec is not None and spec.loader is not None, "fixed v6 judge module is unavailable")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


JUDGE = _load_judge()


@dataclass(frozen=True)
class APIRationaleJudgeConfig:
    schema_version: str
    run_id: str
    base_key: str
    system_kind: str
    generation_dir: str
    output_dir: str
    source: str
    client_max_inflight: int

    @classmethod
    def from_json(cls, path: Path) -> "APIRationaleJudgeConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        _need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "judge config has unknown or missing fields")
        value = cls(**raw); value.validate(); return value

    def validate(self) -> None:
        _need(self.schema_version == "mal2026-api-rationale-judge-v1", "judge config schema differs")
        _need(self.base_key in SUPPORTED_MODELS and self.system_kind in {"bundle", "axis_triplet"}, "judge system identity differs")
        _need(self.source == "validation", "judge must use frozen validation only")
        root = Path(self.generation_dir); output = Path(self.output_dir)
        _need(root.is_absolute() and root.parent == (RESTRICTED_ROOT / "decoder_generation_v1").resolve(), "judge generation root differs")
        _need(output.is_absolute() and output.parent == JUDGE_OUTPUT_ROOT.resolve() and not output.exists(), "judge output must be a fresh restricted direct child")
        # `-001` was preserved after the judge server failed during setup and
        # recorded no usable score distribution.  The fresh `-002` lineage is
        # required because observation files are append-only and never reused.
        expected = f"api-rationale-judge-v1-{self.base_key}-{self.system_kind}-validation-002"
        _need(self.run_id == expected and output.name == expected, "judge run lineage differs")
        _need(self.client_max_inflight == 768, "judge client concurrency differs from fixed v6")


def _baseline() -> Mapping[str, Any]:
    try: value = json.loads(API_BASELINE_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise APIRationaleJudgeError("fixed API baseline report is unavailable") from exc
    _need(isinstance(value, dict) and value.get("status") == "passed" and all(value.get("hard_gates", {}).values()), "fixed API baseline did not pass")
    analysis = value.get("prompt_type_analysis")
    _need(isinstance(analysis, dict) and len(analysis) == 5, "fixed API baseline prompt analysis differs")
    means = {axis: statistics.fmean(float(item["axis_means"][axis]) for item in analysis.values()) for axis in AXES}
    return {"aggregate_score_report_sha256": _sha(API_BASELINE_REPORT), "candidate_count": value.get("counts", {}).get("candidates"), "axis_means": means,
            "macro_mean": statistics.fmean(means.values()), "prompt_type_analysis": analysis}


def _entries(config: APIRationaleJudgeConfig) -> list[dict[str, Any]]:
    generated = load_generated_rationales(
        Path(config.generation_dir), source="validation", task=config.system_kind
    )
    rows = load_writing_rows("validation", include_scores=False)
    _need(len(generated) == len(rows) == 400, "judge population count differs")
    entries = []
    for row in rows:
        diagnoses = generated.get(row.identifier)
        _need(diagnoses is not None and set(diagnoses) == set(AXES), "generated rationale/source linkage differs")
        entries.append({"opaque_generation_key": JUDGE.opaque(config.run_id, row.identifier), "sentences": JUDGE.sentence_list(row.essay),
                        "rationale": {"schema_version": "rationale-only-v1", **{axis: {"rationale": diagnoses[axis]} for axis in AXES}}})
    _need(all(entry["sentences"] for entry in entries), "judge essay sentence segmentation is empty")
    return entries


def _task_stream(config: APIRationaleJudgeConfig, model: str, entries: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    cfg = JUDGE.config()
    for entry in entries:
        for prompt_index, prompt_type in enumerate(cfg["protocol"]["prompt_types"]):
            for seed_index, seed in enumerate(cfg["sampling"]["seeds"]):
                yield {
                    "opaque_request_key": JUDGE.opaque(config.run_id, entry["opaque_generation_key"], prompt_index, seed_index),
                    "opaque_generation_key": entry["opaque_generation_key"], "prompt_type_id": prompt_type["id"], "sampling_seed": seed,
                    "response_contract": cfg["protocol"]["response_contract"],
                    "body": JUDGE.request_body(cfg, model, entry, list(AXES), prompt_type["layout"], seed, prompt_type["review_emphasis"]),
                }


def _summarize(path: Path, expected: int, baseline: Mapping[str, Any]) -> dict[str, Any]:
    rows = [json.loads(line) for line in (path / "score_observations.jsonl").open(encoding="utf-8") if line.strip()]
    _need(len(rows) == expected, "judge observations are incomplete")
    failures = Counter(str(row["failure_category"]) for row in rows if row.get("failure_category"))
    valid = [row for row in rows if row.get("scored")]
    per_axis = {axis: statistics.fmean(float(row["scores"][axis]) for row in valid) if valid else None for axis in AXES}
    histograms = {axis: dict(sorted(Counter(int(row["scores"][axis]) for row in valid).items())) for axis in AXES}
    prompt_analysis: dict[str, Any] = {}
    for prompt in sorted({row.get("prompt_type_id") for row in rows}):
        subset = [row for row in valid if row.get("prompt_type_id") == prompt]
        prompt_analysis[str(prompt)] = {"observations": len(subset), "axis_means": {axis: statistics.fmean(float(row["scores"][axis]) for row in subset) if subset else None for axis in AXES},
                                      "macro_mean": statistics.fmean(statistics.fmean(float(row["scores"][axis]) for axis in AXES) for row in subset) if subset else None}
    ranges = {axis: max(float(item["axis_means"][axis]) for item in prompt_analysis.values()) - min(float(item["axis_means"][axis]) for item in prompt_analysis.values()) for axis in AXES} if prompt_analysis else {axis: None for axis in AXES}
    hard_gates = {"complete_observations": len(rows) == expected, "zero_transport_or_schema_failures": not failures, "all_scores_valid": len(valid) == expected,
                  "five_prompt_forms": len(prompt_analysis) == 5}
    score_mean = {axis: round(float(per_axis[axis]), 6) if per_axis[axis] is not None else None for axis in AXES}
    return {"status": "completed" if all(hard_gates.values()) else "failed_gates", "counts": {"expected_calls": expected, "observations": len(rows), "scored": len(valid),
            "schema_valid": sum(bool(row.get("schema_valid")) for row in rows), "abstain": sum(bool(row.get("abstain")) for row in rows), "generated_candidates": expected // 50},
            "hard_gates": hard_gates, "failure_categories": dict(sorted(failures.items())), "axis_means": score_mean,
            "macro_mean": round(statistics.fmean(score_mean.values()), 6) if all(value is not None for value in score_mean.values()) else None,
            "score_histograms": histograms, "prompt_type_analysis": prompt_analysis, "prompt_type_axis_ranges": {axis: round(float(value), 6) if value is not None else None for axis, value in ranges.items()},
            "api_baseline": baseline, "delta_from_api_baseline": {axis: round(score_mean[axis] - float(baseline["axis_means"][axis]), 6) for axis in AXES} if all(value is not None for value in score_mean.values()) else None,
            "raw_prompts_or_responses_persisted": False, "selection_artifact_constructed": False}


def run_api_rationale_judge(config: APIRationaleJudgeConfig, endpoint: str, server_attestation: Path, model: str) -> dict[str, Any]:
    """Run the unchanged v6 prompt/seed protocol on 400 generated rationales."""
    config.validate(); cfg = JUDGE.config(); JUDGE.validate_server(server_attestation, endpoint, cfg, "full")
    entries, baseline = _entries(config), _baseline(); expected = len(entries) * len(cfg["protocol"]["prompt_types"]) * len(cfg["sampling"]["seeds"])
    output = Path(config.output_dir); output.mkdir(mode=0o700, parents=True)
    manifest = {"schema_version": config.schema_version, "status": "running", "run_id": config.run_id, "config": asdict(config), "expected_calls": expected,
                "fixed_v6_config_sha256": JUDGE.sha(JUDGE.CONFIG_PATH), "server_attestation_sha256": _sha(server_attestation), "api_baseline": baseline,
                "input_provenance": aggregate_input_provenance(), "source_writing_scores_read_or_prompted": False, "candidate_scores_read_or_prompted": False,
                "raw_prompts_or_responses_persisted": False}
    _atomic_json(output / "manifest.json", manifest)
    observations = output / "score_observations.jsonl"; pending: set[Any] = set(); stream = iter(_task_stream(config, model, entries)); exhausted = False
    with observations.open("x", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=config.client_max_inflight) as pool:
        while pending or not exhausted:
            while not exhausted and len(pending) < config.client_max_inflight:
                try: task = next(stream)
                except StopIteration: exhausted = True; break
                pending.add(pool.submit(JUDGE.call, endpoint, task))
            if not pending: continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done: handle.write(json.dumps(future.result(), ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
    report = _summarize(output, expected, baseline)
    report.update({"schema_version": config.schema_version, "run_id": config.run_id, "base_key": config.base_key, "system_kind": config.system_kind,
                   "fixed_v6_config_sha256": JUDGE.sha(JUDGE.CONFIG_PATH), "server_attestation_sha256": _sha(server_attestation), "input_provenance": aggregate_input_provenance()})
    _atomic_json(output / "aggregate_judge_report.json", report)
    manifest["status"] = report["status"]; manifest["aggregate_report_sha256"] = _sha(output / "aggregate_judge_report.json")
    _atomic_json(output / "manifest.json", manifest)
    if not all(report["hard_gates"].values()):
        raise APIRationaleJudgeError("generated-rationale judge hard gate failed")
    return report
