"""Frozen-v6 generation and aggregate-only judging for RLAIF continuations."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import statistics
from typing import Any, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .api_rationale_data import AXES, RESTRICTED_ROOT, ROOT, APIRationaleContractError, axes_for_task, decoder_messages, load_generated_rationales, load_writing_rows, parse_rationale_output, rationale_object, sha256_file
from .api_rationale_sft import SUPPORTED_MODELS
from .rlaif_grpo import CONFIG_PATH, JUDGE, RLAIFGRPOError, RLAIFSettings, canonical_completion


_EVALUATION_STUDY = RLAIFSettings.from_json().schema_version.rsplit("-", 1)[-1]
EVALUATION_ROOT = RESTRICTED_ROOT / f"rlaif_grpo_{_EVALUATION_STUDY}"


class RLAIFEvaluationError(APIRationaleContractError):
    """Raised when a post-RL generation/judge result is not frozen-v6 comparable."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RLAIFEvaluationError(message)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class RLAIFEvaluationConfig:
    schema_version: str
    run_id: str
    base_key: str
    task: str
    arm: str
    rl_phase: str
    rl_training_dir: str
    output_dir: str
    generation_adapter_alias: str
    baseline_kind: str
    baseline_generation_dir: str
    baseline_judge_dir: str
    deterministic_max_new_tokens: int
    character_limit: int

    @classmethod
    def from_json(cls, path: Path) -> "RLAIFEvaluationConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        _need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "RLAIF evaluation config has unknown or missing fields")
        value = cls(**raw); value.validate(); return value

    def validate(self) -> None:
        settings = RLAIFSettings.from_json()
        expected_schema = f"mal2026-rlaif-grpo-evaluation-{settings.schema_version.rsplit('-', 1)[-1]}"
        _need(self.schema_version == expected_schema, "RLAIF evaluation schema differs")
        _need(self.base_key in SUPPORTED_MODELS and self.task in {"bundle", *AXES} and self.arm in settings.arms, "evaluation system identity differs")
        _need(self.rl_phase in ({"full", "pilot"} if settings.schema_version.endswith(("-v3", "-v4", "-v5", "-v6", "-v7", "-v8")) else {"full"}), "evaluation policy phase differs")
        suffix = "validation-001" if self.rl_phase == "full" else "pilot-validation-001"
        _need(self.run_id == f"{settings.run_id_prefix}{self.base_key}-{self.task}-{self.arm}-{suffix}", "evaluation run id differs")
        output = Path(self.output_dir)
        _need(output.is_absolute() and output.parent == EVALUATION_ROOT.resolve() and not output.is_symlink(), "evaluation output root differs")
        expected_kind = "bundle" if self.task == "bundle" else "axis_triplet"
        _need(self.baseline_kind == expected_kind, "baseline kind differs from task contract")
        _need(self.deterministic_max_new_tokens == (512 if self.task == "bundle" else 192) and self.character_limit == 192, "deterministic evaluation decode contract differs")
        training_dir = Path(self.rl_training_dir)
        record = training_dir / "training_complete.json"
        _need(training_dir.is_dir() and record.is_file() and not training_dir.is_symlink(), "completed RLAIF training directory is unavailable")
        training = json.loads(record.read_text(encoding="utf-8"))
        _need(training.get("status") == "completed" and all(training.get(key) == expected for key, expected in (("base_key", self.base_key), ("task", self.task), ("arm", self.arm), ("phase", self.rl_phase))), "RLAIF training provenance differs")
        adapter = training_dir / "adapter"
        _need(adapter.is_dir() and (adapter / "adapter_config.json").is_file(), "completed RLAIF adapter is unavailable")
        _need(self.generation_adapter_alias == f"rlaif_{self.base_key}_{self.task}_{self.arm}", "generation adapter alias differs")
        for path in (Path(self.baseline_generation_dir), Path(self.baseline_judge_dir)):
            _need(path.is_dir() and not path.is_symlink(), "baseline restricted artifact is unavailable")
        baseline = json.loads((Path(self.baseline_judge_dir) / "aggregate_judge_report.json").read_text(encoding="utf-8"))
        _need(baseline.get("status") == "completed" and baseline.get("base_key") == self.base_key and baseline.get("system_kind") == self.baseline_kind and all(baseline.get("hard_gates", {}).values()), "baseline judge report is not fixed-v6 comparable")
        _need(baseline.get("fixed_v6_config_sha256") == settings.fixed_v6_config_sha256, "baseline judge config differs")
        generation = json.loads((Path(self.baseline_generation_dir) / "aggregate_generation_report.json").read_text(encoding="utf-8"))
        _need(generation.get("status") == "completed" and generation.get("source") == "validation" and generation.get("task") == self.baseline_kind and all(generation.get("hard_gates", {}).values()), "baseline generation report differs")


def _response_schema(axes: Sequence[str], character_limit: int) -> dict[str, Any]:
    fields: dict[str, Any] = {"schema_version": {"type": "string", "const": "rationale-only-v1"}}
    for axis in axes:
        fields[str(axis)] = {"type": "object", "properties": {"rationale": {"type": "string", "minLength": 1, "maxLength": character_limit}}, "required": ["rationale"], "additionalProperties": False}
    return {"type": "object", "properties": fields, "required": ["schema_version", *axes], "additionalProperties": False}


def _request(endpoint: str, body: Mapping[str, Any]) -> tuple[str | None, str | None]:
    wire = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    try:
        with urlopen(Request(endpoint + "/v1/chat/completions", data=wire, headers={"Content-Type": "application/json"}, method="POST"), timeout=180) as response:
            outer = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return None, "http_429" if exc.code == 429 else ("http_5xx" if 500 <= exc.code <= 599 else "http_4xx")
    except (URLError, TimeoutError):
        return None, "connection_or_timeout"
    except json.JSONDecodeError:
        return None, "outer_json"
    if not isinstance(outer, dict) or not isinstance(outer.get("choices"), list) or len(outer["choices"]) != 1:
        return None, "envelope"
    choice = outer["choices"][0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop" or not isinstance(choice.get("message"), dict):
        return None, "finish"
    content = choice["message"].get("content")
    return (content, None) if isinstance(content, str) else (None, "missing_content")


def _generation_server(config: RLAIFEvaluationConfig, endpoint: str, attestation_path: Path) -> Mapping[str, Any]:
    parsed = urlparse(endpoint)
    _need(parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port is not None, "generation endpoint must be local HTTP")
    _need(attestation_path.is_file() and not attestation_path.is_symlink(), "generation attestation is unavailable")
    value = json.loads(attestation_path.read_text(encoding="utf-8")); model_id, revision = SUPPORTED_MODELS[config.base_key]
    training_adapter = Path(config.rl_training_dir) / "adapter"
    expected = {
        "schema_version": "mal2026-rlaif-grpo-generation-server-attestation-v1", "server_host": "127.0.0.1", "server_port": parsed.port,
        "physical_gpus": [0, 1, 2, 3], "tensor_parallel_size": 4, "max_model_len": 3072,
        "model_id": model_id, "model_revision": revision, "adapter_path": str(training_adapter.resolve()),
        "adapter_alias": config.generation_adapter_alias, "server_process_environment_verified": True,
    }
    _need(isinstance(value, dict) and all(value.get(key) == expected_value for key, expected_value in expected.items()), "generation server attestation differs")
    return value


def _generation_tasks(config: RLAIFEvaluationConfig) -> Iterator[dict[str, Any]]:
    axes = axes_for_task(config.task); schema = _response_schema(axes, config.character_limit)
    for row in load_writing_rows("validation", include_scores=False):
        yield {
            "source_id": row.identifier,
            "body": {"model": config.generation_adapter_alias, "temperature": 0.0, "top_p": 1.0, "max_tokens": config.deterministic_max_new_tokens,
                     "messages": decoder_messages(row, axes),
                     "response_format": {"type": "json_schema", "json_schema": {"name": "mal2026_rationale_only_v1", "strict": True, "schema": schema}}},
        }


def _baseline_triplet(config: RLAIFEvaluationConfig) -> Mapping[str, Mapping[str, str]]:
    if config.task == "bundle":
        return {}
    value = load_generated_rationales(Path(config.baseline_generation_dir), source="validation", task="axis_triplet")
    _need(len(value) == 400 and all(set(item) == set(AXES) for item in value.values()), "baseline axis triplet cannot be used for hybrid comparison")
    return value


def generate_validation(config: RLAIFEvaluationConfig, endpoint: str, server_attestation: Path) -> dict[str, Any]:
    """Generate one deterministic post-RL output per frozen validation essay."""
    config.validate(); attestation = _generation_server(config, endpoint, server_attestation)
    output = Path(config.output_dir); _need(not output.exists(), "RLAIF evaluation output already exists")
    output.mkdir(mode=0o700, parents=True)
    tasks = list(_generation_tasks(config)); _need(len(tasks) == 400, "frozen validation population differs")
    baseline_triplet = _baseline_triplet(config); axes = axes_for_task(config.task)
    manifest = {
        "schema_version": config.schema_version, "status": "generation_running", "run_id": config.run_id, "config": asdict(config),
        "expected_records": len(tasks), "rlaif_training_complete_sha256": _sha(Path(config.rl_training_dir) / "training_complete.json"),
        "generation_server_attestation_sha256": _sha(server_attestation), "rlaif_config_sha256": _sha(CONFIG_PATH),
        "source_writing_scores_read_or_prompted": False, "candidate_scores_read_or_prompted": False,
        "raw_prompts_or_completions_tracked": False,
    }
    _atomic_json(output / "manifest.json", manifest)
    pending: set[Any] = set(); iterator = iter(tasks); exhausted = False; failures: Counter[str] = Counter(); records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=768) as pool:
        future_tasks: dict[Any, dict[str, Any]] = {}
        while pending or not exhausted:
            while not exhausted and len(pending) < 768:
                try:
                    task = next(iterator)
                except StopIteration:
                    exhausted = True; break
                future = pool.submit(_request, endpoint, task["body"])
                pending.add(future); future_tasks[future] = task
            if not pending:
                continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = future_tasks.pop(future, None)
                _need(task is not None, "generation future lost private task routing")
                content, failure = future.result()
                parsed = canonical_completion(content, axes, config.character_limit) if failure is None else None
                category = failure if failure is not None else (None if parsed is not None else "rationale_schema")
                if category:
                    failures[str(category)] += 1
                diagnoses = None if parsed is None else {axis: parsed[axis]["rationale"] for axis in axes}
                if diagnoses is not None and config.task != "bundle":
                    prior = baseline_triplet.get(task["source_id"])
                    _need(prior is not None, "axis-only baseline source linkage differs")
                    diagnoses = {axis: diagnoses[axis] if axis in diagnoses else prior[axis] for axis in AXES}
                records.append({"source_id": task["source_id"], "rationale": None if diagnoses is None else rationale_object(diagnoses, AXES), "parse_valid": diagnoses is not None, "failure_category": category})
    _need(len(records) == len(tasks), "generation observation count differs")
    records.sort(key=lambda item: str(item["source_id"]))
    source_ids = [item["source_id"] for item in records]
    _need(len(set(source_ids)) == len(tasks), "generation source linkage is non-unique")
    records_path = output / "generated_rationales.jsonl"
    with records_path.open("x", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
    valid = len(records) - sum(failures.values())
    report = {
        "status": "completed" if valid == len(records) else "failed_gates", "run_id": config.run_id, "base_key": config.base_key, "task": config.task, "arm": config.arm,
        "counts": {"expected": len(tasks), "observations": len(records), "parse_valid": valid}, "failure_categories": dict(sorted(failures.items())),
        "hard_gates": {"complete_records": len(records) == len(tasks), "all_outputs_parse_valid": valid == len(records), "zero_transport_or_schema_failures": not failures},
        "generated_rationales_sha256": _sha(records_path), "generation_server_attestation_sha256": _sha(server_attestation),
        "rlaif_training_complete_sha256": manifest["rlaif_training_complete_sha256"], "hybrid_axis_evaluation": config.task != "bundle",
        "source_writing_scores_read_or_prompted": False, "candidate_scores_read_or_prompted": False, "raw_prompts_or_completions_tracked": False,
    }
    _atomic_json(output / "aggregate_generation_report.json", report)
    manifest["status"] = "generation_completed" if all(report["hard_gates"].values()) else "generation_failed_gates"; manifest["aggregate_generation_report_sha256"] = _sha(output / "aggregate_generation_report.json")
    _atomic_json(output / "manifest.json", manifest)
    if not all(report["hard_gates"].values()):
        raise RLAIFEvaluationError("RLAIF validation generation hard gate failed")
    return report


def _generated(config: RLAIFEvaluationConfig) -> Mapping[str, Mapping[str, str]]:
    path = Path(config.output_dir) / "generated_rationales.jsonl"; _need(path.is_file() and not path.is_symlink(), "post-RL generation is unavailable")
    result: dict[str, Mapping[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line); source = item.get("source_id"); rationale = item.get("rationale")
            _need(isinstance(source, str) and source not in result and item.get("parse_valid") is True, "post-RL generation record is invalid")
            parsed = parse_rationale_output(json.dumps(rationale, ensure_ascii=False), AXES)
            _need(parsed is not None, "post-RL generation rationale schema differs")
            result[source] = parsed
    _need(len(result) == 400, "post-RL validation generation population differs")
    return result


def _judge_server(config: RLAIFEvaluationConfig, endpoint: str, attestation_path: Path) -> Mapping[str, Any]:
    settings = RLAIFSettings.from_json(); parsed = urlparse(endpoint)
    _need(parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port is not None and attestation_path.is_file(), "v6 judge endpoint/attestation differs")
    value = json.loads(attestation_path.read_text(encoding="utf-8"))
    frozen_runtime = settings.fixed_prompt_template().get("runtime")
    _need(isinstance(frozen_runtime, Mapping) and isinstance(frozen_runtime.get("enforce_eager"), bool), "fixed-v6 eager-mode setting is unavailable")
    expected = {"schema_version": "mal2026-rlaif-grpo-v6-evaluation-server-attestation-v1", "server_host": "127.0.0.1", "server_port": parsed.port,
                "physical_gpus": [0, 1, 2, 3], "tensor_parallel_size": 1, "data_parallel_size": 4, "max_model_len": 4096,
                "max_num_seqs_per_rank": 192, "enforce_eager": bool(frozen_runtime["enforce_eager"]), "fixed_v6_config_sha256": settings.fixed_v6_config_sha256,
                "server_process_environment_verified": True}
    _need(isinstance(value, dict) and all(value.get(key) == expected_value for key, expected_value in expected.items()), "frozen-v6 judge server attestation differs")
    return value


def _judge_tasks(config: RLAIFEvaluationConfig, generated: Mapping[str, Mapping[str, str]]) -> Iterator[dict[str, Any]]:
    settings = RLAIFSettings.from_json(); template = settings.fixed_prompt_template(); rows = load_writing_rows("validation", include_scores=False)
    for row in rows:
        diagnoses = generated.get(row.identifier); _need(diagnoses is not None, "post-RL rationale/source linkage differs")
        entry = {"sentences": JUDGE.sentence_list(row.essay), "rationale": rationale_object(diagnoses, AXES)}
        source_key = JUDGE.opaque(config.run_id, row.identifier)
        for prompt_index, prompt_type in enumerate(template["protocol"]["prompt_types"]):
            for seed_index, seed in enumerate(template["sampling"]["seeds"]):
                yield {"opaque_request_key": JUDGE.opaque(config.run_id, source_key, prompt_index, seed_index), "opaque_source_key": source_key,
                       "prompt_type_id": prompt_type["id"], "sampling_seed": seed, "response_contract": "required_scores_only_v1",
                       "body": JUDGE.request_body(template, settings.judge["model_id"], entry, list(AXES), prompt_type["layout"], seed, prompt_type["review_emphasis"])}


def _bootstrap_interval(values: Sequence[float], seed: int, repeats: int = 5000) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed); count = len(values); means: list[float] = []
    for _ in range(repeats):
        means.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    means.sort()
    return [round(means[int(0.025 * (repeats - 1))], 6), round(means[int(0.975 * (repeats - 1))], 6)]


def _baseline_per_essay(config: RLAIFEvaluationConfig, rows: Sequence[Any]) -> Mapping[str, Mapping[str, float]]:
    baseline_path = Path(config.baseline_judge_dir); observations = baseline_path / "score_observations.jsonl"; _need(observations.is_file(), "baseline score observations are unavailable")
    baseline_run_id = f"api-rationale-judge-v1-{config.base_key}-{config.baseline_kind}-validation-002"
    key_to_source = {JUDGE.opaque(baseline_run_id, row.identifier): row.identifier for row in rows}
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with observations.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line); source = key_to_source.get(item.get("opaque_generation_key")); scores = item.get("scores")
            if source is not None and item.get("scored") and isinstance(scores, dict):
                for axis in AXES:
                    values[source][axis].append(float(scores[axis]))
    result = {source: {axis: statistics.fmean(axis_values) for axis, axis_values in axes.items()} for source, axes in values.items()}
    _need(len(result) == len(rows) and all(set(item) == set(AXES) and all(len(values[source][axis]) == 50 for axis in AXES) for source, item in result.items()), "baseline per-essay fixed-v6 observations differ")
    return result


def judge_validation(config: RLAIFEvaluationConfig, endpoint: str, server_attestation: Path) -> dict[str, Any]:
    """Score a completed post-RL generation with the unchanged v6 5x10 judge."""
    config.validate(); attestation = _judge_server(config, endpoint, server_attestation)
    output = Path(config.output_dir); generation = json.loads((output / "aggregate_generation_report.json").read_text(encoding="utf-8"))
    _need(generation.get("status") == "completed" and all(generation.get("hard_gates", {}).values()), "post-RL generation gate did not pass")
    _need(not (output / "aggregate_judge_report.json").exists(), "post-RL judge report already exists")
    generated = _generated(config); rows = load_writing_rows("validation", include_scores=False); tasks = list(_judge_tasks(config, generated)); expected = len(rows) * 5 * 10
    _need(len(tasks) == expected == 20000, "fixed-v6 evaluation call population differs")
    observations = output / "score_observations.jsonl"; pending: set[Any] = set(); iterator = iter(tasks); exhausted = False
    with observations.open("x", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=768) as pool:
        future_tasks: dict[Any, dict[str, Any]] = {}
        while pending or not exhausted:
            while not exhausted and len(pending) < 768:
                try:
                    task = next(iterator)
                except StopIteration:
                    exhausted = True; break
                future = pool.submit(JUDGE.call, endpoint, task); pending.add(future); future_tasks[future] = task
            if not pending:
                continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                value = future.result(); future_tasks.pop(future, None)
                handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
    records = [json.loads(line) for line in observations.open(encoding="utf-8") if line.strip()]
    failures = Counter(str(item["failure_category"]) for item in records if item.get("failure_category"))
    valid = [item for item in records if item.get("scored") and isinstance(item.get("scores"), dict)]
    _need(len(records) == expected, "post-RL judge observations are incomplete")
    current_by_key: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    prompt_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item in valid:
        source = str(item["opaque_source_key"])
        for axis in AXES:
            current_by_key[source][axis].append(float(item["scores"][axis])); prompt_values[str(item["prompt_type_id"])][axis].append(float(item["scores"][axis]))
    source_key = {row.identifier: JUDGE.opaque(config.run_id, row.identifier) for row in rows}
    current = {identifier: {axis: statistics.fmean(current_by_key[source_key[identifier]][axis]) for axis in AXES} for identifier in source_key}
    _need(all(all(len(current_by_key[source_key[row.identifier]][axis]) == 50 for axis in AXES) for row in rows), "post-RL per-essay judge population differs")
    baseline = _baseline_per_essay(config, rows); selected_axes = axes_for_task(config.task)
    per_axis_delta = {axis: [current[row.identifier][axis] - baseline[row.identifier][axis] for row in rows] for axis in AXES}
    target_delta = [statistics.fmean(current[row.identifier][axis] - baseline[row.identifier][axis] for axis in selected_axes) for row in rows]
    axis_means = {axis: round(statistics.fmean(current[row.identifier][axis] for row in rows), 6) for axis in AXES}
    prompt_analysis = {prompt: {"observations": sum(len(values) for values in axis_values.values()) // 3, "axis_means": {axis: round(statistics.fmean(values), 6) for axis, values in axis_values.items()}, "macro_mean": round(statistics.fmean(statistics.fmean(values) for values in axis_values.values()), 6)} for prompt, axis_values in sorted(prompt_values.items())}
    report = {
        "status": "completed" if len(valid) == expected and not failures else "failed_gates", "run_id": config.run_id, "base_key": config.base_key, "task": config.task, "arm": config.arm,
        "counts": {"expected_calls": expected, "observations": len(records), "scored": len(valid), "schema_valid": sum(bool(item.get("schema_valid")) for item in records), "abstain": sum(bool(item.get("abstain")) for item in records), "generated_candidates": len(rows)},
        "hard_gates": {"complete_observations": len(records) == expected, "zero_transport_or_schema_failures": not failures, "all_scores_valid": len(valid) == expected, "five_prompt_forms": len(prompt_analysis) == 5},
        "failure_categories": dict(sorted(failures.items())), "axis_means": axis_means, "macro_mean": round(statistics.fmean(axis_means.values()), 6),
        "prompt_type_analysis": prompt_analysis, "prompt_type_axis_ranges": {axis: round(max(item["axis_means"][axis] for item in prompt_analysis.values()) - min(item["axis_means"][axis] for item in prompt_analysis.values()), 6) for axis in AXES},
        "paired_delta_from_sft": {axis: {"mean": round(statistics.fmean(values), 6), "bootstrap95": _bootstrap_interval(values, 2026072209 + index), "wins": sum(value > 0 for value in values), "ties": sum(value == 0 for value in values), "losses": sum(value < 0 for value in values)} for index, (axis, values) in enumerate(per_axis_delta.items())},
        "primary_requested_axes": list(selected_axes), "primary_paired_delta": {"mean": round(statistics.fmean(target_delta), 6), "bootstrap95": _bootstrap_interval(target_delta, 2026072299), "wins": sum(value > 0 for value in target_delta), "ties": sum(value == 0 for value in target_delta), "losses": sum(value < 0 for value in target_delta)},
        "baseline_judge_report_sha256": _sha(Path(config.baseline_judge_dir) / "aggregate_judge_report.json"), "baseline_generation_report_sha256": _sha(Path(config.baseline_generation_dir) / "aggregate_generation_report.json"),
        "fixed_v6_config_sha256": RLAIFSettings.from_json().fixed_v6_config_sha256, "judge_server_attestation_sha256": _sha(server_attestation), "generation_report_sha256": _sha(output / "aggregate_generation_report.json"),
        "source_writing_scores_read_or_prompted": False, "candidate_scores_read_or_prompted": False, "raw_prompts_or_completions_tracked": False,
    }
    _atomic_json(output / "aggregate_judge_report.json", report)
    manifest_path = output / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); manifest["status"] = "completed" if all(report["hard_gates"].values()) else "judge_failed_gates"; manifest["judge_server_attestation_sha256"] = _sha(server_attestation); manifest["aggregate_judge_report_sha256"] = _sha(output / "aggregate_judge_report.json"); _atomic_json(manifest_path, manifest)
    if not all(report["hard_gates"].values()):
        raise RLAIFEvaluationError("post-RL frozen-v6 judge hard gate failed")
    return report
