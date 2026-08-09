"""Validation-only decoder ablation for a train-human-audited scoring prompt.

The human study export and individual model responses are restricted.  Only
aggregate protocol evidence and metrics may leave the restricted/output run
roots.  Prompt rules are derived from train-split reasons whose human score is
within one point of the canonical half-up band and whose secondary audit is
``match`` or ``partial``.  Validation-split human reasons are excluded before
the prompt is evaluated on the validation set.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import gc
import json
import math
import os
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence

from .api_rationale_data import AXES, EXPECTED_ESSAYS, SOURCE_SHA256, load_writing_rows, sha256_file
from .decoder_fewshot_validation import condition_metrics, parse_response, response_schema, round_half_up
from .official_score_prompt import USER_SUPPLIED_EVALUATION, query_text, system_prompt


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "mal2026-decoder-human-audited-prompt-validation-config-v1"
RUN_ID = "decoder-human-audited-prompt-validation-v1-20260809-001"
ARMS = ("official_p0", "public_band_p1", "human_audit_p2")
RESTRICTED_ROOT = ROOT / "data/processed/restricted/decoder_human_audited_prompt_validation_v1"
PUBLIC_ROOT = ROOT / "outputs/analysis"
RUNTIME_ROOT = ROOT / "outputs/decoder-human-audited-prompt-validation-v1"


class HumanAuditedPromptError(RuntimeError):
    """Fail-closed protocol or runtime error."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise HumanAuditedPromptError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def jsonl(path: Path) -> list[dict[str, Any]]:
    need(path.is_file() and not path.is_symlink(), f"JSONL input unavailable: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            need(bool(line.strip()), f"blank JSONL line {line_number}: {path}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HumanAuditedPromptError(f"invalid JSONL line {line_number}: {path}") from exc
            need(isinstance(value, dict), f"non-object JSONL line {line_number}: {path}")
            rows.append(value)
    return rows


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_json_fresh(path: Path, payload: Mapping[str, Any]) -> str:
    need(not path.exists(), f"fresh output required: {path}")
    atomic_json(path, payload)
    return file_sha256(path)


def write_jsonl_fresh(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), f"fresh output required: {path}")
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return file_sha256(path)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    revision: str
    model_path: str
    tensor_parallel_size: int
    disable_thinking: bool


@dataclass(frozen=True)
class DerivationFilter:
    split: str
    maximum_absolute_human_target_difference: int
    accepted_audit_labels: tuple[str, ...]
    expected_selected_axis_reasons: int
    validation_axis_reasons_excluded: int


@dataclass(frozen=True)
class Config:
    schema_version: str
    run_id: str
    seed: int
    bootstrap_seed: int
    bootstrap_replicates: int
    split: str
    validation_rows: int
    prompt_arms: tuple[str, ...]
    official_prompt_path: str
    official_prompt_sha256: str
    public_band_prompt_path: str
    public_band_prompt_sha256: str
    human_audit_prompt_path: str
    human_audit_prompt_sha256: str
    human_responses_path: str
    human_responses_sha256: str
    reason_audit_path: str
    reason_audit_sha256: str
    derivation_filter: DerivationFilter
    model: ModelSpec
    physical_gpu_scope: tuple[int, ...]
    gpu_scope_authorization: str
    max_model_len: int
    max_tokens: int
    retry_max_tokens: int
    temperature: float
    gpu_memory_utilization: float
    max_num_seqs: int
    max_num_batched_tokens: int

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        need(isinstance(raw, dict), "config must be an object")
        model = ModelSpec(**raw.pop("model"))
        derivation = raw.pop("derivation_filter")
        derivation["accepted_audit_labels"] = tuple(derivation["accepted_audit_labels"])
        filt = DerivationFilter(**derivation)
        raw["prompt_arms"] = tuple(raw["prompt_arms"])
        raw["physical_gpu_scope"] = tuple(raw["physical_gpu_scope"])
        config = cls(model=model, derivation_filter=filt, **raw)
        config.validate(path)
        return config

    def validate(self, path: Path) -> None:
        need(self.schema_version == SCHEMA_VERSION and self.run_id == RUN_ID, "config identity differs")
        need(self.split == "validation" and self.validation_rows == EXPECTED_ESSAYS["validation"], "validation population differs")
        need(self.prompt_arms == ARMS, "prompt arms differ")
        need(self.physical_gpu_scope == (6,) and self.model.tensor_parallel_size == 1, "GPU 6 single-device scope differs")
        need("GPU 6 only" in self.gpu_scope_authorization, "GPU authorization record differs")
        need(self.model.key == "qwen35-9b" and self.model.disable_thinking is True, "model contract differs")
        need(self.derivation_filter.split == "train", "prompt derivation must be train-only")
        need(self.derivation_filter.maximum_absolute_human_target_difference == 1, "score agreement filter differs")
        need(self.derivation_filter.accepted_audit_labels == ("match", "partial"), "audit label filter differs")
        need((self.derivation_filter.expected_selected_axis_reasons, self.derivation_filter.validation_axis_reasons_excluded) == (18, 3), "audit selection population differs")
        need(self.temperature == 0.0 and self.max_tokens == 512 and self.retry_max_tokens == 2048, "decoding contract differs")
        need(self.max_model_len == 12288 and self.max_num_seqs == 64 and self.max_num_batched_tokens == 32768, "vLLM capacity differs")
        need(0.5 <= self.gpu_memory_utilization <= 0.95 and self.bootstrap_replicates == 10000, "runtime or bootstrap contract differs")
        for path_value, digest in (
            (self.official_prompt_path, self.official_prompt_sha256),
            (self.public_band_prompt_path, self.public_band_prompt_sha256),
            (self.human_audit_prompt_path, self.human_audit_prompt_sha256),
            (self.human_responses_path, self.human_responses_sha256),
            (self.reason_audit_path, self.reason_audit_sha256),
        ):
            source = Path(path_value)
            need(source.is_absolute() and source.is_file() and not source.is_symlink(), f"bound input unavailable: {source}")
            need(file_sha256(source) == digest, f"bound input checksum differs: {source}")
        model_path = Path(self.model.model_path)
        need(model_path.is_absolute() and model_path.is_dir() and not model_path.is_symlink(), "model snapshot unavailable")


def restricted_dir(config: Config) -> Path:
    return RESTRICTED_ROOT / config.run_id


def public_dir(config: Config) -> Path:
    return PUBLIC_ROOT / config.run_id


def runtime_dir(config: Config) -> Path:
    return RUNTIME_ROOT / config.run_id


def append_ledger(config: Config, stage: str, status: str, evidence: Mapping[str, Any]) -> None:
    path = runtime_dir(config) / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": now(), "run_id": config.run_id, "stage": stage, "status": status,
        "resource_scope": "GPU 6 only" if stage != "prepare" else "no GPU",
        "gpu_scope_authorization": config.gpu_scope_authorization,
        "evidence": dict(evidence),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def prompt_sections(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    markers = {line.strip(): i for i, line in enumerate(lines) if line.strip() in {"[시스템 프롬프트]", "[유저 프롬프트]"}}
    need(set(markers) == {"[시스템 프롬프트]", "[유저 프롬프트]"}, f"prompt markers differ: {path}")
    need(markers["[시스템 프롬프트]"] == 0 and markers["[유저 프롬프트]"] > 1, f"prompt routing differs: {path}")
    split = markers["[유저 프롬프트]"]
    system = "".join(lines[1:split])
    user = "".join(lines[split + 1:])
    need(user.count("{주제 지문}") == 1 and user.count("{논증적 글 본문}") == 1, f"prompt placeholders differ: {path}")
    need("average" not in system.lower() and "평균" in system and "다른 키" in system, f"output guard differs: {path}")
    return system, user


def messages_for(config: Config, arm: str, prompt: str, essay: str) -> list[dict[str, str]]:
    need(arm in ARMS, "unknown prompt arm")
    if arm == "official_p0":
        return [
            {"role": "system", "content": system_prompt(USER_SUPPLIED_EVALUATION)},
            {"role": "user", "content": query_text(prompt, essay, kind=USER_SUPPLIED_EVALUATION)},
        ]
    path = Path(config.public_band_prompt_path if arm == "public_band_p1" else config.human_audit_prompt_path)
    system, template = prompt_sections(path)
    user = template.replace("{주제 지문}", prompt).replace("{논증적 글 본문}", essay)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def filter_reason_audits(
    responses: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    filt: DerivationFilter,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audits = {
        (int(x["item_number"]), str(x["user_name"]), str(x["axis"])): x
        for x in audit["score_reason_audits"]
    }
    selected: list[dict[str, Any]] = []
    validation_excluded: list[dict[str, Any]] = []
    for row in responses:
        item_number = int(row["item_index"]) + 1
        for axis in AXES:
            reason = str(row.get(f"{axis}_reason") or "").strip()
            key = (item_number, str(row["user_name"]), axis)
            if not reason or key not in audits:
                continue
            human_score = row[f"{axis}_score"]
            target = row[f"target_{axis}_band"]
            label = audits[key]["label"]
            if abs(human_score - target) > filt.maximum_absolute_human_target_difference or label not in filt.accepted_audit_labels:
                continue
            entry = {
                "item_number": item_number,
                "source_id": row["source_id"],
                "user_name": row["user_name"],
                "split": row["split"],
                "axis": axis,
                "human_score": human_score,
                "target_band": target,
                "audit_label": label,
                "reason_sha256": sha256(reason.encode("utf-8")).hexdigest(),
                "audit_note_sha256": sha256(str(audits[key]["note"]).encode("utf-8")).hexdigest(),
            }
            if row["split"] == filt.split:
                selected.append(entry)
            elif row["split"] == "validation":
                validation_excluded.append(entry)
    selected.sort(key=lambda x: (x["item_number"], x["user_name"], x["axis"]))
    validation_excluded.sort(key=lambda x: (x["item_number"], x["user_name"], x["axis"]))
    return selected, validation_excluded


def prepare(config: Config, config_path: Path) -> dict[str, Any]:
    need(file_sha256(ROOT / "eval/train.jsonl") == SOURCE_SHA256["train"], "canonical train checksum differs")
    need(file_sha256(ROOT / "eval/validation.jsonl") == SOURCE_SHA256["validation"], "canonical validation checksum differs")
    responses = jsonl(Path(config.human_responses_path))
    audit = json.loads(Path(config.reason_audit_path).read_text(encoding="utf-8"))
    selected, validation_excluded = filter_reason_audits(responses, audit, config.derivation_filter)
    need(len(selected) == config.derivation_filter.expected_selected_axis_reasons, "train audit selection differs")
    need(len(validation_excluded) == config.derivation_filter.validation_axis_reasons_excluded, "validation audit exclusion differs")
    need(all(x["split"] == "train" for x in selected), "validation feedback leaked into prompt derivation")
    manifest = {
        "schema_version": "mal2026-human-audited-prompt-derivation-manifest-v1",
        "run_id": config.run_id,
        "selection_rule": {
            "split": config.derivation_filter.split,
            "maximum_absolute_human_target_difference": config.derivation_filter.maximum_absolute_human_target_difference,
            "accepted_audit_labels": list(config.derivation_filter.accepted_audit_labels),
        },
        "selected": selected,
        "validation_entries_excluded_before_derivation": validation_excluded,
        "no_individual_example_in_prompt": True,
    }
    manifest_sha = write_json_fresh(restricted_dir(config) / "derivation_manifest.json", manifest)
    selected_axis = Counter(x["axis"] for x in selected)
    selected_label = Counter(x["audit_label"] for x in selected)
    selected_user = Counter(x["user_name"] for x in selected)
    protocol = {
        "schema_version": "mal2026-decoder-human-audited-prompt-validation-protocol-v1",
        "status": "prepared",
        "run_id": config.run_id,
        "git_sha": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "config_sha256": file_sha256(config_path),
        "canonical_source_sha256": SOURCE_SHA256,
        "split": config.split,
        "validation_rows": config.validation_rows,
        "prompt_arms": list(config.prompt_arms),
        "prompt_sha256": {
            "official_p0": config.official_prompt_sha256,
            "public_band_p1": config.public_band_prompt_sha256,
            "human_audit_p2": config.human_audit_prompt_sha256,
        },
        "human_feedback_derivation": {
            "train_only": True,
            "selected_axis_reasons": len(selected),
            "validation_axis_reasons_excluded": len(validation_excluded),
            "by_axis": dict(sorted(selected_axis.items())),
            "by_audit_label": dict(sorted(selected_label.items())),
            "by_reviewer": dict(sorted(selected_user.items())),
            "individual_examples_in_prompt": False,
            "derivation_manifest_sha256": manifest_sha,
        },
        "primary_comparison": "human_audit_p2 minus official_p0",
        "secondary_comparison": "human_audit_p2 minus public_band_p1",
        "primary_metrics": ["macro_raw_rmse", "macro_raw_spearman"],
        "validation_status": "locked_descriptive_previously_observed",
        "model": {
            "key": config.model.key, "model_id": config.model.model_id,
            "revision": config.model.revision, "tensor_parallel_size": config.model.tensor_parallel_size,
        },
        "physical_gpu_scope": list(config.physical_gpu_scope),
        "gpu_scope_authorization": config.gpu_scope_authorization,
    }
    protocol_sha = write_json_fresh(public_dir(config) / "protocol.json", protocol)
    append_ledger(config, "prepare", "passed", {
        "protocol_sha256": protocol_sha, "derivation_manifest_sha256": manifest_sha,
        "selected_train_axis_reasons": len(selected), "excluded_validation_axis_reasons": len(validation_excluded),
    })
    return protocol


def verify_prepared(config: Config, config_path: Path) -> dict[str, Any]:
    protocol_path = public_dir(config) / "protocol.json"
    manifest_path = restricted_dir(config) / "derivation_manifest.json"
    need(protocol_path.is_file() and manifest_path.is_file(), "prepared protocol unavailable")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    need(protocol["config_sha256"] == file_sha256(config_path), "prepared config binding differs")
    need(protocol["human_feedback_derivation"]["derivation_manifest_sha256"] == file_sha256(manifest_path), "derivation binding differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    need(all(x["split"] == "train" for x in manifest["selected"]), "prepared derivation includes validation")
    return protocol


def build_requests(config: Config, tokenizer: Any) -> list[dict[str, Any]]:
    validation = load_writing_rows("validation", include_scores=True)
    need(len(validation) == config.validation_rows, "validation rows differ")
    requests: list[dict[str, Any]] = []
    for arm in ARMS:
        for row in validation:
            messages = messages_for(config, arm, row.prompt, row.essay)
            kwargs = {"enable_thinking": False} if config.model.disable_thinking else {}
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
            token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
            need(token_count + config.max_tokens <= config.max_model_len, f"prompt exceeds context: {arm}")
            need(row.scores is not None, "validation gold unavailable")
            requests.append({
                "source_id": row.identifier,
                "arm": arm,
                "prompt": rendered,
                "prompt_tokens": token_count,
                "gold_raw": {axis: float(row.scores[axis]) for axis in AXES},
                "gold_integer": {axis: round_half_up(row.scores[axis]) for axis in AXES},
            })
    requests.sort(key=lambda x: (x["arm"], sha256(x["source_id"].encode()).hexdigest()))
    need(len(requests) == config.validation_rows * len(ARMS), "request population differs")
    need(len({(x["source_id"], x["arm"]) for x in requests}) == len(requests), "request keys differ")
    return requests


def output_record(request: Mapping[str, Any], output: Any) -> dict[str, Any]:
    need(len(output.outputs) == 1, "vLLM choice count differs")
    choice = output.outputs[0]
    response = choice.text
    try:
        parsed = parse_response(response)
        parse_valid, parse_error = True, None
    except Exception as exc:  # exact error retained for integration recovery evidence
        parsed, parse_valid, parse_error = None, False, str(exc)
    return {
        "source_id": request["source_id"], "arm": request["arm"],
        "prompt_tokens": request["prompt_tokens"], "completion_tokens": len(choice.token_ids),
        "finish_reason": choice.finish_reason, "response": response,
        "parse_valid": parse_valid, "parse_error": parse_error,
        "prediction": {axis: parsed[axis]["score"] for axis in AXES} if parsed else None,
        "gold_raw": request["gold_raw"], "gold_integer": request["gold_integer"],
    }


def tail_recall(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for score in range(1, 6):
        matches = [
            int(row["prediction"][axis] == score)
            for row in rows for axis in AXES if row["gold_integer"][axis] == score
        ]
        result[str(score)] = {
            "count": len(matches), "exact": sum(matches),
            "recall": statistics.mean(matches) if matches else None,
        }
    return result


def prompt_pair_diagnostics(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, Any]:
    by_key = {(row["source_id"], row["arm"]): row for row in rows}
    ids = sorted({row["source_id"] for row in rows})
    by_axis: dict[str, Any] = {}
    for axis in AXES:
        deltas = [by_key[(source_id, right)]["prediction"][axis] - by_key[(source_id, left)]["prediction"][axis] for source_id in ids]
        by_axis[axis] = {
            "mean_delta": statistics.mean(deltas),
            "unchanged_rate": statistics.mean(delta == 0 for delta in deltas),
            "higher_rate": statistics.mean(delta > 0 for delta in deltas),
            "lower_rate": statistics.mean(delta < 0 for delta in deltas),
        }
    return {"left": left, "right": right, "by_axis": by_axis}


def macro_rmse_from_indices(squared: Mapping[str, Sequence[float]], indices: Sequence[int]) -> float:
    return statistics.mean(math.sqrt(sum(squared[axis][i] for i in indices) / len(indices)) for axis in AXES)


def paired_rmse_bootstrap(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, *, replicates: int, seed: int,
) -> dict[str, Any]:
    by_key = {(row["source_id"], row["arm"]): row for row in rows}
    ids = sorted({row["source_id"] for row in rows})
    need(all((source_id, arm) in by_key for source_id in ids for arm in (left, right)), "bootstrap pair population differs")
    squared: dict[str, dict[str, list[float]]] = {arm: {axis: [] for axis in AXES} for arm in (left, right)}
    for source_id in ids:
        for arm in (left, right):
            row = by_key[(source_id, arm)]
            for axis in AXES:
                squared[arm][axis].append((float(row["prediction"][axis]) - float(row["gold_raw"][axis])) ** 2)
    full = list(range(len(ids)))
    point = macro_rmse_from_indices(squared[right], full) - macro_rmse_from_indices(squared[left], full)
    rng = random.Random(seed)
    deltas = []
    for _ in range(replicates):
        sample = [rng.randrange(len(ids)) for _ in ids]
        deltas.append(macro_rmse_from_indices(squared[right], sample) - macro_rmse_from_indices(squared[left], sample))
    deltas.sort()
    low = deltas[int(0.025 * replicates)]
    high = deltas[min(replicates - 1, int(0.975 * replicates))]
    return {"left": left, "right": right, "delta_right_minus_left": point, "bootstrap_replicates": replicates, "interval_95": [low, high]}


def run(config: Config, config_path: Path) -> dict[str, Any]:
    protocol = verify_prepared(config, config_path)
    visible = tuple(int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip())
    need(visible == config.physical_gpu_scope, "CUDA_VISIBLE_DEVICES must be exactly physical GPU 6")
    aggregate_path = public_dir(config) / "aggregate.json"
    need(not aggregate_path.exists(), "completed aggregate already exists")
    try:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as exc:  # pragma: no cover
        raise HumanAuditedPromptError("vLLM unavailable in existing environment") from exc

    append_ledger(config, "vllm", "starting", {"physical_gpu_scope": list(visible), "model_revision": config.model.revision})
    llm = LLM(
        model=config.model.model_path, tokenizer=config.model.model_path,
        tensor_parallel_size=config.model.tensor_parallel_size, dtype="auto",
        trust_remote_code=False, max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_seqs=config.max_num_seqs, max_num_batched_tokens=config.max_num_batched_tokens,
        enable_prefix_caching=True, enforce_eager=False, seed=config.seed,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tokenizer = llm.get_tokenizer()
    requests = build_requests(config, tokenizer)
    sampling = SamplingParams(
        temperature=config.temperature, max_tokens=config.max_tokens, seed=config.seed,
        structured_outputs=StructuredOutputsParams(json=response_schema()),
    )
    smoke_requests = [next(x for x in requests if x["arm"] == arm) for arm in ARMS]
    smoke_outputs = llm.generate([x["prompt"] for x in smoke_requests], sampling)
    smoke_records = [output_record(request, output) for request, output in zip(smoke_requests, smoke_outputs)]
    need(all(x["parse_valid"] for x in smoke_records), "smoke parse failed")
    smoke_payload = {
        "schema_version": "mal2026-decoder-human-audited-prompt-smoke-v1",
        "status": "passed", "run_id": config.run_id, "requests": len(smoke_records),
        "arms": list(ARMS), "maximum_prompt_tokens": max(x["prompt_tokens"] for x in requests),
        "max_model_len": config.max_model_len, "max_tokens": config.max_tokens,
        "physical_gpu_scope": list(visible),
    }
    smoke_sha = write_json_fresh(public_dir(config) / "smoke.json", smoke_payload)
    append_ledger(config, "smoke", "passed", {"smoke_sha256": smoke_sha, "requests": len(smoke_records)})

    outputs = llm.generate([x["prompt"] for x in requests], sampling)
    need(len(outputs) == len(requests), "full output population differs")
    initial = [output_record(request, output) for request, output in zip(requests, outputs)]
    initial_sha = write_jsonl_fresh(restricted_dir(config) / "predictions.initial.jsonl", initial)
    failures = [row for row in initial if not row["parse_valid"]]
    retries: list[dict[str, Any]] = []
    final_by_key = {(row["source_id"], row["arm"]): row for row in initial}
    if failures:
        need(all(row["finish_reason"] == "length" and row["completion_tokens"] == config.max_tokens for row in failures), "non-length parse failure requires protocol review")
        request_map = {(x["source_id"], x["arm"]): x for x in requests}
        retry_requests = [request_map[(row["source_id"], row["arm"])] for row in failures]
        retry_sampling = SamplingParams(
            temperature=config.temperature, max_tokens=config.retry_max_tokens, seed=config.seed,
            structured_outputs=StructuredOutputsParams(json=response_schema()),
        )
        retry_outputs = llm.generate([x["prompt"] for x in retry_requests], retry_sampling)
        retries = [output_record(request, output) for request, output in zip(retry_requests, retry_outputs)]
        need(all(row["parse_valid"] for row in retries), "length retry did not resolve all responses")
        for row in retries:
            final_by_key[(row["source_id"], row["arm"])] = row
        retry_sha = write_jsonl_fresh(restricted_dir(config) / "predictions.retry-length.jsonl", retries)
        append_ledger(config, "integration-recovery", "passed", {"length_retries": len(retries), "retry_sha256": retry_sha})
    final = [final_by_key[(request["source_id"], request["arm"])] for request in requests]
    need(all(row["parse_valid"] for row in final), "invalid final response")
    final_sha = write_jsonl_fresh(restricted_dir(config) / "predictions.final.jsonl", final)

    metrics = {
        arm: condition_metrics([row for row in final if row["arm"] == arm], expected_count=config.validation_rows)
        for arm in ARMS
    }
    tails = {arm: tail_recall([row for row in final if row["arm"] == arm]) for arm in ARMS}
    bootstrap = {
        "p2_minus_p0": paired_rmse_bootstrap(final, "official_p0", "human_audit_p2", replicates=config.bootstrap_replicates, seed=config.bootstrap_seed),
        "p2_minus_p1": paired_rmse_bootstrap(final, "public_band_p1", "human_audit_p2", replicates=config.bootstrap_replicates, seed=config.bootstrap_seed + 1),
    }
    aggregate = {
        "schema_version": "mal2026-decoder-human-audited-prompt-validation-result-v1",
        "status": "completed", "run_id": config.run_id,
        "git_sha": protocol["git_sha"], "config_sha256": file_sha256(config_path),
        "model_key": config.model.key, "model_id": config.model.model_id,
        "model_revision": config.model.revision, "physical_gpu_scope": list(visible),
        "validation_rows": config.validation_rows, "requests": len(final),
        "initial_parse_failures": len(failures), "length_retries": len(retries),
        "initial_prediction_sha256": initial_sha, "final_prediction_sha256": final_sha,
        "prompt_tokens": {"minimum": min(x["prompt_tokens"] for x in final), "maximum": max(x["prompt_tokens"] for x in final), "mean": statistics.mean(x["prompt_tokens"] for x in final)},
        "completion_tokens": {"minimum": min(x["completion_tokens"] for x in final), "maximum": max(x["completion_tokens"] for x in final), "mean": statistics.mean(x["completion_tokens"] for x in final)},
        "metrics": metrics, "tail_exact_recall": tails,
        "paired_bootstrap_macro_raw_rmse": bootstrap,
        "prompt_pair_diagnostics": {
            "p0_to_p2": prompt_pair_diagnostics(final, "official_p0", "human_audit_p2"),
            "p1_to_p2": prompt_pair_diagnostics(final, "public_band_p1", "human_audit_p2"),
        },
        "interpretation_boundary": "locked descriptive validation; not an unbiased selection estimate",
    }
    aggregate_sha = write_json_fresh(aggregate_path, aggregate)
    append_ledger(config, "full-validation", "completed", {
        "aggregate_sha256": aggregate_sha, "final_prediction_sha256": final_sha,
        "requests": len(final), "initial_parse_failures": len(failures), "length_retries": len(retries),
    })
    del llm
    gc.collect()
    return aggregate
