"""Validation-only few-shot decoder scoring with the exact evaluation.txt prompt.

Row-level demonstrations, writings, responses, and predictions remain below the
ignored restricted data root.  Public outputs contain aggregate metrics and
provenance only.  Demonstrations are selected from canonical train data only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from .api_rationale_data import AXES, EXPECTED_ESSAYS, SOURCE_SHA256, load_writing_rows, sha256_file
from .official_score_prompt import (
    EVALUATION_PROMPT_SHA256,
    USER_SUPPLIED_EVALUATION,
    query_text,
    system_prompt,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "mal2026-decoder-fewshot-validation-config-v1"
RUN_PATTERN = __import__("re").compile(r"[a-z0-9][a-z0-9-]{2,127}")
CONDITIONS = ("balanced5", "central5")
LENGTH_RETRY_MAX_TOKENS = 2048
RATIONALE_PATH = (
    ROOT / "data/processed/restricted/evaluation_prompt_rationale_v2/"
    "evaluation-prompt-rationale-generation-v2-score-blind-20260729-004/"
    "rationales.train.jsonl"
)
RATIONALE_SHA256 = "d4a2be9a070c786728fde6f64f066ac9d462bc5f83305a2d9161b380abd88e55"
RESTRICTED_ROOT = ROOT / "data/processed/restricted/decoder_fewshot_validation_v1"
PUBLIC_ROOT = ROOT / "outputs/analysis"
CENTRAL_TRIPLETS = ((3, 3, 3), (3, 3, 4), (3, 4, 3), (4, 3, 3), (4, 4, 4))


class DecoderFewshotError(RuntimeError):
    """Fail-closed contract error for the few-shot comparison."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise DecoderFewshotError(message)


def file_sha256(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"ordinary file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    need(path.is_file() and not path.is_symlink(), "JSONL input is unavailable")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            need(bool(line.strip()), f"blank JSONL row {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DecoderFewshotError(f"invalid JSONL row {line_number}") from exc
            need(isinstance(value, dict), "JSONL row must be an object")
            rows.append(value)
    return rows


def round_half_up(value: float) -> int:
    return min(5, max(1, math.floor(float(value) + 0.5)))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    revision: str
    model_path: str
    tensor_parallel_size: int
    disable_thinking: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ModelSpec":
        need(set(raw) == {"key", "model_id", "revision", "model_path", "tensor_parallel_size", "disable_thinking"}, "model schema differs")
        item = cls(**raw)
        need(bool(RUN_PATTERN.fullmatch(item.key)), "model key differs")
        need(all(isinstance(value, str) and value for value in (item.model_id, item.revision, item.model_path)), "model binding is blank")
        need(type(item.tensor_parallel_size) is int and item.tensor_parallel_size in {1, 2, 4}, "model TP differs")
        need(type(item.disable_thinking) is bool, "model thinking flag differs")
        return item


@dataclass(frozen=True)
class FewshotConfig:
    schema_version: str
    run_id: str
    seed: int
    score_prompt_kind: str
    score_prompt_sha256: str
    rationale_path: str
    rationale_sha256: str
    conditions: tuple[str, ...]
    max_model_len: int
    max_tokens: int
    temperature: float
    gpu_memory_utilization: float
    max_num_seqs: int
    max_num_batched_tokens: int
    models: tuple[ModelSpec, ...]

    @classmethod
    def from_json(cls, path: Path, *, require_models: bool = True) -> "FewshotConfig":
        need(path.is_file() and not path.is_symlink(), "config is unavailable")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DecoderFewshotError("config is unreadable") from exc
        need(isinstance(raw, dict) and set(raw) == {
            "schema_version", "run_id", "seed", "score_prompt_kind", "score_prompt_sha256",
            "rationale_path", "rationale_sha256", "conditions", "max_model_len", "max_tokens",
            "temperature", "gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens", "models",
        }, "config schema differs")
        models_raw = raw.pop("models")
        need(isinstance(models_raw, list), "models must be a list")
        conditions_raw = raw.pop("conditions")
        need(isinstance(conditions_raw, list), "conditions must be a list")
        config = cls(models=tuple(ModelSpec.from_mapping(item) for item in models_raw), conditions=tuple(conditions_raw), **raw)
        config.validate(require_models=require_models)
        return config

    def validate(self, *, require_models: bool = True) -> None:
        need(self.schema_version == SCHEMA_VERSION and bool(RUN_PATTERN.fullmatch(self.run_id)), "run identity differs")
        need(self.seed == 2026073104, "seed differs")
        need(self.score_prompt_kind == USER_SUPPLIED_EVALUATION and self.score_prompt_sha256 == EVALUATION_PROMPT_SHA256, "prompt binding differs")
        need(Path(self.rationale_path).resolve() == RATIONALE_PATH.resolve() and self.rationale_sha256 == RATIONALE_SHA256, "rationale binding differs")
        need(self.conditions == CONDITIONS, "few-shot conditions differ")
        need((self.max_model_len, self.max_tokens, self.temperature) == (12288, 512, 0.0), "generation budget differs")
        need(0.5 <= self.gpu_memory_utilization <= 0.95 and self.max_num_seqs > 0 and self.max_num_batched_tokens >= self.max_model_len, "vLLM capacity differs")
        if require_models:
            need(len(self.models) == 5 and len({item.key for item in self.models}) == 5, "five unique model arms are required")

    def model(self, key: str) -> ModelSpec:
        matching = [item for item in self.models if item.key == key]
        need(len(matching) == 1, "unknown model key")
        return matching[0]


@dataclass(frozen=True)
class Shot:
    source_id: str
    prompt: str
    essay: str
    scores: tuple[int, int, int]
    rationales: Mapping[str, str]


def load_train_rationales(path: Path = RATIONALE_PATH) -> dict[str, dict[str, str]]:
    need(file_sha256(path) == RATIONALE_SHA256, "train rationale checksum differs")
    result: dict[str, dict[str, str]] = {}
    for raw in _jsonl(path):
        need(set(raw) == {"source_id", "rationales"}, "train rationale schema differs")
        source_id, values = raw["source_id"], raw["rationales"]
        need(isinstance(source_id, str) and source_id not in result, "train rationale ID differs")
        need(isinstance(values, dict) and set(values) == set(AXES), "train rationale axes differ")
        need(all(isinstance(values[axis], str) and values[axis].strip() for axis in AXES), "train rationale is blank")
        result[source_id] = {axis: values[axis] for axis in AXES}
    need(len(result) == EXPECTED_ESSAYS["train"], "train rationale population differs")
    return result


def _shot_candidates() -> dict[tuple[int, int, int], list[Shot]]:
    rationales = load_train_rationales()
    rows = load_writing_rows("train", include_scores=True)
    result: dict[tuple[int, int, int], list[Shot]] = {}
    for row in rows:
        need(row.scores is not None and row.identifier in rationales, "train shot join differs")
        scores = tuple(round_half_up(row.scores[axis]) for axis in AXES)
        shot = Shot(row.identifier, row.prompt, row.essay, scores, rationales[row.identifier])
        result.setdefault(scores, []).append(shot)
    for values in result.values():
        values.sort(key=lambda item: (
            len(item.prompt) + len(item.essay) + sum(len(item.rationales[axis]) for axis in AXES),
            sha256(item.source_id.encode("utf-8")).hexdigest(),
        ))
    return result


def select_shots() -> dict[str, tuple[Shot, ...]]:
    """Select short train-only anchors without consulting validation."""
    candidates = _shot_candidates()
    balanced: tuple[Shot, ...] | None = None
    balanced_key: tuple[Any, ...] | None = None
    for organization in itertools.permutations(range(1, 6)):
        for expression in itertools.permutations(range(1, 6)):
            triplets = tuple((content, organization[content - 1], expression[content - 1]) for content in range(1, 6))
            if not all(candidates.get(triplet) for triplet in triplets):
                continue
            chosen = tuple(candidates[triplet][0] for triplet in triplets)
            key = (
                sum(len(item.prompt) + len(item.essay) + sum(len(item.rationales[axis]) for axis in AXES) for item in chosen),
                triplets,
            )
            if balanced_key is None or key < balanced_key:
                balanced_key, balanced = key, chosen
    need(balanced is not None, "no exact five-score balanced shot set exists")
    need(all(candidates.get(triplet) for triplet in CENTRAL_TRIPLETS), "central shot triplet is unavailable")
    central = tuple(candidates[triplet][0] for triplet in CENTRAL_TRIPLETS)
    need(len({item.source_id for item in balanced}) == len({item.source_id for item in central}) == 5, "shot IDs are not unique")
    for axis_index in range(3):
        need(sorted(item.scores[axis_index] for item in balanced) == [1, 2, 3, 4, 5], "balanced shots do not cover 1..5")
        need(Counter(item.scores[axis_index] for item in central) == Counter({3: 3, 4: 2}), "central shots do not preserve 3/4 prior")
    return {"balanced5": balanced, "central5": central}


def response_schema() -> dict[str, Any]:
    axis = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "rationale": {"type": "string", "minLength": 1},
        },
        "required": ["score", "rationale"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {name: axis for name in AXES},
        "required": list(AXES),
        "additionalProperties": False,
    }


def assistant_json(shot: Shot) -> str:
    payload = {
        axis: {"score": shot.scores[index], "rationale": shot.rationales[axis]}
        for index, axis in enumerate(AXES)
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def messages_for(shot_rows: Sequence[Shot], prompt: str, essay: str) -> list[dict[str, str]]:
    need(len(shot_rows) == 5, "exactly five demonstrations are required")
    messages = [{"role": "system", "content": system_prompt(USER_SUPPLIED_EVALUATION)}]
    for shot in shot_rows:
        messages.extend((
            {"role": "user", "content": query_text(shot.prompt, shot.essay, kind=USER_SUPPLIED_EVALUATION)},
            {"role": "assistant", "content": assistant_json(shot)},
        ))
    messages.append({"role": "user", "content": query_text(prompt, essay, kind=USER_SUPPLIED_EVALUATION)})
    return messages


def parse_response(text: str) -> dict[str, dict[str, Any]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DecoderFewshotError("decoder response is not exact JSON") from exc
    need(isinstance(value, dict) and set(value) == set(AXES), "decoder response axes differ")
    result: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        item = value[axis]
        need(isinstance(item, dict) and set(item) == {"score", "rationale"}, "decoder axis schema differs")
        score, rationale = item["score"], item["rationale"]
        need(type(score) is int and 1 <= score <= 5, "decoder score differs")
        need(isinstance(rationale, str) and bool(rationale.strip()), "decoder rationale is blank")
        result[axis] = {"score": score, "rationale": rationale}
    return result


def rotation_assignments(source_ids: Sequence[str], seed: int) -> dict[str, int]:
    ordered = sorted(source_ids, key=lambda item: sha256(f"{seed}:{item}".encode("utf-8")).hexdigest())
    need(len(ordered) == len(set(ordered)) and len(ordered) % 5 == 0, "validation rotation population differs")
    return {source_id: index % 5 for index, source_id in enumerate(ordered)}


def rotate(items: Sequence[Shot], amount: int) -> tuple[Shot, ...]:
    need(len(items) == 5 and 0 <= amount < 5, "shot rotation differs")
    return tuple(items[amount:]) + tuple(items[:amount])


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for index in order[cursor:end]:
            result[index] = rank
        cursor = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = _rank(left), _rank(right)
    mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def condition_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = EXPECTED_ESSAYS["validation"],
    total_count: int | None = None,
) -> dict[str, Any]:
    need(len(rows) == expected_count and len(rows) > 0, "metric population differs")
    need(all(row.get("parse_valid") is True for row in rows), "invalid responses cannot enter score metrics")
    denominator = len(rows) if total_count is None else total_count
    need(denominator >= len(rows), "metric total population differs")
    by_axis: dict[str, Any] = {}
    for axis in AXES:
        raw = [float(row["gold_raw"][axis]) for row in rows]
        integer_gold = [int(row["gold_integer"][axis]) for row in rows]
        pred = [int(row["prediction"][axis]) for row in rows]
        histogram = Counter(pred)
        by_axis[axis] = {
            "raw_rmse": math.sqrt(statistics.mean((a - b) ** 2 for a, b in zip(raw, pred))),
            "integer_rmse": math.sqrt(statistics.mean((a - b) ** 2 for a, b in zip(integer_gold, pred))),
            "raw_spearman": spearman(raw, pred),
            "integer_spearman": spearman(integer_gold, pred),
            "integer_accuracy": statistics.mean(a == b for a, b in zip(integer_gold, pred)),
            "prediction_mean": statistics.mean(pred),
            "prediction_std": statistics.pstdev(pred),
            "score_histogram": {str(score): histogram[score] for score in range(1, 6)},
            "score_3_rate": histogram[3] / len(pred),
            "score_4_rate": histogram[4] / len(pred),
        }
    return {
        "count": len(rows),
        "parse_success_rate": len(rows) / denominator,
        "triplet_integer_accuracy": statistics.mean(row["prediction"] == row["gold_integer"] for row in rows),
        "by_axis": by_axis,
        "macro_raw_rmse": statistics.mean(by_axis[axis]["raw_rmse"] for axis in AXES),
        "macro_integer_rmse": statistics.mean(by_axis[axis]["integer_rmse"] for axis in AXES),
        "macro_raw_spearman": statistics.mean(by_axis[axis]["raw_spearman"] for axis in AXES),
        "macro_integer_spearman": statistics.mean(by_axis[axis]["integer_spearman"] for axis in AXES),
        "macro_score_3_rate": statistics.mean(by_axis[axis]["score_3_rate"] for axis in AXES),
        "macro_score_4_rate": statistics.mean(by_axis[axis]["score_4_rate"] for axis in AXES),
    }


def restricted_run_dir(config: FewshotConfig) -> Path:
    path = RESTRICTED_ROOT / config.run_id
    need(path.resolve().is_relative_to(RESTRICTED_ROOT.resolve()), "restricted output escaped root")
    return path


def public_run_dir(config: FewshotConfig) -> Path:
    path = PUBLIC_ROOT / config.run_id
    need(path.resolve().is_relative_to(PUBLIC_ROOT.resolve()), "public output escaped root")
    return path


def write_json_fresh(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), f"fresh output required: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return file_sha256(path)


def write_jsonl_fresh(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), f"fresh output required: {path}")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return file_sha256(path)


def prepare_protocol(config: FewshotConfig, config_path: Path) -> dict[str, Any]:
    need(sha256_file(ROOT / "eval/train.jsonl") == SOURCE_SHA256["train"] and sha256_file(ROOT / "eval/validation.jsonl") == SOURCE_SHA256["validation"], "canonical data checksum differs")
    shots = select_shots()
    restricted = restricted_run_dir(config)
    public = public_run_dir(config)
    manifest_path = restricted / "shot_manifest.json"
    manifest = {
        "schema_version": "mal2026-decoder-fewshot-shot-manifest-v1",
        "run_id": config.run_id,
        "train_only": True,
        "validation_scores_used_for_prompting_or_selection": False,
        "conditions": {
            condition: [
                {
                    "source_id": item.source_id,
                    "score_triplet": list(item.scores),
                    "rationale_sha256": sha256(json.dumps(item.rationales, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
                }
                for item in items
            ]
            for condition, items in shots.items()
        },
    }
    manifest_sha = write_json_fresh(manifest_path, manifest)
    public_payload = {
        "schema_version": "mal2026-decoder-fewshot-protocol-v1",
        "status": "prepared",
        "run_id": config.run_id,
        "git_sha": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "config_sha256": file_sha256(config_path),
        "prompt_sha256": EVALUATION_PROMPT_SHA256,
        "rationale_sha256": RATIONALE_SHA256,
        "canonical_source_sha256": SOURCE_SHA256,
        "shot_manifest_sha256": manifest_sha,
        "score_fields": list(AXES),
        "average_target_used": False,
        "conditions": {
            condition: {
                "shots": 5,
                "score_triplets": [list(item.scores) for item in items],
                "per_axis_histogram": {
                    axis: {str(score): sum(item.scores[index] == score for item in items) for score in range(1, 6)}
                    for index, axis in enumerate(AXES)
                },
            }
            for condition, items in shots.items()
        },
        "models": [{"key": item.key, "model_id": item.model_id, "revision": item.revision, "tensor_parallel_size": item.tensor_parallel_size} for item in config.models],
    }
    write_json_fresh(public / "protocol.json", public_payload)
    return public_payload


def _verify_protocol(config: FewshotConfig) -> tuple[dict[str, tuple[Shot, ...]], str]:
    manifest_path = restricted_run_dir(config) / "shot_manifest.json"
    protocol_path = public_run_dir(config) / "protocol.json"
    need(manifest_path.is_file() and protocol_path.is_file(), "prepared protocol is unavailable")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    need(protocol.get("shot_manifest_sha256") == file_sha256(manifest_path), "shot manifest binding differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shots = select_shots()
    expected = {condition: [item.source_id for item in items] for condition, items in shots.items()}
    actual = {condition: [item["source_id"] for item in manifest["conditions"][condition]] for condition in CONDITIONS}
    need(actual == expected, "shot selection replay differs")
    return shots, file_sha256(manifest_path)


def run_model(config: FewshotConfig, config_path: Path, model_key: str) -> dict[str, Any]:
    spec = config.model(model_key)
    model_path = Path(spec.model_path)
    need(model_path.is_absolute() and model_path.is_dir() and not model_path.is_symlink(), "local model snapshot is unavailable")
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    need(len(visible) == spec.tensor_parallel_size, "visible GPU count differs from model TP")
    shots, shot_manifest_sha = _verify_protocol(config)
    validation = load_writing_rows("validation", include_scores=True)
    assignments = rotation_assignments([row.identifier for row in validation], config.seed)
    try:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise DecoderFewshotError("vLLM is unavailable in the existing environment") from exc

    llm = LLM(
        model=str(model_path), tokenizer=str(model_path), tensor_parallel_size=spec.tensor_parallel_size,
        dtype="auto", trust_remote_code=False, max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization, max_num_seqs=config.max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens, enable_prefix_caching=True,
        enforce_eager=False, seed=config.seed,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tokenizer = llm.get_tokenizer()
    requests: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for row in validation:
            rotation = assignments[row.identifier]
            messages = messages_for(rotate(shots[condition], rotation), row.prompt, row.essay)
            kwargs = {"enable_thinking": False} if spec.disable_thinking else {}
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
            token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
            need(token_count + config.max_tokens <= config.max_model_len, f"prompt exceeds max model length for {model_key}")
            requests.append({"source_id": row.identifier, "condition": condition, "rotation": rotation, "prompt": rendered, "prompt_tokens": token_count})
    requests.sort(key=lambda item: (item["condition"], item["rotation"], sha256(item["source_id"].encode("utf-8")).hexdigest()))
    sampling = SamplingParams(
        temperature=config.temperature, max_tokens=config.max_tokens, seed=config.seed,
        structured_outputs=StructuredOutputsParams(json=response_schema()),
    )
    smoke_requests = [next(item for item in requests if item["condition"] == condition) for condition in CONDITIONS]
    smoke_outputs = llm.generate([item["prompt"] for item in smoke_requests], sampling)
    need(len(smoke_outputs) == len(smoke_requests), "smoke output population differs")
    for output in smoke_outputs:
        need(len(output.outputs) == 1, "smoke choice count differs")
        parse_response(output.outputs[0].text)
    write_json_fresh(public_run_dir(config) / "models" / model_key / "smoke.json", {
        "schema_version": "mal2026-decoder-fewshot-model-smoke-v1",
        "status": "passed",
        "run_id": config.run_id,
        "model_key": model_key,
        "requests": len(smoke_requests),
        "conditions": list(CONDITIONS),
        "maximum_prompt_tokens": max(item["prompt_tokens"] for item in requests),
        "max_model_len": config.max_model_len,
        "max_tokens": config.max_tokens,
    })
    outputs = llm.generate([item["prompt"] for item in requests], sampling)
    need(len(outputs) == len(requests), "vLLM output population differs")
    by_id = {row.identifier: row for row in validation}
    persisted: list[dict[str, Any]] = []
    for request, output in zip(requests, outputs):
        need(len(output.outputs) == 1, "vLLM choice count differs")
        choice = output.outputs[0]
        response = choice.text
        parse_valid, parsed, error = True, None, None
        try:
            parsed = parse_response(response)
        except DecoderFewshotError as exc:
            parse_valid, error = False, str(exc)
        source = by_id[request["source_id"]]
        need(source.scores is not None, "validation gold is unavailable for metrics")
        persisted.append({
            "source_id": source.identifier,
            "condition": request["condition"],
            "rotation": request["rotation"],
            "prompt_tokens": request["prompt_tokens"],
            "completion_tokens": len(output.outputs[0].token_ids),
            "finish_reason": choice.finish_reason,
            "response": response,
            "parse_valid": parse_valid,
            "parse_error": error,
            "prediction": {axis: parsed[axis]["score"] for axis in AXES} if parsed else None,
            "gold_raw": {axis: float(source.scores[axis]) for axis in AXES},
            "gold_integer": {axis: round_half_up(source.scores[axis]) for axis in AXES},
        })
    restricted_path = restricted_run_dir(config) / "models" / model_key / "predictions.jsonl"
    prediction_sha = write_jsonl_fresh(restricted_path, persisted)
    parsed_rows = [row for row in persisted if row["parse_valid"]]
    need(len(parsed_rows) == len(persisted), "structured output produced an invalid response")
    metrics = {condition: condition_metrics([row for row in parsed_rows if row["condition"] == condition]) for condition in CONDITIONS}
    aggregate = {
        "schema_version": "mal2026-decoder-fewshot-model-result-v1",
        "status": "completed",
        "run_id": config.run_id,
        "model_key": model_key,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "physical_gpus": [int(item) for item in visible],
        "tensor_parallel_size": spec.tensor_parallel_size,
        "config_sha256": file_sha256(config_path),
        "shot_manifest_sha256": shot_manifest_sha,
        "prediction_sha256": prediction_sha,
        "validation_rows": len(validation),
        "requests": len(persisted),
        "parse_failures": len(persisted) - len(parsed_rows),
        "prompt_tokens": {
            "minimum": min(row["prompt_tokens"] for row in persisted),
            "maximum": max(row["prompt_tokens"] for row in persisted),
            "mean": statistics.mean(row["prompt_tokens"] for row in persisted),
        },
        "completion_tokens": {
            "minimum": min(row["completion_tokens"] for row in persisted),
            "maximum": max(row["completion_tokens"] for row in persisted),
            "mean": statistics.mean(row["completion_tokens"] for row in persisted),
        },
        "metrics": metrics,
    }
    write_json_fresh(public_run_dir(config) / "models" / model_key / "aggregate.json", aggregate)
    return aggregate


def retry_length_failures(config: FewshotConfig, config_path: Path, model_key: str) -> dict[str, Any]:
    """Retry only schema-truncated rows with a larger integration ceiling.

    The first full output is immutable.  This recovery is allowed only when
    every invalid row ended exactly at the declared 512-token ceiling; model,
    prompt, demonstrations, temperature, seed, and structured schema remain
    unchanged.
    """
    spec = config.model(model_key)
    model_path = Path(spec.model_path)
    need(model_path.is_absolute() and model_path.is_dir() and not model_path.is_symlink(), "local model snapshot is unavailable")
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    need(len(visible) == spec.tensor_parallel_size, "visible GPU count differs from model TP")
    shots, shot_manifest_sha = _verify_protocol(config)
    initial_path = restricted_run_dir(config) / "models" / model_key / "predictions.jsonl"
    aggregate_path = public_run_dir(config) / "models" / model_key / "aggregate.json"
    need(initial_path.is_file() and not aggregate_path.exists(), "length retry requires one failed immutable initial result")
    initial = _jsonl(initial_path)
    need(len(initial) == 2 * EXPECTED_ESSAYS["validation"], "initial retry population differs")
    failures = [row for row in initial if row.get("parse_valid") is False]
    need(bool(failures), "length retry has no failures")
    need(all(row.get("finish_reason") == "length" and row.get("completion_tokens") == config.max_tokens and row.get("parse_error") == "decoder response is not exact JSON" for row in failures), "retry is limited to exact length truncations")
    need(len({(row["source_id"], row["condition"]) for row in initial}) == len(initial), "initial retry keys differ")

    validation = load_writing_rows("validation", include_scores=True)
    by_id = {row.identifier: row for row in validation}
    need(set(row["source_id"] for row in failures) <= set(by_id), "retry source population differs")
    try:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise DecoderFewshotError("vLLM is unavailable in the existing environment") from exc
    llm = LLM(
        model=str(model_path), tokenizer=str(model_path), tensor_parallel_size=spec.tensor_parallel_size,
        dtype="auto", trust_remote_code=False, max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization, max_num_seqs=config.max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens, enable_prefix_caching=True,
        enforce_eager=False, seed=config.seed,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tokenizer = llm.get_tokenizer()
    requests: list[dict[str, Any]] = []
    for row in failures:
        source = by_id[row["source_id"]]
        condition, rotation = row["condition"], int(row["rotation"])
        need(condition in CONDITIONS and 0 <= rotation < 5, "retry condition/rotation differs")
        messages = messages_for(rotate(shots[condition], rotation), source.prompt, source.essay)
        kwargs = {"enable_thinking": False} if spec.disable_thinking else {}
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        prompt_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
        need(prompt_tokens == row["prompt_tokens"] and prompt_tokens + LENGTH_RETRY_MAX_TOKENS <= config.max_model_len, "retry prompt binding or length differs")
        requests.append({"source_id": source.identifier, "condition": condition, "rotation": rotation, "prompt": rendered, "prompt_tokens": prompt_tokens})
    sampling = SamplingParams(
        temperature=config.temperature, max_tokens=LENGTH_RETRY_MAX_TOKENS, seed=config.seed,
        structured_outputs=StructuredOutputsParams(json=response_schema()),
    )
    outputs = llm.generate([item["prompt"] for item in requests], sampling)
    need(len(outputs) == len(requests), "retry output population differs")
    retry_rows: list[dict[str, Any]] = []
    replacements: dict[tuple[str, str], dict[str, Any]] = {}
    for request, output in zip(requests, outputs):
        need(len(output.outputs) == 1, "retry choice count differs")
        choice = output.outputs[0]
        base = next(row for row in failures if (row["source_id"], row["condition"]) == (request["source_id"], request["condition"]))
        try:
            parsed = parse_response(choice.text)
            parse_valid, parse_error = True, None
        except DecoderFewshotError as exc:
            parsed = None
            parse_valid, parse_error = False, str(exc)
        attempted = {
            **base,
            "completion_tokens": len(choice.token_ids),
            "finish_reason": choice.finish_reason,
            "response": choice.text,
            "parse_valid": parse_valid,
            "parse_error": parse_error,
            "prediction": None if parsed is None else {axis: parsed[axis]["score"] for axis in AXES},
            "integration_retry": {"reason": "initial_length_truncation", "max_tokens": LENGTH_RETRY_MAX_TOKENS},
        }
        key = (request["source_id"], request["condition"])
        if parse_valid and choice.finish_reason != "length":
            replacements[key] = attempted
        retry_rows.append(attempted)
    retry_path = restricted_run_dir(config) / "models" / model_key / f"length_retry_predictions.max{LENGTH_RETRY_MAX_TOKENS}.jsonl"
    retry_sha = write_jsonl_fresh(retry_path, retry_rows)
    need(
        len(replacements) == len(failures),
        f"length retry did not resolve every response; preserved at {retry_path}",
    )
    resolved_rows = [replacements.get((row["source_id"], row["condition"]), row) for row in initial]
    need(all(row.get("parse_valid") is True for row in resolved_rows), "retry did not resolve every invalid response")
    resolved_path = restricted_run_dir(config) / "models" / model_key / "predictions.resolved.jsonl"
    resolved_sha = write_jsonl_fresh(resolved_path, resolved_rows)
    metrics = {condition: condition_metrics([row for row in resolved_rows if row["condition"] == condition]) for condition in CONDITIONS}
    aggregate = {
        "schema_version": "mal2026-decoder-fewshot-model-result-v1",
        "status": "completed",
        "run_id": config.run_id,
        "model_key": model_key,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "physical_gpus": [int(item) for item in visible],
        "tensor_parallel_size": spec.tensor_parallel_size,
        "config_sha256": file_sha256(config_path),
        "shot_manifest_sha256": shot_manifest_sha,
        "prediction_sha256": resolved_sha,
        "initial_prediction_sha256": file_sha256(initial_path),
        "length_retry_prediction_sha256": retry_sha,
        "validation_rows": len(validation),
        "requests": len(resolved_rows),
        "parse_failures": 0,
        "integration_retry": {
            "reason": "structured JSON truncated exactly at initial max_tokens",
            "initial_max_tokens": config.max_tokens,
            "retry_max_tokens": LENGTH_RETRY_MAX_TOKENS,
            "retried_requests": len(retry_rows),
            "other_variables_changed": False,
        },
        "prompt_tokens": {
            "minimum": min(row["prompt_tokens"] for row in resolved_rows),
            "maximum": max(row["prompt_tokens"] for row in resolved_rows),
            "mean": statistics.mean(row["prompt_tokens"] for row in resolved_rows),
        },
        "completion_tokens": {
            "minimum": min(row["completion_tokens"] for row in resolved_rows),
            "maximum": max(row["completion_tokens"] for row in resolved_rows),
            "mean": statistics.mean(row["completion_tokens"] for row in resolved_rows),
        },
        "metrics": metrics,
    }
    write_json_fresh(aggregate_path, aggregate)
    return aggregate


def finalize_partial_length_retry(config: FewshotConfig, config_path: Path, model_key: str) -> dict[str, Any]:
    """Finalize a deterministic retry that still contains a generation failure.

    Failed responses remain in the resolved artifact and are excluded from
    score metrics.  Parse coverage is reported per condition, so a malformed
    decoder response cannot silently become a score.
    """
    spec = config.model(model_key)
    _, shot_manifest_sha = _verify_protocol(config)
    initial_path = restricted_run_dir(config) / "models" / model_key / "predictions.jsonl"
    retry_path = restricted_run_dir(config) / "models" / model_key / f"length_retry_predictions.max{LENGTH_RETRY_MAX_TOKENS}.jsonl"
    resolved_path = restricted_run_dir(config) / "models" / model_key / "predictions.resolved.jsonl"
    aggregate_path = public_run_dir(config) / "models" / model_key / "aggregate.json"
    need(initial_path.is_file() and retry_path.is_file(), "partial retry artifacts are unavailable")
    need(not resolved_path.exists() and not aggregate_path.exists(), "partial retry finalization requires fresh outputs")
    initial, retry_rows = _jsonl(initial_path), _jsonl(retry_path)
    failures = [row for row in initial if row.get("parse_valid") is False]
    need(len(initial) == 2 * EXPECTED_ESSAYS["validation"] and len(retry_rows) == len(failures), "partial retry population differs")
    failure_keys = {(row["source_id"], row["condition"]) for row in failures}
    retry_by_key = {(row["source_id"], row["condition"]): row for row in retry_rows}
    need(len(retry_by_key) == len(retry_rows) and set(retry_by_key) == failure_keys, "partial retry keys differ")
    need(all(row.get("integration_retry", {}).get("max_tokens") == LENGTH_RETRY_MAX_TOKENS for row in retry_rows), "partial retry ceiling differs")
    need(all(row.get("parse_valid") is True or (row.get("finish_reason") == "length" and row.get("completion_tokens") == LENGTH_RETRY_MAX_TOKENS) for row in retry_rows), "partial retry has an unsupported failure")
    resolved_rows = [retry_by_key.get((row["source_id"], row["condition"]), row) for row in initial]
    resolved_sha = write_jsonl_fresh(resolved_path, resolved_rows)
    invalid = [row for row in resolved_rows if row.get("parse_valid") is not True]
    need(bool(invalid), "partial retry finalizer requires a remaining failure")
    metrics: dict[str, Any] = {}
    for condition in CONDITIONS:
        valid = [row for row in resolved_rows if row["condition"] == condition and row.get("parse_valid") is True]
        metrics[condition] = condition_metrics(valid, expected_count=len(valid), total_count=EXPECTED_ESSAYS["validation"])
    visible = [int(item) for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    aggregate = {
        "schema_version": "mal2026-decoder-fewshot-model-result-v1",
        "status": "completed",
        "run_id": config.run_id,
        "model_key": model_key,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "physical_gpus": visible,
        "tensor_parallel_size": spec.tensor_parallel_size,
        "config_sha256": file_sha256(config_path),
        "shot_manifest_sha256": shot_manifest_sha,
        "prediction_sha256": resolved_sha,
        "initial_prediction_sha256": file_sha256(initial_path),
        "length_retry_prediction_sha256": file_sha256(retry_path),
        "validation_rows": EXPECTED_ESSAYS["validation"],
        "requests": len(resolved_rows),
        "parse_failures": len(invalid),
        "parse_failure_policy": "exclude from score metrics; retain raw failed response",
        "integration_retry": {
            "reason": "structured JSON truncated exactly at initial max_tokens",
            "initial_max_tokens": config.max_tokens,
            "retry_max_tokens": LENGTH_RETRY_MAX_TOKENS,
            "retried_requests": len(retry_rows),
            "unresolved_requests": len(invalid),
            "other_variables_changed": False,
        },
        "prompt_tokens": {
            "minimum": min(row["prompt_tokens"] for row in resolved_rows),
            "maximum": max(row["prompt_tokens"] for row in resolved_rows),
            "mean": statistics.mean(row["prompt_tokens"] for row in resolved_rows),
        },
        "completion_tokens": {
            "minimum": min(row["completion_tokens"] for row in resolved_rows),
            "maximum": max(row["completion_tokens"] for row in resolved_rows),
            "mean": statistics.mean(row["completion_tokens"] for row in resolved_rows),
        },
        "metrics": metrics,
    }
    write_json_fresh(aggregate_path, aggregate)
    return aggregate


def aggregate_models(config: FewshotConfig, config_path: Path) -> dict[str, Any]:
    model_rows: dict[str, Any] = {}
    for spec in config.models:
        path = public_run_dir(config) / "models" / spec.key / "aggregate.json"
        need(path.is_file() and not path.is_symlink(), f"model aggregate missing: {spec.key}")
        row = json.loads(path.read_text(encoding="utf-8"))
        need(row.get("status") == "completed" and row.get("model_id") == spec.model_id and row.get("model_revision") == spec.revision, "model aggregate binding differs")
        model_rows[spec.key] = row
    comparisons: dict[str, Any] = {}
    for key, row in model_rows.items():
        balanced, central = row["metrics"]["balanced5"], row["metrics"]["central5"]
        comparisons[key] = {
            "balanced_minus_central_macro_raw_rmse": balanced["macro_raw_rmse"] - central["macro_raw_rmse"],
            "balanced_minus_central_macro_raw_spearman": balanced["macro_raw_spearman"] - central["macro_raw_spearman"],
            "balanced_minus_central_score_3_rate": balanced["macro_score_3_rate"] - central["macro_score_3_rate"],
            "balanced_minus_central_score_4_rate": balanced["macro_score_4_rate"] - central["macro_score_4_rate"],
        }
    result = {
        "schema_version": "mal2026-decoder-fewshot-validation-aggregate-v1",
        "status": "completed",
        "run_id": config.run_id,
        "config_sha256": file_sha256(config_path),
        "validation_rows": EXPECTED_ESSAYS["validation"],
        "conditions": list(CONDITIONS),
        "average_target_used": False,
        "models": {key: {"model_id": row["model_id"], "model_revision": row["model_revision"], "metrics": row["metrics"]} for key, row in model_rows.items()},
        "condition_comparisons": comparisons,
    }
    write_json_fresh(public_run_dir(config) / "aggregate.json", result)
    return result
