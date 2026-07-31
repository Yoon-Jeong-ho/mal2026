"""Round-3 Solar prompt search with same-topic, axis-specific score grids."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .api_rationale_data import AXES, WritingRow, load_writing_rows, sha256_file
from .solar_prompt_search import AXIS_BANDS, AXIS_DEFINITIONS, ROOT, _post_json, _write_json_fresh, _write_jsonl_fresh, file_sha256, metrics, need, now
from .solar_prompt_search_v2 import _canonical_map, _retrieval_state


SCHEMA_VERSION = "mal2026-solar-prompt-search-config-v3"
EXPECTED_RUN_ID = "solar-prompt-search-v3-20260801-003"
CANDIDATES = (
    "topic_grid5_axis_direct",
    "topic_grid7_axis_direct",
    "topic_grid9_axis_direct",
    "topic_grid7_axis_survival",
    "topic_grid7_axis_reflect",
)
RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_prompt_search_v3"
PUBLIC_ROOT = ROOT / "outputs/analysis"
RUNTIME_ROOT = ROOT / "outputs/solar-prompt-search-v3"


@dataclass(frozen=True)
class SearchConfigV3:
    schema_version: str
    run_id: str
    seed: int
    split_seed: int
    discovery_rows: int
    confirmation_rows: int
    target_raw_rmse: float
    endpoint: str
    model_alias: str
    model_id: str
    model_path: str
    docker_image: str
    gpu_scope: tuple[int, ...]
    max_model_len: int
    max_tokens: int
    retry_max_tokens: int
    max_inflight: int
    temperature: float
    embedding_rows_path: str
    embedding_rows_sha256: str
    retrieval_pool_policy: str
    candidates: tuple[str, ...]
    canonical_train_sha256: str

    @classmethod
    def from_json(cls, path: Path) -> "SearchConfigV3":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["gpu_scope"] = tuple(raw["gpu_scope"])
        raw["candidates"] = tuple(raw["candidates"])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        need(self.schema_version == SCHEMA_VERSION and self.run_id == EXPECTED_RUN_ID, "v3 identity differs")
        need((self.seed, self.split_seed) == (2026080107, 2026080105), "v3 seeds differ")
        need((self.discovery_rows, self.confirmation_rows) == (160, 400), "v3 split sizes differ")
        need(self.target_raw_rmse == 0.4 and self.temperature == 0.0, "v3 metric contract differs")
        need(self.endpoint == "http://127.0.0.1:19430" and self.gpu_scope == (0, 1, 2, 3), "v3 runtime differs")
        need((self.max_model_len, self.max_tokens, self.retry_max_tokens, self.max_inflight) == (12288, 256, 768, 64), "v3 capacity differs")
        need(self.candidates == CANDIDATES, "v3 candidates differ")
        need(self.retrieval_pool_policy == "same_topic_score_grid_from_pool_excluding_discovery_and_confirmation", "v3 pool policy differs")
        need(file_sha256(Path(self.embedding_rows_path)) == self.embedding_rows_sha256, "v3 embedding differs")


def restricted_dir(config: SearchConfigV3) -> Path:
    return RESTRICTED_ROOT / config.run_id


def public_dir(config: SearchConfigV3) -> Path:
    return PUBLIC_ROOT / config.run_id


def runtime_dir(config: SearchConfigV3) -> Path:
    return RUNTIME_ROOT / config.run_id


def _ledger(config: SearchConfigV3, value: Mapping[str, Any]) -> None:
    path = runtime_dir(config) / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": now(), **value}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def train_splits(config: SearchConfigV3) -> dict[str, list[WritingRow]]:
    need(sha256_file(ROOT / "eval/train.jsonl") == config.canonical_train_sha256, "canonical train differs")
    rows = load_writing_rows("train", include_scores=True)
    ordered = sorted(rows, key=lambda row: sha256(f"{config.split_seed}:{row.identifier}".encode()).hexdigest())
    return {"discovery": ordered[:160], "confirmation": ordered[160:560]}


def _grid_count(candidate: str) -> int:
    if "grid5" in candidate:
        return 5
    if "grid9" in candidate:
        return 9
    return 7


@lru_cache(maxsize=8192)
def score_grid(config: SearchConfigV3, source_id: str, axis: str, count: int) -> tuple[WritingRow, ...]:
    identifiers, embedding, artifact_rows, pool = _retrieval_state(config)  # structural protocol is identical to v2
    canonical = _canonical_map()
    query = canonical[source_id]
    same_topic = [int(index) for index in pool if canonical[artifact_rows[int(index)]["source_id"]].prompt == query.prompt]
    need(len(same_topic) >= count, "same-topic pool too small")
    similarities = {index: float(embedding[index] @ embedding[identifiers[source_id]]) for index in same_topic}
    targets = np.linspace(1.0, 5.0, count)
    unused = set(same_topic)
    selected = []
    for target in targets:
        index = min(unused, key=lambda item: (abs(float(canonical[artifact_rows[item]["source_id"]].scores[axis]) - float(target)) - 0.03 * similarities[item], -similarities[item], item))
        selected.append(index)
        unused.remove(index)
    rows = tuple(sorted((canonical[artifact_rows[index]["source_id"]] for index in selected), key=lambda row: (float(row.scores[axis]), row.identifier)))
    need(len(rows) == count and all(row.identifier != source_id for row in rows), "score grid differs")
    return rows


def _direct_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {"score": {"type": "number", "minimum": 1, "maximum": 5}}, "required": ["score"], "additionalProperties": False}


def _survival_schema(count: int) -> dict[str, Any]:
    return {"type": "object", "properties": {"at_least": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "minItems": count, "maxItems": count}}, "required": ["at_least"], "additionalProperties": False}


def _messages(row: WritingRow, axis: str, examples: Sequence[WritingRow], survival: bool) -> list[dict[str, str]]:
    bands = "\n".join(AXIS_BANDS[axis])
    if survival:
        output = f"아래 {len(examples)}개 예시 각각보다 현재 글의 {axis} 수행이 같거나 높을 확률을 예시 순서대로 at_least 배열에 0~1로 출력하라. 예시 점수가 높아질수록 확률은 증가할 수 없다."
    else:
        output = "현재 글의 연속 점수를 score에 1~5 숫자로 출력하라. 두 예시 사이이면 소수를 사용하라."
    system = f"""너는 한국어 논증적 글의 {axis} 축만 평가한다.
평가 대상: {AXIS_DEFINITIONS[axis]}
점수 기준:\n{bands}
훈련 예시는 같은 주제에서 낮은 점수부터 높은 점수까지 정렬된 비교 anchor다.
현재 글을 각 예시와 직접 비교하되 예시 점수의 평균이나 빈도를 정답으로 사용하지 마라.
길이·표지어·개행 수를 기계적 기준으로 사용하지 말고 담화 수행의 심각도·범위·영향을 보라.
{output} JSON 객체 하나만 출력하라."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for index, example in enumerate(examples, 1):
        messages.extend([
            {"role": "user", "content": f"[같은 주제 예시 {index}]\n{example.essay}"},
            {"role": "assistant", "content": json.dumps({"score": float(example.scores[axis])}, ensure_ascii=False, separators=(",", ":"))},
        ])
    messages.append({"role": "user", "content": f"[평가할 주제]\n{row.prompt}\n\n[평가할 글]\n{row.essay}"})
    return messages


def request_specs(config: SearchConfigV3, candidate: str, row: WritingRow) -> list[dict[str, Any]]:
    need(candidate in CANDIDATES, "unknown v3 candidate")
    count = _grid_count(candidate)
    survival = candidate == "topic_grid7_axis_survival"
    return [{"axis": axis, "anchors": [float(example.scores[axis]) for example in score_grid(config, row.identifier, axis, count)], "messages": _messages(row, axis, score_grid(config, row.identifier, axis, count), survival), "schema": _survival_schema(count) if survival else _direct_schema()} for axis in AXES]


def _payload(config: SearchConfigV3, spec: Mapping[str, Any], max_tokens: int) -> dict[str, Any]:
    response_format = {"type": "json_schema", "json_schema": {"name": "solar_topic_grid_score", "strict": True, "schema": spec["schema"]}}
    return {"model": config.model_alias, "messages": spec["messages"], "temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens, "seed": config.seed, "response_format": response_format, "chat_template_kwargs": {"reasoning_effort": "none", "think_render_option": "preserved", "response_format": response_format}}


def _survival_expected(anchors: Sequence[float], probabilities: Sequence[float]) -> float:
    need(len(anchors) == len(probabilities) and len(anchors) > 1, "survival vector differs")
    repaired = []
    ceiling = 1.0
    for value in probabilities:
        need(math.isfinite(float(value)) and 0 <= float(value) <= 1, "survival probability differs")
        ceiling = min(ceiling, float(value))
        repaired.append(ceiling)
    x = [1.0, *[min(5.0, max(1.0, float(value))) for value in anchors], 5.0]
    q = [1.0, *repaired, 0.0]
    order = sorted(range(len(x)), key=lambda index: (x[index], -q[index]))
    x = [x[index] for index in order]
    q = [q[index] for index in order]
    expected = 1.0 + sum(0.5 * (q[index] + q[index + 1]) * (x[index + 1] - x[index]) for index in range(len(x) - 1))
    return min(5.0, max(1.0, expected))


def _resolve(config: SearchConfigV3, spec: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    for max_tokens in (config.max_tokens, config.retry_max_tokens):
        response = _post_json(config.endpoint + "/v1/chat/completions", _payload(config, spec, max_tokens))
        choice = response["choices"][0]
        text = choice["message"]["content"]
        parsed = None
        error = None
        try:
            value = json.loads(text)
            if "at_least" in value:
                parsed = _survival_expected(spec["anchors"], [float(item) for item in value["at_least"]])
            else:
                parsed = float(value["score"])
            need(math.isfinite(parsed) and 1 <= parsed <= 5, "v3 score differs")
        except Exception as exc:
            error = type(exc).__name__ + ":" + str(exc)
        attempt = {"axis": spec["axis"], "max_tokens": max_tokens, "finish_reason": choice.get("finish_reason"), "response": text, "parse_error": error, "usage": response.get("usage", {})}
        if parsed is not None and choice.get("finish_reason") != "length":
            return parsed, attempt
        if choice.get("finish_reason") != "length":
            break
    raise ValueError("v3 response did not parse")


def _request_one(config: SearchConfigV3, candidate: str, row: WritingRow) -> dict[str, Any]:
    prediction = {}
    attempts = []
    specs = request_specs(config, candidate, row)
    for spec in specs:
        try:
            score, attempt = _resolve(config, spec)
            attempts.append(attempt)
            if candidate == "topic_grid7_axis_reflect":
                reflected = dict(spec)
                reflected["messages"] = [*spec["messages"], {"role": "assistant", "content": json.dumps({"score": score}, separators=(",", ":"))}, {"role": "user", "content": "초기 점수가 실제로 어느 두 anchor 사이인지 다시 비교하여 과대·과소평가를 수정한 최종 score JSON만 출력하라."}]
                score, attempt = _resolve(config, reflected)
                attempts.append(attempt)
            prediction[spec["axis"]] = score
        except Exception as exc:
            return {"source_id": row.identifier, "candidate": candidate, "parse_valid": False, "prediction": None, "parse_error": type(exc).__name__ + ":" + str(exc), "attempts": attempts}
    return {"source_id": row.identifier, "candidate": candidate, "parse_valid": True, "prediction": prediction, "gold_raw": {axis: float(row.scores[axis]) for axis in AXES}, "attempts": attempts}


def prepare(config: SearchConfigV3, config_path: Path) -> dict[str, Any]:
    splits = train_splits(config)
    held = {row.identifier for values in splits.values() for row in values}
    pool = sorted(set(_canonical_map()) - held)
    manifest = {"schema_version": "mal2026-solar-prompt-search-v3-split-v1", "run_id": config.run_id, "discovery_source_ids": [row.identifier for row in splits["discovery"]], "confirmation_source_ids": [row.identifier for row in splits["confirmation"]], "retrieval_pool_source_ids": pool, "validation_records_read": 0}
    manifest_sha = _write_json_fresh(restricted_dir(config) / "split_and_pool_manifest.json", manifest)
    result = {"schema_version": "mal2026-solar-prompt-search-protocol-v3", "status": "prepared", "run_id": config.run_id, "config_sha256": file_sha256(config_path), "embedding_rows_sha256": config.embedding_rows_sha256, "split_and_pool_manifest_sha256": manifest_sha, "splits": {name: len(rows) for name, rows in splits.items()}, "retrieval_pool_rows": len(pool), "retrieval_pool_policy": config.retrieval_pool_policy, "candidates": list(CANDIDATES), "target_raw_rmse": config.target_raw_rmse, "validation_records_read": 0, "gpu_scope": list(config.gpu_scope)}
    _write_json_fresh(public_dir(config) / "protocol.json", result)
    return result


def preflight(config: SearchConfigV3) -> dict[str, Any]:
    from transformers import AutoTokenizer

    rows = train_splits(config)["discovery"]
    _retrieval_state(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True, local_files_only=True)
    lengths = []
    for row in rows:
        for candidate in CANDIDATES:
            for spec in request_specs(config, candidate, row):
                response_format = {"type": "json_schema", "json_schema": {"name": "solar_topic_grid_score", "strict": True, "schema": spec["schema"]}}
                encoded = tokenizer.apply_chat_template(spec["messages"], tokenize=True, add_generation_prompt=True, reasoning_effort="none", think_render_option="preserved", response_format=response_format)
                tokens = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
                lengths.append(len(tokens))
    need(max(lengths) + config.retry_max_tokens <= config.max_model_len, "v3 context preflight failed")
    smoke = [_request_one(config, candidate, rows[0]) for candidate in CANDIDATES]
    need(all(row["parse_valid"] for row in smoke), "v3 smoke failed")
    smoke_sha = _write_jsonl_fresh(restricted_dir(config) / "smoke.jsonl", smoke)
    result = {"schema_version": "mal2026-solar-prompt-search-preflight-v3", "status": "passed", "prompt_shapes_audited": len(lengths), "prompt_tokens_min": min(lengths), "prompt_tokens_max": max(lengths), "retry_max_tokens": config.retry_max_tokens, "real_smoke_candidates": list(CANDIDATES), "smoke_sha256": smoke_sha, "validation_records_read": 0}
    _write_json_fresh(public_dir(config) / "preflight.json", result)
    return result


def run_candidate(config: SearchConfigV3, candidate: str, split: str) -> dict[str, Any]:
    need(candidate in CANDIDATES and split in {"discovery", "confirmation"}, "v3 run selection differs")
    rows = train_splits(config)[split]
    _retrieval_state(config)
    for row in rows:
        for axis in AXES:
            score_grid(config, row.identifier, axis, _grid_count(candidate))
    _ledger(config, {"event": "candidate_started", "candidate": candidate, "split": split, "rows": len(rows), "validation_records_read": 0})
    resolved = {}
    with ThreadPoolExecutor(max_workers=config.max_inflight) as pool:
        futures = {pool.submit(_request_one, config, candidate, row): index for index, row in enumerate(rows)}
        step = max(1, len(rows) // 10)
        for future in as_completed(futures):
            resolved[futures[future]] = future.result()
            if len(resolved) % step == 0 or len(resolved) == len(rows):
                _ledger(config, {"event": "candidate_progress", "candidate": candidate, "split": split, "completed": len(resolved), "total": len(rows)})
    predictions = [resolved[index] for index in range(len(rows))]
    prediction_sha = _write_jsonl_fresh(restricted_dir(config) / split / f"{candidate}.jsonl", predictions)
    multiplier = 2 if candidate == "topic_grid7_axis_reflect" else 1
    result = {"schema_version": "mal2026-solar-prompt-search-result-v3", "status": "completed", "run_id": config.run_id, "candidate": candidate, "split": split, "requests": len(rows) * len(AXES) * multiplier, "prediction_sha256": prediction_sha, "metrics": metrics(predictions, len(rows)), "validation_records_read": 0, "temperature": config.temperature, "seed": config.seed}
    _write_json_fresh(public_dir(config) / split / f"{candidate}.json", result)
    _ledger(config, {"event": "candidate_completed", "candidate": candidate, "split": split, "macro_raw_rmse": result["metrics"]["macro_raw_rmse"], "macro_raw_spearman": result["metrics"]["macro_raw_spearman"], "parse_success_rate": result["metrics"]["parse_success_rate"], "validation_records_read": 0})
    return result


def aggregate_discovery(config: SearchConfigV3) -> dict[str, Any]:
    results = {candidate: json.loads((public_dir(config) / "discovery" / f"{candidate}.json").read_text())["metrics"] for candidate in CANDIDATES}
    ranking = sorted(results, key=lambda candidate: results[candidate]["macro_raw_rmse"])
    result = {"schema_version": "mal2026-solar-prompt-search-discovery-aggregate-v3", "status": "completed", "run_id": config.run_id, "ranking": ranking, "target_raw_rmse": config.target_raw_rmse, "target_met": results[ranking[0]]["macro_raw_rmse"] < config.target_raw_rmse, "results": results, "validation_records_read": 0}
    _write_json_fresh(public_dir(config) / "discovery" / "aggregate.json", result)
    return result
