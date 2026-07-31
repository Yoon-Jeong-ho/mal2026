"""Round-5 Solar base-centered pairwise ternary correction search."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .api_rationale_data import AXES, WritingRow, load_writing_rows, sha256_file
from .solar_prompt_search import (
    AXIS_DEFINITIONS,
    ROOT,
    _post_json,
    _write_json_fresh,
    _write_jsonl_fresh,
    file_sha256,
    need,
    now,
)
from .solar_prompt_search_v2 import _canonical_map, _retrieval_state


SCHEMA_VERSION = "mal2026-solar-prompt-search-config-v5"
EXPECTED_RUN_ID = "solar-prompt-search-v5-20260801-005"
CORRECTIONS = (0.25, 0.35, 0.5)
VERDICTS = ("lower", "same", "higher")
RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_prompt_search_v5"
PUBLIC_ROOT = ROOT / "outputs/analysis"
RUNTIME_ROOT = ROOT / "outputs/solar-prompt-search-v5"


@dataclass(frozen=True)
class SearchConfigV5:
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
    base_prediction_origin: str
    retrieval_pool_policy: str
    corrections: tuple[float, ...]
    direction_threshold: float
    canonical_train_sha256: str

    @classmethod
    def from_json(cls, path: Path) -> "SearchConfigV5":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["gpu_scope"] = tuple(raw["gpu_scope"])
        raw["corrections"] = tuple(float(value) for value in raw["corrections"])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        need(self.schema_version == SCHEMA_VERSION and self.run_id == EXPECTED_RUN_ID, "v5 identity differs")
        need((self.seed, self.split_seed) == (2026080109, 2026080105), "v5 seeds differ")
        need((self.discovery_rows, self.confirmation_rows) == (160, 400), "v5 split sizes differ")
        need(self.endpoint == "http://127.0.0.1:19430" and self.gpu_scope == (0, 1, 2, 3), "v5 runtime differs")
        need((self.max_model_len, self.max_tokens, self.retry_max_tokens, self.max_inflight) == (12288, 128, 256, 64), "v5 capacity differs")
        need(self.temperature == 0.0 and self.target_raw_rmse == 0.4, "v5 metric contract differs")
        need(self.base_prediction_origin == "five_fold_oof_r0", "v5 base origin differs")
        need(self.corrections == CORRECTIONS and self.direction_threshold == 0.25, "v5 correction contract differs")
        need(self.retrieval_pool_policy == "base_centered_low_high_human_anchors_from_1440_pool_prefer_same_topic_then_semantic_similarity", "v5 retrieval policy differs")
        need(file_sha256(Path(self.embedding_rows_path)) == self.embedding_rows_sha256, "v5 embedding artifact differs")


def restricted_dir(config: SearchConfigV5) -> Path:
    return RESTRICTED_ROOT / config.run_id


def public_dir(config: SearchConfigV5) -> Path:
    return PUBLIC_ROOT / config.run_id


def runtime_dir(config: SearchConfigV5) -> Path:
    return RUNTIME_ROOT / config.run_id


def _ledger(config: SearchConfigV5, value: Mapping[str, Any]) -> None:
    path = runtime_dir(config) / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": now(), **value}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def train_splits(config: SearchConfigV5) -> dict[str, list[WritingRow]]:
    need(sha256_file(ROOT / "eval/train.jsonl") == config.canonical_train_sha256, "canonical train differs")
    rows = load_writing_rows("train", include_scores=True)
    ordered = sorted(rows, key=lambda row: sha256(f"{config.split_seed}:{row.identifier}".encode()).hexdigest())
    discovery = ordered[: config.discovery_rows]
    confirmation = ordered[config.discovery_rows : config.discovery_rows + config.confirmation_rows]
    need(not ({row.identifier for row in discovery} & {row.identifier for row in confirmation}), "v5 splits overlap")
    return {"discovery": discovery, "confirmation": confirmation}


@lru_cache(maxsize=1)
def _artifact_map(config: SearchConfigV5) -> dict[str, dict[str, Any]]:
    _, _, rows, _ = _retrieval_state(config)
    return {row["source_id"]: row for row in rows}


def base(config: SearchConfigV5, source_id: str) -> dict[str, float]:
    return {axis: float(_artifact_map(config)[source_id]["base_continuous_prediction"][axis]) for axis in AXES}


@lru_cache(maxsize=8192)
def select_anchors(config: SearchConfigV5, source_id: str, axis: str) -> tuple[WritingRow, WritingRow]:
    """Return human-scored anchors immediately below/above the query OOF base."""
    need(axis in AXES, "unknown v5 axis")
    identifiers, embedding, artifact_rows, pool = _retrieval_state(config)
    canonical = _canonical_map()
    query = canonical[source_id]
    query_base = base(config, source_id)[axis]

    def choose(direction: str) -> WritingRow:
        eligible = []
        for raw_index in pool:
            index = int(raw_index)
            row = canonical[artifact_rows[index]["source_id"]]
            score = float(row.scores[axis])
            if (direction == "low" and score < query_base) or (direction == "high" and score > query_base):
                eligible.append((index, row, score))
        need(bool(eligible), f"no {direction} v5 anchor")
        same_topic = [item for item in eligible if item[1].prompt == query.prompt]
        candidates = same_topic or eligible
        index, row, _ = min(
            candidates,
            key=lambda item: (
                abs(item[2] - query_base),
                -float(embedding[item[0]] @ embedding[identifiers[source_id]]),
                item[1].identifier,
            ),
        )
        need(index != identifiers[source_id], "v5 self anchor occurred")
        return row

    low, high = choose("low"), choose("high")
    need(float(low.scores[axis]) < query_base < float(high.scores[axis]), "v5 anchors do not bracket base")
    return low, high


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)}},
        "required": ["verdict"],
        "additionalProperties": False,
    }


def _messages(row: WritingRow, axis: str, anchors: Sequence[tuple[str, WritingRow]], base_value: float) -> list[dict[str, str]]:
    system = f"""너는 한국어 논증적 글의 {axis} 축만 평가한다.
평가 대상은 {AXIS_DEFINITIONS[axis]}이다.
OOF base는 5-fold OOF 인코더가 현재 글에 부여한 강한 기준값이다.
두 기준 글은 사람 평가에서 각각 base보다 낮거나 높은 수행을 보인 글이다. 기준 글의 숨겨진 숫자 점수를 추측하거나 출력하지 마라.
현재 글이 OOF base보다 명확히 낮으면 lower, 실질적으로 같으면 same, 명확히 높으면 higher로 판정하라.
길이·표지어·개행 수를 기계적 기준으로 쓰지 말고, 기준 글의 제시 순서에도 영향을 받지 마라.
verdict 키에 lower, same, higher 중 하나만 담은 JSON 객체를 출력하라."""
    content = [f"[평가 축]\n{axis}", f"[OOF base]\n{base_value:.6f}"]
    for role, anchor in anchors:
        content.append(f"[{role}]\n{anchor.essay}")
    content.append(f"[평가할 주제]\n{row.prompt}\n\n[평가할 글]\n{row.essay}")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(content)}]


def request_specs(config: SearchConfigV5, row: WritingRow) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for axis in AXES:
        low, high = select_anchors(config, row.identifier, axis)
        orders = (
            ("A", (("base보다 낮은 기준 글", low), ("base보다 높은 기준 글", high))),
            ("B", (("base보다 높은 기준 글", high), ("base보다 낮은 기준 글", low))),
        )
        for order, anchors in orders:
            specs.append({
                "axis": axis,
                "order": order,
                "anchor_source_ids": [anchor.identifier for _, anchor in anchors],
                "messages": _messages(row, axis, anchors, base(config, row.identifier)[axis]),
                "schema": _schema(),
            })
    return specs


def _payload(config: SearchConfigV5, spec: Mapping[str, Any], max_tokens: int) -> dict[str, Any]:
    response_format = {"type": "json_schema", "json_schema": {"name": "solar_pairwise_ternary", "strict": True, "schema": spec["schema"]}}
    return {
        "model": config.model_alias,
        "messages": spec["messages"],
        "temperature": config.temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": config.seed,
        "response_format": response_format,
        "chat_template_kwargs": {"reasoning_effort": "none", "think_render_option": "preserved", "response_format": response_format},
    }


def _resolve(config: SearchConfigV5, spec: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    attempts = []
    for max_tokens in (config.max_tokens, config.retry_max_tokens):
        response = _post_json(config.endpoint + "/v1/chat/completions", _payload(config, spec, max_tokens))
        choice = response["choices"][0]
        text = choice["message"]["content"]
        verdict = None
        error = None
        try:
            value = json.loads(text)
            need(set(value) == {"verdict"} and value["verdict"] in VERDICTS, "v5 verdict differs")
            verdict = value["verdict"]
        except Exception as exc:
            error = type(exc).__name__ + ":" + str(exc)
        attempt = {"max_tokens": max_tokens, "finish_reason": choice.get("finish_reason"), "response": text, "parse_error": error, "usage": response.get("usage", {})}
        attempts.append(attempt)
        if verdict is not None and choice.get("finish_reason") != "length":
            return verdict, {"axis": spec["axis"], "order": spec["order"], "attempts": attempts}
        if choice.get("finish_reason") != "length":
            break
    raise ValueError("v5 response did not parse")


def _request_one(config: SearchConfigV5, row: WritingRow) -> dict[str, Any]:
    judgments: dict[str, dict[str, str]] = {axis: {} for axis in AXES}
    request_records = []
    try:
        for spec in request_specs(config, row):
            verdict, record = _resolve(config, spec)
            judgments[spec["axis"]][spec["order"]] = verdict
            request_records.append(record)
    except Exception as exc:
        return {"source_id": row.identifier, "parse_valid": False, "parse_error": type(exc).__name__ + ":" + str(exc), "requests": request_records}
    final = {}
    agreement = {}
    for axis in AXES:
        agreement[axis] = judgments[axis]["A"] == judgments[axis]["B"]
        final[axis] = judgments[axis]["A"] if agreement[axis] else "same"
    return {
        "source_id": row.identifier,
        "parse_valid": True,
        "base_prediction": base(config, row.identifier),
        "gold_raw": {axis: float(row.scores[axis]) for axis in AXES},
        "judgments": judgments,
        "agreement": agreement,
        "final_verdict": final,
        "requests": request_records,
    }


def _balanced_accuracy(true_values: Sequence[str], predicted_values: Sequence[str]) -> float | None:
    recalls = []
    for label in ("lower", "higher"):
        indices = [index for index, value in enumerate(true_values) if value == label]
        if indices:
            recalls.append(sum(predicted_values[index] == label for index in indices) / len(indices))
    return (sum(recalls) / len(recalls)) if recalls else None


def _macro_rmse(rows: Sequence[Mapping[str, Any]], prediction_key: str, indices: Sequence[int] | None = None) -> tuple[float, dict[str, float]]:
    chosen = list(range(len(rows))) if indices is None else list(indices)
    by_axis = {}
    for axis in AXES:
        squared = [
            (float(rows[index][prediction_key][axis]) - float(rows[index]["gold_raw"][axis])) ** 2
            for index in chosen
        ]
        by_axis[axis] = math.sqrt(sum(squared) / len(squared))
    return sum(by_axis.values()) / len(AXES), by_axis


def _paired_bootstrap_delta_ci(rows: Sequence[Mapping[str, Any]], seed: int, samples: int = 2000) -> list[float] | None:
    if not rows:
        return None
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(len(rows)) for _ in rows]
        candidate, _ = _macro_rmse(rows, "prediction", indices)
        baseline, _ = _macro_rmse(rows, "base_prediction", indices)
        deltas.append(candidate - baseline)
    deltas.sort()
    return [deltas[int(0.025 * samples)], deltas[min(samples - 1, int(0.975 * samples))]]


def evaluate(rows: Sequence[Mapping[str, Any]], correction: float, expected_rows: int, seed: int = 2026080109) -> dict[str, Any]:
    valid = [row for row in rows if row.get("parse_valid")]
    per_axis: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        squared = []
        truths = []
        predictions = []
        agreements = []
        order_values: dict[str, list[int]] = {"A": [], "B": []}
        order_counts = {order: {verdict: 0 for verdict in VERDICTS} for order in ("A", "B")}
        eligible = 0
        for row in valid:
            direction = row["final_verdict"][axis]
            sign = {"lower": -1.0, "same": 0.0, "higher": 1.0}[direction]
            prediction = min(5.0, max(1.0, float(row["base_prediction"][axis]) + correction * sign))
            gold = float(row["gold_raw"][axis])
            squared.append((prediction - gold) ** 2)
            agreements.append(bool(row["agreement"][axis]))
            for order in ("A", "B"):
                ordered_verdict = row["judgments"][axis][order]
                order_counts[order][ordered_verdict] += 1
                order_values[order].append({"lower": -1, "same": 0, "higher": 1}[ordered_verdict])
            residual = gold - float(row["base_prediction"][axis])
            if abs(residual) >= 0.25:
                eligible += 1
                truths.append("higher" if residual > 0 else "lower")
                predictions.append(direction)
        per_axis[axis] = {
            "raw_rmse": math.sqrt(sum(squared) / len(squared)) if squared else None,
            "direction_balanced_accuracy": _balanced_accuracy(truths, predictions),
            "direction_eligible_rows": eligible,
            "consistency_rate": sum(agreements) / len(agreements) if agreements else None,
            "position_bias": {
                "verdict_counts_by_order": order_counts,
                "signed_mean_a_minus_b": (
                    sum(order_values["A"]) / len(order_values["A"])
                    - sum(order_values["B"]) / len(order_values["B"])
                ) if valid else None,
            },
        }
    direction_scores = [per_axis[axis]["direction_balanced_accuracy"] for axis in AXES if per_axis[axis]["direction_balanced_accuracy"] is not None]
    evaluated_rows = []
    for row in valid:
        prediction = {}
        for axis in AXES:
            direction = row["final_verdict"][axis]
            sign = {"lower": -1.0, "same": 0.0, "higher": 1.0}[direction]
            prediction[axis] = min(5.0, max(1.0, float(row["base_prediction"][axis]) + correction * sign))
        evaluated_rows.append({**row, "prediction": prediction})
    if evaluated_rows:
        macro_rmse, candidate_axes = _macro_rmse(evaluated_rows, "prediction")
        base_macro_rmse, base_axes = _macro_rmse(evaluated_rows, "base_prediction")
        delta_axes = {axis: candidate_axes[axis] - base_axes[axis] for axis in AXES}
        paired_ci = _paired_bootstrap_delta_ci(evaluated_rows, seed + int(correction * 1000))
    else:
        macro_rmse = base_macro_rmse = None
        base_axes = delta_axes = {axis: None for axis in AXES}
        paired_ci = None
    macro_direction = sum(direction_scores) / len(direction_scores) if direction_scores else None
    organization_delta = delta_axes["organization"]
    acceptance = {
        "direction_balanced_accuracy_at_least_0_65": macro_direction is not None and macro_direction >= 0.65,
        "macro_rmse_improvement_at_least_0_02": macro_rmse is not None and base_macro_rmse - macro_rmse >= 0.02,
        "organization_rmse_improvement_at_least_0_03": organization_delta is not None and organization_delta <= -0.03,
        "paired_bootstrap_delta_ci_entirely_below_zero": paired_ci is not None and paired_ci[1] < 0.0,
    }
    acceptance["passed_all"] = all(acceptance.values())
    return {
        "correction": correction,
        "parse_success_rate": len(valid) / expected_rows,
        "per_axis": per_axis,
        "macro_raw_rmse": macro_rmse,
        "base_macro_raw_rmse": base_macro_rmse,
        "raw_rmse_delta_vs_base": None if macro_rmse is None else macro_rmse - base_macro_rmse,
        "base_raw_rmse_by_axis": base_axes,
        "raw_rmse_delta_vs_base_by_axis": delta_axes,
        "paired_bootstrap_macro_rmse_delta_95ci": paired_ci,
        "paired_bootstrap_samples": 2000,
        "macro_direction_balanced_accuracy": macro_direction,
        "macro_consistency_rate": sum(per_axis[axis]["consistency_rate"] for axis in AXES) / len(AXES) if valid else None,
        "acceptance": acceptance,
    }


def prepare(config: SearchConfigV5, config_path: Path) -> dict[str, Any]:
    splits = train_splits(config)
    held = {row.identifier for values in splits.values() for row in values}
    pool = sorted(set(_canonical_map()) - held)
    need(len(pool) == 1440, "v5 retrieval pool population differs")
    manifest = {
        "schema_version": "mal2026-solar-prompt-search-v5-split-v1",
        "run_id": config.run_id,
        "discovery_source_ids": [row.identifier for row in splits["discovery"]],
        "confirmation_source_ids": [row.identifier for row in splits["confirmation"]],
        "retrieval_pool_source_ids": pool,
        "validation_records_read": 0,
    }
    manifest_sha = _write_json_fresh(restricted_dir(config) / "split_and_pool_manifest.json", manifest)
    result = {
        "schema_version": "mal2026-solar-prompt-search-protocol-v5",
        "status": "prepared",
        "run_id": config.run_id,
        "config_sha256": file_sha256(config_path),
        "embedding_rows_sha256": config.embedding_rows_sha256,
        "split_and_pool_manifest_sha256": manifest_sha,
        "base_prediction_origin": config.base_prediction_origin,
        "splits": {name: len(rows) for name, rows in splits.items()},
        "retrieval_pool_rows": len(pool),
        "corrections": list(config.corrections),
        "direction_threshold": config.direction_threshold,
        "validation_records_read": 0,
        "gpu_scope": list(config.gpu_scope),
    }
    _write_json_fresh(public_dir(config) / "protocol.json", result)
    _ledger(config, {"event": "prepared", "validation_records_read": 0, "gpu_scope": list(config.gpu_scope)})
    return result


def preflight(config: SearchConfigV5) -> dict[str, Any]:
    from transformers import AutoTokenizer

    row = train_splits(config)["discovery"][0]
    _retrieval_state(config)
    specs = request_specs(config, row)
    need(len(specs) == len(AXES) * 2 and all(spec["schema"] == _schema() for spec in specs), "v5 schema preflight failed")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True, local_files_only=True)
    lengths = []
    for spec in specs:
        response_format = {"type": "json_schema", "json_schema": {"name": "solar_pairwise_ternary", "strict": True, "schema": spec["schema"]}}
        encoded = tokenizer.apply_chat_template(spec["messages"], tokenize=True, add_generation_prompt=True, reasoning_effort="none", think_render_option="preserved", response_format=response_format)
        tokens = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        lengths.append(len(tokens))
    need(max(lengths) + config.retry_max_tokens <= config.max_model_len, "v5 context preflight failed")
    smoke = _request_one(config, row)
    need(smoke["parse_valid"], "v5 real smoke failed")
    smoke_sha = _write_jsonl_fresh(restricted_dir(config) / "smoke.jsonl", [smoke])
    result = {
        "schema_version": "mal2026-solar-prompt-search-preflight-v5",
        "status": "passed",
        "json_schema_audited": True,
        "prompt_shapes_audited": len(lengths),
        "prompt_tokens_min": min(lengths),
        "prompt_tokens_max": max(lengths),
        "retry_max_tokens": config.retry_max_tokens,
        "real_smoke_rows": 1,
        "real_smoke_requests": len(specs),
        "smoke_sha256": smoke_sha,
        "validation_records_read": 0,
    }
    _write_json_fresh(public_dir(config) / "preflight.json", result)
    _ledger(config, {"event": "preflight_passed", "smoke_sha256": smoke_sha, "validation_records_read": 0})
    return result


def run_split(config: SearchConfigV5, split: str) -> dict[str, Any]:
    need(split in {"discovery", "confirmation"}, "v5 split differs")
    rows = train_splits(config)[split]
    _retrieval_state(config)
    for row in rows:
        request_specs(config, row)
    _ledger(config, {"event": "split_started", "split": split, "rows": len(rows), "requests": len(rows) * len(AXES) * 2, "validation_records_read": 0})
    resolved = {}
    with ThreadPoolExecutor(max_workers=config.max_inflight) as pool:
        futures = {pool.submit(_request_one, config, row): index for index, row in enumerate(rows)}
        step = max(1, len(rows) // 10)
        for future in as_completed(futures):
            resolved[futures[future]] = future.result()
            if len(resolved) % step == 0 or len(resolved) == len(rows):
                _ledger(config, {"event": "split_progress", "split": split, "completed": len(resolved), "total": len(rows)})
    predictions = [resolved[index] for index in range(len(rows))]
    prediction_sha = _write_jsonl_fresh(restricted_dir(config) / split / "pairwise_ternary.jsonl", predictions)
    correction_metrics = {str(value): evaluate(predictions, value, len(rows), config.seed) for value in config.corrections}
    ranking = sorted(correction_metrics, key=lambda value: correction_metrics[value]["macro_raw_rmse"] if correction_metrics[value]["macro_raw_rmse"] is not None else math.inf)
    result = {
        "schema_version": "mal2026-solar-prompt-search-result-v5",
        "status": "completed",
        "run_id": config.run_id,
        "split": split,
        "rows": len(rows),
        "requests": len(rows) * len(AXES) * 2,
        "prediction_sha256": prediction_sha,
        "correction_ranking": ranking,
        "correction_metrics": correction_metrics,
        "validation_records_read": 0,
        "temperature": config.temperature,
        "seed": config.seed,
    }
    _write_json_fresh(public_dir(config) / split / "pairwise_ternary.json", result)
    best = ranking[0]
    _ledger(config, {"event": "split_completed", "split": split, "best_correction": best, "best_macro_raw_rmse": correction_metrics[best]["macro_raw_rmse"], "prediction_sha256": prediction_sha, "validation_records_read": 0})
    return result
