"""Round-4 Solar prompt search as a leakage-free correction of OOF R0 scores."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .api_rationale_data import AXES, WritingRow, load_writing_rows, sha256_file
from .solar_prompt_search import ROOT, _post_json, _write_json_fresh, _write_jsonl_fresh, file_sha256, metrics, need, now
from .solar_prompt_search_v2 import _canonical_map, _retrieval_state, nearest
from .solar_prompt_search_v3 import score_grid


SCHEMA_VERSION = "mal2026-solar-prompt-search-config-v4"
EXPECTED_RUN_ID = "solar-prompt-search-v4-20260801-004"
CANDIDATES = (
    "residual8_joint_final",
    "residual12_joint_final",
    "residual8_joint_delta",
    "residual8_axis_delta",
    "residual_topic_grid7_axis_delta",
)
RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_prompt_search_v4"
PUBLIC_ROOT = ROOT / "outputs/analysis"
RUNTIME_ROOT = ROOT / "outputs/solar-prompt-search-v4"


@dataclass(frozen=True)
class SearchConfigV4:
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
    candidates: tuple[str, ...]
    canonical_train_sha256: str

    @classmethod
    def from_json(cls, path: Path) -> "SearchConfigV4":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["gpu_scope"] = tuple(raw["gpu_scope"])
        raw["candidates"] = tuple(raw["candidates"])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        need(self.schema_version == SCHEMA_VERSION and self.run_id == EXPECTED_RUN_ID, "v4 identity differs")
        need((self.seed, self.split_seed) == (2026080108, 2026080105), "v4 seeds differ")
        need((self.discovery_rows, self.confirmation_rows) == (160, 400), "v4 split sizes differ")
        need(self.target_raw_rmse == 0.4 and self.temperature == 0.0, "v4 metric contract differs")
        need(self.endpoint == "http://127.0.0.1:19430" and self.gpu_scope == (0, 1, 2, 3), "v4 runtime differs")
        need((self.max_model_len, self.max_tokens, self.retry_max_tokens, self.max_inflight) == (12288, 256, 768, 64), "v4 capacity differs")
        need(self.base_prediction_origin == "five_fold_oof_r0" and self.candidates == CANDIDATES, "v4 protocol differs")
        need(file_sha256(Path(self.embedding_rows_path)) == self.embedding_rows_sha256, "v4 artifact differs")


def restricted_dir(config: SearchConfigV4) -> Path: return RESTRICTED_ROOT / config.run_id
def public_dir(config: SearchConfigV4) -> Path: return PUBLIC_ROOT / config.run_id
def runtime_dir(config: SearchConfigV4) -> Path: return RUNTIME_ROOT / config.run_id


def _ledger(config: SearchConfigV4, value: Mapping[str, Any]) -> None:
    path = runtime_dir(config) / "ledger.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps({"at": now(), **value}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def train_splits(config: SearchConfigV4) -> dict[str, list[WritingRow]]:
    need(sha256_file(ROOT / "eval/train.jsonl") == config.canonical_train_sha256, "canonical train differs")
    rows = load_writing_rows("train", include_scores=True)
    ordered = sorted(rows, key=lambda row: sha256(f"{config.split_seed}:{row.identifier}".encode()).hexdigest())
    return {"discovery": ordered[:160], "confirmation": ordered[160:560]}


@lru_cache(maxsize=1)
def _artifact_map(config: SearchConfigV4) -> dict[str, dict[str, Any]]:
    _, _, rows, _ = _retrieval_state(config)
    return {row["source_id"]: row for row in rows}


def base(config: SearchConfigV4, source_id: str) -> dict[str, float]:
    return {axis: float(_artifact_map(config)[source_id]["base_continuous_prediction"][axis]) for axis in AXES}


def _schema(keys: Sequence[str], delta: bool) -> dict[str, Any]:
    score = {"type": "number", "minimum": -1 if delta else 1, "maximum": 1 if delta else 5}
    return {"type": "object", "properties": {key: score for key in keys}, "required": list(keys), "additionalProperties": False}


def _system(axis: str | None, delta: bool) -> str:
    target = "세 축" if axis is None else axis
    output = "각 축의 delta를 -1~1로" if delta else "각 축의 최종 점수를 1~5 연속값으로"
    if axis is not None: output = "delta를 -1~1 score 키로" if delta else "최종 점수를 1~5 score 키로"
    return f"""너는 한국어 글의 {target} 점수를 보정한다. base는 5-fold OOF 인코더 예측이며 강한 기준값이다.
훈련 예시는 글, 해당 글의 OOF base, 사람 정답을 함께 보여준다. 현재 글에서 명확한 교정 근거가 있을 때만 base를 바꾸고, 단순히 예시 평균이나 점수 빈도에 맞추지 마라.
내용은 주장·근거·논리, 조직은 담화 기능·순서·연결, 표현은 명료성·어휘·언어 규범으로 판단한다.
{output} JSON 객체 하나만 출력하라."""


def _demo_messages(config: SearchConfigV4, rows: Sequence[WritingRow], axis: str | None, delta: bool) -> list[dict[str, str]]:
    messages = []
    for row in rows:
        b = base(config, row.identifier)
        if axis is None:
            assistant = {name: (float(row.scores[name]) - b[name]) if delta else float(row.scores[name]) for name in AXES}
            shown_base = b
        else:
            assistant = {"score": (float(row.scores[axis]) - b[axis]) if delta else float(row.scores[axis])}
            shown_base = {axis: b[axis]}
        messages.extend([{"role": "user", "content": f"[훈련 글]\n{row.essay}\n\n[OOF base]\n{json.dumps(shown_base, ensure_ascii=False)}"}, {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False, separators=(",", ":"))}])
    return messages


def request_specs(config: SearchConfigV4, candidate: str, row: WritingRow) -> list[dict[str, Any]]:
    need(candidate in CANDIDATES, "unknown v4 candidate")
    delta = "delta" in candidate
    if candidate == "residual_topic_grid7_axis_delta":
        specs = []
        for axis in AXES:
            examples = score_grid(config, row.identifier, axis, 7)
            current = {"role": "user", "content": f"[평가할 주제]\n{row.prompt}\n\n[평가할 글]\n{row.essay}\n\n[OOF base]\n{json.dumps({axis: base(config,row.identifier)[axis]}, ensure_ascii=False)}"}
            specs.append({"axis": axis, "delta": True, "messages": [{"role": "system", "content": _system(axis, True)}, *_demo_messages(config, examples, axis, True), current], "schema": _schema(["score"], True)})
        return specs
    count = 12 if candidate == "residual12_joint_final" else 8
    examples = nearest(config, row.identifier, count)
    if candidate == "residual8_axis_delta":
        return [{"axis": axis, "delta": True, "messages": [{"role": "system", "content": _system(axis, True)}, *_demo_messages(config, examples, axis, True), {"role": "user", "content": f"[평가할 주제]\n{row.prompt}\n\n[평가할 글]\n{row.essay}\n\n[OOF base]\n{json.dumps({axis:base(config,row.identifier)[axis]},ensure_ascii=False)}"}], "schema": _schema(["score"], True)} for axis in AXES]
    current = {"role": "user", "content": f"[평가할 주제]\n{row.prompt}\n\n[평가할 글]\n{row.essay}\n\n[OOF base]\n{json.dumps(base(config,row.identifier),ensure_ascii=False)}"}
    return [{"axis": None, "delta": delta, "messages": [{"role": "system", "content": _system(None, delta)}, *_demo_messages(config, examples, None, delta), current], "schema": _schema(AXES, delta)}]


def _payload(config: SearchConfigV4, spec: Mapping[str, Any], max_tokens: int) -> dict[str, Any]:
    response_format = {"type": "json_schema", "json_schema": {"name": "solar_oof_residual", "strict": True, "schema": spec["schema"]}}
    return {"model": config.model_alias, "messages": spec["messages"], "temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens, "seed": config.seed, "response_format": response_format, "chat_template_kwargs": {"reasoning_effort": "none", "think_render_option": "preserved", "response_format": response_format}}


def _resolve(config: SearchConfigV4, spec: Mapping[str, Any], source_id: str) -> tuple[dict[str, float], dict[str, Any]]:
    for max_tokens in (config.max_tokens, config.retry_max_tokens):
        response = _post_json(config.endpoint + "/v1/chat/completions", _payload(config, spec, max_tokens)); choice = response["choices"][0]; text = choice["message"]["content"]
        parsed = None; error = None
        try:
            value = json.loads(text); axes = AXES if spec["axis"] is None else (spec["axis"],)
            raw = {axis: float(value[axis] if spec["axis"] is None else value["score"]) for axis in axes}
            if spec["delta"]: raw = {axis: base(config, source_id)[axis] + score for axis, score in raw.items()}
            parsed = {axis: min(5.0, max(1.0, score)) for axis, score in raw.items()}
            need(all(math.isfinite(score) for score in parsed.values()), "v4 score differs")
        except Exception as exc: error = type(exc).__name__ + ":" + str(exc)
        attempt = {"axis": spec["axis"], "max_tokens": max_tokens, "finish_reason": choice.get("finish_reason"), "response": text, "parse_error": error, "usage": response.get("usage", {})}
        if parsed is not None and choice.get("finish_reason") != "length": return parsed, attempt
        if choice.get("finish_reason") != "length": break
    raise ValueError("v4 response did not parse")


def _request_one(config: SearchConfigV4, candidate: str, row: WritingRow) -> dict[str, Any]:
    prediction = {}; attempts = []
    for spec in request_specs(config, candidate, row):
        try: values, attempt = _resolve(config, spec, row.identifier); prediction.update(values); attempts.append(attempt)
        except Exception as exc: return {"source_id": row.identifier, "candidate": candidate, "parse_valid": False, "prediction": None, "parse_error": type(exc).__name__ + ":" + str(exc), "attempts": attempts}
    return {"source_id": row.identifier, "candidate": candidate, "parse_valid": True, "prediction": prediction, "base_prediction": base(config,row.identifier), "gold_raw": {axis: float(row.scores[axis]) for axis in AXES}, "attempts": attempts}


def prepare(config: SearchConfigV4, config_path: Path) -> dict[str, Any]:
    splits = train_splits(config); held = {row.identifier for values in splits.values() for row in values}; pool = sorted(set(_canonical_map()) - held)
    manifest = {"schema_version": "mal2026-solar-prompt-search-v4-split-v1", "run_id": config.run_id, "discovery_source_ids": [row.identifier for row in splits["discovery"]], "confirmation_source_ids": [row.identifier for row in splits["confirmation"]], "demonstration_pool_source_ids": pool, "validation_records_read": 0}
    manifest_sha = _write_json_fresh(restricted_dir(config) / "split_and_pool_manifest.json", manifest)
    result = {"schema_version": "mal2026-solar-prompt-search-protocol-v4", "status": "prepared", "run_id": config.run_id, "config_sha256": file_sha256(config_path), "embedding_rows_sha256": config.embedding_rows_sha256, "split_and_pool_manifest_sha256": manifest_sha, "base_prediction_origin": config.base_prediction_origin, "splits": {name: len(rows) for name, rows in splits.items()}, "demonstration_pool_rows": len(pool), "candidates": list(CANDIDATES), "target_raw_rmse": config.target_raw_rmse, "validation_records_read": 0, "gpu_scope": list(config.gpu_scope)}
    _write_json_fresh(public_dir(config) / "protocol.json", result); return result


def preflight(config: SearchConfigV4) -> dict[str, Any]:
    from transformers import AutoTokenizer
    rows = train_splits(config)["discovery"]; _retrieval_state(config); _artifact_map(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True, local_files_only=True); lengths = []
    for row in rows:
        for candidate in CANDIDATES:
            for spec in request_specs(config,candidate,row):
                response_format={"type":"json_schema","json_schema":{"name":"solar_oof_residual","strict":True,"schema":spec["schema"]}}
                encoded=tokenizer.apply_chat_template(spec["messages"],tokenize=True,add_generation_prompt=True,reasoning_effort="none",think_render_option="preserved",response_format=response_format);tokens=encoded["input_ids"] if isinstance(encoded,Mapping) else encoded;lengths.append(len(tokens))
    need(max(lengths)+config.retry_max_tokens<=config.max_model_len,"v4 context preflight failed")
    smoke=[_request_one(config,candidate,rows[0]) for candidate in CANDIDATES];need(all(row["parse_valid"] for row in smoke),"v4 smoke failed");smoke_sha=_write_jsonl_fresh(restricted_dir(config)/"smoke.jsonl",smoke)
    result={"schema_version":"mal2026-solar-prompt-search-preflight-v4","status":"passed","prompt_shapes_audited":len(lengths),"prompt_tokens_min":min(lengths),"prompt_tokens_max":max(lengths),"retry_max_tokens":config.retry_max_tokens,"real_smoke_candidates":list(CANDIDATES),"smoke_sha256":smoke_sha,"validation_records_read":0};_write_json_fresh(public_dir(config)/"preflight.json",result);return result


def run_candidate(config: SearchConfigV4,candidate:str,split:str)->dict[str,Any]:
    need(candidate in CANDIDATES and split in {"discovery","confirmation"},"v4 run selection differs");rows=train_splits(config)[split];_retrieval_state(config);_artifact_map(config)
    for row in rows: request_specs(config,candidate,row)
    _ledger(config,{"event":"candidate_started","candidate":candidate,"split":split,"rows":len(rows),"validation_records_read":0});resolved={}
    with ThreadPoolExecutor(max_workers=config.max_inflight) as pool:
        futures={pool.submit(_request_one,config,candidate,row):index for index,row in enumerate(rows)};step=max(1,len(rows)//10)
        for future in as_completed(futures):
            resolved[futures[future]]=future.result()
            if len(resolved)%step==0 or len(resolved)==len(rows):_ledger(config,{"event":"candidate_progress","candidate":candidate,"split":split,"completed":len(resolved),"total":len(rows)})
    predictions=[resolved[index] for index in range(len(rows))];prediction_sha=_write_jsonl_fresh(restricted_dir(config)/split/f"{candidate}.jsonl",predictions)
    base_rows=[{"parse_valid":True,"prediction":row["base_prediction"],"gold_raw":row["gold_raw"]} for row in predictions if row.get("parse_valid")]
    result={"schema_version":"mal2026-solar-prompt-search-result-v4","status":"completed","run_id":config.run_id,"candidate":candidate,"split":split,"requests":sum(len(request_specs(config,candidate,row)) for row in rows),"prediction_sha256":prediction_sha,"base_metrics":metrics(base_rows,len(rows)),"metrics":metrics(predictions,len(rows)),"validation_records_read":0,"temperature":config.temperature,"seed":config.seed};_write_json_fresh(public_dir(config)/split/f"{candidate}.json",result)
    _ledger(config,{"event":"candidate_completed","candidate":candidate,"split":split,"macro_raw_rmse":result["metrics"]["macro_raw_rmse"],"base_macro_raw_rmse":result["base_metrics"]["macro_raw_rmse"],"macro_raw_spearman":result["metrics"]["macro_raw_spearman"],"parse_success_rate":result["metrics"]["parse_success_rate"],"validation_records_read":0});return result


def aggregate_discovery(config: SearchConfigV4)->dict[str,Any]:
    full={candidate:json.loads((public_dir(config)/"discovery"/f"{candidate}.json").read_text()) for candidate in CANDIDATES};results={candidate:row["metrics"] for candidate,row in full.items()};ranking=sorted(results,key=lambda candidate:results[candidate]["macro_raw_rmse"]);base_metrics=full[CANDIDATES[0]]["base_metrics"]
    result={"schema_version":"mal2026-solar-prompt-search-discovery-aggregate-v4","status":"completed","run_id":config.run_id,"ranking":ranking,"target_raw_rmse":config.target_raw_rmse,"target_met":results[ranking[0]]["macro_raw_rmse"]<config.target_raw_rmse,"base_metrics":base_metrics,"results":results,"validation_records_read":0};_write_json_fresh(public_dir(config)/"discovery"/"aggregate.json",result);return result
