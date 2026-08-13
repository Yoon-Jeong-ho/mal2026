"""Locked zero-shot ablation of the user prompt and a train-derived band rubric.

Individual prompts, writings, identifiers, responses, and predictions remain
under the restricted run root. Public artifacts contain only aggregate metrics
and cryptographic provenance. The revised prompt is derived from the public
three-axis contract and aggregate train-only analysis; it is not represented as
an organizer-authored prompt.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import gc
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
from typing import Any, Mapping, Sequence

from .api_rationale_data import AXES, SOURCE_SHA256, WritingRow, load_writing_rows, sha256_file
from .decoder_fewshot_external import (
    ExternalConfig,
    _assert_gpus_idle,
    _solar_preflight,
    _solar_request,
    _solar_server_command,
    _wait_solar,
)
from .decoder_fewshot_validation import (
    condition_metrics,
    file_sha256,
    parse_response,
    response_schema,
    round_half_up,
    spearman,
)
from .official_score_prompt import (
    EVALUATION_PROMPT_SHA256,
    USER_SUPPLIED_EVALUATION,
    query_text,
    system_prompt,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "mal2026-decoder-prompt-band-ablation-config-v1"
EXPECTED_RUN_ID = "decoder-prompt-band-ablation-v1-20260801-001"
EXPECTED_CONFIG_SHA256 = "f6d9efa80df2e58d81d9de919dce085c49257fec5fd234ab8066b562cbaa799e"
PROMPT_ARMS = ("official_p0", "axis_band_p1")
SPLITS = ("train-probe", "validation")
RESTRICTED_ROOT = ROOT / "data/processed/restricted/decoder_prompt_band_ablation_v1"
PUBLIC_ROOT = ROOT / "outputs/analysis"
RUNTIME_ROOT = ROOT / "outputs/decoder-prompt-band-ablation-v1"


class PromptBandAblationError(RuntimeError):
    """Fail-closed prompt ablation contract error."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise PromptBandAblationError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_json_fresh(path: Path, value: Mapping[str, Any]) -> str:
    need(not path.exists(), f"fresh output required: {path}")
    _atomic_json(path, value)
    return file_sha256(path)


def _write_jsonl_fresh(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), f"fresh output required: {path}")
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return file_sha256(path)


@dataclass(frozen=True)
class QwenSpec:
    model_key: str
    model_id: str
    revision: str
    model_path: str
    tensor_parallel_size: int
    disable_thinking: bool


@dataclass(frozen=True)
class AblationConfig:
    schema_version: str
    run_id: str
    seed: int
    train_probe_seed: int
    train_probe_rows: int
    official_prompt_sha256: str
    revised_prompt_path: str
    revised_prompt_sha256: str
    prompt_arms: tuple[str, ...]
    splits: tuple[str, ...]
    max_model_len: int
    max_tokens: int
    retry_max_tokens: int
    temperature: float
    gpu_memory_utilization: float
    max_num_seqs: int
    max_num_batched_tokens: int
    bootstrap_replicates: int
    bootstrap_seed: int
    qwen: QwenSpec
    solar_base_config_path: str
    solar_base_config_sha256: str

    @classmethod
    def from_json(cls, path: Path) -> "AblationConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version", "run_id", "seed", "train_probe_seed", "train_probe_rows",
            "official_prompt_sha256", "revised_prompt_path", "revised_prompt_sha256",
            "prompt_arms", "splits", "max_model_len", "max_tokens", "retry_max_tokens",
            "temperature", "gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens",
            "bootstrap_replicates", "bootstrap_seed", "qwen", "solar_base_config_path",
            "solar_base_config_sha256",
        }
        need(isinstance(raw, dict) and set(raw) == expected, "ablation config schema differs")
        qwen_raw = raw.pop("qwen")
        arms = tuple(raw.pop("prompt_arms"))
        splits = tuple(raw.pop("splits"))
        need(isinstance(qwen_raw, dict) and set(qwen_raw) == {
            "model_key", "model_id", "revision", "model_path", "tensor_parallel_size", "disable_thinking"
        }, "Qwen spec differs")
        config = cls(qwen=QwenSpec(**qwen_raw), prompt_arms=arms, splits=splits, **raw)
        config.validate(path)
        return config

    def validate(self, path: Path) -> None:
        need(self.schema_version == SCHEMA_VERSION and self.run_id == EXPECTED_RUN_ID, "ablation identity differs")
        need((self.seed, self.train_probe_seed, self.bootstrap_seed) == (2026080101, 2026080102, 2026080103), "ablation seeds differ")
        need(self.train_probe_rows == 400 and self.prompt_arms == PROMPT_ARMS and self.splits == SPLITS, "ablation population differs")
        need(self.official_prompt_sha256 == EVALUATION_PROMPT_SHA256, "official prompt binding differs")
        revised = Path(self.revised_prompt_path)
        need(revised.is_absolute() and revised.is_file() and not revised.is_symlink(), "revised prompt unavailable")
        need(file_sha256(revised) == self.revised_prompt_sha256, "revised prompt binding differs")
        need((self.max_model_len, self.max_tokens, self.retry_max_tokens) == (12288, 512, 2048), "generation budget differs")
        need(self.temperature == 0.0 and (self.max_num_seqs, self.max_num_batched_tokens) == (64, 32768), "decoding capacity differs")
        need(self.bootstrap_replicates == 10000, "bootstrap protocol differs")
        need(self.qwen.tensor_parallel_size == 4 and self.qwen.disable_thinking is True, "Qwen execution contract differs")
        solar_config = Path(self.solar_base_config_path)
        need(solar_config.is_absolute() and file_sha256(solar_config) == self.solar_base_config_sha256, "Solar base binding differs")
        need(EXPECTED_CONFIG_SHA256 != "PENDING" and file_sha256(path) == EXPECTED_CONFIG_SHA256, "ablation config checksum differs")


def restricted_dir(config: AblationConfig) -> Path:
    path = RESTRICTED_ROOT / config.run_id
    need(path.resolve().is_relative_to(RESTRICTED_ROOT.resolve()), "restricted run escaped root")
    return path


def public_dir(config: AblationConfig) -> Path:
    path = PUBLIC_ROOT / config.run_id
    need(path.resolve().is_relative_to(PUBLIC_ROOT.resolve()), "public run escaped root")
    return path


def runtime_dir(config: AblationConfig) -> Path:
    path = RUNTIME_ROOT / config.run_id
    need(path.resolve().is_relative_to(RUNTIME_ROOT.resolve()), "runtime run escaped root")
    return path


def _sections(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    markers = {line.strip(): index for index, line in enumerate(lines) if line.strip() in {"[시스템 프롬프트]", "[유저 프롬프트]"}}
    need(set(markers) == {"[시스템 프롬프트]", "[유저 프롬프트]"}, "revised prompt markers differ")
    need(markers["[시스템 프롬프트]"] == 0 and markers["[유저 프롬프트]"] > 1, "revised prompt routing differs")
    split = markers["[유저 프롬프트]"]
    system = "".join(lines[1:split])
    user = "".join(lines[split + 1:])
    need(user.count("{주제 지문}") == user.count("{논증적 글 본문}") == 1, "revised prompt placeholders differ")
    need("평균" in system and "다른 키" in system, "revised output guard differs")
    return system, user


def messages_for(config: AblationConfig, arm: str, prompt: str, essay: str) -> list[dict[str, str]]:
    need(arm in PROMPT_ARMS, "unknown prompt arm")
    if arm == "official_p0":
        return [
            {"role": "system", "content": system_prompt(USER_SUPPLIED_EVALUATION)},
            {"role": "user", "content": query_text(prompt, essay, kind=USER_SUPPLIED_EVALUATION)},
        ]
    system, template = _sections(Path(config.revised_prompt_path))
    user = template.replace("{주제 지문}", prompt).replace("{논증적 글 본문}", essay)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def train_probe(config: AblationConfig) -> list[WritingRow]:
    rows = load_writing_rows("train", include_scores=True)
    ordered = sorted(rows, key=lambda row: sha256(f"{config.train_probe_seed}:{row.identifier}".encode()).hexdigest())
    selected = ordered[:config.train_probe_rows]
    need(len(selected) == config.train_probe_rows and len({row.identifier for row in selected}) == len(selected), "train probe differs")
    return selected


def split_rows(config: AblationConfig, split: str) -> list[WritingRow]:
    need(split in SPLITS, "unknown split")
    rows = train_probe(config) if split == "train-probe" else load_writing_rows("validation", include_scores=True)
    need(len(rows) == 400, "ablation split population differs")
    return rows


def request_records(config: AblationConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        for row in split_rows(config, split):
            need(row.scores is not None, "gold scores unavailable")
            for arm in PROMPT_ARMS:
                records.append({
                    "source_id": row.identifier,
                    "split": split,
                    "arm": arm,
                    "messages": messages_for(config, arm, row.prompt, row.essay),
                    "gold_raw": {axis: float(row.scores[axis]) for axis in AXES},
                    "gold_integer": {axis: round_half_up(row.scores[axis]) for axis in AXES},
                })
    records.sort(key=lambda row: (row["split"], row["arm"], sha256(row["source_id"].encode()).hexdigest()))
    need(len(records) == 1600 and len({(row["source_id"], row["split"], row["arm"]) for row in records}) == 1600, "request population differs")
    return records


def prepare(config: AblationConfig, config_path: Path) -> dict[str, Any]:
    need(sha256_file(ROOT / "eval/train.jsonl") == SOURCE_SHA256["train"], "canonical train checksum differs")
    need(sha256_file(ROOT / "eval/validation.jsonl") == SOURCE_SHA256["validation"], "canonical validation checksum differs")
    probe = train_probe(config)
    manifest = {
        "schema_version": "mal2026-decoder-prompt-band-train-probe-v1",
        "run_id": config.run_id,
        "selection_seed": config.train_probe_seed,
        "source_ids": [row.identifier for row in probe],
        "source_id_set_sha256": sha256("\n".join(sorted(row.identifier for row in probe)).encode()).hexdigest(),
        "prompt_or_model_selection_independent": False,
        "note": "prompt was derived from aggregate canonical train analysis; this is a descriptive train probe",
    }
    manifest_sha = _write_json_fresh(restricted_dir(config) / "train_probe_manifest.json", manifest)
    protocol = {
        "schema_version": "mal2026-decoder-prompt-band-protocol-v1",
        "status": "prepared",
        "run_id": config.run_id,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": file_sha256(config_path),
        "official_prompt_sha256": config.official_prompt_sha256,
        "revised_prompt_sha256": config.revised_prompt_sha256,
        "revised_prompt_provenance": "public_spec_and_aggregate_train_derived_not_organizer_authored",
        "prompt_arms": list(PROMPT_ARMS),
        "splits": {"train-probe": 400, "validation": 400},
        "requests_per_model": 1600,
        "zero_shot": True,
        "labeled_demonstrations": 0,
        "validation_labels_in_prompt_or_prompt_selection": False,
        "average_target_used": False,
        "train_probe_manifest_sha256": manifest_sha,
        "canonical_source_sha256": dict(SOURCE_SHA256),
        "validation_interpretation": "locked descriptive; canonical validation was observed by prior experiments",
        "models": [config.qwen.model_id, "nota-ai/Solar-Open2-250B-Nota-INT4"],
    }
    _write_json_fresh(public_dir(config) / "protocol.json", protocol)
    ledger = runtime_dir(config) / "ledger.jsonl"
    _write_jsonl_fresh(ledger, [{"event": "prepared", "at": now(), "gpu_scope": [0, 1, 2, 3], "config_sha256": file_sha256(config_path)}])
    return protocol


def _verify_protocol(config: AblationConfig) -> None:
    public = public_dir(config) / "protocol.json"
    restricted = restricted_dir(config) / "train_probe_manifest.json"
    need(public.is_file() and restricted.is_file(), "prepared protocol unavailable")
    protocol = json.loads(public.read_text(encoding="utf-8"))
    need(protocol.get("train_probe_manifest_sha256") == file_sha256(restricted), "train probe binding differs")
    manifest = json.loads(restricted.read_text(encoding="utf-8"))
    need(manifest.get("source_ids") == [row.identifier for row in train_probe(config)], "train probe replay differs")


def _extra_metrics(rows: Sequence[Mapping[str, Any]], total_count: int = 400) -> dict[str, Any]:
    valid = [row for row in rows if row.get("parse_valid") is True]
    need(valid, "no valid rows for extra metrics")
    by_axis: dict[str, Any] = {}
    for axis in AXES:
        raw = [float(row["gold_raw"][axis]) for row in valid]
        integer = [int(row["gold_integer"][axis]) for row in valid]
        pred = [int(row["prediction"][axis]) for row in valid]
        pred_hist = Counter(pred); gold_hist = Counter(integer)
        calibration: dict[str, Any] = {}
        weighted_abs = 0.0; weighted_sq = 0.0
        for score in range(1, 6):
            gold_at_prediction = [g for g, p in zip(raw, pred) if p == score]
            if gold_at_prediction:
                mean_gold = statistics.mean(gold_at_prediction)
                gap = score - mean_gold
                calibration[str(score)] = {"n": len(gold_at_prediction), "mean_raw_gold": mean_gold, "prediction_minus_mean_gold": gap}
                weight = len(gold_at_prediction) / len(valid)
                weighted_abs += weight * abs(gap); weighted_sq += weight * gap * gap
            else:
                calibration[str(score)] = {"n": 0, "mean_raw_gold": None, "prediction_minus_mean_gold": None}
        tails = {}
        for score in (1, 2, 5):
            selected = [p for g, p in zip(integer, pred) if g == score]
            tails[str(score)] = {"n": len(selected), "exact_recall": (sum(p == score for p in selected) / len(selected)) if selected else None}
        low = [(g, p) for g, p in zip(integer, pred) if g in {1, 2}]
        by_axis[axis] = {
            "mean_bias": statistics.mean(p - g for p, g in zip(pred, raw)),
            "histogram_tv": 0.5 * sum(abs(pred_hist[s] / len(valid) - gold_hist[s] / len(valid)) for s in range(1, 6)),
            "ordinal_point_calibration": {"weighted_absolute_error": weighted_abs, "root_mean_square_error": math.sqrt(weighted_sq), "by_predicted_score": calibration},
            "tail_recall": tails,
            "low_tail_1_2_exact_recall": sum(p == g for g, p in low) / len(low) if low else None,
        }
    return {"count": len(valid), "parse_success_rate": len(valid) / total_count, "by_axis": by_axis}


def _arm_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("parse_valid") is True]
    base = condition_metrics(valid, expected_count=len(valid), total_count=400)
    return {**base, "diagnostics": _extra_metrics(rows)}


def _paired_metrics(config: AblationConfig, rows: Sequence[Mapping[str, Any]], split: str, model_key: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split and row.get("parse_valid") is True]
    by_key = {(row["source_id"], row["arm"]): row for row in selected}
    ids = sorted({row["source_id"] for row in selected if (row["source_id"], PROMPT_ARMS[0]) in by_key and (row["source_id"], PROMPT_ARMS[1]) in by_key})
    need(ids, "paired prompt intersection is empty")
    sensitivity = {}
    for axis in AXES:
        left = [int(by_key[(source_id, PROMPT_ARMS[0])]["prediction"][axis]) for source_id in ids]
        right = [int(by_key[(source_id, PROMPT_ARMS[1])]["prediction"][axis]) for source_id in ids]
        sensitivity[axis] = {
            "mean_absolute_delta": statistics.mean(abs(b - a) for a, b in zip(left, right)),
            "mean_signed_delta_revised_minus_official": statistics.mean(b - a for a, b in zip(left, right)),
            "flip_rate": statistics.mean(a != b for a, b in zip(left, right)),
        }

    def sampled(indexes: Sequence[int], arm: str) -> tuple[float, float]:
        rmses=[]; rhos=[]
        for axis in AXES:
            pred=[int(by_key[(ids[i],arm)]["prediction"][axis]) for i in indexes]
            gold=[float(by_key[(ids[i],arm)]["gold_raw"][axis]) for i in indexes]
            rmses.append(math.sqrt(statistics.mean((p-g)**2 for p,g in zip(pred,gold))))
            rhos.append(spearman(pred,gold))
        return statistics.mean(rmses),statistics.mean(rhos)

    derived_seed = config.bootstrap_seed + int(sha256(f"{model_key}:{split}".encode()).hexdigest()[:8],16)
    rng=random.Random(derived_seed)
    rmse_deltas=[];rho_deltas=[]
    for _ in range(config.bootstrap_replicates):
        indexes=[rng.randrange(len(ids)) for _ in ids]
        base_rmse,base_rho=sampled(indexes,PROMPT_ARMS[0]);rev_rmse,rev_rho=sampled(indexes,PROMPT_ARMS[1])
        rmse_deltas.append(rev_rmse-base_rmse);rho_deltas.append(rev_rho-base_rho)
    def interval(values: list[float]) -> dict[str,float]:
        ordered=sorted(values); n=len(ordered)
        return {"mean":statistics.mean(values),"lower_95":ordered[math.floor(0.025*(n-1))],"upper_95":ordered[math.ceil(0.975*(n-1))]}
    return {
        "paired_rows":len(ids),"paired_coverage":len(ids)/400,"by_axis":sensitivity,
        "paired_bootstrap":{"replicates":config.bootstrap_replicates,"seed":derived_seed,"raw_rmse_delta_revised_minus_official":interval(rmse_deltas),"raw_spearman_delta_revised_minus_official":interval(rho_deltas)},
    }


def _finalize_rows(config: AblationConfig, model_key: str, model_id: str, rows: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]) -> dict[str, Any]:
    need(len(rows) == 1600, "model result population differs")
    for split in SPLITS:
        for arm in PROMPT_ARMS:
            need(sum(row["split"] == split and row["arm"] == arm for row in rows) == 400, "model arm population differs")
    prediction_path = restricted_dir(config) / "models" / model_key / "predictions.jsonl"
    prediction_sha = _write_jsonl_fresh(prediction_path, rows)
    metrics = {split:{arm:_arm_metrics([row for row in rows if row["split"]==split and row["arm"]==arm]) for arm in PROMPT_ARMS} for split in SPLITS}
    paired = {split:_paired_metrics(config,rows,split,model_key) for split in SPLITS}
    aggregate = {
        "schema_version":"mal2026-decoder-prompt-band-model-result-v1","status":"completed","run_id":config.run_id,
        "model_key":model_key,"model_id":model_id,"requests":len(rows),"parse_failures":sum(row.get("parse_valid") is not True for row in rows),
        "prediction_sha256":prediction_sha,"metrics":metrics,"paired_prompt_analysis":paired,**dict(provenance),
    }
    _write_json_fresh(public_dir(config)/"models"/model_key/"aggregate.json",aggregate)
    return aggregate


def run_qwen(config: AblationConfig) -> dict[str, Any]:
    _verify_protocol(config)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3", "Qwen physical GPU scope differs")
    _assert_gpus_idle((0, 1, 2, 3))
    model_path=Path(config.qwen.model_path)
    need(model_path.is_dir() and not model_path.is_symlink(),"Qwen model unavailable")
    try:
        from vllm import LLM,SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as exc:
        raise PromptBandAblationError("vLLM unavailable") from exc
    llm=LLM(model=str(model_path),tokenizer=str(model_path),tensor_parallel_size=4,dtype="auto",trust_remote_code=False,max_model_len=config.max_model_len,gpu_memory_utilization=config.gpu_memory_utilization,max_num_seqs=config.max_num_seqs,max_num_batched_tokens=config.max_num_batched_tokens,enable_prefix_caching=True,enforce_eager=False,seed=config.seed,limit_mm_per_prompt={"image":0,"video":0})
    tokenizer=llm.get_tokenizer(); records=request_records(config); requests=[]
    for record in records:
        rendered=tokenizer.apply_chat_template(record["messages"],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prompt_tokens=len(tokenizer.encode(rendered,add_special_tokens=False))
        need(prompt_tokens+config.retry_max_tokens<=config.max_model_len,"Qwen prompt exceeds context")
        requests.append({**record,"prompt":rendered,"prompt_tokens":prompt_tokens})
    sampling=SamplingParams(temperature=0.0,max_tokens=config.max_tokens,seed=config.seed,structured_outputs=StructuredOutputsParams(json=response_schema()))
    smoke=[next(row for row in requests if row["split"]=="train-probe" and row["arm"]==arm) for arm in PROMPT_ARMS]
    smoke_outputs=llm.generate([row["prompt"] for row in smoke],sampling)
    need(len(smoke_outputs)==2 and all(parse_response(output.outputs[0].text) for output in smoke_outputs),"Qwen smoke failed")
    _write_json_fresh(public_dir(config)/"models"/config.qwen.model_key/"smoke.json",{"schema_version":"mal2026-decoder-prompt-band-smoke-v1","status":"passed","requests":2,"prompt_arms":list(PROMPT_ARMS),"maximum_prompt_tokens":max(row["prompt_tokens"] for row in requests),"gpu_scope":[0,1,2,3]})
    outputs=llm.generate([row["prompt"] for row in requests],sampling)
    need(len(outputs)==len(requests),"Qwen output population differs")
    rows=[]; retry_indexes=[]
    for index,(request,output) in enumerate(zip(requests,outputs)):
        choice=output.outputs[0]
        try: parsed=parse_response(choice.text); valid=True; error=None
        except Exception as exc: parsed=None;valid=False;error=type(exc).__name__+":"+str(exc)
        row={k:v for k,v in request.items() if k not in {"messages","prompt"}}
        row.update({"response":choice.text,"parse_valid":valid,"parse_error":error,"prediction":None if parsed is None else {axis:parsed[axis]["score"] for axis in AXES},"completion_tokens":len(choice.token_ids),"finish_reason":choice.finish_reason,"retry":None})
        rows.append(row)
        if not valid and choice.finish_reason=="length" and len(choice.token_ids)==config.max_tokens: retry_indexes.append(index)
    if retry_indexes:
        retry_sampling=SamplingParams(temperature=0.0,max_tokens=config.retry_max_tokens,seed=config.seed,structured_outputs=StructuredOutputsParams(json=response_schema()))
        retried=llm.generate([requests[i]["prompt"] for i in retry_indexes],retry_sampling)
        for index,output in zip(retry_indexes,retried):
            choice=output.outputs[0]
            try: parsed=parse_response(choice.text);valid=True;error=None
            except Exception as exc: parsed=None;valid=False;error=type(exc).__name__+":"+str(exc)
            rows[index].update({"response":choice.text,"parse_valid":valid,"parse_error":error,"prediction":None if parsed is None else {axis:parsed[axis]["score"] for axis in AXES},"completion_tokens":len(choice.token_ids),"finish_reason":choice.finish_reason,"retry":{"reason":"initial_length_truncation","max_tokens":config.retry_max_tokens}})
    aggregate=_finalize_rows(config,config.qwen.model_key,config.qwen.model_id,rows,{"provider":"local vLLM","model_revision":config.qwen.revision,"gpu_scope":[0,1,2,3],"tensor_parallel_size":4,"temperature":0.0,"seed":config.seed,"prompt_tokens":{"minimum":min(row["prompt_tokens"] for row in rows),"maximum":max(row["prompt_tokens"] for row in rows),"mean":statistics.mean(row["prompt_tokens"] for row in rows)},"length_retries":len(retry_indexes)})
    del llm; gc.collect()
    return aggregate


def run_solar(config: AblationConfig) -> dict[str, Any]:
    _verify_protocol(config)
    solar=replace(ExternalConfig.from_json(Path(config.solar_base_config_path)),seed=config.seed); records=request_records(config)
    _assert_gpus_idle((0,1,2,3)); preflight=_solar_preflight(solar,records)
    container=f"mal2026-prompt-band-solar-{solar.solar.port}"
    endpoint=f"http://127.0.0.1:{solar.solar.port}"
    log_path=runtime_dir(config)/"solar-server.log";log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open("x",encoding="utf-8") as log:
        process=subprocess.Popen(_solar_server_command(solar,container),stdout=log,stderr=subprocess.STDOUT,text=True)
        try:
            _wait_solar(process,endpoint)
            smoke_records=[next(row for row in records if row["split"]=="train-probe" and row["arm"]==arm) for arm in PROMPT_ARMS]
            smoke=[_solar_request(solar,endpoint,row) for row in smoke_records]
            need(all(row["parse_valid"] for row in smoke),"Solar smoke failed")
            _write_json_fresh(public_dir(config)/"models"/"solar-open2-int4"/"smoke.json",{"schema_version":"mal2026-decoder-prompt-band-smoke-v1","status":"passed","requests":2,"prompt_arms":list(PROMPT_ARMS),"preflight":preflight,"gpu_scope":[0,1,2,3]})
            with ThreadPoolExecutor(max_workers=solar.solar.max_inflight) as pool:
                futures={pool.submit(_solar_request,solar,endpoint,row):index for index,row in enumerate(records)};resolved={}
                for future in as_completed(futures): resolved[futures[future]]=future.result()
                raw_rows=[resolved[index] for index in range(len(records))]
        finally:
            subprocess.run(["docker","stop","--time","30",container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
            try: process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.terminate();process.wait(timeout=30)
    _assert_gpus_idle((0,1,2,3))
    rows=[{k:v for k,v in row.items() if k!="messages"} for row in raw_rows]
    usage=Counter()
    for row in rows: usage.update({str(k):int(v) for k,v in row.get("usage",{}).items() if type(v) is int})
    return _finalize_rows(config,"solar-open2-int4",solar.solar.model_id,rows,{"provider":"local official Solar vLLM Docker","docker_image":solar.solar.docker_image,"docker_image_id":solar.solar.docker_image_id,"gpu_scope":[0,1,2,3],"tensor_parallel_size":4,"expert_parallel":True,"temperature":0.0,"seed":config.seed,"usage":dict(usage),"preflight":preflight})


def aggregate(config: AblationConfig) -> dict[str, Any]:
    models={}
    for key in (config.qwen.model_key,"solar-open2-int4"):
        path=public_dir(config)/"models"/key/"aggregate.json";need(path.is_file(),f"model aggregate unavailable: {key}")
        row=json.loads(path.read_text(encoding="utf-8"));models[key]={"model_id":row["model_id"],"parse_failures":row["parse_failures"],"metrics":row["metrics"],"paired_prompt_analysis":row["paired_prompt_analysis"]}
    result={"schema_version":"mal2026-decoder-prompt-band-aggregate-v1","status":"completed","run_id":config.run_id,"prompt_arms":list(PROMPT_ARMS),"splits":list(SPLITS),"zero_shot":True,"average_target_used":False,"models":models}
    _write_json_fresh(public_dir(config)/"aggregate.json",result)
    return result
