"""vLLM offline batch evaluation for the standard decoder SFT artifacts.

Generation rows and model text stay in RAM.  The only persistent output is an
aggregate metric/provenance JSON, intentionally unsuitable for reconstructing
student writing or generated feedback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .standard_decoder_data import (
    DEFAULT_MANIFEST, ROOT, SCORE_FIELDS, StandardDecoderContractError,
    load_frozen_validation, load_prepared_split, messages_for_generation,
    parse_decoder_scores, score_mean,
)


def _pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 2:
        return None
    am, bm = sum(a) / len(a), sum(b) / len(b)
    denominator = math.sqrt(sum((x - am) ** 2 for x in a) * sum((y - bm) ** 2 for y in b))
    return None if denominator == 0 else sum((x - am) * (y - bm) for x, y in zip(a, b, strict=True)) / denominator


def _qwk(a: Sequence[float], b: Sequence[float]) -> float | None:
    if not a or len(a) != len(b):
        return None
    observed = [[0.0] * 5 for _ in range(5)]
    for truth, pred in zip(a, b, strict=True):
        i, j = min(5, max(1, int(math.floor(truth + 0.5)))) - 1, min(5, max(1, int(math.floor(pred + 0.5)))) - 1
        observed[i][j] += 1.0
    row, col, n = [sum(line) for line in observed], [sum(observed[i][j] for i in range(5)) for j in range(5)], float(len(a))
    actual = sum(((i - j) ** 2 / 16.0) * observed[i][j] / n for i in range(5) for j in range(5))
    expected = sum(((i - j) ** 2 / 16.0) * row[i] * col[j] / (n * n) for i in range(5) for j in range(5))
    return 1.0 if expected == 0 and actual == 0 else (None if expected == 0 else 1.0 - actual / expected)


def aggregate_metrics(targets: Sequence[dict[str, float]], predictions: Sequence[dict[str, float]], valid: Sequence[bool]) -> dict[str, Any]:
    if not targets or len(targets) != len(predictions) or len(targets) != len(valid):
        raise StandardDecoderContractError("aligned nonempty score vectors required")
    result: dict[str, Any] = {"record_count": len(targets), "decoder_parse_failure_rate": 1.0 - sum(valid) / len(valid)}
    maes = []
    for field in SCORE_FIELDS:
        truth, pred = [row[field] for row in targets], [row[field] for row in predictions]
        mae = sum(abs(x - y) for x, y in zip(truth, pred, strict=True)) / len(truth)
        maes.append(mae)
        result[field] = {
            "mae": mae, "rmse": math.sqrt(sum((x-y)**2 for x, y in zip(truth, pred, strict=True)) / len(truth)),
            "pearson_r": _pearson(truth, pred), "quadratic_weighted_kappa": _qwk(truth, pred),
        }
    result["primary_macro_mae"] = sum(maes) / len(maes)
    return result


@dataclass(frozen=True)
class VLLMEvalConfig:
    run_id: str
    mode: str
    model_path: str
    model_revision: str
    adapter_path: str
    source: str  # selection_dev | frozen_validation
    prepared_manifest: str
    validation_sha256: str
    output_dir: str
    tensor_parallel_size: int = 8
    max_model_len: int = 4096
    max_new_tokens: int = 1536
    gpu_memory_utilization: float = 0.90
    wandb_project: str = "mal2026-korean-writing-scoring"
    wandb_entity: str | None = None

    @classmethod
    def from_json(cls, path: Path) -> "VLLMEvalConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise StandardDecoderContractError("vLLM evaluator config has missing or unknown fields")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in {"direct", "human_feedback"} or self.source not in {"selection_dev", "frozen_validation"}:
            raise StandardDecoderContractError("invalid vLLM mode or source")
        if Path(self.prepared_manifest).resolve() != DEFAULT_MANIFEST.resolve():
            raise StandardDecoderContractError("vLLM must use canonical aggregate prepared manifest")
        if Path(self.output_dir).resolve().parent != (ROOT / "outputs" / "standard-evals").resolve() or Path(self.output_dir).exists():
            raise StandardDecoderContractError("evaluator output must be a new direct child of ignored outputs/standard-evals")
        adapter = Path(self.adapter_path)
        standard_runs = (ROOT / "outputs" / "standard-runs").resolve()
        if not adapter.is_dir() or not adapter.resolve().is_relative_to(standard_runs):
            raise StandardDecoderContractError("adapter must be inside a standard training output")
        # Bind the adapter to a completed standard-Trainer run. Selection-dev
        # evaluation may use its selection adapter; frozen validation accepts
        # only a refit adapter, preventing selection/final lineage swaps.
        completion_path = adapter.parent / "standard_training_complete.json"
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StandardDecoderContractError("adapter is missing standard training completion provenance") from exc
        if not isinstance(completion, dict) or completion.get("status") != "completed":
            raise StandardDecoderContractError("adapter training completion status is invalid")
        if completion.get("mode") != self.mode or completion.get("model_revision") not in {None, self.model_revision}:
            raise StandardDecoderContractError("adapter completion does not match evaluation mode/model revision")
        expected_phase = "selection" if self.source == "selection_dev" else "refit"
        if completion.get("phase") != expected_phase:
            raise StandardDecoderContractError("evaluation source requires matching selection/refit adapter provenance")
        if not (adapter / "adapter_config.json").is_file():
            raise StandardDecoderContractError("adapter directory lacks adapter_config.json")
        expected_len, expected_new = (2048, 256) if self.mode == "direct" else (4096, 1536)
        if (self.max_model_len, self.max_new_tokens) != (expected_len, expected_new):
            raise StandardDecoderContractError("vLLM mode-specific token budget is frozen")
        if self.tensor_parallel_size <= 0 or not (0.5 <= self.gpu_memory_utilization <= 0.95):
            raise StandardDecoderContractError("invalid vLLM parallelism/memory utilization")
        if self.source == "frozen_validation" and not self.validation_sha256:
            raise StandardDecoderContractError("frozen final evaluation requires a pinned validation SHA-256")


def run_vllm_evaluation(config: VLLMEvalConfig) -> dict[str, Any]:
    config.validate()
    rows = load_prepared_split("selection_dev", Path(config.prepared_manifest)) if config.source == "selection_dev" else load_frozen_validation(config.validation_sha256)
    fallback_rows = load_prepared_split("selection_train", Path(config.prepared_manifest))
    fallback = score_mean(fallback_rows)
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise RuntimeError("standard evaluator requires vLLM in the project .venv-standard") from exc
    llm = LLM(
        model=config.model_path, revision=config.model_revision, dtype="bfloat16", trust_remote_code=False,
        enable_lora=True, tensor_parallel_size=config.tensor_parallel_size, max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=config.max_new_tokens, skip_special_tokens=True)
    outputs = llm.chat(
        [messages_for_generation(row, config.mode) for row in rows], sampling,
        lora_request=LoRARequest("standard_adapter", 1, config.adapter_path), use_tqdm=True,
    )
    predictions, valid = [], []
    for output in outputs:
        # Never persist output text. Invalid text is scored through the predeclared train-mean fallback.
        text = output.outputs[0].text if output.outputs else ""
        parsed = parse_decoder_scores(text, config.mode)
        valid.append(parsed is not None)
        predictions.append(parsed if parsed is not None else fallback)
    metrics = aggregate_metrics([row.score for row in rows], predictions, valid)
    result = {
        "status": "completed", "run_id": config.run_id, "source": config.source, "mode": config.mode,
        "model_revision": config.model_revision, "adapter_path": str(Path(config.adapter_path).resolve()),
        "metrics": metrics, "config": asdict(config),
        "privacy": "aggregate_only_no_rows_prompts_essays_feedback_ids_or_model_outputs_persisted",
    }
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "aggregate_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Explicit aggregate-only W&B event. No tables, samples, artifacts, or free text.
    try:
        import wandb
        run = wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=config.run_id, config={"mode": config.mode, "source": config.source, "model_revision": config.model_revision})
        run.log({"eval/primary_macro_mae": metrics["primary_macro_mae"], "eval/parse_failure_rate": metrics["decoder_parse_failure_rate"], "eval/record_count": metrics["record_count"]})
        run.finish()
    except ImportError:
        pass
    return result
