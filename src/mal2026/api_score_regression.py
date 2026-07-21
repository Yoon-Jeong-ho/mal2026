"""Transformers Trainer score regression for direct/API/decoder rationale inputs."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .api_rationale_data import (
    AXES, RESTRICTED_ROOT, ROOT, APIRationaleContractError, aggregate_input_provenance,
    joined_candidates, load_generated_rationales, load_writing_rows, train_writings, validation_writings,
)


RUN_ROOT = ROOT / "outputs" / "api-score-regression-v1"
EVAL_ROOT = ROOT / "outputs" / "api-score-regression-evals-v1"
BACKBONES = {
    "qwen25_7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", "last_nonpad"),
    "kure_v1": ("nlpai-lab/KURE-v1", "d14c8a9423946e268a0c9952fecf3a7aabd73bd9", "cls"),
}


class APIScoreRegressionError(APIRationaleContractError):
    """Raised for invalid score-regression data, model, or evaluation lineage."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise APIScoreRegressionError(message)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, name: str) -> float:
    _need(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be numeric")
    parsed = float(value); _need(math.isfinite(parsed), f"{name} must be finite"); return parsed


@dataclass(frozen=True)
class APIScoreRegressionConfig:
    schema_version: str
    run_id: str
    backbone_key: str
    model_id: str
    model_revision: str
    model_path: str
    input_condition: str
    decoder_generation_dir: str | None
    output_dir: str
    seed: int
    max_length: int
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    logging_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    training_dtype: str

    @classmethod
    def from_json(cls, path: Path) -> "APIScoreRegressionConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        _need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "regression config has unknown or missing fields")
        value = cls(**raw); value.validate(); return value

    def validate(self, *, require_fresh_output: bool = True) -> None:
        _need(self.schema_version == "mal2026-api-score-regression-v1", "regression config schema differs")
        _need(self.backbone_key in BACKBONES and BACKBONES[self.backbone_key][:2] == (self.model_id, self.model_revision), "regression backbone identity differs")
        _need(self.input_condition in {"direct", "api_rationale", "decoder_rationale"}, "regression input condition differs")
        model, output = Path(self.model_path), Path(self.output_dir)
        _need(model.is_absolute() and model.is_dir() and not model.is_symlink() and model.name.endswith(self.model_revision), "regression model snapshot differs")
        _need(output.is_absolute() and output.parent == RUN_ROOT.resolve() and (not output.exists() if require_fresh_output else output.is_dir()), "regression output root differs")
        suffix = "003" if self.backbone_key == "qwen25_7b" else "004"
        _need(self.run_id == f"api-score-regression-v1-{self.backbone_key}-{self.input_condition}-{suffix}", "regression run lineage differs")
        _need(self.seed == 2026072108 and self.max_length == 3072 and self.logging_steps > 0, "regression seed/sequence/logging contract differs")
        expected_optimizer = (2e-5, 2, 8) if self.backbone_key == "qwen25_7b" else (1e-4, 8, 2)
        _need((self.learning_rate, self.per_device_train_batch_size, self.gradient_accumulation_steps) == expected_optimizer, "regression optimizer/global batch differs")
        _need(self.num_train_epochs == (6.0 if self.input_condition == "api_rationale" else 12.0), "regression epoch schedule differs")
        _need((self.lora_r, self.lora_alpha, self.lora_dropout) == (16, 32, 0.05), "regression LoRA contract differs")
        _need(self.training_dtype == "float32", "regression numerical-recovery dtype differs")
        if self.input_condition == "decoder_rationale":
            _need(isinstance(self.decoder_generation_dir, str) and Path(self.decoder_generation_dir).is_absolute(), "decoder-rationale condition requires its immutable generation artifact")
        else:
            _need(self.decoder_generation_dir is None, "non-decoder condition must not receive a decoder artifact")


@dataclass(frozen=True)
class APIScoreRegressionEvalConfig:
    schema_version: str
    run_id: str
    training_metadata_path: str
    output_dir: str
    per_device_eval_batch_size: int

    @classmethod
    def from_json(cls, path: Path) -> "APIScoreRegressionEvalConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        _need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "regression eval config has unknown or missing fields")
        value = cls(**raw); value.validate(); return value

    def validate(self) -> None:
        _need(self.schema_version == "mal2026-api-score-regression-eval-v1", "regression eval config schema differs")
        metadata, output = Path(self.training_metadata_path), Path(self.output_dir)
        _need(metadata.is_absolute() and metadata.name == "training_complete.json" and metadata.parent.parent == RUN_ROOT.resolve(), "regression training metadata differs")
        _need(output.is_absolute() and output.parent == EVAL_ROOT.resolve() and not output.exists(), "regression evaluation output must be fresh ignored direct child")
        _need(self.per_device_eval_batch_size > 0 and self.run_id.startswith("api-score-regression-eval-v1-"), "regression eval identity differs")


def _rationale_text(diagnoses: Mapping[str, str]) -> str:
    _need(set(diagnoses) == set(AXES), "score-regression rationale must cover all three axes")
    # The labels/score fields are intentionally absent from this serialized
    # input; it is only a private model feature in RAM.
    return json.dumps({axis: {"rationale": diagnoses[axis]} for axis in AXES}, ensure_ascii=False, separators=(",", ":"))


def _input_text(prompt: str, essay: str, diagnoses: Mapping[str, str] | None) -> str:
    base = f"<writing_prompt>\n{prompt}\n</writing_prompt>\n<student_essay>\n{essay}\n</student_essay>"
    return base if diagnoses is None else base + f"\n<evaluation_rationales>\n{_rationale_text(diagnoses)}\n</evaluation_rationales>"


def _labels(row: Any) -> list[float]:
    _need(row.scores is not None and set(row.scores) == set(AXES), "score-regression labels are unavailable")
    return [_finite(row.scores[axis], f"label.{axis}") for axis in AXES]


def _training_examples(config: APIScoreRegressionConfig) -> list[dict[str, Any]]:
    if config.input_condition == "direct":
        rows = train_writings()
        return [{"text": _input_text(row.prompt, row.essay, None), "labels": _labels(row)} for row in rows]
    if config.input_condition == "api_rationale":
        labels = {row.identifier: _labels(row) for row in train_writings()}
        values = joined_candidates("train")
        examples = [{"text": _input_text(joined.writing.prompt, joined.writing.essay, joined.candidate.diagnoses), "labels": labels[joined.writing.identifier]} for joined in values]
        _need(len(examples) == 6000, "API-rationale training population differs")
        return examples
    assert config.decoder_generation_dir is not None
    generated = load_generated_rationales(Path(config.decoder_generation_dir), source="train", task="bundle")
    rows = train_writings()
    examples = [{"text": _input_text(row.prompt, row.essay, generated[row.identifier]), "labels": _labels(row)} for row in rows]
    _need(len(examples) == 2000, "decoder-rationale training population differs")
    return examples


def _validation_examples(config: APIScoreRegressionConfig) -> list[dict[str, Any]]:
    rows = validation_writings()
    if config.input_condition == "direct":
        return [{"source_id": row.identifier, "text": _input_text(row.prompt, row.essay, None), "labels": _labels(row)} for row in rows]
    if config.input_condition == "api_rationale":
        labels = {row.identifier: _labels(row) for row in rows}
        values = joined_candidates("validation")
        examples = [{"source_id": joined.writing.identifier, "text": _input_text(joined.writing.prompt, joined.writing.essay, joined.candidate.diagnoses), "labels": labels[joined.writing.identifier]} for joined in values]
        _need(len(examples) == 1200, "API-rationale validation population differs")
        return examples
    assert config.decoder_generation_dir is not None
    train_generation = Path(config.decoder_generation_dir)
    validation_name = train_generation.name.replace("-bundle-train-003", "-bundle-validation-003")
    _need(validation_name != train_generation.name, "decoder-rationale train generation lineage differs")
    validation_generation = train_generation.with_name(validation_name)
    generated = load_generated_rationales(validation_generation, source="validation", task="bundle")
    return [{"source_id": row.identifier, "text": _input_text(row.prompt, row.essay, generated[row.identifier]), "labels": _labels(row)} for row in rows]


def _tokenize_examples(examples: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int, *, include_source: bool) -> Any:
    from datasets import Dataset
    payload = {
        "text": [item["text"] for item in examples], "labels": [item["labels"] for item in examples],
    }
    if include_source: payload["source_id"] = [item["source_id"] for item in examples]
    dataset = Dataset.from_dict(payload)
    encoded = dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])
    return encoded


def _lora_targets(model: Any, backbone_key: str) -> list[str]:
    leaves = {name.rsplit(".", maxsplit=1)[-1] for name, _ in model.named_modules()}
    wanted = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if backbone_key == "qwen25_7b" else ["query", "key", "value", "dense"]
    _need(set(wanted) <= leaves, "regression model lacks reviewed LoRA target modules")
    return wanted


def _build_model(config: APIScoreRegressionConfig) -> Any:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError("score regression requires the project .venv-standard") from exc
    base = AutoModel.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    targets = _lora_targets(base, config.backbone_key)
    peft = get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=targets, bias="none"))
    hidden = getattr(peft.config, "hidden_size", None); _need(type(hidden) is int and hidden > 0, "regression backbone lacks hidden size")
    pooling = BACKBONES[config.backbone_key][2]

    class Regressor(nn.Module):
        def __init__(self) -> None:
            super().__init__(); self.backbone = peft; self.regression_head = nn.Linear(hidden, len(AXES)); self.pooling = pooling

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None, **_: Any) -> Mapping[str, Any]:
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
            if self.pooling == "cls":
                embedding = output[:, 0]
            else:
                positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
                index = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
                _need(bool((index >= 0).all().item()), "all regression sequences require a nonpad token")
                embedding = output[torch.arange(output.shape[0], device=output.device), index]
            logits = 1.0 + 4.0 * torch.sigmoid(self.regression_head(F.normalize(embedding.float(), p=2, dim=-1)))
            result: dict[str, Any] = {"logits": logits}
            if labels is not None:
                _need(tuple(labels.shape[-1:]) == (len(AXES),), "regression label dimension differs")
                result["loss"] = F.mse_loss(logits, labels.float())
            return result
    return Regressor(), targets


def _collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        labels = [feature.pop("labels") for feature in features]
        for feature in features: feature.pop("source_id", None)
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch
    return collate


def _finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {str(key): _finite(value, f"Trainer metric {key}") for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
    _need("train_loss" in result, "Trainer did not emit train_loss")
    return result


def run_api_score_regression(config: APIScoreRegressionConfig) -> dict[str, Any]:
    config.validate()
    try:
        import torch
        from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("score regression requires the project .venv-standard") from exc
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        _need(tokenizer.eos_token is not None, "regression tokenizer lacks pad/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    examples = _training_examples(config); dataset = _tokenize_examples(examples, tokenizer, config.max_length, include_source=False)
    model, targets = _build_model(config)
    args = TrainingArguments(
        output_dir=config.output_dir, run_name=config.run_id, do_train=True, do_eval=False, eval_strategy="no", save_strategy="epoch", save_total_limit=1,
        logging_steps=config.logging_steps, logging_strategy="steps", learning_rate=config.learning_rate, num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size, gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=False, tf32=True, report_to=[], remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=True,
        ddp_find_unused_parameters=config.backbone_key == "kure_v1", max_grad_norm=0.1, optim="adamw_torch", logging_nan_inf_filter=False, seed=config.seed, data_seed=config.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=_collator(tokenizer))
    trained = trainer.train(); trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None; failed = False
    if trainer.is_world_process_zero():
        try:
            metrics = _finite_metrics(trained.metrics)
            _need(all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters() if parameter.requires_grad), "one or more trainable regression parameters are non-finite")
            final = Path(config.output_dir) / "final_model"; trainer.save_model(str(final)); tokenizer.save_pretrained(str(final))
            state = final / "model.safetensors"; _need(state.is_file(), "Trainer did not save a safetensors model state")
            payload = {"status": "completed", "run_id": config.run_id, "backbone_key": config.backbone_key, "model_id": config.model_id, "model_revision": config.model_revision,
                       "input_condition": config.input_condition, "train_records": len(examples), "global_step": int(trainer.state.global_step), "train_metrics": metrics,
                       "lora_targets": targets, "model_state_sha256": _sha(state), "input_provenance": aggregate_input_provenance(), "config": asdict(config),
                       "candidate_scores_read_or_prompted": False, "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_candidate_scores_or_predictions_persisted"}
            path = Path(config.output_dir) / "training_complete.json"; _need(not path.exists(), "regression completion already exists"); path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception: failed = True
    state: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized(): torch.distributed.broadcast_object_list(state, src=0)
    if state[0]: raise APIScoreRegressionError("rank-zero regression persistence/health gate failed")
    _need(isinstance(state[1], dict) and state[1].get("status") == "completed", "regression completion was not published")
    trainer.accelerator.wait_for_everyone(); return state[1]


def _average_rank(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1]); ranks = [0.0] * len(values); start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]: end += 1
        rank = (start + 1 + end) / 2.0
        for index in range(start, end): ranks[indexed[index][0]] = rank
        start = end
    return ranks


def _spearman(truth: Sequence[float], pred: Sequence[float]) -> float:
    _need(len(truth) == len(pred) and len(truth) >= 2, "Spearman needs aligned nontrivial vectors")
    a, b = _average_rank(truth), _average_rank(pred); am, bm = sum(a) / len(a), sum(b) / len(b)
    denom = math.sqrt(sum((value - am) ** 2 for value in a) * sum((value - bm) ** 2 for value in b))
    _need(denom > 0, "Spearman is undefined for constant ranks")
    return sum((x - am) * (y - bm) for x, y in zip(a, b, strict=True)) / denom


def regression_metrics(truth: Sequence[Sequence[float]], predictions: Sequence[Sequence[float]]) -> dict[str, Any]:
    _need(len(truth) == len(predictions) and len(truth) > 0, "metric vectors must align")
    result: dict[str, Any] = {}; rmses=[]; correlations=[]
    for index, axis in enumerate(AXES):
        t, p = [float(row[index]) for row in truth], [float(row[index]) for row in predictions]
        _need(all(math.isfinite(value) for value in t + p), "non-finite score prediction")
        rmse = math.sqrt(sum((a-b) ** 2 for a, b in zip(t, p, strict=True)) / len(t)); correlation = _spearman(t, p)
        result[axis] = {"rmse": rmse, "spearman": correlation}; rmses.append(rmse); correlations.append(correlation)
    result["macro_rmse"] = sum(rmses) / len(rmses); result["macro_spearman"] = sum(correlations) / len(correlations); return result


def _load_training(config: APIScoreRegressionEvalConfig) -> tuple[Mapping[str, Any], APIScoreRegressionConfig, Path]:
    try: metadata = json.loads(Path(config.training_metadata_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise APIScoreRegressionError("training metadata is unreadable") from exc
    _need(isinstance(metadata, dict) and metadata.get("status") == "completed", "training metadata is incomplete")
    raw = metadata.get("config"); _need(isinstance(raw, dict), "training metadata lacks config")
    saved = APIScoreRegressionConfig(**raw); saved.validate(require_fresh_output=False)
    state = Path(saved.output_dir) / "final_model" / "model.safetensors"
    _need(state.is_file() and _sha(state) == metadata.get("model_state_sha256"), "training state checksum differs")
    return metadata, saved, state


def run_api_score_regression_evaluation(config: APIScoreRegressionEvalConfig) -> dict[str, Any]:
    config.validate(); metadata, train, state = _load_training(config)
    try:
        import numpy as np
        import torch
        from safetensors.torch import load_model
        from transformers import AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc: raise RuntimeError("score regression evaluation requires the project .venv-standard") from exc
    tokenizer = AutoTokenizer.from_pretrained(train.model_path, revision=train.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    examples = _validation_examples(train); dataset = _tokenize_examples(examples, tokenizer, train.max_length, include_source=True)
    model, _ = _build_model(train); missing, unexpected = load_model(model, str(state), strict=False); _need(not missing and not unexpected, "saved regression model state differs from rebuilt model")
    output = Path(config.output_dir)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(output), do_train=False, do_eval=False, per_device_eval_batch_size=config.per_device_eval_batch_size, bf16=False, tf32=True, report_to=[], remove_unused_columns=False), data_collator=_collator(tokenizer))
    predicted = trainer.predict(dataset).predictions
    values = predicted.tolist() if isinstance(predicted, np.ndarray) else predicted
    _need(len(values) == len(examples), "prediction count differs")
    grouped: dict[str, list[list[float]]] = defaultdict(list); labels: dict[str, list[float]] = {}
    for item, vector in zip(examples, values, strict=True):
        grouped[item["source_id"]].append([_finite(value, "prediction") for value in vector]); labels[item["source_id"]] = [_finite(value, "label") for value in item["labels"]]
    _need(len(grouped) == 400 and all(identifier in labels for identifier in grouped), "validation source grouping differs")
    averaged = [[sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(AXES))] for _, vectors in sorted(grouped.items())]
    truths = [labels[identifier] for identifier in sorted(grouped)]
    metrics = regression_metrics(truths, averaged)
    payload = {"status": "completed", "run_id": config.run_id, "training_run_id": metadata.get("run_id"), "backbone_key": train.backbone_key, "input_condition": train.input_condition,
               "metrics": metrics, "validation": {"unique_essays": len(grouped), "input_records": len(examples), "api_candidates_averaged_per_essay": train.input_condition == "api_rationale"},
               "model_state_sha256": _sha(state), "config": asdict(config), "candidate_scores_read_or_prompted": False,
               "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_candidate_scores_or_predictions_persisted"}
    trainer.accelerator.wait_for_everyone()
    failed = False
    if trainer.is_world_process_zero():
        try:
            # Current Transformers creates output_dir while constructing
            # TrainingArguments. The no-overwrite guard ran before that
            # construction; only the aggregate completion filename remains
            # unavailable at this point.
            _need(output.is_dir() and not (output / "aggregate_metrics.json").exists(), "regression evaluation output was reused")
            (output / "aggregate_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            failed = True
    state_payload: list[Any] = [failed, payload if trainer.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state_payload, src=0)
    if state_payload[0]:
        raise APIScoreRegressionError("rank-zero regression evaluation persistence failed")
    _need(isinstance(state_payload[1], dict) and state_payload[1].get("status") == "completed", "regression evaluation completion was not published")
    trainer.accelerator.wait_for_everyone()
    return state_payload[1]
