"""Frozen Qwen3-Embedding input and rationale-view improvement experiments.

The module deliberately reuses the already verified AI-Hub LoRA warm start and
changes only the predeclared input/view contract.  Restricted text stays in
memory; checkpoints contain only trainable tensors and public reports contain
aggregate metrics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .api_rationale_data import EXPECTED_ESSAYS, SOURCE_SHA256, load_writing_rows
from .rlaif_qwen3_embedding import (
    AXES,
    MODEL_ID,
    MODEL_PATH,
    MODEL_REVISION,
    WARMSTART_METADATA,
    _atomic_json,
    _sha,
    build_model,
    warmstart_provenance,
)
from .rlaif_top3_encoder import (
    SELECTIONS,
    _input_text,
    _labels,
    _load_generated_rationales,
    generation_dir,
    three_axis_metrics,
)


ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "outputs" / "rlaif-qwen3-embedding-improvement-v1"
EVAL_ROOT = ROOT / "outputs" / "rlaif-qwen3-embedding-improvement-evals-v1"
PROGRAM_ID = "20260726-006"
ARMS = ("essay_only", "essay_instruction", "rationale_instruction", "trait_specific", "multi_rationale")
PHASES = ("gpu0_preflight", "full")
PRIMARY_SOURCE = "rank2_ax4_random1"
MULTI_SOURCES = tuple(SELECTIONS)
FULL_EPOCHS = 4
ESSAY_INSTRUCTION = (
    "Instruct: Predict the content, organization, and expression scores of a Korean student essay "
    "from 1 to 5 using the writing prompt and essay.\nQuery:\n"
)
RATIONALE_INSTRUCTION = (
    "Instruct: Predict the content, organization, and expression scores of a Korean student essay "
    "from 1 to 5 using the writing prompt, essay, and qualitative rationales.\nQuery:\n"
)
TRAIT_RUBRICS = {
    "content": "과제 적합성, 중심 주장과 아이디어의 명료성, 근거의 구체성과 충실성을 판단한다.",
    "organization": "도입·전개·마무리의 구조, 문단과 논리의 흐름, 연결과 응집성을 판단한다.",
    "expression": "문장의 명료성과 자연스러움, 어휘와 문법, 맞춤법과 문체의 적절성을 판단한다.",
}


class ImprovementError(ValueError):
    """Raised when the frozen improvement contract differs."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ImprovementError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImprovementError(f"{label} is unreadable") from exc
    _need(isinstance(value, dict), f"{label} must be an object")
    return value


def training_dir(arm: str, phase: str) -> Path:
    return TRAIN_ROOT / f"rlaif-qwen3-improvement-v1-{arm}-{phase}-006"


def evaluation_dir(arm: str, phase: str) -> Path:
    return EVAL_ROOT / f"rlaif-qwen3-improvement-eval-v1-{arm}-{phase}-006"


def checkpoint_dir(output: Path, epoch: int) -> Path:
    return output / "epoch_checkpoints" / f"epoch-{epoch:02d}"


def _views_per_essay(arm: str) -> int:
    return 3 if arm in {"trait_specific", "multi_rationale"} else 1


def expected_steps(arm: str, phase: str) -> dict[int, int]:
    _need(arm in ARMS and phase in PHASES, "unknown arm or phase")
    if phase == "gpu0_preflight":
        return {1: 1}
    per_epoch = 94 if _views_per_essay(arm) == 3 else 32
    return {epoch: epoch * per_epoch for epoch in range(1, FULL_EPOCHS + 1)}


@dataclass(frozen=True)
class ImprovementTrainConfig:
    schema_version: str
    run_id: str
    arm: str
    phase: str
    output_dir: str
    model_id: str
    model_revision: str
    model_path: str
    warmstart_metadata_path: str
    score_fields: tuple[str, str, str]
    seed: int
    max_length: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    num_train_epochs: float
    max_steps: int
    essay_limit: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    logging_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    training_dtype: str

    @classmethod
    def from_json(cls, path: Path) -> "ImprovementTrainConfig":
        raw = _read_json(path, "improvement training config")
        _need(isinstance(raw.get("score_fields"), list), "score_fields must be a list")
        raw["score_fields"] = tuple(raw["score_fields"])
        _need(set(raw) == set(cls.__dataclass_fields__), "training config fields differ")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self, *, require_fresh_output: bool = True) -> None:
        _need(self.schema_version == "mal2026-rlaif-qwen3-improvement-train-v1", "training schema differs")
        _need(self.arm in ARMS and self.phase in PHASES, "training arm/phase differs")
        _need(self.score_fields == AXES, "only three canonical score fields are allowed")
        _need((self.model_id, self.model_revision, Path(self.model_path).resolve()) == (MODEL_ID, MODEL_REVISION, MODEL_PATH.resolve()), "model snapshot differs")
        _need(Path(self.warmstart_metadata_path).resolve() == WARMSTART_METADATA.resolve(), "warm-start path differs")
        warmstart_provenance()
        output = Path(self.output_dir)
        expected = f"rlaif-qwen3-improvement-v1-{self.arm}-{self.phase}-006"
        _need(output.is_absolute() and output.parent == TRAIN_ROOT.resolve(), "training output root differs")
        _need(output.name == self.run_id == expected, "training identity differs")
        _need((not output.exists()) if require_fresh_output else output.is_dir(), "training output freshness differs")
        _need((self.seed, self.max_length, self.learning_rate, self.weight_decay, self.warmup_ratio) == (2026072601, 2048, 1e-4, 0.01, 0.05), "optimization contract differs")
        _need((self.lora_r, self.lora_alpha, self.lora_dropout, self.training_dtype) == (16, 32, 0.05, "bfloat16"), "LoRA/numeric contract differs")
        if self.phase == "full":
            _need((self.num_train_epochs, self.max_steps, self.essay_limit, self.per_device_train_batch_size, self.gradient_accumulation_steps) == (4.0, -1, 2000, 4, 4), "full schedule differs")
        else:
            _need((self.num_train_epochs, self.max_steps, self.essay_limit, self.per_device_train_batch_size, self.gradient_accumulation_steps) == (1.0, 1, 4, 4, 1), "preflight schedule differs")


def training_config(arm: str, phase: str) -> dict[str, Any]:
    _need(arm in ARMS and phase in PHASES, "unknown training arm/phase")
    full = phase == "full"
    run_id = f"rlaif-qwen3-improvement-v1-{arm}-{phase}-006"
    return {
        "schema_version": "mal2026-rlaif-qwen3-improvement-train-v1",
        "run_id": run_id,
        "arm": arm,
        "phase": phase,
        "output_dir": str(training_dir(arm, phase).resolve()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(MODEL_PATH.resolve()),
        "warmstart_metadata_path": str(WARMSTART_METADATA.resolve()),
        "score_fields": list(AXES),
        "seed": 2026072601,
        "max_length": 2048,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "num_train_epochs": 4.0 if full else 1.0,
        "max_steps": -1 if full else 1,
        "essay_limit": 2000 if full else 4,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4 if full else 1,
        "logging_steps": 5 if full else 1,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "training_dtype": "bfloat16",
    }


def _essay_text(prompt: str, essay: str) -> str:
    return f"<writing_prompt>\n{prompt}\n</writing_prompt>\n<student_essay>\n{essay}\n</student_essay>"


def _trait_text(prompt: str, essay: str, axis: str, rationale: str) -> str:
    return (
        "Instruct: Predict the requested writing-trait score of a Korean student essay from 1 to 5 "
        "using the writing prompt, essay, trait rubric, and qualitative rationale.\nQuery:\n"
        f"<target_trait>{axis}</target_trait>\n<trait_rubric>{TRAIT_RUBRICS[axis]}</trait_rubric>\n"
        f"{_essay_text(prompt, essay)}\n<evaluation_rationale>{rationale}</evaluation_rationale>"
    )


def examples(arm: str, split: str, essay_limit: int) -> list[dict[str, Any]]:
    _need(arm in ARMS and split in EXPECTED_ESSAYS and 0 < essay_limit <= EXPECTED_ESSAYS[split], "example request differs")
    rows = load_writing_rows(split, include_scores=True)[:essay_limit]
    source_keys: Sequence[str] = MULTI_SOURCES if arm == "multi_rationale" else (PRIMARY_SOURCE,)
    generated = {
        source: _load_generated_rationales(generation_dir(source, split, "full"), source, split, "full", EXPECTED_ESSAYS[split])
        for source in source_keys
    }
    result: list[dict[str, Any]] = []
    for row in rows:
        labels = _labels(row)
        if arm == "essay_only":
            result.append({"source_id": row.identifier, "view": "essay", "text": _essay_text(row.prompt, row.essay), "labels": labels})
        elif arm == "essay_instruction":
            result.append({"source_id": row.identifier, "view": "essay_instruction", "text": ESSAY_INSTRUCTION + _essay_text(row.prompt, row.essay), "labels": labels})
        elif arm == "rationale_instruction":
            result.append({"source_id": row.identifier, "view": PRIMARY_SOURCE, "text": RATIONALE_INSTRUCTION + _input_text(row.prompt, row.essay, generated[PRIMARY_SOURCE][row.identifier]), "labels": labels})
        elif arm == "trait_specific":
            rationales = generated[PRIMARY_SOURCE][row.identifier]
            for axis_index, axis in enumerate(AXES):
                result.append({"source_id": row.identifier, "view": axis, "axis_index": axis_index, "text": _trait_text(row.prompt, row.essay, axis, rationales[axis]), "labels": labels})
        else:
            for source in MULTI_SOURCES:
                result.append({"source_id": row.identifier, "view": source, "text": RATIONALE_INSTRUCTION + _input_text(row.prompt, row.essay, generated[source][row.identifier]), "labels": labels})
    _need(len(result) == essay_limit * _views_per_essay(arm), "example count differs")
    return result


def tokenized(items: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int) -> Any:
    from datasets import Dataset

    payload: dict[str, Any] = {
        "text": [item["text"] for item in items],
        "labels": [item["labels"] for item in items],
    }
    if "axis_index" in items[0]:
        payload["axis_index"] = [item["axis_index"] for item in items]
    dataset = Dataset.from_dict(payload)
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])


def collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        labels = [feature.pop("labels") for feature in features]
        axis = [feature.pop("axis_index") for feature in features] if "axis_index" in features[0] else None
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        if axis is not None:
            batch["axis_index"] = torch.tensor(axis, dtype=torch.long)
        return batch

    return collate


def build_warm_model(config: ImprovementTrainConfig) -> tuple[Any, dict[str, Any]]:
    # ``build_model`` needs only this frozen attribute subset.  Keeping the
    # loader centralized preserves the already-audited warm-start slicing of
    # the obsolete average head.
    class Compatible:
        arm = "qwen3_aihub_warmstart"
        model_path = config.model_path
        model_revision = config.model_revision
        lora_r = config.lora_r
        lora_alpha = config.lora_alpha
        lora_dropout = config.lora_dropout

    return build_model(Compatible())


def _trait_wrapper(regressor: Any) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class TraitRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.regressor = regressor

        def forward(self, input_ids: Any, attention_mask: Any, axis_index: Any, labels: Any | None = None, **_: Any) -> Mapping[str, Any]:
            logits = self.regressor(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            _need(axis_index.ndim == 1 and axis_index.shape[0] == logits.shape[0], "trait axis shape differs")
            chosen = logits.gather(1, axis_index[:, None]).squeeze(1)
            result: dict[str, Any] = {"logits": logits}
            if labels is not None:
                target = labels.float().gather(1, axis_index[:, None]).squeeze(1)
                result["loss"] = functional.mse_loss(chosen, target, reduction="mean")
            return result

    return TraitRegressor()


def model_for_arm(config: ImprovementTrainConfig) -> tuple[Any, dict[str, Any]]:
    model, initialization = build_warm_model(config)
    return (_trait_wrapper(model) if config.arm == "trait_specific" else model), initialization


def trainable_state(model: Any) -> dict[str, Any]:
    state = {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters() if parameter.requires_grad}
    _need(bool(state) and any(name.endswith("regression_head.weight") for name in state), "trainable state lacks regression head")
    _need(all("average" not in name for name in state), "trainable state contains average target")
    return state


def _rationale_hashes(arm: str, split: str) -> dict[str, str]:
    if arm in {"essay_only", "essay_instruction"}:
        return {}
    sources = MULTI_SOURCES if arm == "multi_rationale" else (PRIMARY_SOURCE,)
    return {source: _sha(generation_dir(source, split, "full") / "generated_rationales.jsonl") for source in sources}


def run_training(config: ImprovementTrainConfig) -> dict[str, Any]:
    config.validate()
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("improvement training requires .venv-standard") from exc
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    items = examples(config.arm, "train", config.essay_limit)
    dataset = tokenized(items, tokenizer, config.max_length)
    model, initialization = model_for_arm(config)
    expected = expected_steps(config.arm, config.phase)
    output = Path(config.output_dir)

    class EpochCheckpoint(TrainerCallback):
        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            # A structural preflight may deliberately stop after one update
            # before traversing all expanded views; it is checkpoint 1 even
            # though Trainer reports a fractional epoch.
            epoch = 1 if config.phase == "gpu0_preflight" else int(round(float(state.epoch or 0.0)))
            _need(epoch in expected and int(state.global_step) == expected[epoch], "checkpoint boundary differs")
            failed = False
            message = None
            if state.is_world_process_zero:
                try:
                    root = checkpoint_dir(output, epoch)
                    _need(not root.exists(), "checkpoint already exists")
                    root.mkdir(parents=True)
                    state_path = root / "trainable_model.safetensors"
                    tensors = trainable_state(kwargs["model"])
                    save_file(tensors, str(state_path))
                    _atomic_json(root / "checkpoint_metadata.json", {
                        "status": "completed", "run_id": config.run_id, "arm": config.arm,
                        "epoch": epoch, "global_step": int(state.global_step), "score_fields": list(AXES),
                        "average_target_used": False, "trainable_tensor_count": len(tensors),
                        "trainable_state_sha256": _sha(state_path), "initialization": initialization,
                        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
                    })
                except Exception as exc:
                    failed, message = True, f"{type(exc).__name__}: {exc}"
            status: list[Any] = [failed, message]
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(status, src=0)
            if status[0]:
                raise ImprovementError(f"checkpoint persistence failed: {status[1]}")
            return control

    args = TrainingArguments(
        output_dir=config.output_dir, run_name=config.run_id, do_train=True, do_eval=False,
        eval_strategy="no", save_strategy="no", logging_strategy="steps", logging_steps=config.logging_steps,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs, max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps, bf16=True, tf32=True, report_to=[],
        remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=True,
        ddp_find_unused_parameters=False, max_grad_norm=1.0, optim="adamw_torch",
        logging_nan_inf_filter=False, seed=config.seed, data_seed=config.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator(tokenizer), callbacks=[EpochCheckpoint()])
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None
    failed = False
    if trainer.is_world_process_zero():
        try:
            metrics = {str(key): float(value) for key, value in trained.metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
            _need("train_loss" in metrics and all(math.isfinite(value) for value in metrics.values()), "training metrics are non-finite")
            checkpoints = []
            for epoch, step in expected.items():
                root = checkpoint_dir(output, epoch)
                meta = _read_json(root / "checkpoint_metadata.json", "checkpoint metadata")
                state_path = root / "trainable_model.safetensors"
                _need(meta.get("global_step") == step and state_path.is_file() and meta.get("trainable_state_sha256") == _sha(state_path), "checkpoint evidence differs")
                checkpoints.append({"epoch": epoch, "global_step": step, "trainable_state_path": str(state_path.resolve()), "trainable_state_sha256": meta["trainable_state_sha256"]})
            payload = {
                "status": "completed", "run_id": config.run_id, "arm": config.arm, "phase": config.phase,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "score_fields": list(AXES),
                "average_target_used": False, "unique_train_essays": config.essay_limit,
                "train_input_records": len(items), "views_per_essay": _views_per_essay(config.arm),
                "global_step": int(trainer.state.global_step), "train_metrics": metrics,
                "checkpoints": checkpoints, "initialization": initialization,
                "input_provenance": {"canonical_source_sha256": dict(SOURCE_SHA256), "rationale_generation_sha256": _rationale_hashes(config.arm, "train")},
                "config": asdict(config),
                "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
            }
            _atomic_json(output / "training_complete.json", payload)
        except Exception:
            failed = True
    status_payload: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(status_payload, src=0)
    if status_payload[0]:
        raise ImprovementError("rank-zero training finalization failed")
    _need(isinstance(status_payload[1], dict), "training completion was not published")
    trainer.accelerator.wait_for_everyone()
    return status_payload[1]


@dataclass(frozen=True)
class ImprovementEvalConfig:
    schema_version: str
    run_id: str
    arm: str
    phase: str
    training_metadata_path: str
    output_dir: str
    essay_limit: int
    per_device_eval_batch_size: int

    @classmethod
    def from_json(cls, path: Path) -> "ImprovementEvalConfig":
        raw = _read_json(path, "improvement evaluation config")
        _need(set(raw) == set(cls.__dataclass_fields__), "evaluation config fields differ")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self) -> None:
        _need(self.schema_version == "mal2026-rlaif-qwen3-improvement-eval-v1", "evaluation schema differs")
        _need(self.arm in ARMS and self.phase in PHASES, "evaluation arm/phase differs")
        metadata = training_dir(self.arm, self.phase) / "training_complete.json"
        _need(Path(self.training_metadata_path).resolve() == metadata.resolve(), "evaluation lineage differs")
        output = Path(self.output_dir)
        expected = f"rlaif-qwen3-improvement-eval-v1-{self.arm}-{self.phase}-006"
        _need(output.is_absolute() and output.parent == EVAL_ROOT.resolve() and not output.exists(), "evaluation output freshness differs")
        _need(output.name == self.run_id == expected, "evaluation identity differs")
        _need((self.essay_limit, self.per_device_eval_batch_size) == ((400, 8) if self.phase == "full" else (4, 4)), "evaluation schedule differs")


def evaluation_config(arm: str, phase: str) -> dict[str, Any]:
    _need(arm in ARMS and phase in PHASES, "unknown evaluation arm/phase")
    full = phase == "full"
    run_id = f"rlaif-qwen3-improvement-eval-v1-{arm}-{phase}-006"
    return {
        "schema_version": "mal2026-rlaif-qwen3-improvement-eval-v1", "run_id": run_id,
        "arm": arm, "phase": phase,
        "training_metadata_path": str((training_dir(arm, phase) / "training_complete.json").resolve()),
        "output_dir": str(evaluation_dir(arm, phase).resolve()),
        "essay_limit": 400 if full else 4, "per_device_eval_batch_size": 8 if full else 4,
    }


def _aggregate_predictions(arm: str, items: Sequence[Mapping[str, Any]], values: Sequence[Sequence[float]]) -> tuple[list[list[float]], list[list[float]]]:
    essays = len(items) // _views_per_essay(arm)
    _need(len(values) == len(items) and essays > 0, "prediction population differs")
    truth: list[list[float]] = []
    predicted: list[list[float]] = []
    for essay_index in range(essays):
        start = essay_index * _views_per_essay(arm)
        group_items = items[start:start + _views_per_essay(arm)]
        group_values = values[start:start + _views_per_essay(arm)]
        truth.append([float(value) for value in group_items[0]["labels"]])
        _need(all(item["source_id"] == group_items[0]["source_id"] and item["labels"] == group_items[0]["labels"] for item in group_items), "view grouping differs")
        if arm == "trait_specific":
            predicted.append([float(group_values[index][index]) for index in range(len(AXES))])
        elif arm == "multi_rationale":
            predicted.append([sum(float(vector[index]) for vector in group_values) / len(group_values) for index in range(len(AXES))])
        else:
            predicted.append([float(value) for value in group_values[0]])
    return truth, predicted


def run_evaluation(config: ImprovementEvalConfig) -> dict[str, Any]:
    config.validate()
    training = _read_json(Path(config.training_metadata_path), "training completion")
    _need(training.get("status") == "completed" and training.get("arm") == config.arm and training.get("score_fields") == list(AXES) and training.get("average_target_used") is False, "training provenance differs")
    raw = dict(training.get("config", {}))
    _need(isinstance(raw.get("score_fields"), list), "saved training config differs")
    raw["score_fields"] = tuple(raw["score_fields"])
    train = ImprovementTrainConfig(**raw)
    train.validate(require_fresh_output=False)
    try:
        import numpy as np
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("improvement evaluation requires .venv-standard") from exc
    tokenizer = AutoTokenizer.from_pretrained(train.model_path, revision=train.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    items = examples(config.arm, "validation", config.essay_limit)
    dataset = tokenized(items, tokenizer, train.max_length)
    model, _ = model_for_arm(train)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    output = Path(config.output_dir)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(output), do_train=False, do_eval=False, per_device_eval_batch_size=config.per_device_eval_batch_size, bf16=True, tf32=True, report_to=[], remove_unused_columns=False), data_collator=collator(tokenizer))
    rows = []
    for checkpoint in training["checkpoints"]:
        epoch = int(checkpoint["epoch"])
        state_path = Path(checkpoint["trainable_state_path"])
        _need(state_path.is_file() and _sha(state_path) == checkpoint["trainable_state_sha256"], "checkpoint checksum differs")
        state = load_file(str(state_path), device="cpu")
        _need(set(state) == trainable_names, "checkpoint tensor names differ")
        incompatible = model.load_state_dict(state, strict=False)
        _need(not incompatible.unexpected_keys and not (trainable_names & set(incompatible.missing_keys)), "checkpoint load differs")
        raw_prediction = trainer.predict(dataset).predictions
        values = raw_prediction.tolist() if isinstance(raw_prediction, np.ndarray) else raw_prediction
        truth, predicted = _aggregate_predictions(config.arm, items, values)
        metrics = three_axis_metrics(truth, predicted)
        _need(all(math.isfinite(value) for axis in AXES for value in metrics[axis].values()), "evaluation metric is non-finite")
        rows.append({"epoch": epoch, "global_step": checkpoint["global_step"], "metrics": metrics, "trainable_state_sha256": checkpoint["trainable_state_sha256"]})
    best = min(rows, key=lambda row: (float(row["metrics"]["three_axis_macro_rmse"]), -float(row["metrics"]["three_axis_macro_spearman"]), int(row["epoch"])))
    payload = {
        "status": "completed", "run_id": config.run_id, "training_run_id": training["run_id"],
        "arm": config.arm, "phase": config.phase, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "score_fields": list(AXES), "average_target_used": False, "epoch_results": rows,
        "best_epoch_by_validation_macro_rmse_then_spearman": best,
        "validation": {"unique_essays": config.essay_limit, "input_records": len(items),
                       "predictions_per_essay_per_checkpoint": _views_per_essay(config.arm),
                       "view_aggregation": "none" if _views_per_essay(config.arm) == 1 else ("axis_select" if config.arm == "trait_specific" else "uniform_prediction_mean")},
        "selection_caveat": "validation was previously exposed; descriptive development evidence only",
        "input_provenance": {"canonical_source_sha256": dict(SOURCE_SHA256), "rationale_generation_sha256": _rationale_hashes(config.arm, "validation")},
        "config": asdict(config),
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_persisted",
    }
    trainer.accelerator.wait_for_everyone()
    failed = False
    if trainer.is_world_process_zero():
        try:
            _need(output.is_dir(), "evaluation output root was not created")
            _atomic_json(output / "epoch_metrics.json", payload)
        except Exception:
            failed = True
    status_payload: list[Any] = [failed, payload if trainer.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(status_payload, src=0)
    if status_payload[0]:
        raise ImprovementError("rank-zero evaluation persistence failed")
    _need(isinstance(status_payload[1], dict), "evaluation completion was not published")
    trainer.accelerator.wait_for_everyone()
    return status_payload[1]
