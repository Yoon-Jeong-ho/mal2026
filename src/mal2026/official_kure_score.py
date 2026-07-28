"""Leakage-resistant KURE-v1 score regression for exact evaluation prompts.

This is the KURE counterpart to the Qwen3 embedding matrix.  It keeps the
essay-only and essay-plus-rationale inputs separate, selects an epoch using an
internal 1,600/400 train split, reinitializes and refits on all 2,000 train
essays, and only then reads canonical validation for one descriptive report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .official_score_matrix import (
    AXES,
    ScoreRow,
    decode_logits,
    deterministic_internal_split,
    file_sha256,
    load_rationales,
    load_score_rows,
    score_metrics,
    select_epoch,
)
from .official_score_prompt import USER_SUPPLIED_EVALUATION, embedding_input, provenance as prompt_provenance


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "nlpai-lab/KURE-v1"
MODEL_REVISION = "d14c8a9423946e268a0c9952fecf3a7aabd73bd9"
MODEL_PATH = ROOT / "outputs/model-cache/nlpai-lab--KURE-v1-d14c8a9423946e268a0c9952fecf3a7aabd73bd9"
OUTPUT_ROOT = ROOT / "outputs/official-kure-score-evaluation-prompt-v1"
INPUT_VIEWS = ("essay", "rationale")
LORA_TARGETS = ("query", "key", "value", "dense")


class OfficialKUREScoreError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialKUREScoreError(message)


@dataclass(frozen=True)
class KUREScoreConfig:
    schema_version: str
    run_id: str
    model_id: str
    model_revision: str
    model_path: str
    train_path: str
    train_sha256: str
    validation_path: str
    validation_sha256: str
    rationale_key: str
    rationale_train_path: str
    rationale_train_sha256: str
    rationale_validation_path: str
    rationale_validation_sha256: str
    output_root: str
    seed: int
    max_length: int
    selection_epochs: tuple[int, ...]
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    training_dtype: str
    score_prompt_kind: str

    @classmethod
    def from_json(cls, path: Path, *, require_rationales: bool = True) -> "KUREScoreConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        need(isinstance(raw, dict), "KURE config must be an object")
        need(isinstance(raw.get("selection_epochs"), list), "KURE selection epochs differ")
        raw["selection_epochs"] = tuple(raw["selection_epochs"])
        need(set(raw) == set(cls.__dataclass_fields__), "KURE config fields differ")
        value = cls(**raw)
        value.validate(require_rationales=require_rationales)
        return value

    def validate(self, *, require_rationales: bool = True) -> None:
        need(self.schema_version == "mal2026-official-kure-score-v1", "KURE schema differs")
        need(self.run_id == "official-kure-score-evaluation-prompt-v1-20260729-001", "KURE run identity differs")
        need((self.model_id, self.model_revision) == (MODEL_ID, MODEL_REVISION), "KURE model pin differs")
        need(Path(self.model_path).resolve() == MODEL_PATH.resolve() and MODEL_PATH.is_dir() and not MODEL_PATH.is_symlink(), "KURE snapshot differs")
        need(Path(self.output_root).resolve() == OUTPUT_ROOT.resolve(), "KURE output root differs")
        need(self.score_prompt_kind == USER_SUPPLIED_EVALUATION, "KURE score prompt differs")
        need(self.selection_epochs == tuple(range(1, 13)), "KURE selection schedule differs")
        need((self.seed, self.max_length) == (2026072901, 3072), "KURE data/length contract differs")
        need((self.learning_rate, self.weight_decay, self.warmup_ratio) == (1e-4, 0.01, 0.05), "KURE optimizer differs")
        need((self.per_device_train_batch_size, self.per_device_eval_batch_size, self.gradient_accumulation_steps) == (8, 16, 2), "KURE batch contract differs")
        need((self.lora_r, self.lora_alpha, self.lora_dropout, self.training_dtype) == (16, 32, 0.05, "float32"), "KURE LoRA/numeric contract differs")
        train, validation = Path(self.train_path), Path(self.validation_path)
        need(train.resolve() == (ROOT / "eval/train.jsonl").resolve() and train.is_file() and file_sha256(train) == self.train_sha256, "KURE train source differs")
        need(validation.resolve() == (ROOT / "eval/validation.jsonl").resolve() and self.validation_sha256 == "0805445029328848164cf15f34b90b88fb5f7896d7b73f24f1717b733a9a00a4", "KURE validation pin differs")
        if require_rationales:
            need(self.rationale_key and not self.rationale_key.startswith("REQUIRED_"), "KURE rationale key is unresolved")
            restricted = (ROOT / "data/processed/restricted").resolve()
            for raw_path, digest in ((self.rationale_train_path, self.rationale_train_sha256), (self.rationale_validation_path, self.rationale_validation_sha256)):
                path = Path(raw_path)
                need(path.resolve().is_relative_to(restricted) and path.is_file() and file_sha256(path) == digest, "KURE rationale artifact differs")


def render_input(row: ScoreRow, view: str, rationales: Mapping[str, str] | None = None) -> str:
    need(view in INPUT_VIEWS and (view == "essay" or rationales is not None), "KURE input view differs")
    return embedding_input(row.prompt, row.essay, USER_SUPPLIED_EVALUATION, rationales if view == "rationale" else None)


def _dataset(rows: Sequence[ScoreRow], view: str, rationales: Mapping[str, Mapping[str, str]] | None, tokenizer: Any, max_length: int) -> Any:
    from datasets import Dataset
    texts = [render_input(row, view, None if rationales is None else rationales[row.identifier]) for row in rows]
    dataset = Dataset.from_dict({"text": texts, "labels": [list(row.labels) for row in rows]})
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])


def _collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        labels = [feature.pop("labels") for feature in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch
    return collate


def build_model(config: KUREScoreConfig) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel

    base = AutoModel.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    hidden = getattr(base.config, "hidden_size", None)
    need(type(hidden) is int and hidden > 0, "KURE hidden size differs")
    leaves = {name.rsplit(".", 1)[-1] for name, _ in base.named_modules()}
    need(set(LORA_TARGETS) <= leaves, "KURE LoRA target modules differ")
    backbone = get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=list(LORA_TARGETS), bias="none"))

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.score_head = nn.Linear(hidden, 3)

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None, **_: Any) -> Mapping[str, Any]:
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
            pooled = functional.normalize(output[:, 0].float(), p=2, dim=-1)
            logits = self.score_head(pooled)
            result: dict[str, Any] = {"logits": logits}
            if labels is not None:
                continuous, _, _ = decode_logits(logits, "bounded_regression")
                result["loss"] = functional.mse_loss(continuous, labels.float())
            return result

    model = Model()
    need(any("lora_" in name and parameter.requires_grad for name, parameter in model.named_parameters()), "KURE LoRA is absent")
    need(all(parameter.requires_grad for parameter in model.score_head.parameters()), "KURE score head is frozen")
    return model


def _trainable_state(model: Any) -> dict[str, Any]:
    return {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters() if parameter.requires_grad}


def _metrics(result: Any) -> dict[str, float]:
    import torch
    continuous, integers, violations = decode_logits(torch.as_tensor(result.predictions), "bounded_regression")
    value = score_metrics(result.label_ids.tolist(), continuous.tolist(), integers.tolist(), violations.tolist())
    return {key: float(value[key]) for key in ("macro_integer_rmse", "macro_integer_spearman", "macro_continuous_rmse", "macro_continuous_spearman")}


def _predict(trainer: Any, dataset: Any) -> dict[str, Any]:
    import torch
    result = trainer.predict(dataset)
    continuous, integers, violations = decode_logits(torch.as_tensor(result.predictions), "bounded_regression")
    return score_metrics(result.label_ids.tolist(), continuous.tolist(), integers.tolist(), violations.tolist())


def run_arm(config: KUREScoreConfig, view: str, *, smoke: bool = False) -> dict[str, Any]:
    config.validate(require_rationales=view == "rationale")
    need(view in INPUT_VIEWS, "KURE arm differs")
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError("KURE score training requires .venv-standard") from exc

    output = Path(config.output_root) / (f"smoke-{view}" if smoke else view)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        need(not output.exists(), "KURE arm output already exists")
        output.mkdir(parents=True)
    else:
        deadline = time.monotonic() + 60
        while not output.is_dir() and time.monotonic() < deadline:
            time.sleep(0.05)
        need(output.is_dir(), "KURE rank-zero output creation failed")

    all_train = load_score_rows(Path(config.train_path), config.train_sha256, 2000)
    selection_train, selection_dev, split_fingerprint = deterministic_internal_split(all_train, config.seed)
    validation: list[ScoreRow] | None = None
    rationales_train = load_rationales(Path(config.rationale_train_path), config.rationale_train_sha256, all_train) if view == "rationale" else None
    if smoke:
        selection_train, selection_dev = selection_train[:8], selection_dev[:8]

    def initialize() -> tuple[Any, Any]:
        set_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
        return tokenizer, build_model(config)

    tokenizer, model = initialize()
    train_dataset = _dataset(selection_train, view, rationales_train, tokenizer, config.max_length)
    dev_dataset = _dataset(selection_dev, view, rationales_train, tokenizer, config.max_length)
    events: list[dict[str, Any]] = []

    class Capture(TrainerCallback):
        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Mapping[str, Any] | None = None, model: Any | None = None, **_: Any) -> Any:
            if not state.is_world_process_zero:
                return control
            epoch = int(round(float(state.epoch or 0)))
            need(metrics is not None and model is not None, "KURE selection metrics differ")
            path = output / "selection" / f"epoch-{epoch:02d}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(_trainable_state(model), str(path))
            events.append({
                "epoch": epoch,
                "global_step": int(state.global_step),
                "macro_integer_rmse": float(metrics["eval_macro_integer_rmse"]),
                "macro_integer_spearman": float(metrics["eval_macro_integer_spearman"]),
                "macro_continuous_rmse": float(metrics["eval_macro_continuous_rmse"]),
                "macro_continuous_spearman": float(metrics["eval_macro_continuous_spearman"]),
                "state_path": str(path.resolve()),
                "state_sha256": file_sha256(path),
            })
            return control

    args = TrainingArguments(
        output_dir=str(output / "selection/trainer"), do_train=True, do_eval=True, eval_strategy="epoch", save_strategy="no",
        num_train_epochs=1 if smoke else len(config.selection_epochs), max_steps=1 if smoke else -1,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size, per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=1 if smoke else config.gradient_accumulation_steps, bf16=False, tf32=True,
        report_to=[], remove_unused_columns=False, dataloader_num_workers=0, ddp_find_unused_parameters=True,
        logging_steps=1 if smoke else 5, save_only_model=True, seed=config.seed, data_seed=config.seed,
    )
    selector = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=dev_dataset, data_collator=_collator(tokenizer), compute_metrics=_metrics, callbacks=[Capture()])
    selector.train(); selector.accelerator.wait_for_everyone()
    shared: list[Any] = [events if selector.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(shared, src=0)
    events = shared[0]
    need(isinstance(events, list) and events, "KURE selection emitted no metrics")
    selected = select_epoch(events)
    if smoke:
        payload = {"status": "completed", "mode": "gpu0_smoke", "view": view, "selection": events, "split_fingerprint": split_fingerprint, **prompt_provenance(config.score_prompt_kind)}
        if selector.is_world_process_zero():
            (output / "smoke_complete.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload

    del selector, model, tokenizer, train_dataset, dev_dataset
    torch.cuda.empty_cache()
    tokenizer, model = initialize()
    refit_dataset = _dataset(all_train, view, rationales_train, tokenizer, config.max_length)
    refit_args = TrainingArguments(
        output_dir=str(output / "refit/trainer"), do_train=True, do_eval=False, eval_strategy="no", save_strategy="no",
        num_train_epochs=float(selected["epoch"]), learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size, gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=False, tf32=True, report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
        ddp_find_unused_parameters=True, logging_steps=5, save_only_model=True, seed=config.seed, data_seed=config.seed,
    )
    refitter = Trainer(model=model, args=refit_args, train_dataset=refit_dataset, data_collator=_collator(tokenizer))
    trained = refitter.train(); refitter.accelerator.wait_for_everyone()
    if refitter.is_world_process_zero():
        state = output / "final_trainable_state.safetensors"
        save_file(_trainable_state(model), str(state))
    refitter.accelerator.wait_for_everyone()

    # Canonical validation remains unreachable until selection and all-train refit finish.
    validation = load_score_rows(Path(config.validation_path), config.validation_sha256, 400)
    rationales_validation = load_rationales(Path(config.rationale_validation_path), config.rationale_validation_sha256, validation) if view == "rationale" else None
    validation_dataset = _dataset(validation, view, rationales_validation, tokenizer, config.max_length)
    final_metrics = _predict(refitter, validation_dataset)
    payload = {
        "schema_version": "mal2026-official-kure-score-result-v1", "status": "completed", "run_id": config.run_id,
        "view": view, "model_id": config.model_id, "model_revision": config.model_revision,
        "score_fields": list(AXES), "average_read": False, "average_target_used": False,
        "selection_source": "train_internal_1600_400_only", "selection": {"events": events, "selected": selected, "split_fingerprint": split_fingerprint},
        "refit": {"records": 2000, "epochs": int(selected["epoch"]), "global_step": int(refitter.state.global_step), "train_loss": float(trained.metrics["train_loss"])},
        "canonical_validation": {"records": 400, "use": "single_final_descriptive_evaluation_not_selection", "metrics": final_metrics},
        "rationale_source": None if view == "essay" else {"key": config.rationale_key, "train_sha256": config.rationale_train_sha256, "validation_sha256": config.rationale_validation_sha256},
        "state_path": str((output / "final_trainable_state.safetensors").resolve()), "state_sha256": file_sha256(output / "final_trainable_state.safetensors"),
        **prompt_provenance(config.score_prompt_kind), "config": asdict(config),
        "privacy": "aggregate_only_no_rows_text_ids_rationales_or_predictions_persisted",
    }
    if refitter.is_world_process_zero():
        (output / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    refitter.accelerator.wait_for_everyone()
    return payload
