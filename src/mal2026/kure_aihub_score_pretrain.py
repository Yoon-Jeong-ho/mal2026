"""Full-parameter AI-Hub score pretraining for the pinned KURE encoder.

This stage deliberately sees only the private AI-Hub train-derived
selection/refit splits.  It predicts the three analytic axes and never reads
or trains an average head.  Selection is performed on the manifest's
selection-dev split; the selected epoch is replayed from the same seed on all
48,016 records and exported as a complete backbone-plus-head warm start.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .official_aihub_score_pretrain import IntegerScoreRow, load_integer_split
from .official_score_matrix import AXES, decode_logits, file_sha256, score_metrics
from .official_score_prompt import PUBLIC_SPEC_SCORE_ONLY, embedding_input, provenance


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "nlpai-lab/KURE-v1"
MODEL_REVISION = "d14c8a9423946e268a0c9952fecf3a7aabd73bd9"
MODEL_PATH = ROOT / "outputs/model-cache/nlpai-lab--KURE-v1-d14c8a9423946e268a0c9952fecf3a7aabd73bd9"
MANIFEST_PATH = ROOT / "data/manifests/aihub_human_feedback_v1.json"
OUTPUT_ROOT = ROOT / "outputs/official-kure-aihub-score-full-pretrain-v1"


class KUREAIHubPretrainError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise KUREAIHubPretrainError(message)


def directory_inventory(path: Path) -> tuple[list[dict[str, Any]], str]:
    need(path.is_dir() and not path.is_symlink(), "KURE export directory is unavailable")
    rows = [
        {
            "path": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    need(bool(rows), "KURE export is empty")
    digest = sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return rows, digest


@dataclass(frozen=True)
class KUREAIHubConfig:
    schema_version: str
    run_id: str
    model_id: str
    model_revision: str
    model_path: str
    manifest_path: str
    output_root: str
    score_fields: tuple[str, str, str]
    average_target_used: bool
    target_projection: str
    seed: int
    max_length: int
    max_selection_epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    training_dtype: str
    score_prompt_kind: str

    @classmethod
    def from_json(cls, path: Path, *, require_dependencies: bool = True) -> "KUREAIHubConfig":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KUREAIHubPretrainError("KURE AI-Hub config is unreadable") from exc
        need(isinstance(raw, dict) and isinstance(raw.get("score_fields"), list), "KURE AI-Hub config differs")
        raw["score_fields"] = tuple(raw["score_fields"])
        need(set(raw) == set(cls.__dataclass_fields__), "KURE AI-Hub config fields differ")
        value = cls(**raw)
        value.validate(require_dependencies=require_dependencies)
        return value

    def validate(self, *, require_dependencies: bool = True) -> None:
        need(self.schema_version == "mal2026-kure-aihub-score-pretrain-v1", "KURE AI-Hub schema differs")
        need(self.run_id == "official-kure-aihub-score-full-pretrain-v1-20260729-003", "KURE AI-Hub run ID differs")
        need((self.model_id, self.model_revision) == (MODEL_ID, MODEL_REVISION), "KURE pin differs")
        need(Path(self.model_path).resolve() == MODEL_PATH.resolve(), "KURE model path differs")
        need(Path(self.manifest_path).resolve() == MANIFEST_PATH.resolve(), "AI-Hub manifest path differs")
        need(Path(self.output_root).resolve() == OUTPUT_ROOT.resolve(), "KURE AI-Hub output root differs")
        need(self.score_fields == AXES and self.average_target_used is False, "KURE score axes differ")
        need(self.target_projection == "official_half_up", "KURE AI-Hub target projection differs")
        need(self.score_prompt_kind == PUBLIC_SPEC_SCORE_ONLY, "KURE AI-Hub prompt differs")
        need((self.seed, self.max_length) == (2026072902, 1664), "KURE AI-Hub seed/length differs")
        need((self.max_selection_epochs, self.early_stopping_patience) == (6, 2), "KURE AI-Hub selection differs")
        need((self.learning_rate, self.weight_decay, self.warmup_ratio) == (2e-5, 0.01, 0.05), "KURE AI-Hub optimizer differs")
        need(
            (self.per_device_train_batch_size, self.per_device_eval_batch_size, self.gradient_accumulation_steps) == (16, 32, 1),
            "KURE AI-Hub batch contract differs",
        )
        need(self.training_dtype == "float32", "KURE AI-Hub dtype differs")
        if require_dependencies:
            need(MODEL_PATH.is_dir() and not MODEL_PATH.is_symlink(), "KURE snapshot is unavailable")
            need(MANIFEST_PATH.is_file() and not MANIFEST_PATH.is_symlink(), "AI-Hub manifest is unavailable")


def render_input(row: IntegerScoreRow) -> str:
    return embedding_input(row.prompt, row.essay, PUBLIC_SPEC_SCORE_ONLY)


def token_length_audit(rows: Sequence[IntegerScoreRow], tokenizer: Any, max_length: int) -> dict[str, Any]:
    lengths: list[int] = []
    texts = [render_input(row) for row in rows]
    for start in range(0, len(texts), 256):
        encoded = tokenizer(texts[start:start + 256], add_special_tokens=True, truncation=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    need(bool(lengths) and max(lengths) <= max_length, "KURE AI-Hub input would be truncated")
    ordered = sorted(lengths)
    return {
        "records": len(ordered),
        "maximum": ordered[-1],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "max_length": max_length,
        "truncated_records": 0,
    }


def make_dataset(rows: Sequence[IntegerScoreRow], tokenizer: Any, max_length: int) -> Any:
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "text": [render_input(row) for row in rows],
        "labels": [list(row.labels) for row in rows],
    })
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        labels = [feature.pop("labels") for feature in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch

    return collate


def build_model(config: KUREAIHubConfig) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        config.model_path,
        revision=config.model_revision,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        # The XLM-R pooler is not part of KURE's CLS embedding contract.  If
        # instantiated, its two tensors are unused by the score loss and make
        # strict DDP reduction fail on the second batch.
        add_pooling_layer=False,
    )
    hidden = getattr(backbone.config, "hidden_size", None)
    need(type(hidden) is int and hidden > 0, "KURE hidden size differs")

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.score_head = nn.Linear(hidden, 3, dtype=next(backbone.parameters()).dtype)

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Mapping[str, Any] | None = None) -> None:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        def gradient_checkpointing_disable(self) -> None:
            self.backbone.gradient_checkpointing_disable()

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None, **_: Any) -> Mapping[str, Any]:
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
            pooled = functional.normalize(output[:, 0], p=2, dim=-1)
            logits = self.score_head(pooled.to(self.score_head.weight.dtype)).float()
            result: dict[str, Any] = {"logits": logits}
            if labels is not None:
                prediction, _, _ = decode_logits(logits, "bounded_regression")
                result["loss"] = functional.mse_loss(prediction, labels.float())
            return result

    model = Model()
    need(all(parameter.requires_grad for parameter in model.parameters()), "KURE full pretrain has frozen parameters")
    need(not any("lora_" in name for name, _ in model.named_parameters()), "LoRA leaked into KURE full pretrain")
    return model


def compute_metrics(result: Any) -> dict[str, float]:
    import torch

    continuous, integers, violations = decode_logits(torch.as_tensor(result.predictions), "bounded_regression")
    values = score_metrics(result.label_ids.tolist(), continuous.tolist(), integers.tolist(), violations.tolist())
    return {
        "macro_continuous_rmse": float(values["macro_continuous_rmse"]),
        "macro_continuous_spearman": float(values["macro_continuous_spearman"]),
        "macro_integer_rmse": float(values["macro_integer_rmse"]),
    }


def select_epoch(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    need(bool(events), "KURE AI-Hub selection emitted no events")
    return min(
        events,
        key=lambda row: (
            float(row["macro_continuous_rmse"]),
            -float(row["macro_continuous_spearman"]),
            float(row["macro_integer_rmse"]),
            int(row["epoch"]),
        ),
    )


def wait_output(path: Path) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        need(not path.exists(), f"refusing to reuse KURE AI-Hub output: {path}")
        path.mkdir(parents=True)
        return
    deadline = time.monotonic() + 60
    while not path.is_dir() and time.monotonic() < deadline:
        time.sleep(0.05)
    need(path.is_dir(), "rank zero did not create KURE AI-Hub output")


def run(config: KUREAIHubConfig, *, smoke: bool = False) -> dict[str, Any]:
    config.validate(require_dependencies=True)
    try:
        import torch
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("KURE AI-Hub pretraining requires .venv-standard") from exc

    output = Path(config.output_root) / (f"smoke-{config.run_id}" if smoke else config.run_id)
    wait_output(output)
    selection_train, selection_train_sha = load_integer_split("selection_train", Path(config.manifest_path))
    selection_dev, selection_dev_sha = load_integer_split("selection_dev", Path(config.manifest_path))
    if smoke:
        # Exercise enough optimizer updates to expose numeric instability; a
        # one-step smoke has zero LR under the scheduler and cannot do so.
        selection_train, selection_dev = selection_train[:256], selection_dev[:32]

    def initialize() -> tuple[Any, Any]:
        set_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, revision=config.model_revision, local_files_only=True,
            trust_remote_code=False, use_fast=True,
        )
        return tokenizer, build_model(config)

    tokenizer, model = initialize()
    length_audit = token_length_audit(selection_train + selection_dev, tokenizer, config.max_length)
    train_dataset = make_dataset(selection_train, tokenizer, config.max_length)
    dev_dataset = make_dataset(selection_dev, tokenizer, config.max_length)
    events: list[dict[str, Any]] = []

    class Capture(TrainerCallback):
        def __init__(self) -> None:
            self.best: float | None = None
            self.stale = 0

        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Mapping[str, Any] | None = None, **_: Any) -> Any:
            need(metrics is not None, "KURE AI-Hub evaluation emitted no metrics")
            event = {
                "epoch": int(round(float(state.epoch or 0))),
                "global_step": int(state.global_step),
                "macro_continuous_rmse": float(metrics["eval_macro_continuous_rmse"]),
                "macro_continuous_spearman": float(metrics["eval_macro_continuous_spearman"]),
                "macro_integer_rmse": float(metrics["eval_macro_integer_rmse"]),
            }
            events.append(event)
            if self.best is None or event["macro_continuous_rmse"] < self.best:
                self.best, self.stale = event["macro_continuous_rmse"], 0
            else:
                self.stale += 1
                if self.stale >= config.early_stopping_patience:
                    control.should_training_stop = True
            return control

    args = TrainingArguments(
        output_dir=str(output / "selection_trainer"), do_train=True, do_eval=True,
        eval_strategy="steps" if smoke else "epoch", save_strategy="no",
        eval_steps=25 if smoke else None,
        num_train_epochs=1 if smoke else config.max_selection_epochs,
        max_steps=25 if smoke else -1,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio, optim="adamw_torch_fused",
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=1 if smoke else config.gradient_accumulation_steps,
        bf16=False, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[], remove_unused_columns=False, dataloader_num_workers=2,
        dataloader_pin_memory=True, ddp_find_unused_parameters=False,
        logging_steps=1 if smoke else 25, seed=config.seed, data_seed=config.seed,
    )
    selector = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=dev_dataset,
        data_collator=collator(tokenizer), compute_metrics=compute_metrics, callbacks=[Capture()],
    )
    selected_train = selector.train()
    selector.accelerator.wait_for_everyone()
    shared: list[Any] = [events if selector.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(shared, src=0)
    events = shared[0]
    need(isinstance(events, list) and events, "KURE AI-Hub selection metrics differ")
    selected = select_epoch(events)

    if smoke:
        payload = {
            "schema_version": "mal2026-kure-aihub-score-pretrain-completion-v1",
            "status": "completed", "mode": "gpu0_25_update_numeric_smoke",
            "phase": "selection_only", "score_fields": list(AXES),
            "average_read": False, "average_target_used": False,
            "selection": {"events": events, "selected": selected},
            "token_length_audit": length_audit, **provenance(config.score_prompt_kind),
        }
        if selector.is_world_process_zero():
            (output / "smoke_complete.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return payload

    del selector, model, tokenizer, train_dataset, dev_dataset
    torch.cuda.empty_cache()
    refit_rows, refit_sha = load_integer_split("refit_train", Path(config.manifest_path))
    tokenizer, model = initialize()
    refit_dataset = make_dataset(refit_rows, tokenizer, config.max_length)
    refit_args = TrainingArguments(
        output_dir=str(output / "refit_trainer"), do_train=True, do_eval=False,
        eval_strategy="no", save_strategy="no", num_train_epochs=float(selected["epoch"]),
        learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio, optim="adamw_torch_fused",
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=False, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[], remove_unused_columns=False, dataloader_num_workers=2,
        dataloader_pin_memory=True, ddp_find_unused_parameters=False,
        logging_steps=25, seed=config.seed, data_seed=config.seed,
    )
    refitter = Trainer(model=model, args=refit_args, train_dataset=refit_dataset, data_collator=collator(tokenizer))
    refit_train = refitter.train()
    refitter.accelerator.wait_for_everyone()
    artifact = output / "full_model"
    refitter.save_model(str(artifact))
    refitter.accelerator.wait_for_everyone()

    inventory = artifact_sha = None
    if refitter.is_world_process_zero():
        inventory, artifact_sha = directory_inventory(artifact)
    shared_artifact: list[Any] = [{"inventory": inventory, "artifact_sha256": artifact_sha} if refitter.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(shared_artifact, src=0)
    artifact_state = shared_artifact[0]
    need(isinstance(artifact_state, dict), "KURE AI-Hub artifact broadcast differs")

    payload = {
        "schema_version": "mal2026-kure-aihub-score-pretrain-completion-v1",
        "status": "completed", "mode": "full", "run_id": config.run_id,
        "model_id": config.model_id, "model_revision": config.model_revision,
        "score_fields": list(AXES), "average_read": False,
        "average_target_used": False, "target_projection": config.target_projection,
        "training_method": "full_parameter",
        "data": {
            "selection_train_records": 38419, "selection_train_sha256": selection_train_sha,
            "selection_dev_records": 9597, "selection_dev_sha256": selection_dev_sha,
            "refit_records": 48016, "refit_sha256": refit_sha,
            "canonical_validation_access": False,
        },
        "selection": {
            "events": events, "selected": selected,
            "train_metrics": {k: float(v) for k, v in selected_train.metrics.items() if isinstance(v, (int, float))},
        },
        "refit": {
            "epochs": int(selected["epoch"]), "global_step": int(refitter.state.global_step),
            "train_metrics": {k: float(v) for k, v in refit_train.metrics.items() if isinstance(v, (int, float))},
        },
        "state": {
            "artifact_path": str(artifact.resolve()),
            "artifact_sha256": artifact_state["artifact_sha256"],
            "inventory": artifact_state["inventory"],
            "state_scope": "complete_full_parameter_backbone_plus_matched_three_score_head",
        },
        "token_length_audit": length_audit,
        **provenance(config.score_prompt_kind), "config": asdict(config),
        "privacy": "aggregate_only_no_rows_prompts_essays_feedback_ids_predictions_or_model_outputs_persisted",
    }
    completion = output / "training_complete.json"
    if refitter.is_world_process_zero():
        need(not completion.exists(), "KURE AI-Hub completion already exists")
        completion.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    refitter.accelerator.wait_for_everyone()
    return payload
