"""Direct and rationale-aware score encoders for the evaluation.txt matrix.

Both model families start from their complete AI-Hub-trained three-axis state,
attach a fresh MAL LoRA, select an epoch only on a deterministic split of the
2,000 training rows, refit on all training rows, and evaluate the frozen 400
row validation split once.  The source ``score.average`` field is never read.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .evaluation_prompt_matrix import (
    RATIONALE_SCORE_BLIND,
    RATIONALE_SCORE_CONDITIONED,
    SCORE_DIRECT,
    SCORE_KINDS,
    SCORE_RATIONALE_AWARE,
    prompt_provenance,
    score_embedding_input,
)
from .official_score_matrix import AXES, decode_logits, file_sha256, score_metrics
from .rationale_aware_encoder import (
    MODEL_SPECS,
    ContinuousScoreRow,
    build_model,
    collator,
    deterministic_split,
    load_continuous_rows,
    load_rationales,
    read_json,
    select_epoch,
    trainable_state,
    verify_artifact_inventory,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/evaluation-prompt-score-encoder-v1"
RESTRICTED_ROOT = ROOT / "data/processed/restricted/evaluation_prompt_score_encoder_v1"
INPUT_KINDS = ("direct", "rationale_aware")
RATIONALE_VARIANTS = ("score_blind", "score_conditioned")


class EvaluationPromptScoreEncoderError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationPromptScoreEncoderError(message)


@dataclass(frozen=True)
class EvaluationPromptScoreEncoderConfig:
    schema_version: str
    run_id: str
    model_key: str
    model_id: str
    model_revision: str
    model_path: str
    input_kind: str
    rationale_variant: str | None
    train_path: str
    train_sha256: str
    validation_path: str
    validation_sha256: str
    rationale_key: str | None
    rationale_train_path: str | None
    rationale_train_sha256: str | None
    rationale_validation_path: str | None
    rationale_validation_sha256: str | None
    rationale_manifest_path: str | None
    rationale_manifest_sha256: str | None
    warmstart_completion_path: str
    warmstart_completion_sha256: str
    warmstart_artifact_path: str
    warmstart_artifact_sha256: str
    output_root: str
    restricted_output_root: str
    score_fields: tuple[str, str, str]
    average_target_used: bool
    target_projection: str
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
    def from_json(cls, path: Path, *, require_dependencies: bool = True) -> "EvaluationPromptScoreEncoderConfig":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationPromptScoreEncoderError("score encoder config is unreadable") from exc
        need(isinstance(raw, dict), "score encoder config must be an object")
        for field in ("score_fields", "selection_epochs"):
            need(isinstance(raw.get(field), list), f"{field} must be a list")
            raw[field] = tuple(raw[field])
        need(set(raw) == set(cls.__dataclass_fields__), "score encoder config fields differ")
        value = cls(**raw)
        value.validate(require_dependencies=require_dependencies)
        return value

    def validate(self, *, require_dependencies: bool = True) -> None:
        need(self.schema_version == "mal2026-evaluation-prompt-score-encoder-v1", "score encoder schema differs")
        need(self.model_key in MODEL_SPECS and self.input_kind in INPUT_KINDS, "score encoder arm differs")
        spec = MODEL_SPECS[self.model_key]
        need((self.model_id, self.model_revision) == (spec["model_id"], spec["model_revision"]), "score encoder model pin differs")
        model_name = "qwen3-embedding-8b" if self.model_key == "qwen3_embedding_8b" else "kure-v1"
        arm_name = self.input_kind.replace("_", "-")
        if self.input_kind == "rationale_aware":
            need(self.rationale_variant in RATIONALE_VARIANTS, "rationale-aware score variant differs")
            arm_name += "-" + self.rationale_variant.replace("_", "-")
        else:
            need(self.rationale_variant is None, "direct score arm received a rationale variant")
        expected_run = f"evaluation-prompt-score-encoder-v1-{model_name}-{arm_name}-20260729-001"
        need(self.run_id == expected_run, "score encoder run ID differs")
        need(Path(self.output_root).resolve() == OUTPUT_ROOT.resolve(), "score encoder output root differs")
        need(Path(self.restricted_output_root).resolve() == (RESTRICTED_ROOT / self.run_id).resolve(), "restricted score output root differs")
        need(self.score_fields == AXES and self.average_target_used is False, "score encoder axes differ")
        need(self.target_projection == "none_preserve_raw_continuous", "score encoder target projection differs")
        expected_prompt = SCORE_DIRECT if self.input_kind == "direct" else SCORE_RATIONALE_AWARE
        need(self.score_prompt_kind == expected_prompt and self.score_prompt_kind in SCORE_KINDS, "score prompt kind differs")
        need(self.selection_epochs == tuple(range(1, 9)) and self.seed == 2026072905, "score encoder selection protocol differs")
        expected_length = (
            2560 if self.model_key == "qwen3_embedding_8b" and self.input_kind == "rationale_aware"
            else (2304 if self.model_key == "qwen3_embedding_8b" else 2048)
        )
        need(self.max_length == expected_length, "score encoder max length differs")
        need((self.learning_rate, self.weight_decay, self.warmup_ratio) == (1e-4, 0.01, 0.05), "score encoder optimizer differs")
        expected_batch = (4, 8, 2) if self.model_key == "qwen3_embedding_8b" else (8, 16, 2)
        need((self.per_device_train_batch_size, self.per_device_eval_batch_size, self.gradient_accumulation_steps) == expected_batch, "score encoder batch differs")
        need((self.lora_r, self.lora_alpha, self.lora_dropout) == (16, 32, 0.05), "score encoder LoRA differs")
        expected_dtype = "bfloat16" if self.model_key == "qwen3_embedding_8b" else "float32"
        need(self.training_dtype == expected_dtype, "score encoder dtype differs")
        rationale_values = (
            self.rationale_variant, self.rationale_key, self.rationale_train_path, self.rationale_train_sha256,
            self.rationale_validation_path, self.rationale_validation_sha256,
            self.rationale_manifest_path, self.rationale_manifest_sha256,
        )
        if self.input_kind == "direct":
            need(all(value is None for value in rationale_values), "direct score arm received rationale dependencies")
        else:
            need(all(isinstance(value, str) and value for value in rationale_values), "rationale-aware score arm has unresolved rationale dependencies")
        if not require_dependencies:
            return
        for raw_path, expected_sha, directory, label in (
            (self.model_path, None, True, "model"),
            (self.train_path, self.train_sha256, False, "train"),
            (self.validation_path, self.validation_sha256, False, "validation"),
            (self.warmstart_completion_path, self.warmstart_completion_sha256, False, "warmstart completion"),
        ):
            path = Path(raw_path)
            need((path.is_dir() if directory else path.is_file()) and not path.is_symlink(), f"{label} is unavailable")
            if expected_sha is not None:
                need(file_sha256(path) == expected_sha, f"{label} checksum differs")
        artifact = Path(self.warmstart_artifact_path)
        need(artifact.is_dir() and not artifact.is_symlink(), "warmstart artifact is unavailable")
        completion = read_json(Path(self.warmstart_completion_path), "warmstart completion")
        need(completion.get("status") == "completed" and completion.get("training_method") == "full_parameter", "warmstart is not completed full tuning")
        need(completion.get("score_fields") == list(AXES) and completion.get("average_target_used") is False, "warmstart axes differ")
        state = completion.get("state")
        need(isinstance(state, dict) and Path(state.get("artifact_path", "")).resolve() == artifact.resolve(), "warmstart state path differs")
        need(state.get("artifact_sha256") == self.warmstart_artifact_sha256, "warmstart artifact digest differs")
        if self.input_kind == "rationale_aware":
            assert self.rationale_train_path and self.rationale_train_sha256
            assert self.rationale_validation_path and self.rationale_validation_sha256
            assert self.rationale_manifest_path and self.rationale_manifest_sha256
            for raw_path, digest in (
                (self.rationale_train_path, self.rationale_train_sha256),
                (self.rationale_validation_path, self.rationale_validation_sha256),
                (self.rationale_manifest_path, self.rationale_manifest_sha256),
            ):
                path = Path(raw_path)
                need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to((ROOT / "data/processed/restricted").resolve()), "rationale dependency escaped restricted storage")
                need(file_sha256(path) == digest, "rationale dependency checksum differs")
            manifest = read_json(Path(self.rationale_manifest_path), "rationale manifest")
            need(manifest.get("schema_version") == "mal2026-evaluation-prompt-rationale-handoff-v1", "rationale handoff schema differs")
            need(manifest.get("status") == "completed" and manifest.get("structure") == "bundle", "rationale handoff differs")
            need(manifest.get("rationale_key") == self.rationale_key, "rationale key differs")
            need(manifest.get("rationale_train_sha256") == self.rationale_train_sha256 and manifest.get("rationale_validation_sha256") == self.rationale_validation_sha256, "rationale handoff hashes differ")
            expected_rationale_prompt = RATIONALE_SCORE_BLIND if self.rationale_variant == "score_blind" else RATIONALE_SCORE_CONDITIONED
            need(manifest.get("prompt_kind") == expected_rationale_prompt, "rationale handoff prompt differs")
            need(manifest.get("score_conditioning") is (self.rationale_variant == "score_conditioned"), "rationale conditioning lineage differs")
            if self.rationale_variant == "score_blind":
                need(manifest.get("score_train_sha256") is None and manifest.get("score_validation_sha256") is None, "score-blind rationale handoff received scores")
                need(manifest.get("score_source") is None, "score-blind rationale handoff received a score source")
            else:
                need(all(isinstance(manifest.get(key), str) and len(manifest[key]) == 64 for key in ("score_train_sha256", "score_validation_sha256")), "score-conditioned rationale handoff lacks score lineage")
                score_source = manifest.get("score_source")
                need(isinstance(score_source, dict) and set(score_source) == {"run_id", "model_key", "result_sha256"}, "score-conditioned rationale source differs")
                need(all(isinstance(value, str) and value for value in score_source.values()) and len(score_source["result_sha256"]) == 64, "score-conditioned rationale source identity differs")
            need(manifest.get("human_or_reference_score_read_or_prompted") is False, "rationale generator read reference scores")


def render_input(row: ContinuousScoreRow, config: EvaluationPromptScoreEncoderConfig, rationales: Mapping[str, str] | None) -> str:
    try:
        return score_embedding_input(row.prompt, row.essay, config.score_prompt_kind, rationales)
    except ValueError as exc:
        raise EvaluationPromptScoreEncoderError(str(exc)) from exc


def token_length_audit(
    rows: Sequence[ContinuousScoreRow], config: EvaluationPromptScoreEncoderConfig,
    rationales: Mapping[str, Mapping[str, str]] | None, tokenizer: Any,
) -> dict[str, Any]:
    texts = [render_input(row, config, None if rationales is None else rationales[row.identifier]) for row in rows]
    lengths: list[int] = []
    for start in range(0, len(texts), 128):
        encoded = tokenizer(texts[start:start + 128], add_special_tokens=True, truncation=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    need(bool(lengths) and max(lengths) <= config.max_length, "score encoder input would be truncated")
    ordered = sorted(lengths)
    return {
        "records": len(ordered), "maximum": ordered[-1],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "max_length": config.max_length, "truncated_records": 0,
    }


def make_dataset(
    rows: Sequence[ContinuousScoreRow], config: EvaluationPromptScoreEncoderConfig,
    rationales: Mapping[str, Mapping[str, str]] | None, tokenizer: Any,
) -> Any:
    from datasets import Dataset
    dataset = Dataset.from_dict({
        "text": [render_input(row, config, None if rationales is None else rationales[row.identifier]) for row in rows],
        "labels": [list(row.labels) for row in rows],
    })
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=config.max_length),
        batched=True, remove_columns=["text"],
    )


def metric_callback(result: Any) -> dict[str, float]:
    import torch
    continuous, integers, violations = decode_logits(torch.as_tensor(result.predictions), "bounded_regression")
    values = score_metrics(result.label_ids.tolist(), continuous.tolist(), integers.tolist(), violations.tolist())
    return {key: float(values[key]) for key in ("macro_continuous_rmse", "macro_continuous_spearman", "macro_integer_rmse")}


def predict(trainer: Any, dataset: Any) -> tuple[dict[str, Any], list[list[float]], list[list[int]]]:
    import torch
    result = trainer.predict(dataset)
    continuous, integers, violations = decode_logits(torch.as_tensor(result.predictions), "bounded_regression")
    values = score_metrics(result.label_ids.tolist(), continuous.tolist(), integers.tolist(), violations.tolist())
    need(math.isfinite(float(values["macro_continuous_rmse"])), "score encoder prediction is non-finite")
    return values, continuous.tolist(), integers.tolist()


def write_predictions(
    path: Path, rows: Sequence[ContinuousScoreRow], continuous: Sequence[Sequence[float]], integers: Sequence[Sequence[int]], split: str,
) -> str:
    need(not path.exists() and len(rows) == len(continuous) == len(integers), "score prediction output differs")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row, raw_values, int_values in zip(rows, continuous, integers, strict=True):
            need(len(raw_values) == len(int_values) == 3, "score prediction vector differs")
            payload = {
                "source_id": row.identifier, "split": split,
                "continuous_prediction": {axis: float(raw_values[index]) for index, axis in enumerate(AXES)},
                "emitted_integer_prediction": {axis: int(int_values[index]) for index, axis in enumerate(AXES)},
            }
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    return file_sha256(path)


def run(config: EvaluationPromptScoreEncoderConfig, *, smoke: bool = False) -> dict[str, Any]:
    config.validate(require_dependencies=True)
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("evaluation prompt score encoder requires .venv-standard") from exc

    output = Path(config.output_root) / (f"smoke-{config.run_id}" if smoke else config.run_id)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    need(world_size == (1 if smoke else 4), "score encoder world size differs from the GPU0-smoke/DDP4 protocol")
    marker = output / "warmstart_verified.json"
    if rank == 0:
        need(not output.exists(), f"refusing to reuse score encoder output: {output}")
        output.mkdir(parents=True)
        verified = verify_artifact_inventory(config)  # type: ignore[arg-type]
        marker.write_text(json.dumps(verified, indent=2, sort_keys=True) + "\n")
    else:
        deadline = time.monotonic() + 600
        while not marker.is_file() and time.monotonic() < deadline:
            time.sleep(0.1)
        need(marker.is_file(), "rank zero did not verify score warmstart")
        verified = read_json(marker, "warmstart verification")

    all_train = load_continuous_rows(Path(config.train_path), config.train_sha256, 2000)
    selection_train, selection_dev, split_fingerprint = deterministic_split(all_train, config.seed)
    rationales_train = None
    if config.input_kind == "rationale_aware":
        assert config.rationale_train_path and config.rationale_train_sha256
        rationales_train = load_rationales(Path(config.rationale_train_path), config.rationale_train_sha256, all_train)
    if smoke:
        selection_train, selection_dev = selection_train[:4], selection_dev[:4]

    def initialize() -> tuple[Any, Any, dict[str, Any]]:
        set_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, revision=config.model_revision, local_files_only=True,
            trust_remote_code=False, use_fast=True,
        )
        if tokenizer.pad_token is None:
            need(tokenizer.eos_token is not None, "score tokenizer has no pad token")
            tokenizer.pad_token = tokenizer.eos_token
        model, lineage = build_model(config)  # type: ignore[arg-type]
        return tokenizer, model, lineage

    tokenizer, model, initialization = initialize()
    # Even a one-update smoke must prove that the complete declared train and
    # validation populations fit.  Auditing only the four optimization rows can
    # hide a later DDP preflight failure on a long rationale-aware example.
    train_audit = token_length_audit(all_train, config, rationales_train, tokenizer)
    smoke_validation_audit: dict[str, Any] | None = None
    if smoke:
        smoke_validation = load_continuous_rows(Path(config.validation_path), config.validation_sha256, 400)
        smoke_validation_rationales = None
        if config.input_kind == "rationale_aware":
            assert config.rationale_validation_path and config.rationale_validation_sha256
            smoke_validation_rationales = load_rationales(
                Path(config.rationale_validation_path), config.rationale_validation_sha256, smoke_validation,
            )
        smoke_validation_audit = token_length_audit(
            smoke_validation, config, smoke_validation_rationales, tokenizer,
        )
    train_dataset = make_dataset(selection_train, config, rationales_train, tokenizer)
    dev_dataset = make_dataset(selection_dev, config, rationales_train, tokenizer)
    events: list[dict[str, Any]] = []

    class Capture(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **_: Any) -> Any:
            for key in ("loss", "grad_norm"):
                if logs is not None and key in logs:
                    need(math.isfinite(float(logs[key])), f"non-finite score training {key}")
            return control

        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Mapping[str, Any] | None = None, model: Any | None = None, **_: Any) -> Any:
            need(metrics is not None and model is not None, "score selection evaluation differs")
            epoch = int(round(float(state.epoch or 0)))
            event = {
                "epoch": epoch, "global_step": int(state.global_step),
                "macro_continuous_rmse": float(metrics["eval_macro_continuous_rmse"]),
                "macro_continuous_spearman": float(metrics["eval_macro_continuous_spearman"]),
                "macro_integer_rmse": float(metrics["eval_macro_integer_rmse"]),
            }
            if state.is_world_process_zero:
                path = output / "selection" / f"epoch-{epoch:02d}.safetensors"
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file(trainable_state(model), str(path))
                event.update({"state_path": str(path.resolve()), "state_sha256": file_sha256(path)})
            events.append(event)
            return control

    selection_args = TrainingArguments(
        output_dir=str(output / "selection/trainer"), do_train=True, do_eval=True,
        eval_strategy="epoch", save_strategy="no",
        num_train_epochs=1 if smoke else len(config.selection_epochs), max_steps=1 if smoke else -1,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        optim="adamw_torch_fused", per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=1 if smoke else config.gradient_accumulation_steps,
        bf16=config.training_dtype == "bfloat16", tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=True, ddp_find_unused_parameters=False,
        logging_steps=1 if smoke else 5, seed=config.seed, data_seed=config.seed,
    )
    selector = Trainer(
        model=model, args=selection_args, train_dataset=train_dataset, eval_dataset=dev_dataset,
        data_collator=collator(tokenizer), compute_metrics=metric_callback, callbacks=[Capture()],
    )
    selection_train_result = selector.train()
    selector.accelerator.wait_for_everyone()
    shared: list[Any] = [events if selector.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(shared, src=0)
    events = shared[0]
    need(isinstance(events, list) and events, "score selection event broadcast differs")
    selected = select_epoch(events)

    if smoke:
        payload = {
            "schema_version": "mal2026-evaluation-prompt-score-encoder-result-v1",
            "status": "completed", "mode": "gpu0_one_update_smoke", "run_id": config.run_id,
            "model_key": config.model_key, "input_kind": config.input_kind,
            "rationale_variant": config.rationale_variant,
            "physical_gpu_scope": [0], "world_size": world_size,
            "score_fields": list(AXES), "average_read": False, "average_target_used": False,
            "selection": {"events": events, "selected": selected},
            "train_token_length_audit": train_audit,
            "validation_token_length_audit": smoke_validation_audit,
            "warmstart_verification": verified, "initialization": initialization,
            **prompt_provenance(config.score_prompt_kind),
        }
        if selector.is_world_process_zero():
            (output / "smoke_complete.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return payload

    del selector, model, tokenizer, train_dataset, dev_dataset
    torch.cuda.empty_cache()
    tokenizer, model, refit_initialization = initialize()
    need(refit_initialization == initialization, "score selection/refit initialization differs")
    refit_dataset = make_dataset(all_train, config, rationales_train, tokenizer)
    refit_args = TrainingArguments(
        output_dir=str(output / "refit/trainer"), do_train=True, do_eval=False,
        eval_strategy="no", save_strategy="no", num_train_epochs=float(selected["epoch"]),
        learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        optim="adamw_torch_fused", per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=config.training_dtype == "bfloat16", tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=True, ddp_find_unused_parameters=False,
        logging_steps=5, seed=config.seed, data_seed=config.seed,
    )
    refitter = Trainer(model=model, args=refit_args, train_dataset=refit_dataset, data_collator=collator(tokenizer))
    refit_train_result = refitter.train()
    refitter.accelerator.wait_for_everyone()
    final_state = output / "selected_refit_trainable.safetensors"
    if refitter.is_world_process_zero():
        save_file(trainable_state(model), str(final_state))
    refitter.accelerator.wait_for_everyone()

    train_metrics, train_continuous, train_integers = predict(refitter, refit_dataset)
    validation = load_continuous_rows(Path(config.validation_path), config.validation_sha256, 400)
    rationales_validation = None
    if config.input_kind == "rationale_aware":
        assert config.rationale_validation_path and config.rationale_validation_sha256
        rationales_validation = load_rationales(Path(config.rationale_validation_path), config.rationale_validation_sha256, validation)
    validation_audit = token_length_audit(validation, config, rationales_validation, tokenizer)
    validation_dataset = make_dataset(validation, config, rationales_validation, tokenizer)
    validation_metrics, validation_continuous, validation_integers = predict(refitter, validation_dataset)
    prediction_root = Path(config.restricted_output_root)
    train_prediction_path = prediction_root / "scores.train.jsonl"
    validation_prediction_path = prediction_root / "scores.validation.jsonl"
    train_prediction_sha = validation_prediction_sha = None
    if refitter.is_world_process_zero():
        train_prediction_sha = write_predictions(train_prediction_path, all_train, train_continuous, train_integers, "train")
        validation_prediction_sha = write_predictions(validation_prediction_path, validation, validation_continuous, validation_integers, "validation")
    shared_predictions: list[Any] = [{"train": train_prediction_sha, "validation": validation_prediction_sha} if refitter.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(shared_predictions, src=0)
    prediction_hashes = shared_predictions[0]
    need(isinstance(prediction_hashes, dict) and all(isinstance(value, str) for value in prediction_hashes.values()), "score prediction hashes differ")

    shuffled_metrics = None
    if config.input_kind == "rationale_aware":
        assert rationales_validation is not None
        ordered = sorted(validation, key=lambda row: sha256(f"{config.seed}\0{row.identifier}".encode()).hexdigest())
        shifted = {row.identifier: rationales_validation[ordered[(index + 1) % len(ordered)].identifier] for index, row in enumerate(ordered)}
        shuffled_dataset = make_dataset(validation, config, shifted, tokenizer)
        shuffled_metrics, _, _ = predict(refitter, shuffled_dataset)

    result = {
        "schema_version": "mal2026-evaluation-prompt-score-encoder-result-v1",
        "status": "completed", "mode": "full", "run_id": config.run_id,
        "model_key": config.model_key, "model_id": config.model_id, "model_revision": config.model_revision,
        "input_kind": config.input_kind, "rationale_variant": config.rationale_variant,
        "physical_gpu_scope": [0, 1, 2, 3], "world_size": world_size,
        "score_fields": list(AXES),
        "average_read": False, "average_target_used": False, "target_projection": config.target_projection,
        "selection": {
            "source": "train_internal_prompt_stratified_1600_400_only", "split_fingerprint": split_fingerprint,
            "events": events, "selected": selected,
            "rule": "lowest macro continuous RMSE, then highest continuous Spearman, then projected-integer RMSE, then earlier epoch",
            "train_metrics": {key: float(value) for key, value in selection_train_result.metrics.items() if isinstance(value, (int, float))},
        },
        "refit": {
            "records": 2000, "epochs": int(selected["epoch"]), "global_step": int(refitter.state.global_step),
            "train_metrics": {key: float(value) for key, value in refit_train_result.metrics.items() if isinstance(value, (int, float))},
            "descriptive_train_prediction_metrics": train_metrics,
        },
        "canonical_validation": {
            "records": 400, "use": "single_final_descriptive_evaluation_not_selection",
            "metrics": validation_metrics, "shuffled_rationale_diagnostic_metrics": shuffled_metrics,
        },
        "prediction_outputs": {
            "train_path": str(train_prediction_path.resolve()), "train_sha256": prediction_hashes["train"],
            "validation_path": str(validation_prediction_path.resolve()), "validation_sha256": prediction_hashes["validation"],
            "human_or_reference_score_prompted": False,
        },
        "rationale_source": None if config.input_kind == "direct" else {
            "variant": config.rationale_variant, "key": config.rationale_key,
            "train_sha256": config.rationale_train_sha256,
            "validation_sha256": config.rationale_validation_sha256,
        },
        "train_token_length_audit": train_audit, "validation_token_length_audit": validation_audit,
        "warmstart_verification": verified, "initialization": initialization,
        "state_path": str(final_state.resolve()), "state_sha256": file_sha256(final_state),
        **prompt_provenance(config.score_prompt_kind), "config": asdict(config),
        "privacy": "aggregate_result_only_row_predictions_restricted_no_text_or_scores_in_result",
    }
    if refitter.is_world_process_zero():
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    refitter.accelerator.wait_for_everyone()
    return result
