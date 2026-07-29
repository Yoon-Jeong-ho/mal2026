"""Train rationale-aware encoders with source-disjoint Solar augmentation.

Epoch selection uses only original MAL train rows: 1,600 source groups plus
their 4,800 augmentations train the model, while the remaining 400 original
source groups are the development set and none of their augmentations is used.
After selection, refit restarts from the same AI-Hub full-tuned initialization
on all 2,000 originals plus all 6,000 augmentations. Canonical validation is
loaded only after refit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .augmented_bundle_rationale import AugmentedRow, SOLAR_RESULT, load_completed_solar
from .official_score_matrix import AXES, file_sha256
from .official_score_prompt import provenance as prompt_provenance
from .rationale_aware_encoder import (
    ContinuousScoreRow,
    RationaleEncoderConfig,
    build_model,
    collator,
    deterministic_split,
    load_continuous_rows,
    load_rationales,
    make_dataset,
    metrics,
    need as base_need,
    predict_metrics,
    read_json,
    select_epoch,
    shuffled_rationales,
    token_length_audit,
    trainable_state,
    verify_artifact_inventory,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/augmented-rationale-aware-encoder-v1"
RESTRICTED_ROOT = (ROOT / "data/processed/restricted").resolve()
AUGMENTED_RATIONALE_RUN_ID = "official-dpo-bundle-solar-augmented-rationales-v1-20260729-001"
AUGMENTED_RATIONALE_RESULT = (
    ROOT / "outputs/solar-augmented-bundle-rationale-v1" /
    AUGMENTED_RATIONALE_RUN_ID / "result.json"
)


class AugmentedRationaleEncoderError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise AugmentedRationaleEncoderError(message)


@dataclass(frozen=True)
class AugmentedEncoderConfig:
    schema_version: str
    run_id: str
    model_key: str
    base_config_path: str
    base_config_sha256: str
    solar_result_path: str
    augmented_rationale_result_path: str
    output_root: str
    score_fields: tuple[str, str, str]
    average_target_used: bool
    target_projection: str
    seed: int
    selection_epochs: tuple[int, ...]
    training_data_contract: str
    selection_dev_contract: str
    validation_contract: str

    @classmethod
    def from_json(cls, path: Path, *, require_dependencies: bool = True) -> "AugmentedEncoderConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        need(isinstance(raw, dict), "augmented encoder config must be an object")
        for key in ("score_fields", "selection_epochs"):
            need(isinstance(raw.get(key), list), f"{key} must be a list")
            raw[key] = tuple(raw[key])
        need(set(raw) == set(cls.__dataclass_fields__), "augmented encoder config fields differ")
        config = cls(**raw)
        config.validate(require_dependencies=require_dependencies)
        return config

    def validate(self, *, require_dependencies: bool = True) -> None:
        need(self.schema_version == "mal2026-augmented-rationale-aware-encoder-v1", "augmented encoder schema differs")
        need(self.model_key in {"qwen3_embedding_8b", "kure_v1"}, "augmented encoder model differs")
        expected_run = {
            "qwen3_embedding_8b": "augmented-rationale-aware-qwen3-embedding-8b-v1-20260729-001",
            "kure_v1": "augmented-rationale-aware-kure-v1-v1-20260729-001",
        }[self.model_key]
        need(self.run_id == expected_run, "augmented encoder run identity differs")
        need(Path(self.output_root).resolve() == OUTPUT_ROOT.resolve(), "augmented encoder output root differs")
        need(self.score_fields == AXES and self.average_target_used is False, "augmented encoder axes differ")
        need(self.target_projection == "none_preserve_raw_continuous", "augmented target projection differs")
        need(self.seed == 2026072903 and self.selection_epochs == (1, 2, 3, 4), "augmented selection schedule differs")
        need(self.training_data_contract == "original_one_plus_three_axis_degraded_per_source", "augmented data contract differs")
        need(self.selection_dev_contract == "original_only_source_disjoint_from_all_training_augmentations", "augmented dev contract differs")
        need(self.validation_contract == "frozen_original_validation_once_after_selected_epoch_refit", "validation contract differs")
        need(Path(self.solar_result_path).resolve() == SOLAR_RESULT.resolve(), "Solar result binding differs")
        need(Path(self.augmented_rationale_result_path).resolve() == AUGMENTED_RATIONALE_RESULT.resolve(), "augmented rationale result binding differs")
        base_path = Path(self.base_config_path)
        need(base_path.is_file() and file_sha256(base_path) == self.base_config_sha256, "base encoder config binding differs")
        base = RationaleEncoderConfig.from_json(base_path, require_dependencies=require_dependencies)
        need(base.model_key == self.model_key and base.seed == self.seed, "base encoder identity differs")
        if not require_dependencies:
            return
        solar_result = Path(self.solar_result_path)
        rationale_result = Path(self.augmented_rationale_result_path)
        need(solar_result.is_file() and not solar_result.is_symlink(), "Solar result is unavailable")
        need(rationale_result.is_file() and not rationale_result.is_symlink(), "augmented rationale result is unavailable")


def base_config(config: AugmentedEncoderConfig) -> RationaleEncoderConfig:
    return RationaleEncoderConfig.from_json(Path(config.base_config_path), require_dependencies=True)


def augmented_score_rows(rows: Sequence[AugmentedRow]) -> list[ContinuousScoreRow]:
    result: list[ContinuousScoreRow] = []
    for row in rows:
        result.append(ContinuousScoreRow(
            identifier=row.identifier,
            document_id=row.source_id,
            prompt_num=f"solar-{row.target_axis}",
            prompt=row.prompt,
            essay=row.essay,
            labels=row.score,
        ))
    need(len(result) == 6000 and len({row.identifier for row in result}) == 6000, "augmented score rows differ")
    return result


def load_augmented_rationales(
    config: AugmentedEncoderConfig, rows: Sequence[AugmentedRow], solar_result: Mapping[str, Any],
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    result_path = Path(config.augmented_rationale_result_path)
    result = read_json(result_path, "augmented rationale result")
    need(result.get("schema_version") == "mal2026-solar-augmented-bundle-rationale-result-v1", "augmented rationale result schema differs")
    need(result.get("status") == "completed" and result.get("run_id") == AUGMENTED_RATIONALE_RUN_ID, "augmented rationale result differs")
    need(result.get("records") == 6000 and result.get("parse_valid") == 6000, "augmented rationale counts differ")
    need(result.get("structure") == "bundle" and result.get("axis_triplet_used") is False, "augmented rationale is not bundle-only")
    need(result.get("human_or_reference_score_read_or_prompted") is False, "augmented rationale read a protected score")
    rationale_path = Path(str(result.get("rationale_path", "")))
    handoff_path = Path(str(result.get("handoff_path", "")))
    for path in (rationale_path, handoff_path):
        need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(RESTRICTED_ROOT), "augmented rationale path differs")
    need(file_sha256(rationale_path) == result.get("rationale_sha256"), "augmented rationale checksum differs")
    need(file_sha256(handoff_path) == result.get("handoff_sha256"), "augmented rationale handoff checksum differs")
    handoff = read_json(handoff_path, "augmented rationale handoff")
    need(handoff.get("status") == "completed" and handoff.get("structure") == "bundle", "augmented rationale handoff differs")
    need(handoff.get("axis_triplet_used_for_training_or_selection") is False, "augmented axis-triplet lineage is forbidden")
    need(handoff.get("solar_augmented_train_sha256") == solar_result.get("augmented_train_sha256"), "Solar/rationale lineage differs")
    need(handoff.get("rationale_train_augmented_sha256") == result.get("rationale_sha256"), "augmented rationale handoff digest differs")
    expected = {row.identifier: row for row in rows}
    mapping: dict[str, dict[str, str]] = {}
    with rationale_path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            need(set(raw) == {"source_id", "source_train_id", "target_axis", "rationales", "attempts"}, "augmented rationale row schema differs")
            identifier = raw["source_id"]
            need(identifier in expected and identifier not in mapping, "augmented rationale linkage differs")
            row = expected[identifier]
            need(raw["source_train_id"] == row.source_id and raw["target_axis"] == row.target_axis, "augmented rationale source metadata differs")
            rationales = raw["rationales"]
            need(isinstance(rationales, dict) and set(rationales) == set(AXES), "augmented rationale axes differ")
            need(all(isinstance(rationales[axis], str) and rationales[axis].strip() for axis in AXES), "augmented rationale text differs")
            mapping[identifier] = {axis: rationales[axis].strip() for axis in AXES}
    need(set(mapping) == set(expected), "augmented rationale population is incomplete")
    return mapping, {
        "result_sha256": file_sha256(result_path),
        "handoff_sha256": file_sha256(handoff_path),
        "rationale_sha256": file_sha256(rationale_path),
        "score_kind": handoff.get("score_kind"),
        "synthetic_scores_prompted": handoff.get("synthetic_scores_prompted"),
    }


def _wait_output_and_verify(base: RationaleEncoderConfig, output: Path) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    marker = output / "warmstart_verified.json"
    if rank == 0:
        need(not output.exists(), f"refusing to reuse augmented encoder output: {output}")
        output.mkdir(parents=True)
        verified = verify_artifact_inventory(base)
        marker.write_text(json.dumps(verified, indent=2, sort_keys=True) + "\n")
        return verified
    deadline = time.monotonic() + 600
    while not marker.is_file() and time.monotonic() < deadline:
        time.sleep(0.1)
    need(marker.is_file(), "rank zero did not verify augmented encoder warmstart")
    verified = read_json(marker, "augmented warmstart verification")
    need(verified.get("artifact_sha256") == base.warmstart_artifact_sha256, "augmented warmstart marker differs")
    return verified


def run(config: AugmentedEncoderConfig, *, smoke: bool = False) -> dict[str, Any]:
    config.validate(require_dependencies=True)
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("augmented encoder training requires .venv-standard") from exc

    base = base_config(config)
    output = Path(config.output_root) / (f"smoke-{config.run_id}" if smoke else config.run_id)
    warmstart_verification = _wait_output_and_verify(base, output)
    original = load_continuous_rows(Path(base.train_path), base.train_sha256, 2000)
    selection_original, dev_original, split_fingerprint = deterministic_split(original, config.seed)
    original_rationales = load_rationales(Path(base.rationale_train_path), base.rationale_train_sha256, original)
    augmented_raw, solar_result = load_completed_solar()
    augmented = augmented_score_rows(augmented_raw)
    augmented_rationales, augmented_lineage = load_augmented_rationales(config, augmented_raw, solar_result)
    train_sources = {row.identifier for row in selection_original}
    dev_sources = {row.identifier for row in dev_original}
    augmented_by_id = {row.identifier: row for row in augmented_raw}
    selection_augmented = [row for row in augmented if augmented_by_id[row.identifier].source_id in train_sources]
    excluded_dev_augmented = [row for row in augmented if augmented_by_id[row.identifier].source_id in dev_sources]
    need((len(selection_augmented), len(excluded_dev_augmented)) == (4800, 1200), "source-disjoint augmented split differs")
    selection_train = [*selection_original, *selection_augmented]
    refit_rows = [*original, *augmented]
    all_rationales = {**original_rationales, **augmented_rationales}
    need(len(all_rationales) == 8000 and not (train_sources & dev_sources), "combined rationale/source split differs")
    if smoke:
        selection_train, dev_original = selection_train[:4], dev_original[:4]

    def initialize() -> tuple[Any, Any, dict[str, Any]]:
        set_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            base.model_path, revision=base.model_revision, local_files_only=True,
            trust_remote_code=False, use_fast=True,
        )
        if tokenizer.pad_token is None:
            base_need(tokenizer.eos_token is not None, "tokenizer has no pad token")
            tokenizer.pad_token = tokenizer.eos_token
        model, lineage = build_model(base)
        return tokenizer, model, lineage

    tokenizer, model, initialization = initialize()
    audit_rows = selection_train + dev_original if smoke else refit_rows
    train_token_audit = token_length_audit(audit_rows, all_rationales, tokenizer, base.max_length)
    train_dataset = make_dataset(selection_train, all_rationales, tokenizer, base.max_length)
    dev_dataset = make_dataset(dev_original, original_rationales, tokenizer, base.max_length)
    events: list[dict[str, Any]] = []

    class Capture(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **_: Any) -> Any:
            for key in ("loss", "grad_norm"):
                if logs is not None and key in logs:
                    need(math.isfinite(float(logs[key])), f"non-finite augmented training {key}")
            return control

        def on_evaluate(self, args: Any, state: Any, control: Any, metrics_value: Mapping[str, Any] | None = None, model: Any | None = None, **kwargs: Any) -> Any:
            reported = metrics_value if metrics_value is not None else kwargs.get("metrics")
            need(reported is not None and model is not None, "augmented selection evaluation differs")
            epoch = int(round(float(state.epoch or 0)))
            event = {
                "epoch": epoch, "global_step": int(state.global_step),
                "macro_continuous_rmse": float(reported["eval_macro_continuous_rmse"]),
                "macro_continuous_spearman": float(reported["eval_macro_continuous_spearman"]),
                "macro_integer_rmse": float(reported["eval_macro_integer_rmse"]),
            }
            if state.is_world_process_zero:
                path = output / "selection" / f"epoch-{epoch:02d}.safetensors"
                path.parent.mkdir(parents=True, exist_ok=True)
                need(not path.exists(), "augmented epoch checkpoint already exists")
                save_file(trainable_state(model), str(path))
                event.update({"state_path": str(path.resolve()), "state_sha256": file_sha256(path)})
            events.append(event)
            return control

    train_args = TrainingArguments(
        output_dir=str(output / "selection/trainer"), do_train=True, do_eval=True,
        eval_strategy="epoch", save_strategy="no",
        num_train_epochs=1 if smoke else len(config.selection_epochs), max_steps=1 if smoke else -1,
        learning_rate=base.learning_rate, weight_decay=base.weight_decay, warmup_ratio=base.warmup_ratio,
        optim="adamw_torch_fused", per_device_train_batch_size=base.per_device_train_batch_size,
        per_device_eval_batch_size=base.per_device_eval_batch_size,
        gradient_accumulation_steps=1 if smoke else base.gradient_accumulation_steps,
        bf16=base.training_dtype == "bfloat16", tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=True, ddp_find_unused_parameters=False,
        logging_steps=1 if smoke else 5, seed=config.seed, data_seed=config.seed,
    )
    selector = Trainer(
        model=model, args=train_args, train_dataset=train_dataset, eval_dataset=dev_dataset,
        data_collator=collator(tokenizer), compute_metrics=metrics, callbacks=[Capture()],
    )
    selection_train_result = selector.train()
    selector.accelerator.wait_for_everyone()
    shared: list[Any] = [events if selector.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(shared, src=0)
    events = shared[0]
    need(isinstance(events, list) and events, "augmented selection event broadcast differs")
    selected = select_epoch(events)

    if smoke:
        payload = {
            "schema_version": "mal2026-augmented-rationale-aware-encoder-result-v1",
            "status": "completed", "mode": "gpu0_one_update_smoke", "model_key": config.model_key,
            "score_fields": list(AXES), "average_read": False, "average_target_used": False,
            "selection": {"events": events, "selected": selected},
            "source_disjoint_counts": {"selection_original": 1600, "selection_augmented": 4800, "dev_original": 400, "excluded_dev_augmented": 1200},
            "train_token_length_audit": train_token_audit,
            "warmstart_verification": warmstart_verification, "initialization": initialization,
        }
        if selector.is_world_process_zero():
            (output / "smoke_complete.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload

    del selector, model, tokenizer, train_dataset, dev_dataset
    torch.cuda.empty_cache()
    tokenizer, model, refit_initialization = initialize()
    need(refit_initialization == initialization, "augmented selection/refit initialization differs")
    refit_dataset = make_dataset(refit_rows, all_rationales, tokenizer, base.max_length)
    refit_args = TrainingArguments(
        output_dir=str(output / "refit/trainer"), do_train=True, do_eval=False,
        eval_strategy="no", save_strategy="no", num_train_epochs=float(selected["epoch"]),
        learning_rate=base.learning_rate, weight_decay=base.weight_decay, warmup_ratio=base.warmup_ratio,
        optim="adamw_torch_fused", per_device_train_batch_size=base.per_device_train_batch_size,
        gradient_accumulation_steps=base.gradient_accumulation_steps,
        bf16=base.training_dtype == "bfloat16", tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=True, ddp_find_unused_parameters=False,
        logging_steps=5, seed=config.seed, data_seed=config.seed,
    )
    refitter = Trainer(model=model, args=refit_args, train_dataset=refit_dataset, data_collator=collator(tokenizer))
    refit_result = refitter.train()
    refitter.accelerator.wait_for_everyone()
    final_state = output / "selected_refit_trainable.safetensors"
    if refitter.is_world_process_zero():
        save_file(trainable_state(model), str(final_state))
    refitter.accelerator.wait_for_everyone()

    # Canonical validation is first loaded only after selection and refit.
    validation = load_continuous_rows(Path(base.validation_path), base.validation_sha256, 400)
    validation_rationales = load_rationales(Path(base.rationale_validation_path), base.rationale_validation_sha256, validation)
    validation_token_audit = token_length_audit(validation, validation_rationales, tokenizer, base.max_length)
    validation_dataset = make_dataset(validation, validation_rationales, tokenizer, base.max_length)
    aligned_metrics = predict_metrics(refitter, validation_dataset)
    shuffled = shuffled_rationales(validation, validation_rationales, config.seed)
    shuffled_metrics = predict_metrics(refitter, make_dataset(validation, shuffled, tokenizer, base.max_length))
    baseline_path = Path(base.output_root) / base.run_id / "result.json"
    baseline = read_json(baseline_path, "baseline rationale-aware encoder result")
    baseline_metric = float(baseline["canonical_validation"]["aligned_bundle_metrics"]["macro_continuous_rmse"])
    current_metric = float(aligned_metrics["macro_continuous_rmse"])
    result = {
        "schema_version": "mal2026-augmented-rationale-aware-encoder-result-v1",
        "status": "completed", "mode": "full", "run_id": config.run_id,
        "model_key": base.model_key, "model_id": base.model_id, "model_revision": base.model_revision,
        "score_fields": list(AXES), "average_read": False, "average_target_used": False,
        "target_projection": config.target_projection,
        "selection": {
            "source": "train_internal_source_disjoint_original_dev_only",
            "split_fingerprint": split_fingerprint, "events": events, "selected": selected,
            "rule": "lowest macro continuous RMSE, then highest continuous Spearman, then projected-integer RMSE, then earlier epoch",
            "train_metrics": {key: float(value) for key, value in selection_train_result.metrics.items() if isinstance(value, (int, float))},
        },
        "data": {
            "selection_original": 1600, "selection_augmented": 4800,
            "selection_dev_original": 400, "excluded_dev_source_augmentations": 1200,
            "refit_original": 2000, "refit_augmented": 6000,
            "source_group_leakage": False, "validation_used_for_training_or_selection": False,
            "solar_result_sha256": file_sha256(Path(config.solar_result_path)),
            "solar_augmented_train_sha256": solar_result["augmented_train_sha256"],
            "augmented_rationale_lineage": augmented_lineage,
        },
        "refit": {
            "records": 8000, "epochs": int(selected["epoch"]), "global_step": int(refitter.state.global_step),
            "train_metrics": {key: float(value) for key, value in refit_result.metrics.items() if isinstance(value, (int, float))},
        },
        "canonical_validation": {
            "records": 400, "use": "single_final_descriptive_evaluation_not_selection",
            "aligned_bundle_metrics": aligned_metrics,
            "shuffled_bundle_diagnostic_metrics": shuffled_metrics,
            "rationale_shuffle_used_for_training_or_selection": False,
            "baseline_result_sha256": file_sha256(baseline_path),
            "baseline_macro_continuous_rmse": baseline_metric,
            "delta_macro_continuous_rmse": current_metric - baseline_metric,
        },
        "train_token_length_audit": train_token_audit,
        "validation_token_length_audit": validation_token_audit,
        "warmstart_verification": warmstart_verification, "initialization": initialization,
        "state_path": str(final_state.resolve()), "state_sha256": file_sha256(final_state),
        **prompt_provenance(base.score_prompt_kind), "base_config": asdict(base), "config": asdict(config),
        "privacy": "aggregate_only_no_rows_text_ids_rationales_scores_or_predictions_persisted",
    }
    result_path = output / "result.json"
    if refitter.is_world_process_zero():
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    refitter.accelerator.wait_for_everyone()
    return result
