"""Select the best frozen-validation arm and fit its final train+validation model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .augmented_rationale_encoder import (
    AugmentedEncoderConfig,
    augmented_score_rows,
    load_augmented_rationales,
)
from .augmented_bundle_rationale import load_completed_solar
from .official_score_matrix import AXES, file_sha256
from .official_score_prompt import provenance as prompt_provenance
from .rationale_aware_encoder import (
    RationaleEncoderConfig,
    build_model,
    collator,
    load_continuous_rows,
    load_rationales,
    make_dataset,
    read_json,
    token_length_audit,
    trainable_state,
    verify_artifact_inventory,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/final-rationale-aware-score-encoder-v1"
BASE_CONFIGS = {
    "qwen3_embedding_8b": ROOT / "configs/rationale_aware_qwen3_embedding_8b_aihub_mal.v1.json",
    "kure_v1": ROOT / "configs/rationale_aware_kure_v1_aihub_mal.v1.json",
}
AUGMENTED_CONFIGS = {
    "qwen3_embedding_8b": ROOT / "configs/augmented_rationale_aware_qwen3_embedding_8b.v1.json",
    "kure_v1": ROOT / "configs/augmented_rationale_aware_kure_v1.v1.json",
}
METHODS = ("original_bundle_rationale", "original_plus_solar_augmented_bundle_rationale")


class FinalRationaleEncoderError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise FinalRationaleEncoderError(message)


@dataclass(frozen=True)
class FinalEncoderConfig:
    schema_version: str
    run_id: str
    candidates: tuple[Mapping[str, Any], ...]
    selection_rule: str
    output_root: str
    score_fields: tuple[str, str, str]
    average_target_used: bool
    target_projection: str
    train_plus_validation: bool
    final_evaluation_performed: bool

    @classmethod
    def from_json(cls, path: Path, *, require_dependencies: bool = True) -> "FinalEncoderConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        need(isinstance(raw, dict), "final encoder config must be an object")
        need(isinstance(raw.get("candidates"), list) and isinstance(raw.get("score_fields"), list), "final config arrays differ")
        raw["candidates"] = tuple(raw["candidates"])
        raw["score_fields"] = tuple(raw["score_fields"])
        need(set(raw) == set(cls.__dataclass_fields__), "final encoder config fields differ")
        config = cls(**raw)
        config.validate(require_dependencies=require_dependencies)
        return config

    def validate(self, *, require_dependencies: bool = True) -> None:
        need(self.schema_version == "mal2026-final-rationale-aware-score-encoder-v1", "final encoder schema differs")
        need(self.run_id == "final-rationale-aware-score-encoder-v1-20260729-001", "final encoder run identity differs")
        need(Path(self.output_root).resolve() == OUTPUT_ROOT.resolve(), "final encoder output root differs")
        need(self.score_fields == AXES and self.average_target_used is False, "final score axes differ")
        need(self.target_projection == "none_preserve_raw_continuous", "final target projection differs")
        need(self.train_plus_validation is True and self.final_evaluation_performed is False, "final fit/evaluation contract differs")
        need(self.selection_rule == "lowest frozen-validation macro continuous RMSE, then highest macro continuous Spearman, then fixed candidate order", "final selection rule differs")
        expected = [
            ("original_bundle_rationale", "qwen3_embedding_8b"),
            ("original_bundle_rationale", "kure_v1"),
            ("original_plus_solar_augmented_bundle_rationale", "qwen3_embedding_8b"),
            ("original_plus_solar_augmented_bundle_rationale", "kure_v1"),
        ]
        actual: list[tuple[str, str]] = []
        for candidate in self.candidates:
            need(isinstance(candidate, Mapping) and set(candidate) == {"method", "model_key", "result_path", "expected_sha256"}, "final candidate fields differ")
            actual.append((candidate["method"], candidate["model_key"]))
            need(candidate["method"] in METHODS and candidate["model_key"] in BASE_CONFIGS, "final candidate identity differs")
            expected_sha = candidate["expected_sha256"]
            if candidate["method"] == "original_bundle_rationale":
                need(isinstance(expected_sha, str) and len(expected_sha) == 64, "baseline candidate digest is unresolved")
            else:
                need(expected_sha is None, "future augmented candidate digest was prefilled")
            if require_dependencies:
                path = Path(candidate["result_path"])
                need(path.is_file() and not path.is_symlink(), "final candidate result is unavailable")
                if expected_sha is not None:
                    need(file_sha256(path) == expected_sha, "baseline candidate result checksum differs")
        need(actual == expected, "final candidate order differs")


def candidate_table(config: FinalEncoderConfig) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for order, candidate in enumerate(config.candidates):
        path = Path(candidate["result_path"])
        result = read_json(path, "final candidate result")
        method, model_key = candidate["method"], candidate["model_key"]
        expected_schema = (
            "mal2026-rationale-aware-encoder-result-v1"
            if method == "original_bundle_rationale"
            else "mal2026-augmented-rationale-aware-encoder-result-v1"
        )
        need(result.get("schema_version") == expected_schema and result.get("status") == "completed" and result.get("mode") == "full", "candidate result contract differs")
        need(result.get("model_key") == model_key, "candidate model identity differs")
        metrics = result.get("canonical_validation", {}).get("aligned_bundle_metrics", {})
        rmse, spearman = metrics.get("macro_continuous_rmse"), metrics.get("macro_continuous_spearman")
        need(type(rmse) in {int, float} and type(spearman) in {int, float}, "candidate validation metrics differ")
        need(math.isfinite(float(rmse)) and math.isfinite(float(spearman)), "candidate validation metric is non-finite")
        selected = result.get("selection", {}).get("selected", {})
        epoch = selected.get("epoch")
        need(type(epoch) is int and epoch >= 1, "candidate selected epoch differs")
        table.append({
            "order": order, "method": method, "model_key": model_key,
            "result_path": str(path.resolve()), "result_sha256": file_sha256(path),
            "macro_continuous_rmse": float(rmse), "macro_continuous_spearman": float(spearman),
            "selected_epoch": epoch,
        })
    return table


def select_candidate(table: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    need(len(table) == 4, "final candidate population differs")
    return min(table, key=lambda row: (float(row["macro_continuous_rmse"]), -float(row["macro_continuous_spearman"]), int(row["order"])))


def _wait_output_and_verify(base: RationaleEncoderConfig, output: Path) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    marker = output / "warmstart_verified.json"
    if rank == 0:
        need(not output.exists(), f"refusing to reuse final encoder output: {output}")
        output.mkdir(parents=True)
        verified = verify_artifact_inventory(base)
        marker.write_text(json.dumps(verified, indent=2, sort_keys=True) + "\n")
        return verified
    deadline = time.monotonic() + 600
    while not marker.is_file() and time.monotonic() < deadline:
        time.sleep(0.1)
    need(marker.is_file(), "rank zero did not verify final warmstart")
    verified = read_json(marker, "final warmstart verification")
    need(verified.get("artifact_sha256") == base.warmstart_artifact_sha256, "final warmstart marker differs")
    return verified


def final_data(selected: Mapping[str, Any]) -> tuple[RationaleEncoderConfig, list[Any], dict[str, dict[str, str]], dict[str, Any]]:
    model_key, method = selected["model_key"], selected["method"]
    base = RationaleEncoderConfig.from_json(BASE_CONFIGS[model_key], require_dependencies=True)
    train = load_continuous_rows(Path(base.train_path), base.train_sha256, 2000)
    validation = load_continuous_rows(Path(base.validation_path), base.validation_sha256, 400)
    need(not ({row.identifier for row in train} & {row.identifier for row in validation}), "train/validation IDs overlap")
    train_rationales = load_rationales(Path(base.rationale_train_path), base.rationale_train_sha256, train)
    validation_rationales = load_rationales(Path(base.rationale_validation_path), base.rationale_validation_sha256, validation)
    rows: list[Any] = [*train, *validation]
    rationales = {**train_rationales, **validation_rationales}
    lineage: dict[str, Any] = {
        "original_train_records": 2000, "original_validation_records": 400,
        "validation_role": "training_data_after_final_method_selection",
        "augmented_records": 0,
    }
    if method == "original_plus_solar_augmented_bundle_rationale":
        augmented_config = AugmentedEncoderConfig.from_json(AUGMENTED_CONFIGS[model_key], require_dependencies=True)
        augmented_raw, solar_result = load_completed_solar()
        augmented = augmented_score_rows(augmented_raw)
        augmented_rationales, augmented_lineage = load_augmented_rationales(augmented_config, augmented_raw, solar_result)
        rows.extend(augmented)
        rationales.update(augmented_rationales)
        lineage.update({
            "augmented_records": 6000,
            "solar_result_sha256": file_sha256(Path(augmented_config.solar_result_path)),
            "solar_augmented_train_sha256": solar_result["augmented_train_sha256"],
            "augmented_rationale_lineage": augmented_lineage,
        })
    expected = 8400 if method == "original_plus_solar_augmented_bundle_rationale" else 2400
    need(len(rows) == expected and len(rationales) == expected, "final training population differs")
    return base, rows, rationales, lineage


def run(config: FinalEncoderConfig, *, smoke: bool = False) -> dict[str, Any]:
    config.validate(require_dependencies=True)
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("final encoder training requires .venv-standard") from exc

    table = candidate_table(config)
    selected = select_candidate(table)
    base, rows, rationales, data_lineage = final_data(selected)
    output = Path(config.output_root) / (f"smoke-{config.run_id}" if smoke else config.run_id)
    warmstart = _wait_output_and_verify(base, output)

    set_seed(base.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        base.model_path, revision=base.model_revision, local_files_only=True,
        trust_remote_code=False, use_fast=True,
    )
    if tokenizer.pad_token is None:
        need(tokenizer.eos_token is not None, "final tokenizer has no pad token")
        tokenizer.pad_token = tokenizer.eos_token
    model, initialization = build_model(base)
    audit_rows = rows[:8] if smoke else rows
    token_audit = token_length_audit(audit_rows, rationales, tokenizer, base.max_length)
    train_rows = rows[:8] if smoke else rows
    dataset = make_dataset(train_rows, rationales, tokenizer, base.max_length)

    class FiniteTraining(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **_: Any) -> Any:
            for key in ("loss", "grad_norm"):
                if logs is not None and key in logs:
                    need(math.isfinite(float(logs[key])), f"non-finite final training {key}")
            return control

    training_args = TrainingArguments(
        output_dir=str(output / "trainer"), do_train=True, do_eval=False,
        eval_strategy="no", save_strategy="no",
        num_train_epochs=float(selected["selected_epoch"]), max_steps=1 if smoke else -1,
        learning_rate=base.learning_rate, weight_decay=base.weight_decay, warmup_ratio=base.warmup_ratio,
        optim="adamw_torch_fused", per_device_train_batch_size=base.per_device_train_batch_size,
        gradient_accumulation_steps=1 if smoke else base.gradient_accumulation_steps,
        bf16=base.training_dtype == "bfloat16", tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=True, ddp_find_unused_parameters=False,
        logging_steps=1 if smoke else 5, seed=base.seed, data_seed=base.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator(tokenizer), callbacks=[FiniteTraining()])
    training = trainer.train()
    trainer.accelerator.wait_for_everyone()
    state_path = output / "final_trainable.safetensors"
    if trainer.is_world_process_zero():
        save_file(trainable_state(model), str(state_path))
    trainer.accelerator.wait_for_everyone()
    result = {
        "schema_version": "mal2026-final-rationale-aware-score-encoder-result-v1",
        "status": "completed", "mode": "gpu0_one_update_smoke" if smoke else "final_train_plus_validation",
        "run_id": config.run_id, "candidate_table": table, "selected": dict(selected),
        "selection_source": "previously_completed_frozen_validation_results",
        "final_evaluation_performed": False,
        "model_key": base.model_key, "model_id": base.model_id, "model_revision": base.model_revision,
        "score_fields": list(AXES), "average_read": False, "average_target_used": False,
        "target_projection": config.target_projection,
        "training": {
            "records": len(train_rows), "epochs": int(selected["selected_epoch"]),
            "global_step": int(trainer.state.global_step),
            "metrics": {key: float(value) for key, value in training.metrics.items() if isinstance(value, (int, float))},
        },
        "data_lineage": data_lineage, "token_length_audit": token_audit,
        "warmstart_verification": warmstart, "initialization": initialization,
        "state_path": str(state_path.resolve()), "state_sha256": file_sha256(state_path),
        **prompt_provenance(base.score_prompt_kind), "base_config": asdict(base), "config": asdict(config),
        "privacy": "aggregate_only_no_rows_text_ids_rationales_scores_or_predictions_persisted",
    }
    result_path = output / ("smoke_complete.json" if smoke else "result.json")
    if trainer.is_world_process_zero():
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trainer.accelerator.wait_for_everyone()
    return result
