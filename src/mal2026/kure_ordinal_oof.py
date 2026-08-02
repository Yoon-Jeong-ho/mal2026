"""Exact train-only five-fold KURE ordinal OOF runner.

The module deliberately has no validation-path field.  Phase-2 method choice is
accepted only from a checksum-bound fixed-feature aggregate, and row predictions
and checkpoints are written only below the restricted output root.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .iterative_tail_metrics import AXES, compute_iterative_tail_metrics, metric_improvements
from .kure_axis_contrastive import (
    MODEL_CONFIG_SHA256, build_model, render_input, token_length_audit,
)
from .official_score_matrix import ScoreRow, file_sha256, load_score_rows
from .ordinal_tail_fixed_feature import (
    CandidateSpec, coral_pmf, corn_pmf, corn_targets, effective_number_weights,
    rps_loss, slace_loss,
)
from .r0_ordinal_residual import load_embedding_artifact


SUPPORTED_FAMILIES = {"softmax_ce", "rps", "coral", "corn", "slace"}


class KUREOrdinalOOFError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise KUREOrdinalOOFError(message)


def derived_seed(base: int, outer_fold: int, method: str, axis: str, phase: str) -> int:
    payload = f"{base}\0{outer_fold}\0{method}\0{axis}\0{phase}".encode()
    return int.from_bytes(sha256(payload).digest()[:4], "big") % (2**31 - 1)


def seed_runtime(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_sha256(config: "KUREOrdinalOOFConfig") -> str:
    payload = json.dumps(asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def _git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _environment() -> Mapping[str, Any]:
    import torch
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "transformers": transformers_version, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def _secure_dir(path: Path) -> None:
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor); cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        os.chmod(directory, 0o700)
    os.chmod(path, 0o700)
    # The project volume enforces a group-restricted 0770 ACL even after a
    # 0700 chmod.  Accept user-only or project-group-only access, but fail
    # closed if any world bit is present or the owner lacks full access.
    need(all(
        ((directory.stat().st_mode & 0o007) == 0)
        and ((directory.stat().st_mode & 0o700) == 0o700)
        for directory in [path, *missing]
    ), "restricted directory permission differs")


@dataclass(frozen=True)
class BackboneSpec:
    arm: str
    model_id: str
    model_revision: str
    model_path: str
    model_config_sha256: str
    warmstart_completion_path: str
    warmstart_completion_sha256: str
    warmstart_artifact_path: str
    warmstart_artifact_sha256: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float


@dataclass(frozen=True)
class KUREOrdinalOOFConfig:
    schema_version: str
    run_id: str
    train_path: str
    train_sha256: str
    fold_manifest_path: str
    fold_manifest_sha256: str
    fold_rows_path: str
    fold_rows_sha256: str
    r0_oof_prediction_path: str
    r0_oof_prediction_sha256: str
    stage2_aggregate_path: str
    stage2_aggregate_sha256: str
    stage2_config_sha256: str
    backbone: BackboneSpec
    output_root: str
    restricted_output_root: str
    seed: int
    epochs: int
    crt_epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    raw_rmse_auxiliary_weight: float
    axes: tuple[str, ...]
    average_target_forbidden: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "KUREOrdinalOOFConfig":
        need(isinstance(raw, Mapping), "KURE OOF config must be an object")
        def forbidden(value: Any) -> bool:
            if isinstance(value, Mapping):
                return any("validation" in str(key).lower() or forbidden(nested) for key, nested in value.items())
            if isinstance(value, (list, tuple)):
                return any(forbidden(item) for item in value)
            return isinstance(value, str) and "validation" in value.lower()
        need(not forbidden(raw), "validation fields and paths are forbidden")
        value_raw = dict(raw)
        value_raw["axes"] = tuple(value_raw.get("axes", ()))
        need(isinstance(value_raw.get("backbone"), Mapping), "backbone provenance block is required")
        need(set(value_raw["backbone"]) == set(BackboneSpec.__dataclass_fields__), "backbone provenance fields differ")
        value_raw["backbone"] = BackboneSpec(**value_raw["backbone"])
        need(set(value_raw) == set(cls.__dataclass_fields__), "KURE OOF config fields differ")
        value = cls(**value_raw)
        value.validate(require_dependencies=False)
        return value

    @classmethod
    def from_json(cls, path: str | Path, *, require_dependencies: bool = False) -> "KUREOrdinalOOFConfig":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KUREOrdinalOOFError("KURE OOF config is unreadable") from exc
        value = cls.from_mapping(raw)
        value.validate(require_dependencies=require_dependencies)
        return value

    def validate(self, *, require_dependencies: bool) -> None:
        need(self.schema_version == "mal2026-kure-ordinal-oof-v1", "KURE OOF schema differs")
        need(self.axes == AXES and self.average_target_forbidden is True, "independent-axis/average contract differs")
        need(self.backbone.arm == "aihub_full_backbone" and self.backbone.model_config_sha256 == MODEL_CONFIG_SHA256,
             "KURE/AI-Hub backbone contract differs")
        need((self.backbone.lora_r, self.backbone.lora_alpha, self.backbone.lora_dropout) == (16, 32, 0.05),
             "LoRA contract differs")
        need(Path(self.output_root).resolve() != Path(self.restricted_output_root).resolve(), "public/restricted roots must differ")
        need(self.epochs > 0 and self.crt_epochs > 0 and self.batch_size > 0 and self.max_length == 1536,
             "training contract differs")
        need(self.raw_rmse_auxiliary_weight == 0.25, "raw RMSE auxiliary contract differs")
        need(self.learning_rate > 0 and self.weight_decay >= 0 and self.gradient_accumulation_steps > 0,
             "optimizer contract differs")
        for digest in (self.train_sha256, self.fold_manifest_sha256, self.fold_rows_sha256,
                       self.r0_oof_prediction_sha256, self.stage2_aggregate_sha256,
                       self.stage2_config_sha256):
            need(isinstance(digest, str) and len(digest) == 64, "checksum contract differs")
        if require_dependencies:
            for path, digest, label in (
                (self.train_path, self.train_sha256, "train"),
                (self.fold_manifest_path, self.fold_manifest_sha256, "fold manifest"),
                (self.fold_rows_path, self.fold_rows_sha256, "fold rows"),
                (self.r0_oof_prediction_path, self.r0_oof_prediction_sha256, "exact R0 OOF"),
                (self.stage2_aggregate_path, self.stage2_aggregate_sha256, "stage2 aggregate"),
            ):
                need(Path(path).is_file() and file_sha256(Path(path)) == digest, f"{label} binding differs")


def _candidate(identifier: str, family: str) -> CandidateSpec:
    if identifier.startswith("slace-a"):
        return CandidateSpec(identifier, family, "slace_internal", alpha=float(identifier.removeprefix("slace-a")))
    if identifier.startswith("ce-effective-b"):
        return CandidateSpec(identifier, family, "effective_number", beta=float(identifier.removeprefix("ce-effective-b")))
    if identifier == "ce-sqrt-sampler":
        return CandidateSpec(identifier, family, "sqrt_sampler")
    return CandidateSpec(identifier, family, "natural")


def load_recommended_methods(config: KUREOrdinalOOFConfig) -> tuple[CandidateSpec, CandidateSpec]:
    path = Path(config.stage2_aggregate_path)
    need(path.is_file() and file_sha256(path) == config.stage2_aggregate_sha256, "stage2 aggregate binding differs")
    try:
        aggregate = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KUREOrdinalOOFError("stage2 aggregate is unreadable") from exc
    need(aggregate.get("status") == "completed" and aggregate.get("config_sha256") == config.stage2_config_sha256,
         "stage2 completion/config binding differs")
    recommended = aggregate.get("phase2_recommended_distinct_families")
    need(isinstance(recommended, list) and len(recommended) == 2, "exactly two phase2 recommendations are required")
    methods = tuple(_candidate(str(item["candidate_id"]), str(item["family"])) for item in recommended)
    need(len({method.family for method in methods}) == 2 and all(method.family in SUPPORTED_FAMILIES for method in methods),
         "phase2 methods must be two supported distinct families")
    return methods  # type: ignore[return-value]


def load_train_and_folds(config: KUREOrdinalOOFConfig) -> tuple[list[ScoreRow], Mapping[str, int]]:
    rows = load_score_rows(Path(config.train_path), config.train_sha256, 2000)
    need(file_sha256(Path(config.fold_manifest_path)) == config.fold_manifest_sha256, "fold manifest binding differs")
    need(file_sha256(Path(config.fold_rows_path)) == config.fold_rows_sha256, "fold rows binding differs")
    manifest, fold_rows = load_embedding_artifact(config.fold_manifest_path, config.fold_rows_path)
    need(manifest.split_role == "train" and manifest.base_prediction_origin == "oof"
         and not manifest.contains_average_target, "fold artifact contract differs")
    folds = {row.source_id: row.oof_fold for row in fold_rows}
    need(set(folds) == {row.identifier for row in rows} and set(folds.values()) == set(range(5)), "fold population differs")
    return rows, {key: int(value) for key, value in folds.items() if value is not None}


def load_exact_r0(config: KUREOrdinalOOFConfig) -> Mapping[str, tuple[float, float, float]]:
    need(file_sha256(Path(config.r0_oof_prediction_path)) == config.r0_oof_prediction_sha256,
         "exact R0 OOF binding differs")
    _, fold_rows = load_embedding_artifact(config.fold_manifest_path, config.fold_rows_path)
    expected = {row.source_id: row for row in fold_rows}
    result: dict[str, tuple[float, float, float]] = {}
    with Path(config.r0_oof_prediction_path).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line); source_id = item["source_id"]
            need(source_id in expected and source_id not in result and item["fold"] == expected[source_id].oof_fold,
                 "exact R0 row/fold differs")
            prediction = tuple(float(item["continuous_prediction"][axis]) for axis in AXES)
            need(prediction == expected[source_id].base_predictions, "exact R0 axis values differ")
            result[source_id] = prediction  # type: ignore[assignment]
    need(set(result) == set(expected), "exact R0 population differs")
    return result


def load_raw_axis_gold(path: str | Path, expected_sha256: str) -> Mapping[str, tuple[float, float, float]]:
    """Load only the three raw axes; the canonical ``average`` is never indexed."""
    value = Path(path)
    need(value.is_file() and file_sha256(value) == expected_sha256, "raw gold train binding differs")
    result: dict[str, tuple[float, float, float]] = {}
    with value.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            need(set(raw) == {"id", "document_id", "prompt_num", "prompt", "essay", "score"}, "raw gold schema differs")
            identifier, scores = raw["id"], raw["score"]
            need(identifier not in result and isinstance(scores, Mapping) and set(scores) == {*AXES, "average"},
                 "raw gold row differs")
            axes = tuple(float(scores[axis]) for axis in AXES)
            need(all(np.isfinite(item) and 1.0 <= item <= 5.0 for item in axes), "raw gold axis differs")
            result[str(identifier)] = axes  # type: ignore[assignment]
    need(len(result) == 2000, "raw gold population differs")
    return result


def outer_split(rows: Sequence[ScoreRow], folds: Mapping[str, int], outer_fold: int) -> tuple[list[ScoreRow], list[ScoreRow]]:
    need(0 <= outer_fold < 5, "outer fold must be 0..4")
    fit = [row for row in rows if folds[row.identifier] != outer_fold]
    held = [row for row in rows if folds[row.identifier] == outer_fold]
    need((len(fit), len(held)) == (1600, 400), "outer 1600/400 split differs")
    need(not ({row.identifier for row in fit} & {row.identifier for row in held}), "outer fold leakage")
    return fit, held


def ordinal_loss(logits: Any, labels: Any, spec: CandidateSpec, fit_labels: Any) -> Any:
    import torch
    import torch.nn.functional as functional
    pmf = corn_pmf(logits) if spec.family == "corn" else coral_pmf(logits) if spec.family == "coral" else torch.softmax(logits, 1)
    if spec.family == "rps":
        return rps_loss(pmf, labels)
    if spec.family == "coral":
        target = (labels[:, None] > torch.arange(1, 5, device=labels.device)).float()
        return functional.binary_cross_entropy_with_logits(logits, target)
    if spec.family == "corn":
        target, mask = corn_targets(labels)
        return functional.binary_cross_entropy_with_logits(logits[mask], target[mask])
    if spec.family == "slace":
        prior = torch.bincount(fit_labels.long(), minlength=6)[1:].float()
        return slace_loss(logits, labels, prior / prior.sum(), float(spec.alpha))
    weights = effective_number_weights(fit_labels, float(spec.beta)).to(logits.device) if spec.prior_treatment == "effective_number" else None
    return functional.cross_entropy(logits, labels - 1, weight=weights)


def hybrid_loss(logits: Any, labels: Any, raw_labels: Any, spec: CandidateSpec, fit_labels: Any,
                auxiliary_weight: float = 0.25, *, crt: bool = False) -> Any:
    import torch
    import torch.nn.functional as functional
    need(auxiliary_weight == 0.25, "raw RMSE auxiliary contract differs")
    if crt:
        ordinal = functional.cross_entropy(logits, labels - 1)
        pmf = torch.softmax(logits, 1)
    else:
        ordinal = ordinal_loss(logits, labels, spec, fit_labels)
        pmf = corn_pmf(logits) if spec.family == "corn" else coral_pmf(logits) if spec.family == "coral" else torch.softmax(logits, 1)
    expected = (pmf * torch.arange(1, 6, device=logits.device).float()).sum(1)
    return ordinal + auxiliary_weight * functional.mse_loss(expected, raw_labels.float())


def _build_axis_model(backbone_config: BackboneSpec, spec: CandidateSpec) -> tuple[Any, Mapping[str, Any]]:
    import torch
    import torch.nn as nn
    base, lineage = build_model(backbone_config)
    backbone = base.backbone
    output_dim = 4 if spec.family in {"coral", "corn"} else 5

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.crt = False
            if spec.family == "coral":
                self.score = nn.Linear(1024, 1)
                self.cut_base = nn.Parameter(torch.tensor(-1.0))
                self.cut_gaps = nn.Parameter(torch.zeros(3))
            else:
                self.head = nn.Linear(1024, output_dim)

        def enable_crt(self) -> None:
            self.crt = True
            self.head = nn.Linear(1024, 5)

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None) -> Mapping[str, Any]:
            hidden = torch.nn.functional.normalize(
                self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state[:, 0].float(),
                p=2, dim=-1,
            )
            if self.crt or spec.family != "coral":
                logits = self.head(hidden)
            else:
                cuts = torch.cat((self.cut_base.view(1), self.cut_base + torch.cumsum(torch.nn.functional.softplus(self.cut_gaps), 0)))
                logits = self.score(hidden) - cuts.view(1, 4)
            return {"logits": logits}

    del base
    return Model(), lineage


def validate_backbone_without_validation(config: BackboneSpec) -> None:
    """Bind local KURE and AI-Hub tensors without touching canonical validation."""
    model_path = Path(config.model_path)
    need(model_path.is_dir() and not model_path.is_symlink(), "KURE snapshot is unavailable")
    need(file_sha256(model_path / "config.json") == config.model_config_sha256 == MODEL_CONFIG_SHA256, "KURE config checksum differs")
    need(config.arm == "aihub_full_backbone", "exact AI-Hub warm backbone is required")
    for path, digest, label in (
        (config.warmstart_completion_path, config.warmstart_completion_sha256, "AI-Hub completion"),
        (config.warmstart_artifact_path, config.warmstart_artifact_sha256, "AI-Hub backbone"),
    ):
        value = Path(path)
        need(value.is_file() and not value.is_symlink() and file_sha256(value) == digest, f"{label} binding differs")


class _RowsDataset:
    def __init__(self, rows: Sequence[ScoreRow], axis: str, tokenizer: Any, max_length: int,
                 raw_gold: Mapping[str, tuple[float, float, float]]) -> None:
        self.labels = [row.labels[AXES.index(axis)] for row in rows]
        self.raw_labels = [raw_gold[row.identifier][AXES.index(axis)] for row in rows]
        self.ids = [row.identifier for row in rows]
        self.encoded = tokenizer([render_input(row) for row in rows], truncation=True, max_length=max_length)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return {"input_ids": self.encoded["input_ids"][index], "attention_mask": self.encoded["attention_mask"][index],
                "labels": self.labels[index], "raw_labels": self.raw_labels[index]}


def _train_phase(model: Any, dataset: _RowsDataset, tokenizer: Any, spec: CandidateSpec,
                 config: KUREOrdinalOOFConfig, output: Path, *, seed: int, crt: bool = False,
                 max_steps: int = -1) -> Any:
    import torch
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
    fit_labels = torch.tensor(dataset.labels)
    _secure_dir(output)

    class OrdinalTrainer(Trainer):
        def compute_loss(self, current_model: Any, inputs: Mapping[str, Any], return_outputs: bool = False, **_: Any):
            labels = inputs.pop("labels")
            raw_labels = inputs.pop("raw_labels")
            result = current_model(**inputs)
            loss = hybrid_loss(result["logits"], labels, raw_labels, spec, fit_labels,
                               config.raw_rmse_auxiliary_weight, crt=crt)
            return (loss, result) if return_outputs else loss

        def get_train_dataloader(self):
            if crt or spec.prior_treatment != "sqrt_sampler":
                return super().get_train_dataloader()
            from torch.utils.data import DataLoader, WeightedRandomSampler
            counts = torch.bincount(fit_labels, minlength=6)[1:].float()
            weights = torch.tensor([1.0 / float(counts[label - 1]).__pow__(0.5) for label in dataset.labels])
            generator = torch.Generator().manual_seed(seed)
            sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
            return DataLoader(dataset, batch_size=self.args.train_batch_size, sampler=sampler,
                              collate_fn=self.data_collator, generator=generator)

    arguments = TrainingArguments(
        output_dir=str(output), num_train_epochs=config.crt_epochs if crt else config.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=config.batch_size, gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay, save_strategy="no", report_to=[],
        seed=seed, data_seed=seed, remove_unused_columns=False,
    )
    trainer = OrdinalTrainer(model=model, args=arguments, train_dataset=dataset,
                             data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train()
    return trainer


def _write_json_fresh(path: Path, value: Mapping[str, Any], *, private: bool = False) -> str:
    need(not path.exists(), f"refusing to overwrite {path}")
    if not private:
        _validate_public_payload(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if private:
        _secure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    need(not temporary.exists(), "temporary artifact already exists")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if private:
        os.chmod(path, 0o600)
        mode = path.stat().st_mode & 0o777
        need((mode & 0o007) == 0 and (mode & 0o600) == 0o600,
             "restricted JSON permission differs")
    return file_sha256(path)


def _validate_public_payload(value: Any) -> None:
    forbidden = {"source_id", "document_id", "prompt", "essay", "prediction", "raw_gold",
                 "checkpoint_path", "restricted_path"}
    if isinstance(value, Mapping):
        need(not (set(value) & forbidden), "restricted content cannot enter public output")
        for nested in value.values(): _validate_public_payload(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value: _validate_public_payload(nested)


def _write_jsonl_fresh(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    need(not path.exists(), f"refusing to overwrite {path}")
    _secure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path); os.chmod(path, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    mode = path.stat().st_mode & 0o777
    need((mode & 0o007) == 0 and (mode & 0o600) == 0o600,
         "restricted JSONL permission differs")
    return file_sha256(path)


def _save_checkpoint_fresh(path: Path, state: Mapping[str, Any]) -> str:
    from safetensors.torch import save_file
    need(not path.exists(), f"refusing to overwrite {path}")
    _secure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(dict(state), str(temporary)); os.chmod(temporary, 0o600); os.replace(temporary, path)
    mode = path.stat().st_mode & 0o777
    need((mode & 0o007) == 0 and (mode & 0o600) == 0o600,
         "restricted checkpoint permission differs")
    return file_sha256(path)


def run(config: KUREOrdinalOOFConfig | str | Path, *, outer_fold: int, validate_only: bool = False,
        smoke: bool = False) -> Mapping[str, Any]:
    config_argument = str(Path(config).resolve()) if isinstance(config, (str, Path)) else "<in-memory-config>"
    value = KUREOrdinalOOFConfig.from_json(config, require_dependencies=not validate_only) if isinstance(config, (str, Path)) else config
    value.validate(require_dependencies=not validate_only)
    methods = load_recommended_methods(value)
    need(not smoke or outer_fold == 0, "smoke is restricted to outer fold 0")
    if validate_only:
        return {"status": "validated", "methods": [item.identifier for item in methods], "gpu_used": False,
                "config_sha256": config_sha256(value), "validation_rows_loaded": False, "average_target_used": False}
    import torch
    from transformers import AutoTokenizer
    need(torch.cuda.is_available(), "KURE OOF training requires an explicitly launched GPU job")
    if smoke:
        need(os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and torch.cuda.device_count() == 1,
             "smoke requires only physical GPU0 exposed")
    rows, folds = load_train_and_folds(value)
    raw_gold = load_raw_axis_gold(value.train_path, value.train_sha256)
    fit, held = outer_split(rows, folds, outer_fold)
    if smoke:
        subset_seed = derived_seed(value.seed, 0, methods[0].identifier, "content", "smoke_subset")
        rng = random.Random(subset_seed)
        buckets = {score: [row for row in fit if row.labels[0] == score] for score in range(1, 6)}
        for bucket in buckets.values(): rng.shuffle(bucket)
        need(all(len(bucket) >= 2 for bucket in buckets.values()), "smoke fit lacks five-class support")
        fit = [row for score in range(1, 6) for row in buckets[score][:2]]
        rng.shuffle(held); held = held[:8]
    run_methods = methods[:1] if smoke else methods
    run_axes = ("content",) if smoke else AXES
    restricted = Path(value.restricted_output_root) / ("smoke/outer-00" if smoke else f"outer-{outer_fold:02d}")
    _secure_dir(restricted)
    public_results = []
    for method in run_methods:
        axis_predictions = []
        axis_bindings = []
        for axis in run_axes:
            phase1_seed = derived_seed(value.seed, outer_fold, method.identifier, axis, "phase1")
            crt_seed = derived_seed(value.seed, outer_fold, method.identifier, axis, "crt")
            seed_runtime(phase1_seed)
            backbone_config = value.backbone
            validate_backbone_without_validation(backbone_config)
            tokenizer = AutoTokenizer.from_pretrained(backbone_config.model_path, revision=backbone_config.model_revision,
                                                      local_files_only=True, trust_remote_code=False, use_fast=True)
            fit_token_audit = token_length_audit(fit, tokenizer, value.max_length)
            held_token_audit = token_length_audit(held, tokenizer, value.max_length)
            train_data = _RowsDataset(fit, axis, tokenizer, value.max_length, raw_gold)
            held_data = _RowsDataset(held, axis, tokenizer, value.max_length, raw_gold)
            model, lineage = _build_axis_model(backbone_config, method)
            trainer = _train_phase(model, train_data, tokenizer, method, value,
                                   restricted / method.identifier / axis / "lora", seed=phase1_seed,
                                   max_steps=2 if smoke else -1)
            for parameter in model.parameters():
                parameter.requires_grad = False
            seed_runtime(crt_seed)
            model.enable_crt()
            model.head.to(torch.device("cuda"))
            crt_trainer = _train_phase(model, train_data, tokenizer, method, value,
                                       restricted / method.identifier / axis / "crt", seed=crt_seed, crt=True,
                                       max_steps=1 if smoke else -1)
            logits = torch.tensor(crt_trainer.predict(held_data).predictions)
            pmf = torch.softmax(logits, 1)
            axis_predictions.append((pmf * torch.arange(1, 6)).sum(1).numpy())
            state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
                     if "lora_" in name or name.startswith(("head.", "score.", "cut_"))}
            state_path = restricted / method.identifier / axis / "trainable.safetensors"
            state_sha = _save_checkpoint_fresh(state_path, state)
            axis_bindings.append({"axis": axis, "phase1_seed": phase1_seed, "crt_seed": crt_seed,
                                  "checkpoint_sha256": state_sha, "lineage": lineage,
                                  "fit_token_length_audit": fit_token_audit,
                                  "held_token_length_audit": held_token_audit,
                                  "pooling": "cls_l2"})
            del trainer, crt_trainer, model
            torch.cuda.empty_cache()
        predictions = np.column_stack(axis_predictions)
        private_rows = [
            {"source_id": row.identifier, "outer_fold": outer_fold,
             "prediction": {axis: float(predictions[i, j]) for j, axis in enumerate(run_axes)}}
            for i, row in enumerate(held)]
        private_path = restricted / method.identifier / "predictions.jsonl"
        private_sha = _write_jsonl_fresh(private_path, private_rows)
        method_result = {"method": method.identifier, "family": method.family,
                               "crt": {"representation_and_lora_frozen": True, "head_prior": "natural",
                                       "epochs": value.crt_epochs, "loss": "five_way_cross_entropy",
                                       "raw_expected_score_mse_weight": value.raw_rmse_auxiliary_weight,
                                       "inverse_prior_decode": False},
                               "objective": {"phase1": "family_ordinal_loss_plus_raw_expected_score_mse",
                                             "crt": "natural_ce_plus_raw_expected_score_mse",
                                             "raw_rmse_auxiliary_weight": value.raw_rmse_auxiliary_weight},
                               "phase1_disclosure": "family ordinal loss plus 0.25 raw expected-score MSE trains LoRA representation and ordinal head; final reported OOF prediction is the natural-prior cRT head with the same raw auxiliary",
                               "axis_bindings": axis_bindings,
                               "restricted_prediction_sha256": private_sha}
        if not smoke:
            truth = np.asarray([raw_gold[row.identifier] for row in held], dtype=float)
            method_result["metrics"] = compute_iterative_tail_metrics(truth, predictions)
        public_results.append(method_result)
    result = {"schema_version": "mal2026-kure-ordinal-oof-outer-v1", "status": "completed",
              "mode": "smoke" if smoke else "outer_fold", "run_id": value.run_id, "outer_fold": outer_fold,
              "records": len(held), "methods": public_results,
              "config_sha256": config_sha256(value), "git_sha": _git_sha(),
              "logical_command": (f"python scripts/run_kure_ordinal_oof.py --config {config_argument} --outer-fold {outer_fold}"
                                  + (" --smoke" if smoke else "")),
              "environment": _environment(), "stage2_aggregate_sha256": value.stage2_aggregate_sha256,
              "fold_manifest_sha256": value.fold_manifest_sha256, "fold_rows_sha256": value.fold_rows_sha256,
              "kure_model_revision": value.backbone.model_revision,
              "aihub_completion_sha256": value.backbone.warmstart_completion_sha256,
              "aihub_backbone_sha256": value.backbone.warmstart_artifact_sha256,
              "validation_rows_loaded": False, "average_target_used": False,
              "smoke_not_reusable_for_selection_or_scientific_results": smoke,
              "privacy": "aggregate_only_public_predictions_and_checkpoints_restricted"}
    public_path = Path(value.output_root) / ("smoke/outer-00.json" if smoke else f"outer-{outer_fold:02d}.json")
    _write_json_fresh(public_path, result)
    return result


def aggregate(config: KUREOrdinalOOFConfig | str | Path) -> Mapping[str, Any]:
    """Combine the five restricted outer predictions into exact train OOF metrics."""
    config_argument = str(Path(config).resolve()) if isinstance(config, (str, Path)) else "<in-memory-config>"
    value = KUREOrdinalOOFConfig.from_json(config, require_dependencies=True) if isinstance(config, (str, Path)) else config
    value.validate(require_dependencies=True)
    methods = load_recommended_methods(value)
    rows, folds = load_train_and_folds(value)
    truth_by_id = load_raw_axis_gold(value.train_path, value.train_sha256)
    exact_r0_by_id = load_exact_r0(value)
    ordered_ids = [row.identifier for row in rows]
    exact_r0_metrics = compute_iterative_tail_metrics(
        [truth_by_id[key] for key in ordered_ids], [exact_r0_by_id[key] for key in ordered_ids])
    results = []
    for method in methods:
        prediction_by_id: dict[str, Sequence[float]] = {}
        bindings = []
        for fold in range(5):
            public_path = Path(value.output_root) / f"outer-{fold:02d}.json"
            private_path = Path(value.restricted_output_root) / f"outer-{fold:02d}" / method.identifier / "predictions.jsonl"
            need(public_path.is_file() and private_path.is_file(), "outer result is incomplete")
            public = json.loads(public_path.read_text(encoding="utf-8"))
            _validate_outer_report(public, value, fold, methods)
            method_public = next((item for item in public.get("methods", ()) if item.get("method") == method.identifier), None)
            need(method_public is not None and method_public["restricted_prediction_sha256"] == file_sha256(private_path),
                 "outer public/restricted binding differs")
            expected_ids = {source_id for source_id, assigned_fold in folds.items() if assigned_fold == fold}
            private = _load_private_fold(private_path, fold, expected_ids)
            for source_id, prediction in private.items():
                need(source_id not in prediction_by_id, "outer prediction duplicated")
                prediction_by_id[source_id] = prediction
            bindings.append({"outer_fold": fold, "public_sha256": file_sha256(public_path),
                             "restricted_prediction_sha256": file_sha256(private_path),
                             "axis_bindings": method_public["axis_bindings"],
                             "environment": public["environment"]})
        need(set(prediction_by_id) == set(truth_by_id), "full OOF coverage differs")
        metrics = compute_iterative_tail_metrics(
            [truth_by_id[key] for key in ordered_ids], [prediction_by_id[key] for key in ordered_ids])
        results.append({"method": method.identifier, "family": method.family,
                        "metrics": metrics, "improvements_vs_exact_r0": metric_improvements(exact_r0_metrics, metrics),
                        "fold_bindings": bindings,
                        "objective": {"phase1": "family_ordinal_loss_plus_raw_expected_score_mse",
                                      "crt": "natural_ce_plus_raw_expected_score_mse",
                                      "raw_rmse_auxiliary_weight": value.raw_rmse_auxiliary_weight},
                        "crt": {"representation_and_lora_frozen": True, "head_prior": "natural",
                                "inverse_prior_decode": False}})
    result = {"schema_version": "mal2026-kure-ordinal-oof-aggregate-v1", "status": "completed",
              "run_id": value.run_id, "records": 2000, "folds": 5, "methods": results,
              "exact_r0_metrics": exact_r0_metrics, "r0_oof_prediction_sha256": value.r0_oof_prediction_sha256,
              "protected_output": "exact_r0",
              "protection_disclosure": "Stage3 is exploratory; exact R0 remains protected unless a separately preregistered Stage2 promotion gate is passed",
              "config_sha256": config_sha256(value), "git_sha": _git_sha(),
              "logical_command": f"python scripts/run_kure_ordinal_oof.py --config {config_argument} --aggregate",
              "environment": _environment(), "fold_manifest_sha256": value.fold_manifest_sha256,
              "fold_rows_sha256": value.fold_rows_sha256,
              "kure_model_revision": value.backbone.model_revision,
              "aihub_completion_sha256": value.backbone.warmstart_completion_sha256,
              "aihub_backbone_sha256": value.backbone.warmstart_artifact_sha256,
              "stage2_aggregate_sha256": value.stage2_aggregate_sha256,
              "validation_rows_loaded": False, "average_target_used": False,
              "privacy": "aggregate_only_no_rows_text_ids_embeddings_or_predictions"}
    _write_json_fresh(Path(value.output_root) / "aggregate.json", result)
    return result


def _load_private_fold(path: Path, outer_fold: int, expected_ids: set[str]) -> Mapping[str, Sequence[float]]:
    need(len(expected_ids) == 400, "expected outer fold population differs")
    result: dict[str, Sequence[float]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise KUREOrdinalOOFError("restricted outer JSONL is invalid") from exc
            need(isinstance(item, Mapping) and set(item) == {"source_id", "outer_fold", "prediction"},
                 "restricted outer row schema differs")
            source_id, prediction = item["source_id"], item["prediction"]
            need(isinstance(source_id, str) and source_id in expected_ids and source_id not in result
                 and item["outer_fold"] == outer_fold, "restricted outer row/fold differs")
            need(isinstance(prediction, Mapping) and set(prediction) == set(AXES),
                 "restricted outer prediction axes differ")
            values = [prediction[axis] for axis in AXES]
            need(all(type(number) in {int, float} and not isinstance(number, bool)
                     and np.isfinite(float(number)) and 1.0 <= float(number) <= 5.0 for number in values),
                 "restricted outer prediction value differs")
            result[source_id] = [float(number) for number in values]
    need(len(result) == 400 and set(result) == expected_ids, "restricted outer population differs")
    return result


def _validate_outer_report(public: Mapping[str, Any], config: KUREOrdinalOOFConfig, outer_fold: int,
                           methods: Sequence[CandidateSpec]) -> None:
    """Reject smoke, stale, incomplete, or cross-run outer reports."""
    expected = {
        "schema_version": "mal2026-kure-ordinal-oof-outer-v1", "status": "completed",
        "mode": "outer_fold", "run_id": config.run_id, "config_sha256": config_sha256(config),
        "outer_fold": outer_fold, "records": 400, "stage2_aggregate_sha256": config.stage2_aggregate_sha256,
        "fold_manifest_sha256": config.fold_manifest_sha256, "fold_rows_sha256": config.fold_rows_sha256,
        "kure_model_revision": config.backbone.model_revision,
        "aihub_completion_sha256": config.backbone.warmstart_completion_sha256,
        "aihub_backbone_sha256": config.backbone.warmstart_artifact_sha256,
        "validation_rows_loaded": False, "average_target_used": False,
    }
    need(all(public.get(key) == value for key, value in expected.items()), "outer report binding differs")
    reported = public.get("methods")
    need(isinstance(reported, list) and len(reported) == 2
         and [item.get("method") for item in reported] == [method.identifier for method in methods],
         "outer method inventory differs")
    for item, method in zip(reported, methods, strict=True):
        need(item.get("family") == method.family and isinstance(item.get("restricted_prediction_sha256"), str)
             and len(item["restricted_prediction_sha256"]) == 64, "outer method binding differs")
        objective = item.get("objective")
        need(isinstance(objective, Mapping)
             and objective.get("phase1") == "family_ordinal_loss_plus_raw_expected_score_mse"
             and objective.get("crt") == "natural_ce_plus_raw_expected_score_mse"
             and objective.get("raw_rmse_auxiliary_weight") == config.raw_rmse_auxiliary_weight,
             "outer objective binding differs")
        axes = item.get("axis_bindings")
        need(isinstance(axes, list) and len(axes) == 3 and [axis.get("axis") for axis in axes] == list(AXES),
             "outer axis inventory differs")
        for axis in axes:
            need(type(axis.get("phase1_seed")) is int and type(axis.get("crt_seed")) is int
                 and isinstance(axis.get("checkpoint_sha256"), str) and len(axis["checkpoint_sha256"]) == 64
                 and isinstance(axis.get("lineage"), Mapping), "outer seed/checkpoint lineage differs")


__all__ = ["BackboneSpec", "KUREOrdinalOOFConfig", "KUREOrdinalOOFError", "aggregate",
           "_load_private_fold", "_validate_outer_report", "config_sha256", "derived_seed", "load_raw_axis_gold", "load_recommended_methods",
           "hybrid_loss", "load_train_and_folds", "ordinal_loss", "outer_split", "run", "seed_runtime",
           "validate_backbone_without_validation"]
