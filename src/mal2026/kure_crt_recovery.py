"""Train-only recovery of the Stage3 KURE CORAL cRT head.

This deliberately preserves the failed Stage3 experiment: it replays only its
CORAL-natural phase 1, freezes that representation, then learns a fresh
prior-initialized five-class head from cached *fit-fold* CLS-L2 features.  Held
rows are deliberately absent from cRT fitting and the fit-only sanity gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .iterative_tail_metrics import AXES, compute_iterative_tail_metrics, metric_improvements
from .kure_axis_contrastive import MODEL_CONFIG_SHA256, render_input, token_length_audit
from .kure_ordinal_oof import (
    BackboneSpec, KUREOrdinalOOFConfig, _RowsDataset, _build_axis_model, _train_phase,
    derived_seed, load_exact_r0, load_train_and_folds,
    seed_runtime, validate_backbone_without_validation,
)
from .official_score_matrix import ScoreRow, file_sha256, official_half_up
from .ordinal_tail_fixed_feature import CandidateSpec
from .r0_ordinal_residual import load_embedding_artifact
from .stage3_coral_promotion import promotion_gate

SCHEMA_VERSION = "mal2026-kure-crt-recovery-v1"
METHOD = "coral-natural"
FAMILY = "coral"
CONFIG_FILE_PATH = Path(__file__).resolve().parents[2] / "configs/kure_crt_recovery.v1.json"


class KURECRTRecoveryError(RuntimeError):
    """Raised when the recovery's fixed, train-only protocol is violated."""


@dataclass(frozen=True)
class _TextRow:
    """Held-fold inference input that cannot expose a score label."""

    identifier: str
    document_id: str
    prompt_num: str
    prompt: str
    essay: str


def need(condition: bool, message: str) -> None:
    if not condition:
        raise KURECRTRecoveryError(message)


def _contains_validation(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any("validation" in str(key).lower() or _contains_validation(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_validation(child) for child in value)
    return isinstance(value, str) and "validation" in value.lower()


def _config_sha256(config: "KURECRTRecoveryConfig") -> str:
    return sha256(json.dumps(asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _config_file_sha256() -> str:
    need(CONFIG_FILE_PATH.is_file(), "canonical recovery config file is unavailable")
    return file_sha256(CONFIG_FILE_PATH)


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()


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


@dataclass(frozen=True)
class KURECRTRecoveryConfig:
    schema_version: str
    run_id: str
    source_stage3_config_path: str
    source_stage3_config_sha256: str
    train_path: str
    train_sha256: str
    fold_manifest_path: str
    fold_manifest_sha256: str
    fold_rows_path: str
    fold_rows_sha256: str
    r0_oof_prediction_path: str
    r0_oof_prediction_sha256: str
    backbone: BackboneSpec
    output_root: str
    restricted_output_root: str
    seed: int
    phase1_epochs: int
    phase1_learning_rate: float
    phase1_weight_decay: float
    batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    raw_rmse_auxiliary_weight: float
    crt_learning_rate: float
    crt_weight_decay: float
    crt_epochs: int
    crt_updates: int
    prior_sanity_steps: int
    prior_sanity_learning_rate: float
    prior_sanity_weight_decay: float
    axes: tuple[str, ...]
    average_target_forbidden: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "KURECRTRecoveryConfig":
        need(isinstance(raw, Mapping) and not _contains_validation(raw), "validation fields and paths are forbidden")
        value = dict(raw)
        value["axes"] = tuple(value.get("axes", ()))
        need(isinstance(value.get("backbone"), Mapping), "backbone provenance block is required")
        need(set(value["backbone"]) == set(BackboneSpec.__dataclass_fields__), "backbone provenance fields differ")
        value["backbone"] = BackboneSpec(**value["backbone"])
        need(set(value) == set(cls.__dataclass_fields__), "recovery config fields differ")
        result = cls(**value)
        result.validate(require_dependencies=False)
        return result

    @classmethod
    def from_json(cls, path: str | Path, *, require_dependencies: bool = False) -> "KURECRTRecoveryConfig":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KURECRTRecoveryError("recovery config is unreadable") from exc
        result = cls.from_mapping(raw)
        result.validate(require_dependencies=require_dependencies)
        return result

    def validate(self, *, require_dependencies: bool) -> None:
        need(self.schema_version == SCHEMA_VERSION and self.run_id == "kure-ordinal-crt-recovery-v1-20260803-001", "schema/run identity differs")
        need(self.axes == AXES and self.average_target_forbidden is True, "independent-axis/average contract differs")
        need(self.backbone.arm == "aihub_full_backbone" and self.backbone.model_config_sha256 == MODEL_CONFIG_SHA256,
             "KURE/AI-Hub backbone contract differs")
        need((self.backbone.lora_r, self.backbone.lora_alpha, self.backbone.lora_dropout) == (16, 32, 0.05), "LoRA contract differs")
        need(Path(self.output_root).resolve() != Path(self.restricted_output_root).resolve(), "public/restricted roots must differ")
        need((self.phase1_epochs, self.phase1_learning_rate, self.phase1_weight_decay) == (6, 5e-5, .01), "phase1 replay contract differs")
        need((self.batch_size, self.gradient_accumulation_steps, self.max_length) == (20, 2, 1536), "batch/token contract differs")
        need(self.raw_rmse_auxiliary_weight == .25, "raw MSE contract differs")
        need((self.crt_learning_rate, self.crt_weight_decay, self.crt_epochs, self.crt_updates) == (5e-3, .01, 20, 800), "cRT recovery contract differs")
        need((self.prior_sanity_steps, self.prior_sanity_learning_rate, self.prior_sanity_weight_decay) == (160, .05, .01),
             "prior sanity optimizer contract differs")
        for digest in (self.source_stage3_config_sha256, self.train_sha256, self.fold_manifest_sha256,
                       self.fold_rows_sha256, self.r0_oof_prediction_sha256):
            need(isinstance(digest, str) and len(digest) == 64, "checksum format differs")
        if require_dependencies:
            source_path = Path(self.source_stage3_config_path)
            need(source_path.is_file() and file_sha256(source_path) == self.source_stage3_config_sha256, "source Stage3 config binding differs")
            for path, digest, label in ((self.train_path, self.train_sha256, "train"),
                                        (self.fold_manifest_path, self.fold_manifest_sha256, "fold manifest"),
                                        (self.fold_rows_path, self.fold_rows_sha256, "fold rows"),
                                        (self.r0_oof_prediction_path, self.r0_oof_prediction_sha256, "exact R0 OOF")):
                need(Path(path).is_file() and file_sha256(Path(path)) == digest, f"{label} binding differs")


def _source_stage3_config(config: KURECRTRecoveryConfig) -> KUREOrdinalOOFConfig:
    source = KUREOrdinalOOFConfig.from_json(config.source_stage3_config_path, require_dependencies=True)
    need((source.train_path, source.train_sha256, source.fold_manifest_path, source.fold_manifest_sha256,
          source.fold_rows_path, source.fold_rows_sha256, source.r0_oof_prediction_path, source.r0_oof_prediction_sha256,
          source.backbone, source.seed, source.epochs, source.learning_rate, source.weight_decay,
          source.batch_size, source.gradient_accumulation_steps, source.max_length, source.raw_rmse_auxiliary_weight)
         == (config.train_path, config.train_sha256, config.fold_manifest_path, config.fold_manifest_sha256,
             config.fold_rows_path, config.fold_rows_sha256, config.r0_oof_prediction_path, config.r0_oof_prediction_sha256,
             config.backbone, config.seed, config.phase1_epochs, config.phase1_learning_rate, config.phase1_weight_decay,
             config.batch_size, config.gradient_accumulation_steps, config.max_length, config.raw_rmse_auxiliary_weight),
         "source Stage3 replay binding differs")
    return source


def _assert_private_file(path: Path) -> None:
    mode = path.stat().st_mode
    need(mode & 0o007 == 0 and mode & 0o600 == 0o600, "restricted file ACL/mode is not project-private")


def _secure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o770)
    need(path.stat().st_mode & 0o007 == 0 and path.stat().st_mode & 0o700 == 0o700, "restricted directory ACL/mode differs")


def _atomic_private_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    need(not path.exists(), "refusing to overwrite restricted recovery predictions")
    _secure_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary, 0o660); os.replace(temporary, path); os.chmod(path, 0o660); _assert_private_file(path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return file_sha256(path)


def _validate_public(value: Any) -> None:
    forbidden = {"source_id", "document_id", "essay", "prompt", "embedding", "raw_gold", "prediction", "predictions"}
    if isinstance(value, Mapping):
        need(not (set(value) & forbidden), "restricted row content cannot enter public output")
        for child in value.values(): _validate_public(child)
    elif isinstance(value, (list, tuple)):
        for child in value: _validate_public(child)


def _atomic_public_json(path: Path, value: Mapping[str, Any]) -> str:
    need(not path.exists(), "refusing to overwrite public recovery output")
    _validate_public(value); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True); raise
    return file_sha256(path)


def _fit_raw_gold(path: Path, expected_sha256: str, fit_ids: set[str]) -> Mapping[str, tuple[float, float, float]]:
    """Read score fields only for fit IDs; held score fields remain untouched until reporting."""
    need(file_sha256(path) == expected_sha256, "train checksum differs")
    result: dict[str, tuple[float, float, float]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            identifier = str(raw["id"])
            if identifier not in fit_ids:
                continue
            score = raw["score"]
            need(set(score) == {*AXES, "average"}, "train score schema differs")
            values = tuple(float(score[axis]) for axis in AXES)
            need(all(math.isfinite(item) and 1 <= item <= 5 for item in values), "fit raw score differs")
            result[identifier] = values  # type: ignore[assignment]
    need(set(result) == fit_ids, "fit raw-gold coverage differs")
    return result


def _load_fit_and_held_text(source: KUREOrdinalOOFConfig, outer_fold: int) -> tuple[list[ScoreRow], list[_TextRow]]:
    """Load fit labels while never indexing the held-fold score object."""
    need(file_sha256(Path(source.train_path)) == source.train_sha256, "train checksum differs")
    need(file_sha256(Path(source.fold_manifest_path)) == source.fold_manifest_sha256, "fold manifest binding differs")
    need(file_sha256(Path(source.fold_rows_path)) == source.fold_rows_sha256, "fold rows binding differs")
    manifest, fold_rows = load_embedding_artifact(source.fold_manifest_path, source.fold_rows_path)
    need(manifest.split_role == "train" and manifest.base_prediction_origin == "oof" and not manifest.contains_average_target,
         "fold artifact contract differs")
    folds = {row.source_id: row.oof_fold for row in fold_rows}
    need(len(folds) == 2000 and set(folds.values()) == set(range(5)), "fold population differs")
    fit: list[ScoreRow] = []
    held: list[_TextRow] = []
    seen: set[str] = set()
    with Path(source.train_path).open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            need(set(raw) == {"id", "document_id", "prompt_num", "prompt", "essay", "score"}, "canonical row schema differs")
            identifier = raw["id"]
            need(isinstance(identifier, str) and identifier in folds and identifier not in seen, "canonical row ID/fold differs")
            seen.add(identifier)
            text = (raw["document_id"], raw["prompt_num"], raw["prompt"], raw["essay"])
            need(all(isinstance(item, (str, int)) for item in text[:2]) and all(isinstance(item, str) and item.strip() for item in text[2:]),
                 "canonical text/group fields differ")
            common = (identifier, str(text[0]), str(text[1]), text[2], text[3])
            if folds[identifier] == outer_fold:
                # Deliberately do not access raw["score"] for a held row.
                held.append(_TextRow(*common))
                continue
            scores = raw["score"]
            need(isinstance(scores, Mapping) and set(scores) == {*AXES, "average"}, "fit score schema differs")
            labels = []
            for axis in AXES:
                value = scores[axis]
                need(type(value) in {int, float} and not isinstance(value, bool) and math.isfinite(float(value)) and 1 <= float(value) <= 5,
                     "fit score differs")
                labels.append(official_half_up(float(value)))
            # scores["average"] is never indexed.
            fit.append(ScoreRow(*common, tuple(labels)))  # type: ignore[arg-type]
    need(seen == set(folds) and (len(fit), len(held)) == (1600, 400), "outer fit/held split differs")
    return fit, held


def _labels_for_axis(rows: Sequence[ScoreRow], axis: str) -> list[int]:
    return [int(row.labels[AXES.index(axis)]) for row in rows]


def initialize_crt_head(head: Any, labels: Sequence[int]) -> tuple[np.ndarray, float]:
    """Zero weights and initialize bias to fit-fold ordinal empirical log-prior."""
    import torch
    values = torch.as_tensor(labels, dtype=torch.long)
    need(values.ndim == 1 and len(values) > 0 and bool(torch.all((1 <= values) & (values <= 5))), "cRT labels lack five-class support")
    counts = torch.bincount(values, minlength=6)[1:].double().cpu().numpy()
    need(bool(np.all(counts > 0)), "cRT fit lacks five-class support")
    q = counts / counts.sum()
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.as_tensor(np.log(q), dtype=head.bias.dtype, device=head.bias.device))
    return q, float(np.dot(q, np.arange(1, 6)))


def prior_sanity_gate(labels: Sequence[int], *, steps: int, learning_rate: float, weight_decay: float) -> Mapping[str, Any]:
    """Fit-only bias-only optimizer check; never receives held labels or features."""
    import torch
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_tensor, minlength=6)[1:].double(); q = counts / counts.sum()
    need(bool(torch.all(counts > 0)), "cRT fit lacks five-class support")
    bias = torch.nn.Parameter(torch.zeros(5, dtype=torch.float32))
    optimizer = torch.optim.AdamW([bias], lr=learning_rate, weight_decay=weight_decay)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(bias.expand(len(labels_tensor), -1), labels_tensor - 1)
        loss.backward(); optimizer.step()
    pmf = torch.softmax(bias.detach(), 0).double()
    expected = float(torch.dot(pmf, torch.arange(1, 6, dtype=torch.float64)))
    label_mean = float(labels_tensor.double().mean())
    entropy = float(-(q * q.log()).sum())
    ce = float(torch.nn.functional.cross_entropy(bias.detach().expand(len(labels_tensor), -1), labels_tensor - 1))
    max_error = float(torch.max(torch.abs(pmf - q)))
    need(max_error <= .02 and abs(expected - label_mean) <= .02 and ce <= entropy + .005,
         "fit-only bias-prior sanity gate failed")
    return {"pmf_max_abs_error": max_error, "expected_ordinal_label_mean_error": abs(expected - label_mean),
            "cross_entropy": ce, "empirical_entropy": entropy, "steps": steps,
            "optimizer": "AdamW", "learning_rate": learning_rate, "weight_decay": weight_decay,
            "role": "fit_only_bias_optimizer_integration_sanity_not_crt_hyperparameter"}


def expected_score(logits: Any) -> Any:
    import torch
    return (torch.softmax(logits, 1) * torch.arange(1, 6, device=logits.device, dtype=logits.dtype)).sum(1)


def cRT_update_budget(fit_records: int, batch_size: int, gradient_accumulation_steps: int, epochs: int) -> int:
    need(fit_records % batch_size == 0 and (fit_records // batch_size) % gradient_accumulation_steps == 0, "exact cRT update budget requires divisible fit batches")
    return fit_records // batch_size // gradient_accumulation_steps * epochs


def train_cached_crt_head(features: Any, labels: Sequence[int], raw_labels: Sequence[float], config: KURECRTRecoveryConfig,
                            *, seed: int, device: Any, max_updates: int | None = None) -> tuple[Any, Mapping[str, Any]]:
    """Head-only AdamW, constant LR/no warmup, exactly 800 full-run updates."""
    import torch
    need(features.ndim == 2 and features.shape == (len(labels), 1024) and len(labels) == len(raw_labels), "cached fit feature shape differs")
    target_updates = config.crt_updates if max_updates is None else max_updates
    need(len(labels) % (config.batch_size * config.gradient_accumulation_steps) == 0,
         "cRT records must fill every accumulated optimizer batch")
    if max_updates is None:
        need(cRT_update_budget(len(labels), config.batch_size, config.gradient_accumulation_steps, config.crt_epochs) == config.crt_updates,
             "configured cRT update budget differs")
    seed_runtime(seed)
    head = torch.nn.Linear(1024, 5).to(device)
    q, q_mean = initialize_crt_head(head, labels)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.crt_learning_rate, weight_decay=config.crt_weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cpu_features = features.detach().cpu().float(); label_tensor = torch.as_tensor(labels, dtype=torch.long); raw_tensor = torch.as_tensor(raw_labels, dtype=torch.float32)
    updates = 0; completed_epochs = 0
    updates_per_epoch = len(labels) // (config.batch_size * config.gradient_accumulation_steps)
    need(0 < target_updates <= updates_per_epoch * config.crt_epochs, "cRT target update budget exceeds configured epochs")
    for epoch in range(config.crt_epochs):
        # One deterministic permutation per epoch; two adjacent microbatches make
        # each optimizer update, exactly matching the Stage3 accumulation count.
        order = torch.randperm(len(labels), generator=generator)
        for start in range(0, len(labels), config.batch_size * config.gradient_accumulation_steps):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(config.gradient_accumulation_steps):
                indices = order[start + micro * config.batch_size:start + (micro + 1) * config.batch_size]
                batch_features = cpu_features[indices].to(device, non_blocking=True)
                batch_labels = label_tensor[indices].to(device, non_blocking=True)
                batch_raw = raw_tensor[indices].to(device, non_blocking=True)
                logits = head(batch_features)
                loss = torch.nn.functional.cross_entropy(logits, batch_labels - 1) + config.raw_rmse_auxiliary_weight * torch.nn.functional.mse_loss(expected_score(logits), batch_raw)
                need(bool(torch.isfinite(loss)), "cRT loss became non-finite")
                (loss / config.gradient_accumulation_steps).backward()
            optimizer.step(); updates += 1
            need(all(bool(torch.isfinite(parameter).all()) for parameter in head.parameters()), "cRT head became non-finite")
            if updates == target_updates: break
        completed_epochs = epoch + 1
        if updates == target_updates: break
    need(updates == target_updates, "cRT optimizer update budget differs")
    return head, {"updates": updates, "configured_epochs": config.crt_epochs, "completed_epochs": completed_epochs,
                  "updates_per_complete_epoch": updates_per_epoch, "learning_rate": config.crt_learning_rate,
                  "weight_decay": config.crt_weight_decay, "scheduler": "constant_no_warmup", "head_weight_init": "zero",
                  "head_bias_init": "fit_ordinal_empirical_log_prior", "fit_ordinal_empirical_prior": q.tolist(),
                  "fit_ordinal_empirical_expected_score": q_mean}


def cache_cls_l2_features(model: Any, rows: Sequence[Any], tokenizer: Any, *, max_length: int, device: Any) -> Any:
    """Cache only the provided rows. Full runs call this with fit rows before cRT."""
    import torch
    model.eval(); cached = []
    with torch.no_grad():
        for start in range(0, len(rows), 20):
            batch = rows[start:start + 20]
            encoded = tokenizer([render_input(row) for row in batch], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            hidden = model.backbone(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device), return_dict=True).last_hidden_state[:, 0].float()
            cached.append(torch.nn.functional.normalize(hidden, p=2, dim=-1).cpu())
    result = torch.cat(cached, 0)
    need(result.shape == (len(rows), 1024) and bool(torch.isfinite(result).all()), "cached CLS-L2 feature contract differs")
    return result


def _phase1_and_fit_cache(source: KUREOrdinalOOFConfig, recovery: KURECRTRecoveryConfig, fit: Sequence[ScoreRow], axis: str,
                           fit_raw: Mapping[str, tuple[float, float, float]], restricted: Path, *, smoke: bool) -> tuple[Any, Any, Mapping[str, Any], int, int, Any]:
    import torch
    from transformers import AutoTokenizer
    phase1_seed = derived_seed(recovery.seed, 0 if smoke else int(restricted.name.removeprefix("outer-")), METHOD, axis, "phase1")
    crt_seed = derived_seed(recovery.seed, 0 if smoke else int(restricted.name.removeprefix("outer-")), METHOD, axis, "crt_recovery")
    seed_runtime(phase1_seed); validate_backbone_without_validation(recovery.backbone)
    tokenizer = AutoTokenizer.from_pretrained(recovery.backbone.model_path, revision=recovery.backbone.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    token_audit = token_length_audit(fit, tokenizer, recovery.max_length)
    model, lineage = _build_axis_model(recovery.backbone, CandidateSpec(METHOD, FAMILY, "natural"))
    dataset = _RowsDataset(fit, axis, tokenizer, recovery.max_length, fit_raw)
    trainer = _train_phase(model, dataset, tokenizer, CandidateSpec(METHOD, FAMILY, "natural"), source,
                           restricted / axis / "phase1", seed=phase1_seed, max_steps=2 if smoke else -1)
    del trainer
    for parameter in model.parameters(): parameter.requires_grad = False
    features = cache_cls_l2_features(model, fit, tokenizer, max_length=recovery.max_length, device=torch.device("cuda"))
    return model, tokenizer, {"lineage": lineage, "fit_token_length_audit": token_audit}, phase1_seed, crt_seed, features


def run(config: KURECRTRecoveryConfig | str | Path, *, outer_fold: int, validate_only: bool = False, smoke: bool = False) -> Mapping[str, Any]:
    recovery = KURECRTRecoveryConfig.from_json(config, require_dependencies=not validate_only) if isinstance(config, (str, Path)) else config
    recovery.validate(require_dependencies=not validate_only); need(0 <= outer_fold < 5, "outer fold must be 0..4")
    need(not smoke or outer_fold == 0, "smoke is restricted to outer fold 0")
    if validate_only:
        return {"status": "validated", "method": METHOD, "gpu_used": False, "config_sha256": _config_sha256(recovery),
                "config_file_sha256": _config_file_sha256(), "validation_rows_loaded": False, "average_target_used": False}
    import torch
    need(torch.cuda.is_available(), "recovery training requires an explicitly launched GPU job")
    if smoke: need(os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and torch.cuda.device_count() == 1, "smoke requires only physical GPU0 exposed")
    source = _source_stage3_config(recovery)
    fit, held = _load_fit_and_held_text(source, outer_fold)
    if smoke:
        rng = random.Random(derived_seed(recovery.seed, 0, METHOD, "content", "smoke_fit")); buckets = {score: [row for row in fit if row.labels[0] == score] for score in range(1, 6)}
        for bucket in buckets.values(): rng.shuffle(bucket)
        need(all(len(bucket) >= 8 for bucket in buckets.values()), "smoke fit lacks eight examples per class")
        fit = [row for score in range(1, 6) for row in buckets[score][:8]]
    fit_ids = {row.identifier for row in fit}; fit_raw = _fit_raw_gold(Path(recovery.train_path), recovery.train_sha256, fit_ids)
    restricted = Path(recovery.restricted_output_root) / ("smoke/outer-00" if smoke else f"outer-{outer_fold:02d}")
    _secure_directory(restricted)
    run_axes = ("content",) if smoke else AXES
    axis_predictions = []; bindings = []
    for axis in run_axes:
        model, tokenizer, phase, phase1_seed, crt_seed, features = _phase1_and_fit_cache(source, recovery, fit, axis, fit_raw, restricted, smoke=smoke)
        labels = _labels_for_axis(fit, axis); raw_labels = [fit_raw[row.identifier][AXES.index(axis)] for row in fit]
        sanity = prior_sanity_gate(labels, steps=recovery.prior_sanity_steps,
                                   learning_rate=recovery.prior_sanity_learning_rate,
                                   weight_decay=recovery.prior_sanity_weight_decay)
        head, crt = train_cached_crt_head(features, labels, raw_labels, recovery, seed=crt_seed, device=torch.device("cuda"), max_updates=2 if smoke else None)
        bindings.append({"axis": axis, "phase1_seed": phase1_seed, "crt_seed": crt_seed, **phase, "fit_records": len(fit), "prior_sanity": sanity, "crt": crt})
        if not smoke:
            # Held rows enter only after cRT fit and its fit-only sanity gate have passed.
            held_features = cache_cls_l2_features(model, held, tokenizer, max_length=recovery.max_length, device=torch.device("cuda"))
            with torch.no_grad(): axis_predictions.append(expected_score(head(held_features.to(torch.device("cuda")))).cpu().numpy())
        del head, model, tokenizer; torch.cuda.empty_cache()
    if smoke:
        result = {"schema_version": SCHEMA_VERSION, "status": "completed", "mode": "smoke", "run_id": recovery.run_id, "outer_fold": 0,
                  "method": METHOD, "axes": list(run_axes), "axis_bindings": bindings, "config_sha256": _config_sha256(recovery),
                  "config_file_sha256": _config_file_sha256(), "git_sha": _git_sha(),
                  "validation_rows_loaded": False, "average_target_used": False, "nonselectable": True,
                  "smoke_not_reusable_for_selection_or_scientific_results": True,
                  "logical_command": f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} PYTHONPATH=src {sys.executable} scripts/run_kure_crt_recovery.py --config configs/kure_crt_recovery.v1.json --outer-fold 0 --smoke",
                  "environment": _environment()}
        _atomic_public_json(Path(recovery.output_root) / "smoke/outer-00.json", result); return result
    predictions = np.column_stack(axis_predictions)
    # Held rows first enter at final inference and the restricted atomic OOF write.
    private_sha = _atomic_private_jsonl(restricted / METHOD / "predictions.jsonl", (
        {"source_id": row.identifier, "outer_fold": outer_fold, "prediction": {axis: float(predictions[i, j]) for j, axis in enumerate(AXES)}}
        for i, row in enumerate(held)))
    # Held labels are read only after the OOF artifact is durably written, for
    # aggregate-only fold metrics; they never enter cRT fitting or sanity.
    held_truth = _fit_raw_gold(Path(recovery.train_path), recovery.train_sha256, {row.identifier for row in held})
    metrics = compute_iterative_tail_metrics([held_truth[row.identifier] for row in held], predictions)
    result = {"schema_version": SCHEMA_VERSION, "status": "completed", "mode": "outer_fold", "run_id": recovery.run_id, "outer_fold": outer_fold,
              "records": 400, "method": METHOD, "family": FAMILY, "axis_bindings": bindings, "metrics": metrics,
              "restricted_prediction_sha256": private_sha, "config_sha256": _config_sha256(recovery),
              "config_file_sha256": _config_file_sha256(), "git_sha": _git_sha(),
              "fold_manifest_sha256": recovery.fold_manifest_sha256, "fold_rows_sha256": recovery.fold_rows_sha256,
              "r0_oof_prediction_sha256": recovery.r0_oof_prediction_sha256, "validation_rows_loaded": False, "average_target_used": False,
              "negative_stage3_preserved": True, "privacy": "aggregate_only_public_predictions_restricted",
              "logical_command": f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} PYTHONPATH=src {sys.executable} scripts/run_kure_crt_recovery.py --config configs/kure_crt_recovery.v1.json --outer-fold {outer_fold}",
              "environment": _environment()}
    _atomic_public_json(Path(recovery.output_root) / f"outer-{outer_fold:02d}.json", result)
    return result


def _load_private_fold(path: Path, fold: int, expected_ids: set[str]) -> Mapping[str, Sequence[float]]:
    result: dict[str, Sequence[float]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line); identifier = item.get("source_id"); prediction = item.get("prediction")
            need(set(item) == {"source_id", "outer_fold", "prediction"} and isinstance(identifier, str) and identifier in expected_ids and identifier not in result and item["outer_fold"] == fold,
                 "restricted recovery row/fold differs")
            need(isinstance(prediction, Mapping) and set(prediction) == set(AXES) and all(math.isfinite(float(prediction[axis])) and 1 <= float(prediction[axis]) <= 5 for axis in AXES), "restricted recovery prediction differs")
            result[identifier] = tuple(float(prediction[axis]) for axis in AXES)
    need(set(result) == expected_ids, "restricted recovery fold coverage differs")
    return result


def aggregate(config: KURECRTRecoveryConfig | str | Path) -> Mapping[str, Any]:
    recovery = KURECRTRecoveryConfig.from_json(config, require_dependencies=True) if isinstance(config, (str, Path)) else config
    recovery.validate(require_dependencies=True); source = _source_stage3_config(recovery)
    rows, folds = load_train_and_folds(source); ids = [row.identifier for row in rows]
    truth = _fit_raw_gold(Path(recovery.train_path), recovery.train_sha256, set(ids)); r0 = load_exact_r0(source)
    predictions: dict[str, Sequence[float]] = {}; bindings = []
    for fold in range(5):
        public_path = Path(recovery.output_root) / f"outer-{fold:02d}.json"; private_path = Path(recovery.restricted_output_root) / f"outer-{fold:02d}" / METHOD / "predictions.jsonl"
        need(public_path.is_file() and private_path.is_file(), "recovery outer result is incomplete")
        public = json.loads(public_path.read_text(encoding="utf-8"))
        need(public.get("schema_version") == SCHEMA_VERSION and public.get("status") == "completed" and public.get("mode") == "outer_fold"
             and public.get("run_id") == recovery.run_id and public.get("outer_fold") == fold and public.get("method") == METHOD
             and public.get("config_sha256") == _config_sha256(recovery) and public.get("config_file_sha256") == _config_file_sha256()
             and public.get("validation_rows_loaded") is False and public.get("average_target_used") is False
             and public.get("restricted_prediction_sha256") == file_sha256(private_path), "recovery outer report binding differs")
        expected = {identifier for identifier, assigned in folds.items() if assigned == fold}
        predictions.update(_load_private_fold(private_path, fold, expected))
        need(isinstance(public.get("logical_command"), str) and isinstance(public.get("environment"), Mapping),
             "recovery execution evidence differs")
        bindings.append({"outer_fold": fold, "public_sha256": file_sha256(public_path),
                         "restricted_prediction_sha256": file_sha256(private_path),
                         "logical_command": public["logical_command"], "environment": public["environment"]})
    need(set(predictions) == set(ids), "recovery OOF coverage differs")
    metrics = compute_iterative_tail_metrics([truth[key] for key in ids], [predictions[key] for key in ids])
    r0_metrics = compute_iterative_tail_metrics([truth[key] for key in ids], [r0[key] for key in ids])
    decision = promotion_gate(np.asarray([truth[key] for key in ids], dtype=float),
                              np.asarray([r0[key] for key in ids], dtype=float),
                              np.asarray([predictions[key] for key in ids], dtype=float),
                              ids, seed=recovery.seed)
    result = {"schema_version": "mal2026-kure-crt-recovery-aggregate-v1", "status": "completed", "mode": "full_oof", "run_id": recovery.run_id,
              "records": 2000, "folds": 5, "method": METHOD, "family": FAMILY, "metrics": metrics,
              "exact_r0_metrics": r0_metrics, "improvements_vs_exact_r0": metric_improvements(r0_metrics, metrics), "fold_bindings": bindings,
              "protected_output": "exact_r0", "protection_disclosure": "recovery is outside the frozen Stage6 trust chain; exact R0 remains protected even if the reused common criteria pass",
              "stage6_common_gate_required": True, "common_stage3_promotion_gate": decision,
              "common_stage3_promotion_gate_passed": decision["eligible"],
              "automatic_stage6_deployment_eligible": False,
              "automatic_stage6_deployment_disclosure": "a pass requires explicit scientific authorization and a new hash-bound decision",
              "negative_stage3_preserved": True, "config_sha256": _config_sha256(recovery),
              "config_file_sha256": _config_file_sha256(), "git_sha": _git_sha(),
              "fold_manifest_sha256": recovery.fold_manifest_sha256, "fold_rows_sha256": recovery.fold_rows_sha256,
              "r0_oof_prediction_sha256": recovery.r0_oof_prediction_sha256, "validation_rows_loaded": False, "average_target_used": False,
              "privacy": "aggregate_only_no_rows_text_ids_embeddings_or_predictions",
              "logical_command": f"PYTHONPATH=src {sys.executable} scripts/run_kure_crt_recovery.py --config configs/kure_crt_recovery.v1.json --aggregate",
              "environment": _environment()}
    _atomic_public_json(Path(recovery.output_root) / "aggregate.json", result)
    return result


__all__ = ["KURECRTRecoveryConfig", "KURECRTRecoveryError", "aggregate", "cRT_update_budget", "expected_score", "initialize_crt_head", "prior_sanity_gate", "run", "train_cached_crt_head"]
