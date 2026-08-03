"""Hash-bound direct evaluation of the saved Stage3 CORAL phase-1 heads.

This is inference only.  Each outer-fold checkpoint restores its LoRA tensors,
``score.*`` and ``cut_*`` tensors; the later cRT ``head.*`` tensors are checked
and deliberately ignored.  Held labels are not read until the corresponding
restricted predictions have been atomically persisted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .iterative_tail_metrics import AXES, compute_iterative_tail_metrics, metric_improvements
from .kure_axis_contrastive import render_input, token_length_audit
from .kure_ordinal_oof import (
    KUREOrdinalOOFConfig, _build_axis_model, load_exact_r0, validate_backbone_without_validation,
)
from .official_score_matrix import file_sha256, official_half_up
from .ordinal_tail_fixed_feature import CandidateSpec, coral_pmf
from .stage3_coral_promotion import promotion_gate


SCHEMA_VERSION = "mal2026-kure-phase1-direct-oof-v1"
METHOD = "coral-natural-phase1-direct"
SOURCE_METHOD = "coral-natural"
CONFIG_FILE_PATH = Path(__file__).resolve().parents[2] / "configs/kure_phase1_direct_oof.v1.json"


class KUREPhase1DirectOOFError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise KUREPhase1DirectOOFError(message)


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any("validation" in str(key).lower() or _contains_forbidden(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden(child) for child in value)
    return isinstance(value, str) and "validation" in value.lower()


@dataclass(frozen=True)
class CheckpointBinding:
    outer_fold: int
    axis: str
    path: str
    sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CheckpointBinding":
        need(isinstance(raw, Mapping) and set(raw) == set(cls.__dataclass_fields__), "checkpoint binding fields differ")
        value = cls(int(raw["outer_fold"]), str(raw["axis"]), str(raw["path"]), str(raw["sha256"]))
        need(value.outer_fold in range(5) and value.axis in AXES and len(value.sha256) == 64,
             "checkpoint binding identity differs")
        return value


@dataclass(frozen=True)
class FoldMembershipBinding:
    outer_fold: int
    path: str
    sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FoldMembershipBinding":
        need(isinstance(raw, Mapping) and set(raw) == set(cls.__dataclass_fields__), "membership binding fields differ")
        value = cls(int(raw["outer_fold"]), str(raw["path"]), str(raw["sha256"]))
        need(value.outer_fold in range(5) and len(value.sha256) == 64, "membership binding identity differs")
        return value


@dataclass(frozen=True)
class KUREPhase1DirectOOFConfig:
    schema_version: str
    status: str
    execution_authorized: bool
    task_card_path: str
    task_card_sha256: str
    task_card_commit: str
    preparer_path: str
    preparer_sha256: str
    preparer_commit: str
    preparation_request_config_sha256: str
    run_id: str
    source_stage3_config_path: str
    source_stage3_config_sha256: str
    source_stage3_report_config_sha256: str
    source_stage3_aggregate_path: str
    source_stage3_aggregate_sha256: str
    train_path: str
    train_sha256: str
    fold_manifest_path: str
    fold_manifest_sha256: str
    fold_rows_path: str
    fold_rows_sha256: str
    r0_oof_prediction_path: str
    r0_oof_prediction_sha256: str
    fold_membership_bindings: tuple[FoldMembershipBinding, ...]
    label_free_projection_path: str
    label_free_projection_sha256: str
    label_free_manifest_path: str
    label_free_manifest_sha256: str
    checkpoint_bindings: tuple[CheckpointBinding, ...]
    output_root: str
    restricted_output_root: str
    seed: int
    batch_size: int
    max_length: int
    smoke_gpu: int
    full_gpu_scope: tuple[int, ...]
    fold_gpu_mapping: Mapping[str, int]
    telemetry_columns: tuple[str, ...]
    smoke_minimum_samples: int
    full_minimum_samples: int
    telemetry_interval_seconds: int
    axes: tuple[str, ...]
    average_target_forbidden: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "KUREPhase1DirectOOFConfig":
        need(isinstance(raw, Mapping) and not _contains_forbidden(raw), "validation fields and paths are forbidden")
        value = dict(raw)
        value["axes"] = tuple(value.get("axes", ()))
        value["full_gpu_scope"] = tuple(value.get("full_gpu_scope", ()))
        value["telemetry_columns"] = tuple(value.get("telemetry_columns", ()))
        need(isinstance(value.get("checkpoint_bindings"), list), "checkpoint_bindings must be a list")
        value["checkpoint_bindings"] = tuple(CheckpointBinding.from_mapping(item) for item in value["checkpoint_bindings"])
        need(isinstance(value.get("fold_membership_bindings"), list), "fold_membership_bindings must be a list")
        value["fold_membership_bindings"] = tuple(FoldMembershipBinding.from_mapping(item) for item in value["fold_membership_bindings"])
        need(set(value) == set(cls.__dataclass_fields__), "phase1-direct config fields differ")
        result = cls(**value)
        result.validate(require_dependencies=False)
        return result

    @classmethod
    def from_json(cls, path: str | Path, *, require_dependencies: bool = False) -> "KUREPhase1DirectOOFConfig":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KUREPhase1DirectOOFError("phase1-direct config is unreadable") from exc
        result = cls.from_mapping(raw)
        result.validate(require_dependencies=require_dependencies)
        return result

    def validate(self, *, require_dependencies: bool) -> None:
        need(self.schema_version == SCHEMA_VERSION and self.run_id == "kure-phase1-direct-oof-v1-20260803-001",
             "schema/run identity differs")
        need(self.status in {"pending_scientific_authorization", "authorized"}, "authorization status differs")
        need(self.execution_authorized is (self.status == "authorized"), "authorization fields disagree")
        need(bool(self.task_card_path) and bool(self.preparer_path), "task-card/preparer path differs")
        if self.status == "authorized":
            need(len(self.task_card_sha256) == len(self.preparer_sha256) == 64, "authorized code/card checksum differs")
            need(len(self.preparation_request_config_sha256) == 64, "preparation-request config binding differs")
            need(len(self.task_card_commit) == len(self.preparer_commit) == 40, "authorized code/card commit differs")
            need(len(self.label_free_projection_sha256) == len(self.label_free_manifest_sha256) == 64,
                 "authorized label-free input binding differs")
        else:
            need(all(value == "" for value in (self.task_card_sha256, self.task_card_commit, self.preparer_sha256,
                                                self.preparer_commit, self.preparation_request_config_sha256,
                                                self.label_free_projection_sha256,
                                                self.label_free_manifest_sha256)),
                 "pending config must retain unfilled post-commit/projection placeholders")
        need(self.axes == AXES and self.average_target_forbidden is True, "independent-axis/average contract differs")
        need((self.batch_size, self.max_length) == (20, 1536), "fixed inference batch/token contract differs")
        need(self.smoke_gpu == 0 and self.full_gpu_scope == (0, 1, 2, 3)
             and self.fold_gpu_mapping == {"0": 0, "1": 1, "2": 2, "3": 3, "4": 0},
             "fixed GPU/fold mapping differs")
        need(self.telemetry_columns == ("timestamp", "index", "uuid", "name", "memory.total", "driver_version",
                                        "utilization.gpu", "memory.used")
             and (self.smoke_minimum_samples, self.full_minimum_samples, self.telemetry_interval_seconds) == (1, 2, 30),
             "fixed telemetry runtime contract differs")
        need(Path(self.output_root).resolve() != Path(self.restricted_output_root).resolve(), "public/restricted roots must differ")
        expected_inventory = tuple((fold, axis) for fold in range(5) for axis in AXES)
        actual_inventory = tuple((item.outer_fold, item.axis) for item in self.checkpoint_bindings)
        need(actual_inventory == expected_inventory, "exact ordered 5-fold x 3-axis checkpoint inventory differs")
        need(tuple(item.outer_fold for item in self.fold_membership_bindings) == tuple(range(5)),
             "exact ordered five membership bindings differ")
        need(bool(self.label_free_projection_path) and bool(self.label_free_manifest_path), "label-free input paths differ")
        need(Path(self.label_free_projection_path) != Path(self.label_free_manifest_path)
             and Path("data/processed/restricted").resolve() in Path(self.label_free_projection_path).resolve().parents
             and Path("data/processed/restricted").resolve() in Path(self.label_free_manifest_path).resolve().parents,
             "label-free inputs must be distinct restricted artifacts")
        for digest in (self.source_stage3_config_sha256, self.source_stage3_report_config_sha256,
                       self.source_stage3_aggregate_sha256, self.train_sha256, self.fold_manifest_sha256,
                       self.fold_rows_sha256, self.r0_oof_prediction_sha256):
            need(isinstance(digest, str) and len(digest) == 64, "checksum format differs")
        if require_dependencies:
            self.validate_safe_dependencies()

    def validate_safe_dependencies(self) -> None:
        """Validate inference dependencies without touching any gold-bearing file."""
        need(self.status == "authorized" and self.execution_authorized is True, "safe runtime dependencies require authorization")
        for path, digest, label in (
            (self.task_card_path, self.task_card_sha256, "task card"),
            (self.preparer_path, self.preparer_sha256, "projection preparer"),
            (self.source_stage3_config_path, self.source_stage3_config_sha256, "Stage3 config"),
            (self.source_stage3_aggregate_path, self.source_stage3_aggregate_sha256, "Stage3 aggregate"),
            (self.label_free_projection_path, self.label_free_projection_sha256, "label-free projection"),
            (self.label_free_manifest_path, self.label_free_manifest_sha256, "label-free manifest"),
        ):
            _verify_ordinary_file(Path(path), digest, label, private="label-free" in label)
        for binding in self.fold_membership_bindings:
            _verify_ordinary_file(Path(binding.path), binding.sha256, "fold membership", private=True)
        for binding in self.checkpoint_bindings:
            _verify_ordinary_file(Path(binding.path), binding.sha256, "Stage3 checkpoint", private=True)
        for commit, path, digest, label in (
            (self.task_card_commit, self.task_card_path, self.task_card_sha256, "task-card"),
            (self.preparer_commit, self.preparer_path, self.preparer_sha256, "preparer"),
        ):
            committed = subprocess.run(["git", "show", f"{commit}:{path}"], check=False, capture_output=True)
            need(committed.returncode == 0 and sha256(committed.stdout).hexdigest() == digest,
                 f"{label} commit/file binding differs")
        request_config = subprocess.run(
            ["git", "show", f"{self.preparer_commit}:configs/kure_phase1_direct_oof.v1.json"],
            check=False, capture_output=True,
        )
        need(request_config.returncode == 0
             and sha256(request_config.stdout).hexdigest() == self.preparation_request_config_sha256,
             "preparation-request config commit binding differs")
        _validate_stage3_contract(self)
        _load_label_free_projection(self)
        validate_backbone_without_validation(_source_config(self).backbone)

    def require_execution_authorization(self) -> None:
        need(self.status == "authorized" and self.execution_authorized is True,
             "scientific execution is not authorized by the hash-bound task card")
        self.validate_safe_dependencies()


def config_sha256(config: KUREPhase1DirectOOFConfig) -> str:
    payload = json.dumps(asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def _config_file_sha256() -> str:
    need(CONFIG_FILE_PATH.is_file(), "canonical phase1-direct config is unavailable")
    return file_sha256(CONFIG_FILE_PATH)


def _source_config(config: KUREPhase1DirectOOFConfig) -> KUREOrdinalOOFConfig:
    # Parsing this safe config must not validate/hash its gold-bearing paths.
    source = KUREOrdinalOOFConfig.from_json(config.source_stage3_config_path, require_dependencies=False)
    need((source.train_path, source.train_sha256, source.fold_manifest_path, source.fold_manifest_sha256,
          source.fold_rows_path, source.fold_rows_sha256, source.r0_oof_prediction_path,
          source.r0_oof_prediction_sha256, source.seed, source.batch_size, source.max_length)
         == (config.train_path, config.train_sha256, config.fold_manifest_path, config.fold_manifest_sha256,
             config.fold_rows_path, config.fold_rows_sha256, config.r0_oof_prediction_path,
             config.r0_oof_prediction_sha256, config.seed, config.batch_size, config.max_length),
         "Stage3 source/fold/R0 contract differs")
    return source


@dataclass(frozen=True)
class _TextRow:
    identifier: str
    document_id: str
    prompt_num: str
    prompt: str
    essay: str


def _private_parents(path: Path) -> Sequence[Path]:
    anchor = Path("data/processed/restricted").resolve(); parent = path.resolve().parent
    need(anchor == parent or anchor in parent.parents, "private artifact is outside restricted anchor")
    chain = [parent]
    while chain[-1] != anchor: chain.append(chain[-1].parent)
    return tuple(reversed(chain))


def _verify_ordinary_file(path: Path, expected_sha256: str, label: str, *, private: bool) -> None:
    need(path.is_file() and not path.is_symlink() and file_sha256(path) == expected_sha256, f"{label} binding differs")
    if private:
        need(path.stat().st_mode & 0o007 == 0, f"{label} is world-accessible")
        need(all(parent.is_dir() and not parent.is_symlink() and parent.stat().st_mode & 0o007 == 0
                 for parent in _private_parents(path)), f"{label} parent ACL differs")


def _verify_private_file_at(path: Path, expected_sha256: str, label: str, anchor: Path) -> None:
    need(path.is_file() and not path.is_symlink() and file_sha256(path) == expected_sha256,
         f"{label} post-prediction binding differs")
    resolved_anchor = anchor.resolve(); cursor = path.resolve().parent
    need(resolved_anchor == cursor or resolved_anchor in cursor.parents, f"{label} private anchor differs")
    need(path.stat().st_mode & 0o007 == 0, f"{label} is world-accessible")
    while True:
        need(cursor.is_dir() and not cursor.is_symlink() and cursor.stat().st_mode & 0o007 == 0,
             f"{label} parent ACL differs")
        if cursor == resolved_anchor: break
        cursor = cursor.parent


def verify_post_prediction_gold_dependencies(config: KUREPhase1DirectOOFConfig) -> None:
    """Hash and ACL-check gold-bearing files only after full predictions exist."""
    _verify_private_file_at(Path(config.train_path), config.train_sha256, "canonical train", Path("eval"))
    for path, digest, label in (
        (config.fold_manifest_path, config.fold_manifest_sha256, "canonical fold manifest"),
        (config.fold_rows_path, config.fold_rows_sha256, "canonical fold rows"),
        (config.r0_oof_prediction_path, config.r0_oof_prediction_sha256, "exact R0 OOF"),
    ):
        _verify_private_file_at(Path(path), digest, label, Path("data/processed/restricted"))


def _load_label_free_projection(config: KUREPhase1DirectOOFConfig) -> tuple[Mapping[int, tuple[_TextRow, ...]], Mapping[str, int]]:
    """Validate the steward manifest and decode only the label-free projection."""
    manifest = json.loads(Path(config.label_free_manifest_path).read_text(encoding="utf-8"))
    expected_memberships = [asdict(item) for item in config.fold_membership_bindings]
    need(manifest.get("schema_version") == "mal2026-kure-phase1-direct-input-manifest-v1"
         and manifest.get("status") == "completed" and manifest.get("records") == 2000
         and manifest.get("fold_counts") == {str(fold): 400 for fold in range(5)}
         and manifest.get("projection_path") == config.label_free_projection_path
         and manifest.get("projection_sha256") == config.label_free_projection_sha256
         and manifest.get("projection_schema") == ["id", "document_id", "prompt_num", "prompt", "essay", "outer_fold"]
         and manifest.get("labels_present") is False and manifest.get("average_present") is False
         and manifest.get("gold_present") is False and manifest.get("source_train_sha256") == config.train_sha256
         and manifest.get("source_stage3_aggregate_sha256") == config.source_stage3_aggregate_sha256
         and manifest.get("generator_path") == config.preparer_path
         and manifest.get("generator_sha256") == config.preparer_sha256
         and manifest.get("generator_git_sha") == config.preparer_commit,
         "label-free manifest contract differs")
    need(manifest.get("preparation_request_config_sha256") == config.preparation_request_config_sha256,
         "label-free preparation-request config lineage differs")
    reported_memberships = [{key: item[key] for key in ("outer_fold", "path", "sha256")}
                            for item in manifest.get("fold_membership_bindings", ())]
    need(reported_memberships == expected_memberships, "label-free manifest membership lineage differs")
    by_fold: dict[int, list[_TextRow]] = {fold: [] for fold in range(5)}; folds: dict[str, int] = {}
    with Path(config.label_free_projection_path).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            need(set(item) == {"id", "document_id", "prompt_num", "prompt", "essay", "outer_fold"},
                 "label-free projection schema differs")
            need(not ({"score", "average", "gold", "label", "raw_gold"} & set(item)), "gold field entered projection")
            identifier, fold = item["id"], item["outer_fold"]
            need(isinstance(identifier, str) and identifier not in folds and fold in range(5), "projection ID/fold differs")
            row = _TextRow(identifier, str(item["document_id"]), str(item["prompt_num"]), item["prompt"], item["essay"])
            need(isinstance(row.prompt, str) and isinstance(row.essay, str), "projection text differs")
            folds[identifier] = fold; by_fold[fold].append(row)
    need(len(folds) == 2000 and all(len(by_fold[fold]) == 400 for fold in range(5)), "projection population differs")
    return {fold: tuple(rows) for fold, rows in by_fold.items()}, folds


def _validate_stage3_contract(config: KUREPhase1DirectOOFConfig) -> None:
    source = _source_config(config)
    from .kure_ordinal_oof import config_sha256 as stage3_config_sha256
    need(stage3_config_sha256(source) == config.source_stage3_report_config_sha256,
         "Stage3 report-config hash differs")
    try:
        aggregate = json.loads(Path(config.source_stage3_aggregate_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KUREPhase1DirectOOFError("Stage3 aggregate is invalid") from exc
    need(aggregate.get("schema_version") == "mal2026-kure-ordinal-oof-aggregate-v1"
         and aggregate.get("status") == "completed" and aggregate.get("run_id") == source.run_id
         and aggregate.get("config_sha256") == config.source_stage3_report_config_sha256
         and aggregate.get("fold_manifest_sha256") == config.fold_manifest_sha256
         and aggregate.get("fold_rows_sha256") == config.fold_rows_sha256
         and aggregate.get("r0_oof_prediction_sha256") == config.r0_oof_prediction_sha256
         and aggregate.get("validation_rows_loaded") is False and aggregate.get("average_target_used") is False,
         "Stage3 aggregate provenance differs")
    methods = aggregate.get("methods")
    coral = next((item for item in methods if item.get("method") == SOURCE_METHOD), None) if isinstance(methods, list) else None
    need(isinstance(coral, Mapping), "Stage3 coral-natural aggregate entry is missing")
    reported: dict[tuple[int, str], str] = {}
    for fold in coral.get("fold_bindings", ()):
        outer = fold.get("outer_fold")
        for axis in fold.get("axis_bindings", ()):
            reported[(outer, axis.get("axis"))] = axis.get("checkpoint_sha256")
    configured = {(item.outer_fold, item.axis): item.sha256 for item in config.checkpoint_bindings}
    need(reported == configured, "Stage3 aggregate/all 15 checkpoint hashes differ")
    reported_memberships = {int(item["outer_fold"]): item["restricted_prediction_sha256"]
                            for item in coral.get("fold_bindings", ())}
    need(reported_memberships == {item.outer_fold: item.sha256 for item in config.fold_membership_bindings},
         "Stage3 aggregate/five membership hashes differ")
    for item in config.checkpoint_bindings:
        expected = Path(source.restricted_output_root) / f"outer-{item.outer_fold:02d}" / SOURCE_METHOD / item.axis / "trainable.safetensors"
        need(Path(item.path) == expected, "Stage3 deterministic checkpoint path differs")


def _assert_private_file(path: Path) -> None:
    need(path.is_file() and not path.is_symlink(), "restricted file must be ordinary")
    mode = path.stat().st_mode
    need(mode & 0o007 == 0 and mode & 0o600 == 0o600
         and all(parent.stat().st_mode & 0o007 == 0 for parent in _private_parents(path)),
         "restricted file or parent is not project-private")


def _secure_directory(path: Path) -> None:
    for directory in _private_parents(path / "placeholder"):
        directory.mkdir(parents=False, exist_ok=True); os.chmod(directory, 0o770)
        need(not directory.is_symlink() and directory.stat().st_mode & 0o007 == 0
             and directory.stat().st_mode & 0o700 == 0o700, "restricted directory ACL differs")


def _publish_no_clobber(temporary: Path, path: Path) -> None:
    os.link(temporary, path)
    temporary.unlink()
    directory = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)


def _atomic_private_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    need(not path.exists(), "refusing to overwrite restricted phase1-direct predictions")
    _secure_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary, 0o660); _publish_no_clobber(Path(temporary), path); os.chmod(path, 0o660)
        _assert_private_file(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return file_sha256(path)


def _validate_public(value: Any) -> None:
    forbidden = {"source_id", "document_id", "essay", "prompt", "raw_gold", "prediction", "predictions"}
    if isinstance(value, Mapping):
        need(not (set(value) & forbidden), "restricted row content cannot enter public output")
        for child in value.values(): _validate_public(child)
    elif isinstance(value, (list, tuple)):
        for child in value: _validate_public(child)


def _atomic_public_json(path: Path, value: Mapping[str, Any]) -> str:
    need(not path.exists(), "refusing to overwrite public phase1-direct output")
    _validate_public(value); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        _publish_no_clobber(Path(temporary), path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return file_sha256(path)


def load_phase1_state(path: Path, expected_sha256: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Load only LoRA+CORAL tensors and prove that the saved cRT head is ignored."""
    from safetensors.torch import load_file
    need(path.is_file() and file_sha256(path) == expected_sha256, "phase1 checkpoint hash differs")
    state = load_file(str(path), device="cpu")
    required = {"score.weight", "score.bias", "cut_base", "cut_gaps"}
    ignored = {"head.weight", "head.bias"}
    need(required <= set(state) and ignored <= set(state), "saved CORAL/cRT tensor inventory differs")
    allowed = required | ignored | {key for key in state if "lora_" in key}
    need(set(state) == allowed and len({key for key in state if "lora_" in key}) > 0,
         "checkpoint contains tensors outside LoRA/CORAL/cRT contract")
    selected = {key: tensor for key, tensor in state.items() if key not in ignored}
    disclosure = {"loaded_tensor_count": len(selected), "loaded_lora_tensor_count": len(selected) - len(required),
                  "loaded_coral_tensors": sorted(required), "ignored_crt_tensors": sorted(ignored),
                  "ignored_crt_tensor_count": len(ignored)}
    return selected, disclosure


def restore_phase1_model(source: KUREOrdinalOOFConfig, binding: CheckpointBinding) -> tuple[Any, Mapping[str, Any]]:
    spec = CandidateSpec(SOURCE_METHOD, "coral", "natural")
    model, lineage = _build_axis_model(source.backbone, spec)
    selected, disclosure = load_phase1_state(Path(binding.path), binding.sha256)
    model_keys = set(model.state_dict())
    need(set(selected) <= model_keys, "saved phase1 tensors do not match the CORAL model")
    incompatible = model.load_state_dict(selected, strict=False)
    need(not incompatible.unexpected_keys and not (set(selected) & set(incompatible.missing_keys)),
         "phase1 checkpoint restore differs")
    return model, {"lineage": lineage, **disclosure}


def direct_coral_expected_score(logits: Any) -> Any:
    import torch
    pmf = coral_pmf(logits)
    need(pmf.ndim == 2 and pmf.shape[1] == 5 and bool(torch.isfinite(pmf).all()), "CORAL PMF differs")
    need(bool(torch.allclose(pmf.sum(1), torch.ones(len(pmf), device=pmf.device), atol=1e-5)), "CORAL PMF is not normalized")
    return (pmf * torch.arange(1, 6, device=pmf.device, dtype=pmf.dtype)).sum(1)


def _predict(model: Any, rows: Sequence[_TextRow], tokenizer: Any, *, batch_size: int, max_length: int, device: Any) -> np.ndarray:
    import torch
    model.to(device); model.eval(); values = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = tokenizer([render_input(row) for row in batch], truncation=True, max_length=max_length,
                                padding=True, return_tensors="pt")
            logits = model(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device))["logits"]
            values.append(direct_coral_expected_score(logits).cpu())
    result = torch.cat(values).numpy()
    need(result.shape == (len(rows),) and np.all(np.isfinite(result)) and np.all((1 <= result) & (result <= 5)),
         "direct CORAL predictions differ")
    return result


def _held_gold_after_persist(path: Path, expected_sha256: str, held_ids: set[str], persisted: Path) -> Mapping[str, tuple[float, float, float]]:
    """The persisted-file argument makes the held-label ordering an enforced contract."""
    need(persisted.is_file(), "held labels remain unavailable until predictions are persisted")
    _assert_private_file(persisted)
    _verify_private_file_at(path, expected_sha256, "canonical train", Path("eval"))
    result: dict[str, tuple[float, float, float]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line); identifier = str(raw["id"])
            if identifier not in held_ids:
                continue
            scores = raw["score"]
            need(set(scores) == {*AXES, "average"}, "held score schema differs")
            axes = tuple(float(scores[axis]) for axis in AXES)
            need(all(math.isfinite(value) and 1 <= value <= 5 for value in axes), "held score differs")
            result[identifier] = axes  # type: ignore[assignment]
    need(set(result) == held_ids, "held score coverage differs")
    return result


def set_process_title(stage: str) -> str:
    """Expose the current MAL2026 direct-evaluation stage in process listings."""
    need(stage and len(stage) <= 96 and all(char.isalnum() or char in "._:-" for char in stage),
         "process-title stage differs")
    import setproctitle
    title = f"mal2026:direct:{stage}"
    setproctitle.setproctitle(title)
    need(setproctitle.getproctitle() == title, "setproctitle did not preserve the requested title")
    return title


def _environment() -> Mapping[str, Any]:
    import torch
    import setproctitle
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "transformers": transformers_version, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "process_title": setproctitle.getproctitle()}


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def run(config: KUREPhase1DirectOOFConfig | str | Path, *, outer_fold: int,
        validate_only: bool = False, smoke: bool = False) -> Mapping[str, Any]:
    value = KUREPhase1DirectOOFConfig.from_json(config) if isinstance(config, (str, Path)) else config
    need(outer_fold in range(5), "outer fold must be 0..4")
    need(not smoke or outer_fold == 0, "smoke requires outer fold 0")
    if validate_only:
        set_process_title("validate")
        # Pending drafts intentionally have unfilled post-commit projection/card hashes.
        if value.status == "authorized": value.validate_safe_dependencies()
        return {"schema_version": SCHEMA_VERSION, "status": "validated", "execution_authorized": value.execution_authorized,
                "checkpoint_bindings": 15, "validation_rows_loaded": False, "average_target_used": False, "gpu_used": False}
    value.require_execution_authorization()
    set_process_title(f"{'smoke' if smoke else 'oof'}:f{outer_fold}:load")
    import torch
    from transformers import AutoTokenizer
    need(torch.cuda.is_available(), "phase1-direct inference requires an explicitly launched GPU job")
    if smoke:
        need(os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and torch.cuda.device_count() == 1,
             "smoke requires only physical GPU0 exposed")
    source = _source_config(value); validate_backbone_without_validation(source.backbone)
    projection, _ = _load_label_free_projection(value)
    held = list(projection[outer_fold])
    if smoke: held = held[:8]
    tokenizer = AutoTokenizer.from_pretrained(source.backbone.model_path, revision=source.backbone.model_revision,
                                              local_files_only=True, trust_remote_code=False, use_fast=True)
    held_audit = token_length_audit(held, tokenizer, value.max_length)
    axes = ("content",) if smoke else AXES
    predictions = []
    axis_bindings = []
    for axis in axes:
        set_process_title(f"{'smoke' if smoke else 'oof'}:f{outer_fold}:{axis}")
        binding = next(item for item in value.checkpoint_bindings if item.outer_fold == outer_fold and item.axis == axis)
        model, disclosure = restore_phase1_model(source, binding)
        prediction = _predict(model, held, tokenizer, batch_size=value.batch_size, max_length=value.max_length,
                              device=torch.device("cuda"))
        predictions.append(prediction)
        axis_bindings.append({"axis": axis, "checkpoint_sha256": binding.sha256,
                              "decode": "saved_phase1_CORAL_PMF_expected_score", **disclosure})
        del model; torch.cuda.empty_cache()
    matrix = np.column_stack(predictions)
    set_process_title(f"{'smoke' if smoke else 'oof'}:f{outer_fold}:persist")
    restricted = Path(value.restricted_output_root) / ("smoke/outer-00" if smoke else f"outer-{outer_fold:02d}")
    private_path = restricted / METHOD / "predictions.jsonl"
    private_sha = _atomic_private_jsonl(private_path, (
        {"source_id": row.identifier, "outer_fold": outer_fold,
         "prediction": {axis: float(matrix[index, column]) for column, axis in enumerate(axes)}}
        for index, row in enumerate(held)
    ))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "status": "completed", "mode": "smoke" if smoke else "outer_fold",
        "nonselectable": smoke, "run_id": value.run_id, "outer_fold": outer_fold, "records": len(held),
        "method": METHOD, "source_method": SOURCE_METHOD, "training_performed": False,
        "calibration_performed": False, "selection_performed": False,
        "decode": "saved_phase1_CORAL_PMF_expected_score", "axis_bindings": axis_bindings,
        "held_token_length_audit": held_audit, "restricted_prediction_sha256": private_sha,
        "config_sha256": config_sha256(value), "config_file_sha256": _config_file_sha256(),
        "task_card_sha256": value.task_card_sha256, "task_card_commit": value.task_card_commit,
        "source_stage3_aggregate_sha256": value.source_stage3_aggregate_sha256,
        "fold_manifest_sha256": value.fold_manifest_sha256, "fold_rows_sha256": value.fold_rows_sha256,
        "r0_oof_prediction_sha256": value.r0_oof_prediction_sha256,
        "validation_rows_loaded": False, "average_target_used": False,
        "logical_command": f"PYTHONPATH=src {sys.executable} scripts/run_kure_phase1_direct_oof.py --config configs/kure_phase1_direct_oof.v1.json --outer-fold {outer_fold}" + (" --smoke" if smoke else ""),
        "environment": _environment(), "privacy": "aggregate_only_public_predictions_restricted",
    }
    if not smoke:
        # This call is intentionally after the durable restricted write above.
        gold = _held_gold_after_persist(Path(value.train_path), value.train_sha256,
                                        {row.identifier for row in held}, private_path)
        result["metrics"] = compute_iterative_tail_metrics([gold[row.identifier] for row in held], matrix)
    public_path = Path(value.output_root) / ("smoke/outer-00.json" if smoke else f"outer-{outer_fold:02d}.json")
    _atomic_public_json(public_path, result)
    return result


def _load_private_unbound(path: Path, fold: int) -> Mapping[str, tuple[float, float, float]]:
    _assert_private_file(path)
    result: dict[str, tuple[float, float, float]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            need(set(item) == {"source_id", "outer_fold", "prediction"} and item["outer_fold"] == fold,
                 "restricted prediction row differs")
            identifier = item["source_id"]; prediction = item["prediction"]
            need(isinstance(identifier, str) and identifier not in result and set(prediction) == set(AXES),
                 "restricted prediction identity/axes differ")
            values = tuple(float(prediction[axis]) for axis in AXES)
            need(all(math.isfinite(value) and 1 <= value <= 5 for value in values), "restricted prediction value differs")
            result[identifier] = values  # type: ignore[assignment]
    need(len(result) == 400, "restricted prediction fold size differs")
    return result


def _validate_outer_public(public: Mapping[str, Any], config: KUREPhase1DirectOOFConfig, fold: int,
                           private_path: Path) -> None:
    expected_bindings = [item for item in config.checkpoint_bindings if item.outer_fold == fold]
    need(public.get("schema_version") == SCHEMA_VERSION and public.get("status") == "completed"
         and public.get("mode") == "outer_fold" and public.get("nonselectable") is False
         and public.get("run_id") == config.run_id and public.get("outer_fold") == fold
         and public.get("records") == 400 and public.get("method") == METHOD
         and public.get("source_method") == SOURCE_METHOD
         and public.get("decode") == "saved_phase1_CORAL_PMF_expected_score"
         and public.get("training_performed") is False and public.get("calibration_performed") is False
         and public.get("selection_performed") is False
         and public.get("config_sha256") == config_sha256(config)
         and public.get("config_file_sha256") == _config_file_sha256()
         and public.get("task_card_sha256") == config.task_card_sha256
         and public.get("task_card_commit") == config.task_card_commit
         and public.get("source_stage3_aggregate_sha256") == config.source_stage3_aggregate_sha256
         and public.get("fold_manifest_sha256") == config.fold_manifest_sha256
         and public.get("fold_rows_sha256") == config.fold_rows_sha256
         and public.get("r0_oof_prediction_sha256") == config.r0_oof_prediction_sha256
         and public.get("restricted_prediction_sha256") == file_sha256(private_path)
         and public.get("validation_rows_loaded") is False and public.get("average_target_used") is False,
         "phase1-direct outer identity/contract differs")
    axes = public.get("axis_bindings")
    need(isinstance(axes, list) and [item.get("axis") for item in axes] == list(AXES), "outer axis order differs")
    source = _source_config(config)
    for item, expected in zip(axes, expected_bindings, strict=True):
        need(set(item) == {"axis", "checkpoint_sha256", "decode", "lineage", "loaded_tensor_count",
                           "loaded_lora_tensor_count", "loaded_coral_tensors", "ignored_crt_tensors",
                           "ignored_crt_tensor_count"}
             and item.get("checkpoint_sha256") == expected.sha256
             and item.get("decode") == "saved_phase1_CORAL_PMF_expected_score"
             and item.get("ignored_crt_tensors") == ["head.bias", "head.weight"]
             and item.get("ignored_crt_tensor_count") == 2
             and item.get("loaded_coral_tensors") == ["cut_base", "cut_gaps", "score.bias", "score.weight"]
             and type(item.get("loaded_lora_tensor_count")) is int and item["loaded_lora_tensor_count"] > 0
             and item.get("loaded_tensor_count") == item["loaded_lora_tensor_count"] + 4
             and isinstance(item.get("lineage"), Mapping)
             and item["lineage"].get("arm") == "aihub_full_backbone"
             and item["lineage"].get("pooling") == "cls_l2"
             and item["lineage"].get("artifact_sha256") == source.backbone.warmstart_artifact_sha256,
             "outer axis checkpoint/disclosure differs")


def prediction_band_diagnostics(predictions: np.ndarray) -> Mapping[str, Any]:
    need(predictions.ndim == 2 and predictions.shape[1] == 3, "prediction diagnostic shape differs")
    bands = np.asarray([[official_half_up(float(value)) for value in row] for row in predictions], dtype=int)
    counts = {axis: {str(score): int(np.sum(bands[:, index] == score)) for score in range(1, 6)}
              for index, axis in enumerate(AXES)}
    total = int(bands.size); collapsed = int(np.sum((bands == 3) | (bands == 4)))
    return {"half_up_band_counts": counts, "total_axis_predictions": total,
            "band_3_4_count": collapsed, "band_3_4_collapse_rate": collapsed / total}


def summarize_gpu_telemetry(path: Path, selected_gpus: Sequence[int], minimum_samples: int) -> Mapping[str, Any]:
    need(path.is_file() and not path.is_symlink() and minimum_samples > 0, "telemetry input contract differs")
    selected = set(selected_gpus); need(selected and len(selected) == len(selected_gpus), "selected GPU inventory differs")
    grouped: dict[int, list[Mapping[str, str]]] = {gpu: [] for gpu in selected}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp", "index", "uuid", "name", "memory.total", "driver_version", "utilization.gpu", "memory.used"}
        need(set(reader.fieldnames or ()) == required, "telemetry columns differ")
        for row in reader:
            gpu = int(row["index"].strip()); need(gpu in selected, "telemetry contains an unselected GPU")
            need(all(row[field].strip() for field in ("timestamp", "uuid", "name", "driver_version")),
                 "telemetry identity/timestamp is empty")
            grouped[gpu].append(row)
    need(all(len(grouped[gpu]) >= minimum_samples for gpu in selected), "telemetry selected-GPU/minimum-sample contract failed")
    summaries = []
    for gpu in sorted(selected):
        rows = grouped[gpu]
        identities = {(row["uuid"].strip(), row["name"].strip(), int(row["memory.total"]), row["driver_version"].strip()) for row in rows}
        need(len(identities) == 1, "telemetry hardware identity changed")
        uuid, name, total, driver = identities.pop()
        utilization = [float(row["utilization.gpu"]) for row in rows]
        memory = [float(row["memory.used"]) for row in rows]
        need(total > 0 and all(math.isfinite(value) and 0 <= value <= 100 for value in utilization)
             and all(math.isfinite(value) and 0 <= value <= total for value in memory),
             "telemetry numeric sanity differs")
        summaries.append({"physical_gpu": gpu, "uuid": uuid, "name": name, "memory_total_mib": total,
                          "driver_version": driver, "samples": len(rows),
                          "mean_utilization_percent": sum(utilization) / len(utilization),
                          "peak_utilization_percent": max(utilization),
                          "mean_memory_used_mib": sum(memory) / len(memory),
                          "peak_memory_used_mib": max(memory)})
    return {"schema_version": "mal2026-gpu-telemetry-summary-v1",
            "minimum_samples_per_gpu": minimum_samples, "gpus": summaries}


def scheduler_state_conflict(state: Mapping[str, Any], selected_gpus: Sequence[int], *, age_seconds: float,
                             expected_run_id: str = "vllm-soak-gpu0-3-120h-20260803-004") -> str | None:
    """Return a fail-closed conflict reason for an explicitly named idle scheduler."""
    allowed = {"delayed", "armed", "launched", "launched_then_stopped_by_user",
               "superseded_before_launch", "expired_without_launch"}
    need(isinstance(state, Mapping) and state.get("schema_version") == "mal2026-vllm-idle-arm-state-v1",
         "active scheduler schema is malformed")
    status = state.get("status")
    need(status in allowed, "active scheduler status is unknown")
    need(state.get("physical_gpus") == [0, 1, 2, 3], "active scheduler GPU inventory is malformed")
    identity = state.get("run_id") if status in {"launched", "launched_then_stopped_by_user"} else state.get("planned_run_id")
    need(expected_run_id in {"vllm-soak-gpu0-3-120h-20260803-004",
                             "vllm-soak-gpu0-3-120h-20260803-005"}
         and identity == expected_run_id, "active scheduler run identity is malformed")
    selected = set(selected_gpus)
    need(selected and selected <= {0, 1, 2, 3}, "scheduler selected GPU inventory differs")
    if not selected.intersection(state["physical_gpus"]): return None
    if status == "launched": return "active named idle-scheduler already launched overlapping work"
    if status in {"armed", "delayed"}:
        # Scheduler state may be written through a filesystem/client whose
        # wall clock leads this launcher slightly. Bound that integration-only
        # skew instead of treating a fresh state as stale or unsafe.
        need(math.isfinite(age_seconds) and age_seconds >= -300,
             "active scheduler state age is malformed")
        age_seconds = max(0.0, age_seconds)
        if age_seconds > 120: return "stale active named idle-scheduler state"
        required = state.get("idle_required_seconds", 1800)
        consecutive = state.get("consecutive_idle_seconds", 0)
        need(type(required) in {int, float} and type(consecutive) in {int, float}
             and math.isfinite(float(required)) and math.isfinite(float(consecutive))
             and float(required) > 0 and 0 <= float(consecutive) <= float(required),
             "active scheduler idle counters are malformed")
        if float(required) - float(consecutive) < 300: return "active named idle-scheduler has less than five-minute margin"
    return None


def aggregate(config: KUREPhase1DirectOOFConfig | str | Path) -> Mapping[str, Any]:
    value = KUREPhase1DirectOOFConfig.from_json(config) if isinstance(config, (str, Path)) else config
    set_process_title("aggregate")
    value.require_execution_authorization(); source = _source_config(value)
    # Collect and authenticate all predictions before any gold labels are read.
    predictions: dict[str, tuple[float, float, float]] = {}; bindings = []
    for fold in range(5):
        public_path = Path(value.output_root) / f"outer-{fold:02d}.json"
        private_path = Path(value.restricted_output_root) / f"outer-{fold:02d}" / METHOD / "predictions.jsonl"
        need(public_path.is_file() and private_path.is_file(), "phase1-direct outer output is incomplete")
        public = json.loads(public_path.read_text(encoding="utf-8"))
        _assert_private_file(private_path)
        _validate_outer_public(public, value, fold, private_path)
        fold_predictions = _load_private_unbound(private_path, fold)
        need(not (set(predictions) & set(fold_predictions)), "OOF fold predictions overlap")
        predictions.update(fold_predictions)
        bindings.append({"outer_fold": fold, "public_sha256": file_sha256(public_path),
                         "restricted_prediction_sha256": file_sha256(private_path),
                         "checkpoint_sha256": [item.sha256 for item in value.checkpoint_bindings if item.outer_fold == fold]})
    need(len(predictions) == 2000, "full phase1-direct OOF size differs")
    # Projection folds are safe; canonical fold rows/R0/gold are opened only now.
    _, folds = _load_label_free_projection(value)
    need(set(predictions) == set(folds)
         and all(all(folds[identifier] == fold for identifier in _load_private_unbound(
             Path(value.restricted_output_root) / f"outer-{fold:02d}" / METHOD / "predictions.jsonl", fold))
                 for fold in range(5)), "full phase1-direct fold assignment differs")
    verify_post_prediction_gold_dependencies(value)
    from .r0_ordinal_residual import load_embedding_artifact
    _, canonical_fold_rows = load_embedding_artifact(value.fold_manifest_path, value.fold_rows_path)
    canonical_folds = {row.source_id: int(row.oof_fold) for row in canonical_fold_rows}
    need(canonical_folds == folds, "post-prediction canonical fold assignment differs")
    from .kure_ordinal_oof import load_raw_axis_gold
    truth = load_raw_axis_gold(value.train_path, value.train_sha256); r0 = load_exact_r0(source)
    ordered = list(truth)
    truth_array = np.asarray([truth[key] for key in ordered], dtype=float)
    r0_array = np.asarray([r0[key] for key in ordered], dtype=float)
    candidate = np.asarray([predictions[key] for key in ordered], dtype=float)
    metrics = compute_iterative_tail_metrics(truth_array, candidate)
    baseline_metrics = compute_iterative_tail_metrics(truth_array, r0_array)
    decision = promotion_gate(truth_array, r0_array, candidate, ordered, seed=value.seed)
    result = {
        "schema_version": "mal2026-kure-phase1-direct-oof-aggregate-v1", "status": "completed",
        "mode": "full_oof", "run_id": value.run_id, "records": 2000, "folds": 5,
        "method": METHOD, "source_method": SOURCE_METHOD, "training_performed": False,
        "calibration_performed": False, "selection_performed": False,
        "metrics": metrics, "exact_r0_metrics": baseline_metrics,
        "prediction_diagnostics": prediction_band_diagnostics(candidate),
        "improvements_vs_exact_r0": metric_improvements(baseline_metrics, metrics),
        "common_stage3_promotion_gate": decision, "common_stage3_promotion_gate_passed": decision["eligible"],
        "automatic_stage6_deployment_eligible": False,
        "automatic_stage6_deployment_disclosure": "diagnostic recovery is outside the frozen Stage6 trust chain regardless of gate result",
        "protected_output": "exact_r0", "fold_bindings": bindings,
        "config_sha256": config_sha256(value), "config_file_sha256": _config_file_sha256(),
        "task_card_sha256": value.task_card_sha256, "task_card_commit": value.task_card_commit,
        "source_stage3_config_sha256": value.source_stage3_config_sha256,
        "source_stage3_report_config_sha256": value.source_stage3_report_config_sha256,
        "source_stage3_aggregate_sha256": value.source_stage3_aggregate_sha256,
        "fold_manifest_sha256": value.fold_manifest_sha256, "fold_rows_sha256": value.fold_rows_sha256,
        "r0_oof_prediction_sha256": value.r0_oof_prediction_sha256,
        "validation_rows_loaded": False, "average_target_used": False,
        "git_sha": _git_sha(), "environment": _environment(),
        "logical_command": f"PYTHONPATH=src {sys.executable} scripts/run_kure_phase1_direct_oof.py --config configs/kure_phase1_direct_oof.v1.json --aggregate",
        "privacy": "aggregate_only_no_rows_ids_text_or_predictions",
    }
    _atomic_public_json(Path(value.output_root) / "aggregate.json", result)
    return result


__all__ = ["CheckpointBinding", "FoldMembershipBinding", "KUREPhase1DirectOOFConfig",
           "KUREPhase1DirectOOFError", "aggregate", "config_sha256", "direct_coral_expected_score",
           "load_phase1_state", "prediction_band_diagnostics", "restore_phase1_model", "run",
           "scheduler_state_conflict", "set_process_title", "summarize_gpu_telemetry",
           "verify_post_prediction_gold_dependencies"]
