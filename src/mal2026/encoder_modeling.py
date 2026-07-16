"""Strict encoder-regression model construction for writing-score experiments.

This module deliberately separates the two requested embedding backbones.  In
particular, NV-Embed-v2 is not allowed to execute remote code until a pinned
snapshot has been reviewed and its review record verifies every Python source
file in that snapshot.  There is intentionally no "best effort" fallback for
unknown forward outputs, LoRA targets, pooling, or normalization.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
NV_EMBED_MODEL = "nvidia/NV-Embed-v2"
SCORE_FIELDS = ("content", "organization", "expression", "average")
_IMMUTABLE_REVISION_LENGTHS = frozenset((40, 64))


class EncoderContractError(ValueError):
    """Raised when an encoder configuration is unsafe or underspecified."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EncoderContractError(message)


def is_immutable_revision(value: object) -> bool:
    """Return whether *value* is a full Git object ID, not a branch/tag."""

    return (
        isinstance(value, str)
        and len(value) in _IMMUTABLE_REVISION_LENGTHS
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class NVRemoteCodeReview:
    """A version-controlled approval record for a local NV remote-code snapshot."""

    model_id: str
    revision: str
    license_acknowledged: bool
    use_case: str
    reviewer: str
    outcome: str
    reviewed_files: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NVRemoteCodeReview":
        """Parse the one canonical NV review schema without coercion.

        In particular, ``bool("false")`` is true in Python; accepting that
        conversion would silently authorize remote code.  Require exact keys
        and an actual JSON boolean instead.
        """
        required = {"model_id", "revision", "license_acknowledged", "use_case", "reviewer", "outcome", "reviewed_files"}
        _require(isinstance(value, Mapping) and set(value) == required, "NV remote-code review has unknown or missing fields")
        _require(isinstance(value["license_acknowledged"], bool), "NV license_acknowledged must be a JSON boolean")
        _require(isinstance(value["reviewed_files"], Mapping), "NV reviewed_files must be an object")
        try:
            review = cls(
                model_id=value["model_id"],
                revision=value["revision"],
                license_acknowledged=value["license_acknowledged"],
                use_case=value["use_case"],
                reviewer=value["reviewer"],
                outcome=value["outcome"],
                reviewed_files=dict(value["reviewed_files"]),
            )
        except (KeyError, TypeError) as error:
            raise EncoderContractError("NV remote-code review record is incomplete") from error
        _require(all(isinstance(item, str) for item in (review.model_id, review.revision, review.use_case, review.reviewer, review.outcome)), "NV review text fields must be strings")
        review.validate_record()
        return review

    def validate_record(self) -> None:
        _require(self.model_id == NV_EMBED_MODEL, "NV review model_id must be nvidia/NV-Embed-v2")
        _require(is_immutable_revision(self.revision), "NV review revision must be an immutable commit hash")
        _require(self.license_acknowledged, "NV run requires CC-BY-NC-4.0 acknowledgement")
        _require(
            self.use_case == "research_noncommercial",
            "NV remote code is permitted only for acknowledged research_noncommercial use",
        )
        _require(bool(self.reviewer.strip()), "NV remote-code reviewer is required")
        _require(self.outcome == "approved", "NV remote-code reviewer outcome must be approved")
        _require(bool(self.reviewed_files), "NV review must hash reviewed remote-code files")
        for relative_path, digest in self.reviewed_files.items():
            candidate = Path(relative_path)
            _require(
                not candidate.is_absolute() and ".." not in candidate.parts and candidate.suffix == ".py",
                "NV reviewed_files must contain safe relative Python paths only",
            )
            _require(
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest.lower()),
                f"NV reviewed hash for {relative_path!r} is not SHA-256",
            )


def load_nv_remote_code_review(path: str | Path) -> NVRemoteCodeReview:
    """Load a review record without executing any remote model code."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise EncoderContractError("NV remote-code review must be a JSON object")
    return NVRemoteCodeReview.from_mapping(raw)


def verify_nv_snapshot(snapshot_dir: str | Path, review: NVRemoteCodeReview) -> None:
    """Fail closed unless the reviewed file list exactly covers local Python code.

    The snapshot must have been downloaded at ``review.revision`` by a separate
    provenance-aware step.  This verifier performs no network access and must
    run immediately before ``trust_remote_code=True``.
    """

    root = Path(snapshot_dir).resolve()
    _require(root.is_dir() and not Path(snapshot_dir).is_symlink(), "NV local snapshot directory does not exist or is a symlink")
    _require(root.name == review.revision, "NV snapshot directory must be named by the reviewed immutable revision")
    _require(not any(candidate.is_symlink() for candidate in root.rglob("*")), "NV snapshot may not contain symlinks")
    actual_python = {
        file.relative_to(root).as_posix()
        for file in root.rglob("*.py")
        if file.is_file()
    }
    expected_python = set(review.reviewed_files)
    _require(actual_python == expected_python, "NV snapshot Python files differ from approved review record")
    for relative_path, expected_hash in review.reviewed_files.items():
        file = root / relative_path
        _require(file.is_file() and not file.is_symlink(), f"reviewed NV file missing: {relative_path}")
        _require(_sha256_file(file) == expected_hash, f"NV remote-code hash mismatch: {relative_path}")


@dataclass(frozen=True)
class EncoderModelSpec:
    """The architecture-critical settings that must be frozen before a run."""

    backbone: str
    model_id: str
    revision: str
    tokenizer_revision: str
    pooling: str
    normalize_embeddings: bool
    lora_target_modules: tuple[str, ...]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    regression_loss: str
    loss_reduction: str
    nv_snapshot_dir: str | None = None
    nv_remote_code_review: NVRemoteCodeReview | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EncoderModelSpec":
        try:
            targets = value["lora_target_modules"]
            normalization = value["normalize_embeddings"]
            _require(isinstance(normalization, bool), "normalize_embeddings must be a boolean")
            spec = cls(
                backbone=str(value["backbone"]),
                model_id=str(value["model_id"]),
                revision=str(value["revision"]),
                tokenizer_revision=str(value["tokenizer_revision"]),
                pooling=str(value["pooling"]),
                normalize_embeddings=normalization,
                lora_target_modules=tuple(str(item) for item in targets),
                lora_rank=int(value["lora_rank"]),
                lora_alpha=int(value["lora_alpha"]),
                lora_dropout=float(value["lora_dropout"]),
                regression_loss=str(value["regression_loss"]),
                loss_reduction=str(value["loss_reduction"]),
                nv_snapshot_dir=value.get("nv_snapshot_dir"),
                nv_remote_code_review=(NVRemoteCodeReview.from_mapping(value["nv_remote_code_review"])
                                       if value.get("nv_remote_code_review") is not None else None),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise EncoderContractError("encoder model specification is incomplete") from error
        spec.validate()
        return spec

    def validate(self) -> None:
        _require(self.backbone in {"qwen3_embedding", "nv_embed_v2"}, "unsupported encoder backbone")
        expected_model = QWEN3_EMBEDDING_MODEL if self.backbone == "qwen3_embedding" else NV_EMBED_MODEL
        _require(self.model_id == expected_model, "backbone and model_id do not match")
        _require(is_immutable_revision(self.revision), "model revision must be an immutable commit hash")
        _require(is_immutable_revision(self.tokenizer_revision), "tokenizer revision must be an immutable commit hash")
        expected_pooling = "last_nonpad" if self.backbone == "qwen3_embedding" else "remote_sentence_embedding"
        _require(self.pooling == expected_pooling, f"{self.backbone} requires pooling={expected_pooling!r}")
        _require(self.normalize_embeddings is True, "L2 normalization must be explicit and enabled")
        _require(bool(self.lora_target_modules), "explicit LoRA target modules are required")
        _require(all(target and "." not in target for target in self.lora_target_modules), "LoRA targets must be leaf module names")
        _require(len(set(self.lora_target_modules)) == len(self.lora_target_modules), "LoRA target modules must be unique")
        _require(self.lora_rank > 0 and self.lora_alpha > 0 and 0 <= self.lora_dropout < 1, "invalid LoRA hyperparameters")
        _require(self.regression_loss == "mse", "encoder regression_loss must explicitly be mse")
        _require(self.loss_reduction == "mean", "encoder loss_reduction must explicitly be mean")
        is_nv = self.backbone == "nv_embed_v2"
        _require(is_nv == bool(self.nv_snapshot_dir), "NV snapshot path is required only for NV-Embed-v2")
        _require(is_nv == (self.nv_remote_code_review is not None), "NV typed review is required only for NV-Embed-v2")
        if is_nv:
            assert self.nv_remote_code_review is not None
            _require(self.tokenizer_revision == self.revision, "NV local tokenizer revision must match reviewed model revision")
            _require(self.nv_remote_code_review.model_id == self.model_id and self.nv_remote_code_review.revision == self.revision, "NV review must bind this exact model revision")

    def validate_nv_runtime(self) -> None:
        """Validate the local reviewed snapshot immediately before remote code."""
        _require(self.backbone == "nv_embed_v2", "NV runtime validation applies only to NV-Embed-v2")
        assert self.nv_snapshot_dir is not None and self.nv_remote_code_review is not None
        verify_nv_snapshot(self.nv_snapshot_dir, self.nv_remote_code_review)


def validate_lora_targets(model: Any, targets: Sequence[str]) -> None:
    """Require every configured target to match at least one actual leaf module."""

    available = {name.rsplit(".", maxsplit=1)[-1] for name, _ in model.named_modules()}
    missing = sorted(set(targets) - available)
    _require(not missing, f"configured LoRA target modules are absent: {', '.join(missing)}")


def _last_nonpad_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Pool the final non-padding token for either left- or right-padding."""

    import torch

    _require(last_hidden_state.ndim == 3, "Qwen encoder must return [batch, tokens, hidden]")
    _require(attention_mask.ndim == 2, "attention_mask must be [batch, tokens]")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
    masked_positions = positions.masked_fill(attention_mask.to(torch.bool).logical_not(), -1)
    final_positions = masked_positions.max(dim=1).values
    _require(bool((final_positions >= 0).all().item()), "all encoder inputs must contain at least one token")
    return last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), final_positions]


def _remote_sentence_embedding(outputs: Any) -> Any:
    """Extract only NV's explicit remote-code sentence embedding output."""

    if isinstance(outputs, Mapping):
        embedding = outputs.get("sentence_embedding")
    else:
        embedding = getattr(outputs, "sentence_embedding", None)
    _require(embedding is not None, "NV remote model must return sentence_embedding explicitly")
    _require(getattr(embedding, "ndim", None) == 2, "NV sentence_embedding must be [batch, hidden]")
    return embedding


def build_encoder_regressor(spec: EncoderModelSpec) -> Any:
    """Build a PEFT encoder and four-target regression head.

    Imports of model libraries are deferred so static contract tests can run in
    the minimal repository environment. Automatic device placement is never
    requested here; placement is owned by Accelerate/DDP.
    """

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel
    except ImportError as error:  # pragma: no cover - exercised in runtime environment
        raise RuntimeError("encoder training requires torch, transformers, and peft") from error

    model_kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if spec.backbone == "nv_embed_v2":
        # The sole typed review source is part of spec/config and validates
        # snapshot revision plus every Python file before remote code executes.
        spec.validate_nv_runtime()
        model_kwargs.update({"trust_remote_code": True, "local_files_only": True})
        model_source = spec.nv_snapshot_dir
    else:
        model_kwargs["trust_remote_code"] = False
        model_source = spec.model_id

    backbone = AutoModel.from_pretrained(model_source, **model_kwargs)
    validate_lora_targets(backbone, spec.lora_target_modules)
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=spec.lora_rank,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=list(spec.lora_target_modules),
        bias="none",
    )
    backbone = get_peft_model(backbone, peft_config)
    hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
    _require(isinstance(hidden_size, int) and hidden_size > 0, "encoder config must expose hidden_size")

    class EncoderRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.regression_head = nn.Linear(hidden_size, len(SCORE_FIELDS))

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None) -> Mapping[str, Any]:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            if spec.pooling == "last_nonpad":
                embedding = _last_nonpad_pool(outputs.last_hidden_state, attention_mask)
            elif spec.pooling == "remote_sentence_embedding":
                embedding = _remote_sentence_embedding(outputs)
            else:  # Config validation makes this unreachable; retain fail-closed behavior.
                raise EncoderContractError(f"unsupported pooling: {spec.pooling}")
            embedding = functional.normalize(embedding, p=2, dim=-1)
            predictions = self.regression_head(embedding.float())
            result: dict[str, Any] = {"predictions": predictions, "embeddings": embedding}
            if labels is not None:
                _require(tuple(labels.shape[-1:]) == (len(SCORE_FIELDS),), "labels must have four score targets")
                result["loss"] = functional.mse_loss(predictions, labels.float(), reduction="mean")
            return result

    return EncoderRegressor()


def finite_score_matrix(rows: Iterable[Sequence[float]]) -> None:
    """Small dependency-free validation used before tensors are constructed."""

    for row_number, row in enumerate(rows):
        _require(len(row) == len(SCORE_FIELDS), f"row {row_number} does not have four scores")
        _require(all(math.isfinite(float(value)) for value in row), f"row {row_number} has non-finite score")
