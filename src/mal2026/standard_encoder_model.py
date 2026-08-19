"""Model construction for the standard Hugging Face Trainer encoder runs.

The training lifecycle lives in :mod:`standard_encoder_train`; this module only
constructs configurable canonical-score regression models.  It deliberately has no optimizer,
DDP, or manual training loop.  NV-Embed-v2 remote code is admitted only from a
reviewed immutable local snapshot and is made offline before Transformers is
allowed to import it.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

QWEN3_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
NV_MODEL_ID = "nvidia/NV-Embed-v2"
SCORE_FIELDS = ("content", "organization", "expression", "average")
NV_EMBEDDING_DIM = 4096
LORA_TARGETS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})


class StandardEncoderContractError(ValueError):
    """Raised before an unsafe or ambiguous encoder model is constructed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise StandardEncoderContractError(message)


def _immutable_revision(value: object) -> bool:
    return isinstance(value, str) and len(value) in {40, 64} and all(char in "0123456789abcdef" for char in value.lower())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class NVReview:
    """The exact approved source-file inventory for one NV snapshot."""

    model_id: str
    revision: str
    license_acknowledged: bool
    use_case: str
    reviewer: str
    outcome: str
    reviewed_files: Mapping[str, str]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NVReview":
        required = {"model_id", "revision", "license_acknowledged", "use_case", "reviewer", "outcome", "reviewed_files"}
        _need(set(raw) == required, "NV review has unknown or missing fields")
        _need(isinstance(raw["license_acknowledged"], bool), "NV license acknowledgement must be a JSON boolean")
        _need(isinstance(raw["reviewed_files"], Mapping), "NV reviewed_files must be an object")
        review = cls(
            model_id=raw["model_id"], revision=raw["revision"], license_acknowledged=raw["license_acknowledged"],
            use_case=raw["use_case"], reviewer=raw["reviewer"], outcome=raw["outcome"], reviewed_files=dict(raw["reviewed_files"]),
        )
        _need(all(isinstance(value, str) for value in (review.model_id, review.revision, review.use_case, review.reviewer, review.outcome)), "NV review text fields must be strings")
        _need(review.model_id == NV_MODEL_ID and _immutable_revision(review.revision), "NV review must pin NV-Embed-v2 to an immutable revision")
        _need(review.license_acknowledged and review.use_case == "research_noncommercial" and review.outcome == "approved" and bool(review.reviewer.strip()), "NV review approval gate failed")
        _need(bool(review.reviewed_files), "NV review must list source files")
        for name, digest in review.reviewed_files.items():
            item = Path(name)
            _need(not item.is_absolute() and ".." not in item.parts and item.suffix == ".py", "NV review file path is unsafe")
            _need(isinstance(digest, str) and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest.lower()), "NV review file hash is invalid")
        return review


def verify_nv_snapshot(snapshot: str | Path, review: NVReview) -> Path:
    """Fail closed unless *all* local Python source matches the approval record."""
    root = Path(snapshot).resolve()
    _need(root.is_dir() and not Path(snapshot).is_symlink(), "NV snapshot must be a real local directory")
    _need(root.name.endswith(review.revision), "NV snapshot directory name must end in the reviewed immutable revision")
    _need(not any(path.is_symlink() for path in root.rglob("*")), "NV snapshot must not contain symlinks")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*.py") if path.is_file()}
    _need(actual == set(review.reviewed_files), "NV Python source differs from reviewed inventory")
    for name, expected in review.reviewed_files.items():
        path = root / name
        _need(path.is_file() and _file_sha256(path) == expected, f"NV reviewed source hash mismatch: {name}")
    for required in ("config.json", "tokenizer_config.json"):
        _need((root / required).is_file(), f"NV snapshot is missing {required}")
    _need(any((root / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")), "NV snapshot lacks a tokenizer vocabulary")
    return root


def _activate_nv_offline(snapshot: str | Path, review: NVReview) -> Path:
    root = verify_nv_snapshot(snapshot, review)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _need(os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "unable to activate NV offline guards")
    return root


@dataclass(frozen=True)
class EncoderModelSpec:
    backbone: str  # qwen3_embedding | nv_embed_v2
    model_id: str
    revision: str
    tokenizer_revision: str
    model_path: str
    pooling: str
    normalize_embeddings: bool
    lora_target_modules: tuple[str, ...]
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    nv_snapshot_dir: str | None = None
    nv_review: NVReview | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EncoderModelSpec":
        required = {"backbone", "model_id", "revision", "tokenizer_revision", "model_path", "pooling", "normalize_embeddings", "lora_target_modules", "lora_r", "lora_alpha", "lora_dropout", "nv_snapshot_dir", "nv_review"}
        _need(set(raw) == required, "encoder model specification has unknown or missing fields")
        _need(isinstance(raw["normalize_embeddings"], bool), "normalize_embeddings must be a boolean")
        _need(isinstance(raw["lora_target_modules"], list) and all(isinstance(name, str) for name in raw["lora_target_modules"]), "LoRA targets must be a string list")
        review = NVReview.from_mapping(raw["nv_review"]) if raw["nv_review"] is not None else None
        spec = cls(
            backbone=raw["backbone"], model_id=raw["model_id"], revision=raw["revision"], tokenizer_revision=raw["tokenizer_revision"],
            model_path=raw["model_path"], pooling=raw["pooling"], normalize_embeddings=raw["normalize_embeddings"],
            lora_target_modules=tuple(raw["lora_target_modules"]), lora_r=raw["lora_r"], lora_alpha=raw["lora_alpha"], lora_dropout=raw["lora_dropout"],
            nv_snapshot_dir=raw["nv_snapshot_dir"], nv_review=review,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        _need(self.backbone in {"qwen3_embedding", "nv_embed_v2"}, "unsupported encoder backbone")
        _need(self.model_id == (QWEN3_MODEL_ID if self.backbone == "qwen3_embedding" else NV_MODEL_ID), "backbone/model ID mismatch")
        _need(_immutable_revision(self.revision) and _immutable_revision(self.tokenizer_revision), "model and tokenizer revisions must be immutable")
        _need(isinstance(self.model_path, str) and bool(self.model_path), "model_path is required")
        _need(self.pooling == ("last_nonpad" if self.backbone == "qwen3_embedding" else "remote_sentence_embedding"), "pooling policy differs from frozen backbone contract")
        _need(self.normalize_embeddings, "embedding normalization must be explicitly enabled")
        _need(bool(self.lora_target_modules) and len(set(self.lora_target_modules)) == len(self.lora_target_modules) and set(self.lora_target_modules) <= LORA_TARGETS, "LoRA targets must be reviewed Mistral/Qwen projection leaves")
        _need(isinstance(self.lora_r, int) and self.lora_r > 0 and isinstance(self.lora_alpha, int) and self.lora_alpha > 0 and isinstance(self.lora_dropout, float) and 0 <= self.lora_dropout < 1, "invalid LoRA hyperparameters")
        if self.backbone == "nv_embed_v2":
            _need(isinstance(self.nv_snapshot_dir, str) and self.nv_review is not None, "NV requires a local reviewed snapshot")
            _need(self.nv_review.model_id == self.model_id and self.nv_review.revision == self.revision and self.tokenizer_revision == self.revision, "NV review/revision mismatch")
        else:
            _need(self.nv_snapshot_dir is None and self.nv_review is None, "Qwen must not enable NV remote code")


def _last_nonpad(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch
    _need(getattr(last_hidden_state, "ndim", None) == 3 and getattr(attention_mask, "ndim", None) == 2, "invalid Qwen hidden-state shape")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
    final = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    _need(bool((final >= 0).all().item()), "encoder examples must contain at least one token")
    return last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), final]


def _nv_sentence_embeddings(outputs: Any, batch_size: int) -> Any:
    embedding = outputs.get("sentence_embeddings") if isinstance(outputs, Mapping) else getattr(outputs, "sentence_embeddings", None)
    _need(embedding is not None and getattr(embedding, "ndim", None) == 2 and tuple(embedding.shape) == (batch_size, NV_EMBEDDING_DIM), "NV must return sentence_embeddings [batch, 4096]")
    return embedding


def _validate_targets(model: Any, targets: Sequence[str]) -> None:
    leaves = {name.rsplit(".", maxsplit=1)[-1] for name, _ in model.named_modules()}
    missing = sorted(set(targets) - leaves)
    _need(not missing, "configured LoRA target modules are absent: " + ", ".join(missing))


def build_encoder_regressor(spec: EncoderModelSpec, score_fields: Sequence[str] = SCORE_FIELDS) -> Any:
    """Build the PEFT model and regression head for a Hugging Face ``Trainer``."""
    spec.validate()
    fields = tuple(score_fields)
    _need(bool(fields) and len(set(fields)) == len(fields) and set(fields) <= set(SCORE_FIELDS), "regressor requires unique canonical score fields")
    local_path = Path(spec.model_path).resolve()
    _need(local_path.is_dir() and not Path(spec.model_path).is_symlink(), "model_path must be an existing local non-symlink snapshot")
    if spec.backbone == "nv_embed_v2":
        assert spec.nv_snapshot_dir is not None and spec.nv_review is not None
        reviewed = _activate_nv_offline(spec.nv_snapshot_dir, spec.nv_review)
        _need(local_path == reviewed, "NV model_path must equal the reviewed snapshot")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:  # pragma: no cover - runtime-only imports
        raise RuntimeError("standard encoder requires torch, Transformers, and PEFT") from exc

    if spec.backbone == "nv_embed_v2":
        config = AutoConfig.from_pretrained(str(local_path), revision=spec.revision, trust_remote_code=True, local_files_only=True)
        text_config = getattr(config, "text_config", None)
        _need(text_config is not None, "NV remote config must expose text_config")
        setattr(text_config, "_name_or_path", str(local_path))
        _need(getattr(text_config, "_name_or_path", None) == str(local_path), "NV tokenizer source was not pinned locally")
        backbone = AutoModel.from_pretrained(str(local_path), revision=spec.revision, config=config, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    else:
        backbone = AutoModel.from_pretrained(str(local_path), revision=spec.revision, trust_remote_code=False, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    _validate_targets(backbone, spec.lora_target_modules)
    peft = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=spec.lora_r, lora_alpha=spec.lora_alpha, lora_dropout=spec.lora_dropout, target_modules=list(spec.lora_target_modules), bias="none")
    backbone = get_peft_model(backbone, peft)
    hidden = getattr(backbone.config, "hidden_size", None)
    _need(isinstance(hidden, int) and hidden > 0, "encoder config lacks hidden_size")

    class FourScoreRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.regression_head = nn.Linear(hidden, len(fields))

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None, **_: Any) -> Mapping[str, Any]:
            if spec.pooling == "remote_sentence_embedding":
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, pool_mask=attention_mask, return_dict=True)
                embedding = _nv_sentence_embeddings(outputs, int(input_ids.shape[0]))
            else:
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                embedding = _last_nonpad(outputs.last_hidden_state, attention_mask)
            logits = self.regression_head(F.normalize(embedding, p=2, dim=-1).float())
            result: dict[str, Any] = {"logits": logits}
            if labels is not None:
                _need(tuple(labels.shape[-1:]) == (len(fields),), "labels shape does not match configured score fields")
                result["loss"] = F.mse_loss(logits, labels.float(), reduction="mean")
            return result

    return FourScoreRegressor()


def build_encoder_tokenizer(spec: EncoderModelSpec) -> Any:
    """Load the pinned local tokenizer; NV guards run before remote code import."""
    spec.validate()
    local_path = Path(spec.model_path).resolve()
    if spec.backbone == "nv_embed_v2":
        assert spec.nv_snapshot_dir is not None and spec.nv_review is not None
        reviewed = _activate_nv_offline(spec.nv_snapshot_dir, spec.nv_review)
        _need(local_path == reviewed, "NV model_path must equal the reviewed snapshot")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("standard encoder requires Transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_path), revision=spec.tokenizer_revision,
        trust_remote_code=spec.backbone == "nv_embed_v2", local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        _need(tokenizer.eos_token_id is not None, "tokenizer requires EOS padding token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
