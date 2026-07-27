"""Integer, three-axis AI-Hub pretraining for Qwen3-Embedding-8B.

This module is intentionally isolated from the historical four-target
continuous regressor.  It reads only ``selection_train``, ``selection_dev``,
and ``refit_train`` from the canonical AI-Hub manifest, projects each analytic
axis with deterministic Decimal ``ROUND_HALF_UP``, and never indexes the
source ``average`` member.

The required arm is genuine full-parameter backbone tuning.  Its refit exports
the complete Qwen3 backbone plus the selected three-axis ``score_head``.  A
downstream MAL2026 arm streams the full backbone tensors into a fresh model,
then attaches a new LoRA adapter; it does not mistake AI-Hub LoRA pretraining
for full tuning.  Historical continuous four-axis state is not an input to
this training lifecycle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

from .official_score_prompt import (
    LEGACY_COMPACT,
    SCORE_PROMPT_KINDS,
    embedding_input,
    provenance as score_prompt_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_MANIFEST = ROOT / "data" / "manifests" / "aihub_human_feedback_v1.json"
CANONICAL_DATA_ROOT = ROOT / "data" / "processed" / "aihub_human_feedback_v1"
OUTPUT_ROOT = ROOT / "outputs" / "official-aihub-integer-score-full-pretrain-v1"
AXES = ("content", "organization", "expression")
HEADS = ("bounded_regression", "ordinal_cumulative")
MODEL_ID = "Qwen/Qwen3-Embedding-8B"
MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
STATE_SCHEMA = "mal2026-aihub-integer-score-full-state-v2"
COMPLETION_SCHEMA = "mal2026-aihub-integer-score-pretrain-completion-v2"


class AIHubIntegerScoreError(ValueError):
    """Raised when the integer-score experiment contract is violated."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AIHubIntegerScoreError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_bytes(tensor: Any) -> bytes:
    """Return dtype-agnostic raw tensor bytes (NumPy lacks bfloat16)."""
    import torch
    value = tensor.detach().cpu().contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def official_half_up(value: Any) -> int:
    """Project one source decimal deterministically to the official 1--5 integer."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise AIHubIntegerScoreError("analytic score must be numeric")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as exc:
        raise AIHubIntegerScoreError("analytic score is not decimal-compatible") from exc
    _need(decimal.is_finite() and Decimal("1") <= decimal <= Decimal("5"), "analytic score is outside [1,5]")
    projected = int(decimal.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    _need(1 <= projected <= 5, "integer projection is outside [1,5]")
    return projected


def project_three_axes(scores: Mapping[str, Any]) -> tuple[int, int, int]:
    """Read only the three analytic axes; ``average`` is deliberately untouched."""
    _need(isinstance(scores, Mapping), "score must be an object")
    _need(set(scores) == {*AXES, "average"}, "source score keys differ from the canonical contract")
    return tuple(official_half_up(scores[axis]) for axis in AXES)  # type: ignore[return-value]


@dataclass(frozen=True)
class IntegerScoreRow:
    prompt: str
    essay: str
    labels: tuple[int, int, int]


def _nonblank(value: Any, field: str) -> str:
    _need(isinstance(value, str) and bool(value.strip()), f"{field} must be nonblank text")
    return value


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            _need(bool(line.strip()), f"blank JSONL line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AIHubIntegerScoreError(f"invalid JSONL line {line_number}") from exc
            _need(isinstance(value, dict), f"JSONL line {line_number} is not an object")
            yield value


def _manifest_split(split: str, manifest_path: Path) -> tuple[Path, int, str]:
    _need(split in {"selection_train", "selection_dev", "refit_train"}, "unsupported AI-Hub split")
    _need(manifest_path.resolve() == CANONICAL_MANIFEST.resolve(), "only the canonical AI-Hub manifest is allowed")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AIHubIntegerScoreError("canonical AI-Hub manifest is unreadable") from exc
    _need(isinstance(manifest, dict) and manifest.get("dataset_id") == "aihub_human_feedback_v1", "AI-Hub manifest identity differs")
    details = manifest.get("files", {}).get(split)
    _need(isinstance(details, dict) and set(details) == {"filename", "sha256", "record_count"}, "AI-Hub split manifest differs")
    path = CANONICAL_DATA_ROOT / details["filename"]
    _need(path.parent.resolve() == CANONICAL_DATA_ROOT.resolve() and path.is_file() and not path.is_symlink(), "AI-Hub split path is unsafe")
    _need(file_sha256(path) == details["sha256"], "AI-Hub split checksum differs")
    return path, int(details["record_count"]), str(details["sha256"])


def load_integer_split(split: str, manifest_path: Path = CANONICAL_MANIFEST) -> tuple[list[IntegerScoreRow], str]:
    """Hash-check and load one canonical private split without retaining IDs/feedback."""
    path, expected_count, digest = _manifest_split(split, manifest_path)
    rows: list[IntegerScoreRow] = []
    for raw in _jsonl(path):
        _need(set(raw) == {"id", "prompt", "essay", "score", "feedback"}, "AI-Hub source row schema differs")
        rows.append(IntegerScoreRow(
            prompt=_nonblank(raw["prompt"], "prompt"),
            essay=_nonblank(raw["essay"], "essay"),
            labels=project_three_axes(raw["score"]),
        ))
    _need(len(rows) == expected_count and rows, "AI-Hub split record count differs")
    return rows, digest


def render_input(row: IntegerScoreRow, score_prompt_kind: str = LEGACY_COMPACT) -> str:
    """Render a reproducible legacy or public-spec score-only encoder input."""
    return embedding_input(row.prompt, row.essay, score_prompt_kind)


def ordinal_targets(labels: Any) -> Any:
    import torch
    values = labels.long().unsqueeze(-1)
    thresholds = torch.arange(1, 5, device=labels.device).view(1, 1, 4)
    return (values > thresholds).float()


def decode_logits(logits: Any, head: str) -> tuple[Any, Any, Any]:
    """Return continuous values, half-up integers, and ordinal violations."""
    import torch
    _need(head in HEADS, "unknown score head")
    if head == "bounded_regression":
        _need(logits.ndim == 2 and logits.shape[-1] == 3, "bounded head logits must be [batch,3]")
        continuous = 1.0 + 4.0 * torch.sigmoid(logits.float())
        violations = torch.zeros(logits.shape[0], 3, 0, dtype=torch.bool, device=logits.device)
    else:
        _need(logits.ndim == 2 and logits.shape[-1] == 12, "ordinal head logits must be [batch,12]")
        probabilities = torch.sigmoid(logits.float().reshape(-1, 3, 4))
        violations = probabilities[:, :, 1:] > probabilities[:, :, :-1]
        probabilities = torch.cummin(probabilities, dim=-1).values
        continuous = 1.0 + probabilities.sum(dim=-1)
    integers = torch.floor(continuous + 0.5).clamp(1, 5).to(torch.int64)
    return continuous, integers, violations


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for position in range(cursor, end):
            result[order[position]] = rank
        cursor = end
    return result


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    ranked_left, ranked_right = _rank(left), _rank(right)
    mean_left = sum(ranked_left) / len(ranked_left)
    mean_right = sum(ranked_right) / len(ranked_right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(ranked_left, ranked_right, strict=True))
    left_norm = math.sqrt(sum((a - mean_left) ** 2 for a in ranked_left))
    right_norm = math.sqrt(sum((b - mean_right) ** 2 for b in ranked_right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def score_metrics(labels: Sequence[Sequence[float]], continuous: Sequence[Sequence[float]], integers: Sequence[Sequence[int]], violations: Sequence[Any]) -> dict[str, float]:
    _need(bool(labels) and len(labels) == len(continuous) == len(integers), "metric rows differ")
    integer_rmse, integer_rho, continuous_rmse = [], [], []
    for axis in range(3):
        target = [float(row[axis]) for row in labels]
        integer = [float(row[axis]) for row in integers]
        raw = [float(row[axis]) for row in continuous]
        integer_rmse.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(target, integer, strict=True)) / len(target)))
        continuous_rmse.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(target, raw, strict=True)) / len(target)))
        integer_rho.append(_spearman(target, integer))
    violation_count = sum(int(value) for row in violations for axis in row for value in axis)
    violation_total = sum(len(axis) for row in violations for axis in row)
    return {
        "macro_integer_rmse": sum(integer_rmse) / 3,
        "macro_integer_spearman": sum(integer_rho) / 3,
        "macro_continuous_rmse": sum(continuous_rmse) / 3,
        "ordinal_violation_rate": violation_count / violation_total if violation_total else 0.0,
    }


def select_event(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    _need(bool(events), "selection emitted no evaluation events")
    return min(events, key=lambda row: (
        float(row["macro_integer_rmse"]),
        -float(row["macro_integer_spearman"]),
        float(row["macro_continuous_rmse"]),
        int(row["global_step"]),
    ))


@dataclass(frozen=True)
class PretrainConfig:
    schema_version: str
    run_id: str
    model_id: str
    model_revision: str
    model_path: str
    manifest_path: str
    output_root: str
    score_fields: tuple[str, str, str]
    integer_target_used: bool
    target_projection: str
    average_target_used: bool
    heads: tuple[str, str]
    seed: int
    max_length: int
    training_method: str
    distributed_strategy: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_selection_epochs: float
    eval_steps: int
    early_stopping_patience: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    activation_checkpointing: bool
    fsdp_transformer_layer_class: str
    fsdp_state_dict_type: str
    training_dtype: str
    historical_reference_selected_step: int
    historical_reference_classification: str
    # Transformers 5.14 defaults to FSDP2.  Adafactor in the pinned
    # Transformers/PyTorch stack cannot update DTensor parameters, so the
    # repaired lineage explicitly selects the still-supported FSDP1 backend.
    fsdp_version: int = 2
    score_prompt_kind: str = LEGACY_COMPACT

    @classmethod
    def from_json(cls, path: Path, *, require_dependencies: bool = True) -> "PretrainConfig":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AIHubIntegerScoreError("pretraining config is unreadable") from exc
        _need(isinstance(raw, dict), "pretraining config must be an object")
        # Configs from the two preserved failed lineages predate the explicit
        # version field and therefore reproduce Transformers' FSDP2 default.
        raw.setdefault("fsdp_version", 2)
        raw.setdefault("score_prompt_kind", LEGACY_COMPACT)
        for field in ("score_fields", "heads"):
            _need(isinstance(raw.get(field), list), f"{field} must be a list")
            raw[field] = tuple(raw[field])
        _need(set(raw) == set(cls.__dataclass_fields__), "pretraining config has missing or unknown fields")
        config = cls(**raw)
        config.validate(require_dependencies=require_dependencies)
        return config

    def validate(self, *, require_dependencies: bool = True) -> None:
        _need(self.schema_version == "mal2026-aihub-integer-score-pretrain-config-v2", "config schema differs")
        _need(self.model_id == MODEL_ID and self.model_revision == MODEL_REVISION, "Qwen3 embedding pin differs")
        _need(self.score_fields == AXES and self.heads == HEADS, "three-axis/head contract differs")
        _need(self.integer_target_used is True, "integer_target_used must be true")
        _need(self.target_projection == "official_half_up", "target projection differs")
        _need(self.average_target_used is False, "average_target_used must be false")
        _need(Path(self.manifest_path).resolve() == CANONICAL_MANIFEST.resolve(), "canonical AI-Hub manifest is required")
        _need(Path(self.output_root).resolve() == OUTPUT_ROOT.resolve(), "output root differs")
        _need(self.seed == 2026 and self.max_length == 2048, "historical seed/length provenance differs")
        _need(self.training_method == "full_parameter", "required AI-Hub arm must tune the full backbone")
        _need(self.distributed_strategy == "fsdp_full_shard_auto_wrap", "full training must use the declared FSDP strategy")
        _need(self.fsdp_version in {1, 2}, "FSDP version differs")
        _need(self.optimizer == "adafactor", "memory-prudent full-parameter optimizer differs")
        _need(self.learning_rate == 1e-5 and self.weight_decay == 0.01 and self.warmup_ratio == 0.05, "optimization provenance differs")
        _need(self.max_selection_epochs == 20.0 and self.eval_steps == 100 and self.early_stopping_patience == 3, "selection schedule differs")
        _need((self.per_device_train_batch_size, self.per_device_eval_batch_size, self.gradient_accumulation_steps) == (1, 1, 8), "batch schedule differs")
        _need(self.activation_checkpointing is True, "full tuning requires activation checkpointing")
        _need(self.fsdp_transformer_layer_class == "Qwen3DecoderLayer" and self.fsdp_state_dict_type == "FULL_STATE_DICT", "FSDP save/wrap contract differs")
        _need(self.training_dtype == "bfloat16", "training dtype differs")
        _need(self.score_prompt_kind in SCORE_PROMPT_KINDS, "score prompt kind differs")
        _need(self.historical_reference_selected_step == 1900, "historical provenance step differs")
        _need(self.historical_reference_classification == "reference_only_continuous_four_axis_not_loaded", "historical warmstate must remain reference-only")
        if require_dependencies:
            model = Path(self.model_path)
            _need(model.is_dir() and not model.is_symlink(), "local Qwen3 embedding snapshot is unavailable")
            _need(Path(self.manifest_path).is_file(), "canonical AI-Hub manifest is unavailable")

    def identity(self, head: str) -> dict[str, Any]:
        _need(head in self.heads, "head is outside config")
        raw = asdict(self)
        for key in ("run_id", "output_root", "heads", "historical_reference_selected_step", "historical_reference_classification"):
            raw.pop(key)
        # Identity is persisted as JSON and read back for the fresh refit.
        # Normalize tuple-valued config fields before comparison so an
        # otherwise identical selection/refit pair is not rejected merely
        # because JSON arrays deserialize as lists.
        raw["score_fields"] = list(raw["score_fields"])
        raw["head"] = head
        return raw


def downstream_target_contract(config: PretrainConfig) -> dict[str, Any]:
    """Exact target semantics embedded in every downstream-visible artifact."""
    config.validate(require_dependencies=False)
    return {
        "integer_target_used": True,
        "target_projection": "official_half_up",
        "score_fields": list(AXES),
        "average_target_used": False,
    }


def _dataset(
    rows: Sequence[IntegerScoreRow],
    tokenizer: Any,
    max_length: int,
    score_prompt_kind: str = LEGACY_COMPACT,
) -> Any:
    from datasets import Dataset
    dataset = Dataset.from_dict({
        "text": [render_input(row, score_prompt_kind) for row in rows],
        "labels": [list(row.labels) for row in rows],
    })
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])


def _collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        labels = [feature.pop("labels") for feature in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch
    return collate


def build_model(config: PretrainConfig, head: str) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from transformers import AutoModel

    _need(head in config.heads, "unknown head")
    base = AutoModel.from_pretrained(
        config.model_path, revision=config.model_revision, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    backbone = base
    hidden = getattr(backbone.config, "hidden_size", None)
    _need(type(hidden) is int and hidden > 0, "embedding hidden size is unavailable")

    class IntegerScoreModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            # FSDP1 flattens each wrapper's managed parameters before its
            # mixed-precision policy is applied, so the root head must already
            # match the BF16 backbone dtype.
            self.score_head = nn.Linear(
                hidden, 3 if head == "bounded_regression" else 12,
                dtype=next(backbone.parameters()).dtype,
            )

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Mapping[str, Any] | None = None) -> None:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        def gradient_checkpointing_disable(self) -> None:
            self.backbone.gradient_checkpointing_disable()

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None, **_: Any) -> Mapping[str, Any]:
            hidden_state = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
            positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
            final = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
            _need(bool((final >= 0).all().item()), "encoder input has no non-padding token")
            pooled = functional.normalize(
                hidden_state[torch.arange(hidden_state.shape[0], device=hidden_state.device), final],
                p=2,
                dim=-1,
            )
            # FSDP's BF16 mixed policy also casts this non-wrapped head.  The
            # single-GPU smoke leaves it in FP32, so explicitly follow the
            # live parameter dtype for GEMM and expose FP32 logits/loss.
            logits = self.score_head(pooled.to(self.score_head.weight.dtype)).float()
            result: dict[str, Any] = {"logits": logits}
            if labels is not None:
                _need(labels.ndim == 2 and labels.shape[-1] == 3, "labels must be [batch,3]")
                if head == "bounded_regression":
                    prediction, _, _ = decode_logits(logits, head)
                    result["loss"] = functional.mse_loss(prediction, labels.float())
                else:
                    result["loss"] = functional.binary_cross_entropy_with_logits(logits.reshape(-1, 3, 4), ordinal_targets(labels))
            return result

    model = IntegerScoreModel()
    _need(all(parameter.requires_grad for parameter in model.backbone.parameters()), "full backbone contains frozen parameters")
    _need(not any("lora_" in name for name, _ in model.named_parameters()), "LoRA leaked into the required full-parameter arm")
    return model


def _initial_head_sha256(model: Any) -> str:
    digest = sha256()
    state = {f"score_head.{name}": value.detach().cpu().contiguous() for name, value in model.score_head.state_dict().items()}
    _need(set(state) == {"score_head.weight", "score_head.bias"}, "score head state differs")
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def initialization_contract_sha256(config: PretrainConfig, head: str, initial_head_sha256: str) -> str:
    """Bind fresh initialization without rereading the 17 GB immutable snapshot."""
    payload = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "model_path": str(Path(config.model_path).resolve()),
        "head": head,
        "initial_head_sha256": initial_head_sha256,
        "seed": config.seed,
        "training_method": config.training_method,
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact_inventory(directory: Path) -> tuple[list[dict[str, Any]], str]:
    """Hash every regular exported model file and then hash the canonical inventory."""
    _need(directory.is_dir() and not directory.is_symlink(), "full model artifact directory is unavailable")
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            _need(not path.is_symlink(), "full model artifact contains a symlink")
            entries.append({"path": path.relative_to(directory).as_posix(), "size": path.stat().st_size, "sha256": file_sha256(path)})
    _need(bool(entries), "full model artifact is empty")
    digest = sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return entries, digest


def exported_tensor_contract(directory: Path, head: str) -> dict[str, Any]:
    """Verify full-backbone/matched-head tensor scope and hash the head alone."""
    from safetensors import safe_open

    expected_head = 3 if head == "bounded_regression" else 12
    tensor_files = sorted(directory.glob("*.safetensors"))
    _need(bool(tensor_files), "full model export contains no safetensors state")
    backbone_count = 0
    head_shapes: dict[str, list[int]] = {}
    head_tensors: dict[str, Any] = {}
    for path in tensor_files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in sorted(handle.keys()):
                if name.startswith("backbone."):
                    backbone_count += 1
                elif name.startswith("score_head."):
                    _need(name not in head_tensors, "full model export repeats a matched head tensor")
                    tensor = handle.get_tensor(name).detach().cpu().contiguous()
                    head_shapes[name] = list(tensor.shape)
                    head_tensors[name] = tensor
    _need(backbone_count > 0, "full model export contains no backbone tensors")
    _need(set(head_shapes) == {"score_head.weight", "score_head.bias"}, "full model export matched head differs")
    _need(head_shapes["score_head.weight"][0] == expected_head and head_shapes["score_head.bias"] == [expected_head], "full model export head width differs")
    head_digest = sha256()
    for name, tensor in sorted(head_tensors.items()):
        head_digest.update(name.encode())
        head_digest.update(str(tensor.dtype).encode())
        head_digest.update(json.dumps(list(tensor.shape)).encode())
        head_digest.update(_tensor_bytes(tensor))
    return {
        "backbone_tensor_count": backbone_count,
        "score_head_tensor_shapes": head_shapes,
        "score_head_state_sha256": head_digest.hexdigest(),
    }


def _compute_metrics(head: str):
    def compute(result: Any) -> dict[str, float]:
        import torch
        continuous, integers, violations = decode_logits(torch.as_tensor(result.predictions), head)
        return score_metrics(result.label_ids.tolist(), continuous.tolist(), integers.tolist(), violations.tolist())
    return compute


def _wait_for_directory(path: Path) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        _need(not path.exists(), f"refusing to reuse output: {path}")
        path.mkdir(parents=True)
        return
    deadline = time.monotonic() + 60
    while not path.is_dir() and time.monotonic() < deadline:
        time.sleep(0.05)
    _need(path.is_dir(), "rank zero did not create output directory")


def _load_selection_metadata(config: PretrainConfig, head: str, path: Path) -> tuple[int, int, str, Mapping[str, Any]]:
    _need(path.is_file() and not path.is_symlink(), "selection completion artifact is unavailable")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AIHubIntegerScoreError("selection completion artifact is unreadable") from exc
    _need(isinstance(payload, dict) and payload.get("status") == "completed" and payload.get("phase") == "selection", "artifact is not a completed selection")
    _need(payload.get("head") == head and payload.get("identity") == config.identity(head), "selection/refit identity differs")
    selected = payload.get("selection", {}).get("selected_event", {})
    step = selected.get("global_step")
    _need(type(step) is int and step > 0, "selection has no positive selected step")
    schedule_steps = selected.get("selection_schedule_max_steps")
    _need(type(schedule_steps) is int and schedule_steps >= step, "selection scheduler horizon is unavailable")
    initial_hash = payload.get("initialization_contract_sha256")
    _need(isinstance(initial_hash, str) and len(initial_hash) == 64, "selection initial-state hash differs")
    return step, schedule_steps, initial_hash, selected


def run_training(config: PretrainConfig, head: str, phase: str, *, smoke: bool = False, selection_metadata: Path | None = None) -> dict[str, Any]:
    """Run selection or refit under Trainer; canonical evaluation is unreachable."""
    config.validate(require_dependencies=True)
    _need(head in config.heads and phase in {"selection", "refit"}, "training head/phase differs")
    _need((phase == "selection") == (selection_metadata is None), "refit alone requires selection metadata")
    try:
        import torch
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("AI-Hub integer score pretraining requires .venv-standard") from exc

    rank = int(os.environ.get("RANK", "0"))
    output = Path(config.output_root) / config.run_id / (f"smoke-{head}-{phase}" if smoke else f"{head}-{phase}")
    _wait_for_directory(output)
    split = "selection_train" if phase == "selection" else "refit_train"
    train_rows, train_sha = load_integer_split(split, Path(config.manifest_path))
    dev_rows: list[IntegerScoreRow] | None = None
    dev_sha: str | None = None
    if phase == "selection":
        dev_rows, dev_sha = load_integer_split("selection_dev", Path(config.manifest_path))
    if smoke:
        train_rows = train_rows[:4]
        if dev_rows is not None:
            dev_rows = dev_rows[:4]

    selected_steps: int | None = None
    selection_schedule_steps: int | None = None
    selection_initial_hash: str | None = None
    selected_event: Mapping[str, Any] | None = None
    if phase == "refit":
        assert selection_metadata is not None
        selected_steps, selection_schedule_steps, selection_initial_hash, selected_event = _load_selection_metadata(config, head, selection_metadata)

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, revision=config.model_revision, local_files_only=True,
        trust_remote_code=False, use_fast=True,
    )
    if tokenizer.pad_token is None:
        _need(tokenizer.eos_token is not None, "tokenizer has no pad/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(config, head)
    initial_head_hash = _initial_head_sha256(model)
    initial_hash = initialization_contract_sha256(config, head, initial_head_hash)
    if phase == "refit":
        _need(initial_hash == selection_initial_hash, "selection/refit initialization replay differs")
    train_dataset = _dataset(train_rows, tokenizer, config.max_length, config.score_prompt_kind)
    eval_dataset = _dataset(dev_rows, tokenizer, config.max_length, config.score_prompt_kind) if dev_rows is not None else None
    events: list[dict[str, Any]] = []

    class SelectionCallback(TrainerCallback):
        def __init__(self) -> None:
            self.best_key: tuple[float, float, float, int] | None = None
            self.stale = 0

        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Mapping[str, Any] | None = None, **_: Any) -> Any:
            if metrics is None:
                raise AIHubIntegerScoreError("selection evaluation has no metrics")
            event = {
                "global_step": int(state.global_step),
                "selection_schedule_max_steps": int(state.max_steps),
                "epoch": float(state.epoch),
                "macro_integer_rmse": float(metrics["eval_macro_integer_rmse"]),
                "macro_integer_spearman": float(metrics["eval_macro_integer_spearman"]),
                "macro_continuous_rmse": float(metrics["eval_macro_continuous_rmse"]),
                "ordinal_violation_rate": float(metrics["eval_ordinal_violation_rate"]),
            }
            events.append(event)
            key = (event["macro_integer_rmse"], -event["macro_integer_spearman"], event["macro_continuous_rmse"], event["global_step"])
            if self.best_key is None or key < self.best_key:
                self.best_key, self.stale = key, 0
            else:
                self.stale += 1
                if self.stale >= config.early_stopping_patience:
                    control.should_training_stop = True
            return control

    class StopRefitAtSelectedStep(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            if selected_steps is not None and int(state.global_step) >= selected_steps:
                control.should_training_stop = True
            return control

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    fsdp_config = {
        "version": config.fsdp_version,
        "transformer_layer_cls_to_wrap": [config.fsdp_transformer_layer_class],
        "activation_checkpointing": config.activation_checkpointing,
        "use_orig_params": True,
        "state_dict_type": config.fsdp_state_dict_type,
        "sync_module_states": True,
        "limit_all_gathers": True,
    } if distributed else None
    args = TrainingArguments(
        output_dir=str(output / "trainer"), do_train=True, do_eval=phase == "selection",
        eval_strategy="steps" if phase == "selection" else "no", save_strategy="no",
        eval_steps=(1 if smoke else config.eval_steps) if phase == "selection" else None,
        # Refit keeps the selection scheduler horizon and stops at the chosen
        # optimizer step, reproducing the learning-rate trajectory exactly.
        max_steps=1 if smoke else (selection_schedule_steps if selection_schedule_steps is not None else -1),
        num_train_epochs=config.max_selection_epochs if phase == "selection" else 1.0,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        optim=config.optimizer,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=1 if smoke else config.gradient_accumulation_steps,
        bf16=True, tf32=True, report_to=[], remove_unused_columns=False,
        # Single-GPU smoke uses the model hook; FSDP full runs use the plugin's
        # layer wrapping. Enabling both would checkpoint each block twice.
        gradient_checkpointing=config.activation_checkpointing and not distributed,
        gradient_checkpointing_kwargs={"use_reentrant": False} if not distributed else None,
        fsdp="full_shard auto_wrap" if distributed else None,
        fsdp_config=fsdp_config,
        dataloader_num_workers=0, dataloader_pin_memory=True, ddp_find_unused_parameters=False,
        logging_steps=1 if smoke else 5, save_only_model=True, seed=config.seed, data_seed=config.seed,
    )
    callbacks = [SelectionCallback()] if phase == "selection" else [StopRefitAtSelectedStep()]
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=_collator(tokenizer), compute_metrics=_compute_metrics(head) if phase == "selection" else None,
        callbacks=callbacks,
    )
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()

    shared_events: list[Any] = [events if trainer.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(shared_events, src=0)
    events = shared_events[0]
    _need(isinstance(events, list), "selection event broadcast differs")
    if phase == "selection":
        selected_event = select_event(events)
    else:
        _need(int(trainer.state.global_step) == selected_steps, "refit did not reproduce selected global steps")

    payload: dict[str, Any] = {
        "schema_version": COMPLETION_SCHEMA,
        "status": "completed",
        "mode": "gpu0_one_update_smoke" if smoke else "full",
        "run_id": config.run_id,
        "phase": phase,
        "head": head,
        "identity": config.identity(head),
        **downstream_target_contract(config),
        **score_prompt_provenance(config.score_prompt_kind),
        "integer_projection": "Decimal ROUND_HALF_UP to 1..5",
        "average_read": False,
        "canonical_validation_access": False,
        "data": {"split": split, "records": len(train_rows), "sha256": train_sha, "selection_dev_sha256": dev_sha},
        "training_method": "full_parameter",
        "backbone_parameter_policy": "all_backbone_parameters_trainable",
        "initial_head_sha256": initial_head_hash,
        "initialization_contract_sha256": initial_hash,
        "selection": {"events": events, "selected_event": selected_event, "selection_source": "AI-Hub selection_dev only"},
        "trainer": {
            "fsdp_version": config.fsdp_version if distributed else None,
            "global_step": int(trainer.state.global_step),
            "scheduler_horizon_steps": int(trainer.state.max_steps),
            "exact_selected_step_stop": phase == "refit",
            "train_metrics": {k: float(v) for k, v in trained.metrics.items() if isinstance(v, (int, float))},
        },
        "historical_reference": {"selected_global_step": config.historical_reference_selected_step, "classification": config.historical_reference_classification, "loaded": False},
        "privacy": "aggregate_only_no_rows_prompts_essays_feedback_ids_predictions_or_model_outputs_persisted",
    }
    if phase == "refit":
        if smoke:
            payload["state"] = {
                "schema_version": STATE_SCHEMA,
                "training_method": "full_parameter",
                "export_skipped": True,
                "reason": "one_update_smoke_avoids_a_redundant_17GB_full_state_export",
            }
        else:
            model_path = output / "full_model"
            metadata_path = output / "full_model_state.json"
            state_metadata = {
                "schema_version": STATE_SCHEMA,
                "head": head,
                **downstream_target_contract(config),
                **score_prompt_provenance(config.score_prompt_kind),
                "model_id": config.model_id,
                "model_revision": config.model_revision,
                "training_method": "full_parameter",
                "state_scope": "complete_full_parameter_backbone_plus_matched_score_head",
                "compatible_load_modes": {
                    "matched_full": "requires exact head, axes, model revision, and tensor shapes",
                    "full_backbone_and_matched_head_then_mal_lora": "stream all backbone.* tensors and the matched score_head.* into a fresh Qwen3 score model, then attach a fresh MAL2026 LoRA; train the LoRA and retained matched head",
                },
                "forbidden_primary_source": "historical_continuous_four_axis_warmstate",
            }
            # All ranks participate because FSDP must gather a FULL_STATE_DICT.
            trainer.save_model(str(model_path))
            trainer.accelerator.wait_for_everyone()
            if trainer.is_world_process_zero():
                inventory, artifact_sha = artifact_inventory(model_path)
                state_metadata.update({
                    "artifact_path": str(model_path.resolve()),
                    "artifact_sha256": artifact_sha,
                    "inventory": inventory,
                    **exported_tensor_contract(model_path, head),
                })
                metadata_path.write_text(json.dumps(state_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                payload["state"] = {
                    **state_metadata,
                    "metadata_path": str(metadata_path.resolve()),
                    "metadata_sha256": file_sha256(metadata_path),
                }
            shared_state: list[Any] = [payload.get("state") if trainer.is_world_process_zero() else None]
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(shared_state, src=0)
            payload["state"] = shared_state[0]

    completion = output / "training_complete.json"
    if trainer.is_world_process_zero():
        _need(not completion.exists(), "completion artifact already exists")
        completion.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trainer.accelerator.wait_for_everyone()
    return payload
