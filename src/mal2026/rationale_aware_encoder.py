"""Rationale-aware continuous three-axis score encoders for MAL2026.

Both supported encoders start from a complete AI-Hub-trained
backbone-plus-three-score-head state, attach a fresh LoRA, and then learn from
the 2,000 MAL2026 training essays.  A deterministic train-internal split is
the only source used for epoch selection.  Frozen validation is loaded once,
after refit, and is reported both with the aligned rationale bundle and with a
deterministically shuffled bundle as a reliance diagnostic.

The source ``score.average`` value is deliberately never indexed.  MAL labels
remain continuous; no integer projection is applied to supervision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

from .official_score_matrix import AXES, decode_logits, file_sha256, score_metrics
from .official_score_prompt import (
    RATIONALE_AWARE_ENCODER,
    embedding_input,
    provenance as prompt_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/rationale-aware-encoder-v1"
RESTRICTED_ROOT = (ROOT / "data/processed/restricted").resolve()
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "qwen3_embedding_8b": {
        "model_id": "Qwen/Qwen3-Embedding-8B",
        "model_revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "pooling": "last_nonpadding",
        "hidden_size": 4096,
        "lora_targets": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    },
    "kure_v1": {
        "model_id": "nlpai-lab/KURE-v1",
        "model_revision": "d14c8a9423946e268a0c9952fecf3a7aabd73bd9",
        "pooling": "cls",
        "hidden_size": 1024,
        "lora_targets": ("query", "key", "value", "dense"),
    },
}


class RationaleAwareEncoderError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RationaleAwareEncoderError(message)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RationaleAwareEncoderError(f"{label} is unreadable") from exc
    need(isinstance(value, dict), f"{label} must be an object")
    return value


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            need(bool(line.strip()), f"blank JSONL line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RationaleAwareEncoderError(f"invalid JSONL line {line_number}") from exc
            need(isinstance(value, dict), f"JSONL line {line_number} must be an object")
            yield value


@dataclass(frozen=True)
class ContinuousScoreRow:
    identifier: str
    document_id: str
    prompt_num: str
    prompt: str
    essay: str
    labels: tuple[float, float, float]


@dataclass(frozen=True)
class RationaleEncoderConfig:
    schema_version: str
    run_id: str
    model_key: str
    model_id: str
    model_revision: str
    model_path: str
    train_path: str
    train_sha256: str
    validation_path: str
    validation_sha256: str
    rationale_key: str
    rationale_train_path: str
    rationale_train_sha256: str
    rationale_validation_path: str
    rationale_validation_sha256: str
    rationale_manifest_path: str
    rationale_manifest_sha256: str
    warmstart_completion_path: str
    warmstart_completion_sha256: str
    warmstart_artifact_path: str
    warmstart_artifact_sha256: str
    output_root: str
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
    def from_json(cls, path: Path, *, require_dependencies: bool = True) -> "RationaleEncoderConfig":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RationaleAwareEncoderError("rationale encoder config is unreadable") from exc
        need(isinstance(raw, dict), "rationale encoder config must be an object")
        for field in ("score_fields", "selection_epochs"):
            need(isinstance(raw.get(field), list), f"{field} must be a list")
            raw[field] = tuple(raw[field])
        need(set(raw) == set(cls.__dataclass_fields__), "rationale encoder config fields differ")
        value = cls(**raw)
        value.validate(require_dependencies=require_dependencies)
        return value

    def validate(self, *, require_dependencies: bool = True) -> None:
        need(self.schema_version == "mal2026-rationale-aware-encoder-v1", "rationale encoder schema differs")
        need(self.model_key in MODEL_SPECS, "rationale encoder model key differs")
        spec = MODEL_SPECS[self.model_key]
        need((self.model_id, self.model_revision) == (spec["model_id"], spec["model_revision"]), "model pin differs")
        expected_run = {
            "qwen3_embedding_8b": "rationale-aware-qwen3-embedding-8b-aihub-mal-v1-20260729-002",
            "kure_v1": "rationale-aware-kure-v1-aihub-mal-v1-20260729-002",
        }[self.model_key]
        need(self.run_id == expected_run, "rationale encoder run ID differs")
        need(Path(self.output_root).resolve() == OUTPUT_ROOT.resolve(), "rationale encoder output root differs")
        need(self.score_fields == AXES and self.average_target_used is False, "rationale encoder axes differ")
        need(self.target_projection == "none_preserve_raw_continuous", "MAL target projection differs")
        need(self.score_prompt_kind == RATIONALE_AWARE_ENCODER, "rationale-aware prompt differs")
        need(self.selection_epochs == tuple(range(1, 9)), "rationale encoder selection schedule differs")
        expected_length = 2304 if self.model_key == "qwen3_embedding_8b" else 2048
        need(self.max_length == expected_length, "rationale encoder max length differs")
        need(self.seed == 2026072903, "rationale encoder seed differs")
        need((self.learning_rate, self.weight_decay, self.warmup_ratio) == (1e-4, 0.01, 0.05), "MAL optimizer differs")
        expected_batch = (4, 8, 2) if self.model_key == "qwen3_embedding_8b" else (8, 16, 2)
        need(
            (self.per_device_train_batch_size, self.per_device_eval_batch_size, self.gradient_accumulation_steps) == expected_batch,
            "MAL batch contract differs",
        )
        need((self.lora_r, self.lora_alpha, self.lora_dropout) == (16, 32, 0.05), "MAL LoRA contract differs")
        expected_dtype = "bfloat16" if self.model_key == "qwen3_embedding_8b" else "float32"
        need(self.training_dtype == expected_dtype, "MAL dtype differs")
        if not require_dependencies:
            return
        for raw_path, expected_sha, label, directory in (
            (self.model_path, None, "model snapshot", True),
            (self.train_path, self.train_sha256, "canonical train", False),
            (self.validation_path, self.validation_sha256, "canonical validation", False),
            (self.rationale_train_path, self.rationale_train_sha256, "train rationales", False),
            (self.rationale_validation_path, self.rationale_validation_sha256, "validation rationales", False),
            (self.rationale_manifest_path, self.rationale_manifest_sha256, "rationale manifest", False),
            (self.warmstart_completion_path, self.warmstart_completion_sha256, "warmstart completion", False),
        ):
            path = Path(raw_path)
            need((path.is_dir() if directory else path.is_file()) and not path.is_symlink(), f"{label} is unavailable")
            if expected_sha is not None:
                need(len(expected_sha) == 64 and file_sha256(path) == expected_sha, f"{label} checksum differs")
        for rationale_path in (Path(self.rationale_train_path), Path(self.rationale_validation_path), Path(self.rationale_manifest_path)):
            need(rationale_path.resolve().is_relative_to(RESTRICTED_ROOT), "rationale path escaped restricted storage")
        artifact = Path(self.warmstart_artifact_path)
        need(artifact.is_dir() and not artifact.is_symlink(), "warmstart artifact is unavailable")
        need(len(self.warmstart_artifact_sha256) == 64, "warmstart artifact checksum is unresolved")
        handoff = read_json(Path(self.rationale_manifest_path), "rationale handoff")
        need(handoff.get("schema_version") == "mal2026-official-rationale-score-matrix-handoff-v1", "rationale handoff schema differs")
        need(handoff.get("status") == "completed" and handoff.get("structure") == "bundle", "only completed bundle rationales are allowed")
        need(handoff.get("axis_triplet_used_for_training_or_selection") is False, "axis-triplet lineage is forbidden")
        need(handoff.get("rationale_key") == self.rationale_key, "rationale key differs")
        need(handoff.get("rationale_train_sha256") == self.rationale_train_sha256, "train rationale handoff differs")
        need(handoff.get("rationale_validation_sha256") == self.rationale_validation_sha256, "validation rationale handoff differs")
        need(handoff.get("human_or_reference_score_read_or_prompted") is False, "rationale generator read a protected score")
        completion = read_json(Path(self.warmstart_completion_path), "warmstart completion")
        need(completion.get("status") == "completed" and completion.get("training_method") == "full_parameter", "warmstart is not completed full tuning")
        need(completion.get("score_fields") == list(AXES), "warmstart score fields differ")
        need(completion.get("average_target_used") is False, "warmstart used an average target")
        state = completion.get("state")
        need(isinstance(state, dict), "warmstart state differs")
        need(Path(state.get("artifact_path", "")).resolve() == artifact.resolve(), "warmstart artifact path differs")
        need(state.get("artifact_sha256") == self.warmstart_artifact_sha256, "warmstart artifact digest differs")


def load_continuous_rows(path: Path, expected_sha: str, expected_count: int) -> list[ContinuousScoreRow]:
    """Load only content/organization/expression and preserve decimals."""
    need(path.is_file() and file_sha256(path) == expected_sha, "canonical MAL source differs")
    rows: list[ContinuousScoreRow] = []
    seen: set[str] = set()
    for raw in jsonl(path):
        need(set(raw) == {"id", "document_id", "prompt_num", "prompt", "essay", "score"}, "canonical MAL row schema differs")
        identifier = raw["id"]
        need(isinstance(identifier, str) and identifier and identifier not in seen, "canonical MAL ID differs")
        seen.add(identifier)
        scores = raw["score"]
        need(isinstance(scores, dict) and set(scores) == {*AXES, "average"}, "canonical MAL score keys differ")
        # Do not access scores["average"].
        labels: list[float] = []
        for axis in AXES:
            value = scores[axis]
            need(type(value) in {int, float} and not isinstance(value, bool), f"{axis} target is nonnumeric")
            value = float(value)
            need(math.isfinite(value) and 1.0 <= value <= 5.0, f"{axis} target is outside [1,5]")
            labels.append(value)
        need(isinstance(raw["prompt"], str) and raw["prompt"].strip(), "MAL prompt is blank")
        need(isinstance(raw["essay"], str) and raw["essay"].strip(), "MAL essay is blank")
        rows.append(ContinuousScoreRow(
            identifier=identifier, document_id=str(raw["document_id"]), prompt_num=str(raw["prompt_num"]),
            prompt=raw["prompt"], essay=raw["essay"], labels=tuple(labels),  # type: ignore[arg-type]
        ))
    need(len(rows) == expected_count, "canonical MAL record count differs")
    return rows


def deterministic_split(rows: Sequence[ContinuousScoreRow], seed: int) -> tuple[list[ContinuousScoreRow], list[ContinuousScoreRow], str]:
    need(len(rows) == 2000, "MAL internal split requires 2,000 records")
    groups: dict[tuple[str, str], ContinuousScoreRow] = {}
    for row in rows:
        key = (row.prompt_num, row.document_id)
        need(key not in groups, "MAL grouping key is not unique")
        groups[key] = row
    counts: dict[str, int] = {}
    for prompt_num, _ in groups:
        counts[prompt_num] = counts.get(prompt_num, 0) + 1
    quotas = {key: int(value * 0.2) for key, value in counts.items()}
    remainder = 400 - sum(quotas.values())
    ranking = sorted(counts, key=lambda key: (-(counts[key] * 0.2 - quotas[key]), sha256(f"{seed}\0{key}".encode()).hexdigest()))
    for key in ranking[:remainder]:
        quotas[key] += 1
    dev_keys: set[tuple[str, str]] = set()
    for prompt_num, quota in quotas.items():
        candidates = sorted(
            (key for key in groups if key[0] == prompt_num),
            key=lambda key: sha256(f"{seed}\0{key[0]}\0{key[1]}".encode()).hexdigest(),
        )
        dev_keys.update(candidates[:quota])
    train = [row for row in rows if (row.prompt_num, row.document_id) not in dev_keys]
    dev = [row for row in rows if (row.prompt_num, row.document_id) in dev_keys]
    need((len(train), len(dev)) == (1600, 400), "MAL internal split size differs")
    fingerprint = sha256("\n".join(sorted(sha256(f"{row.prompt_num}\0{row.document_id}".encode()).hexdigest() for row in dev)).encode()).hexdigest()
    return train, dev, fingerprint


def load_rationales(path: Path, expected_sha: str, rows: Sequence[ContinuousScoreRow]) -> dict[str, dict[str, str]]:
    need(path.is_file() and file_sha256(path) == expected_sha, "rationale source differs")
    expected = {row.identifier for row in rows}
    result: dict[str, dict[str, str]] = {}
    for raw in jsonl(path):
        need(set(raw) >= {"source_id", "rationales"}, "rationale row schema differs")
        source_id = raw["source_id"]
        rationales = raw["rationales"]
        need(source_id in expected and source_id not in result, "rationale linkage differs")
        need(isinstance(rationales, dict) and set(rationales) == set(AXES), "rationale bundle axes differ")
        need(all(isinstance(rationales[axis], str) and rationales[axis].strip() for axis in AXES), "rationale bundle text differs")
        result[source_id] = {axis: rationales[axis] for axis in AXES}
    need(set(result) == expected, "rationale population is incomplete")
    return result


def render_input(row: ContinuousScoreRow, rationales: Mapping[str, str]) -> str:
    try:
        return embedding_input(row.prompt, row.essay, RATIONALE_AWARE_ENCODER, rationales)
    except ValueError as exc:
        raise RationaleAwareEncoderError(str(exc)) from exc


def token_length_audit(
    rows: Sequence[ContinuousScoreRow], rationales: Mapping[str, Mapping[str, str]], tokenizer: Any, max_length: int,
) -> dict[str, Any]:
    texts = [render_input(row, rationales[row.identifier]) for row in rows]
    lengths: list[int] = []
    for start in range(0, len(texts), 128):
        encoded = tokenizer(texts[start:start + 128], add_special_tokens=True, truncation=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    need(bool(lengths) and max(lengths) <= max_length, "rationale-aware encoder input would be truncated")
    ordered = sorted(lengths)
    return {
        "records": len(ordered), "maximum": ordered[-1],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "max_length": max_length, "truncated_records": 0,
    }


def make_dataset(
    rows: Sequence[ContinuousScoreRow], rationales: Mapping[str, Mapping[str, str]], tokenizer: Any, max_length: int,
) -> Any:
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "text": [render_input(row, rationales[row.identifier]) for row in rows],
        "labels": [list(row.labels) for row in rows],
    })
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True, remove_columns=["text"],
    )


def collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        labels = [feature.pop("labels") for feature in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch

    return collate


def verify_artifact_inventory(config: RationaleEncoderConfig) -> dict[str, Any]:
    completion = read_json(Path(config.warmstart_completion_path), "warmstart completion")
    state = completion["state"]
    inventory = state.get("inventory")
    need(isinstance(inventory, list) and inventory, "warmstart inventory differs")
    artifact = Path(config.warmstart_artifact_path)
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for entry in inventory:
        need(isinstance(entry, dict) and set(entry) == {"path", "size", "sha256"}, "warmstart inventory entry differs")
        relative = entry["path"]
        need(isinstance(relative, str) and relative not in seen, "warmstart inventory path differs")
        seen.add(relative)
        path = artifact / relative
        need(path.is_file() and not path.is_symlink(), "warmstart inventory file is unavailable")
        need(path.stat().st_size == entry["size"] and file_sha256(path) == entry["sha256"], "warmstart inventory file checksum differs")
        normalized.append({"path": relative, "size": entry["size"], "sha256": entry["sha256"]})
    actual = {path.relative_to(artifact).as_posix() for path in artifact.rglob("*") if path.is_file()}
    need(actual == seen, "warmstart artifact contains undeclared files")
    normalized.sort(key=lambda row: row["path"])
    digest = sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    need(digest == config.warmstart_artifact_sha256, "warmstart artifact checksum differs")
    return {"artifact_sha256": digest, "files": len(normalized), "verified": True}


def wait_output_and_verify(config: RationaleEncoderConfig, output: Path) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    marker = output / "warmstart_verified.json"
    if rank == 0:
        need(not output.exists(), f"refusing to reuse rationale encoder output: {output}")
        output.mkdir(parents=True)
        verified = verify_artifact_inventory(config)
        marker.write_text(json.dumps(verified, indent=2, sort_keys=True) + "\n")
        return verified
    deadline = time.monotonic() + 600
    while not marker.is_file() and time.monotonic() < deadline:
        time.sleep(0.1)
    need(marker.is_file(), "rank zero did not verify warmstart")
    verified = read_json(marker, "warmstart verification")
    need(verified.get("artifact_sha256") == config.warmstart_artifact_sha256, "warmstart verification marker differs")
    return verified


def build_model(config: RationaleEncoderConfig) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from peft import LoraConfig, TaskType, get_peft_model
    from safetensors import safe_open
    from transformers import AutoModel

    spec = MODEL_SPECS[config.model_key]
    load_kwargs: dict[str, Any] = {
        "revision": config.model_revision,
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16 if config.training_dtype == "bfloat16" else torch.float32,
        "low_cpu_mem_usage": True,
    }
    if config.model_key == "kure_v1":
        load_kwargs["add_pooling_layer"] = False
    backbone = AutoModel.from_pretrained(config.model_path, **load_kwargs)
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    hidden = getattr(backbone.config, "hidden_size", None)
    need(hidden == spec["hidden_size"], "encoder hidden size differs")

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
            hidden_state = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
            if spec["pooling"] == "cls":
                pooled = hidden_state[:, 0]
            else:
                positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
                final = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
                need(bool((final >= 0).all().item()), "encoder input has no nonpadding token")
                pooled = hidden_state[torch.arange(hidden_state.shape[0], device=hidden_state.device), final]
            pooled = functional.normalize(pooled, p=2, dim=-1)
            logits = self.score_head(pooled.to(self.score_head.weight.dtype)).float()
            result: dict[str, Any] = {"logits": logits}
            if labels is not None:
                prediction, _, _ = decode_logits(logits, "bounded_regression")
                result["loss"] = functional.mse_loss(prediction, labels.float())
            return result

    model = Model()
    targets = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    required_parameters = set(dict(model.named_parameters()))
    loaded: set[str] = set()
    artifact = Path(config.warmstart_artifact_path)
    tensor_files = sorted(artifact.rglob("*.safetensors"))
    need(bool(tensor_files), "warmstart has no safetensors")
    for tensor_file in tensor_files:
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                need(name in targets and name not in loaded, f"warmstart tensor differs: {name}")
                tensor = handle.get_tensor(name)
                need(tuple(tensor.shape) == tuple(targets[name].shape), f"warmstart tensor shape differs: {name}")
                targets[name].data.copy_(tensor.to(dtype=targets[name].dtype))
                loaded.add(name)
    need(required_parameters <= loaded, "warmstart is missing model parameters")
    model.backbone = get_peft_model(model.backbone, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, target_modules=list(spec["lora_targets"]), bias="none",
    ))
    if hasattr(model.backbone, "enable_input_require_grads"):
        model.backbone.enable_input_require_grads()
    need(any("lora_" in name and parameter.requires_grad for name, parameter in model.named_parameters()), "fresh MAL LoRA is absent")
    need(all(parameter.requires_grad for parameter in model.score_head.parameters()), "three-score head is frozen")
    need(not any(parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("backbone.") and "lora_" not in name), "AI-Hub backbone is not frozen")
    return model, {
        "load_mode": "full_aihub_backbone_and_matched_three_score_head_then_fresh_mal_lora",
        "loaded_tensor_count": len(loaded), "pooling": spec["pooling"],
        "artifact_sha256": config.warmstart_artifact_sha256,
    }


def trainable_state(model: Any) -> dict[str, Any]:
    state = {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters() if parameter.requires_grad}
    need(any(name.startswith("score_head.") for name in state), "three-score head is absent")
    need(any("lora_" in name for name in state), "LoRA state is absent")
    need(not any("average" in name for name in state), "average state leaked")
    return state


def metrics(result: Any) -> dict[str, float]:
    import torch

    continuous, integers, violations = decode_logits(torch.as_tensor(result.predictions), "bounded_regression")
    values = score_metrics(result.label_ids.tolist(), continuous.tolist(), integers.tolist(), violations.tolist())
    reported = {
        "macro_continuous_rmse": float(values["macro_continuous_rmse"]),
        "macro_continuous_spearman": float(values["macro_continuous_spearman"]),
        "macro_integer_rmse": float(values["macro_integer_rmse"]),
    }
    need(all(math.isfinite(value) for value in reported.values()), "non-finite MAL selection metric")
    return reported


def predict_metrics(trainer: Any, dataset: Any) -> dict[str, Any]:
    import torch

    prediction = trainer.predict(dataset)
    continuous, integers, violations = decode_logits(torch.as_tensor(prediction.predictions), "bounded_regression")
    values = score_metrics(prediction.label_ids.tolist(), continuous.tolist(), integers.tolist(), violations.tolist())
    need(math.isfinite(float(values["macro_continuous_rmse"])), "non-finite MAL validation metric")
    return values


def select_epoch(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    need(bool(events), "MAL selection emitted no events")
    need(
        all(
            all(math.isfinite(float(row[key])) for key in ("macro_continuous_rmse", "macro_continuous_spearman", "macro_integer_rmse"))
            for row in events
        ),
        "MAL selection emitted a non-finite metric",
    )
    return min(
        events,
        key=lambda row: (
            float(row["macro_continuous_rmse"]),
            -float(row["macro_continuous_spearman"]),
            float(row["macro_integer_rmse"]),
            int(row["epoch"]),
        ),
    )


def shuffled_rationales(
    rows: Sequence[ContinuousScoreRow], aligned: Mapping[str, Mapping[str, str]], seed: int,
) -> dict[str, Mapping[str, str]]:
    ordered = sorted(rows, key=lambda row: sha256(f"{seed}\0{row.identifier}".encode()).hexdigest())
    need(len(ordered) > 1, "cannot shuffle one rationale")
    result = {row.identifier: aligned[ordered[(index + 1) % len(ordered)].identifier] for index, row in enumerate(ordered)}
    need(all(result[row.identifier] is not aligned[row.identifier] for row in ordered), "rationale shuffle retained an aligned bundle")
    return result


def run(config: RationaleEncoderConfig, *, smoke: bool = False) -> dict[str, Any]:
    config.validate(require_dependencies=True)
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("rationale-aware encoder training requires .venv-standard") from exc

    output = Path(config.output_root) / (f"smoke-{config.run_id}" if smoke else config.run_id)
    warmstart_verification = wait_output_and_verify(config, output)
    all_train = load_continuous_rows(Path(config.train_path), config.train_sha256, 2000)
    selection_train, selection_dev, split_fingerprint = deterministic_split(all_train, config.seed)
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
            need(tokenizer.eos_token is not None, "tokenizer has no pad token")
            tokenizer.pad_token = tokenizer.eos_token
        model, lineage = build_model(config)
        return tokenizer, model, lineage

    tokenizer, model, initialization = initialize()
    train_token_audit = token_length_audit(all_train if not smoke else selection_train + selection_dev, rationales_train, tokenizer, config.max_length)
    train_dataset = make_dataset(selection_train, rationales_train, tokenizer, config.max_length)
    dev_dataset = make_dataset(selection_dev, rationales_train, tokenizer, config.max_length)
    events: list[dict[str, Any]] = []

    class Capture(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None, **_: Any) -> Any:
            for key in ("loss", "grad_norm"):
                if logs is not None and key in logs:
                    need(math.isfinite(float(logs[key])), f"non-finite MAL training {key}")
            return control

        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Mapping[str, Any] | None = None, model: Any | None = None, **_: Any) -> Any:
            need(metrics is not None and model is not None, "MAL selection evaluation differs")
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
                need(not path.exists(), "MAL epoch checkpoint already exists")
                save_file(trainable_state(model), str(path))
                event.update({"state_path": str(path.resolve()), "state_sha256": file_sha256(path)})
            events.append(event)
            return control

    args = TrainingArguments(
        output_dir=str(output / "selection/trainer"), do_train=True, do_eval=True,
        eval_strategy="epoch", save_strategy="no",
        num_train_epochs=1 if smoke else len(config.selection_epochs), max_steps=1 if smoke else -1,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        optim="adamw_torch_fused", per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=1 if smoke else config.gradient_accumulation_steps,
        bf16=config.training_dtype == "bfloat16", tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
        dataloader_pin_memory=True, ddp_find_unused_parameters=False,
        logging_steps=1 if smoke else 5, seed=config.seed, data_seed=config.seed,
    )
    selector = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=dev_dataset,
        data_collator=collator(tokenizer), compute_metrics=metrics, callbacks=[Capture()],
    )
    selection_train_result = selector.train()
    selector.accelerator.wait_for_everyone()
    shared: list[Any] = [events if selector.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(shared, src=0)
    events = shared[0]
    need(isinstance(events, list) and events, "MAL selection event broadcast differs")
    selected = select_epoch(events)

    if smoke:
        payload = {
            "schema_version": "mal2026-rationale-aware-encoder-result-v1",
            "status": "completed", "mode": "gpu0_one_update_smoke",
            "model_key": config.model_key, "score_fields": list(AXES),
            "average_read": False, "average_target_used": False,
            "target_projection": config.target_projection,
            "selection": {"events": events, "selected": selected},
            "train_token_length_audit": train_token_audit,
            "warmstart_verification": warmstart_verification,
            "initialization": initialization, **prompt_provenance(config.score_prompt_kind),
        }
        if selector.is_world_process_zero():
            (output / "smoke_complete.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return payload

    del selector, model, tokenizer, train_dataset, dev_dataset
    torch.cuda.empty_cache()
    tokenizer, model, refit_initialization = initialize()
    need(refit_initialization == initialization, "selection/refit initialization lineage differs")
    refit_dataset = make_dataset(all_train, rationales_train, tokenizer, config.max_length)
    refit_args = TrainingArguments(
        output_dir=str(output / "refit/trainer"), do_train=True, do_eval=False,
        eval_strategy="no", save_strategy="no", num_train_epochs=float(selected["epoch"]),
        learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        optim="adamw_torch_fused", per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=config.training_dtype == "bfloat16", tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
        dataloader_pin_memory=True, ddp_find_unused_parameters=False,
        logging_steps=5, seed=config.seed, data_seed=config.seed,
    )
    refitter = Trainer(model=model, args=refit_args, train_dataset=refit_dataset, data_collator=collator(tokenizer))
    refit_train_result = refitter.train()
    refitter.accelerator.wait_for_everyone()
    final_state = output / "selected_refit_trainable.safetensors"
    if refitter.is_world_process_zero():
        save_file(trainable_state(model), str(final_state))
    refitter.accelerator.wait_for_everyone()

    # Frozen validation is intentionally first loaded after selection and refit.
    validation = load_continuous_rows(Path(config.validation_path), config.validation_sha256, 400)
    rationales_validation = load_rationales(Path(config.rationale_validation_path), config.rationale_validation_sha256, validation)
    validation_token_audit = token_length_audit(validation, rationales_validation, tokenizer, config.max_length)
    validation_dataset = make_dataset(validation, rationales_validation, tokenizer, config.max_length)
    aligned_metrics = predict_metrics(refitter, validation_dataset)
    shuffled = shuffled_rationales(validation, rationales_validation, config.seed)
    shuffled_dataset = make_dataset(validation, shuffled, tokenizer, config.max_length)
    shuffled_metrics = predict_metrics(refitter, shuffled_dataset)
    result = {
        "schema_version": "mal2026-rationale-aware-encoder-result-v1",
        "status": "completed", "mode": "full", "run_id": config.run_id,
        "model_key": config.model_key, "model_id": config.model_id, "model_revision": config.model_revision,
        "score_fields": list(AXES), "average_read": False, "average_target_used": False,
        "target_projection": config.target_projection,
        "selection": {
            "source": "train_internal_prompt_stratified_1600_400_only",
            "split_fingerprint": split_fingerprint, "events": events, "selected": selected,
            "rule": "lowest macro continuous RMSE, then highest continuous Spearman, then projected-integer RMSE, then earlier epoch",
            "train_metrics": {k: float(v) for k, v in selection_train_result.metrics.items() if isinstance(v, (int, float))},
        },
        "refit": {
            "records": 2000, "epochs": int(selected["epoch"]), "global_step": int(refitter.state.global_step),
            "train_metrics": {k: float(v) for k, v in refit_train_result.metrics.items() if isinstance(v, (int, float))},
        },
        "canonical_validation": {
            "records": 400, "use": "single_final_descriptive_evaluation_not_selection",
            "aligned_bundle_metrics": aligned_metrics,
            "shuffled_bundle_diagnostic_metrics": shuffled_metrics,
            "rationale_shuffle_used_for_training_or_selection": False,
        },
        "rationale_source": {
            "key": config.rationale_key, "structure": "bundle",
            "train_sha256": config.rationale_train_sha256,
            "validation_sha256": config.rationale_validation_sha256,
            "manifest_sha256": config.rationale_manifest_sha256,
        },
        "train_token_length_audit": train_token_audit,
        "validation_token_length_audit": validation_token_audit,
        "warmstart_verification": warmstart_verification,
        "initialization": initialization,
        "state_path": str(final_state.resolve()), "state_sha256": file_sha256(final_state),
        **prompt_provenance(config.score_prompt_kind), "config": asdict(config),
        "privacy": "aggregate_only_no_rows_text_ids_rationales_scores_or_predictions_persisted",
    }
    result_path = output / "result.json"
    if refitter.is_world_process_zero():
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    refitter.accelerator.wait_for_everyone()
    return result
