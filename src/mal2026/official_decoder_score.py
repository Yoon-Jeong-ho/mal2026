"""Official integer-score comparisons built on a causal decoder.

The experiment has three architectures: a causal LM that emits one of the
125 canonical three-integer JSON objects, and two decoder-backbone models with
fresh bounded-regression or cumulative-ordinal score heads.  Each architecture
crosses public/AI-Hub initialization and essay/rationale input, producing 12
target-data arms.  Selection is train-internal; canonical validation is
unreachable until a selected epoch has been refit on all 2,000 train rows.

Private rows and generated predictions remain in memory.  Only aggregate
metrics and hash-bound model state are persisted under the ignored output root.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .official_score_matrix import (
    AXES, ScoreRow, decode_logits, deterministic_internal_split, file_sha256,
    load_rationales, load_score_rows, ordinal_targets, score_metrics,
    select_epoch,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
ARCHITECTURES = ("generative", "bounded_regression", "ordinal_cumulative")
INITIALIZATIONS = ("public", "aihub_matched")
INPUT_VIEWS = ("essay", "rationale")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
STATE_SCHEMA = "mal2026-official-decoder-integer-score-state-v1"
COMPLETION_SCHEMA = "mal2026-official-decoder-integer-score-completion-v1"
_GENERATED_RE = re.compile(r'\A\{"content":([1-5]),"organization":([1-5]),"expression":([1-5])\}\Z')


class OfficialDecoderScoreError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialDecoderScoreError(message)


def arm_names() -> tuple[str, ...]:
    return tuple(f"{arch}__{init}__{view}" for arch in ARCHITECTURES for init in INITIALIZATIONS for view in INPUT_VIEWS)


def parse_arm(arm: str) -> tuple[str, str, str]:
    values = tuple(arm.split("__"))
    _need(len(values) == 3 and values[0] in ARCHITECTURES and values[1] in INITIALIZATIONS and values[2] in INPUT_VIEWS, "unknown decoder score arm")
    return values  # type: ignore[return-value]


def render_target(labels: Sequence[int]) -> str:
    _need(len(labels) == 3 and all(type(value) is int and 1 <= value <= 5 for value in labels), "target labels must be three 1..5 integers")
    return '{"content":%d,"organization":%d,"expression":%d}' % tuple(labels)


def parse_generated(text: str) -> tuple[int, int, int] | None:
    if not isinstance(text, str):
        return None
    matched = _GENERATED_RE.fullmatch(text)
    return tuple(map(int, matched.groups())) if matched else None  # type: ignore[return-value]


def render_input(row: Any, input_view: str, rationales: Mapping[str, str] | None = None) -> str:
    _need(input_view in INPUT_VIEWS, "unknown decoder input view")
    text = (
        "<writing_prompt>\n" + row.prompt + "\n</writing_prompt>\n"
        "<student_essay>\n" + row.essay + "\n</student_essay>"
    )
    if input_view == "rationale":
        _need(rationales is not None and set(rationales) == set(AXES), "three rationale axes are required")
        text += "\n<evaluation_rationales>\n" + "\n".join(f"<{axis}>{rationales[axis]}</{axis}>" for axis in AXES) + "\n</evaluation_rationales>"
    return (
        "다음 한국어 글을 평가하십시오. content, organization, expression의 1~5 정수 점수만 "
        "정확한 JSON으로 출력하고 설명이나 average를 출력하지 마십시오.\n" + text
    )


def chat_prompt(tokenizer: Any, row: Any, input_view: str, rationales: Mapping[str, str] | None = None) -> str:
    messages = [
        {"role": "system", "content": "당신은 한국어 글쓰기 평가자입니다. 과제와 학생 글만 근거로 세 정수 점수만 출력하십시오."},
        {"role": "user", "content": render_input(row, input_view, rationales)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@dataclass(frozen=True)
class WarmArtifact:
    completion_path: str
    completion_sha256: str
    artifact_path: str
    artifact_sha256: str
    state_metadata_path: str
    state_metadata_sha256: str


@dataclass(frozen=True)
class DecoderScoreConfig:
    schema_version: str
    run_id: str
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
    aihub_manifest_path: str
    aihub_artifacts: Mapping[str, WarmArtifact]
    output_root: str
    score_fields: tuple[str, str, str]
    selection_epochs: tuple[int, ...]
    seed: int
    max_length: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    training_dtype: str
    historical_result_classification: str

    @classmethod
    def from_json(cls, path: Path, *, require_dependencies: bool = True) -> "DecoderScoreConfig":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OfficialDecoderScoreError("decoder config is unreadable") from exc
        _need(isinstance(raw, dict), "decoder config must be an object")
        for key in ("score_fields", "selection_epochs", "lora_target_modules"):
            _need(isinstance(raw.get(key), list), f"{key} must be a list")
            raw[key] = tuple(raw[key])
        artifacts = raw.get("aihub_artifacts")
        _need(isinstance(artifacts, dict) and set(artifacts) == set(ARCHITECTURES), "AI-Hub artifacts must cover all decoder architectures")
        raw["aihub_artifacts"] = {name: WarmArtifact(**value) for name, value in artifacts.items()}
        _need(set(raw) == set(cls.__dataclass_fields__), "decoder config has missing or unknown fields")
        config = cls(**raw)
        config.validate(require_dependencies=require_dependencies)
        return config

    def validate(self, *, require_dependencies: bool = True) -> None:
        _need(self.schema_version == "mal2026-official-decoder-score-config-v1", "decoder config schema differs")
        _need(self.run_id == "official-decoder-score-matrix-v1-20260727-001", "decoder run identity differs")
        _need((self.model_id, self.model_revision) == (MODEL_ID, MODEL_REVISION), "decoder model pin differs")
        _need(self.score_fields == AXES, "only three analytic axes are allowed")
        _need(self.selection_epochs == (1, 2, 3, 4), "selection epochs differ")
        _need(self.seed == 2026072702 and self.max_length == 2048, "seed/length contract differs")
        _need((self.learning_rate, self.weight_decay, self.warmup_ratio) == (2e-5, 0.01, 0.05), "optimizer contract differs")
        _need((self.per_device_train_batch_size, self.per_device_eval_batch_size, self.gradient_accumulation_steps) == (1, 2, 8), "batch contract differs")
        _need((self.lora_r, self.lora_alpha, self.lora_dropout) == (16, 32, 0.05), "LoRA contract differs")
        _need(self.lora_target_modules == LORA_TARGETS and self.training_dtype == "bfloat16", "LoRA target/dtype differs")
        _need(self.historical_result_classification == "descriptive_only_continuous_or_four_axis_not_loaded", "historical-result boundary differs")
        _need(Path(self.output_root).resolve() == (ROOT / "outputs" / "official-decoder-score-matrix-v1").resolve(), "output root differs")
        _need(Path(self.validation_path).resolve() == (ROOT / "eval" / "validation.jsonl").resolve(), "canonical validation path differs")
        _need(self.validation_sha256 == "0805445029328848164cf15f34b90b88fb5f7896d7b73f24f1717b733a9a00a4", "canonical validation pin differs")
        if not require_dependencies:
            return
        model = Path(self.model_path)
        _need(model.is_dir() and not model.is_symlink(), "local decoder snapshot is unavailable")
        _need(Path(self.aihub_manifest_path).is_file() and not Path(self.aihub_manifest_path).is_symlink(), "canonical AI-Hub manifest is unavailable")
        # Canonical validation is path/hash pinned above but deliberately not
        # opened or hashed here. Its bytes and labels remain unreachable until
        # the post-refit call to load_score_rows.
        for source, digest, label in ((self.train_path, self.train_sha256, "canonical train"),):
            path = Path(source)
            _need(path.is_file() and not path.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", digest) is not None and file_sha256(path) == digest, f"{label} checksum differs")
        _need(Path(self.aihub_manifest_path).resolve() == (ROOT / "data" / "manifests" / "aihub_human_feedback_v1.json").resolve(), "AI-Hub manifest differs")

    def validate_rationales(self, split: str) -> None:
        _need(split in {"train", "validation"}, "rationale split differs")
        _need(self.rationale_key and not self.rationale_key.startswith("REQUIRED_"), "selected rationale key is unresolved")
        restricted = (ROOT / "data" / "processed" / "restricted").resolve()
        source, digest = (
            (self.rationale_train_path, self.rationale_train_sha256)
            if split == "train" else
            (self.rationale_validation_path, self.rationale_validation_sha256)
        )
        path = Path(source)
        _need(path.resolve().is_relative_to(restricted) and path.is_file() and not path.is_symlink(), "rationale source is unavailable or outside restricted storage")
        _need(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and file_sha256(path) == digest, "rationale checksum differs")

    def validate_warm_artifact(self, architecture: str) -> WarmArtifact:
        _need(architecture in ARCHITECTURES, "unknown warm architecture")
        artifact = self.aihub_artifacts[architecture]
        for source, digest, label in ((artifact.completion_path, artifact.completion_sha256, "completion"), (artifact.state_metadata_path, artifact.state_metadata_sha256, "state metadata")):
            path = Path(source)
            _need(path.is_file() and not path.is_symlink(), f"AI-Hub {label} is unavailable")
            _need(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and file_sha256(path) == digest, f"AI-Hub {label} checksum differs")
        artifact_path = Path(artifact.artifact_path)
        _need(artifact_path.is_dir() and not artifact_path.is_symlink(), "AI-Hub full-model artifact is unavailable")
        completion = json.loads(Path(artifact.completion_path).read_text(encoding="utf-8"))
        _need(completion.get("schema_version") == COMPLETION_SCHEMA and completion.get("status") == "completed", "AI-Hub completion schema/status differs")
        _need(completion.get("phase") == "refit" and completion.get("architecture") == architecture, "AI-Hub completion lineage differs")
        _need(completion.get("initialization") == "public" and completion.get("input_view") == "essay", "AI-Hub initialization/input lineage differs")
        _need(completion.get("score_fields") == list(AXES) and completion.get("integer_target_used") is True and completion.get("average_target_used") is False, "AI-Hub target contract differs")
        _need(completion.get("rationale_output_used") is False and completion.get("canonical_validation") is None, "AI-Hub pretraining crossed a forbidden output/evaluation boundary")
        _need(completion.get("data", {}).get("dataset") == "aihub_human_feedback_v1", "AI-Hub dataset lineage differs")
        reproducibility = completion.get("reproducibility", {})
        _need((reproducibility.get("model_id"), reproducibility.get("model_revision")) == (self.model_id, self.model_revision), "AI-Hub decoder model lineage differs")
        state = completion.get("state", {})
        _need(state.get("artifact_sha256") == artifact.artifact_sha256 and state.get("schema_version") == STATE_SCHEMA, "AI-Hub state binding differs")
        _need(state.get("metadata_sha256") == artifact.state_metadata_sha256 and state.get("training_method") == "full_parameter", "AI-Hub state metadata/training method differs")
        metadata = json.loads(Path(artifact.state_metadata_path).read_text(encoding="utf-8"))
        _need(metadata.get("artifact_sha256") == artifact.artifact_sha256 and metadata.get("architecture") == architecture, "AI-Hub artifact metadata differs")
        inventory = metadata.get("inventory")
        _need(isinstance(inventory, list) and bool(inventory), "AI-Hub artifact inventory is absent")
        expected_by_path: dict[str, Mapping[str, Any]] = {}
        for item in inventory:
            _need(isinstance(item, dict) and set(item) == {"path", "size", "sha256"}, "AI-Hub inventory entry differs")
            _need(item["path"] not in expected_by_path, "AI-Hub inventory has a duplicate path")
            expected_by_path[item["path"]] = item
            path = artifact_path / item["path"]
            _need(path.is_file() and not path.is_symlink() and path.stat().st_size == item["size"] and file_sha256(path) == item["sha256"], "AI-Hub inventory file differs")
        actual_paths = {path.relative_to(artifact_path).as_posix() for path in artifact_path.rglob("*") if path.is_file()}
        _need(actual_paths == set(expected_by_path), "AI-Hub artifact contains an unbound or missing file")
        canonical_inventory = [expected_by_path[name] for name in sorted(expected_by_path)]
        actual_sha = sha256(json.dumps(canonical_inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        _need(actual_sha == artifact.artifact_sha256, "AI-Hub artifact inventory digest differs")
        return artifact


def generative_metrics(labels: Sequence[Sequence[int]], predictions: Sequence[Sequence[int]]) -> dict[str, Any]:
    _need(bool(labels) and len(labels) == len(predictions), "generative metric population differs")
    continuous = [[float(value) for value in row] for row in predictions]
    metrics = score_metrics(labels, continuous, predictions)
    metrics["strict_parse_rate"] = 1.0
    return metrics


def canonical_targets() -> tuple[str, ...]:
    return tuple(render_target((a, b, c)) for a in range(1, 6) for b in range(1, 6) for c in range(1, 6))


def _load_full_warmstate(base: Any, artifact: WarmArtifact, architecture: str) -> tuple[int, dict[str, Any]]:
    from safetensors import safe_open
    targets = dict(base.state_dict())
    loaded = 0
    source_backbone_names: set[str] = set()
    head_state: dict[str, Any] = {}
    for state_file in sorted(Path(artifact.artifact_path).glob("*.safetensors")):
        with safe_open(state_file, framework="pt", device="cpu") as handle:
            for source_name in handle.keys():
                if architecture == "generative":
                    target_name = source_name
                elif source_name.startswith("backbone."):
                    target_name = source_name[len("backbone."):]
                elif source_name.startswith("score_head."):
                    head_state[source_name] = handle.get_tensor(source_name).detach().cpu()
                    continue
                else:
                    raise OfficialDecoderScoreError(f"AI-Hub dedicated-head artifact has an unexpected tensor: {source_name}")
                _need(target_name in targets, f"AI-Hub full backbone tensor is unmatched: {source_name}")
                tensor = handle.get_tensor(source_name)
                _need(tuple(tensor.shape) == tuple(targets[target_name].shape), f"AI-Hub tensor shape differs: {source_name}")
                targets[target_name].data.copy_(tensor.to(dtype=targets[target_name].dtype))
                source_backbone_names.add(target_name)
                loaded += 1
    _need(loaded > 0, "AI-Hub state loaded no matching tensors")
    if architecture == "generative":
        _need(not head_state, "generative AI-Hub state contains a score head")
    else:
        _need(set(head_state) == {"score_head.weight", "score_head.bias"}, "AI-Hub matched score head is incomplete")
    _need(source_backbone_names <= set(targets), "AI-Hub artifact names exceed the decoder state")
    loaded_storage = {targets[name].untyped_storage().data_ptr() for name in source_backbone_names}
    missing_uncovered = [name for name in set(targets) - source_backbone_names if targets[name].untyped_storage().data_ptr() not in loaded_storage]
    _need(not missing_uncovered, "AI-Hub artifact is not a complete full-parameter backbone")
    return loaded, head_state


def build_model(config: DecoderScoreConfig, architecture: str, initialization: str) -> tuple[Any, Mapping[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel, AutoModelForCausalLM

    _need(architecture in ARCHITECTURES and initialization in INITIALIZATIONS, "model arm differs")
    task = TaskType.CAUSAL_LM if architecture == "generative" else TaskType.FEATURE_EXTRACTION
    loader = AutoModelForCausalLM if architecture == "generative" else AutoModel
    base = loader.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    base.config.use_cache = False
    artifact = config.validate_warm_artifact(architecture) if initialization == "aihub_matched" else None
    loaded_count = 0
    warm_head: dict[str, Any] = {}
    if artifact is not None:
        loaded_count, warm_head = _load_full_warmstate(base, artifact, architecture)
    backbone = get_peft_model(base, LoraConfig(task_type=task, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=list(config.lora_target_modules), bias="none"))
    if architecture == "generative":
        model = backbone
    else:
        hidden = getattr(backbone.config, "hidden_size", None)
        _need(type(hidden) is int and hidden > 0, "decoder hidden size is unavailable")

        class DecoderScoreHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = backbone
                self.score_head = nn.Linear(hidden, 3 if architecture == "bounded_regression" else 12)

            def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None, **_: Any) -> Mapping[str, Any]:
                hidden_state = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
                positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
                final = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
                _need(bool((final >= 0).all().item()), "decoder input has no non-padding token")
                logits = self.score_head(hidden_state[torch.arange(hidden_state.shape[0], device=hidden_state.device), final].float())
                result: dict[str, Any] = {"logits": logits}
                if labels is not None:
                    if architecture == "bounded_regression":
                        result["loss"] = functional.mse_loss(decode_logits(logits, architecture)[0], labels.float())
                    else:
                        result["loss"] = functional.binary_cross_entropy_with_logits(logits.reshape(-1, 3, 4), ordinal_targets(labels))
                return result

        model = DecoderScoreHead()
        if warm_head:
            model.score_head.load_state_dict({name.removeprefix("score_head."): value for name, value in warm_head.items()}, strict=True)
    provenance: dict[str, Any] = {"initialization": initialization, "model_revision": config.model_revision}
    if initialization == "aihub_matched":
        assert artifact is not None
        provenance.update({"loaded_full_backbone_tensor_count": loaded_count, "artifact_sha256": artifact.artifact_sha256, "matched_architecture": architecture, "mal_adaptation": "fresh_LoRA_after_full_AIHub_tuning", "matched_score_head_retained": architecture != "generative"})
    return model, provenance


def trainable_state(model: Any, architecture: str) -> dict[str, Any]:
    state = {name: value.detach().cpu().contiguous() for name, value in model.named_parameters() if value.requires_grad}
    _need(any("lora_" in name for name in state), "decoder LoRA state is absent")
    if architecture == "generative":
        _need(not any(name.startswith("score_head.") for name in state), "generative model unexpectedly has a score head")
    else:
        _need({name for name in state if name.startswith("score_head.")} == {"score_head.weight", "score_head.bias"}, "dedicated score head state differs")
    return state


def state_sha256(state: Mapping[str, Any]) -> str:
    digest = sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(json.dumps(list(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def experiment_contract(config: DecoderScoreConfig) -> dict[str, Any]:
    """Public aggregate contract consumed by orchestrators and reports."""
    config.validate(require_dependencies=False)
    return {
        "arms": list(arm_names()), "arm_count": 12, "architectures": list(ARCHITECTURES),
        "initializations": list(INITIALIZATIONS), "input_views": list(INPUT_VIEWS),
        "score_fields": list(AXES), "integer_target_used": True,
        "target_projection": "official_half_up", "average_target_used": False,
        "selection_source": "deterministic train-internal 1600/400 only",
        "refit_source": "all 2000 train rows", "canonical_validation_use": "single final descriptive evaluation only",
        "aihub_pretraining": "same decoder architecture; full-parameter integer three-axis AI-Hub selection/exact-step refit; then fresh MAL LoRA; no rationale or canonical validation",
        "generative_output_space": {"format": '{"content":I,"organization":I,"expression":I}', "canonical_outputs": 125, "rationale": False, "average": False},
        "historical_results": config.historical_result_classification,
        "privacy": "aggregate_only_no_rows_text_ids_rationales_or_predictions_persisted",
    }


def _head_dataset(rows: Sequence[Any], tokenizer: Any, config: DecoderScoreConfig, input_view: str, rationales: Mapping[str, Mapping[str, str]] | None) -> Any:
    from datasets import Dataset
    texts = [chat_prompt(tokenizer, row, input_view, None if rationales is None else rationales[row.identifier]) for row in rows]
    dataset = Dataset.from_dict({"text": texts, "labels": [list(row.labels) for row in rows]})
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=config.max_length), batched=True, remove_columns=["text"])


def _head_collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        labels = [feature.pop("labels") for feature in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch
    return collate


def _causal_dataset(rows: Sequence[Any], tokenizer: Any, config: DecoderScoreConfig, input_view: str, rationales: Mapping[str, Mapping[str, str]] | None) -> Any:
    from datasets import Dataset
    items: list[dict[str, Any]] = []
    _need(tokenizer.eos_token_id is not None, "decoder tokenizer has no EOS token")
    for row in rows:
        prompt = chat_prompt(tokenizer, row, input_view, None if rationales is None else rationales[row.identifier])
        prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=config.max_length - 32)["input_ids"]
        target_ids = tokenizer(render_target(row.labels), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        _need(len(prompt_ids) + len(target_ids) <= config.max_length, "decoder training example exceeds max_length")
        items.append({"input_ids": prompt_ids + target_ids, "attention_mask": [1] * (len(prompt_ids) + len(target_ids)), "labels": [-100] * len(prompt_ids) + target_ids})
    return Dataset.from_list(items)


def _causal_collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        width = max(len(item["input_ids"]) for item in features)
        pad = tokenizer.pad_token_id
        _need(pad is not None, "decoder tokenizer has no pad token")
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            count = width - len(item["input_ids"])
            result["input_ids"].append(item["input_ids"] + [pad] * count)
            result["attention_mask"].append(item["attention_mask"] + [0] * count)
            result["labels"].append(item["labels"] + [-100] * count)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}
    return collate


def _token_trie(tokenizer: Any) -> tuple[dict[tuple[int, ...], tuple[int, ...]], int]:
    _need(tokenizer.eos_token_id is not None, "decoder tokenizer has no EOS token")
    sequences = [tuple(tokenizer(value, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]) for value in canonical_targets()]
    _need(all(sequence for sequence in sequences), "canonical decoder target tokenization differs")
    trie: dict[tuple[int, ...], set[int]] = {}
    for sequence in sequences:
        for offset, token in enumerate(sequence):
            trie.setdefault(sequence[:offset], set()).add(token)
    return {key: tuple(sorted(value)) for key, value in trie.items()}, max(map(len, sequences))


def generate_integer_predictions(model: Any, tokenizer: Any, rows: Sequence[Any], config: Any, input_view: str = "essay", rationales: Mapping[str, Mapping[str, str]] | None = None) -> tuple[list[tuple[int, int, int]], int]:
    """Free-run one decoder inside the frozen 125-output token trie."""
    import torch
    trie, max_new_tokens = _token_trie(tokenizer)
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    predictions: list[tuple[int, int, int]] = []
    invalid = 0
    model.eval()
    for start in range(0, len(rows), config.per_device_eval_batch_size):
        batch_rows = rows[start:start + config.per_device_eval_batch_size]
        prompts = [chat_prompt(tokenizer, row, input_view, None if rationales is None else rationales[row.identifier]) for row in batch_rows]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=config.max_length - max_new_tokens, return_tensors="pt").to(next(model.parameters()).device)
        prefix_width = encoded["input_ids"].shape[1]

        def allowed(_: int, sent: Any) -> list[int]:
            prefix = tuple(int(value) for value in sent[prefix_width:].tolist())
            return list(trie.get(prefix, (tokenizer.eos_token_id,)))

        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens, prefix_allowed_tokens_fn=allowed, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        for sequence in generated[:, prefix_width:]:
            parsed = parse_generated(tokenizer.decode(sequence, skip_special_tokens=True))
            if parsed is None:
                invalid += 1
                predictions.append((3, 3, 3))
            else:
                predictions.append(parsed)
    tokenizer.padding_side = previous_padding
    return predictions, invalid


def _evaluate_generative(model: Any, tokenizer: Any, rows: Sequence[Any], config: DecoderScoreConfig, input_view: str, rationales: Mapping[str, Mapping[str, str]] | None) -> dict[str, Any]:
    predictions, invalid = generate_integer_predictions(model, tokenizer, rows, config, input_view, rationales)
    metrics = score_metrics([row.labels for row in rows], predictions, predictions)
    metrics["strict_parse_rate"] = 1.0 - invalid / len(rows)
    metrics["invalid_output_count"] = invalid
    metrics["constraint_space_size"] = 125
    return metrics


def _training_args(config: DecoderScoreConfig, output: Path, epochs: int, smoke: bool) -> Any:
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=str(output), do_train=True, do_eval=False, eval_strategy="no", save_strategy="no",
        num_train_epochs=float(epochs), max_steps=1 if smoke else -1,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=1 if smoke else config.gradient_accumulation_steps,
        bf16=True, tf32=True, report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
        ddp_find_unused_parameters=False, logging_steps=1 if smoke else 5, save_only_model=True,
        seed=config.seed, data_seed=config.seed,
    )


def _fit_once(config: DecoderScoreConfig, architecture: str, initialization: str, train_rows: Sequence[Any], eval_rows: Sequence[Any] | None, input_view: str, rationales_train: Mapping[str, Mapping[str, str]] | None, rationales_eval: Mapping[str, Mapping[str, str]] | None, epochs: int, output: Path, smoke: bool) -> tuple[Any, Any, Mapping[str, Any], dict[str, Any] | None, str]:
    import torch
    from transformers import AutoTokenizer, Trainer, set_seed
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        _need(tokenizer.eos_token is not None, "decoder tokenizer has no pad/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model, provenance = build_model(config, architecture, initialization)
    initial_hash = state_sha256(trainable_state(model, architecture))
    if architecture == "generative":
        train_dataset = _causal_dataset(train_rows, tokenizer, config, input_view, rationales_train)
        collator = _causal_collator(tokenizer)
    else:
        train_dataset = _head_dataset(train_rows, tokenizer, config, input_view, rationales_train)
        collator = _head_collator(tokenizer)
    trainer = Trainer(model=model, args=_training_args(config, output, epochs, smoke), train_dataset=train_dataset, data_collator=collator)
    trained = trainer.train()
    trainer.accelerator.wait_for_everyone()
    metrics: dict[str, Any] | None = None
    if eval_rows is not None:
        if architecture == "generative":
            if trainer.is_world_process_zero():
                metrics = _evaluate_generative(model, tokenizer, eval_rows, config, input_view, rationales_eval)
            shared: list[Any] = [metrics]
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(shared, src=0)
            metrics = shared[0]
        else:
            dataset = _head_dataset(eval_rows, tokenizer, config, input_view, rationales_eval)
            prediction = trainer.predict(dataset)
            continuous, integers, violations = decode_logits(torch.as_tensor(prediction.predictions), architecture)
            metrics = score_metrics(prediction.label_ids.tolist(), continuous.tolist(), integers.tolist(), violations.tolist())
    return trainer, tokenizer, provenance, metrics, initial_hash


def _run_selection_refit(config: DecoderScoreConfig, architecture: str, initialization: str, input_view: str, selection_train: Sequence[Any], selection_dev: Sequence[Any], refit_rows: Sequence[Any], rationales_train: Mapping[str, Mapping[str, str]] | None, rationales_dev: Mapping[str, Mapping[str, str]] | None, output: Path, phase: str, smoke: bool, split_fingerprint: str | None, source_provenance: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    if int(os.environ.get("RANK", "0")) == 0:
        _need(not output.exists(), f"refusing to reuse output: {output}")
        output.mkdir(parents=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    epochs = (1,) if smoke else config.selection_epochs
    events: list[dict[str, Any]] = []
    initial_hash: str | None = None
    for epoch in epochs:
        trainer, _, _, metrics, candidate_hash = _fit_once(config, architecture, initialization, selection_train, selection_dev, input_view, rationales_train, rationales_dev, epoch, output / f"selection-epoch-{epoch}", smoke)
        _need(metrics is not None, "selection evaluation emitted no metrics")
        if initial_hash is None:
            initial_hash = candidate_hash
        _need(candidate_hash == initial_hash, "selection candidates did not replay identical initialization")
        events.append({"epoch": epoch, "macro_integer_rmse": metrics["macro_integer_rmse"], "macro_integer_spearman": metrics["macro_integer_spearman"], "macro_continuous_rmse": metrics["macro_continuous_rmse"], "metrics": metrics})
        del trainer
        torch.cuda.empty_cache()
    best = select_epoch(events)
    refitter, refit_tokenizer, provenance, _, refit_hash = _fit_once(config, architecture, initialization, refit_rows, None, input_view, rationales_train, None, int(best["epoch"]), output / "refit", smoke)
    _need(refit_hash == initial_hash, "selection/refit initialization differs")
    final_metrics = None
    final_rows: Sequence[Any] | None = None
    rationales_final: Mapping[str, Mapping[str, str]] | None = None
    if phase == "target_refit":
        # Canonical validation is deliberately not loaded until selection and
        # all-train refit have both completed.
        final_rows = load_score_rows(Path(config.validation_path), config.validation_sha256, 400)
        if input_view == "rationale":
            config.validate_rationales("validation")
            rationales_final = load_rationales(Path(config.rationale_validation_path), config.rationale_validation_sha256, final_rows)
        if smoke:
            final_rows = final_rows[:4]
        if architecture == "generative":
            if refitter.is_world_process_zero():
                final_metrics = _evaluate_generative(refitter.model, refit_tokenizer, final_rows, config, input_view, rationales_final)
        else:
            prediction = refitter.predict(_head_dataset(final_rows, refit_tokenizer, config, input_view, rationales_final))
            continuous, integers, violations = decode_logits(torch.as_tensor(prediction.predictions), architecture)
            final_metrics = score_metrics(prediction.label_ids.tolist(), continuous.tolist(), integers.tolist(), violations.tolist())
    state_path = output / "trainable_state.safetensors"
    if refitter.is_world_process_zero():
        save_file(trainable_state(refitter.model, architecture), str(state_path), metadata={"schema_version": STATE_SCHEMA, "architecture": architecture, "score_fields": ",".join(AXES)})
    refitter.accelerator.wait_for_everyone()
    payload = {
        "schema_version": COMPLETION_SCHEMA, "status": "completed", "run_id": config.run_id, "phase": phase,
        "architecture": architecture, "initialization": initialization, "input_view": input_view,
        "score_fields": list(AXES), "integer_target_used": True, "target_projection": "official_half_up",
        "average_read": False, "average_target_used": False, "rationale_output_used": False,
        "selection": {"events": events, "selected_epoch": best["epoch"], "rule": "integer RMSE, integer Spearman, continuous RMSE, earlier epoch", "initial_state_sha256": initial_hash, "split_fingerprint": split_fingerprint},
        "refit": {"records": len(refit_rows), "epochs": best["epoch"], "initial_state_sha256": refit_hash},
        "state": {"path": str(state_path.resolve()), "sha256": file_sha256(state_path), "schema_version": STATE_SCHEMA},
        "initialization_provenance": provenance,
        "canonical_validation": None if final_rows is None else {"use": "single_final_descriptive_evaluation_not_selection", "records": len(final_rows), "metrics": final_metrics},
        "data_provenance": {
            "train_sha256": None if phase == "aihub_refit" else config.train_sha256,
            "validation_sha256": None if phase == "aihub_refit" else config.validation_sha256,
            "rationale_key": config.rationale_key if phase == "target_refit" and input_view == "rationale" else None,
            "rationale_train_sha256": config.rationale_train_sha256 if phase == "target_refit" and input_view == "rationale" else None,
            "rationale_validation_sha256": config.rationale_validation_sha256 if phase == "target_refit" and input_view == "rationale" else None,
            **source_provenance,
        },
        "reproducibility": {
            "seed": config.seed, "model_id": config.model_id, "model_revision": config.model_revision,
            "training_dtype": config.training_dtype, "visible_gpu_scope": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "config_identity_sha256": sha256(json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        },
        "privacy": "aggregate_only_no_rows_text_ids_rationales_or_predictions_persisted",
    }
    if refitter.is_world_process_zero():
        (output / "training_complete.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    refitter.accelerator.wait_for_everyone()
    return payload


def run_target_arm(config: DecoderScoreConfig, arm: str, *, smoke: bool = False) -> dict[str, Any]:
    """Select/refit one target arm and only then load canonical validation once."""
    config.validate(require_dependencies=True)
    architecture, initialization, input_view = parse_arm(arm)
    if initialization == "aihub_matched":
        config.validate_warm_artifact(architecture)
    train_rows = load_score_rows(Path(config.train_path), config.train_sha256, 2000)
    selection_train, selection_dev, split_fingerprint = deterministic_internal_split(train_rows, config.seed)
    rationales = None
    if input_view == "rationale":
        config.validate_rationales("train")
        rationales = load_rationales(Path(config.rationale_train_path), config.rationale_train_sha256, train_rows)
    if smoke:
        selection_train, selection_dev = selection_train[:4], selection_dev[:4]
    output = Path(config.output_root) / (f"smoke-{arm}" if smoke else arm)
    return _run_selection_refit(config, architecture, initialization, input_view, selection_train, selection_dev, train_rows if not smoke else train_rows[:4], rationales, rationales, output, "target_refit", smoke, split_fingerprint, {"dataset": "mal2026_official_train_validation"})


def rank_results(results: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Rank completed target arms by the frozen integer-primary rule."""
    _need(bool(results), "decoder results are empty")
    for result in results:
        _need(result.get("status") == "completed" and result.get("phase") == "target_refit", "decoder result is not a completed target refit")
        metrics = result.get("canonical_validation", {}).get("metrics", {})
        _need(all(key in metrics for key in ("macro_integer_rmse", "macro_integer_spearman", "macro_continuous_rmse")), "decoder result metrics are incomplete")
    return sorted(results, key=lambda result: (
        float(result["canonical_validation"]["metrics"]["macro_integer_rmse"]),
        -float(result["canonical_validation"]["metrics"]["macro_integer_spearman"]),
        float(result["canonical_validation"]["metrics"]["macro_continuous_rmse"]),
        str(result["architecture"]), str(result["initialization"]), str(result["input_view"]),
    ))


def aggregate_results(config: DecoderScoreConfig) -> dict[str, Any]:
    results: list[Mapping[str, Any]] = []
    for arm in arm_names():
        path = Path(config.output_root) / arm / "training_complete.json"
        _need(path.is_file() and not path.is_symlink(), f"decoder result is unavailable: {arm}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        _need(payload.get("run_id") == config.run_id and payload.get("score_fields") == list(AXES), "decoder result identity differs")
        _need(payload.get("average_read") is False and payload.get("average_target_used") is False, "decoder result used average")
        _need(payload.get("data_provenance", {}).get("validation_sha256") == config.validation_sha256, "decoder result validation lineage differs")
        results.append(payload)
    ranked = rank_results(results)
    compact = []
    for rank, result in enumerate(ranked, 1):
        compact.append({
            "rank": rank, "arm": f'{result["architecture"]}__{result["initialization"]}__{result["input_view"]}',
            "architecture": result["architecture"], "initialization": result["initialization"], "input_view": result["input_view"],
            "metrics": result["canonical_validation"]["metrics"],
        })
    return {
        "schema_version": "mal2026-official-decoder-score-aggregate-v1", "status": "completed",
        "run_id": config.run_id, "arm_count": 12, "ranking_rule": "integer RMSE, integer Spearman, continuous RMSE, lexical identity",
        "score_fields": list(AXES), "average_target_used": False, "canonical_validation_sha256": config.validation_sha256,
        "ranking": compact, "winner": compact[0],
        "privacy": "aggregate_only_no_rows_text_ids_rationales_or_predictions_persisted",
    }
