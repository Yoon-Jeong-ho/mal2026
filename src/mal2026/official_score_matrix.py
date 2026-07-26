"""Leakage-resistant eight-arm Qwen3-Embedding score experiment.

The matrix crosses two score heads, two initializations, and two input views.
Model selection uses only a deterministic train-internal split.  The canonical
validation split is loaded exactly once, after the selected epoch has been
refit on all 2,000 training essays.  Only the three analytic integer scores
are read; the source ``average`` value is never accessed.
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


ROOT = Path(__file__).resolve().parents[2]
AXES = ("content", "organization", "expression")
HEADS = ("bounded_regression", "ordinal_cumulative")
INITIALIZATIONS = ("public", "aihub_matched_full")
INPUTS = ("essay", "rationale")
MODEL_ID = "Qwen/Qwen3-Embedding-8B"
MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class OfficialScoreMatrixError(ValueError):
    """Raised before a matrix contract is violated."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialScoreMatrixError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_inventory_sha256(directory: Path, inventory: Any) -> str:
    """Verify every declared full-model file and hash the canonical inventory."""
    _need(isinstance(inventory, list) and bool(inventory), "AI-Hub artifact inventory differs")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in inventory:
        _need(isinstance(entry, dict) and set(entry) == {"path", "size", "sha256"}, "AI-Hub artifact inventory entry differs")
        relative = entry["path"]
        _need(isinstance(relative, str) and relative not in seen, "AI-Hub artifact inventory path differs")
        seen.add(relative)
        path = directory / relative
        _need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()), "AI-Hub artifact file is unsafe")
        _need(path.stat().st_size == entry["size"] and file_sha256(path) == entry["sha256"], "AI-Hub artifact inventory file differs")
        normalized.append({"path": relative, "size": entry["size"], "sha256": entry["sha256"]})
    actual_files = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    _need(actual_files == seen, "AI-Hub artifact has undeclared or missing files")
    normalized.sort(key=lambda entry: entry["path"])
    return sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def official_half_up(value: float) -> int:
    """Project a positive score to the official integer scale, half upward."""
    _need(math.isfinite(float(value)), "score must be finite")
    return min(5, max(1, int(math.floor(float(value) + 0.5))))


def arm_names() -> tuple[str, ...]:
    return tuple(f"{head}__{initialization}__{input_view}" for head in HEADS for initialization in INITIALIZATIONS for input_view in INPUTS)


def parse_arm(name: str) -> tuple[str, str, str]:
    parts = tuple(name.split("__"))
    _need(len(parts) == 3 and parts[0] in HEADS and parts[1] in INITIALIZATIONS and parts[2] in INPUTS, "unknown matrix arm")
    return parts  # type: ignore[return-value]


@dataclass(frozen=True)
class MatrixConfig:
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
    rationale_manifest_path: str
    rationale_manifest_sha256: str
    bootstrap_selection_path: str
    bootstrap_selection_sha256: str
    aihub_bounded_completion_path: str
    aihub_bounded_completion_sha256: str
    aihub_bounded_artifact_path: str
    aihub_bounded_artifact_sha256: str
    aihub_ordinal_completion_path: str
    aihub_ordinal_completion_sha256: str
    aihub_ordinal_artifact_path: str
    aihub_ordinal_artifact_sha256: str
    aihub_warmstart_load_mode: str
    historical_warmstate_metadata_path: str
    historical_warmstate_path: str
    historical_warmstate_sha256: str
    historical_warmstate_classification: str
    output_root: str
    restricted_bootstrap_output_root: str
    score_fields: tuple[str, str, str]
    seed: int
    max_length: int
    selection_epochs: tuple[int, int, int, int]
    internal_dev_fraction: float
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

    @classmethod
    def from_json(cls, path: Path, *, require_dependencies: bool = False) -> "MatrixConfig":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OfficialScoreMatrixError("matrix config is unreadable") from exc
        _need(isinstance(raw, dict), "matrix config must be an object")
        for field in ("score_fields", "selection_epochs"):
            _need(isinstance(raw.get(field), list), f"{field} must be a list")
            raw[field] = tuple(raw[field])
        _need(set(raw) == set(cls.__dataclass_fields__), "matrix config has missing or unknown fields")
        value = cls(**raw)
        value.validate()
        if require_dependencies:
            value.validate_dependencies("all")
        return value

    def validate(self) -> None:
        _need(self.schema_version == "mal2026-official-score-matrix-v1", "matrix schema differs")
        _need(self.run_id == "official-score-matrix-v1-20260727-001", "matrix run identity differs")
        _need((self.model_id, self.model_revision) == (MODEL_ID, MODEL_REVISION), "model pin differs")
        _need(self.score_fields == AXES, "only content/organization/expression targets are allowed")
        _need(self.selection_epochs == (1, 2, 3, 4) and self.internal_dev_fraction == 0.2, "selection protocol differs")
        _need((self.seed, self.max_length) == (2026072701, 2048), "data/numeric seed contract differs")
        _need((self.learning_rate, self.weight_decay, self.warmup_ratio) == (1e-4, 0.01, 0.05), "optimization contract differs")
        _need((self.per_device_train_batch_size, self.per_device_eval_batch_size, self.gradient_accumulation_steps) == (2, 4, 4), "batch contract differs")
        _need((self.lora_r, self.lora_alpha, self.lora_dropout, self.training_dtype) == (16, 32, 0.05, "bfloat16"), "LoRA/numeric contract differs")
        _need(self.aihub_warmstart_load_mode == "full_backbone_and_matched_head_then_fresh_mal_lora", "AI-Hub warmstart semantics differ")
        _need(self.historical_warmstate_classification == "historical_continuous_four_axis_backbone_only_not_primary", "historical warmstate classification differs")
        output = Path(self.output_root)
        _need(output.resolve() == (ROOT / "outputs" / "official-score-matrix-v1").resolve(), "matrix output root differs")
        _need(Path(self.bootstrap_selection_path).resolve() == (output / "bootstrap_selection.json").resolve(), "bootstrap selection path differs")
        _need(Path(self.restricted_bootstrap_output_root).resolve() == (ROOT / "data" / "processed" / "restricted" / "official_prompt_alignment_v1" / "score_matrix_bootstrap" / self.run_id).resolve(), "restricted bootstrap output root differs")
        _need(
            Path(self.aihub_bounded_artifact_path).resolve() != Path(self.historical_warmstate_path).resolve()
            and Path(self.aihub_ordinal_artifact_path).resolve() != Path(self.historical_warmstate_path).resolve(),
            "historical continuous four-axis warmstate cannot initialize a primary arm",
        )
        pretrain = ROOT / "outputs" / "official-aihub-integer-score-full-pretrain-v1" / "official-aihub-integer-score-full-pretrain-v1-20260727-002"
        for head, completion, artifact in (
            ("bounded_regression", self.aihub_bounded_completion_path, self.aihub_bounded_artifact_path),
            ("ordinal_cumulative", self.aihub_ordinal_completion_path, self.aihub_ordinal_artifact_path),
        ):
            _need(Path(completion).resolve() == (pretrain / f"{head}-refit" / "training_complete.json").resolve(), f"{head} AI-Hub completion path differs")
            _need(Path(artifact).resolve() == (pretrain / f"{head}-refit" / "full_model").resolve(), f"{head} AI-Hub artifact path differs")

    def _aihub_dependency(self, head: str) -> tuple[Path, str, Path, str]:
        _need(head in HEADS, "AI-Hub dependency head differs")
        if head == "bounded_regression":
            return Path(self.aihub_bounded_completion_path), self.aihub_bounded_completion_sha256, Path(self.aihub_bounded_artifact_path), self.aihub_bounded_artifact_sha256
        return Path(self.aihub_ordinal_completion_path), self.aihub_ordinal_completion_sha256, Path(self.aihub_ordinal_artifact_path), self.aihub_ordinal_artifact_sha256

    def validate_dependencies(self, stage: str, head: str | None = None, *, require_aihub: bool = True) -> None:
        """Validate only dependencies reachable in the requested bootstrap stage."""
        self.validate()
        _need(stage in {"bootstrap", "rationale", "all"}, "dependency stage differs")
        restricted = (ROOT / "data" / "processed" / "restricted").resolve()
        dependencies = (
            (Path(self.model_path), None, "model snapshot", True),
            (Path(self.train_path), self.train_sha256, "canonical train", False),
            (Path(self.validation_path), self.validation_sha256, "canonical validation", False),
        )
        for path, expected_sha, label, directory in dependencies:
            _need((path.is_dir() if directory else path.is_file()) and not path.is_symlink(), f"{label} is unavailable")
            if expected_sha is not None:
                _need(len(expected_sha) == 64, f"{label} checksum is unresolved")
                _need(file_sha256(path) == expected_sha, f"{label} checksum differs")
        heads = (HEADS if head is None else (head,)) if require_aihub else ()
        for dependency_head in heads:
            completion_path, completion_sha, artifact_path, expected_sha = self._aihub_dependency(dependency_head)
            _need(completion_path.is_file() and artifact_path.is_dir() and not completion_path.is_symlink() and not artifact_path.is_symlink(), f"{dependency_head} integer AI-Hub warmstate is unavailable")
            _need(len(completion_sha) == 64 and file_sha256(completion_path) == completion_sha, f"{dependency_head} AI-Hub completion checksum differs")
            _need(len(expected_sha) == 64, f"{dependency_head} AI-Hub artifact checksum is unresolved")
            marker = os.environ.get(f"MAL2026_VERIFIED_AIHUB_{dependency_head.upper()}_SHA256")
            metadata = _read_json(completion_path, f"{dependency_head} AI-Hub completion")
            state = metadata.get("state")
            _need(isinstance(state, dict) and state.get("artifact_sha256") == expected_sha, "AI-Hub completion/artifact digest differs")
            inventory = state.get("inventory")
            _need(marker == expected_sha or artifact_inventory_sha256(artifact_path, inventory) == expected_sha, f"{dependency_head} AI-Hub artifact checksum differs")
            _need(metadata.get("schema_version") == "mal2026-aihub-integer-score-pretrain-completion-v2" and metadata.get("status") == "completed" and metadata.get("phase") == "refit", "AI-Hub completion identity differs")
            _need(metadata.get("head") == dependency_head and metadata.get("score_fields") == list(AXES), "AI-Hub head/axis lineage differs")
            _need(metadata.get("integer_target_used") is True and metadata.get("average_target_used") is False and metadata.get("target_projection") == "official_half_up", "AI-Hub warmstate is not integer three-axis only")
            _need(state.get("schema_version") == "mal2026-aihub-integer-score-full-state-v2" and state.get("head") == dependency_head, "AI-Hub full-state identity differs")
            _need(state.get("model_id") == self.model_id and state.get("model_revision") == self.model_revision, "AI-Hub full-state model pin differs")
            _need(state.get("score_fields") == list(AXES) and state.get("integer_target_used") is True and state.get("average_target_used") is False and state.get("target_projection") == "official_half_up", "AI-Hub full-state score contract differs")
            _need(metadata.get("training_method") == "full_parameter" and state.get("training_method") == "full_parameter" and state.get("state_scope") == "complete_full_parameter_backbone_plus_matched_score_head", "AI-Hub artifact is not a full-parameter matched-head refit")
            _need(Path(state.get("artifact_path", "")).resolve() == artifact_path.resolve(), "AI-Hub completion/artifact path differs")
            state_metadata = Path(state.get("metadata_path", ""))
            _need(state_metadata.is_file() and file_sha256(state_metadata) == state.get("metadata_sha256"), "AI-Hub full-state metadata checksum differs")
            expected_head_shape = [3, 4096] if dependency_head == "bounded_regression" else [12, 4096]
            _need(state.get("score_head_tensor_shapes") == {"score_head.bias": [expected_head_shape[0]], "score_head.weight": expected_head_shape}, "AI-Hub matched score-head shape differs")
            _need(type(state.get("backbone_tensor_count")) is int and state["backbone_tensor_count"] > 0 and isinstance(state.get("score_head_state_sha256"), str) and len(state["score_head_state_sha256"]) == 64, "AI-Hub full-state tensor audit differs")
        if stage in {"rationale", "all"}:
            rationale_paths = (Path(self.rationale_train_path), Path(self.rationale_validation_path), Path(self.rationale_manifest_path))
            _need(all(path.resolve().is_relative_to(restricted) for path in rationale_paths), "rationale artifacts must remain under the restricted root")
            rationale_dependencies = (
                (Path(self.bootstrap_selection_path), self.bootstrap_selection_sha256, "bootstrap selection"),
                (Path(self.rationale_train_path), self.rationale_train_sha256, "selected train rationale"),
                (Path(self.rationale_validation_path), self.rationale_validation_sha256, "selected validation rationale"),
                (Path(self.rationale_manifest_path), self.rationale_manifest_sha256, "selected rationale manifest"),
            )
            for path, expected_sha, label in rationale_dependencies:
                _need(path.is_file() and not path.is_symlink(), f"{label} is unavailable")
                _need(len(expected_sha) == 64 and file_sha256(path) == expected_sha, f"{label} checksum differs")
            _need(self.rationale_key and not self.rationale_key.startswith("REQUIRED_"), "selected rationale key is unresolved")
            bootstrap = _read_json(Path(self.bootstrap_selection_path), "bootstrap selection")
            _need(bootstrap.get("status") == "stage_a_completed" and bootstrap.get("selection_source") == "train_internal_dev_only", "bootstrap selection contract differs")
            rationale_manifest = _read_json(Path(self.rationale_manifest_path), "selected rationale manifest")
            _need(rationale_manifest.get("schema_version") == "mal2026-official-rationale-score-matrix-handoff-v1" and rationale_manifest.get("status") == "completed", "rationale handoff manifest identity differs")
            _need(rationale_manifest.get("structure") in {"bundle", "axis_triplet"} and rationale_manifest.get("score_kind") == "bootstrap_model_actual_emitted_integer_prediction" and rationale_manifest.get("human_or_reference_score_read_or_prompted") is False, "rationale handoff data contract differs")
            _need(rationale_manifest.get("judge_contract_sha256") == file_sha256(ROOT / "src" / "mal2026" / "official_writing_contract.py") and rationale_manifest.get("judge_model_sha256") == "b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772", "rationale handoff fixed judge differs")
            _need(all(isinstance(rationale_manifest.get(key), str) and len(rationale_manifest[key]) == 64 for key in ("winner_selection_sha256", "winner_candidate_identity_sha256", "winner_evaluation_sha256", "directional_gate_sha256", "injection_gate_sha256")), "rationale handoff provenance SHA differs")
            selected_scores = bootstrap.get("selected_score_files")
            _need(isinstance(selected_scores, dict), "bootstrap selected score binding differs")
            selected_result = Path(bootstrap.get("selected_result_path", ""))
            _need(selected_result.is_file() and selected_result.resolve().is_relative_to(Path(self.output_root).resolve()) and file_sha256(selected_result) == bootstrap.get("selected_result_sha256"), "bootstrap selected result SHA differs")
            for split in ("train", "validation"):
                score_path = Path(selected_scores.get(f"{split}_path", ""))
                _need(score_path.is_file() and score_path.resolve().is_relative_to(restricted) and file_sha256(score_path) == selected_scores.get(f"{split}_sha256"), f"bootstrap {split} score SHA differs")
            expected_binding = {
                "bootstrap_selection_sha256": self.bootstrap_selection_sha256,
                "bootstrap_selected_result_sha256": bootstrap.get("selected_result_sha256"),
                "score_train_sha256": selected_scores.get("train_sha256"),
                "score_validation_sha256": selected_scores.get("validation_sha256"),
                "rationale_train_sha256": self.rationale_train_sha256,
                "rationale_validation_sha256": self.rationale_validation_sha256,
                "rationale_key": self.rationale_key,
            }
            _need(all(rationale_manifest.get(key) == value for key, value in expected_binding.items()), "rationale/bootstrap SHA binding differs")


@dataclass(frozen=True)
class ScoreRow:
    identifier: str
    document_id: str
    prompt_num: str
    prompt: str
    essay: str
    labels: tuple[int, int, int]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialScoreMatrixError(f"{label} is unreadable") from exc
    _need(isinstance(value, dict), f"{label} must be an object")
    return value


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            _need(bool(line.strip()), f"blank JSONL line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OfficialScoreMatrixError(f"invalid JSONL line {line_number}") from exc
            _need(isinstance(value, dict), f"JSONL line {line_number} must be an object")
            yield value


def load_score_rows(path: Path, expected_sha: str, expected_count: int) -> list[ScoreRow]:
    """Load analytic labels without ever indexing the source average value."""
    _need(path.is_file() and file_sha256(path) == expected_sha, "canonical score source differs")
    result: list[ScoreRow] = []
    seen: set[str] = set()
    for raw in _jsonl(path):
        _need(set(raw) == {"id", "document_id", "prompt_num", "prompt", "essay", "score"}, "canonical row schema differs")
        identifier = raw["id"]
        _need(isinstance(identifier, str) and identifier and identifier not in seen, "canonical row ID differs")
        seen.add(identifier)
        scores = raw["score"]
        _need(isinstance(scores, dict) and set(scores) == {*AXES, "average"}, "canonical score keys differ")
        # Deliberately do not access scores["average"].  It is neither a feature
        # nor a target anywhere in this experiment.
        labels: list[int] = []
        for axis in AXES:
            value = scores[axis]
            _need(type(value) in {int, float} and not isinstance(value, bool), f"{axis} score is nonnumeric")
            _need(math.isfinite(float(value)) and 1.0 <= float(value) <= 5.0, f"{axis} score is outside [1,5]")
            labels.append(official_half_up(float(value)))
        text_fields = (raw["document_id"], raw["prompt_num"], raw["prompt"], raw["essay"])
        _need(all(isinstance(item, (str, int)) for item in text_fields[:2]) and all(isinstance(item, str) and item.strip() for item in text_fields[2:]), "canonical text/group fields differ")
        result.append(ScoreRow(identifier, str(raw["document_id"]), str(raw["prompt_num"]), raw["prompt"], raw["essay"], tuple(labels)))  # type: ignore[arg-type]
    _need(len(result) == expected_count, "canonical score row count differs")
    return result


def deterministic_internal_split(rows: Sequence[ScoreRow], seed: int) -> tuple[list[ScoreRow], list[ScoreRow], str]:
    """Exact prompt-stratified 80/20 split by indivisible document groups."""
    _need(len(rows) == 2000, "internal split requires all 2,000 train essays")
    groups: dict[tuple[str, str], list[ScoreRow]] = {}
    for row in rows:
        groups.setdefault((row.prompt_num, row.document_id), []).append(row)
    _need(all(len(group) == 1 for group in groups.values()), "canonical document_id groups are unexpectedly non-unique")
    prompt_counts: dict[str, int] = {}
    for prompt_num, _ in groups:
        prompt_counts[prompt_num] = prompt_counts.get(prompt_num, 0) + 1
    quotas = {prompt: int(count * 0.2) for prompt, count in prompt_counts.items()}
    remainder = 400 - sum(quotas.values())
    ranked_remainders = sorted(
        prompt_counts,
        key=lambda prompt: (-(prompt_counts[prompt] * 0.2 - quotas[prompt]), sha256(f"{seed}\0{prompt}".encode()).hexdigest()),
    )
    for prompt in ranked_remainders[:remainder]:
        quotas[prompt] += 1
    dev_keys: set[tuple[str, str]] = set()
    for prompt, quota in quotas.items():
        ranked = sorted(
            (key for key in groups if key[0] == prompt),
            key=lambda key: sha256(f"{seed}\0{key[0]}\0{key[1]}".encode()).hexdigest(),
        )
        dev_keys.update(ranked[:quota])
    _need(len(dev_keys) == 400, "prompt-stratified split cannot realize exact 80/20 counts")
    train = [row for row in rows if (row.prompt_num, row.document_id) not in dev_keys]
    dev = [row for row in rows if (row.prompt_num, row.document_id) in dev_keys]
    _need((len(train), len(dev)) == (1600, 400), "internal split counts differ")
    _need(not ({(r.prompt_num, r.document_id) for r in train} & {(r.prompt_num, r.document_id) for r in dev}), "internal split leaks a group")
    fingerprint = sha256("\n".join(sorted(sha256(f"{r.prompt_num}\0{r.document_id}".encode()).hexdigest() for r in dev)).encode()).hexdigest()
    return train, dev, fingerprint


def load_rationales(path: Path, expected_sha: str, rows: Sequence[ScoreRow]) -> dict[str, Mapping[str, str]]:
    _need(path.is_file() and file_sha256(path) == expected_sha, "selected rationale source differs")
    expected_ids = {row.identifier for row in rows}
    result: dict[str, Mapping[str, str]] = {}
    for raw in _jsonl(path):
        _need(set(raw) >= {"source_id", "rationales"}, "rationale schema differs")
        source_id, rationales = raw["source_id"], raw["rationales"]
        _need(source_id in expected_ids and source_id not in result, "rationale linkage differs")
        _need(isinstance(rationales, dict) and set(rationales) == set(AXES), "rationale axes differ")
        _need(all(isinstance(rationales[axis], str) and rationales[axis].strip() for axis in AXES), "rationale text differs")
        result[source_id] = {axis: rationales[axis] for axis in AXES}
    _need(set(result) == expected_ids, "rationale population is incomplete")
    return result


def render_input(row: ScoreRow, input_view: str, rationales: Mapping[str, str] | None = None) -> str:
    _need(input_view in INPUTS, "input view differs")
    prefix = "Instruct: Predict three integer Korean writing scores (content, organization, expression), each from 1 to 5.\nQuery:\n"
    text = f"<writing_prompt>\n{row.prompt}\n</writing_prompt>\n<student_essay>\n{row.essay}\n</student_essay>"
    if input_view == "rationale":
        _need(rationales is not None and set(rationales) == set(AXES), "rationale input is unavailable")
        text += "\n<evaluation_rationales>\n" + "\n".join(f"<{axis}>{rationales[axis]}</{axis}>" for axis in AXES) + "\n</evaluation_rationales>"
    return prefix + text


def ordinal_targets(labels: Any) -> Any:
    import torch
    values = labels.long().unsqueeze(-1)
    thresholds = torch.arange(1, 5, device=labels.device).view(1, 1, 4)
    return (values > thresholds).float()


def decode_logits(logits: Any, head: str) -> tuple[Any, Any, Any]:
    """Return continuous scores, half-up integers, and ordinal violation mask."""
    import torch
    _need(head in HEADS, "head differs")
    if head == "bounded_regression":
        _need(logits.ndim == 2 and logits.shape[-1] == 3, "bounded regression logits differ")
        continuous = 1.0 + 4.0 * torch.sigmoid(logits.float())
        violations = torch.zeros(logits.shape[0], 3, 0, dtype=torch.bool, device=logits.device)
    else:
        _need(logits.ndim == 2 and logits.shape[-1] == 12, "ordinal logits differ")
        probabilities = torch.sigmoid(logits.float().reshape(-1, 3, 4))
        violations = probabilities[:, :, 1:] > probabilities[:, :, :-1]
        projected = torch.cummin(probabilities, dim=-1).values
        continuous = 1.0 + projected.sum(dim=-1)
    integers = torch.floor(continuous + 0.5).clamp(1, 5).to(torch.int64)
    return continuous, integers, violations


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for index in order[cursor:end]:
            result[index] = rank
        cursor = end
    return result


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = _rank(left), _rank(right)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def score_metrics(labels: Sequence[Sequence[float]], continuous: Sequence[Sequence[float]], integers: Sequence[Sequence[int]], violations: Sequence[Sequence[Sequence[bool]]] | None = None) -> dict[str, Any]:
    _need(len(labels) == len(continuous) == len(integers) and bool(labels), "metric population differs")
    result: dict[str, Any] = {}
    for index, axis in enumerate(AXES):
        truth = [float(row[index]) for row in labels]
        pred = [float(row[index]) for row in continuous]
        integer = [float(row[index]) for row in integers]
        result[axis] = {
            "continuous_rmse": math.sqrt(sum((a - b) ** 2 for a, b in zip(truth, pred)) / len(truth)),
            "integer_rmse": math.sqrt(sum((a - b) ** 2 for a, b in zip(truth, integer)) / len(truth)),
            "continuous_spearman": _spearman(truth, pred),
            "integer_spearman": _spearman(truth, integer),
            "integer_accuracy": sum(a == b for a, b in zip(truth, integer)) / len(truth),
        }
    result["macro_continuous_rmse"] = sum(result[axis]["continuous_rmse"] for axis in AXES) / 3
    result["macro_integer_rmse"] = sum(result[axis]["integer_rmse"] for axis in AXES) / 3
    result["macro_continuous_spearman"] = sum(result[axis]["continuous_spearman"] for axis in AXES) / 3
    result["macro_integer_spearman"] = sum(result[axis]["integer_spearman"] for axis in AXES) / 3
    flat = [bool(value) for row in (violations or []) for axis in row for value in axis]
    result["ordinal_monotonic_violation_count"] = sum(flat)
    result["ordinal_monotonic_violation_rate"] = sum(flat) / len(flat) if flat else 0.0
    result["ordinal_monotonic_projection_applied"] = bool(flat)
    return result


def select_epoch(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Apply the frozen integer-primary internal-dev selection rule."""
    _need(bool(rows), "selection rows are empty")
    return min(
        rows,
        key=lambda row: (
            float(row["macro_integer_rmse"]),
            -float(row["macro_integer_spearman"]),
            float(row["macro_continuous_rmse"]),
            int(row["epoch"]),
        ),
    )


def select_bootstrap_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Choose among essay-only arms without consulting canonical validation."""
    _need(bool(rows), "bootstrap candidates are empty")
    return min(
        rows,
        key=lambda row: (
            float(row["macro_integer_rmse"]),
            -float(row["macro_integer_spearman"]),
            float(row["macro_continuous_rmse"]),
            str(row["arm"]),
        ),
    )


def _examples(rows: Sequence[ScoreRow], input_view: str, rationales: Mapping[str, Mapping[str, str]] | None) -> list[dict[str, Any]]:
    return [{"text": render_input(row, input_view, None if rationales is None else rationales[row.identifier]), "labels": list(row.labels)} for row in rows]


def _dataset(items: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int) -> Any:
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": [item["text"] for item in items], "labels": [item["labels"] for item in items]})
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])


def _collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        labels = [feature.pop("labels") for feature in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch
    return collate


def build_model(config: MatrixConfig, head: str, initialization: str) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from peft import LoraConfig, TaskType, get_peft_model
    from safetensors import safe_open
    from transformers import AutoModel

    _need(head in HEADS and initialization in INITIALIZATIONS, "model arm differs")
    base = AutoModel.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    base.config.use_cache = False
    hidden = getattr(base.config, "hidden_size", None)
    _need(type(hidden) is int and hidden > 0, "embedding hidden size is unavailable")

    class Model(nn.Module):
        def __init__(self, backbone: Any) -> None:
            super().__init__()
            self.backbone = backbone
            self.score_head = nn.Linear(hidden, 3 if head == "bounded_regression" else 12)

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None, **_: Any) -> Mapping[str, Any]:
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
            positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
            index = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
            _need(bool((index >= 0).all().item()), "an input has no nonpad token")
            pooled = functional.normalize(
                output[torch.arange(output.shape[0], device=output.device), index], p=2, dim=-1
            )
            logits = self.score_head(pooled.to(self.score_head.weight.dtype)).float()
            result: dict[str, Any] = {"logits": logits}
            if labels is not None:
                if head == "bounded_regression":
                    prediction, _, _ = decode_logits(logits, head)
                    result["loss"] = functional.mse_loss(prediction, labels.float())
                else:
                    result["loss"] = functional.binary_cross_entropy_with_logits(logits.reshape(-1, 3, 4), ordinal_targets(labels))
            return result

    model = Model(base)
    provenance = {"initialization": initialization, "source_score_head_loaded": initialization == "aihub_matched_full"}
    if initialization == "aihub_matched_full":
        completion_path, _, artifact_path, expected_sha = config._aihub_dependency(head)
        completion = _read_json(completion_path, "AI-Hub matched-full completion")
        source_head_sha = completion.get("state", {}).get("score_head_state_sha256")
        targets = {**dict(model.named_parameters()), **dict(model.named_buffers())}
        parameter_names = set(dict(model.named_parameters()))
        loaded_names: set[str] = set()
        loaded_head: set[str] = set()
        source_head_tensors: dict[str, Any] = {}
        tensor_files = sorted(artifact_path.rglob("*.safetensors"))
        _need(bool(tensor_files), "matched-full AI-Hub artifact has no safetensors")
        for tensor_file in tensor_files:
            with safe_open(tensor_file, framework="pt", device="cpu") as handle:
                for name in sorted(handle.keys()):
                    _need(name in targets and name not in loaded_names, f"AI-Hub artifact tensor differs: {name}")
                    _need(name.startswith("backbone.") or name.startswith("score_head."), "AI-Hub artifact contains an out-of-scope tensor")
                    tensor = handle.get_tensor(name)
                    _need(tuple(tensor.shape) == tuple(targets[name].shape), f"AI-Hub artifact shape differs: {name}")
                    targets[name].data.copy_(tensor.to(dtype=targets[name].dtype))
                    loaded_names.add(name)
                    if name.startswith("score_head."):
                        loaded_head.add(name)
                        source_head_tensors[name] = tensor.detach().cpu().contiguous()
        _need(parameter_names <= loaded_names and loaded_head == {"score_head.weight", "score_head.bias"}, "matched-full AI-Hub model is incomplete")
        source_head_digest = sha256()
        for name, source in sorted(source_head_tensors.items()):
            source_head_digest.update(name.encode())
            source_head_digest.update(str(source.dtype).encode())
            source_head_digest.update(json.dumps(list(source.shape)).encode())
            source_head_digest.update(source.view(torch.uint8).numpy().tobytes())
        _need(source_head_digest.hexdigest() == source_head_sha, "matched AI-Hub source score-head hash differs")
        provenance.update({
            "load_mode": "full_backbone_and_matched_head_then_fresh_mal_lora",
            "matched_head": head,
            "full_parameter_tensor_count_loaded": len(loaded_names),
            "completion_path": str(completion_path.resolve()),
            "artifact_path": str(artifact_path.resolve()),
            "artifact_sha256": expected_sha,
            "source_score_head_tensor_count_loaded": len(loaded_head),
            "source_score_head_sha256": source_head_sha,
            "integer_target_used": True,
            "average_target_used": False,
        })
    model.backbone = get_peft_model(model.backbone, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, target_modules=list(LORA_TARGETS), bias="none",
    ))
    _need(all(parameter.requires_grad for parameter in model.score_head.parameters()), "matched score head must remain trainable")
    _need(any("lora_" in name and parameter.requires_grad for name, parameter in model.named_parameters()), "fresh MAL LoRA is absent")
    _need(not any(parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("backbone.") and "lora_" not in name), "full AI-Hub backbone must be frozen during MAL continuation")
    return model, provenance


def _trainable_state(model: Any) -> dict[str, Any]:
    state = {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters() if parameter.requires_grad}
    _need(any(name.startswith("score_head.") for name in state), "fresh score head is absent")
    _need(not any("average" in name or name.startswith("regression_head.") for name in state), "forbidden historical head/average state leaked")
    return state


def score_head_initial_sha256(model: Any) -> str:
    """Hash only the freshly initialized score head, independent of backbone."""
    import torch
    digest = sha256()
    head = getattr(model, "score_head", None)
    _need(head is not None, "score head is unavailable")
    tensors = [(f"score_head.{name}", tensor) for name, tensor in head.state_dict().items()]
    _need({name for name, _ in tensors} == {"score_head.weight", "score_head.bias"}, "score head state differs")
    for name, tensor in sorted(tensors):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _predict(trainer: Any, dataset: Any, head: str) -> tuple[dict[str, Any], list[list[float]], list[list[int]], list[list[list[bool]]]]:
    import torch
    prediction = trainer.predict(dataset)
    logits = prediction.predictions
    continuous, integers, violations = decode_logits(torch.as_tensor(logits), head)
    labels = prediction.label_ids.tolist()
    continuous_list = continuous.cpu().tolist()
    integer_list = integers.cpu().tolist()
    violation_list = violations.cpu().tolist()
    return score_metrics(labels, continuous_list, integer_list, violation_list), continuous_list, integer_list, violation_list


def write_integer_scores(path: Path, rows: Sequence[ScoreRow], integers: Sequence[Sequence[int]], arm: str, split: str) -> str:
    """Persist only identifiers and emitted three-axis integers in restricted storage."""
    _need(split in {"train", "validation"} and len(rows) == len(integers), "integer score emission population differs")
    _need(not path.exists(), "refusing to replace emitted integer scores")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _need(not temporary.exists(), "integer score temporary already exists")
    with temporary.open("x", encoding="utf-8") as handle:
        for row, vector in zip(rows, integers, strict=True):
            _need(len(vector) == 3 and all(type(value) is int and 1 <= value <= 5 for value in vector), "emitted score is not an official integer")
            payload = {"source_id": row.identifier, "split": split, "arm": arm, "scores": {axis: vector[index] for index, axis in enumerate(AXES)}}
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)
    return file_sha256(path)


def run_arm(config: MatrixConfig, arm: str, *, smoke: bool = False) -> dict[str, Any]:
    """Run one arm.  Intended to be invoked under torchrun for full runs."""
    head, initialization, input_view = parse_arm(arm)
    config.validate_dependencies("bootstrap" if input_view == "essay" else "rationale", head=head, require_aihub=initialization == "aihub_matched_full")
    try:
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("official score matrix requires .venv-standard") from exc

    output = Path(config.output_root) / (f"smoke-{arm}" if smoke else arm)
    process_rank = int(os.environ.get("RANK", "0"))
    if process_rank == 0:
        _need(not output.exists(), "refusing to reuse an arm output")
        output.mkdir(parents=True)
    else:
        deadline = time.monotonic() + 60
        while not output.is_dir() and time.monotonic() < deadline:
            time.sleep(0.05)
        _need(output.is_dir(), "rank zero did not create the arm output")
    train_rows = load_score_rows(Path(config.train_path), config.train_sha256, 2000)
    selection_train, selection_dev, split_fingerprint = deterministic_internal_split(train_rows, config.seed)
    if smoke:
        selection_train, selection_dev = selection_train[:4], selection_dev[:4]
    rationales_train = load_rationales(Path(config.rationale_train_path), config.rationale_train_sha256, train_rows) if input_view == "rationale" else None
    # Selection and refit deliberately repeat this exact RNG -> tokenizer ->
    # model -> data order so both start from the same immutable initialization.
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        _need(tokenizer.eos_token is not None, "tokenizer has no pad/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model, initialization_provenance = build_model(config, head, initialization)
    selection_head_initial_sha256 = score_head_initial_sha256(model)
    train_dataset = _dataset(_examples(selection_train, input_view, rationales_train), tokenizer, config.max_length)
    dev_dataset = _dataset(_examples(selection_dev, input_view, rationales_train), tokenizer, config.max_length)
    epoch_rows: list[dict[str, Any]] = []
    selection_root = output / "selection"

    def compute_metrics(result: Any) -> dict[str, float]:
        continuous, integers, violations = decode_logits(torch.as_tensor(result.predictions), head)
        metrics = score_metrics(result.label_ids.tolist(), continuous.tolist(), integers.tolist(), violations.tolist())
        return {
            "macro_integer_rmse": metrics["macro_integer_rmse"],
            "macro_integer_spearman": metrics["macro_integer_spearman"],
            "macro_continuous_rmse": metrics["macro_continuous_rmse"],
            "macro_continuous_spearman": metrics["macro_continuous_spearman"],
        }

    class EpochArtifact(TrainerCallback):
        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Mapping[str, Any] | None = None, model: Any | None = None, **_: Any) -> Any:
            epoch = int(round(float(state.epoch or 0)))
            if not state.is_world_process_zero:
                return control
            _need(epoch in config.selection_epochs or smoke, "selection epoch boundary differs")
            path = selection_root / f"epoch-{epoch:02d}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            _need(not path.exists() and model is not None, "selection checkpoint differs")
            save_file(_trainable_state(model), str(path))
            epoch_rows.append({
                "epoch": epoch, "global_step": int(state.global_step),
                "macro_integer_rmse": float((metrics or {})["eval_macro_integer_rmse"]),
                "macro_integer_spearman": float((metrics or {})["eval_macro_integer_spearman"]),
                "macro_continuous_rmse": float((metrics or {})["eval_macro_continuous_rmse"]),
                "macro_continuous_spearman": float((metrics or {})["eval_macro_continuous_spearman"]),
                "state_path": str(path.resolve()), "state_sha256": file_sha256(path),
            })
            return control

    selection_args = TrainingArguments(
        output_dir=str(selection_root), do_train=True, do_eval=True, eval_strategy="epoch", save_strategy="no",
        num_train_epochs=1 if smoke else 4, max_steps=1 if smoke else -1, learning_rate=config.learning_rate,
        weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio, per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size, gradient_accumulation_steps=1 if smoke else config.gradient_accumulation_steps,
        bf16=True, tf32=True, report_to=[], remove_unused_columns=False, dataloader_num_workers=0, ddp_find_unused_parameters=False,
        logging_steps=1 if smoke else 5, save_only_model=True, seed=config.seed, data_seed=config.seed,
    )
    selector = Trainer(model=model, args=selection_args, train_dataset=train_dataset, eval_dataset=dev_dataset, data_collator=_collator(tokenizer), compute_metrics=compute_metrics, callbacks=[EpochArtifact()])
    selector.train()
    selector.accelerator.wait_for_everyone()
    shared_selection: list[Any] = [epoch_rows if selector.is_world_process_zero() else None]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(shared_selection, src=0)
    epoch_rows = shared_selection[0]
    _need(isinstance(epoch_rows, list) and epoch_rows, "selection emitted no epoch metrics")
    best = select_epoch(epoch_rows)
    if smoke:
        payload = {"status": "completed", "mode": "gpu0_smoke", "arm": arm, "score_fields": list(AXES), "average_read": False, "average_target_used": False, "split_fingerprint": split_fingerprint, "selection": epoch_rows, "initialization": initialization_provenance, "score_head_initial_sha256": selection_head_initial_sha256}
        if selector.is_world_process_zero():
            (output / "smoke_complete.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload

    # Refit is deliberately reinitialized and sees all 2,000 train essays.
    del selector, model, tokenizer, train_dataset, dev_dataset
    torch.cuda.empty_cache()
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, revision=config.model_revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        _need(tokenizer.eos_token is not None, "tokenizer has no pad/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    refit_model, refit_initialization = build_model(config, head, initialization)
    refit_head_initial_sha256 = score_head_initial_sha256(refit_model)
    _need(refit_head_initial_sha256 == selection_head_initial_sha256, "selection/refit score-head initialization differs")
    full_dataset = _dataset(_examples(train_rows, input_view, rationales_train), tokenizer, config.max_length)
    refit_root = output / "refit"
    refitter = Trainer(model=refit_model, args=TrainingArguments(
        output_dir=str(refit_root), do_train=True, do_eval=False, eval_strategy="no", save_strategy="no", num_train_epochs=float(best["epoch"]),
        learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size, gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=True, tf32=True, report_to=[], remove_unused_columns=False, dataloader_num_workers=0, ddp_find_unused_parameters=False,
        logging_steps=5, save_only_model=True, seed=config.seed, data_seed=config.seed,
    ), train_dataset=full_dataset, data_collator=_collator(tokenizer))
    trained = refitter.train()
    refitter.accelerator.wait_for_everyone()
    final_state = output / "selected_refit_trainable.safetensors"
    if refitter.is_world_process_zero():
        save_file(_trainable_state(refit_model), str(final_state))
    refitter.accelerator.wait_for_everyone()

    # Bootstrap models also emit all-train integers for the downstream
    # rationale generator.  These predictions never participate in selection.
    train_prediction_metrics: dict[str, Any] | None = None
    train_integer_predictions: list[list[int]] | None = None
    if input_view == "essay":
        train_prediction_metrics, _, train_integer_predictions, _ = _predict(refitter, full_dataset, head)

    # Canonical validation is intentionally first loaded here and evaluated once.
    validation_rows = load_score_rows(Path(config.validation_path), config.validation_sha256, 400)
    rationales_validation = load_rationales(Path(config.rationale_validation_path), config.rationale_validation_sha256, validation_rows) if input_view == "rationale" else None
    validation_dataset = _dataset(_examples(validation_rows, input_view, rationales_validation), tokenizer, config.max_length)
    final_metrics, _, validation_integer_predictions, _ = _predict(refitter, validation_dataset, head)
    emitted: dict[str, Any] | None = None
    if input_view == "essay":
        _need(train_integer_predictions is not None, "bootstrap train predictions are absent")
        restricted_root = Path(config.restricted_bootstrap_output_root) / arm
        train_scores = restricted_root / "scores.train.jsonl"
        validation_scores = restricted_root / "scores.validation.jsonl"
        if refitter.is_world_process_zero():
            train_score_sha = write_integer_scores(train_scores, train_rows, train_integer_predictions, arm, "train")
            validation_score_sha = write_integer_scores(validation_scores, validation_rows, validation_integer_predictions, arm, "validation")
        else:
            train_score_sha = validation_score_sha = None
        shared_scores: list[Any] = [{"train_sha256": train_score_sha, "validation_sha256": validation_score_sha} if refitter.is_world_process_zero() else None]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(shared_scores, src=0)
        score_hashes = shared_scores[0]
        _need(isinstance(score_hashes, dict), "bootstrap score SHA broadcast differs")
        emitted = {
            "train_path": str(train_scores.resolve()), "train_sha256": score_hashes["train_sha256"], "train_records": 2000,
            "validation_path": str(validation_scores.resolve()), "validation_sha256": score_hashes["validation_sha256"], "validation_records": 400,
            "schema": "source_id,split,arm,scores{content,organization,expression}; official integer 1..5",
        }
    payload = {
        "status": "completed", "run_id": config.run_id, "arm": arm, "head": head, "initialization": initialization,
        "input_view": input_view, "score_fields": list(AXES), "average_read": False, "average_target_used": False,
        "integer_projection": "official_half_up", "ordinal_projection": "cumulative_minimum_nonincreasing",
        "internal_split": {"train": 1600, "dev": 400, "group_fields": ["prompt_num", "document_id"], "fingerprint": split_fingerprint},
        "selection": {
            "epochs": epoch_rows,
            "rule": "lowest internal-dev macro integer RMSE, then highest integer Spearman, then lowest continuous RMSE, then earlier epoch",
            "selected_epoch": best["epoch"], "selected_event": best, "score_head_initial_sha256": selection_head_initial_sha256,
        },
        "refit": {"records": 2000, "epochs": best["epoch"], "train_metrics": {k: float(v) for k, v in trained.metrics.items() if isinstance(v, (int, float))}, "state_path": str(final_state.resolve()), "state_sha256": file_sha256(final_state)},
        "canonical_validation": {"use": "single_final_descriptive_evaluation_not_selection", "records": 400, "metrics": final_metrics},
        "bootstrap_train_prediction": None if input_view != "essay" else {"use": "downstream_rationale_conditioning_only_not_selection", "records": 2000, "metrics": train_prediction_metrics},
        "emitted_integer_score_files": emitted,
        "rationale_source": None if input_view == "essay" else {"key": config.rationale_key, "train_sha256": config.rationale_train_sha256, "validation_sha256": config.rationale_validation_sha256},
        "initialization_provenance": refit_initialization,
        "initialization_replay": {"order": ["reset_rng", "load_tokenizer", "build_model", "build_data"], "selection_score_head_sha256": selection_head_initial_sha256, "refit_score_head_sha256": refit_head_initial_sha256, "equal": True},
        "config": asdict(config), "privacy": "aggregate_only_no_rows_text_ids_rationales_or_predictions_persisted",
    }
    if refitter.is_world_process_zero():
        (output / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    refitter.accelerator.wait_for_everyone()
    return payload
