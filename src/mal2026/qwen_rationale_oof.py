"""Train-only OOF experiments for rationale-aware Qwen score prediction.

The module deliberately keeps row-level labels, predictions, rationales and
identifiers under the restricted data root.  Public outputs contain aggregate
metrics and immutable lineage only.  Canonical validation is not parsed until
``refit_and_validate`` after all OOF choices have been frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
AXES = ("content", "organization", "expression")
STAGE1_ARMS = ("raw_mse", "rationale_mse", "dropout_mse")
STAGE2_ARMS = ("huber", "ordinal", "multitask", "multitask_tail")
INPUT_VARIANTS = ("raw", "rationale", "dropout")
LOSS_KINDS = ("mse", "huber", "ordinal", "multitask")


class QwenRationaleOOFError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise QwenRationaleOOFError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count, file_sha256(path)


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            need(bool(line.strip()), f"blank JSONL line {number}: {path}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QwenRationaleOOFError(f"invalid JSONL line {number}: {path}") from exc
            need(isinstance(value, dict), f"non-object JSONL line {number}: {path}")
            yield value


def resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Config:
    raw: Mapping[str, Any]
    path: Path

    @classmethod
    def from_json(cls, path: Path, *, verify_validation_hash: bool = True) -> "Config":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QwenRationaleOOFError("OOF config is unreadable") from exc
        need(isinstance(raw, dict), "OOF config must be an object")
        expected = {
            "schema_version", "run_id", "train_path", "train_sha256", "validation_path", "validation_sha256",
            "fold_rows_path", "fold_rows_sha256", "rationale_handoff_path", "rationale_handoff_sha256",
            "rationale_prompt_path", "rationale_prompt_sha256", "raw_prompt_path", "raw_prompt_sha256",
            "model_id", "model_revision", "model_path", "aihub_completion_path", "aihub_completion_sha256",
            "aihub_artifact_path", "aihub_artifact_sha256", "aihub_manifest_path", "baseline_result_path",
            "baseline_result_sha256", "output_root", "restricted_output_root", "seed", "fold_count", "epochs",
            "learning_rate", "weight_decay", "warmup_ratio", "max_length", "per_device_train_batch_size",
            "per_device_eval_batch_size", "gradient_accumulation_steps", "gradient_checkpointing", "lora_r",
            "lora_alpha", "lora_dropout", "huber_beta", "ordinal_auxiliary_weight",
            "rationale_dropout_probability", "tail_weight_cap", "stage1_arms", "stage2_arms",
            "aihub_tail_max_records", "primary_metric", "selection_tiebreakers", "target_rmse", "gpu_scope",
            "smoke_gpu", "validation_policy", "average_target_used", "user_authorization",
        }
        need(set(raw) == expected, "OOF config fields differ")
        value = cls(raw=raw, path=path.resolve())
        value.validate(verify_validation_hash=verify_validation_hash)
        return value

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def output_root(self) -> Path:
        return resolve(str(self["output_root"]))

    @property
    def restricted_root(self) -> Path:
        return resolve(str(self["restricted_output_root"]))

    @property
    def config_sha256(self) -> str:
        return file_sha256(self.path)

    def validate(self, *, verify_validation_hash: bool) -> None:
        need(self["schema_version"] == "mal2026-qwen-rationale-oof-multistage-v1", "OOF schema differs")
        authorized_runs = {
            "qwen-rationale-oof-multistage-v1-20260812-001": ([0, 1, 2, 3], 0),
            "qwen-rationale-oof-stage1-gpu7-v1-20260819-003": ([7], 7),
        }
        need(self["run_id"] in authorized_runs, "OOF run ID differs")
        need(self["model_id"] == "Qwen/Qwen3-Embedding-8B" and self["model_revision"] == "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", "model pin differs")
        need(self["fold_count"] == 5 and self["epochs"] == 7 and self["seed"] == 2026081201, "OOF fold/epoch/seed differs")
        need(self["stage1_arms"] == list(STAGE1_ARMS) and self["stage2_arms"] == list(STAGE2_ARMS), "OOF arm inventory differs")
        expected_scope, expected_smoke_gpu = authorized_runs[self["run_id"]]
        need(
            self["gpu_scope"] == expected_scope
            and self["smoke_gpu"] == expected_smoke_gpu
            and bool(str(self["user_authorization"]).strip()),
            "GPU authorization differs",
        )
        need(self["average_target_used"] is False and self["validation_policy"].startswith("never_train_or_select;"), "validation/average policy differs")
        need((self["learning_rate"], self["weight_decay"], self["warmup_ratio"]) == (1e-4, 0.01, 0.05), "optimizer contract differs")
        need((self["per_device_train_batch_size"], self["per_device_eval_batch_size"], self["gradient_accumulation_steps"]) == (8, 8, 4), "batch contract differs")
        need(self["gradient_checkpointing"] is True and self["max_length"] == 2560, "memory/length contract differs")
        need((self["lora_r"], self["lora_alpha"], self["lora_dropout"]) == (16, 32, 0.05), "LoRA contract differs")
        need((self["huber_beta"], self["ordinal_auxiliary_weight"], self["rationale_dropout_probability"], self["tail_weight_cap"]) == (0.5, 0.5, 0.5, 2.5), "loss ablation contract differs")
        need(self["aihub_tail_max_records"] == 2000 and self["primary_metric"] == "macro_integer_rmse", "selection contract differs")
        need(self["selection_tiebreakers"] == ["macro_tail_rmse", "negative_macro_integer_spearman", "arm"], "selection tiebreakers differ")
        need(float(self["target_rmse"]) == 0.4, "target RMSE differs")
        checks = (
            ("train_path", "train_sha256", False), ("fold_rows_path", "fold_rows_sha256", False),
            ("rationale_handoff_path", "rationale_handoff_sha256", False),
            ("rationale_prompt_path", "rationale_prompt_sha256", False), ("raw_prompt_path", "raw_prompt_sha256", False),
            ("aihub_completion_path", "aihub_completion_sha256", False),
            ("baseline_result_path", "baseline_result_sha256", False),
            ("validation_path", "validation_sha256", not verify_validation_hash),
        )
        for path_key, sha_key, skip in checks:
            if skip:
                continue
            path = resolve(str(self[path_key]))
            need(path.is_file() and file_sha256(path) == self[sha_key], f"dependency differs: {path_key}")
        model = resolve(str(self["model_path"])); artifact = resolve(str(self["aihub_artifact_path"]))
        need(model.is_dir() and (model / "config.json").is_file(), "model snapshot unavailable")
        need(artifact.is_dir() and any(artifact.glob("*.safetensors")), "AI-Hub artifact unavailable")
        need(resolve(str(self["aihub_manifest_path"])).is_file(), "AI-Hub manifest unavailable")
        need(self.output_root.resolve().is_relative_to((ROOT / "outputs").resolve()), "public output escaped outputs")
        need(self.restricted_root.resolve().is_relative_to((ROOT / "data/processed/restricted").resolve()), "restricted output escaped restricted root")


@dataclass(frozen=True)
class Row:
    identifier: str
    prompt: str
    essay: str
    labels: tuple[float, float, float]
    source: str = "mal"


def round_half_up(value: float) -> int:
    parsed = Decimal(str(value))
    need(parsed.is_finite() and Decimal("1") <= parsed <= Decimal("5"), "score outside [1,5]")
    return int(parsed.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def load_mal_rows(config: Config, *, validation: bool = False) -> list[Row]:
    path = resolve(str(config["validation_path"] if validation else config["train_path"]))
    expected_sha = str(config["validation_sha256"] if validation else config["train_sha256"])
    expected_count = 400 if validation else 2000
    need(file_sha256(path) == expected_sha, "canonical MAL checksum differs")
    result: list[Row] = []
    seen: set[str] = set()
    for raw in jsonl(path):
        need(set(raw) == {"id", "document_id", "prompt_num", "prompt", "essay", "score"}, "canonical MAL schema differs")
        identifier = raw["id"]
        need(isinstance(identifier, str) and identifier and identifier not in seen, "canonical MAL identifier differs")
        seen.add(identifier)
        scores = raw["score"]
        need(isinstance(scores, dict) and set(scores) == {*AXES, "average"}, "canonical MAL score fields differ")
        # Deliberately never index score.average.
        labels = tuple(float(scores[axis]) for axis in AXES)
        need(all(math.isfinite(value) and 1 <= value <= 5 for value in labels), "canonical MAL score differs")
        need(isinstance(raw["prompt"], str) and raw["prompt"].strip() and isinstance(raw["essay"], str) and raw["essay"].strip(), "canonical MAL text differs")
        result.append(Row(identifier, raw["prompt"], raw["essay"], labels))
    need(len(result) == expected_count, "canonical MAL population differs")
    return result


def load_fold_map(config: Config, expected_ids: set[str]) -> dict[str, int]:
    path = resolve(str(config["fold_rows_path"]))
    need(file_sha256(path) == config["fold_rows_sha256"], "fold rows checksum differs")
    result: dict[str, int] = {}
    for raw in jsonl(path):
        # Only the previously frozen source_id -> oof_fold assignment is read.
        identifier, fold = raw.get("source_id"), raw.get("oof_fold")
        need(identifier in expected_ids and identifier not in result and type(fold) is int and 0 <= fold < 5, "fold assignment differs")
        result[str(identifier)] = fold
    need(set(result) == expected_ids, "fold coverage differs")
    counts = {fold: sum(value == fold for value in result.values()) for fold in range(5)}
    need(counts == {fold: 400 for fold in range(5)}, "fold sizes differ")
    return result


def _rationale_multimap(path: Path, expected_ids: set[str], expected_records: int, *, single: bool = False) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for raw in jsonl(path):
        identifier, value = str(raw.get("source_id")), raw.get("rationales")
        need(identifier in expected_ids and isinstance(value, dict) and set(value) == set(AXES), "rationale linkage differs")
        normalized = {axis: str(value[axis]).strip() for axis in AXES}
        need(all(normalized.values()), "blank rationale")
        result.setdefault(identifier, []).append(normalized)
    need(set(result) == expected_ids and sum(map(len, result.values())) == expected_records, "rationale population differs")
    need(not single or all(len(values) == 1 for values in result.values()), "single rationale view differs")
    return result


def load_rationale_views(config: Config, train_ids: set[str], validation_ids: set[str] | None = None) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]], dict[str, Any]]:
    handoff_path = resolve(str(config["rationale_handoff_path"]))
    need(file_sha256(handoff_path) == config["rationale_handoff_sha256"], "rationale handoff checksum differs")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    need(handoff.get("schema_version") == "mal2026-rationale-pipeline-encoder-ratio-handoff-v2" and handoff.get("status") == "completed" and handoff.get("arm") == "1to2", "rationale handoff differs")
    need(handoff.get("teacher_use") == "train_only_label_aware_augmentation_never_validation_or_selection_dev", "teacher rationale boundary differs")
    need(handoff.get("selection_dev_view") == handoff.get("validation_view") == "student_score_blind_single_only", "student evaluation view differs")
    paths, digests, records = handoff["paths"], handoff["sha256"], handoff["records"]
    for key in ("teacher_train_all", "student_train_ratio", "student_train_single", "student_validation_single"):
        path = Path(paths[key]); need(path.is_file() and file_sha256(path) == digests[key], f"rationale file differs: {key}")
        need(path.resolve().is_relative_to((ROOT / "data/processed/restricted").resolve()), "rationale path escaped restricted storage")
    teacher = _rationale_multimap(Path(paths["teacher_train_all"]), train_ids, int(records["teacher_train_all"]))
    student = _rationale_multimap(Path(paths["student_train_ratio"]), train_ids, int(records["student_train_ratio"]))
    combined = {identifier: teacher[identifier] + student[identifier] for identifier in train_ids}
    student_train = _rationale_multimap(Path(paths["student_train_single"]), train_ids, int(records["student_train_single"]), single=True)
    if validation_ids is not None:
        student_validation = _rationale_multimap(Path(paths["student_validation_single"]), validation_ids, int(records["student_validation_single"]), single=True)
    else:
        student_validation = {}
    return combined, {**student_train, **student_validation}, handoff


def _split_prompt(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    first, second = "[시스템 프롬프트]", "[인코더 입력 템플릿]"
    need(text.count(first) == text.count(second) == 1 and not text.split(first, 1)[0].strip(), "raw prompt markers differ")
    system, template = text.split(first, 1)[1].split(second, 1)
    need(system.strip() and template.strip(), "raw prompt section blank")
    return system.strip(), template.strip()


def raw_score_text(config: Config, prompt: str, essay: str) -> str:
    path = resolve(str(config["raw_prompt_path"]))
    need(file_sha256(path) == config["raw_prompt_sha256"], "raw prompt checksum differs")
    system, template = _split_prompt(path)
    need(template.count("{prompt_text_json_string}") == template.count("{essay_text_json_string}") == 1, "raw prompt placeholders differ")
    rendered = template.replace("{prompt_text_json_string}", json.dumps(prompt, ensure_ascii=False), 1)
    rendered = rendered.replace("{essay_text_json_string}", json.dumps(essay, ensure_ascii=False), 1)
    result = system + "\n\n" + rendered
    need("reference_scores" not in result and "predicted_score" not in result and "evaluation_rationales" not in result, "raw encoder input leaked a forbidden field")
    return result


def rationale_score_text(config: Config, prompt: str, essay: str, rationales: Mapping[str, str]) -> str:
    from mal2026.rationale_pipeline_prompts import rationale_to_score_text
    need(file_sha256(resolve(str(config["rationale_prompt_path"]))) == config["rationale_prompt_sha256"], "rationale prompt checksum differs")
    return rationale_to_score_text(prompt, essay, rationales)


def _stable_int(*parts: Any) -> int:
    return int.from_bytes(sha256("\0".join(str(part) for part in parts).encode()).digest()[:8], "big")


def mild_tail_weights(labels: Sequence[Sequence[float]], cap: float) -> tuple[list[list[float]], dict[str, Any]]:
    need(labels and cap == 2.5, "tail-weight inputs differ")
    integer = [[round_half_up(float(value)) for value in row] for row in labels]
    weights = [[1.0] * 3 for _ in labels]
    audit: dict[str, Any] = {"mode": "inverse_sqrt_frequency_capped_and_axis_mean_normalized", "cap": cap, "axes": {}}
    for axis_index, axis in enumerate(AXES):
        counts = {score: sum(row[axis_index] == score for row in integer) for score in range(1, 6)}
        need(all(counts.values()), f"tail weighting lacks a class: {axis}")
        raw = {score: math.sqrt(len(integer) / (5.0 * counts[score])) for score in range(1, 6)}
        # Find a common scale whose capped per-example mean is exactly one.
        low, high = 0.0, cap / min(raw.values()) * 2.0
        for _ in range(100):
            middle = (low + high) / 2.0
            mean = statistics.fmean(min(cap, middle * raw[row[axis_index]]) for row in integer)
            if mean < 1.0: low = middle
            else: high = middle
        scale = (low + high) / 2.0
        normalized = {score: min(cap, scale * value) for score, value in raw.items()}
        need(max(normalized.values()) <= cap + 1e-6, "tail weights exceeded cap")
        for row_index, row in enumerate(integer):
            weights[row_index][axis_index] = normalized[row[axis_index]]
        audit["axes"][axis] = {"counts": {str(k): v for k, v in counts.items()}, "cell_weights": {str(k): v for k, v in normalized.items()}, "example_mean": statistics.fmean(weights[i][axis_index] for i in range(len(weights)))}
    return weights, audit


class SourceBalancedDataset:
    """One item per unique source; rationale view changes by deterministic epoch."""

    def __init__(
        self, rows: Sequence[Row], pools: Mapping[str, Sequence[Mapping[str, str]]], config: Config,
        input_variant: str, *, loss_weighting: str, training: bool,
    ) -> None:
        need(input_variant in INPUT_VARIANTS and loss_weighting in {"natural", "mild_tail"}, "dataset arm differs")
        need(len({row.identifier for row in rows}) == len(rows), "dataset contains repeated source")
        if input_variant != "raw":
            need(all(row.identifier in pools and pools[row.identifier] for row in rows), "dataset rationale coverage differs")
        self.rows = list(rows); self.pools = pools; self.config = config; self.input_variant = input_variant; self.training = training
        self.epoch = 0
        if loss_weighting == "mild_tail":
            self.weights, self.weight_audit = mild_tail_weights([row.labels for row in self.rows], float(config["tail_weight_cap"]))
        else:
            self.weights = [[1.0] * 3 for _ in self.rows]
            self.weight_audit = {"mode": "natural", "records": len(self.rows)}

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        need(type(epoch) is int and epoch >= 0, "dataset epoch differs")
        self.epoch = epoch

    def _rationale(self, row: Row) -> Mapping[str, str]:
        values = self.pools[row.identifier]
        index = _stable_int(self.config["seed"], self.epoch, row.identifier, "view") % len(values)
        return values[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        variant = self.input_variant
        if variant == "dropout" and self.training:
            probability = _stable_int(self.config["seed"], self.epoch, row.identifier, "dropout") / float(2**64)
            variant = "raw" if probability < float(self.config["rationale_dropout_probability"]) else "rationale"
        elif variant == "dropout":
            variant = "rationale"
        text = raw_score_text(self.config, row.prompt, row.essay) if variant == "raw" else rationale_score_text(self.config, row.prompt, row.essay, self._rationale(row))
        return {"text": text, "labels": list(row.labels), "loss_weights": self.weights[index]}


def collator(tokenizer: Any, max_length: int):
    def collect(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        texts = [str(feature["text"]) for feature in features]
        labels = [feature["labels"] for feature in features]
        weights = [feature["loss_weights"] for feature in features]
        batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        batch["loss_weights"] = torch.tensor(weights, dtype=torch.float32)
        return batch
    return collect


def arm_spec(phase: str, arm: str, stage1_winner: str | None = None, stage2_winner: Mapping[str, str] | None = None) -> tuple[str, str, str, bool]:
    if phase == "stage1":
        mapping = {
            "raw_mse": ("raw", "mse", "natural", False),
            "rationale_mse": ("rationale", "mse", "natural", False),
            "dropout_mse": ("dropout", "mse", "natural", False),
        }
        need(arm in mapping, "stage1 arm differs")
        return mapping[arm]
    need(phase in {"stage2", "stage4"} and stage1_winner in STAGE1_ARMS, "later-stage input selection differs")
    input_variant = {"raw_mse": "raw", "rationale_mse": "rationale", "dropout_mse": "dropout"}[str(stage1_winner)]
    if phase == "stage4":
        need(arm == "aihub_tail" and stage2_winner is not None, "stage4 arm differs")
        need(stage2_winner["input_variant"] == input_variant and stage2_winner["loss_kind"] in LOSS_KINDS and stage2_winner["loss_weighting"] in {"natural", "mild_tail"}, "stage4 inherited winner differs")
        return input_variant, stage2_winner["loss_kind"], stage2_winner["loss_weighting"], True
    mapping = {
        "huber": (input_variant, "huber", "natural", False),
        "ordinal": (input_variant, "ordinal", "natural", False),
        "multitask": (input_variant, "multitask", "natural", False),
        "multitask_tail": (input_variant, "multitask", "mild_tail", False),
    }
    need(arm in mapping, "stage2 arm differs")
    return mapping[arm]


def ordered_thresholds(first: Any, gap_raw: Any) -> Any:
    import torch
    import torch.nn.functional as F
    gaps = F.softplus(gap_raw) + 0.05
    return torch.cat((first.unsqueeze(-1), first.unsqueeze(-1) + torch.cumsum(gaps, dim=-1)), dim=-1)


def build_model(config: Config, loss_kind: str) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from peft import LoraConfig, TaskType, get_peft_model
    from safetensors import safe_open
    from transformers import AutoModel

    need(loss_kind in LOSS_KINDS, "model loss kind differs")
    backbone = AutoModel.from_pretrained(
        resolve(str(config["model_path"])), revision=config["model_revision"], local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    need(int(getattr(backbone.config, "hidden_size", -1)) == 4096, "Qwen hidden size differs")

    default_thresholds = torch.tensor([-1.945910149, -0.510825624, 0.510825624, 1.945910149], dtype=torch.float32)
    gaps = default_thresholds[1:] - default_thresholds[:-1] - 0.05
    gap_raw = torch.log(torch.expm1(gaps))

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.score_head = nn.Linear(4096, 3, dtype=next(backbone.parameters()).dtype)
            self.ordinal_first = nn.Parameter(default_thresholds[0].repeat(3))
            self.ordinal_gap_raw = nn.Parameter(gap_raw.repeat(3, 1))

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Mapping[str, Any] | None = None) -> None:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        def gradient_checkpointing_disable(self) -> None:
            self.backbone.gradient_checkpointing_disable()

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None = None, loss_weights: Any | None = None, **_: Any) -> Mapping[str, Any]:
            state = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
            positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
            final = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
            need(bool((final >= 0).all().item()), "encoder input has no token")
            pooled = state[torch.arange(state.shape[0], device=state.device), final]
            latent = self.score_head(F.normalize(pooled, p=2, dim=-1).to(self.score_head.weight.dtype)).float()
            regression = 1.0 + 4.0 * torch.sigmoid(latent)
            thresholds = ordered_thresholds(self.ordinal_first, self.ordinal_gap_raw)
            ordinal_logits = latent.unsqueeze(-1) - thresholds.unsqueeze(0)
            ordinal_expected = 1.0 + torch.sigmoid(ordinal_logits).sum(dim=-1)
            if loss_kind in {"mse", "huber"}:
                emitted = regression
            elif loss_kind == "ordinal":
                emitted = ordinal_expected
            else:
                emitted = (regression + ordinal_expected) / 2.0
            result: dict[str, Any] = {"logits": emitted}
            if labels is not None:
                need(loss_weights is not None and tuple(loss_weights.shape) == tuple(labels.shape), "loss weight shape differs")
                weights = loss_weights.float()
                if loss_kind == "mse":
                    per_axis = F.mse_loss(regression, labels.float(), reduction="none")
                elif loss_kind == "huber":
                    per_axis = F.smooth_l1_loss(regression, labels.float(), reduction="none", beta=float(config["huber_beta"]))
                else:
                    gold_integer = torch.floor(labels.float() + 0.5).clamp(1, 5)
                    ordinal_targets = (gold_integer.unsqueeze(-1) > torch.arange(1, 5, device=labels.device).view(1, 1, 4)).float()
                    ordinal_loss = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets, reduction="none").mean(dim=-1)
                    if loss_kind == "ordinal":
                        per_axis = ordinal_loss
                    else:
                        regression_loss = F.smooth_l1_loss(regression, labels.float(), reduction="none", beta=float(config["huber_beta"]))
                        per_axis = regression_loss + float(config["ordinal_auxiliary_weight"]) * ordinal_loss
                result["loss"] = (per_axis * weights).mean()
            return result

    model = Model()
    artifact = resolve(str(config["aihub_artifact_path"]))
    parameters = dict(model.named_parameters())
    targets = {**parameters, **dict(model.named_buffers())}
    # The AI-Hub artifact contract covers every learned backbone tensor and
    # the matched score head, not derived/non-persistent rotary buffers.
    required = {name for name in parameters if name.startswith("backbone.")} | {"score_head.weight", "score_head.bias"}
    loaded: set[str] = set()
    for tensor_path in sorted(artifact.glob("*.safetensors")):
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name not in required:
                    continue
                need(name in targets and name not in loaded, f"AI-Hub tensor differs: {name}")
                tensor = handle.get_tensor(name)
                need(tuple(tensor.shape) == tuple(targets[name].shape), f"AI-Hub tensor shape differs: {name}")
                targets[name].data.copy_(tensor.to(dtype=targets[name].dtype)); loaded.add(name)
    need(required <= loaded, "AI-Hub warmstart is incomplete")
    model.backbone = get_peft_model(model.backbone, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=int(config["lora_r"]), lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], bias="none",
    ))
    if hasattr(model.backbone, "enable_input_require_grads"):
        model.backbone.enable_input_require_grads()
    need(any("lora_" in name and parameter.requires_grad for name, parameter in model.named_parameters()), "LoRA is absent")
    return model, {
        "mode": "aihub_full_backbone_and_matched_bounded_head_then_fresh_lora",
        "loaded_tensors": len(loaded), "artifact_sha256": config["aihub_artifact_sha256"],
        "ordinal_threshold_initialization": default_thresholds.tolist(),
    }


def ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index]); result = [0.0] * len(values); start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]: result[index] = rank
        start = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = ranks(left), ranks(right); mx, my = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def emit_integers(continuous: Sequence[Sequence[float]], cutpoints: Mapping[str, Sequence[float]] | None = None) -> list[list[int]]:
    result: list[list[int]] = []
    for row in continuous:
        emitted: list[int] = []
        for axis_index, axis in enumerate(AXES):
            value = min(5.0, max(1.0, float(row[axis_index])))
            if cutpoints is None:
                emitted.append(round_half_up(value))
            else:
                thresholds = cutpoints[axis]
                need(len(thresholds) == 4 and all(thresholds[i] <= thresholds[i + 1] for i in range(3)), "cutpoints differ")
                emitted.append(1 + sum(value >= float(threshold) for threshold in thresholds))
        result.append(emitted)
    return result


def metric_bundle(labels: Sequence[Sequence[float]], continuous: Sequence[Sequence[float]], integers: Sequence[Sequence[int]] | None = None) -> dict[str, Any]:
    need(len(labels) == len(continuous) and bool(labels), "metric population differs")
    gold = [[round_half_up(float(value)) for value in row] for row in labels]
    predicted = emit_integers(continuous) if integers is None else [list(map(int, row)) for row in integers]
    result: dict[str, Any] = {"record_count": len(labels), "axes": {}}
    all_squared: list[float] = []
    tail_rmses: list[float] = []
    for axis_index, axis in enumerate(AXES):
        truth = [row[axis_index] for row in gold]; emitted = [row[axis_index] for row in predicted]
        errors = [(a - b) ** 2 for a, b in zip(truth, emitted, strict=True)]; all_squared.extend(errors)
        confusion = {str(score): {str(pred): 0 for pred in range(1, 6)} for score in range(1, 6)}
        for score, pred in zip(truth, emitted, strict=True): confusion[str(score)][str(pred)] += 1
        tail_errors = [error for score, error in zip(truth, errors, strict=True) if score in {1, 2, 5}]
        tail_rmse = math.sqrt(statistics.fmean(tail_errors)); tail_rmses.append(tail_rmse)
        continuous_errors = [(float(label[axis_index]) - float(pred[axis_index])) ** 2 for label, pred in zip(labels, continuous, strict=True)]
        result["axes"][axis] = {
            "integer_rmse": math.sqrt(statistics.fmean(errors)), "integer_spearman": spearman(truth, emitted),
            "integer_accuracy": statistics.fmean(a == b for a, b in zip(truth, emitted, strict=True)),
            "continuous_rmse_raw_decimal_gold": math.sqrt(statistics.fmean(continuous_errors)), "tail_1_2_5_rmse": tail_rmse,
            "gold_support": {str(score): sum(confusion[str(score)].values()) for score in range(1, 6)},
            "per_gold_recall": {str(score): (confusion[str(score)][str(score)] / sum(confusion[str(score)].values()) if sum(confusion[str(score)].values()) else None) for score in range(1, 6)},
            "confusion_gold_by_prediction": confusion,
        }
    result["macro_integer_rmse"] = statistics.fmean(result["axes"][axis]["integer_rmse"] for axis in AXES)
    result["overall_integer_rmse"] = math.sqrt(statistics.fmean(all_squared))
    result["macro_integer_spearman"] = statistics.fmean(result["axes"][axis]["integer_spearman"] for axis in AXES)
    result["macro_continuous_rmse_raw_decimal_gold"] = statistics.fmean(result["axes"][axis]["continuous_rmse_raw_decimal_gold"] for axis in AXES)
    result["macro_tail_rmse"] = statistics.fmean(tail_rmses)
    need(all(math.isfinite(float(result[key])) for key in ("macro_integer_rmse", "macro_integer_spearman", "macro_tail_rmse")), "metric is non-finite")
    return result


def selection_key(arm: str, metrics: Mapping[str, Any]) -> tuple[float, float, float, str]:
    return (float(metrics["macro_integer_rmse"]), float(metrics["macro_tail_rmse"]), -float(metrics["macro_integer_spearman"]), arm)


def trainable_state(model: Any) -> dict[str, Any]:
    state = {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters() if parameter.requires_grad}
    need(any("lora_" in name for name in state) and "score_head.weight" in state and "score_head.bias" in state, "trainable state differs")
    return state


def _attempt_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (parent / f"attempt-{attempt:04d}").exists():
        attempt += 1
    path = parent / f"attempt-{attempt:04d}"; path.mkdir()
    return path


def _winner_from_stage1(config: Config) -> str:
    path = config.output_root / "stage1" / "aggregate.json"
    need(path.is_file(), "stage1 aggregate unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    winner = value.get("selection", {}).get("arm")
    need(winner in STAGE1_ARMS, "stage1 winner differs")
    return str(winner)


def _winner_from_stage2(config: Config) -> dict[str, str]:
    path = config.output_root / "stage2" / "aggregate.json"
    need(path.is_file(), "stage2 aggregate unavailable")
    value = json.loads(path.read_text(encoding="utf-8")); selection = value.get("selection", {})
    need(selection.get("input_variant") in INPUT_VARIANTS and selection.get("loss_kind") in LOSS_KINDS, "stage2 winner differs")
    return {"arm": str(selection["arm"]), "input_variant": str(selection["input_variant"]), "loss_kind": str(selection["loss_kind"]), "loss_weighting": str(selection["loss_weighting"])}


def _load_aihub_prepared(config: Config) -> tuple[list[Row], dict[str, list[dict[str, str]]], dict[str, Any]]:
    report_path = config.output_root / "stage4" / "aihub_audit.json"
    need(report_path.is_file(), "AI-Hub audit unavailable")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    need(report.get("status") == "admitted", "AI-Hub tail data was not admitted")
    path = Path(report["restricted_selected_path"])
    need(path.is_file() and file_sha256(path) == report["restricted_selected_sha256"], "AI-Hub selected data differs")
    rows: list[Row] = []; pools: dict[str, list[dict[str, str]]] = {}
    for raw in jsonl(path):
        need(set(raw) == {"source_id", "prompt", "essay", "labels", "rationales"}, "AI-Hub selected row differs")
        row = Row(str(raw["source_id"]), str(raw["prompt"]), str(raw["essay"]), tuple(map(float, raw["labels"])), "aihub")
        rationales = raw["rationales"]; need(isinstance(rationales, dict) and set(rationales) == set(AXES), "AI-Hub rationale axes differ")
        rows.append(row); pools[row.identifier] = [{axis: str(rationales[axis]) for axis in AXES}]
    need(len(rows) == int(report["selected_records"]) == int(config["aihub_tail_max_records"]), "AI-Hub selected population differs")
    return rows, pools, report


def run_fold(config: Config, phase: str, arm: str, fold: int, *, smoke: bool = False) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    from setproctitle import setproctitle
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

    need(phase in {"stage1", "stage2", "stage4"} and type(fold) is int and 0 <= fold < 5, "fold invocation differs")
    stage1_winner = _winner_from_stage1(config) if phase != "stage1" else None
    stage2_winner = _winner_from_stage2(config) if phase == "stage4" else None
    input_variant, loss_kind, loss_weighting, include_aihub = arm_spec(phase, arm, stage1_winner, stage2_winner)
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    setproctitle(f"mal2026:qwen-oof:{'smoke' if smoke else phase}:{arm}:f{fold}:gpu{physical_gpu}"[:255])
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, "fold runner requires exactly one visible GPU")
    task_root = config.output_root / ("smoke" if smoke else phase) / arm / f"fold-{fold:02d}"
    result_path = task_root / ("smoke_complete.json" if smoke else "result.json")
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        need(existing.get("status") == "completed" and existing.get("config_sha256") == config.config_sha256, "existing fold result lineage differs")
        return existing
    attempt = _attempt_dir(task_root)
    restricted_task = config.restricted_root / ("smoke" if smoke else phase) / arm / f"fold-{fold:02d}" / attempt.name
    restricted_task.mkdir(parents=True, exist_ok=False)

    all_rows = load_mal_rows(config)
    fold_map = load_fold_map(config, {row.identifier for row in all_rows})
    combined_views, student_views, handoff = load_rationale_views(config, {row.identifier for row in all_rows})
    train_rows = [row for row in all_rows if fold_map[row.identifier] != fold]
    held_rows = [row for row in all_rows if fold_map[row.identifier] == fold]
    train_pools: dict[str, Sequence[Mapping[str, str]]] = {row.identifier: combined_views[row.identifier] for row in train_rows}
    held_pools: dict[str, Sequence[Mapping[str, str]]] = {row.identifier: student_views[row.identifier] for row in held_rows}
    aihub_report = None
    if include_aihub:
        aihub_rows, aihub_pools, aihub_report = _load_aihub_prepared(config)
        train_rows += aihub_rows; train_pools.update(aihub_pools)
    if smoke:
        # The smoke covers multiple integer bands without changing the full protocol.
        ranked = sorted(train_rows, key=lambda row: (_stable_int(config["seed"], row.identifier, "smoke"), row.identifier))
        chosen: list[Row] = []; cells: set[tuple[int, int]] = set()
        for row in ranked:
            row_cells = {(axis, round_half_up(row.labels[axis])) for axis in range(3)}
            if row_cells - cells or len(chosen) < 16:
                chosen.append(row); cells |= row_cells
            if len(chosen) >= 32 and len(cells) == 15: break
        train_rows = chosen[:64]; train_pools = {row.identifier: train_pools[row.identifier] for row in train_rows}
        ranked_held = sorted(held_rows, key=lambda row: (_stable_int(config["seed"], row.identifier, "smoke-held"), row.identifier))
        chosen_held: list[Row] = []; held_cells: set[tuple[int, int]] = set()
        for row in ranked_held:
            row_cells = {(axis, round_half_up(row.labels[axis])) for axis in range(3)}
            if row_cells - held_cells or len(chosen_held) < 16:
                chosen_held.append(row); held_cells |= row_cells
            if len(chosen_held) >= 32 and all(any(cell[0] == axis and cell[1] in {1, 2, 5} for cell in held_cells) for axis in range(3)): break
        held_rows = chosen_held[:64]; held_pools = {row.identifier: held_pools[row.identifier] for row in held_rows}
    need(len({row.identifier for row in train_rows}) == len(train_rows) and set(row.identifier for row in train_rows).isdisjoint(row.identifier for row in held_rows), "source-disjoint fold differs")

    tokenizer = AutoTokenizer.from_pretrained(resolve(str(config["model_path"])), revision=config["model_revision"], local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_data = SourceBalancedDataset(train_rows, train_pools, config, input_variant, loss_weighting=loss_weighting, training=True)
    held_data = SourceBalancedDataset(held_rows, held_pools, config, input_variant, loss_weighting="natural", training=False)
    model, initialization = build_model(config, loss_kind)

    class EpochCallback(TrainerCallback):
        def on_epoch_begin(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            train_data.set_epoch(int(math.floor(float(state.epoch or 0.0))))
            return control

    set_seed(int(config["seed"]) + fold)
    args = TrainingArguments(
        output_dir=str(attempt / "trainer"), do_train=True, do_eval=False, eval_strategy="no", save_strategy="no",
        num_train_epochs=float(config["epochs"]), max_steps=2 if smoke else -1,
        learning_rate=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]), warmup_ratio=float(config["warmup_ratio"]),
        optim="adamw_torch_fused", per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]), gradient_accumulation_steps=1 if smoke else int(config["gradient_accumulation_steps"]),
        bf16=True, tf32=True, gradient_checkpointing=bool(config["gradient_checkpointing"]), gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[], remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=True,
        logging_steps=1 if smoke else 10, seed=int(config["seed"]) + fold, data_seed=int(config["seed"]) + fold,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_data, data_collator=collator(tokenizer, int(config["max_length"])), callbacks=[EpochCallback()])
    torch.cuda.reset_peak_memory_stats(); trained = trainer.train(); peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    prediction = trainer.predict(held_data)
    continuous = prediction.predictions.tolist(); labels = prediction.label_ids.tolist(); integers = emit_integers(continuous)
    metrics = metric_bundle(labels, continuous, integers)
    common = {
        "schema_version": "mal2026-qwen-rationale-oof-fold-v1", "status": "completed", "completed_at": now(),
        "run_id": config["run_id"], "phase": phase, "arm": arm, "outer_fold": fold, "smoke": smoke,
        "input_variant": input_variant, "loss_kind": loss_kind, "loss_weighting": loss_weighting,
        "source_balance": "each_unique_source_exactly_once_per_epoch_with_deterministic_epoch_varying_rationale_view",
        "train_records": len(train_rows), "held_records": len(held_rows), "mal_train_records": sum(row.source == "mal" for row in train_rows),
        "aihub_train_records": sum(row.source == "aihub" for row in train_rows), "teacher_rationales_train_only": True,
        "held_view": "student_score_blind_single_only", "average_target_used": False, "validation_access": False,
        "fixed_epochs": int(config["epochs"]), "global_step": int(trainer.state.global_step), "effective_batch_size": int(config["per_device_train_batch_size"]) * (1 if smoke else int(config["gradient_accumulation_steps"])),
        "peak_memory_mib": peak, "metrics": metrics, "weight_audit": train_data.weight_audit,
        "initialization": initialization, "rationale_handoff_sha256": config["rationale_handoff_sha256"],
        "aihub_audit_sha256": file_sha256(config.output_root / "stage4" / "aihub_audit.json") if aihub_report else None,
        "config_path": str(config.path), "config_sha256": config.config_sha256, "physical_gpu": physical_gpu,
        "train_metrics": {key: float(value) for key, value in trained.metrics.items() if isinstance(value, (int, float))},
        "privacy": "aggregate_only;row_predictions_and_trainable_state_are_restricted_or_ignored",
    }
    if smoke:
        atomic_json(result_path, common)
        return common
    prediction_path = restricted_task / "predictions.jsonl"
    count, digest = write_jsonl(prediction_path, (
        {"source_id": row.identifier, "outer_fold": fold, "labels_raw": list(label), "predictions_continuous": list(pred), "predictions_integer": list(integer)}
        for row, label, pred, integer in zip(held_rows, labels, continuous, integers, strict=True)
    ))
    need(count == len(held_rows), "prediction count differs")
    state_path = attempt / "trainable.safetensors"; save_file(trainable_state(model), str(state_path))
    result = {**common, "prediction_path": str(prediction_path.resolve()), "prediction_sha256": digest, "state_path": str(state_path.resolve()), "state_sha256": file_sha256(state_path)}
    atomic_json(result_path, result)
    return result


def aggregate_phase(config: Config, phase: str) -> dict[str, Any]:
    need(phase in {"stage1", "stage2", "stage4"}, "aggregate phase differs")
    if phase == "stage4":
        audit_path = config.output_root / "stage4" / "aihub_audit.json"
        need(audit_path.is_file(), "AI-Hub audit unavailable")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "admitted":
            value = {"schema_version": "mal2026-qwen-rationale-oof-aggregate-v1", "status": "not_admitted", "run_id": config["run_id"], "phase": phase, "reason": audit.get("reason"), "validation_access": False, "average_target_used": False}
            atomic_json(config.output_root / phase / "aggregate.json", value); return value
        arms = ("aihub_tail",)
    else:
        arms = STAGE1_ARMS if phase == "stage1" else STAGE2_ARMS
    summaries: dict[str, Any] = {}
    for arm in arms:
        fold_results: list[dict[str, Any]] = []; restricted_rows: list[dict[str, Any]] = []
        for fold in range(5):
            path = config.output_root / phase / arm / f"fold-{fold:02d}" / "result.json"
            need(path.is_file(), f"missing fold result: {phase}/{arm}/{fold}")
            result = json.loads(path.read_text(encoding="utf-8")); need(result.get("status") == "completed" and result.get("config_sha256") == config.config_sha256, "fold result lineage differs")
            prediction_path = Path(result["prediction_path"]); need(file_sha256(prediction_path) == result["prediction_sha256"], "fold prediction checksum differs")
            fold_results.append(result); restricted_rows.extend(jsonl(prediction_path))
        need(len(restricted_rows) == 2000 and len({row["source_id"] for row in restricted_rows}) == 2000, "OOF coverage differs")
        metrics = metric_bundle([row["labels_raw"] for row in restricted_rows], [row["predictions_continuous"] for row in restricted_rows], [row["predictions_integer"] for row in restricted_rows])
        merged_path = config.restricted_root / phase / arm / "oof_predictions.jsonl"
        count, digest = write_jsonl(merged_path, sorted(restricted_rows, key=lambda row: (int(row["outer_fold"]), str(row["source_id"]))))
        summaries[arm] = {
            "arm": arm, "input_variant": fold_results[0]["input_variant"], "loss_kind": fold_results[0]["loss_kind"], "loss_weighting": fold_results[0]["loss_weighting"],
            "metrics": metrics, "fold_metrics": {str(result["outer_fold"]): result["metrics"] for result in fold_results},
            "oof_prediction_path": str(merged_path.resolve()), "oof_prediction_sha256": digest, "records": count,
            "state_sha256_by_fold": {str(result["outer_fold"]): result["state_sha256"] for result in fold_results},
        }
    if phase == "stage2":
        stage1 = json.loads((config.output_root / "stage1" / "aggregate.json").read_text(encoding="utf-8"))
        prior_arm = str(stage1["selection"]["arm"]); prior = stage1["arms"][prior_arm]
        candidates = {f"stage1::{prior_arm}": prior, **{f"stage2::{arm}": summary for arm, summary in summaries.items()}}
    else:
        candidates = summaries
    winner_key, winner = min(candidates.items(), key=lambda item: selection_key(item[0], item[1]["metrics"]))
    selection = {
        "candidate": winner_key, "arm": winner["arm"], "input_variant": winner["input_variant"], "loss_kind": winner["loss_kind"],
        "loss_weighting": winner["loss_weighting"], "metrics": winner["metrics"],
        "rule": "lowest macro integer RMSE, then lowest macro {1,2,5} tail RMSE, then highest macro integer Spearman, then lexical arm",
    }
    result = {
        "schema_version": "mal2026-qwen-rationale-oof-aggregate-v1", "status": "completed", "completed_at": now(),
        "run_id": config["run_id"], "phase": phase, "arms": summaries, "selection": selection,
        "fold_count": 5, "records": 2000, "selection_source": "train_only_exact_5fold_oof",
        "validation_access": False, "average_target_used": False, "config_sha256": config.config_sha256,
        "privacy": "aggregate_only;OOF_rows_are_restricted",
    }
    atomic_json(config.output_root / phase / "aggregate.json", result)
    return result


def fit_cutpoints(values: Sequence[float], gold: Sequence[int]) -> list[float]:
    """Deterministic coordinate minimization of ordinal integer squared error."""
    import numpy as np
    x = np.asarray(values, dtype=np.float64); y = np.asarray(gold, dtype=np.int64)
    need(x.ndim == y.ndim == 1 and len(x) == len(y) and len(x) >= 20 and np.isfinite(x).all(), "cutpoint fit population differs")
    unique = np.unique(x)
    candidates = np.concatenate(([unique[0] - 1e-6], (unique[:-1] + unique[1:]) / 2.0, [unique[-1] + 1e-6]))
    thresholds = np.asarray([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    for _ in range(6):
        changed = False
        for index in range(4):
            lower = thresholds[index - 1] if index else -np.inf; upper = thresholds[index + 1] if index < 3 else np.inf
            allowed = candidates[(candidates >= lower) & (candidates <= upper)]
            other = 1 + sum((x >= thresholds[j]).astype(np.int64) for j in range(4) if j != index)
            best = thresholds[index]; best_key = (float("inf"), float("inf"), float("inf"))
            for candidate in allowed:
                loss = float(np.mean((y - (other + (x >= candidate).astype(np.int64))) ** 2))
                key = (loss, abs(float(candidate) - float(thresholds[index])), float(candidate))
                if key < best_key: best_key, best = key, float(candidate)
            changed |= best != thresholds[index]; thresholds[index] = best
        if not changed: break
    need(bool(np.all(thresholds[:-1] <= thresholds[1:])), "fitted cutpoints are not monotonic")
    return thresholds.tolist()


def _prediction_source(config: Config, source: str) -> tuple[str, dict[str, Any]]:
    need(source in {"stage2", "stage4"}, "calibration source differs")
    aggregate = json.loads((config.output_root / source / "aggregate.json").read_text(encoding="utf-8"))
    need(aggregate.get("status") == "completed", "calibration aggregate unavailable")
    selected = aggregate["selection"]
    if source == "stage2" and str(selected["candidate"]).startswith("stage1::"):
        phase = "stage1"
        phase_aggregate = json.loads((config.output_root / phase / "aggregate.json").read_text(encoding="utf-8"))
        summary = phase_aggregate["arms"][selected["arm"]]
    else:
        phase = source; summary = aggregate["arms"][selected["arm"]]
    return phase, summary


def calibrate(config: Config, source: str) -> dict[str, Any]:
    if source == "stage4":
        aggregate_path = config.output_root / "stage4" / "aggregate.json"
        need(aggregate_path.is_file(), "stage4 aggregate unavailable")
        if json.loads(aggregate_path.read_text(encoding="utf-8")).get("status") != "completed":
            result = {"schema_version": "mal2026-qwen-rationale-oof-calibration-v1", "status": "not_admitted", "source": source, "validation_access": False}
            atomic_json(config.output_root / source / "calibration.json", result); return result
    phase, summary = _prediction_source(config, source)
    path = Path(summary["oof_prediction_path"]); need(file_sha256(path) == summary["oof_prediction_sha256"], "calibration OOF checksum differs")
    rows = list(jsonl(path)); need(len(rows) == 2000, "calibration OOF population differs")
    crossfit_integer = [[0, 0, 0] for _ in rows]; fold_thresholds: dict[str, Any] = {}
    for fold in range(5):
        train_indices = [i for i, row in enumerate(rows) if int(row["outer_fold"]) != fold]
        held_indices = [i for i, row in enumerate(rows) if int(row["outer_fold"]) == fold]
        need((len(train_indices), len(held_indices)) == (1600, 400), "calibration crossfit split differs")
        fold_thresholds[str(fold)] = {}
        for axis_index, axis in enumerate(AXES):
            thresholds = fit_cutpoints([float(rows[i]["predictions_continuous"][axis_index]) for i in train_indices], [round_half_up(float(rows[i]["labels_raw"][axis_index])) for i in train_indices])
            fold_thresholds[str(fold)][axis] = thresholds
            for i in held_indices:
                value = float(rows[i]["predictions_continuous"][axis_index]); crossfit_integer[i][axis_index] = 1 + sum(value >= threshold for threshold in thresholds)
    labels = [row["labels_raw"] for row in rows]; continuous = [row["predictions_continuous"] for row in rows]
    calibrated_metrics = metric_bundle(labels, continuous, crossfit_integer); uncalibrated_metrics = metric_bundle(labels, continuous, [row["predictions_integer"] for row in rows])
    promoted = selection_key("calibrated", calibrated_metrics) < selection_key("uncalibrated", uncalibrated_metrics)
    full_thresholds = {axis: fit_cutpoints([float(row["predictions_continuous"][axis_index]) for row in rows], [round_half_up(float(row["labels_raw"][axis_index])) for row in rows]) for axis_index, axis in enumerate(AXES)}
    restricted_path = config.restricted_root / source / "calibrated_oof_predictions.jsonl"
    count, digest = write_jsonl(restricted_path, ({**row, "calibrated_integer": integer} for row, integer in zip(rows, crossfit_integer, strict=True)))
    result = {
        "schema_version": "mal2026-qwen-rationale-oof-calibration-v1", "status": "completed", "completed_at": now(),
        "run_id": config["run_id"], "source": source, "source_phase": phase, "arm": summary["arm"],
        "input_variant": summary["input_variant"], "loss_kind": summary["loss_kind"], "loss_weighting": summary["loss_weighting"],
        "method": "axiswise_monotonic_cutpoints_fit_on_other_four_OOF_folds_and_applied_to_held_fold",
        "uncalibrated_metrics": uncalibrated_metrics, "crossfit_calibrated_metrics": calibrated_metrics,
        "promoted": promoted, "chosen_variant": "calibrated" if promoted else "uncalibrated",
        "chosen_metrics": calibrated_metrics if promoted else uncalibrated_metrics,
        "fold_thresholds": fold_thresholds, "full_oof_fit_thresholds_for_final_model": full_thresholds,
        "restricted_prediction_path": str(restricted_path.resolve()), "restricted_prediction_sha256": digest, "records": count,
        "selection_source": "train_only_crossfit_OOF", "validation_access": False, "average_target_used": False,
    }
    atomic_json(config.output_root / source / "calibration.json", result)
    return result


def audit_aihub_tail(config: Config) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from mal2026.official_aihub_rationale_data import load_argumentative, projected_rationales

    output = config.output_root / "stage4" / "aihub_audit.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    manifest = json.loads(resolve(str(config["aihub_manifest_path"])).read_text(encoding="utf-8"))
    need(manifest.get("score_contract", {}).get("average_excluded") is True and manifest.get("score_contract", {}).get("axes") == list(AXES), "AI-Hub score contract differs")
    mal_rows = load_mal_rows(config); mal_hashes = {sha256((row.prompt + "\0" + row.essay).encode()).hexdigest() for row in mal_rows}
    candidates: dict[str, tuple[Any, dict[str, str], tuple[int, int, int]]] = {}
    duplicate_count = 0; overlap_count = 0
    for raw in load_argumentative("refit_train"):
        integer = tuple(round_half_up(float(raw.score[axis])) for axis in AXES)
        if not any(score in {1, 2, 5} for score in integer): continue
        text_hash = sha256((raw.prompt + "\0" + raw.essay).encode()).hexdigest()
        if text_hash in mal_hashes: overlap_count += 1; continue
        if text_hash in candidates: duplicate_count += 1; continue
        candidates[text_hash] = (raw, projected_rationales(raw), integer)
    cells = [(axis, score) for axis in range(3) for score in (1, 2, 5)]
    availability = {(axis, score): sum(value[2][axis] == score for value in candidates.values()) for axis, score in cells}
    quotas = {cell: min(250, availability[cell]) for cell in cells}
    ordered_by_cell = {cell: sorted((key for key, value in candidates.items() if value[2][cell[0]] == cell[1]), key=lambda key: sha256(f"{config['seed']}\0{cell}\0{key}".encode()).hexdigest()) for cell in cells}
    pointers = {cell: 0 for cell in cells}; selected: set[str] = set(); achieved = {cell: 0 for cell in cells}
    while len(selected) < int(config["aihub_tail_max_records"]):
        progressed = False
        for cell in sorted(cells, key=lambda value: (availability[value], value)):
            if achieved[cell] >= quotas[cell]: continue
            values = ordered_by_cell[cell]
            while pointers[cell] < len(values) and values[pointers[cell]] in selected: pointers[cell] += 1
            if pointers[cell] >= len(values): continue
            key = values[pointers[cell]]; pointers[cell] += 1; selected.add(key); progressed = True
            integer = candidates[key][2]
            for other in cells:
                if integer[other[0]] == other[1]: achieved[other] += 1
            if len(selected) >= int(config["aihub_tail_max_records"]): break
        if not progressed or all(achieved[cell] >= quotas[cell] for cell in cells): break
    for key in sorted(candidates, key=lambda value: sha256(f"{config['seed']}\0fill\0{value}".encode()).hexdigest()):
        if len(selected) >= int(config["aihub_tail_max_records"]): break
        selected.add(key)
    reason = None
    if len(selected) != int(config["aihub_tail_max_records"]): reason = "insufficient_unique_tail_records"
    if overlap_count: reason = "MAL_prompt_essay_overlap_detected"
    selected_rows: list[dict[str, Any]] = []
    for key in sorted(selected):
        raw, rationales, _ = candidates[key]
        selected_rows.append({"source_id": f"aihub-tail:{raw.identifier}", "prompt": raw.prompt, "essay": raw.essay, "labels": [float(raw.score[axis]) for axis in AXES], "rationales": rationales})
    selected_path = config.restricted_root / "stage4" / "aihub_tail_selected.jsonl"
    count, digest = write_jsonl(selected_path, selected_rows)
    tokenizer = AutoTokenizer.from_pretrained(resolve(str(config["model_path"])), revision=config["model_revision"], local_files_only=True, trust_remote_code=False, use_fast=True)
    winner = _winner_from_stage2(config); lengths: list[int] = []
    for start in range(0, len(selected_rows), 64):
        texts = []
        for row in selected_rows[start:start + 64]:
            texts.append(raw_score_text(config, row["prompt"], row["essay"]) if winner["input_variant"] == "raw" else rationale_score_text(config, row["prompt"], row["essay"], row["rationales"]))
        lengths.extend(len(ids) for ids in tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"])
    maximum = max(lengths) if lengths else 0
    if maximum > int(config["max_length"]): reason = "selected_AIHub_input_exceeds_frozen_max_length"
    histograms = {axis: {str(score): sum(round_half_up(row["labels"][axis_index]) == score for row in selected_rows) for score in range(1, 6)} for axis_index, axis in enumerate(AXES)}
    result = {
        "schema_version": "mal2026-qwen-rationale-oof-aihub-audit-v1", "status": "admitted" if reason is None else "not_admitted",
        "completed_at": now(), "run_id": config["run_id"], "reason": reason, "source": "canonical_AIHub_argumentative_Training_only",
        "candidate_unique_tail_records": len(candidates), "selected_records": count, "requested_records": int(config["aihub_tail_max_records"]),
        "deduplicated_records": duplicate_count, "MAL_exact_prompt_essay_overlaps": overlap_count,
        "cell_availability": {f"{AXES[axis]}:{score}": value for (axis, score), value in availability.items()},
        "cell_quotas": {f"{AXES[axis]}:{score}": value for (axis, score), value in quotas.items()},
        "cell_achieved": {f"{AXES[axis]}:{score}": value for (axis, score), value in achieved.items()},
        "selected_integer_histograms": histograms, "token_length": {"maximum": maximum, "limit": int(config["max_length"]), "truncated": sum(length > int(config["max_length"]) for length in lengths)},
        "restricted_selected_path": str(selected_path.resolve()), "restricted_selected_sha256": digest,
        "manifest_path": str(resolve(str(config["aihub_manifest_path"])).resolve()), "manifest_sha256": file_sha256(resolve(str(config["aihub_manifest_path"]))),
        "average_target_used": False, "validation_access": False, "privacy": "aggregate_only;selected_rows_are_restricted",
    }
    atomic_json(output, result); return result


def final_candidate(config: Config) -> dict[str, Any]:
    stage2 = json.loads((config.output_root / "stage2" / "calibration.json").read_text(encoding="utf-8"))
    need(stage2.get("status") == "completed", "stage2 calibration unavailable")
    candidates = [("stage2", stage2)]
    stage4_path = config.output_root / "stage4" / "calibration.json"
    if stage4_path.is_file():
        stage4 = json.loads(stage4_path.read_text(encoding="utf-8"))
        if stage4.get("status") == "completed": candidates.append(("stage4", stage4))
    source, winner = min(candidates, key=lambda item: selection_key(item[0], item[1]["chosen_metrics"]))
    return {
        "source": source, "arm": winner["arm"], "input_variant": winner["input_variant"], "loss_kind": winner["loss_kind"],
        "loss_weighting": winner["loss_weighting"], "calibration_variant": winner["chosen_variant"],
        "oof_metrics": winner["chosen_metrics"], "cutpoints": winner["full_oof_fit_thresholds_for_final_model"] if winner["chosen_variant"] == "calibrated" else None,
        "rule": "lowest train-only OOF macro integer RMSE, then tail RMSE, Spearman, lexical source",
    }


def refit_and_validate(config: Config) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    from setproctitle import setproctitle
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

    output = config.output_root / "final" / "result.json"
    if output.is_file(): return json.loads(output.read_text(encoding="utf-8"))
    selected = final_candidate(config)
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    setproctitle(f"mal2026:qwen-oof:final:{selected['source']}:{selected['arm']}:gpu{physical_gpu}"[:255])
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, "final refit requires exactly one visible GPU")
    attempt = _attempt_dir(config.output_root / "final"); restricted = config.restricted_root / "final" / attempt.name; restricted.mkdir(parents=True, exist_ok=False)
    train_rows = load_mal_rows(config); train_ids = {row.identifier for row in train_rows}
    combined_views, _, _ = load_rationale_views(config, train_ids)
    train_pools: dict[str, Sequence[Mapping[str, str]]] = combined_views
    if selected["source"] == "stage4":
        aihub_rows, aihub_pools, _ = _load_aihub_prepared(config); train_rows += aihub_rows; train_pools = {**train_pools, **aihub_pools}
    tokenizer = AutoTokenizer.from_pretrained(resolve(str(config["model_path"])), revision=config["model_revision"], local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_data = SourceBalancedDataset(train_rows, train_pools, config, selected["input_variant"], loss_weighting=selected["loss_weighting"], training=True)
    model, initialization = build_model(config, selected["loss_kind"])

    class EpochCallback(TrainerCallback):
        def on_epoch_begin(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            train_data.set_epoch(int(math.floor(float(state.epoch or 0.0)))); return control

    set_seed(int(config["seed"]))
    args = TrainingArguments(
        output_dir=str(attempt / "trainer"), do_train=True, do_eval=False, eval_strategy="no", save_strategy="no", num_train_epochs=float(config["epochs"]),
        learning_rate=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]), warmup_ratio=float(config["warmup_ratio"]), optim="adamw_torch_fused",
        per_device_train_batch_size=int(config["per_device_train_batch_size"]), per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]), bf16=True, tf32=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=True, logging_steps=10, seed=int(config["seed"]), data_seed=int(config["seed"]),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_data, data_collator=collator(tokenizer, int(config["max_length"])), callbacks=[EpochCallback()])
    torch.cuda.reset_peak_memory_stats(); trained = trainer.train(); peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    state_path = attempt / "trainable.safetensors"; save_file(trainable_state(model), str(state_path))

    # This is the single point where canonical validation content and labels are parsed.
    validation_rows = load_mal_rows(config, validation=True); validation_ids = {row.identifier for row in validation_rows}
    _, validation_views, handoff = load_rationale_views(config, train_ids, validation_ids)
    validation_data = SourceBalancedDataset(validation_rows, validation_views, config, selected["input_variant"], loss_weighting="natural", training=False)
    prediction = trainer.predict(validation_data); continuous = prediction.predictions.tolist(); labels = prediction.label_ids.tolist()
    integers = emit_integers(continuous, selected["cutpoints"])
    metrics = metric_bundle(labels, continuous, integers)
    prediction_path = restricted / "validation_predictions.jsonl"
    count, digest = write_jsonl(prediction_path, ({"source_id": row.identifier, "labels_raw": label, "predictions_continuous": pred, "predictions_integer": integer} for row, label, pred, integer in zip(validation_rows, labels, continuous, integers, strict=True)))
    result = {
        "schema_version": "mal2026-qwen-rationale-oof-final-v1", "status": "completed", "completed_at": now(), "run_id": config["run_id"],
        "selection": selected, "training": {"fixed_epochs": int(config["epochs"]), "records": len(train_rows), "mal_records": 2000, "aihub_records": len(train_rows) - 2000, "global_step": int(trainer.state.global_step), "effective_batch_size": int(config["per_device_train_batch_size"]) * int(config["gradient_accumulation_steps"]), "peak_memory_mib": peak, "weight_audit": train_data.weight_audit, "train_metrics": {key: float(value) for key, value in trained.metrics.items() if isinstance(value, (int, float))}},
        "canonical_validation": {"records": count, "view": "raw_only" if selected["input_variant"] == "raw" else "student_score_blind_single_only", "use": "single_final_descriptive_evaluation_not_selection", "metrics": metrics},
        "state_path": str(state_path.resolve()), "state_sha256": file_sha256(state_path), "validation_prediction_path": str(prediction_path.resolve()), "validation_prediction_sha256": digest,
        "initialization": initialization, "rationale_handoff_sha256": config["rationale_handoff_sha256"], "config_sha256": config.config_sha256,
        "average_target_used": False, "validation_access_count_in_this_program": 1, "physical_gpu": physical_gpu,
        "privacy": "aggregate_only;validation_predictions_are_restricted",
    }
    atomic_json(output, result); return result


def preflight(config: Config) -> dict[str, Any]:
    from setproctitle import setproctitle
    setproctitle("mal2026:qwen-oof:preflight")
    rows = load_mal_rows(config); ids = {row.identifier for row in rows}; fold_map = load_fold_map(config, ids)
    combined, student, handoff = load_rationale_views(config, ids)
    need(all(len(combined[identifier]) >= 2 and len(student[identifier]) == 1 for identifier in ids), "rationale multiplicity differs")
    # Render every raw source and the longest-declared rationale paths were already
    # proven below 2,100 tokens by the immutable handoff's existing token audit.
    samples = [raw_score_text(config, row.prompt, row.essay) for row in rows[:10]]
    need(all("evaluation_rationales" not in text for text in samples), "raw input ablation differs")
    result = {
        "schema_version": "mal2026-qwen-rationale-oof-preflight-v1", "status": "completed", "completed_at": now(), "run_id": config["run_id"],
        "train_records": len(rows), "fold_counts": {str(fold): sum(value == fold for value in fold_map.values()) for fold in range(5)},
        "teacher_records": int(handoff["records"]["teacher_train_all"]), "student_ratio_records": int(handoff["records"]["student_train_ratio"]),
        "source_balanced_records_per_epoch": 1600, "config_path": str(config.path), "config_sha256": config.config_sha256,
        "git_sha": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(), "gpu_scope": config["gpu_scope"], "user_authorization": config["user_authorization"],
        "validation_content_parsed": False, "validation_hash_verified": True, "average_target_used": False,
    }
    atomic_json(config.output_root / "preflight.json", result)
    task_card = {
        "run_id": config["run_id"], "stage_and_deliverable": "Stage1 input OOF -> Stage2 loss OOF -> OOF calibration -> compatible AI-Hub tail OOF -> one final validation",
        "completion_predicate": "final/result.json and final_report.json completed with finite metrics",
        "permitted_inputs": [config["train_path"], config["fold_rows_path"], config["rationale_handoff_path"], config["aihub_manifest_path"], "validation only after all OOF selection"],
        "resource_scope": "GPUs 0-3; GPU0 smoke first", "gpu_authorization": config["user_authorization"],
        "scientific_variables_preapproved": {key: config[key] for key in ("stage1_arms", "stage2_arms", "epochs", "learning_rate", "max_length", "huber_beta", "ordinal_auxiliary_weight", "rationale_dropout_probability", "tail_weight_cap", "aihub_tail_max_records", "primary_metric", "selection_tiebreakers")},
        "integration_recovery": "at most three evidence-distinct repairs; no data/prompt/objective/selection changes", "ledger_path": str((config.output_root / "ledger.jsonl").resolve()),
    }
    atomic_json(config.output_root / "task_card.json", task_card)
    return result


def final_report(config: Config) -> dict[str, Any]:
    final = json.loads((config.output_root / "final" / "result.json").read_text(encoding="utf-8")); need(final.get("status") == "completed", "final result unavailable")
    baseline = json.loads(resolve(str(config["baseline_result_path"])).read_text(encoding="utf-8"))
    baseline_metrics = baseline["canonical_validation"]["metrics"]; metrics = final["canonical_validation"]["metrics"]
    stage1 = json.loads((config.output_root / "stage1" / "aggregate.json").read_text(encoding="utf-8"))
    stage2 = json.loads((config.output_root / "stage2" / "aggregate.json").read_text(encoding="utf-8"))
    stage2_cal = json.loads((config.output_root / "stage2" / "calibration.json").read_text(encoding="utf-8"))
    audit = json.loads((config.output_root / "stage4" / "aihub_audit.json").read_text(encoding="utf-8"))
    stage4_path = config.output_root / "stage4" / "aggregate.json"; stage4 = json.loads(stage4_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": "mal2026-qwen-rationale-oof-final-report-v1", "status": "completed", "completed_at": now(), "run_id": config["run_id"],
        "hypothesis": "source-balanced rationale views, input ablation, ordinal multi-task loss, train-only cutpoint calibration, and compatible unique tail essays may reduce central compression",
        "stage1_selection": stage1["selection"], "stage2_selection": stage2["selection"], "stage2_calibration": {key: stage2_cal[key] for key in ("chosen_variant", "promoted", "uncalibrated_metrics", "crossfit_calibrated_metrics")},
        "stage4_aihub_audit": {key: audit.get(key) for key in ("status", "reason", "candidate_unique_tail_records", "selected_records", "selected_integer_histograms", "token_length")},
        "stage4_selection": stage4.get("selection"), "final_selection": final["selection"],
        "prior_internal_validation": {"run_id": baseline["run_id"], "macro_integer_rmse": baseline_metrics["macro_integer_rmse"], "macro_integer_spearman": baseline_metrics["macro_integer_spearman"]},
        "new_single_final_validation": metrics,
        "rmse_change_vs_prior": float(metrics["macro_integer_rmse"]) - float(baseline_metrics["macro_integer_rmse"]),
        "target_rmse": float(config["target_rmse"]), "remaining_gap_to_target": float(metrics["macro_integer_rmse"]) - float(config["target_rmse"]),
        "validation_used_for_selection": False, "average_target_used": False, "config_sha256": config.config_sha256,
        "conclusion": "improved" if float(metrics["macro_integer_rmse"]) < float(baseline_metrics["macro_integer_rmse"]) else "not_improved",
    }
    atomic_json(config.output_root / "final_report.json", result)
    doc = ROOT / "docs/experiments/20260812_qwen_rationale_oof_multistage_v1.md"; doc.parent.mkdir(parents=True, exist_ok=True)
    recalls = metrics["axes"]
    lines = [
        "# Qwen rationale-aware OOF multistage v1", "", f"- Run: `{config['run_id']}`", f"- Git SHA: `{json.loads((config.output_root / 'preflight.json').read_text())['git_sha']}`",
        f"- Config SHA-256: `{config.config_sha256}`", f"- Data: train `{config['train_sha256']}`, validation `{config['validation_sha256']}`", f"- GPUs: `0,1,2,3`; seed `{config['seed']}`", "",
        "## Result", "", f"- Final arm: `{final['selection']['source']} / {final['selection']['arm']} / {final['selection']['calibration_variant']}`",
        f"- Macro integer RMSE: `{metrics['macro_integer_rmse']:.6f}` (prior internal `{baseline_metrics['macro_integer_rmse']:.6f}`)",
        f"- Macro Spearman: `{metrics['macro_integer_spearman']:.6f}`", f"- Macro tail RMSE: `{metrics['macro_tail_rmse']:.6f}`", f"- Gap to 0.4: `{result['remaining_gap_to_target']:.6f}`", "",
        "| Axis | RMSE | recall@1 | recall@2 | recall@5 |", "|---|---:|---:|---:|---:|",
    ]
    for axis in AXES:
        axis_metrics = recalls[axis]; per = axis_metrics["per_gold_recall"]
        lines.append(f"| {axis} | {axis_metrics['integer_rmse']:.6f} | {per['1']:.4f} | {per['2']:.4f} | {per['5']:.4f} |")
    lines += ["", "Validation was read once after train-only five-fold OOF selection. Average was never a target. Row-level predictions and text remain in ignored restricted storage.", ""]
    temporary = doc.with_name(f".{doc.name}.{os.getpid()}.tmp"); temporary.write_text("\n".join(lines), encoding="utf-8"); temporary.replace(doc)
    return result
