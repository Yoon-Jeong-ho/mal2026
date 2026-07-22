"""Train-only, score-blind Qwen-RLAIF continuations for rationale decoders.

This module intentionally keeps essays, completions, judge requests, source
identifiers, and judge observations in RAM (or in ignored runtime roots).  The
only tracked output of a run is its aggregate completion record.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
from threading import Lock
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .api_rationale_data import (
    AXES,
    RESTRICTED_ROOT,
    ROOT,
    APIRationaleContractError,
    aggregate_input_provenance,
    axes_for_task,
    decoder_messages,
    load_writing_rows,
    rationale_object,
    sha256_file,
)
from .api_rationale_sft import SUPPORTED_MODELS


DEFAULT_CONFIG_PATH = ROOT / "configs" / "rlaif_grpo_prompt_ensemble.v1.json"
STRUCTURED_SCHEMA_SUFFIXES = ("-v2", "-v3", "-v4", "-v5", "-v6", "-v7", "-v8")
PILOT_SCHEMA_SUFFIXES = ("-v3", "-v4", "-v5", "-v6", "-v7", "-v8")
TP2_SCHEMA_SUFFIXES = ("-v6", "-v7", "-v8")


def active_config_path() -> Path:
    """Resolve an explicitly versioned RLAIF config without accepting paths outside the repo."""
    configured = os.environ.get("MAL2026_RLAIF_CONFIG")
    if configured is None:
        return DEFAULT_CONFIG_PATH
    candidate = (ROOT / configured).resolve()
    _root = ROOT.resolve()
    if candidate.parent != _root / "configs" or candidate.suffix != ".json":
        raise RuntimeError("MAL2026_RLAIF_CONFIG must name a top-level configs JSON file")
    return candidate


CONFIG_PATH = active_config_path()
SFT_ROOT = ROOT / "outputs" / "api-rationale-sft-v1"


class RLAIFGRPOError(APIRationaleContractError):
    """Raised before a run can blur score, split, model, or reward boundaries."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RLAIFGRPOError(message)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _judge_module() -> Any:
    target = ROOT / "scripts" / "score_rationale_distribution_vllm_dp4.py"
    spec = importlib.util.spec_from_file_location("mal2026_rlaif_fixed_judge_template", target)
    _need(spec is not None and spec.loader is not None, "fixed judge-template module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JUDGE = _judge_module()


def _call_reward_judge(endpoint: str, task: Mapping[str, Any], max_transport_attempts: int) -> tuple[dict[str, Any], int]:
    """Return the terminal fixed-judge result and bounded transport retries.

    The fixed client retries only HTTP/connection failures and vLLM's explicit
    internal ``finish_reason=error``. Incomplete ``length`` finishes, parsed
    invalid scores, and abstentions remain terminal. The counter separates
    logical judge observations from transient recovery in aggregate records.
    """
    _need(max_transport_attempts >= 1, "reward transport-attempt limit is invalid")
    value = JUDGE.call_with_transport_attempts(endpoint, dict(task), max_transport_attempts)
    _need(isinstance(value, dict), "fixed judge call did not return an object")
    attempts = value.get("attempts")
    _need(type(attempts) is int and 1 <= attempts <= max_transport_attempts, "fixed judge retry count is invalid")
    return value, attempts - 1


@dataclass(frozen=True)
class RLAIFSettings:
    """Versioned constants shared by every arm; no raw data belongs here."""

    schema_version: str
    run_id_prefix: str
    fixed_v6_config: str
    fixed_v6_config_sha256: str
    judge: Mapping[str, Any]
    policy: Mapping[str, Any]
    arms: tuple[str, ...]
    inputs: Mapping[str, Any]
    reward: Mapping[str, Any]
    runtime: Mapping[str, Any]
    hard_gates: Mapping[str, Any]
    privacy: Mapping[str, Any]

    @classmethod
    def from_json(cls, path: Path = CONFIG_PATH) -> "RLAIFSettings":
        raw = json.loads(path.read_text(encoding="utf-8"))
        _need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "RLAIF config has unknown or missing fields")
        value = cls(
            schema_version=str(raw["schema_version"]), run_id_prefix=str(raw["run_id_prefix"]),
            fixed_v6_config=str(raw["fixed_v6_config"]), fixed_v6_config_sha256=str(raw["fixed_v6_config_sha256"]),
            judge=raw["judge"], policy=raw["policy"], arms=tuple(raw["arms"]), inputs=raw["inputs"],
            reward=raw["reward"], runtime=raw["runtime"], hard_gates=raw["hard_gates"], privacy=raw["privacy"],
        )
        value.validate()
        return value

    @property
    def fixed_v6_path(self) -> Path:
        path = ROOT / self.fixed_v6_config
        _need(path.is_file() and not path.is_symlink(), "fixed v6 judge config is unavailable")
        return path

    def fixed_prompt_template(self) -> dict[str, Any]:
        """Copy only the immutable score-blind prompt template into RAM.

        The copied template is not an RL configuration: its v6 permission flag
        remains false.  The distinct RLAIF config above is the authorization and
        seed/runtime contract for train-only reward calls.
        """
        _need(_sha(self.fixed_v6_path) == self.fixed_v6_config_sha256, "fixed v6 config digest changed")
        raw = json.loads(self.fixed_v6_path.read_text(encoding="utf-8"))
        _need(isinstance(raw, dict), "fixed v6 judge config is invalid")
        protocol = raw.get("protocol")
        _need(isinstance(protocol, dict), "fixed v6 protocol is absent")
        _need(protocol.get("candidate_isolated") is True and protocol.get("reference_score_in_prompt") is False, "judge template is not score blind")
        _need(protocol.get("candidate_projection") == "diagnosis_only_rationale_v1", "judge template rationale projection differs")
        _need(protocol.get("response_contract") == "required_scores_only_v1", "judge template response contract differs")
        _need(protocol.get("sft_dpo_grpo_permitted") is False, "v6 evaluation config must remain training prohibited")
        prompt_types = protocol.get("prompt_types")
        _need(isinstance(prompt_types, list) and len(prompt_types) == 5, "five fixed judge prompt forms are required")
        return raw

    def validate(self) -> None:
        _need(self.schema_version in {"mal2026-rlaif-grpo-prompt-ensemble-v1", "mal2026-rlaif-grpo-prompt-ensemble-v2", "mal2026-rlaif-grpo-prompt-ensemble-v3", "mal2026-rlaif-grpo-prompt-ensemble-v4", "mal2026-rlaif-grpo-prompt-ensemble-v5", "mal2026-rlaif-grpo-prompt-ensemble-v6", "mal2026-rlaif-grpo-prompt-ensemble-v7", "mal2026-rlaif-grpo-prompt-ensemble-v8"}, "unexpected RLAIF config schema")
        expected_prefix = self.schema_version.removeprefix("mal2026-") + "-"
        _need(self.run_id_prefix == expected_prefix, "RLAIF run prefix differs")
        _need(self.arms == ("all5", "random1"), "RLAIF arms must be all5 and random1")
        _need(isinstance(self.judge, Mapping) and isinstance(self.policy, Mapping), "judge/policy configuration is invalid")
        _need(self.judge.get("model_id") == "Qwen/Qwen3.6-35B-A3B-FP8", "judge identity differs")
        judge_path = ROOT / str(self.judge.get("model_path", ""))
        _need(judge_path.is_dir() and not judge_path.is_symlink(), "local immutable Qwen judge snapshot is unavailable")
        _need(self.judge.get("response_contract") == "required_scores_only_v1", "judge response contract differs")
        seeds = self.judge.get("training_seed_by_prompt_index")
        _need(isinstance(seeds, list) and len(seeds) == 5 and len(set(seeds)) == 5 and all(type(seed) is int for seed in seeds), "training judge seed schedule differs")
        _need(self.policy.get("algorithm") == "grpo" and self.policy.get("num_generations") == 4, "GRPO group contract differs")
        _need(self.policy.get("loss_type") == "dr_grpo" and self.policy.get("scale_rewards") == "none", "GRPO loss/normalization differs")
        _need(self.policy.get("beta") == 0.02 and self.policy.get("learning_rate") == 1e-6, "KL or learning rate differs")
        _need(self.policy.get("training_dtype") == "float32", "RLAIF numerical policy requires float32")
        expected_per_device_batch = 16 if self.schema_version.endswith(TP2_SCHEMA_SUFFIXES) else 8
        _need(self.policy.get("per_device_train_batch_size_full") == expected_per_device_batch and self.policy.get("generation_batch_size_full") == 64 and self.policy.get("steps_per_generation_full") == 4, "full batching contract differs")
        _need(self.inputs.get("train_expected_essays") == 2000 and self.inputs.get("contrastive_holdout_count") == 80, "train population/holdout contract differs")
        _need(self.inputs.get("validation_expected_essays") == 400, "validation population differs")
        _need(self.reward.get("invalid_completion_reward") == -1.0, "invalid completion reward differs")
        expected_character_limit = 128 if self.schema_version.endswith(("-v3", "-v4")) else 192
        _need(self.reward.get("field_character_limit") == expected_character_limit, "policy rationale length contract differs")
        if self.schema_version.endswith("-v1"):
            _need(self.policy.get("rollout_backend") is None, "v1 must retain its native policy rollout declaration")
            _need(self.runtime.get("physical_gpus") == [0, 1, 2, 3] and self.runtime.get("full_reward_gpus") == [0, 3] and self.runtime.get("full_policy_gpus") == [1, 2], "authorized GPU partition differs")
            _need(self.runtime.get("reward_data_parallel_size") == 2 and self.runtime.get("enforce_eager") is True, "reward server topology differs")
        else:
            _need(self.policy.get("rollout_backend") == "vllm_structured_outputs_http_v1", "structured rollout backend differs")
            _need(self.runtime.get("physical_gpus") == [0, 1, 2, 3] and self.runtime.get("full_reward_gpus") == [3], "structured rollout GPU scope differs")
            if self.schema_version.endswith(TP2_SCHEMA_SUFFIXES):
                _need(self.runtime.get("full_rollout_gpus") == [0, 1] and self.runtime.get("full_policy_gpus") == [2] and self.runtime.get("reward_data_parallel_size") == 1 and self.runtime.get("rollout_tensor_parallel_size") == 2, "TP2 resource partition differs")
            else:
                _need(self.runtime.get("full_rollout_gpus") == [0] and self.runtime.get("full_policy_gpus") == [1, 2] and self.runtime.get("reward_data_parallel_size") == 1 and self.runtime.get("rollout_tensor_parallel_size") == 1, "structured rollout/reward topology differs")
            _need(self.runtime.get("rollout_max_model_len") == 3072 and self.runtime.get("rollout_max_num_seqs") == 192 and self.runtime.get("enforce_eager") is False, "structured vLLM rollout settings differ")
            if self.schema_version.endswith("-v4"):
                _need(self.policy.get("rollout_json_schema_enforces_field_limit") is False, "v4 must leave the field cap to canonical parsing")
            if self.schema_version.endswith("-v5"):
                _need(self.policy.get("rollout_structured_output_mode") == "json_object", "v5 must use vLLM JSON-object mode")
                _need(self.policy.get("rollout_json_schema_enforces_field_limit") is False, "v5 cannot claim a JSON-schema field cap")
            if self.schema_version.endswith(TP2_SCHEMA_SUFFIXES):
                _need(self.policy.get("rollout_structured_output_mode") == "json_object", "TP2 policy must use vLLM JSON-object mode")
                _need(self.policy.get("rollout_json_schema_enforces_field_limit") is False, "TP2 policy cannot claim a JSON-schema field cap")
            if self.schema_version.endswith("-v7"):
                _need(self.runtime.get("policy_training_cuda_alloc_conf") == "expandable_segments:True", "v7 allocator repair differs")
            if self.schema_version.endswith("-v8"):
                _need(self.runtime.get("policy_training_cuda_alloc_conf") == "expandable_segments:True", "v8 allocator repair differs")
                _need(self.reward.get("unscorable_judge_group_policy") == "discard_generation_group", "v8 judge-failure policy differs")
                _need(self.reward.get("unscorable_judge_group_reward") == 0.0, "v8 discarded-group reward differs")
                _need(self.reward.get("max_unscorable_judge_fraction") == 0.001, "v8 judge-failure ceiling differs")
                _need(self.reward.get("max_discarded_reward_group_fraction") == 0.01, "v8 discarded-group ceiling differs")
        _need(self.privacy == {"source_writing_scores_read_or_prompted": False, "candidate_scores_read_or_prompted": False, "raw_prompts_or_completions_tracked": False}, "privacy contract differs")
        self.fixed_prompt_template()


@dataclass(frozen=True)
class RLAIFRunConfig:
    """Arm-specific runtime configuration written under ignored outputs."""

    schema_version: str
    run_id: str
    base_key: str
    task: str
    arm: str
    phase: str
    output_dir: str
    source_adapter: str
    reward_endpoint: str
    reward_server_attestation: str
    train_limit: int
    max_steps: int
    per_device_train_batch_size: int
    generation_batch_size: int
    steps_per_generation: int
    seed: int

    @classmethod
    def from_json(cls, path: Path) -> "RLAIFRunConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        _need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "RLAIF run config has unknown or missing fields")
        value = cls(**raw)
        value.validate(RLAIFSettings.from_json())
        return value

    def validate(self, settings: RLAIFSettings) -> None:
        _need(self.schema_version == "mal2026-rlaif-grpo-run-v1", "RLAIF run schema differs")
        _need(self.base_key in SUPPORTED_MODELS and self.task in {"bundle", *AXES}, "unknown source adapter")
        allowed_phases = {"gpu0_actual", "full"}
        if settings.schema_version.endswith(PILOT_SCHEMA_SUFFIXES):
            allowed_phases.add("pilot")
        _need(self.arm in settings.arms and self.phase in allowed_phases, "RLAIF arm or phase differs")
        expected_prefix = f"{settings.run_id_prefix}{self.base_key}-{self.task}-{self.arm}-{self.phase}-"
        _need(self.run_id.startswith(expected_prefix), "run id does not bind source/task/arm/phase")
        root = Path(self.output_dir)
        _need(root.is_absolute() and root.parent.name == settings.run_id_prefix.rstrip("-"), "output root differs")
        adapter = Path(self.source_adapter)
        _need(adapter.is_dir() and not adapter.is_symlink() and (adapter / "adapter_config.json").is_file(), "source SFT adapter is unavailable")
        _need(self.reward_endpoint.startswith("http://127.0.0.1:"), "reward endpoint must be local")
        _need(Path(self.reward_server_attestation).is_file() and not Path(self.reward_server_attestation).is_symlink(), "reward attestation is unavailable")
        if self.phase == "full":
            expected_batch = 16 if settings.schema_version.endswith(TP2_SCHEMA_SUFFIXES) else 8
            _need((self.train_limit, self.max_steps, self.per_device_train_batch_size, self.generation_batch_size, self.steps_per_generation) == (1920, 480, expected_batch, 64, 4), "full population/batch/update contract differs")
        elif self.phase == "pilot":
            expected_batch = 16 if settings.schema_version.endswith(TP2_SCHEMA_SUFFIXES) else 8
            _need(settings.schema_version.endswith(PILOT_SCHEMA_SUFFIXES) and (self.train_limit, self.max_steps, self.per_device_train_batch_size, self.generation_batch_size, self.steps_per_generation) == (320, 80, expected_batch, 64, 4), "pilot population/batch/update contract differs")
        else:
            _need((self.train_limit, self.max_steps, self.per_device_train_batch_size, self.generation_batch_size, self.steps_per_generation) == (4, 1, 4, 4, 1), "GPU0 actual preflight contract differs")
        _need(self.seed == int(settings.policy["seed"]), "RLAIF seed differs")


def canonical_completion(text: Any, axes: Sequence[str], character_limit: int) -> dict[str, Any] | None:
    """Accept only a complete, bounded rationale-only JSON response."""
    selected = tuple(axes)
    if not isinstance(text, str):
        return None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict) or set(raw) != {"schema_version", *selected} or raw.get("schema_version") != "rationale-only-v1":
        return None
    diagnoses: dict[str, str] = {}
    for axis in selected:
        item = raw.get(axis)
        if not isinstance(item, dict) or set(item) != {"rationale"} or not isinstance(item.get("rationale"), str):
            return None
        rationale = item["rationale"].strip()
        if not rationale or len(rationale) > character_limit or not any("가" <= character <= "힣" for character in rationale):
            return None
        diagnoses[axis] = rationale
    return rationale_object(diagnoses, selected)


def canonical_completion_text(text: Any, axes: Sequence[str], character_limit: int) -> str | None:
    value = canonical_completion(text, axes, character_limit)
    return None if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def random_prompt_index(seed: int, arm: str, source_key: str, canonical_text: str) -> int:
    """Stable one-of-five selection without source IDs or model scores."""
    _need(arm == "random1", "random prompt selection is only valid for random1")
    digest = sha256(f"{seed}|{arm}|{source_key}|{canonical_text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % 5


def _completion_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        content = value[0].get("content")
        return content if isinstance(content, str) else None
    return None


def _contrastive_holdout(settings: RLAIFSettings, rows: Sequence[Any]) -> set[str]:
    """Reconstruct and bind the previously reserved 80-group train holdout."""
    path = RESTRICTED_ROOT / "judge_contrastive_validity_v1" / str(settings.inputs["contrastive_holdout_run_id"]) / "manifest.json"
    _need(path.is_file() and not path.is_symlink(), "contrastive holdout manifest is unavailable")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    provenance = manifest.get("provenance") if isinstance(manifest, dict) else None
    _need(isinstance(provenance, dict) and provenance.get("calibration_source_id_set_sha256") == settings.inputs["contrastive_holdout_digest"], "reserved contrastive digest differs")
    template = settings.fixed_prompt_template()
    selected = sorted(rows, key=lambda row: JUDGE.opaque(template["seed"], "contrastive-validity-v1", row.identifier))[:int(settings.inputs["contrastive_holdout_count"])]
    selected_ids = {str(row.identifier) for row in selected}
    digest = sha256("".join(f"{value}\n" for value in sorted(selected_ids)).encode("utf-8")).hexdigest()
    _need(len(selected_ids) == int(settings.inputs["contrastive_holdout_count"]) and digest == settings.inputs["contrastive_holdout_digest"], "reconstructed contrastive holdout differs")
    return selected_ids


def train_examples(settings: RLAIFSettings, run: RLAIFRunConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return private score-blind policy prompts; never access writing labels."""
    rows = load_writing_rows("train", include_scores=False)
    _need(len(rows) == settings.inputs["train_expected_essays"], "canonical train population differs")
    held_out = _contrastive_holdout(settings, rows)
    eligible = [row for row in rows if row.identifier not in held_out]
    _need(len(eligible) == 1920, "eligible train population differs")
    if run.phase == "pilot":
        # A fixed opaque rank avoids a first-rows subset whose content could
        # depend on the original source-file ordering.  Identifiers are used
        # only in memory to form the deterministic selection and are never
        # written to the aggregate record.
        chosen = sorted(
            eligible,
            key=lambda row: JUDGE.opaque(settings.policy["seed"], "rlaif-v3-pilot-subset", row.identifier),
        )[:run.train_limit]
        selection_method = "opaque_sha256_rank(seed:rlaif-v3-pilot-subset:source_id)"
    else:
        chosen = eligible[:run.train_limit]
        selection_method = "canonical_source_order_all_eligible_prefix"
    _need(len(chosen) == run.train_limit, "RLAIF train limit exceeds eligible train population")
    axes = axes_for_task(run.task)
    values: list[dict[str, Any]] = []
    for row in chosen:
        values.append({
            "prompt": decoder_messages(row, axes),
            "source_key": JUDGE.opaque(run.run_id, "policy-source", row.identifier),
            "judge_entry": {"sentences": JUDGE.sentence_list(row.essay)},
        })
    _need(all(value["judge_entry"]["sentences"] for value in values), "empty train essay sentence segmentation")
    provenance = {
        "train_rows_loaded": len(rows), "contrastive_holdout_rows": len(held_out), "eligible_train_rows": len(eligible),
        "policy_train_rows": len(values), "policy_selection_method": selection_method, "train_source_sha256": _sha(ROOT / "eval" / "train.jsonl"),
        "contrastive_holdout_digest": settings.inputs["contrastive_holdout_digest"],
        "source_writing_scores_read_or_prompted": False, "candidate_scores_read_or_prompted": False,
        "validation_source_text_opened": 0,
    }
    return values, provenance


def _score_to_reward(scores: Mapping[str, Any], axes: Sequence[str]) -> float:
    selected = tuple(axes)
    _need(set(scores) == set(AXES), "judge score axes differ")
    values = [scores[axis] for axis in selected]
    _need(all(type(value) in {int, float} and not isinstance(value, bool) and 1.0 <= float(value) <= 5.0 for value in values), "judge score is outside 1--5")
    return (statistics.fmean(float(value) for value in values) - 3.0) / 2.0


class QwenPointReward:
    """Synchronous batched local Qwen reward callable for ``GRPOTrainer``."""

    __name__ = "qwen_pointwise_rationale_reward"

    def __init__(self, settings: RLAIFSettings, run: RLAIFRunConfig):
        self.settings, self.run = settings, run
        self.axes = axes_for_task(run.task)
        self.template = settings.fixed_prompt_template()
        self.prompt_types = list(self.template["protocol"]["prompt_types"])
        self.totals: Counter[str] = Counter()
        self.form_counts: Counter[str] = Counter()
        self.failure_categories: Counter[str] = Counter()

    def _task(self, *, source_key: str, entry: Mapping[str, Any], rationale: Mapping[str, Any], prompt_index: int, canonical_text: str) -> dict[str, Any]:
        prompt_type = self.prompt_types[prompt_index]
        seed = self.settings.judge["training_seed_by_prompt_index"][prompt_index]
        request_template = copy.deepcopy(self.template)
        request_template["sampling"] = {"temperature": self.settings.judge["temperature"]}
        request_template["request"] = {
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": self.settings.judge["max_tokens"],
            "top_p": self.settings.judge["top_p"],
        }
        request_entry = {"sentences": list(entry["sentences"]), "rationale": rationale}
        key = JUDGE.opaque(self.run.run_id, "reward", source_key, sha256(canonical_text.encode("utf-8")).hexdigest(), prompt_index)
        return {
            "opaque_request_key": key, "prompt_type_id": prompt_type["id"], "sampling_seed": seed,
            "response_contract": self.settings.judge["response_contract"],
            "body": JUDGE.request_body(request_template, self.settings.judge["model_id"], request_entry, list(self.axes), prompt_type["layout"], seed, prompt_type["review_emphasis"]),
        }

    def _all_reduce(self, values: Counter[str], forms: Counter[str]) -> tuple[Counter[str], Counter[str]]:
        try:
            import torch
            import torch.distributed as dist
        except ImportError:  # pragma: no cover - runtime dependency
            return values, forms
        names = ("completions", "parse_valid", "parse_invalid", "judge_requests", "judge_calls", "judge_unscorable", "transport_retries", "discarded_reward_groups", "discarded_reward_completions", "reward_scaled_sum", "reward_scaled_sq_sum")
        form_ids = tuple(str(item["id"]) for item in self.prompt_types)
        packed = torch.tensor([float(values.get(name, 0.0)) for name in names] + [float(forms.get(name, 0.0)) for name in form_ids], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        reduced_values = Counter({name: float(packed[index].item()) for index, name in enumerate(names)})
        offset = len(names)
        reduced_forms = Counter({name: float(packed[offset + index].item()) for index, name in enumerate(form_ids)})
        return reduced_values, reduced_forms

    def __call__(self, prompts: list[Any], completions: list[Any], completion_ids: list[Any], source_key: list[str], judge_entry: list[Mapping[str, Any]], **_: Any) -> list[float]:
        _need(len(prompts) == len(completions) == len(completion_ids) == len(source_key) == len(judge_entry), "GRPO reward batch columns differ")
        groups: dict[str, list[int]] = {}
        for index, value in enumerate(source_key):
            groups.setdefault(str(value), []).append(index)
        _need(all(len(indices) == int(self.settings.policy["num_generations"]) for indices in groups.values()), "GRPO reward groups differ from the declared generation count")
        rewards: list[float | None] = [None] * len(completions)
        tasks: list[tuple[int, dict[str, Any]]] = []
        local = Counter(completions=len(completions))
        local_forms: Counter[str] = Counter()
        local_failures: Counter[str] = Counter()
        limit = int(self.settings.reward["field_character_limit"])
        for index, completion in enumerate(completions):
            raw_text = _completion_text(completion)
            canonical_text = canonical_completion_text(raw_text, self.axes, limit)
            if canonical_text is None:
                rewards[index] = float(self.settings.reward["invalid_completion_reward"])
                local["parse_invalid"] += 1
                continue
            rationale = json.loads(canonical_text)
            indices = range(5) if self.run.arm == "all5" else (random_prompt_index(self.run.seed, self.run.arm, str(source_key[index]), canonical_text),)
            for prompt_index in indices:
                task = self._task(source_key=str(source_key[index]), entry=judge_entry[index], rationale=rationale, prompt_index=prompt_index, canonical_text=canonical_text)
                tasks.append((index, task)); local_forms[str(task["prompt_type_id"])] += 1
            local["parse_valid"] += 1
        scores: dict[int, list[float]] = {index: [] for index in range(len(completions))}
        unscorable_indices: set[int] = set()
        if tasks:
            local["judge_requests"] += len(tasks)
            workers = min(int(self.settings.reward["client_max_inflight_per_rank"]), len(tasks))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                attempts = int(self.settings.reward["max_transport_attempts"])
                futures = {pool.submit(_call_reward_judge, self.run.reward_endpoint, task, attempts): index for index, task in tasks}
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        result, extra_retries = future.result()
                    except Exception as exc:  # network/worker integration failure, never a policy label
                        raise RLAIFGRPOError("reward judge worker failed") from exc
                    local["transport_retries"] += int(extra_retries)
                    if not result.get("scored") or not result.get("schema_valid") or result.get("failure_category") is not None or not isinstance(result.get("scores"), dict):
                        # Never convert an incomplete/failed judge response into
                        # a low-quality rationale label. v8 discards the whole
                        # four-completion GRPO group, giving it equal rewards and
                        # therefore no relative-advantage update. Earlier
                        # protocols retain their fail-closed behavior.
                        if self.settings.reward.get("unscorable_judge_group_policy") != "discard_generation_group":
                            raise RLAIFGRPOError(f"reward judge transport/schema failure: {result.get('failure_category')}")
                        unscorable_indices.add(index)
                        local["judge_unscorable"] += 1
                        local_failures[str(result.get("failure_category") or "unscored_response")] += 1
                        continue
                    scores[index].append(_score_to_reward(result["scores"], self.axes))
                    local["judge_calls"] += 1
        if unscorable_indices:
            discarded_sources = {str(source_key[index]) for index in unscorable_indices}
            local["discarded_reward_groups"] += len(discarded_sources)
            for source in discarded_sources:
                indices = groups[source]
                local["discarded_reward_completions"] += len(indices)
                for index in indices:
                    rewards[index] = float(self.settings.reward["unscorable_judge_group_reward"])
        for index, value in enumerate(rewards):
            if value is None:
                observations = scores[index]
                _need(bool(observations) and (len(observations) == 5 if self.run.arm == "all5" else len(observations) == 1), "reward observation count differs from arm contract")
                rewards[index] = statistics.fmean(observations)
            local["reward_scaled_sum"] += float(rewards[index])
            local["reward_scaled_sq_sum"] += float(rewards[index]) ** 2
        reduced, reduced_forms = self._all_reduce(local, local_forms)
        self.totals.update(reduced); self.form_counts.update(reduced_forms); self.failure_categories.update(local_failures)
        return [float(value) for value in rewards]

    def aggregate(self) -> dict[str, Any]:
        completed = float(self.totals.get("completions", 0.0))
        parsed = float(self.totals.get("parse_valid", 0.0))
        requested = float(self.totals.get("judge_requests", 0.0))
        group_count = completed / float(self.settings.policy["num_generations"]) if completed else 0.0
        mean = self.totals.get("reward_scaled_sum", 0.0) / completed if completed else None
        variance = self.totals.get("reward_scaled_sq_sum", 0.0) / completed - float(mean) ** 2 if completed and mean is not None else None
        return {
            "policy_completions": int(completed), "parse_valid": int(parsed), "parse_invalid": int(self.totals.get("parse_invalid", 0.0)),
            "parse_valid_rate": round(parsed / completed, 6) if completed else None,
            "judge_requests": int(requested), "judge_calls": int(self.totals.get("judge_calls", 0.0)),
            "judge_unscorable": int(self.totals.get("judge_unscorable", 0.0)),
            "judge_unscorable_fraction": round(float(self.totals.get("judge_unscorable", 0.0)) / requested, 8) if requested else None,
            "judge_failure_categories": dict(sorted(self.failure_categories.items())),
            "discarded_reward_groups": int(self.totals.get("discarded_reward_groups", 0.0)),
            "discarded_reward_completions": int(self.totals.get("discarded_reward_completions", 0.0)),
            "discarded_reward_group_fraction": round(float(self.totals.get("discarded_reward_groups", 0.0)) / group_count, 8) if group_count else None,
            "transport_retries": int(self.totals.get("transport_retries", 0.0)),
            "mapped_reward_mean": round(float(mean), 6) if mean is not None else None,
            "mapped_reward_std": round(math.sqrt(max(0.0, float(variance))), 6) if variance is not None else None,
            "prompt_form_calls": {key: int(value) for key, value in sorted(self.form_counts.items())},
        }


def _policy_response_schema(axes: Sequence[str], character_limit: int, *, enforce_character_limit: bool = True) -> dict[str, Any]:
    """Express the rationale object contract as a strict JSON schema.

    Canonical parsing always applies the private-policy character bound.  In
    v4 the schema-side cap is intentionally disabled because v3 showed a
    pathological constrained-decoding tail while valid outputs stayed below
    that same parser-enforced cap.
    """
    _need(character_limit >= 1, "policy rationale character limit is invalid")
    rationale: dict[str, Any] = {"type": "string", "minLength": 1}
    if enforce_character_limit:
        rationale["maxLength"] = character_limit
    axis = {
        "type": "object",
        "properties": {"rationale": rationale},
        "required": ["rationale"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"schema_version": {"const": "rationale-only-v1"}, **{name: axis for name in axes}},
        "required": ["schema_version", *axes],
        "additionalProperties": False,
    }


class StructuredVLLMRollout:
    """vLLM JSON-schema rollout with rank-zero, step-bound LoRA synchronization.

    TRL 0.29.1 cannot use its integrated vLLM path with the installed vLLM
    0.25.1.  Its maintained ``rollout_func`` hook still lets the trainer own
    policy/ref log probabilities and optimization while this class supplies
    only constrained sampled token IDs.  No prompt or completion leaves RAM.
    """

    def __init__(self, settings: RLAIFSettings, run: RLAIFRunConfig, tokenizer: Any):
        self.settings, self.run, self.tokenizer = settings, run, tokenizer
        self.axes = tuple(axes_for_task(run.task))
        self.endpoint = os.environ.get("MAL2026_RLAIF_ROLLOUT_ENDPOINT", "")
        self.attestation_path = Path(os.environ.get("MAL2026_RLAIF_ROLLOUT_ATTESTATION", ""))
        self.sync_root = Path(os.environ.get("MAL2026_RLAIF_ROLLOUT_SYNC_DIR", ""))
        self.alias = f"rlaif_policy_{run.run_id.replace('-', '_')}"
        self.last_synced_step: int | None = None
        self.active_snapshot: Path | None = None
        self.sync_calls = 0
        self.request_count = 0
        self.completion_count = 0
        # A completed HTTP response can carry a non-``stop`` terminal reason
        # (notably ``length``).  The actual policy-quality boundary is the
        # canonical rationale parser below, not this transport metadata.  Keep
        # only aggregate reason counts so a runtime anomaly is auditable
        # without retaining generated text.
        self.finish_reason_counts: Counter[str] = Counter()
        self.finish_reason_lock = Lock()
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        parsed = urlparse(self.endpoint)
        _need(parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port is not None, "vLLM rollout endpoint must be local HTTP")
        output = Path(self.run.output_dir).resolve()
        _need(self.sync_root.is_absolute() and self.sync_root.parent == output and not self.sync_root.is_symlink(), "vLLM rollout sync root differs")
        _need(self.attestation_path.is_file() and not self.attestation_path.is_symlink(), "vLLM rollout attestation is unavailable")
        value = json.loads(self.attestation_path.read_text(encoding="utf-8"))
        model_id, revision = SUPPORTED_MODELS[self.run.base_key]
        expected = {
            "schema_version": "mal2026-rlaif-grpo-vllm-policy-server-attestation-v1",
            "server_host": "127.0.0.1", "server_port": parsed.port,
            "physical_gpus": list(self.settings.runtime["full_rollout_gpus"]), "tensor_parallel_size": int(self.settings.runtime["rollout_tensor_parallel_size"]),
            "max_model_len": int(self.settings.runtime["rollout_max_model_len"]),
            "max_num_seqs": int(self.settings.runtime["rollout_max_num_seqs"]),
            "model_id": model_id, "model_revision": revision,
            "dynamic_lora": True, "structured_outputs_json_schema": True,
            "enforce_eager": bool(self.settings.runtime["enforce_eager"]),
            "server_process_environment_verified": True,
        }
        _need(isinstance(value, dict) and all(value.get(key) == expected_value for key, expected_value in expected.items()), "vLLM rollout attestation differs")

    def _request(self, messages: Any, seed: int, num_choices: int) -> list[str]:
        _need(isinstance(messages, list) and all(isinstance(item, dict) for item in messages), "rollout messages differ")
        _need(1 <= num_choices <= int(self.settings.policy["num_generations"]), "vLLM rollout choice count differs")
        mode = str(self.settings.policy.get("rollout_structured_output_mode", "json_schema"))
        _need(mode in {"json_schema", "json_object"}, "unsupported vLLM structured output mode")
        response_format: dict[str, Any]
        if mode == "json_schema":
            schema = _policy_response_schema(
                self.axes,
                int(self.settings.reward["field_character_limit"]),
                enforce_character_limit=bool(self.settings.policy.get("rollout_json_schema_enforces_field_limit", True)),
            )
            response_format = {"type": "json_schema", "json_schema": {"name": "mal2026_rationale_only_v1", "strict": True, "schema": schema}}
        else:
            response_format = {"type": "json_object"}
        body = {
            "model": self.alias,
            "messages": messages,
            "n": num_choices,
            "temperature": float(self.settings.policy["sampling_temperature"]),
            "top_p": float(self.settings.policy["sampling_top_p"]),
            "max_tokens": int(self.settings.policy["max_completion_tokens"]),
            "seed": seed,
            "response_format": response_format,
        }
        wire = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        last_error: Exception | None = None
        for _ in range(int(self.settings.reward["max_transport_attempts"])):
            try:
                with urlopen(Request(self.endpoint + "/v1/chat/completions", data=wire, headers={"Content-Type": "application/json"}, method="POST"), timeout=600) as response:
                    outer = json.loads(response.read().decode("utf-8"))
                choices = outer.get("choices") if isinstance(outer, dict) else None
                _need(isinstance(choices, list) and len(choices) == num_choices, "vLLM rollout envelope differs")
                outputs: list[str] = []
                for choice in choices:
                    _need(isinstance(choice, dict), "vLLM rollout choice differs")
                    finish_reason = choice.get("finish_reason")
                    _need(isinstance(finish_reason, str) or finish_reason is None, "vLLM rollout finish reason differs")
                    message = choice.get("message")
                    content = message.get("content") if isinstance(message, dict) else None
                    _need(isinstance(content, str), "vLLM rollout content is missing")
                    # Do not retry or discard a returned policy completion
                    # solely due to its terminal tag.  If the text is
                    # complete, the canonical parser/reward path accepts and
                    # scores it; if it is truncated or malformed, that same
                    # parser applies the existing policy-invalid reward.  A
                    # strict ``finish_reason == stop`` check instead aborts
                    # the whole training run before either outcome is known.
                    with self.finish_reason_lock:
                        self.finish_reason_counts["<none>" if finish_reason is None else finish_reason] += 1
                    outputs.append(content)
                return outputs
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, RLAIFGRPOError) as exc:
                last_error = exc
        raise RLAIFGRPOError("vLLM policy rollout transport/schema failure") from last_error

    def _sync_adapter(self, trainer: Any, step: int) -> None:
        accelerator = trainer.accelerator
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.sync_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            next_snapshot = self.sync_root / f"step-{step:06d}"
            _need(not next_snapshot.exists(), "refusing to overwrite a vLLM rollout adapter snapshot")
            model = accelerator.unwrap_model(trainer.model)
            model.save_pretrained(str(next_snapshot), selected_adapters=["default"], safe_serialization=True)
            _need((next_snapshot / "adapter_config.json").is_file(), "vLLM rollout adapter snapshot is incomplete")
            body = json.dumps({"lora_name": self.alias, "lora_path": str(next_snapshot), "load_inplace": True}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            try:
                with urlopen(Request(self.endpoint + "/v1/load_lora_adapter", data=body, headers={"Content-Type": "application/json"}, method="POST"), timeout=180) as response:
                    _need(200 <= response.status < 300, "vLLM rollout adapter reload failed")
                    response.read()
            except (HTTPError, URLError, TimeoutError) as exc:
                raise RLAIFGRPOError("vLLM rollout adapter reload transport failure") from exc
            if self.active_snapshot is not None and self.active_snapshot.exists():
                shutil.rmtree(self.active_snapshot)
            self.active_snapshot = next_snapshot
            self.sync_calls += 1
        accelerator.wait_for_everyone()

    def __call__(self, prompts: list[Any], trainer: Any) -> dict[str, Any]:
        step = int(trainer.state.global_step)
        if self.last_synced_step != step:
            self._sync_adapter(trainer, step)
            self.last_synced_step = step
        # GRPO's RepeatSampler already places each source prompt in the local
        # batch once per configured generation.  A custom rollout must return
        # exactly one completion for each of those rows—not multiply them by
        # ``num_generations`` again.  Coalesce only adjacent identical rows
        # into one vLLM ``n=group_size`` request, then restore their original
        # order.  This retains vLLM batching while preserving TRL's reward
        # column alignment.
        prompt_ids: list[list[int]] = []
        completion_ids: list[list[int]] = []
        requests: list[tuple[Any, list[int], int, int]] = []
        current_messages: Any | None = None
        current_digest: bytes | None = None
        current_ids: list[int] | None = None
        current_count = 0

        def flush_group() -> None:
            nonlocal current_messages, current_digest, current_ids, current_count
            if current_messages is None:
                return
            _need(current_digest is not None and current_ids is not None, "vLLM rollout group differs")
            _need(current_count == int(self.settings.policy["num_generations"]), "GRPO rollout prompt group differs")
            seed = (int.from_bytes(current_digest[:8], "big") ^ self.run.seed ^ step ^ len(requests)) % (2**31 - 1)
            requests.append((current_messages, current_ids, current_count, seed))
            current_messages = None
            current_digest = None
            current_ids = None
            current_count = 0

        for messages in prompts:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            # Midm's fast tokenizer returns a tokenizers.Encoding while most
            # Transformers tokenizers return a Python list.  Normalize only
            # token IDs; never stringify or log the private rendered prompt.
            if isinstance(rendered, Mapping):
                rendered_ids = rendered.get("input_ids")
            elif hasattr(rendered, "ids"):
                rendered_ids = list(rendered.ids)
            elif hasattr(rendered, "tolist"):
                rendered_ids = rendered.tolist()
                if rendered_ids and isinstance(rendered_ids[0], list):
                    _need(len(rendered_ids) == 1, "policy chat-template batch dimension differs")
                    rendered_ids = rendered_ids[0]
            else:
                rendered_ids = rendered
            _need(isinstance(rendered_ids, list) and all(isinstance(item, int) for item in rendered_ids), "policy chat-template tokenization differs")
            digest = sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).digest()
            if current_messages is None:
                current_messages, current_digest, current_ids, current_count = messages, digest, rendered_ids, 1
            elif digest == current_digest:
                _need(rendered_ids == current_ids, "GRPO rollout repeated prompt tokenization differs")
                current_count += 1
            else:
                flush_group()
                current_messages, current_digest, current_ids, current_count = messages, digest, rendered_ids, 1
        flush_group()
        _need(sum(count for _, _, count, _ in requests) == len(prompts), "GRPO rollout prompt group total differs")
        with ThreadPoolExecutor(max_workers=min(len(requests), 32)) as pool:
            grouped = list(pool.map(lambda item: self._request(item[0], item[3], item[2]), requests))
        eos = self.tokenizer.eos_token_id
        _need(isinstance(eos, int), "policy tokenizer lacks EOS")
        for request, outputs in zip(requests, grouped, strict=True):
            _, rendered_ids, expected_count, _ = request
            _need(len(outputs) == expected_count, "vLLM rollout group response differs")
            for text in outputs:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                _need(isinstance(ids, list) and ids, "vLLM rollout tokenization is empty")
                prompt_ids.append(rendered_ids)
                completion_ids.append([*ids, eos])
        expected = len(prompts)
        _need(len(completion_ids) == expected, "vLLM rollout group size differs")
        self.request_count += len(requests)
        self.completion_count += len(completion_ids)
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": None}

    def aggregate(self) -> dict[str, Any]:
        return {
            "backend": "vllm_structured_outputs_http_v1",
            "sync_calls_rank_zero": self.sync_calls,
            "rollout_requests_rank_zero": self.request_count,
            "rollout_completions_rank_zero": self.completion_count,
            "rollout_finish_reason_counts": dict(sorted(self.finish_reason_counts.items())),
            "rollout_non_stop_completions": int(sum(count for reason, count in self.finish_reason_counts.items() if reason != "stop")),
            "structured_output_mode": str(self.settings.policy.get("rollout_structured_output_mode", "json_schema")),
            "structured_json_schema_field_max_length_enforced": bool(self.settings.policy.get("rollout_json_schema_enforces_field_limit", True)),
            "raw_prompts_or_completions_persisted": False,
        }


def _adapter_hash(model: Any, adapter_name: str) -> str:
    digest = sha256()
    matched = 0
    marker = f".{adapter_name}."
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if marker in name:
            digest.update(name.encode("utf-8")); digest.update(parameter.detach().cpu().contiguous().numpy().tobytes()); matched += 1
    _need(matched > 0, f"PEFT adapter {adapter_name} has no parameters")
    return digest.hexdigest()


def _finite_numeric(values: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in values:
        for key, value in item.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                parsed = float(value)
                _need(math.isfinite(parsed), f"non-finite trainer metric {key}")
                result[str(key)] = parsed
    return result


def _mean_log(history: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in history if isinstance(item.get(key), (int, float)) and math.isfinite(float(item[key]))]
    return round(statistics.fmean(values), 6) if values else None


def _validate_reward_server(settings: RLAIFSettings, run: RLAIFRunConfig) -> dict[str, Any]:
    from urllib.parse import urlparse

    parsed = urlparse(run.reward_endpoint)
    _need(parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port is not None, "reward endpoint must be local HTTP")
    value = json.loads(Path(run.reward_server_attestation).read_text(encoding="utf-8"))
    frozen_runtime = settings.fixed_prompt_template().get("runtime")
    _need(isinstance(frozen_runtime, Mapping) and isinstance(frozen_runtime.get("enforce_eager"), bool), "fixed-v6 eager-mode setting is unavailable")
    expected = {
        "schema_version": "mal2026-rlaif-grpo-reward-server-attestation-v1", "server_host": "127.0.0.1", "server_port": parsed.port,
        "model_id": settings.judge["model_id"], "model_revision": settings.judge["model_revision"],
        "data_parallel_size": 1 if run.phase == "gpu0_actual" else int(settings.runtime["reward_data_parallel_size"]), "max_model_len": settings.runtime["reward_max_model_len"],
        "max_num_seqs_per_rank": settings.runtime["reward_max_num_seqs_per_rank"], "enforce_eager": bool(frozen_runtime["enforce_eager"]),
        "fixed_v6_config_sha256": settings.fixed_v6_config_sha256, "server_process_environment_verified": True,
    }
    # The TP2 layouts keep their one-update gate on the exact GPU3 judge;
    # topology; prior structured versions used GPU1 for that small gate.
    expected_gpus = list(settings.runtime["full_reward_gpus"]) if settings.schema_version.endswith(TP2_SCHEMA_SUFFIXES) else ([1] if run.phase == "gpu0_actual" else list(settings.runtime["full_reward_gpus"]))
    _need(isinstance(value, dict) and all(value.get(key) == expected_value for key, expected_value in expected.items()) and value.get("physical_gpus") == expected_gpus, "reward server attestation differs")
    return value


def run_rlaif_grpo(run: RLAIFRunConfig) -> dict[str, Any]:
    """Continue one SFT adapter under the declared GRPO arm and save one adapter."""
    settings = RLAIFSettings.from_json(); run.validate(settings); attestation = _validate_reward_server(settings, run)
    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError("RLAIF GRPO requires the project .venv-standard") from exc
    output = Path(run.output_dir); output.mkdir(mode=0o700, parents=True, exist_ok=True)
    completion_path = output / "training_complete.json"
    _need(not completion_path.exists(), "RLAIF completion already exists")
    examples, input_provenance = train_examples(settings, run)
    model_id, revision = SUPPORTED_MODELS[run.base_key]
    source_complete = Path(run.source_adapter).parent / "training_complete.json"
    _need(source_complete.is_file(), "source adapter completion record is absent")
    source_record = json.loads(source_complete.read_text(encoding="utf-8"))
    _need(source_record.get("status") == "completed" and source_record.get("base_key") == run.base_key and source_record.get("task") == run.task and source_record.get("model_id") == model_id and source_record.get("model_revision") == revision, "source adapter provenance differs")
    model_path = Path(source_record["config"]["model_path"])
    _need(model_path.is_dir() and not model_path.is_symlink(), "immutable base snapshot is unavailable")
    set_seed(run.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        _need(tokenizer.eos_token is not None, "tokenizer lacks both PAD and EOS")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, run.source_adapter, adapter_name="default", is_trainable=True)
    model.set_adapter("default")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".default." in name)
        if parameter.requires_grad:
            parameter.data = parameter.data.to(dtype=torch.float32)
    _need(any(parameter.requires_grad for _, parameter in model.named_parameters()), "default LoRA adapter is not trainable")
    args = GRPOConfig(
        output_dir=str(output), run_name=run.run_id, seed=run.seed, max_steps=run.max_steps, num_train_epochs=float(settings.policy["num_train_epochs"]),
        learning_rate=float(settings.policy["learning_rate"]), per_device_train_batch_size=run.per_device_train_batch_size,
        gradient_accumulation_steps=int(settings.policy["gradient_accumulation_steps"]), generation_batch_size=run.generation_batch_size,
        num_generations=int(settings.policy["num_generations"]), max_completion_length=int(settings.policy["max_completion_tokens"]),
        temperature=float(settings.policy["sampling_temperature"]), top_p=float(settings.policy["sampling_top_p"]), top_k=0,
        beta=float(settings.policy["beta"]), loss_type=str(settings.policy["loss_type"]), scale_rewards=str(settings.policy["scale_rewards"]),
        num_iterations=1, epsilon=0.2, mask_truncated_completions=False, use_vllm=False, bf16=False, tf32=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, logging_strategy="steps", logging_steps=1,
        save_strategy="no", report_to=[], remove_unused_columns=False, logging_nan_inf_filter=False, disable_dropout=True,
        ddp_find_unused_parameters=False, dataloader_num_workers=0, dataloader_drop_last=False, log_completions=False,
    )
    _need(args.steps_per_generation == run.steps_per_generation, "GRPO derived steps-per-generation differs")
    _need(args.generation_batch_size == run.generation_batch_size and args.generation_batch_size % args.num_generations == 0, "GRPO generation grouping differs")
    reward = QwenPointReward(settings, run)
    rollout: StructuredVLLMRollout | None = None
    if settings.schema_version.endswith(STRUCTURED_SCHEMA_SUFFIXES):
        rollout = StructuredVLLMRollout(settings, run, tokenizer)
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=args, train_dataset=Dataset.from_list(examples), processing_class=tokenizer, rollout_func=rollout)
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    _need("ref" in unwrapped.peft_config, "frozen SFT reference adapter was not created")
    for name, parameter in unwrapped.named_parameters():
        if ".ref." in name:
            parameter.requires_grad_(False)
    _need(not any(parameter.requires_grad for name, parameter in unwrapped.named_parameters() if ".ref." in name), "reference adapter is trainable")
    reference_before = _adapter_hash(unwrapped, "ref")
    try:
        trained = trainer.train()
    except Exception as exc:
        # Preserve an aggregate-only runtime failure even when the trainer
        # cannot reach its normal post-update hard-gate block.
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero:
            _atomic_json(output / "training_failed_runtime.json", {
                "status": "failed_runtime", "run_id": run.run_id, "base_key": run.base_key, "task": run.task,
                "arm": run.arm, "phase": run.phase, "failure_type": type(exc).__name__,
                "global_step": int(trainer.state.global_step), "rlaif_config_sha256": _sha(CONFIG_PATH),
                "fixed_v6_config_sha256": settings.fixed_v6_config_sha256,
                "source_adapter_completion_sha256": _sha(source_complete), "reward_summary": reward.aggregate(),
                "rollout": None if rollout is None else rollout.aggregate(),
                "source_writing_scores_read_or_prompted": False, "candidate_scores_read_or_prompted": False,
                "raw_prompts_or_completions_persisted_outside_ignored_roots": False,
            })
        raise
    trainer.accelerator.wait_for_everyone()
    payload: dict[str, Any] | None = None
    failed = False
    if trainer.is_world_process_zero():
        try:
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            reference_after = _adapter_hash(unwrapped, "ref")
            _need(reference_before == reference_after, "reference SFT adapter changed during GRPO")
            history = [item for item in trainer.state.log_history if isinstance(item, dict)]
            metrics = _finite_numeric(history + [trained.metrics])
            reward_summary = reward.aggregate()
            parse_rate = reward_summary["parse_valid_rate"]
            zero_std = _mean_log(history, "frac_reward_zero_std")
            _need(parse_rate is not None and parse_rate >= float(settings.hard_gates["policy_parse_valid_rate_min"]), "policy parse-validity gate failed")
            _need(zero_std is not None and zero_std < float(settings.hard_gates["reward_zero_std_fraction_max"]), "GRPO reward zero-variance gate failed")
            expected_calls = reward_summary["parse_valid"] * (5 if run.arm == "all5" else 1)
            _need(reward_summary["judge_requests"] == expected_calls, "judge request count differs from reward arm")
            _need(reward_summary["judge_calls"] + reward_summary["judge_unscorable"] == expected_calls, "judge terminal outcome count differs from reward arm")
            if settings.schema_version.endswith("-v8"):
                _need(reward_summary["judge_unscorable_fraction"] is not None and reward_summary["judge_unscorable_fraction"] <= float(settings.reward["max_unscorable_judge_fraction"]), "judge unscorable-rate gate failed")
                _need(reward_summary["discarded_reward_group_fraction"] is not None and reward_summary["discarded_reward_group_fraction"] <= float(settings.reward["max_discarded_reward_group_fraction"]), "discarded reward-group rate gate failed")
            else:
                _need(reward_summary["judge_unscorable"] == 0 and reward_summary["judge_calls"] == expected_calls, "judge call count differs from reward arm")
            adapter = output / "adapter"; _need(not adapter.exists(), "RLAIF adapter output already exists")
            unwrapped.save_pretrained(str(adapter), selected_adapters=["default"], safe_serialization=True)
            tokenizer.save_pretrained(str(adapter))
            _need((adapter / "adapter_config.json").is_file(), "RLAIF adapter was not exported")
            payload = {
                "status": "completed", "run_id": run.run_id, "base_key": run.base_key, "task": run.task, "arm": run.arm, "phase": run.phase,
                "model_id": model_id, "model_revision": revision, "global_step": int(trainer.state.global_step), "policy_train_rows": len(examples),
                "source_adapter_completion_sha256": _sha(source_complete), "source_adapter_path": str(Path(run.source_adapter).resolve()),
                "reference_adapter_sha256_before": reference_before, "reference_adapter_sha256_after": reference_after,
                "rlaif_config_sha256": _sha(CONFIG_PATH), "fixed_v6_config_sha256": settings.fixed_v6_config_sha256,
                "reward_server_attestation_sha256": _sha(Path(run.reward_server_attestation)), "reward_server": {key: attestation[key] for key in ("physical_gpus", "data_parallel_size", "max_model_len", "max_num_seqs_per_rank")},
                "input_provenance": input_provenance, "reward_summary": reward_summary,
                "trainer_metrics": metrics, "trainer_reward_zero_std_fraction_mean": zero_std,
                "rollout": None if rollout is None else rollout.aggregate(),
                "adapter_precision": "float32", "config": asdict(run),
                "source_writing_scores_read_or_prompted": False, "candidate_scores_read_or_prompted": False,
                "raw_prompts_or_completions_persisted_outside_ignored_roots": False,
            }
            _atomic_json(completion_path, payload)
        except Exception as exc:
            failure = {
                "status": "failed_gate", "run_id": run.run_id, "base_key": run.base_key, "task": run.task,
                "arm": run.arm, "phase": run.phase, "failure_type": type(exc).__name__,
                "global_step": int(trainer.state.global_step), "rlaif_config_sha256": _sha(CONFIG_PATH),
                "fixed_v6_config_sha256": settings.fixed_v6_config_sha256,
                "source_adapter_completion_sha256": _sha(source_complete),
                "reward_summary": reward.aggregate(),
                "rollout": None if rollout is None else rollout.aggregate(),
                "source_writing_scores_read_or_prompted": False, "candidate_scores_read_or_prompted": False,
                "raw_prompts_or_completions_persisted_outside_ignored_roots": False,
            }
            _atomic_json(output / "training_failed_gate.json", failure)
            failed = True
    state: list[Any] = [failed, payload]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(state, src=0)
    if state[0]:
        raise RLAIFGRPOError("rank-zero RLAIF completion/health gate failed")
    _need(isinstance(state[1], dict), "RLAIF completion payload was not broadcast")
    return state[1]
