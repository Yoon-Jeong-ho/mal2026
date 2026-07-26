"""Official-contract rationale-only preference and RL utilities.

This module deliberately does not claim access to the organizer's hidden
prompt.  It continues the public-spec-aligned, score-conditioned rationale
contract and uses the repository's single frozen Q4 proxy judge.  All raw
rows handled here are train-only and must remain below the restricted data
root.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import statistics
from threading import Lock
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .api_rationale_data import ROOT, SOURCE_SHA256, load_writing_rows, sha256_file
from .official_rationale_data import (
    AXES,
    axes_for_task,
    load_candidates,
    messages,
    parse_rationale_output,
    rationale_object,
    rationale_schema,
)
from .official_rationale_sft import MODEL_ID, MODEL_PATH, MODEL_REVISION
from .official_writing_contract import FROZEN_PROXY_JUDGE_SYSTEM_PROMPT, JUDGE_DIMENSIONS, judge_json_schema, judge_messages, parse_judge_output


RESTRICTED_ROOT = (ROOT / "data/processed/restricted").resolve()
OUTPUT_ROOT = (ROOT / "outputs/official-rationale-rl-v1").resolve()
CONTRASTIVE_GATE = ROOT / (
    "outputs/official-prompt-alignment-v1/judge-contrastive/"
    "official-judge-contrastive-train32-001/aggregate_contrastive_gate.json"
)
RL_SAFETY_GATE = ROOT / (
    "outputs/official-prompt-alignment-v1/judge-prompt-injection/"
    "official-judge-prompt-injection-train32-001/aggregate_rl_safety_gate.json"
)
Q4_MODEL_SHA256 = "b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772"
LLAMA_REVISION = "571d0d540df04f25298d0e159e520d9fc62ed121"
LLAMA_TAG = "b10068"
TRL_VERSION = "0.29.1"
VLLM_VERSION = "0.25.1"
JUDGE_PROMPT_SHA256 = sha256(FROZEN_PROXY_JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
TASKS = ("bundle", *AXES)


class OfficialRationaleRLError(RuntimeError):
    """Raised when an official rationale RL invariant is violated."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialRationaleRLError(message)


def file_sha(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"required file is unavailable: {path}")
    return sha256_file(path)


def restricted_fresh(path: Path) -> Path:
    resolved = path.resolve()
    need(resolved.is_relative_to(RESTRICTED_ROOT), "row-level RL artifacts must remain restricted")
    need(not resolved.exists(), "RL row-level output must be fresh")
    return resolved


def output_fresh(path: Path) -> Path:
    resolved = path.resolve()
    need(resolved.is_relative_to(OUTPUT_ROOT), "RL aggregate/model output root differs")
    need(not resolved.exists(), "RL output must be fresh")
    return resolved


def local_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    need(parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port is not None, "endpoint must be local HTTP")
    return endpoint.rstrip("/")


def validate_runtime_versions() -> dict[str, str]:
    observed = {"trl": version("trl"), "vllm": version("vllm")}
    need(observed == {"trl": TRL_VERSION, "vllm": VLLM_VERSION}, "installed TRL/vLLM versions differ")
    return observed


def assert_contrastive_gate(path: Path = CONTRASTIVE_GATE) -> dict[str, Any]:
    """Require and bind the completed directional gate before any RL action."""
    digest = file_sha(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), "contrastive gate is not an object")
    need(value.get("schema_version") == "mal2026-official-proxy-judge-contrastive-gate-v1", "contrastive gate schema differs")
    need(value.get("status") == "passed" and value.get("rl_with_this_proxy_judge_allowed") is True, "contrastive gate did not authorize RL")
    thresholds = value.get("thresholds_frozen_before_results")
    need(thresholds == {"minimum_target_mean_decrease": 0.25, "minimum_paired_decrease_rate": 0.5}, "contrastive thresholds differ")
    tests = value.get("tests")
    need(isinstance(tests, dict) and len(tests) == 3 and all(isinstance(item, dict) and item.get("passed") is True for item in tests.values()), "contrastive directional tests did not all pass")
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "status": value["status"],
        "rl_with_this_proxy_judge_allowed": True,
    }


def assert_rl_safety_gate(path: Path = RL_SAFETY_GATE) -> dict[str, Any]:
    """Require the combined directional and prompt-injection safety gate."""
    digest = file_sha(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), "RL safety gate is not an object")
    need(value.get("schema_version") == "mal2026-official-proxy-judge-rl-safety-gate-v1", "RL safety gate schema differs")
    need(value.get("status") == "passed" and value.get("rl_allowed") is True, "RL safety gate did not authorize RL")
    need(value.get("directional_contrastive_gate_passed") is True, "RL safety gate lacks directional passage")
    need(value.get("prompt_injection_gate_passed") is True, "RL safety gate lacks prompt-injection passage")
    need(value.get("failure_policy") == "preserve all artifacts and exit 2; do not run RL with this proxy judge", "RL safety failure policy differs")
    return {"path": str(path.resolve()), "sha256": digest, "status": "passed", "rl_allowed": True}


def validate_q4_attestation(path: Path, endpoints: Sequence[str]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = [local_endpoint(item) for item in endpoints]
    need(value.get("schema_version") == "mal2026-official-q4-judge-server-attestation-v1", "Q4 attestation schema differs")
    need(value.get("model_sha256") == Q4_MODEL_SHA256, "Q4 model digest differs")
    need(value.get("llama_revision") == LLAMA_REVISION and value.get("llama_tag") == LLAMA_TAG, "llama.cpp provenance differs")
    need(value.get("server_endpoints") == expected, "Q4 endpoints differ from attestation")
    return value


def validate_policy_attestation(
    path: Path,
    endpoint: str,
    aliases: Mapping[str, str],
    *,
    expected_model_id: str = MODEL_ID,
    expected_model_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_endpoint = local_endpoint(endpoint)
    need(value.get("schema_version") == "mal2026-official-rl-policy-server-attestation-v1", "policy attestation schema differs")
    need(value.get("endpoint") == expected_endpoint, "policy endpoint differs from attestation")
    need(value.get("vllm_version") == VLLM_VERSION, "policy vLLM version differs")
    need(value.get("model_id") == expected_model_id and value.get("model_revision") == expected_model_revision, "policy model provenance differs")
    attested_aliases = value.get("adapter_aliases")
    need(isinstance(attested_aliases, dict) and all(attested_aliases.get(task) == alias for task, alias in aliases.items()), "policy adapter aliases differ")
    need(value.get("train_split_only") is True and value.get("dynamic_lora") is True, "policy server contract differs")
    return value


def judge_total(output: Mapping[str, Any]) -> int:
    """Return the exact integer sum of all 12 frozen-judge cells."""
    parsed = parse_judge_output(output)
    total = sum(int(parsed[axis][dimension]["score"]) for axis in AXES for dimension in JUDGE_DIMENSIONS)
    need(12 <= total <= 60, "judge total differs")
    return total


def axis_judge_total(output: Mapping[str, Any], axis: str) -> int:
    need(axis in AXES, "judge axis differs")
    parsed = parse_judge_output(output)
    total = sum(int(parsed[axis][dimension]["score"]) for dimension in JUDGE_DIMENSIONS)
    need(4 <= total <= 20, "axis judge total differs")
    return total


def select_preference(scored: Sequence[tuple[str, Mapping[str, Any]]], minimum_total_difference: int = 1) -> dict[str, Any] | None:
    """Select max/min by 12-cell total; exclude exact ties.

    A margin of one is not an arbitrary floating threshold: judge cells are
    integer-valued, so one is the smallest observable non-tie in their sum.
    """
    need(len(scored) >= 2 and minimum_total_difference == 1, "preference selection contract differs")
    ranked = sorted(((judge_total(judge), index, text) for index, (text, judge) in enumerate(scored)), key=lambda item: (item[0], -item[1]))
    low, high = ranked[0], ranked[-1]
    difference = high[0] - low[0]
    if difference < minimum_total_difference:
        return None
    return {
        "chosen": high[2],
        "rejected": low[2],
        "chosen_judge_total": high[0],
        "rejected_judge_total": low[0],
        "judge_total_difference": difference,
    }


def select_axis_preference(scored: Sequence[tuple[str, Mapping[str, Any]]], axis: str, minimum_total_difference: int = 1) -> dict[str, Any] | None:
    """Select an axis pair by that axis's four integer judge cells only."""
    need(axis in AXES and len(scored) >= 2 and minimum_total_difference == 1, "axis preference selection contract differs")
    ranked = sorted(((axis_judge_total(judge, axis), index, text) for index, (text, judge) in enumerate(scored)), key=lambda item: (item[0], -item[1]))
    low, high = ranked[0], ranked[-1]
    difference = high[0] - low[0]
    if difference < minimum_total_difference:
        return None
    return {
        "chosen": high[2],
        "rejected": low[2],
        "chosen_axis_judge_total": high[0],
        "rejected_axis_judge_total": low[0],
        "axis_judge_total_difference": difference,
        "axis": axis,
    }


def completion_text(rationales: Mapping[str, str], task: str) -> str:
    axes = axes_for_task(task)
    return json.dumps(rationale_object(rationales, axes), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def parse_completion(text: Any, task: str, character_limit: int) -> dict[str, str] | None:
    if not isinstance(text, str):
        return None
    try:
        parsed = parse_rationale_output(text, axes_for_task(task))
    except Exception:
        return None
    if any(len(value) > character_limit or not any("가" <= char <= "힣" for char in value) for value in parsed.values()):
        return None
    return parsed


def participant(scores: Mapping[str, int], generated: Mapping[str, str], frozen: Mapping[str, str] | None = None) -> dict[str, Any]:
    need(set(scores) == set(AXES) and all(type(scores[axis]) is int and 1 <= scores[axis] <= 5 for axis in AXES), "frozen emitted integer scores differ")
    rationales = dict(frozen or {})
    rationales.update(generated)
    need(set(rationales) == set(AXES) and all(isinstance(rationales[axis], str) and rationales[axis].strip() for axis in AXES), "participant rationale axes differ")
    return {axis: {"score": scores[axis], "rationale": rationales[axis].strip()} for axis in AXES}


def official_train_rows(task: str, limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build score-conditioned train rows without reading a validation split."""
    axes_for_task(task)
    writings = {row.identifier: row for row in load_writing_rows("train", include_scores=False)}
    candidates = load_candidates()
    if limit is not None:
        need(0 < limit <= len(candidates), "RL train limit differs")
        candidates = candidates[:limit]
    rows: list[dict[str, Any]] = []
    for item in candidates:
        writing = writings[item.source_id]
        source_key = sha256(f"{item.source_id}:{item.candidate_number}".encode()).hexdigest()
        rows.append({
            "prompt": messages(writing.prompt, writing.essay, item.scores, axes_for_task(task)),
            "source_key": source_key,
            "source_id": item.source_id,
            "candidate_number": item.candidate_number,
            "prompt_text": writing.prompt,
            "essay_text": writing.essay,
            "scores": dict(item.scores),
            "frozen_rationales": dict(item.rationales),
            "split": "train",
        })
    return rows, {
        "split": "train",
        "source_sha256": SOURCE_SHA256["train"],
        "rows": len(rows),
        "human_or_reference_score_read_or_prompted": False,
        "score_kind": "frozen_api_emitted_integer_prediction",
    }


@dataclass(frozen=True)
class RLSettings:
    schema_version: str
    run_id_prefix: str
    algorithm: str
    gate: Mapping[str, Any]
    judge: Mapping[str, Any]
    policy: Mapping[str, Any]
    reward: Mapping[str, Any]
    warm_starts: Mapping[str, str]
    runtime: Mapping[str, Any]
    legacy_ablations: Sequence[Mapping[str, Any]]
    privacy: Mapping[str, Any]

    @classmethod
    def from_json(cls, path: Path) -> "RLSettings":
        raw = json.loads(path.read_text(encoding="utf-8"))
        need(isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__), "RL settings fields differ")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self) -> None:
        expected_schema = f"mal2026-official-rationale-{self.algorithm}-v1"
        need(self.algorithm in {"dpo", "grpo"} and self.schema_version == expected_schema, "RL algorithm/schema differs")
        need(self.run_id_prefix == f"official-rationale-{self.algorithm}-v1-", "RL run prefix differs")
        need(self.gate == {
            "path": str(CONTRASTIVE_GATE.relative_to(ROOT)),
            "safety_path": str(RL_SAFETY_GATE.relative_to(ROOT)),
            "required_schema_version": "mal2026-official-proxy-judge-contrastive-gate-v1",
            "required_status": "passed",
            "required_rl_allowed": True,
            "required_safety_schema_version": "mal2026-official-proxy-judge-rl-safety-gate-v1",
            "required_safety_status": "passed",
            "required_combined_rl_allowed": True,
            "bind_sha256_in_every_artifact": True,
        }, "RL contrastive-gate declaration differs")
        need(self.judge.get("model_sha256") == Q4_MODEL_SHA256 and self.judge.get("llama_revision") == LLAMA_REVISION and self.judge.get("llama_tag") == LLAMA_TAG, "RL judge pin differs")
        need(self.judge.get("prompt_kind") == "single_frozen_public-spec-aligned_proxy_not_verbatim_organizer_prompt", "RL judge prompt provenance differs")
        need(self.judge.get("prompt_sha256") == JUDGE_PROMPT_SHA256, "RL judge prompt digest differs")
        need(self.judge.get("score_projection") == "bundle_12_cell_sum_axis_4_cell_sum", "RL judge projection differs")
        need(set(self.warm_starts) == set(TASKS), "RL warm-start tasks differ")
        for task, raw_path in self.warm_starts.items():
            path = Path(raw_path)
            need(path.is_absolute() and path.is_dir() and not path.is_symlink() and (path / "adapter_config.json").is_file(), f"warm-start adapter unavailable: {task}")
        need(self.reward.get("invalid_completion_reward") == -1.0, "invalid completion reward differs")
        need(self.reward.get("parse_valid_rate_min") == 0.98, "parse-validity gate differs")
        need(self.reward.get("preference_minimum_total_difference") == 1, "preference margin differs")
        need(self.reward.get("preference_margin_basis") == "smallest_observable_non_tie_in_integer_cell_sum_bundle12_axis4", "preference margin basis differs")
        need(self.reward.get("tie_policy") == "exclude", "preference tie policy differs")
        need(self.reward.get("max_zero_variance_group_fraction") == 0.8, "reward variance gate differs")
        need(self.reward.get("variance_gate_basis") == "frozen_before_run_from_previously_executed_v8_operational_gate_not_a_quality_claim", "reward variance gate basis differs")
        need(self.runtime.get("gpu_scope") == [0, 1, 2, 3] and self.runtime.get("vllm_version") == VLLM_VERSION and self.runtime.get("trl_version") == TRL_VERSION, "RL runtime versions/scope differ")
        need(self.runtime.get("integrated_vllm") is False and self.runtime.get("external_vllm_http") is True, "TRL/vLLM integration boundary differs")
        legacy_names = [item.get("name") for item in self.legacy_ablations]
        need(legacy_names == ["midm_bundle_random1_v8_replication", "ax4_bundle_random1_v8_replication", "ax4_bundle_all5_v8_replication"], "legacy top-three method list differs")
        need(all(item.get("classification") == "legacy_method_replication_ablation_not_direct_official_arm" for item in self.legacy_ablations), "legacy arms are mislabeled")
        required_legacy_fields = {"name", "classification", "contract_shift", "model_id", "model_revision", "model_path", "adapter_path", "adapter_model_sha256", "completion_path", "completion_sha256"}
        for item in self.legacy_ablations:
            need(set(item) == required_legacy_fields, "legacy ablation provenance fields differ")
            need(item.get("contract_shift") == "score_blind_legacy_warmstart_to_score_conditioned_official_prompt_descriptive_only", "legacy contract-shift label differs")
            for key in ("model_path", "adapter_path"):
                candidate = Path(str(item[key]))
                need(candidate.is_absolute() and candidate.is_dir() and not candidate.is_symlink(), f"legacy {key} is unavailable")
            adapter_digest = item.get("adapter_model_sha256")
            need(isinstance(adapter_digest, str) and len(adapter_digest) == 64 and all(character in "0123456789abcdef" for character in adapter_digest), "legacy adapter model digest declaration differs")
            completion = Path(str(item["completion_path"]))
            need(completion.is_absolute() and file_sha(completion) == item["completion_sha256"], "legacy completion digest differs")
            record = json.loads(completion.read_text(encoding="utf-8"))
            need(record.get("status") == "completed" and record.get("model_id") == item["model_id"] and record.get("model_revision") == item["model_revision"], "legacy completion provenance differs")
        need(self.privacy == {
            "split": "train_only",
            "validation_used_for_preferences_or_reward": False,
            "human_or_reference_score_read_or_prompted": False,
            "row_artifacts": "restricted_only",
            "aggregate_artifacts": "outputs_only",
        }, "RL privacy/split contract differs")
        if self.algorithm == "dpo":
            need(self.policy.get("trainer") == "trl.DPOTrainer" and self.policy.get("offline_preferences") is True, "DPO trainer contract differs")
            need(self.policy.get("loss_type") == "sigmoid" and self.policy.get("beta") == 0.1, "DPO loss contract differs")
            need((self.runtime.get("generation_tensor_parallel_size"), self.runtime.get("generation_max_model_len"), self.runtime.get("generation_max_num_seqs"), self.runtime.get("generation_max_num_batched_tokens")) == (4, 4096, 256, 32768), "DPO generation topology/batching differs")
        else:
            need(self.policy.get("trainer") == "trl.GRPOTrainer" and self.policy.get("rollout_backend") == "external_vllm_http_rollout_func", "GRPO rollout contract differs")
            need(self.policy.get("use_vllm") is False and self.policy.get("num_generations") == 4, "GRPO integrated-vLLM/group contract differs")
            need(self.policy.get("loss_type") == "dr_grpo" and self.policy.get("scale_rewards") == "none", "GRPO objective differs")
            need((self.policy.get("max_steps"), self.policy.get("full_train_limit"), self.policy.get("pilot_max_steps"), self.policy.get("pilot_train_limit")) == (480, 1920, 80, 320), "GRPO full/pilot bounds differ")
            need(self.runtime.get("rollout_gpus") == [0, 1] and self.runtime.get("policy_gpu") == [2] and self.runtime.get("reward_gpu") == [3], "GRPO GPU partition differs")
            need((self.runtime.get("rollout_tensor_parallel_size"), self.runtime.get("rollout_max_model_len"), self.runtime.get("rollout_max_num_seqs"), self.runtime.get("rollout_max_num_batched_tokens"), self.runtime.get("gpu_memory_utilization")) == (2, 4096, 192, 65536, 0.9), "GRPO rollout topology/batching differs")

    def gate_evidence(self) -> dict[str, Any]:
        return {
            "directional": assert_contrastive_gate(ROOT / str(self.gate["path"])),
            "combined_safety": assert_rl_safety_gate(ROOT / str(self.gate["safety_path"])),
        }


def load_preferences(path: Path, task: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    need(task in TASKS, "preference task differs")
    resolved = path.resolve()
    need(resolved.is_relative_to(RESTRICTED_ROOT) and resolved.is_file() and not resolved.is_symlink(), "preference file must be restricted")
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            need(bool(line.strip()), f"blank preference row at {line_number}")
            raw = json.loads(line)
            need(raw.get("schema_version") == "mal2026-official-rationale-preference-v1", "preference schema differs")
            need(raw.get("split") == "train" and raw.get("task") == task, "preference split/task differs")
            need(raw.get("score_kind") == "frozen_api_emitted_integer_prediction", "preference score kind differs")
            if task == "bundle":
                need(raw.get("selection_projection") == "sum_of_all_12_integer_cells", "bundle preference projection differs")
                need(type(raw.get("chosen_judge_total")) is int and type(raw.get("rejected_judge_total")) is int, "bundle preference totals differ")
                need(type(raw.get("judge_total_difference")) is int and raw["judge_total_difference"] >= 1, "bundle preference margin differs")
                need(raw["chosen_judge_total"] - raw["rejected_judge_total"] == raw["judge_total_difference"], "bundle preference total arithmetic differs")
            else:
                need(raw.get("selection_projection") == f"sum_of_4_integer_cells_for_{task}", "axis preference projection differs")
                need(type(raw.get("chosen_axis_judge_total")) is int and type(raw.get("rejected_axis_judge_total")) is int, "axis preference totals differ")
                need(type(raw.get("axis_judge_total_difference")) is int and raw["axis_judge_total_difference"] >= 1, "axis preference margin differs")
                need(raw["chosen_axis_judge_total"] - raw["rejected_axis_judge_total"] == raw["axis_judge_total_difference"], "axis preference total arithmetic differs")
            prompt, chosen, rejected = raw.get("prompt"), raw.get("chosen"), raw.get("rejected")
            need(isinstance(prompt, list) and isinstance(chosen, list) and isinstance(rejected, list), "preference conversational fields differ")
            need(chosen != rejected, "preference pair is tied")
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    need(bool(rows), "preference dataset is empty")
    return rows, {"path": str(resolved), "sha256": file_sha(resolved), "rows": len(rows), "split": "train"}


def validate_preference_report(path: Path, preference_path: Path, task: str, settings: RLSettings, gate: Mapping[str, Any]) -> dict[str, Any]:
    need(path.resolve().is_relative_to(OUTPUT_ROOT) and path.is_file() and not path.is_symlink(), "preference aggregate must be an outputs artifact")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_arm = "bundle" if task == "bundle" else "axis_triplet"
    need(value.get("schema_version") == "mal2026-official-rationale-preference-aggregate-v1", "preference aggregate schema differs")
    need(value.get("status") == "completed" and value.get("stage") == "assemble" and value.get("arm") == expected_arm, "preference aggregate status/arm differs")
    need(value.get("split") == "train" and value.get("validation_used") is False, "preference aggregate split differs")
    need(value.get("raw_sha256") == file_sha(preference_path), "preference file digest differs from aggregate")
    need(value.get("contrastive_gate_sha256") == gate["directional"]["sha256"], "preference directional gate binding differs")
    need(value.get("rl_safety_gate_sha256") == gate["combined_safety"]["sha256"], "preference safety gate binding differs")
    need(value.get("judge_model_sha256") == Q4_MODEL_SHA256, "preference judge model binding differs")
    need(value.get("judge_prompt_sha256") == JUDGE_PROMPT_SHA256, "preference judge prompt binding differs")
    need(value.get("reward_variance_gate_passed") is True, "preference reward-variance gate failed")
    per_task = value.get("per_task_reward_variance")
    need(isinstance(per_task, dict) and task in per_task, "preference per-task variance report differs")
    expected_tasks = {"bundle"} if expected_arm == "bundle" else set(AXES)
    need(set(per_task) == expected_tasks, "preference variance task population differs")
    for reported_task, diagnostic in per_task.items():
        need(isinstance(diagnostic, dict) and diagnostic.get("passed") is True, f"preference variance gate failed for {reported_task}")
        fraction = diagnostic.get("zero_variance_group_fraction")
        need(isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and float(fraction) <= float(settings.reward["max_zero_variance_group_fraction"]), f"preference zero-variance fraction differs for {reported_task}")
    return {"path": str(path.resolve()), "sha256": file_sha(path), "status": "completed", "reward_variance_gate_passed": True}


def legacy_ablation(settings: RLSettings, name: str) -> Mapping[str, Any]:
    matches = [item for item in settings.legacy_ablations if item.get("name") == name]
    need(len(matches) == 1, "legacy ablation name differs")
    return matches[0]


def legacy_grpo_producer_spec(settings: RLSettings, name: str) -> dict[str, Any]:
    """Validate one pinned legacy checkpoint as a GRPO warm-start producer.

    This is deliberately a static, CPU-only compatibility gate.  A real
    one-update smoke remains mandatory before a pilot or full producer may be
    considered eligible for downstream rationale generation.
    """
    item = legacy_ablation(settings, name)
    model_path = Path(str(item["model_path"]))
    adapter_path = Path(str(item["adapter_path"]))
    completion_path = Path(str(item["completion_path"]))
    model_config_path = model_path / "config.json"
    adapter_config_path = adapter_path / "adapter_config.json"
    need(model_config_path.is_file() and not model_config_path.is_symlink(), "legacy GRPO model config is unavailable")
    need(adapter_config_path.is_file() and not adapter_config_path.is_symlink(), "legacy GRPO adapter config is unavailable")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    architectures = model_config.get("architectures")
    need(
        isinstance(architectures, list)
        and len(architectures) == 1
        and architectures[0] in {"LlamaForCausalLM", "Qwen2ForCausalLM"},
        "legacy GRPO base architecture is unsupported by the frozen producer",
    )
    need(adapter_config.get("peft_type") == "LORA" and adapter_config.get("task_type") == "CAUSAL_LM", "legacy GRPO adapter type differs")
    rank = adapter_config.get("r")
    need(type(rank) is int and 0 < rank <= 32, "legacy GRPO LoRA rank exceeds the vLLM producer contract")
    configured_base = Path(str(adapter_config.get("base_model_name_or_path", ""))).resolve()
    need(configured_base == model_path.resolve(), "legacy GRPO adapter base-model linkage differs")
    need(file_sha(adapter_path / "adapter_model.safetensors") == item["adapter_model_sha256"], "legacy GRPO adapter model digest differs")
    need(file_sha(completion_path) == item["completion_sha256"], "legacy GRPO source completion digest differs")
    return {
        "legacy_arm": name,
        "classification": item["classification"],
        "contract_shift": item["contract_shift"],
        "model_id": item["model_id"],
        "model_revision": item["model_revision"],
        "model_path": str(model_path.resolve()),
        "model_architecture": architectures[0],
        "model_config_sha256": file_sha(model_config_path),
        "warm_start_adapter": str(adapter_path.resolve()),
        "warm_start_adapter_config_sha256": file_sha(adapter_config_path),
        "warm_start_adapter_model_sha256": item["adapter_model_sha256"],
        "legacy_completion_path": str(completion_path.resolve()),
        "legacy_completion_sha256": item["completion_sha256"],
        "static_compatibility_status": "supported_pending_real_one_update_smoke",
    }


def http_json(endpoint: str, body: Mapping[str, Any], *, timeout: int = 600, attempts: int = 2) -> dict[str, Any]:
    wire = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    error: Exception | None = None
    for _ in range(attempts):
        try:
            request = Request(local_endpoint(endpoint) + "/v1/chat/completions", data=wire, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            need(isinstance(value, dict), "HTTP response is not an object")
            return value
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OfficialRationaleRLError) as exc:
            error = exc
    raise OfficialRationaleRLError("local model HTTP request failed") from error


def q4_score(endpoint: str, model: str, prompt_text: str, essay_text: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "model": model,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": 1800,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": judge_messages(prompt_text, essay_text, candidate),
        "response_format": {"type": "json_object", "schema": judge_json_schema()},
    }
    outer = http_json(endpoint, body)
    try:
        choice = outer["choices"][0]
        need(choice.get("finish_reason") == "stop", "Q4 judge finish reason differs")
        return parse_judge_output(choice["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise OfficialRationaleRLError("Q4 judge response envelope differs") from exc


class ExactQ4Reward:
    """TRL reward callable backed only by the pinned local Q4 judge."""

    __name__ = "official_exact_q4_rationale_reward"

    def __init__(self, settings: RLSettings, task: str, endpoints: Sequence[str], model_alias: str):
        self.settings = settings
        self.task = task
        self.endpoints = tuple(local_endpoint(item) for item in endpoints)
        self.model_alias = model_alias
        self.counts: Counter[str] = Counter()
        self.reward_values: list[float] = []
        self.lock = Lock()

    def __call__(self, prompts: list[Any], completions: list[Any], prompt_text: list[str], essay_text: list[str], scores: list[Mapping[str, int]], frozen_rationales: list[Mapping[str, str]], **_: Any) -> list[float]:
        need(len(prompts) == len(completions) == len(prompt_text) == len(essay_text) == len(scores) == len(frozen_rationales), "reward columns differ")
        values: list[float] = []
        for index, completion in enumerate(completions):
            raw = completion
            if isinstance(completion, list) and len(completion) == 1 and isinstance(completion[0], dict):
                raw = completion[0].get("content")
            parsed = parse_completion(raw, self.task, int(self.settings.reward["field_character_limit"]))
            if parsed is None:
                values.append(float(self.settings.reward["invalid_completion_reward"]))
                self.counts["parse_invalid"] += 1
                continue
            candidate = participant(scores[index], parsed, frozen_rationales[index] if self.task != "bundle" else None)
            judged = q4_score(self.endpoints[index % len(self.endpoints)], self.model_alias, prompt_text[index], essay_text[index], candidate)
            if self.task == "bundle":
                reward = judge_total(judged) / 12.0
            else:
                reward = statistics.fmean(int(judged[self.task][dimension]["score"]) for dimension in JUDGE_DIMENSIONS)
            values.append(float(reward))
            self.counts["parse_valid"] += 1
            self.counts["judge_calls"] += 1
        with self.lock:
            self.counts["completions"] += len(values)
            self.reward_values.extend(values)
        return values

    def aggregate(self) -> dict[str, Any]:
        total = self.counts["completions"]
        return {
            "completions": total,
            "parse_valid": self.counts["parse_valid"],
            "parse_invalid": self.counts["parse_invalid"],
            "parse_valid_rate": self.counts["parse_valid"] / total if total else None,
            "judge_calls": self.counts["judge_calls"],
            "reward_mean": statistics.fmean(self.reward_values) if self.reward_values else None,
            "reward_std": statistics.pstdev(self.reward_values) if len(self.reward_values) > 1 else 0.0 if self.reward_values else None,
            "projection": "all_12_cell_mean" if self.task == "bundle" else f"{self.task}_4_cell_mean_from_full_12_cell_judgment",
        }


class ExternalVLLMRollout:
    """TRL 0.29.1 experimental rollout_func using external vLLM 0.25.1.

    This is the same repository-proven integration boundary used by the v8
    legacy experiment: TRL owns policy/ref log probabilities and optimization;
    the external server supplies sampled token IDs.  Integrated TRL vLLM is
    intentionally disabled.
    """

    def __init__(self, settings: RLSettings, task: str, endpoint: str, alias: str, tokenizer: Any, sync_root: Path):
        self.settings, self.task, self.endpoint, self.alias, self.tokenizer = settings, task, local_endpoint(endpoint), alias, tokenizer
        self.sync_root = sync_root.resolve()
        self.last_step: int | None = None
        self.active_snapshot: Path | None = None
        self.counts: Counter[str] = Counter()

    def _request(self, prompt: list[dict[str, str]], count: int, seed: int) -> list[str]:
        body = {
            "model": self.alias,
            "messages": prompt,
            "n": count,
            "temperature": float(self.settings.policy["sampling_temperature"]),
            "top_p": float(self.settings.policy["sampling_top_p"]),
            "seed": seed,
            "max_tokens": int(self.settings.policy["max_completion_tokens"]),
            "response_format": {"type": "json_schema", "json_schema": {"name": f"official_rl_{self.task}", "strict": True, "schema": rationale_schema(axes_for_task(self.task))}},
        }
        outer = http_json(self.endpoint, body)
        choices = outer.get("choices")
        need(isinstance(choices, list) and len(choices) == count, "rollout response count differs")
        result: list[str] = []
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            need(isinstance(content, str), "rollout completion differs")
            self.counts[f"finish_{choice.get('finish_reason')}"] += 1
            result.append(content)
        self.counts["requests"] += 1
        self.counts["completions"] += len(result)
        return result

    def _sync(self, trainer: Any, step: int) -> None:
        accelerator = trainer.accelerator
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.sync_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            snapshot = self.sync_root / f"step-{step:06d}"
            need(not snapshot.exists(), "rollout adapter snapshot would be overwritten")
            accelerator.unwrap_model(trainer.model).save_pretrained(str(snapshot), selected_adapters=["default"], safe_serialization=True)
            need((snapshot / "adapter_config.json").is_file(), "rollout adapter snapshot is incomplete")
            wire = json.dumps({"lora_name": self.alias, "lora_path": str(snapshot), "load_inplace": True}, separators=(",", ":")).encode()
            request = Request(self.endpoint + "/v1/load_lora_adapter", data=wire, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urlopen(request, timeout=180) as response:
                    need(200 <= response.status < 300, "rollout adapter reload failed")
                    response.read()
            except (HTTPError, URLError, TimeoutError) as exc:
                raise OfficialRationaleRLError("rollout adapter reload transport failure") from exc
            if self.active_snapshot is not None and self.active_snapshot.exists():
                shutil.rmtree(self.active_snapshot)
            self.active_snapshot = snapshot
            self.counts["syncs"] += 1
        accelerator.wait_for_everyone()

    def __call__(self, prompts: list[Any], trainer: Any) -> dict[str, Any]:
        step = int(trainer.state.global_step)
        if self.last_step != step:
            self._sync(trainer, step)
            self.last_step = step
        group_size = int(self.settings.policy["num_generations"])
        prompt_ids: list[list[int]] = []
        completion_ids: list[list[int]] = []
        index = 0
        while index < len(prompts):
            group = prompts[index:index + group_size]
            need(len(group) == group_size and all(item == group[0] for item in group), "GRPO repeated prompt group differs")
            rendered = self.tokenizer.apply_chat_template(group[0], tokenize=True, add_generation_prompt=True)
            if hasattr(rendered, "ids"):
                rendered = list(rendered.ids)
            elif hasattr(rendered, "tolist"):
                rendered = rendered.tolist()
                if rendered and isinstance(rendered[0], list):
                    rendered = rendered[0]
            need(isinstance(rendered, list) and all(type(token) is int for token in rendered), "rollout prompt tokenization differs")
            digest = sha256(json.dumps(group[0], ensure_ascii=False, sort_keys=True).encode()).digest()
            seed = (int.from_bytes(digest[:8], "big") ^ int(self.settings.policy["seed"]) ^ step ^ index) % (2**31 - 1)
            texts = self._request(group[0], group_size, seed)
            eos = self.tokenizer.eos_token_id
            need(type(eos) is int, "tokenizer EOS differs")
            for text in texts:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                need(isinstance(ids, list) and bool(ids), "rollout completion tokenization differs")
                prompt_ids.append(rendered)
                completion_ids.append([*ids, eos])
            index += group_size
        need(len(prompt_ids) == len(prompts), "rollout output population differs")
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": None}

    def aggregate(self) -> dict[str, Any]:
        return {"backend": "external_vllm_http_rollout_func", **dict(sorted(self.counts.items())), "raw_prompts_or_completions_persisted": False}


def finite_metrics(values: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in values:
        for key, value in item.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                parsed = float(value)
                need(math.isfinite(parsed), f"non-finite metric: {key}")
                result[str(key)] = parsed
    return result


def mean_history(history: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in history if isinstance(item.get(key), (int, float)) and math.isfinite(float(item[key]))]
    return statistics.fmean(values) if values else None
