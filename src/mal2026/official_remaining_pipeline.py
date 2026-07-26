"""Artifact resolvers for the remaining public-spec MAL2026 experiment.

The functions in this module do not train models or read writing rows.  They
only turn completed, checksum-bound producer artifacts into the runtime
configs consumed by the already versioned score/rationale runners.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .official_rationale_candidate_evaluation import BINDING_SCHEMA
from .official_rationale_handoff import ALLOWED_HISTORICAL_CONTINUATIONS, AXES


ROOT = Path(__file__).resolve().parents[2]
RL_RUN_ID = "official-rationale-rl-experiment-v1-20260727-001"
RL_ROOT = ROOT / "outputs/official-rationale-rl-v1/orchestration" / RL_RUN_ID
AIHUB_RATIONALE_RUN = (
    ROOT
    / "outputs/official-aihub-rationale-full-sft-v1"
    / "official-aihub-rationale-full-sft-v1-ax4-axis_triplet-full-002"
)
AIHUB_RATIONALE_MODEL = AIHUB_RATIONALE_RUN / "final_model"
AIHUB_RATIONALE_LORA_ROOT = ROOT / "outputs/official-aihub-then-api-rationale-lora-v1"
EMBEDDING_PRETRAIN_ROOT = (
    ROOT
    / "outputs/official-aihub-integer-score-full-pretrain-v1"
    / "official-aihub-integer-score-full-pretrain-v1-20260727-003"
)
RATIONALE_RESTRICTED_ROOT = (
    ROOT
    / "data/processed/restricted/official_prompt_alignment_v1/final_rationale_handoff"
    / "official-rationale-handoff-v1-20260727-001"
)


class RemainingPipelineError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RemainingPipelineError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RemainingPipelineError(f"{label} is unreadable") from exc
    need(isinstance(value, dict), f"{label} must be an object")
    return value


def _completion_from_stage(stage: str, method: str) -> Path:
    report_path = RL_ROOT / "stages" / f"{stage}.json"
    report = read_json(report_path, f"RL stage {stage}")
    need(report.get("status") == "completed" and report.get("stage") == stage, f"RL stage is incomplete: {stage}")
    evidence = report.get("evidence")
    need(isinstance(evidence, dict), f"RL stage evidence differs: {stage}")
    completion = Path(str(evidence.get("completion", "")))
    expected = evidence.get("completion_sha256")
    need(completion.is_file() and isinstance(expected, str) and file_sha256(completion) == expected, f"RL completion binding differs: {stage}")
    value = read_json(completion, f"RL completion {stage}")
    need(value.get("schema_version") == f"mal2026-official-rationale-{method}-complete-v1", f"RL completion method differs: {stage}")
    return completion


def build_candidate_bindings(template: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the fixed three SFT and six historical-method RL candidates."""
    candidates = template.get("candidates")
    need(isinstance(candidates, list) and len(candidates) == 9, "handoff template must declare nine candidates")
    by_key = {candidate.get("key"): candidate for candidate in candidates if isinstance(candidate, dict)}
    need(len(by_key) == 9, "handoff candidate keys differ")

    ax_model = ROOT / "outputs/model-cache/skt--A.X-4.0-Light-ba21c20ea1b31ded1ec3e2fb432335077dc4be98"
    sft_root = ROOT / "outputs/official-rationale-sft-v1"
    fixed: dict[str, dict[str, Any]] = {
        "official_sft_bundle": {
            "model_path": str(ax_model), "model_binding_path": str(ax_model / "config.json"),
            "adapters": {"bundle": str(sft_root / "official-rationale-sft-v1-ax4-bundle-full-001/training_complete.json")},
        },
        "official_sft_axis_triplet": {
            "model_path": str(ax_model), "model_binding_path": str(ax_model / "config.json"),
            "adapters": {
                axis: str(sft_root / f"official-rationale-sft-v1-ax4-{axis}-full-001/training_complete.json")
                for axis in AXES
            },
        },
        "aihub_sft_axis_triplet": {
            "model_path": str(AIHUB_RATIONALE_MODEL),
            "model_binding_path": str(AIHUB_RATIONALE_RUN / "training_complete.json"),
            "adapters": {
                axis: str(
                    AIHUB_RATIONALE_LORA_ROOT
                    / f"official-aihub-then-api-rationale-lora-v1-ax4-axis_triplet-{axis}-full-001/training_complete.json"
                )
                for axis in AXES
            },
        },
    }
    for (method, key), policy in ALLOWED_HISTORICAL_CONTINUATIONS.items():
        legacy = str(policy["legacy_arm"])
        completion = _completion_from_stage(f"{method}-legacy-{legacy}-full", method)
        value = read_json(completion, f"{method} completion {legacy}")
        model = Path(str(value.get("model_path", "")))
        need(model.is_dir() and (model / "config.json").is_file(), f"RL base model differs: {key}")
        fixed[key] = {
            "model_path": str(model.resolve()), "model_binding_path": str((model / "config.json").resolve()),
            "adapters": {"bundle": str(completion.resolve())},
        }
    need(set(fixed) == set(by_key), "candidate binding coverage differs")
    for key, binding in fixed.items():
        binding["evaluation_path"] = by_key[key]["evaluation_path"]
    return {"schema_version": BINDING_SCHEMA, "candidates": fixed}


def resolve_embedding_score_config(
    template: Mapping[str, Any], *, include_rationales: bool,
) -> dict[str, Any]:
    """Bind AI-Hub full states and, for stage B, the final rationale handoff."""
    raw = deepcopy(dict(template))
    for head, prefix in (("bounded_regression", "aihub_bounded"), ("ordinal_cumulative", "aihub_ordinal")):
        completion = EMBEDDING_PRETRAIN_ROOT / f"{head}-refit/training_complete.json"
        value = read_json(completion, f"embedding AI-Hub {head} completion")
        state = value.get("state")
        need(
            value.get("status") == "completed" and value.get("phase") == "refit"
            and value.get("head") == head and isinstance(state, dict),
            f"embedding AI-Hub {head} completion differs",
        )
        artifact = EMBEDDING_PRETRAIN_ROOT / f"{head}-refit/full_model"
        need(artifact.is_dir() and state.get("artifact_path") == str(artifact.resolve()), f"embedding AI-Hub artifact path differs: {head}")
        raw[f"{prefix}_completion_path"] = str(completion.resolve())
        raw[f"{prefix}_completion_sha256"] = file_sha256(completion)
        raw[f"{prefix}_artifact_path"] = str(artifact.resolve())
        raw[f"{prefix}_artifact_sha256"] = state.get("artifact_sha256")

    if include_rationales:
        bootstrap = Path(raw["bootstrap_selection_path"])
        manifest = RATIONALE_RESTRICTED_ROOT / "aggregate_handoff_manifest.json"
        train = RATIONALE_RESTRICTED_ROOT / "rationales.train.jsonl"
        validation = RATIONALE_RESTRICTED_ROOT / "rationales.validation.jsonl"
        handoff = read_json(manifest, "final rationale handoff")
        need(handoff.get("status") == "completed", "final rationale handoff is incomplete")
        raw.update({
            "bootstrap_selection_sha256": file_sha256(bootstrap),
            "rationale_key": handoff.get("rationale_key"),
            "rationale_train_path": str(train.resolve()), "rationale_train_sha256": file_sha256(train),
            "rationale_validation_path": str(validation.resolve()), "rationale_validation_sha256": file_sha256(validation),
            "rationale_manifest_path": str(manifest.resolve()), "rationale_manifest_sha256": file_sha256(manifest),
        })
    return raw


def resolve_decoder_score_config(template: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the selected final rationale; decoder AI-Hub states bind later."""
    raw = deepcopy(dict(template))
    manifest = RATIONALE_RESTRICTED_ROOT / "aggregate_handoff_manifest.json"
    train = RATIONALE_RESTRICTED_ROOT / "rationales.train.jsonl"
    validation = RATIONALE_RESTRICTED_ROOT / "rationales.validation.jsonl"
    handoff = read_json(manifest, "final rationale handoff")
    need(handoff.get("status") == "completed", "final rationale handoff is incomplete")
    raw.update({
        "rationale_key": handoff.get("rationale_key"),
        "rationale_train_path": str(train.resolve()), "rationale_train_sha256": file_sha256(train),
        "rationale_validation_path": str(validation.resolve()), "rationale_validation_sha256": file_sha256(validation),
    })
    return raw
