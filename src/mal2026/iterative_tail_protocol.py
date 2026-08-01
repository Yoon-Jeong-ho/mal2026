"""Immutable, leakage-safe contract for iterative R0 tail refinement.

This module is deliberately orchestration-only: it validates the predeclared
train OOF inputs and produces aggregate task-card/report metadata.  It does
not train a model, access validation, use a GPU, or call an external API.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

from mal2026.r0_ordinal_residual import ResidualRow, load_embedding_artifact

SCHEMA_VERSION = "mal2026-iterative-tail-refinement-v1"
RUN_ID = "iterative-tail-refinement-v1-20260801-001"
CONFIG_PATH = Path("configs/iterative_tail_refinement.v1.json")
EXPECTED_BINDINGS = {
    "canonical_train_path": "eval/train.jsonl",
    "canonical_train_sha256": "b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737",
    "r0_oof_prediction_path": "data/processed/restricted/r0_exact_oof_v1/r0-exact-oof-20260731-002/merged/oof_predictions.jsonl",
    "r0_oof_prediction_sha256": "823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04",
    "embedding_manifest_path": "data/processed/restricted/r0_ordinal_residual_embeddings_v1/r0-public-frozen-embedding-20260731-001/train/merged/manifest.json",
    "embedding_provenance_path": "data/processed/restricted/r0_ordinal_residual_embeddings_v1/r0-public-frozen-embedding-20260731-001/train/merged/provenance.json",
    "embedding_rows_path": "data/processed/restricted/r0_ordinal_residual_embeddings_v1/r0-public-frozen-embedding-20260731-001/train/merged/rows.jsonl",
    "embedding_rows_sha256": "949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16",
    "fold_assignment_fingerprint": "8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db",
    "embedding_model_id": "Qwen/Qwen3-Embedding-8B",
    "embedding_model_revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
    "score_blind_rationale_train_path": "data/processed/restricted/evaluation_prompt_rationale_v2/evaluation-prompt-rationale-generation-v2-score-blind-20260729-004/rationales.train.jsonl",
    "score_blind_rationale_train_sha256": "d4a2be9a070c786728fde6f64f066ac9d462bc5f83305a2d9161b380abd88e55",
    "score_blind_rationale_manifest_path": "data/processed/restricted/evaluation_prompt_rationale_v2/evaluation-prompt-rationale-generation-v2-score-blind-20260729-004/aggregate_handoff_manifest.json",
    "score_blind_rationale_manifest_sha256": "9b5ca18dd9a993cf8cb7d24fcc0c19a8d3ceff05f17abafe1a58e8841df0a41f",
    "score_blind_agent_a_train_path": "data/processed/restricted/evaluation_prompt_rationale_v1/evaluation-prompt-rationale-generation-v1-score-blind-20260729-001/rationales.train.jsonl",
    "score_blind_agent_a_train_sha256": "1a10524f79823e097e6f56f8c7ac3a499baf2d79b81cbb5b69184ddc88610223",
    "score_blind_agent_b_train_path": "data/processed/restricted/evaluation_prompt_rationale_v1/evaluation-prompt-rationale-generation-v1-score-blind-20260729-002/rationales.train.jsonl",
    "score_blind_agent_b_train_sha256": "f5cce4058f56ea7dedc7de07bfb20b343345676054dd29bf541adeed3d594e3c",
    "score_blind_agent_d_train_path": "data/processed/restricted/official_prompt_alignment_v1/final_rationale_handoff/official-rationale-dpo-selected-handoff-exact-bundle-20260729-021/rationales.train.jsonl",
    "score_blind_agent_d_train_sha256": "45dc9bfd05d60c75214221e34149ed7bff6dae0d571a90fde287ab193bb6f347",
}
ROUND_NAMES = (
    "baseline", "ridge", "MLP", "CORAL", "Huber+ordinal",
    "uncertainty gate", "low-tail weighting", "high-tail weighting",
    "equal-band replay", "grouped loss", "3vs4 head", "thresholds",
    "adjacent contrastive", "score-blind content evidence",
    "org/expression evidence", "agent consensus/disagreement",
    "evidence distillation", "text-vs-structured fusion",
    "top structurally distinct ensemble", "bounded calibration/freeze",
)
ROUND_METHODS = (
    ("frozen_r0", {"prediction_source": "bound_r0_oof", "trainable_parameters": 0}),
    ("linear_residual", {"alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0], "fit_intercept": True}),
    ("nonlinear_residual", {"hidden_dims": [256, 128], "dropout": 0.1, "learning_rate": 0.001, "max_epochs": 100}),
    ("ordinal_coral", {"classes": 5, "shared_thresholds": True, "learning_rate": 0.001}),
    ("joint_continuous_ordinal", {"huber_delta": 1.0, "ordinal_loss_weight": 0.5}),
    ("selective_residual", {"uncertainty": "predictive_entropy", "coverage_grid": [0.25, 0.5, 0.75, 1.0]}),
    ("tail_weighted_loss", {"gold_classes": [1, 2], "weight_grid": [1.5, 2.0, 3.0]}),
    ("tail_weighted_loss", {"gold_classes": [4, 5], "weight_grid": [1.5, 2.0, 3.0]}),
    ("balanced_replay", {"bands": [1, 2, 3, 4, 5], "sampling": "equal_per_gold_band"}),
    ("group_robust_loss", {"group_key": "prompt_num", "aggregation": "equal_group_mean"}),
    ("adjacent_binary_head", {"gold_classes": [3, 4], "loss": "binary_cross_entropy"}),
    ("bounded_threshold_calibration", {"thresholds": 4, "fit_source": "train_oof_only", "monotonic": True}),
    ("contrastive_ordinal", {"adjacent_margin": 0.25, "contrastive_loss_weight": 0.25}),
    ("rationale_evidence", {"axes": ["content"], "rationale_artifact": "score_blind_rationale_v2", "score_blind": True}),
    ("rationale_evidence", {"axes": ["organization", "expression"], "rationale_artifact": "score_blind_rationale_v2", "score_blind": True}),
    ("evidence_consensus", {"rationale_artifact": "score_blind_rationale_v2", "features": ["consensus", "disagreement"], "agent_count": 4}),
    ("evidence_distillation", {"rationale_artifact": "score_blind_rationale_v2", "teacher": "round16_consensus", "temperature": 2.0}),
    ("multiview_fusion", {"rationale_artifact": "score_blind_rationale_v2", "views": ["text", "structured"], "fusion": "gated"}),
    ("diverse_ensemble", {"maximum_members": 3, "diversity_key": "method_family", "weighting": "nonnegative_oof"}),
    ("bounded_calibration", {"calibration": "monotonic_bounded", "lower_bound": 1.0, "upper_bound": 5.0, "freeze": True}),
)
PROMOTION_GATE = {
    "fixed": True,
    "selection_split": "train_oof_only",
    "primary_metric": "macro_axis_rmse",
    "direction": "lower",
    "macro_rmse_min_improvement": 0.005,
    "equal_group_rmse_min_improvement": 0.010,
    "low_tail_must_improve": True,
    "high_tail_must_improve": True,
    "gold_3_4_balanced_accuracy_min_improvement": 0.01,
    "max_axis_rmse_worsening": 0.01,
    "max_macro_spearman_fall": 0.005,
    "score1_descriptive_only": True,
    "require_all_five_folds": True,
    "require_finite_metrics": True,
}
FINAL_GATE = {
    "fixed": True,
    "eligible_candidates": "promotion_pass_only",
    "ranking_metric": "macro_axis_rmse",
    "direction": "lower",
    "macro_rmse_min_improvement": 0.01,
    "paired_bootstrap_candidate_minus_baseline_ci_upper_below_zero": True,
    "bootstrap_resamples": 10000,
    "bootstrap_seed": 2026080101,
    "validation_selection": False,
    "require_fresh_refit": True,
    "require_baseline_fallback": True,
    "freeze_after_round": 20,
}


class IterativeTailProtocolError(ValueError):
    """Raised before a mutable, leaky, or row-revealing run can proceed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise IterativeTailProtocolError(message)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    _need(set(raw) == expected, f"{label} has unknown or missing fields")


@dataclass(frozen=True)
class IterativeTailProtocol:
    """Fully validated JSON protocol with immutable artifact bindings."""

    path: Path
    raw: Mapping[str, Any]

    @property
    def run_id(self) -> str:
        return str(self.raw["run_id"])

    @property
    def bindings(self) -> Mapping[str, str]:
        return self.raw["bindings"]

    @property
    def rounds(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw["rounds"])


def validate_protocol_mapping(raw: Mapping[str, Any], *, path: str | Path = CONFIG_PATH) -> IterativeTailProtocol:
    """Validate the complete protocol, including fixed gates and round order."""
    _need(isinstance(raw, Mapping), "protocol must be a JSON object")
    _exact_keys(raw, {"schema_version", "run_id", "bindings", "data_contract", "execution", "optional_api", "rounds", "promotion_gate", "final_gate"}, "protocol")
    _need(raw["schema_version"] == SCHEMA_VERSION, "protocol schema differs")
    _need(raw["run_id"] == RUN_ID, "run ID differs")
    _need(isinstance(raw["bindings"], Mapping) and dict(raw["bindings"]) == EXPECTED_BINDINGS, "immutable artifact bindings differ")

    data = raw["data_contract"]
    _need(isinstance(data, Mapping), "data contract must be an object")
    _exact_keys(data, {"split_role", "base_prediction_origin", "records", "folds", "records_per_fold", "contains_average_target", "validation_loaded", "validation_selection"}, "data contract")
    _need(dict(data) == {"split_role": "train", "base_prediction_origin": "oof", "records": 2000, "folds": 5, "records_per_fold": 400, "contains_average_target": False, "validation_loaded": False, "validation_selection": False}, "train-only OOF data contract differs")

    execution = raw["execution"]
    _need(isinstance(execution, Mapping), "execution contract must be an object")
    _need(dict(execution) == {"authorized_gpus": [0, 1, 2, 3], "smoke_gpu": 0, "fresh_initialization_per_round": True, "reuse_round_checkpoints": False}, "GPU or fresh-initialization contract differs")
    api = raw["optional_api"]
    _need(isinstance(api, Mapping) and dict(api) == {"enabled": False, "required": False, "external_calls_allowed": False}, "optional API must remain disabled by default")

    rounds = raw["rounds"]
    _need(isinstance(rounds, list) and len(rounds) == 20, "protocol requires exactly 20 rounds")
    for index, (round_raw, expected_name, method) in enumerate(zip(rounds, ROUND_NAMES, ROUND_METHODS, strict=True), start=1):
        family, hyperparameters = method
        expected_round = {
            "number": index,
            "name": expected_name,
            "method_family": family,
            "hyperparameters": hyperparameters,
            "fresh_initialization": True,
        }
        _need(isinstance(round_raw, Mapping) and dict(round_raw) == expected_round, f"round {index} method or hyperparameters differ")

    promotion = raw["promotion_gate"]
    _need(isinstance(promotion, Mapping), "promotion gate must be an object")
    _exact_keys(promotion, set(PROMOTION_GATE), "promotion gate")
    _need(dict(promotion) == PROMOTION_GATE, "fixed promotion gate differs")
    final = raw["final_gate"]
    _need(isinstance(final, Mapping), "final gate must be an object")
    _exact_keys(final, set(FINAL_GATE), "final gate")
    _need(dict(final) == FINAL_GATE, "fixed final gate differs")
    return IterativeTailProtocol(Path(path), raw)


def load_protocol(path: str | Path = CONFIG_PATH) -> IterativeTailProtocol:
    """Read and validate the immutable protocol JSON."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IterativeTailProtocolError(f"protocol is unreadable: {config_path}") from exc
    return validate_protocol_mapping(raw, path=config_path)


def _bound_path(root: Path, relative: str) -> Path:
    _need(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "bound artifact path must be repository-relative")
    return root / relative


def load_bound_training_rows(protocol: IterativeTailProtocol, *, root: str | Path = ".") -> tuple[ResidualRow, ...]:
    """Validate every canonical binding and load exactly five 400-row folds."""
    root_path = Path(root)
    bindings = protocol.bindings
    canonical = _bound_path(root_path, bindings["canonical_train_path"])
    oof = _bound_path(root_path, bindings["r0_oof_prediction_path"])
    manifest_path = _bound_path(root_path, bindings["embedding_manifest_path"])
    provenance_path = _bound_path(root_path, bindings["embedding_provenance_path"])
    rows_path = _bound_path(root_path, bindings["embedding_rows_path"])
    rationale_path = _bound_path(root_path, bindings["score_blind_rationale_train_path"])
    rationale_manifest_path = _bound_path(root_path, bindings["score_blind_rationale_manifest_path"])
    agent_a_path = _bound_path(root_path, bindings["score_blind_agent_a_train_path"])
    agent_b_path = _bound_path(root_path, bindings["score_blind_agent_b_train_path"])
    agent_d_path = _bound_path(root_path, bindings["score_blind_agent_d_train_path"])
    for artifact, checksum, label in (
        (canonical, bindings["canonical_train_sha256"], "canonical train"),
        (oof, bindings["r0_oof_prediction_sha256"], "R0 OOF predictions"),
        (rows_path, bindings["embedding_rows_sha256"], "embedding rows"),
        (rationale_path, bindings["score_blind_rationale_train_sha256"], "score-blind rationale train"),
        (rationale_manifest_path, bindings["score_blind_rationale_manifest_sha256"], "score-blind rationale manifest"),
        (agent_a_path, bindings["score_blind_agent_a_train_sha256"], "score-blind rationale agent A"),
        (agent_b_path, bindings["score_blind_agent_b_train_sha256"], "score-blind rationale agent B"),
        (agent_d_path, bindings["score_blind_agent_d_train_sha256"], "score-blind rationale agent D"),
    ):
        _need(artifact.is_file() and not artifact.is_symlink(), f"{label} path is missing or mutable")
        _need(_sha256(artifact) == checksum, f"{label} checksum differs")
    _need(manifest_path.is_file() and not manifest_path.is_symlink(), "embedding manifest path is missing or mutable")
    _need(provenance_path.is_file() and not provenance_path.is_symlink(), "embedding provenance path is missing or mutable")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IterativeTailProtocolError("embedding provenance is unreadable") from exc
    expected_provenance = provenance.get("input_provenance", {})
    _need(expected_provenance.get("canonical_source_sha256") == bindings["canonical_train_sha256"], "provenance canonical train differs")
    _need(expected_provenance.get("base_prediction_provenance", {}).get("prediction_sha256") == bindings["r0_oof_prediction_sha256"], "provenance R0 OOF prediction differs")
    _need(expected_provenance.get("embedding_model_revision") == bindings["embedding_model_revision"], "provenance embedding revision differs")
    _need(provenance.get("rows_sha256") == bindings["embedding_rows_sha256"], "provenance embedding rows differs")

    try:
        manifest, rows = load_embedding_artifact(manifest_path, rows_path)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise IterativeTailProtocolError("bound embedding artifact failed validation") from exc
    _need(manifest.split_role == "train" and manifest.base_prediction_origin == "oof", "only train OOF embeddings may be loaded")
    _need(not manifest.evaluation_only and not manifest.contains_average_target, "validation or average target is forbidden")
    _need((manifest.embedding_model_id, manifest.embedding_model_revision) == (bindings["embedding_model_id"], bindings["embedding_model_revision"]), "embedding model binding differs")
    _need(len(rows) == 2000, "embedding population must be exactly 2,000")
    counts = Counter(row.oof_fold for row in rows)
    _need(counts == Counter({fold: 400 for fold in range(5)}), "each OOF fold must contain exactly 400 rows")
    assignments = "\n".join(f"{row.source_id}:{row.oof_fold}" for row in sorted(rows, key=lambda item: item.source_id))
    _need(sha256(assignments.encode("utf-8")).hexdigest() == bindings["fold_assignment_fingerprint"], "fold assignment fingerprint differs")
    return rows


_FORBIDDEN_AGGREGATE_KEYS = {"source_id", "group_id", "document_id", "essay", "text", "rationale", "rows", "predictions"}


def _aggregate_metrics(raw: Mapping[str, Any]) -> dict[str, int | float | bool | None]:
    _need(isinstance(raw, Mapping), "aggregate metrics must be an object")
    result: dict[str, int | float | bool | None] = {}
    for key, value in raw.items():
        _need(isinstance(key, str) and key not in _FORBIDDEN_AGGREGATE_KEYS, f"row-level field is forbidden in aggregate output: {key}")
        _need(value is None or isinstance(value, (bool, Real)), f"aggregate metric must be scalar: {key}")
        if isinstance(value, Real) and not isinstance(value, bool):
            _need(math.isfinite(float(value)), f"aggregate metric must be finite: {key}")
        result[key] = value
    return result


def build_task_card(protocol: IterativeTailProtocol) -> dict[str, Any]:
    """Build the non-sensitive, aggregate-only execution task card."""
    return {
        "schema_version": "mal2026-iterative-tail-task-card-v1",
        "run_id": protocol.run_id,
        "status": "authorized_not_started",
        "gpu_scope": list(protocol.raw["execution"]["authorized_gpus"]),
        "smoke_gpu": protocol.raw["execution"]["smoke_gpu"],
        "round_count": len(protocol.rounds),
        "train_records": protocol.raw["data_contract"]["records"],
        "fold_count": protocol.raw["data_contract"]["folds"],
        "records_per_fold": protocol.raw["data_contract"]["records_per_fold"],
        "fresh_initialization_per_round": True,
        "validation_selection": False,
        "optional_api_enabled": False,
    }


def build_protocol_summary(protocol: IterativeTailProtocol, round_aggregates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create a public-safe summary containing scalar round aggregates only."""
    _need(len(round_aggregates) <= len(protocol.rounds), "more aggregate results than protocol rounds")
    rounds = []
    for round_spec, metrics in zip(protocol.rounds, round_aggregates, strict=False):
        rounds.append({"number": round_spec["number"], "name": round_spec["name"], "aggregate_metrics": _aggregate_metrics(metrics)})
    return {
        "schema_version": "mal2026-iterative-tail-protocol-summary-v1",
        "run_id": protocol.run_id,
        "completed_rounds": len(rounds),
        "planned_rounds": len(protocol.rounds),
        "validation_selection": False,
        "optional_api_enabled": False,
        "rounds": rounds,
    }


# Explicit aliases keep call sites readable without introducing another API.
load_config = load_protocol
validate_bound_inputs = load_bound_training_rows
protocol_summary = build_protocol_summary
