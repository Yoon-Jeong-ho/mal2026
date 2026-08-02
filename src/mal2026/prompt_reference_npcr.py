"""Train-only prompt-reference neural pairwise comparative regression (NPCR).

This fixed-feature competitor intentionally consumes only frozen public 4096-d
embeddings.  It never exposes an R0 score to a model: embeddings, canonical
prompt identifiers, canonical raw analytic-axis targets, and the exact OOF
fold assignment are the complete input contract.  All pair pools, anchors,
and selection statistics are scoped to the current fit fold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .iterative_tail_metrics import AXES, compute_iterative_tail_metrics, metric_improvements, paired_bootstrap_delta_ci
from .r0_ordinal_residual import load_embedding_artifact


SCHEMA_VERSION = "mal2026-prompt-reference-npcr-v1"
RESTRICTED_KEYS = frozenset({
    "source_id", "document_id", "prompt_num", "essay", "prompt", "embedding",
    "raw_gold", "raw_scores", "row_prediction", "row_predictions",
})


class PromptReferenceNPCRError(ValueError):
    """Raised when the train-only NPCR contract is violated."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise PromptReferenceNPCRError(message)


def file_sha256(path: str | Path) -> str:
    value = Path(path)
    need(value.is_file() and not value.is_symlink(), "checksum input must be an ordinary file")
    digest = sha256()
    with value.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_hash(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _contains_validation(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_validation(key) or _contains_validation(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_validation(child) for child in value)
    return isinstance(value, str) and "validation" in value.lower()


def _raw_axis_value(value: object, axis: str) -> float:
    need(type(value) in {int, float} and not isinstance(value, bool), f"{axis} score is nonnumeric")
    result = float(value)
    need(math.isfinite(result) and 1.0 <= result <= 5.0, f"{axis} score lies outside [1,5]")
    return result


def ordinal_band(raw_score: float) -> int:
    """Official positive half-up band, used only for pair-gap strata."""
    return min(5, max(1, int(math.floor(float(raw_score) + 0.5))))


@dataclass(frozen=True)
class PairSamplingSpec:
    identifier: str
    gaps: tuple[int, ...]
    references_per_gap: int
    anchors_per_query: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PairSamplingSpec":
        need(set(raw) == {"id", "gaps", "references_per_gap", "anchors_per_query"}, "pair candidate fields differ")
        need(isinstance(raw["gaps"], list), "pair gaps must be a list")
        value = cls(str(raw["id"]), tuple(raw["gaps"]), int(raw["references_per_gap"]), int(raw["anchors_per_query"]))
        value.validate()
        return value

    def validate(self) -> None:
        need(bool(self.identifier) and all(character.isalnum() or character in "-_" for character in self.identifier), "pair candidate ID is unsafe")
        need(self.gaps == tuple(sorted(set(self.gaps))), "pair gaps must be unique and sorted")
        need(1 in self.gaps and any(gap >= 2 for gap in self.gaps) and all(1 <= gap <= 4 for gap in self.gaps),
             "each NPCR candidate requires adjacent and skip-gap pairs")
        need(1 <= self.references_per_gap <= 8 and 1 <= self.anchors_per_query <= 32, "pair/anchor caps differ")


@dataclass(frozen=True)
class NPCRConfig:
    schema_version: str
    run_id: str
    train_path: str
    train_sha256: str
    embedding_manifest_path: str
    embedding_manifest_sha256: str
    embedding_rows_path: str
    embedding_rows_sha256: str
    r0_oof_prediction_path: str
    r0_oof_prediction_sha256: str
    output_root: str
    restricted_output_root: str
    seed: int
    outer_folds: int
    inner_folds: int
    axes: tuple[str, ...]
    average_target_forbidden: bool
    candidates: tuple[PairSamplingSpec, ...]
    hidden_dim: int
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    inner_selection: tuple[str, ...]
    promotion_gate: Mapping[str, Any]
    config_sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, config_sha256: str | None = None) -> "NPCRConfig":
        need(isinstance(raw, Mapping), "NPCR config must be an object")
        need(not _contains_validation(raw), "validation paths or strings are forbidden")
        normalized = dict(raw)
        for name in ("axes", "candidates", "inner_selection"):
            need(isinstance(normalized.get(name), list), f"{name} must be a list")
        normalized["axes"] = tuple(normalized["axes"])
        normalized["candidates"] = tuple(PairSamplingSpec.from_mapping(item) for item in normalized["candidates"])
        normalized["inner_selection"] = tuple(normalized["inner_selection"])
        normalized["config_sha256"] = config_sha256 or _canonical_json_hash(raw)
        need(set(normalized) == set(cls.__dataclass_fields__), "NPCR config fields differ")
        value = cls(**normalized)
        value.validate()
        return value

    @classmethod
    def from_json(cls, path: str | Path) -> "NPCRConfig":
        location = Path(path)
        try:
            raw = json.loads(location.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PromptReferenceNPCRError("NPCR config is unreadable") from exc
        return cls.from_mapping(raw, config_sha256=file_sha256(location))

    def validate(self) -> None:
        need(self.schema_version == SCHEMA_VERSION, "NPCR schema differs")
        need(bool(re.fullmatch(r"[a-z0-9][a-z0-9-]{2,127}", self.run_id)) and self.axes == AXES and self.average_target_forbidden is True,
             "run ID/axis/average contract differs")
        need(self.outer_folds == 5 and self.inner_folds == 4, "NPCR requires fixed 5x4 folds")
        need(1 <= len(self.candidates) <= 4 and len({item.identifier for item in self.candidates}) == len(self.candidates), "small preregistered pair matrix differs")
        need(self.hidden_dim > 0 and self.epochs > 0 and self.batch_size > 0 and self.learning_rate > 0 and self.weight_decay >= 0,
             "NPCR optimization settings differ")
        need(self.inner_selection == ("macro_rmse", "equal_group_rmse", "low_tail_rmse", "high_tail_rmse", "candidate_id"),
             "NPCR candidate-only inner selection differs")
        required_gate = {"minimum_macro_rmse_improvement", "maximum_axis_rmse_worsening", "maximum_gold_3_4_balanced_accuracy_drop",
                         "low_tail_noninferior", "high_tail_noninferior", "paired_bootstrap_resamples", "paired_bootstrap_lower_bound_above_zero"}
        need(set(self.promotion_gate) == required_gate and float(self.promotion_gate["minimum_macro_rmse_improvement"]) > 0
             and float(self.promotion_gate["maximum_axis_rmse_worsening"]) >= 0 and float(self.promotion_gate["maximum_gold_3_4_balanced_accuracy_drop"]) >= 0
             and self.promotion_gate["low_tail_noninferior"] is True and self.promotion_gate["high_tail_noninferior"] is True
             and type(self.promotion_gate["paired_bootstrap_resamples"]) is int and self.promotion_gate["paired_bootstrap_resamples"] >= 100
             and self.promotion_gate["paired_bootstrap_lower_bound_above_zero"] is True, "NPCR promotion gate differs")
        need(len(self.train_sha256) == len(self.embedding_manifest_sha256) == len(self.embedding_rows_sha256)
             == len(self.r0_oof_prediction_sha256) == len(self.config_sha256) == 64,
             "NPCR checksum format differs")
        need(Path(self.output_root).resolve() != Path(self.restricted_output_root).resolve(), "public and restricted roots must differ")


@dataclass(frozen=True)
class NPCRRow:
    source_id: str
    document_id: str
    prompt_num: str
    embedding: tuple[float, ...]
    raw_scores: tuple[float, float, float]
    oof_fold: int


@dataclass(frozen=True)
class Pair:
    query: int
    reference: int
    gap: int


def load_canonical_raw_rows(path: str | Path, expected_sha256: str) -> dict[str, tuple[str, str, tuple[float, float, float]]]:
    """Read canonical prompt/group/raw-axis values without indexing ``average``."""
    location = Path(path)
    need(file_sha256(location) == expected_sha256, "canonical train checksum differs")
    result: dict[str, tuple[str, str, tuple[float, float, float]]] = {}
    with location.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromptReferenceNPCRError(f"invalid canonical JSONL at line {line_number}") from exc
            need(isinstance(raw, Mapping) and set(raw) == {"id", "document_id", "prompt_num", "prompt", "essay", "score"},
                 "canonical train row schema differs")
            identifier = raw["id"]
            need(isinstance(identifier, str) and identifier and identifier not in result, "canonical ID differs")
            score = raw["score"]
            need(isinstance(score, Mapping) and set(score) == {*AXES, "average"}, "canonical score schema differs")
            # Do not read score["average"].
            values = tuple(_raw_axis_value(score[axis], axis) for axis in AXES)
            prompt_num, document_id = raw["prompt_num"], raw["document_id"]
            need(isinstance(prompt_num, (str, int)) and isinstance(document_id, (str, int)), "canonical group fields differ")
            result[identifier] = (str(prompt_num), str(document_id), values)  # type: ignore[arg-type]
    need(len(result) == 2000, "NPCR expects 2,000 canonical training rows")
    return result


def load_rows(config: NPCRConfig) -> tuple[NPCRRow, ...]:
    """Join canonical raw targets/prompt keys to frozen embeddings and OOF folds.

    ``ResidualRow.base_predictions`` is intentionally never read or copied.
    """
    config.validate()
    need(file_sha256(config.embedding_manifest_path) == config.embedding_manifest_sha256, "embedding manifest checksum differs")
    # The exact R0 OOF artifact is a lineage/fold binding only.  It is not
    # parsed, featurized, calibrated against, or exposed to the utility net.
    need(file_sha256(config.r0_oof_prediction_path) == config.r0_oof_prediction_sha256, "exact R0 OOF checksum differs")
    canonical = load_canonical_raw_rows(config.train_path, config.train_sha256)
    manifest, embedded = load_embedding_artifact(config.embedding_manifest_path, config.embedding_rows_path)
    need(manifest.split_role == "train" and manifest.embedding_source == "public" and manifest.embedding_frozen
         and manifest.embedding_dim == 4096 and manifest.fold_count == 5 and not manifest.contains_average_target,
         "frozen public embedding contract differs")
    need(manifest.rows_sha256 == config.embedding_rows_sha256 and file_sha256(config.embedding_rows_path) == config.embedding_rows_sha256,
         "embedding rows checksum differs")
    need(len(embedded) == len(canonical) == 2000, "NPCR population differs")
    result: list[NPCRRow] = []
    seen_groups: set[tuple[str, str]] = set()
    for item in embedded:
        need(item.source_id in canonical and item.oof_fold is not None, "embedding/canonical linkage differs")
        prompt_num, document_id, raw_scores = canonical[item.source_id]
        need(item.group_id == document_id, "embedding group must equal canonical document_id")
        group = (prompt_num, document_id)
        need(group not in seen_groups, "canonical document group is non-unique")
        seen_groups.add(group)
        result.append(NPCRRow(item.source_id, document_id, prompt_num, item.shared_embedding, raw_scores, int(item.oof_fold)))
    need({row.source_id for row in result} == set(canonical) and {row.oof_fold for row in result} == set(range(5)), "NPCR OOF folds differ")
    _bound_r0_predictions(config, canonical, embedded)
    return tuple(result)


def _axis_mapping(value: object, label: str) -> Mapping[str, float]:
    need(isinstance(value, Mapping) and set(value) == set(AXES), f"R0 {label} axis schema differs")
    result = {axis: _raw_axis_value(value[axis], f"R0 {label}/{axis}") for axis in AXES}
    return result


def _bound_r0_predictions(
    config: NPCRConfig,
    canonical: Mapping[str, tuple[str, str, tuple[float, float, float]]],
    embedded: Sequence[Any],
) -> dict[str, Mapping[str, float]]:
    """Verify exact-R0 lineage without ever returning it as an NPCR feature."""
    by_id = {item.source_id: item for item in embedded}
    result: dict[str, Mapping[str, float]] = {}
    expected = {"source_id", "fold", "continuous_prediction", "half_up_integer_prediction", "reference_score"}
    with Path(config.r0_oof_prediction_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromptReferenceNPCRError(f"invalid exact R0 OOF JSONL at line {line_number}") from exc
            need(isinstance(raw, Mapping) and set(raw) == expected, "exact R0 OOF row schema differs")
            identifier, fold = raw["source_id"], raw["fold"]
            need(isinstance(identifier, str) and identifier in canonical and identifier not in result and type(fold) is int and 0 <= fold < 5,
                 "exact R0 OOF ID/fold differs")
            prediction = _axis_mapping(raw["continuous_prediction"], "continuous prediction")
            reference = _axis_mapping(raw["reference_score"], "reference score")
            integer = raw["half_up_integer_prediction"]
            need(isinstance(integer, Mapping) and set(integer) == set(AXES)
                 and all(type(integer[axis]) is int and 1 <= integer[axis] <= 5 for axis in AXES), "exact R0 integer schema differs")
            item = by_id[identifier]
            need(fold == item.oof_fold, "exact R0/embedding fold differs")
            raw_scores = canonical[identifier][2]
            need(all(abs(reference[axis] - raw_scores[position]) <= 1e-9 for position, axis in enumerate(AXES)),
                 "exact R0/canonical raw-axis equality differs")
            need(all(abs(prediction[axis] - float(item.base_predictions[position])) <= 1e-9 for position, axis in enumerate(AXES)),
                 "exact R0/embedding three-axis prediction equality differs")
            result[identifier] = prediction
    need(set(result) == set(canonical) and len(result) == 2000, "exact R0 OOF coverage differs")
    return result


def bound_r0_predictions(config: NPCRConfig, rows: Sequence[NPCRRow]) -> dict[str, Mapping[str, float]]:
    """Read comparison-only exact-R0 predictions after the binding made in ``load_rows``."""
    result: dict[str, Mapping[str, float]] = {}
    with Path(config.r0_oof_prediction_path).open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            identifier, fold = raw["source_id"], raw["fold"]
            expected = next((row for row in rows if row.source_id == identifier), None)
            need(isinstance(identifier, str) and identifier not in result and expected is not None and fold == expected.oof_fold,
                 "exact R0 aggregate ID/fold differs")
            result[identifier] = _axis_mapping(raw["continuous_prediction"], "aggregate continuous prediction")
    need(set(result) == {row.source_id for row in rows}, "exact R0 aggregate coverage differs")
    return result


def outer_and_inner_indices(rows: Sequence[NPCRRow], outer_fold: int) -> tuple[tuple[int, ...], dict[int, tuple[tuple[int, ...], tuple[int, ...]]]]:
    need(0 <= outer_fold < 5 and len(rows) == 2000, "outer fold/population differs")
    outer = tuple(index for index, row in enumerate(rows) if row.oof_fold == outer_fold)
    inner: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for dev_fold in range(5):
        if dev_fold == outer_fold:
            continue
        fit = tuple(index for index, row in enumerate(rows) if row.oof_fold not in {outer_fold, dev_fold})
        dev = tuple(index for index, row in enumerate(rows) if row.oof_fold == dev_fold)
        inner[dev_fold] = (fit, dev)
    need(len(outer) == 400 and len(inner) == 4 and all(len(fit) == 1200 and len(dev) == 400 for fit, dev in inner.values()),
         "nested 5x4 sizes differ")
    outer_set = set(outer)
    need(all(outer_set.isdisjoint(fit) and outer_set.isdisjoint(dev) and set(fit).isdisjoint(dev) for fit, dev in inner.values()),
         "outer or inner leakage")
    return outer, inner


def _ordered(indices: Iterable[int], rows: Sequence[NPCRRow], seed: int, *parts: object) -> list[int]:
    return sorted(indices, key=lambda index: sha256(("\0".join(map(str, (seed, *parts, rows[index].source_id)))).encode()).hexdigest())


def build_prompt_pairs(rows: Sequence[NPCRRow], fit_indices: Sequence[int], axis: int, candidate: PairSamplingSpec, seed: int) -> tuple[Pair, ...]:
    """Build deterministic same-prompt, fit-only adjacent/skip-gap pairs."""
    need(axis in range(len(AXES)), "axis differs")
    candidate.validate()
    fit = tuple(fit_indices)
    need(bool(fit) and len(set(fit)) == len(fit), "fit indices differ")
    fit_set = set(fit)
    by_prompt: dict[str, list[int]] = {}
    for index in fit:
        by_prompt.setdefault(rows[index].prompt_num, []).append(index)
    pairs: list[Pair] = []
    for query in fit:
        query_band = ordinal_band(rows[query].raw_scores[axis])
        for gap in candidate.gaps:
            choices = [
                reference for reference in by_prompt[rows[query].prompt_num]
                if reference != query and rows[reference].document_id != rows[query].document_id
                and abs(ordinal_band(rows[reference].raw_scores[axis]) - query_band) == gap
            ]
            for reference in _ordered(choices, rows, seed, candidate.identifier, axis, "pair", query, gap)[:candidate.references_per_gap]:
                need(query in fit_set and reference in fit_set and rows[query].prompt_num == rows[reference].prompt_num,
                     "pair locality leakage")
                pairs.append(Pair(query, reference, gap))
    need(bool(pairs), "fit fold has no eligible prompt-local NPCR pairs")
    return tuple(pairs)


def select_anchors(rows: Sequence[NPCRRow], fit_indices: Sequence[int], query_index: int, axis: int, anchor_count: int, seed: int) -> tuple[int, ...]:
    """Select only same-prompt fit rows; held-row gold is never consulted."""
    need(axis in range(len(AXES)) and anchor_count > 0, "anchor parameters differ")
    query = rows[query_index]
    candidates = [
        index for index in fit_indices
        if index != query_index and rows[index].prompt_num == query.prompt_num and rows[index].document_id != query.document_id
    ]
    selected = tuple(_ordered(candidates, rows, seed, "anchor", axis, query_index)[:anchor_count])
    need(bool(selected), "query has no same-prompt fit-fold anchor")
    need(all(index in set(fit_indices) and rows[index].prompt_num == query.prompt_num for index in selected), "anchor locality leakage")
    return selected


def build_utility_network(input_dim: int, hidden_dim: int) -> Any:
    import torch.nn as nn
    need(input_dim == 4096 and hidden_dim > 0, "NPCR scalar utility dimensions differ")
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))


def utility_difference(model: Any, query_embedding: Any, reference_embedding: Any) -> Any:
    """The only learned pair score: antisymmetry follows algebraically."""
    return model(query_embedding).squeeze(-1) - model(reference_embedding).squeeze(-1)


def _features(rows: Sequence[NPCRRow], indices: Sequence[int]) -> np.ndarray:
    value = np.asarray([rows[index].embedding for index in indices], dtype=np.float32)
    need(value.ndim == 2 and value.shape == (len(indices), 4096) and np.isfinite(value).all(), "frozen embedding features differ")
    return value


def _seed(seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_utility_network(
    rows: Sequence[NPCRRow], fit_indices: Sequence[int], axis: int, candidate: PairSamplingSpec, config: NPCRConfig,
    *, seed: int, device: str = "cpu",
) -> tuple[Any, tuple[Pair, ...]]:
    """Fit one independent axis utility network from fit-only pair differences."""
    import torch
    import torch.nn.functional as functional

    _seed(seed)
    pairs = build_prompt_pairs(rows, fit_indices, axis, candidate, seed)
    model = build_utility_network(4096, config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    features = torch.tensor(_features(rows, range(len(rows))), dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(config.epochs):
        order = torch.randperm(len(pairs), generator=generator).tolist()
        for start in range(0, len(order), config.batch_size):
            batch = [pairs[position] for position in order[start:start + config.batch_size]]
            query = torch.tensor([item.query for item in batch], device=device)
            reference = torch.tensor([item.reference for item in batch], device=device)
            target = torch.tensor([rows[item.query].raw_scores[axis] - rows[item.reference].raw_scores[axis] for item in batch],
                                  dtype=torch.float32, device=device)
            predicted = utility_difference(model, features[query], features[reference])
            loss = functional.huber_loss(predicted, target, delta=0.5)
            need(bool(torch.isfinite(loss)), "NPCR pair loss is non-finite")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model, pairs


def recover_absolute_scores(
    model: Any, rows: Sequence[NPCRRow], query_indices: Sequence[int], fit_indices: Sequence[int], axis: int,
    candidate: PairSamplingSpec, *, seed: int, device: str = "cpu",
) -> np.ndarray:
    """Recover absolute scores from same-prompt, fit-only labeled anchors."""
    import torch
    model.eval()
    features = torch.tensor(_features(rows, range(len(rows))), dtype=torch.float32, device=device)
    values: list[float] = []
    with torch.inference_mode():
        for query in query_indices:
            anchors = select_anchors(rows, fit_indices, query, axis, candidate.anchors_per_query, seed)
            query_value = model(features[query:query + 1]).squeeze().float()
            anchor_tensor = torch.tensor(anchors, dtype=torch.long, device=device)
            anchor_utility = model(features[anchor_tensor]).squeeze(-1).float()
            anchor_scores = torch.tensor([rows[index].raw_scores[axis] for index in anchors], dtype=torch.float32, device=device)
            restored_values = torch.sort(anchor_scores + query_value - anchor_utility).values
            middle = len(restored_values) // 2
            # Use the conventional median: average the two middle anchors for
            # an even anchor count (``torch.median`` would choose the lower).
            restored = (restored_values[middle] if len(restored_values) % 2 else
                        0.5 * (restored_values[middle - 1] + restored_values[middle])).item()
            values.append(float(min(5.0, max(1.0, restored))))
    return np.asarray(values, dtype=np.float64)


def _derived_seed(base: int, outer: int, candidate: str, axis: int, phase: str) -> int:
    return int(sha256(f"{base}\0{outer}\0{candidate}\0{axis}\0{phase}".encode()).hexdigest()[:15], 16) % (2**31 - 1)


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _fit_predict(
    rows: Sequence[NPCRRow], fit: Sequence[int], query: Sequence[int], axis: int, candidate: PairSamplingSpec,
    config: NPCRConfig, *, seed: int, device: str,
) -> tuple[np.ndarray, int]:
    model, pairs = train_utility_network(rows, fit, axis, candidate, config, seed=seed, device=device)
    prediction = recover_absolute_scores(model, rows, query, fit, axis, candidate, seed=seed, device=device)
    return prediction, len(pairs)


def _inner_oof(rows: Sequence[NPCRRow], outer_fold: int, candidate: PairSamplingSpec, config: NPCRConfig, device: str) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], dict[str, int]]:
    _, inner = outer_and_inner_indices(rows, outer_fold)
    ordered_indices: list[int] = []
    predictions: list[np.ndarray] = []
    pair_counts: dict[str, int] = {axis: 0 for axis in AXES}
    for dev_fold in sorted(inner):
        fit, dev = inner[dev_fold]
        axis_predictions = []
        for axis, axis_name in enumerate(AXES):
            predicted, count = _fit_predict(rows, fit, dev, axis, candidate, config,
                                            seed=_derived_seed(config.seed, outer_fold, candidate.identifier, axis, f"inner-{dev_fold}"), device=device)
            axis_predictions.append(predicted); pair_counts[axis_name] += count
        ordered_indices.extend(dev)
        predictions.append(np.column_stack(axis_predictions))
    prediction = np.vstack(predictions)
    truth = np.asarray([rows[index].raw_scores for index in ordered_indices], dtype=float)
    need(prediction.shape == truth.shape == (1600, 3), "inner OOF shape differs")
    return truth, prediction, tuple(ordered_indices), pair_counts


def _tail_support(metrics: Mapping[str, Any]) -> bool:
    return all(
        int(metrics["axes"][axis]["bands"]["1"]["count"]) + int(metrics["axes"][axis]["bands"]["2"]["count"]) > 0
        and int(metrics["axes"][axis]["bands"]["5"]["count"]) > 0
        for axis in AXES
    )


def global_promotion_gate(
    truth: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, document_ids: Sequence[str], config: NPCRConfig, *, seed: int,
) -> dict[str, Any]:
    """Apply the post-OOF protected-R0 promotion gate; never used for selection."""
    base_metrics, candidate_metrics = compute_iterative_tail_metrics(truth, baseline), compute_iterative_tail_metrics(truth, candidate)
    delta = metric_improvements(base_metrics, candidate_metrics)
    bootstrap = paired_bootstrap_delta_ci(truth, baseline, candidate, document_ids=document_ids,
                                          n_resamples=int(config.promotion_gate["paired_bootstrap_resamples"]), seed=seed)
    gates = {
        "pooled_macro_rmse": delta["rmse"] is not None and delta["rmse"] >= float(config.promotion_gate["minimum_macro_rmse_improvement"]),
        "axis_rmse_cap": all(value >= -float(config.promotion_gate["maximum_axis_rmse_worsening"]) for value in delta["axis_rmse"].values()),
        "gold_3_4_ba_floor": delta["gold_3_4_balanced_accuracy"] is not None and delta["gold_3_4_balanced_accuracy"] >= -float(config.promotion_gate["maximum_gold_3_4_balanced_accuracy_drop"]),
        "tail_support": _tail_support(base_metrics),
        "low_tail_noninferior": delta["low_tail_rmse"] is not None and delta["low_tail_rmse"] >= 0.0,
        "high_tail_noninferior": delta["high_tail_rmse"] is not None and delta["high_tail_rmse"] >= 0.0,
        "paired_bootstrap": bootstrap["intervals"]["rmse"]["lower"] is not None and bootstrap["intervals"]["rmse"]["lower"] > 0.0,
    }
    return {"eligible": all(gates.values()), "gates": gates, "improvements": delta, "candidate_metrics": candidate_metrics,
            "exact_r0_metrics": base_metrics, "paired_bootstrap": bootstrap,
            "score1_used_for_promotion": False}


def _secure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o770)
    need(path.stat().st_mode & 0o007 == 0, "restricted directory has world permissions")


def _assert_private_file(path: Path) -> None:
    mode = path.stat().st_mode
    need(mode & 0o007 == 0 and mode & 0o111 == 0 and mode & 0o660 == 0o660, "restricted file ACL/mode is not project-private")


def _atomic_jsonl_private(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    need(not path.exists(), "refusing to overwrite restricted NPCR predictions")
    _secure_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary, 0o660)
        os.replace(temporary, path)
        os.chmod(path, 0o660); _assert_private_file(path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return file_sha256(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    need(not path.exists(), "refusing to overwrite NPCR public output")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return file_sha256(path)


def _validate_public_payload(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            need(str(key) not in RESTRICTED_KEYS, "public NPCR payload contains restricted row material")
            _validate_public_payload(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_public_payload(child)


def _execution_metadata(seed: int, device: str) -> dict[str, Any]:
    import torch
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unavailable"
    return {"device": device, "seed": seed, "deterministic_algorithms": True, "torch": torch.__version__,
            "cuda": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_sha": git_sha, "command": os.environ.get("MAL2026_NPCR_COMMAND", "unrecorded"),
            "environment": {"python": sys.version.split()[0], "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}}


def run_outer_fold(config: NPCRConfig, outer_fold: int, *, device: str | None = None) -> dict[str, Any]:
    """Nested selection and one held-out prediction pass for a single outer fold."""
    rows = load_rows(config)
    outer, _ = outer_and_inner_indices(rows, outer_fold)
    execution_device = _device() if device is None else device
    restricted_path = Path(config.restricted_output_root) / config.run_id / f"outer-{outer_fold:02d}.jsonl"
    public_path = Path(config.output_root) / config.run_id / f"outer-{outer_fold:02d}.json"
    need(not restricted_path.exists() and not public_path.exists(), "stale NPCR outer output exists")
    screen = []
    for candidate in config.candidates:
        truth, prediction, _, pair_counts = _inner_oof(rows, outer_fold, candidate, config, execution_device)
        metrics = compute_iterative_tail_metrics(truth, prediction)
        screen.append({"candidate_id": candidate.identifier, "inner_pair_counts": pair_counts, "metrics": metrics})
    selected = min(screen, key=lambda item: (float(item["metrics"]["macro"]["rmse"]), float(item["metrics"]["macro"]["equal_group_rmse"]),
                                              float(item["metrics"]["macro"]["low_tail_rmse"]), float(item["metrics"]["macro"]["high_tail_rmse"]), str(item["candidate_id"])))
    selected_id = str(selected["candidate_id"])
    candidate = next(item for item in config.candidates if item.identifier == selected_id)
    fit = tuple(index for index in range(len(rows)) if index not in set(outer))
    per_axis: list[np.ndarray] = []
    refit_pairs: dict[str, int] = {}
    for axis, axis_name in enumerate(AXES):
        predicted, count = _fit_predict(rows, fit, outer, axis, candidate, config,
                                        seed=_derived_seed(config.seed, outer_fold, candidate.identifier, axis, "outer-refit"), device=execution_device)
        per_axis.append(predicted); refit_pairs[axis_name] = count
    prediction = np.column_stack(per_axis)
    truth = np.asarray([rows[index].raw_scores for index in outer], dtype=float)
    rows_sha = _atomic_jsonl_private(restricted_path, (
        {"source_id": rows[index].source_id, "outer_fold": outer_fold,
         "raw_gold": {axis: float(truth[position, axis_index]) for axis_index, axis in enumerate(AXES)},
         "row_prediction": {axis: float(prediction[position, axis_index]) for axis_index, axis in enumerate(AXES)}}
        for position, index in enumerate(outer)
    ))
    result = {
        "schema_version": SCHEMA_VERSION, "status": "completed", "mode": "outer_fold", "run_id": config.run_id,
        "outer_fold": outer_fold, "records": 400, "selected_candidate": selected_id, "candidate_inventory": [item.identifier for item in config.candidates],
        "inner_selection_metric": "candidate_only_lexicographic_raw_metrics", "inner_screen": screen, "outer_metrics": compute_iterative_tail_metrics(truth, prediction),
        "refit_pair_counts": refit_pairs, "config_sha256": config.config_sha256, "train_sha256": config.train_sha256,
        "embedding_manifest_sha256": config.embedding_manifest_sha256, "embedding_rows_sha256": config.embedding_rows_sha256,
        "r0_oof_prediction_sha256": config.r0_oof_prediction_sha256,
        "restricted_predictions_sha256": rows_sha, "validation_rows_loaded": False, "average_target_used": False,
        "r0_score_feature_used": False, "execution": _execution_metadata(config.seed, execution_device),
        "privacy": "aggregate_only_public_row_predictions_restricted",
    }
    _validate_public_payload(result)
    _atomic_json(public_path, result)
    return result


def aggregate_full(config: NPCRConfig) -> dict[str, Any]:
    """Aggregate five already-completed outer predictions without re-selection."""
    rows = load_rows(config)
    r0 = bound_r0_predictions(config, rows)
    by_id: dict[str, Mapping[str, Any]] = {}
    bindings = []
    for outer_fold in range(5):
        public_path = Path(config.output_root) / config.run_id / f"outer-{outer_fold:02d}.json"
        restricted_path = Path(config.restricted_output_root) / config.run_id / f"outer-{outer_fold:02d}.jsonl"
        public = json.loads(public_path.read_text(encoding="utf-8"))
        expected_public = {"schema_version", "status", "mode", "run_id", "outer_fold", "records", "selected_candidate", "candidate_inventory",
                           "inner_selection_metric", "inner_screen", "outer_metrics", "refit_pair_counts", "config_sha256",
                           "train_sha256", "embedding_manifest_sha256", "embedding_rows_sha256", "r0_oof_prediction_sha256", "restricted_predictions_sha256",
                           "validation_rows_loaded", "average_target_used", "r0_score_feature_used", "execution", "privacy"}
        need(set(public) == expected_public and public.get("schema_version") == SCHEMA_VERSION and public.get("status") == "completed"
             and public.get("mode") == "outer_fold" and public.get("run_id") == config.run_id and public.get("outer_fold") == outer_fold
             and public.get("records") == 400 and public.get("config_sha256") == config.config_sha256
             and public.get("train_sha256") == config.train_sha256 and public.get("embedding_rows_sha256") == config.embedding_rows_sha256
             and public.get("r0_oof_prediction_sha256") == config.r0_oof_prediction_sha256
             and public.get("embedding_manifest_sha256") == config.embedding_manifest_sha256
             and public.get("candidate_inventory") == [item.identifier for item in config.candidates]
             and public.get("selected_candidate") in {item.identifier for item in config.candidates}
             and public.get("restricted_predictions_sha256") == file_sha256(restricted_path)
             and public.get("validation_rows_loaded") is False and public.get("average_target_used") is False and public.get("r0_score_feature_used") is False,
             "outer NPCR binding differs")
        with restricted_path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                identifier = item.get("source_id")
                expected_row = next((row for row in rows if row.source_id == identifier), None)
                need(isinstance(item, Mapping) and set(item) == {"source_id", "outer_fold", "raw_gold", "row_prediction"}
                     and isinstance(identifier, str) and identifier not in by_id and expected_row is not None
                     and item.get("outer_fold") == outer_fold == expected_row.oof_fold, "outer NPCR ID/fold/schema differs")
                gold, prediction = item["raw_gold"], item["row_prediction"]
                need(isinstance(gold, Mapping) and isinstance(prediction, Mapping) and set(gold) == set(prediction) == set(AXES),
                     "outer NPCR axis schema differs")
                need(all(abs(_raw_axis_value(gold[axis], axis) - expected_row.raw_scores[position]) <= 1e-9 for position, axis in enumerate(AXES)),
                     "restricted NPCR gold differs from canonical row")
                need(all(math.isfinite(float(prediction[axis])) and 1.0 <= float(prediction[axis]) <= 5.0 for axis in AXES),
                     "restricted NPCR prediction range differs")
                by_id[identifier] = item
        bindings.append({"outer_fold": outer_fold, "public_sha256": file_sha256(public_path), "restricted_predictions_sha256": file_sha256(restricted_path),
                         "selected_candidate": public["selected_candidate"]})
    need(len(by_id) == len(rows) == 2000, "NPCR full OOF coverage differs")
    ordered = [by_id[row.source_id] for row in rows]
    truth = [row.raw_scores for row in rows]
    prediction = [[item["row_prediction"][axis] for axis in AXES] for item in ordered]
    baseline = [[r0[row.source_id][axis] for axis in AXES] for row in rows]
    metrics, exact_r0_metrics = compute_iterative_tail_metrics(truth, prediction), compute_iterative_tail_metrics(truth, baseline)
    comparison = metric_improvements(exact_r0_metrics, metrics)
    gate = global_promotion_gate(np.asarray(truth, dtype=float), np.asarray(baseline, dtype=float), np.asarray(prediction, dtype=float),
                                 [row.source_id for row in rows], config, seed=config.seed)
    result = {
        "schema_version": SCHEMA_VERSION, "status": "completed", "mode": "full", "run_id": config.run_id,
        "records": 2000, "folds": 5, "metrics": metrics, "exact_r0_metrics": exact_r0_metrics, "improvements_vs_exact_r0": comparison,
        "promotion_gate_vs_exact_r0": gate, "global_recommendation": "npcr" if gate["eligible"] else "exact_r0_identity",
        "fold_bindings": bindings, "candidate_inventory": [item.identifier for item in config.candidates],
        "config_sha256": config.config_sha256, "train_sha256": config.train_sha256,
        "embedding_manifest_sha256": config.embedding_manifest_sha256, "embedding_rows_sha256": config.embedding_rows_sha256,
        "r0_oof_prediction_sha256": config.r0_oof_prediction_sha256,
        "validation_rows_loaded": False, "average_target_used": False, "r0_score_feature_used": False,
        "privacy": "aggregate_only_no_ids_embeddings_or_row_predictions",
    }
    _validate_public_payload(result)
    aggregate_path = Path(config.output_root) / config.run_id / "aggregate.json"
    need(not aggregate_path.exists(), "stale NPCR aggregate output exists")
    _atomic_json(aggregate_path, result)
    return result


def run(config: NPCRConfig | str | Path, *, mode: str, outer_fold: int | None = None, device: str | None = None) -> dict[str, Any]:
    value = NPCRConfig.from_json(config) if isinstance(config, (str, Path)) else config
    need(mode in {"outer_fold", "full"}, "NPCR mode differs")
    if mode == "full":
        need(outer_fold is None, "full aggregation has no outer fold")
        return aggregate_full(value)
    need(isinstance(outer_fold, int), "outer fold is required")
    return run_outer_fold(value, outer_fold, device=device)


__all__ = [
    "NPCRConfig", "NPCRRow", "Pair", "PairSamplingSpec", "PromptReferenceNPCRError", "aggregate_full",
    "build_prompt_pairs", "build_utility_network", "file_sha256", "load_canonical_raw_rows", "load_rows",
    "ordinal_band", "outer_and_inner_indices", "recover_absolute_scores", "run", "run_outer_fold", "select_anchors",
    "train_utility_network", "utility_difference",
]
