"""Validation-free, OOF-only conservative calibration and combination.

Standard five-fold OOF predictions cannot identify a leakage-free learned
stacker for a second outer cross-fit: predictions on its four fit folds may
come from base models trained on the current held fold.  This implementation
therefore fails closed for learned fitting and evaluates one preregistered
rowwise candidate only: 0.8 exact-R0 identity + 0.2 Stage3 ``coral-natural``.
Exact R0 remains protected until a single post-OOF promotion gate passes.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .iterative_tail_metrics import AXES, compute_iterative_tail_metrics, metric_improvements, paired_bootstrap_delta_ci
from .r0_ordinal_residual import load_embedding_artifact


SCHEMA_VERSION = "mal2026-conservative-oof-combiner-v1"
CALIBRATION_STATUS = "unavailable_requires_genuinely_outer_nested_base_predictions"


class ConservativeCombinerError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ConservativeCombinerError(message)


def file_sha256(path: str | Path) -> str:
    value = Path(path)
    need(value.is_file() and not value.is_symlink(), "checksum input must be an ordinary file")
    digest = sha256()
    with value.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contains_validation(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_validation(key) or _contains_validation(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_validation(child) for child in value)
    return isinstance(value, str) and "validation" in value.lower()


@dataclass(frozen=True)
class FoldFile:
    outer_fold: int
    public_path: str
    public_sha256: str
    restricted_path: str
    restricted_sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FoldFile":
        need(set(raw) == {"outer_fold", "public_path", "public_sha256", "restricted_path", "restricted_sha256"}, "source fold fields differ")
        value = cls(int(raw["outer_fold"]), str(raw["public_path"]), str(raw["public_sha256"]),
                    str(raw["restricted_path"]), str(raw["restricted_sha256"]))
        need(0 <= value.outer_fold < 5 and len(value.public_sha256) == len(value.restricted_sha256) == 64, "source fold binding differs")
        return value


@dataclass(frozen=True)
class SourceSpec:
    identifier: str
    kind: str
    provenance: str
    upstream_run_id: str
    upstream_config_sha256: str
    upstream_outer_schema: str
    upstream_aggregate_schema: str
    upstream_method_id: str
    upstream_method_inventory: tuple[str, ...]
    aggregate_path: str
    aggregate_sha256: str
    fold_files: tuple[FoldFile, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SourceSpec":
        expected = {"id", "kind", "provenance", "upstream_run_id", "upstream_config_sha256", "upstream_outer_schema", "upstream_aggregate_schema",
                    "upstream_method_id", "upstream_method_inventory", "aggregate_path", "aggregate_sha256", "fold_files"}
        need(set(raw) == expected and isinstance(raw["fold_files"], list) and isinstance(raw["upstream_method_inventory"], list), "source fields differ")
        value = cls(str(raw["id"]), str(raw["kind"]), str(raw["provenance"]), str(raw["upstream_run_id"]),
                    str(raw["upstream_config_sha256"]), str(raw["upstream_outer_schema"]), str(raw["upstream_aggregate_schema"]), str(raw["upstream_method_id"]),
                    tuple(str(item) for item in raw["upstream_method_inventory"]),
                    str(raw["aggregate_path"]), str(raw["aggregate_sha256"]),
                    tuple(FoldFile.from_mapping(item) for item in raw["fold_files"]))
        need(value.kind in {"stage3_kure", "stage4_npcr"} and value.identifier not in {"", "exact_r0"}, "source identity differs")
        need(value.provenance == "standard_5fold_oof" and len(value.upstream_config_sha256) == len(value.aggregate_sha256) == 64,
             "only standard OOF provenance is accepted for fixed rowwise use")
        need(bool(value.upstream_method_inventory) and len(set(value.upstream_method_inventory)) == len(value.upstream_method_inventory),
             "upstream method inventory differs")
        need(tuple(item.outer_fold for item in value.fold_files) == tuple(range(5)), "source needs ordered five-fold files")
        return value


@dataclass(frozen=True)
class CombinerConfig:
    schema_version: str
    run_id: str
    train_path: str
    train_sha256: str
    fold_manifest_path: str
    fold_manifest_sha256: str
    fold_rows_path: str
    fold_rows_sha256: str
    r0_oof_prediction_path: str
    r0_oof_prediction_sha256: str
    sources: tuple[SourceSpec, ...]
    output_root: str
    restricted_output_root: str
    seed: int
    axes: tuple[str, ...]
    average_target_forbidden: bool
    combination_mode: str
    fixed_partner_source_id: str
    fixed_partner_method_id: str
    fixed_partner_weight: float
    calibration_status: str
    promotion_gate: Mapping[str, Any]
    config_sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, config_sha256: str | None = None) -> "CombinerConfig":
        need(isinstance(raw, Mapping) and not _contains_validation(raw), "validation content is forbidden")
        normalized = dict(raw)
        for key in ("sources", "axes"):
            need(isinstance(normalized.get(key), list), f"{key} must be a list")
        normalized["sources"] = tuple(SourceSpec.from_mapping(item) for item in normalized["sources"])
        normalized["axes"] = tuple(normalized["axes"])
        normalized["config_sha256"] = config_sha256 or sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        need(set(normalized) == set(cls.__dataclass_fields__), "combiner config fields differ")
        value = cls(**normalized); value.validate(); return value

    @classmethod
    def from_json(cls, path: str | Path) -> "CombinerConfig":
        location = Path(path)
        try:
            raw = json.loads(location.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConservativeCombinerError("combiner config is unreadable") from exc
        return cls.from_mapping(raw, config_sha256=file_sha256(location))

    def validate(self) -> None:
        need(self.schema_version == SCHEMA_VERSION and bool(self.run_id), "combiner identity differs")
        need(self.axes == AXES and self.average_target_forbidden is True, "axis/average contract differs")
        need(self.combination_mode == "preregistered_fixed_standard_oof"
             and self.fixed_partner_method_id == "coral-natural" and self.fixed_partner_weight == 0.2
             and self.calibration_status == CALIBRATION_STATUS, "fixed candidate contract differs")
        need(bool(self.sources) and len({item.identifier for item in self.sources}) == len(self.sources), "source inventory differs")
        partner = next((item for item in self.sources if item.identifier == self.fixed_partner_source_id), None)
        need(partner is not None and partner.kind == "stage3_kure" and partner.upstream_method_id == "coral-natural",
             "fixed Stage3 coral partner is absent")
        need(Path(self.output_root).resolve() != Path(self.restricted_output_root).resolve(), "public/restricted roots must differ")
        for digest in (self.train_sha256, self.fold_manifest_sha256, self.fold_rows_sha256,
                       self.r0_oof_prediction_sha256, self.config_sha256):
            need(len(digest) == 64, "checksum format differs")
        gate = self.promotion_gate
        expected = {"minimum_macro_rmse_improvement", "maximum_axis_rmse_worsening",
                    "maximum_gold_3_4_balanced_accuracy_drop", "maximum_spearman_drop", "low_tail_noninferior",
                    "high_tail_noninferior", "paired_bootstrap_resamples",
                    "paired_bootstrap_lower_bound_above_zero"}
        need(set(gate) == expected and float(gate["minimum_macro_rmse_improvement"]) == 0.005
             and float(gate["maximum_axis_rmse_worsening"]) == 0.01
             and float(gate["maximum_gold_3_4_balanced_accuracy_drop"]) == 0.01
             and float(gate["maximum_spearman_drop"]) == 0.005
             and gate["low_tail_noninferior"] is True and gate["high_tail_noninferior"] is True
             and int(gate["paired_bootstrap_resamples"]) == 10000
             and gate["paired_bootstrap_lower_bound_above_zero"] is True, "promotion gate differs")


def fit_outer_combiner(*_: Any, **__: Any) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Fail closed: learned fitting is unidentified from standard five-fold OOF."""
    raise ConservativeCombinerError(CALIBRATION_STATUS)


def fixed_candidate(r0: np.ndarray, partner: np.ndarray, weight: float = 0.2) -> np.ndarray:
    need(r0.shape == partner.shape and r0.ndim == 2 and r0.shape[1] == 3 and weight == 0.2,
         "fixed candidate tensors/weight differ")
    need(np.isfinite(r0).all() and np.isfinite(partner).all(), "fixed candidate is non-finite")
    return np.clip(0.8 * r0 + 0.2 * partner, 1.0, 5.0)


def _load_train(path: str, digest: str) -> Mapping[str, tuple[float, float, float]]:
    need(file_sha256(path) == digest, "train binding differs")
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line); score = raw["score"]
            need(set(raw) == {"id", "document_id", "prompt_num", "prompt", "essay", "score"}
                 and isinstance(score, Mapping) and set(score) == {*AXES, "average"}, "train schema differs")
            need(all(type(score[axis]) in {int, float} and not isinstance(score[axis], bool) for axis in AXES), "train axis type differs")
            values = tuple(float(score[axis]) for axis in AXES)
            need(isinstance(raw["id"], str) and raw["id"] not in result
                 and all(math.isfinite(item) and 1 <= item <= 5 for item in values), "train row differs")
            result[raw["id"]] = values
    need(len(result) == 2000, "train population differs")
    return result


def _load_source_fold(spec: SourceSpec, binding: FoldFile, expected_ids: set[str], raw_gold: Mapping[str, tuple[float, float, float]]) -> Mapping[str, tuple[float, float, float]]:
    need(file_sha256(binding.public_path) == binding.public_sha256
         and file_sha256(binding.restricted_path) == binding.restricted_sha256
         and len(expected_ids) == 400, "source fold checksum/population differs")
    public = json.loads(Path(binding.public_path).read_text(encoding="utf-8"))
    need(public.get("schema_version") == spec.upstream_outer_schema and public.get("status") == "completed"
         and public.get("mode") == "outer_fold" and public.get("run_id") == spec.upstream_run_id
         and public.get("config_sha256") == spec.upstream_config_sha256
         and public.get("outer_fold") == binding.outer_fold and public.get("records") == 400,
         "upstream outer report binding differs")
    if spec.kind == "stage3_kure":
        methods = public.get("methods")
        need(isinstance(methods, list) and tuple(item.get("method") for item in methods) == spec.upstream_method_inventory,
             "upstream Stage3 method inventory differs")
        method = next((item for item in methods if item.get("method") == spec.upstream_method_id), None) if isinstance(methods, list) else None
        need(method is not None and method.get("restricted_prediction_sha256") == binding.restricted_sha256,
             "upstream Stage3 method/restricted binding differs")
    else:
        need(tuple(public.get("candidate_inventory", ())) == spec.upstream_method_inventory
             and public.get("selected_candidate") in set(spec.upstream_method_inventory)
             and public.get("restricted_predictions_sha256") == binding.restricted_sha256,
             "upstream Stage4 selected/restricted binding differs")
    result = {}
    with Path(binding.restricted_path).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            expected_keys = {"source_id", "outer_fold", "prediction"} if spec.kind == "stage3_kure" else {"source_id", "outer_fold", "raw_gold", "row_prediction"}
            need(isinstance(item, Mapping) and set(item) == expected_keys, "source row schema differs")
            source_id = item["source_id"]
            need(source_id in expected_ids and source_id not in result and item["outer_fold"] == binding.outer_fold, "source ID/fold differs")
            field = "prediction" if spec.kind == "stage3_kure" else "row_prediction"
            value = item[field]
            need(isinstance(value, Mapping) and set(value) == set(AXES), "source prediction axes differ")
            need(all(type(value[axis]) in {int, float} and not isinstance(value[axis], bool) for axis in AXES), "source prediction type differs")
            prediction = tuple(float(value[axis]) for axis in AXES)
            need(all(math.isfinite(number) and 1 <= number <= 5 for number in prediction), "source prediction differs")
            if spec.kind == "stage4_npcr":
                gold = item["raw_gold"]
                need(isinstance(gold, Mapping) and set(gold) == set(AXES)
                     and all(abs(float(gold[axis]) - raw_gold[source_id][index]) <= 1e-9 for index, axis in enumerate(AXES)), "source raw gold differs")
            result[source_id] = prediction
    need(set(result) == expected_ids, "source fold coverage differs")
    return result


def _validate_upstream_aggregate(spec: SourceSpec) -> None:
    need(file_sha256(spec.aggregate_path) == spec.aggregate_sha256, "upstream aggregate checksum differs")
    aggregate = json.loads(Path(spec.aggregate_path).read_text(encoding="utf-8"))
    need(aggregate.get("schema_version") == spec.upstream_aggregate_schema and aggregate.get("status") == "completed"
         and aggregate.get("run_id") == spec.upstream_run_id and aggregate.get("config_sha256") == spec.upstream_config_sha256
         and aggregate.get("records") == 2000 and aggregate.get("folds") == 5
         and aggregate.get("validation_rows_loaded") is False and aggregate.get("average_target_used") is False,
         "upstream aggregate binding differs")
    if spec.kind == "stage3_kure":
        methods = aggregate.get("methods")
        need(isinstance(methods, list) and tuple(item.get("method") for item in methods) == spec.upstream_method_inventory
             and spec.upstream_method_id in spec.upstream_method_inventory,
             "upstream aggregate method inventory differs")
        method = next(item for item in methods if item.get("method") == spec.upstream_method_id)
        bindings = method.get("fold_bindings")
        need(isinstance(bindings, list) and len(bindings) == len(spec.fold_files)
             and all(binding.get("outer_fold") == configured.outer_fold
                     and binding.get("public_sha256") == configured.public_sha256
                     and binding.get("restricted_prediction_sha256") == configured.restricted_sha256
                     for binding, configured in zip(bindings, spec.fold_files)),
             "upstream aggregate Stage3 fold bindings differ")
    else:
        need(spec.upstream_method_id == "selected_npcr_process"
             and tuple(aggregate.get("candidate_inventory", ())) == spec.upstream_method_inventory,
             "upstream aggregate NPCR inventory differs")


def load_inputs(config: CombinerConfig) -> tuple[list[str], np.ndarray, np.ndarray, Mapping[str, np.ndarray]]:
    config.validate(); raw = _load_train(config.train_path, config.train_sha256)
    need(file_sha256(config.fold_manifest_path) == config.fold_manifest_sha256
         and file_sha256(config.fold_rows_path) == config.fold_rows_sha256, "fold artifact binding differs")
    manifest, embedded = load_embedding_artifact(config.fold_manifest_path, config.fold_rows_path)
    need(manifest.split_role == "train" and manifest.base_prediction_origin == "oof" and not manifest.contains_average_target, "fold artifact contract differs")
    ids = [row.source_id for row in embedded]
    need(set(ids) == set(raw) and len(ids) == 2000, "fold/train population differs")
    folds = np.asarray([row.oof_fold for row in embedded], dtype=int)
    truth = np.asarray([raw[source_id] for source_id in ids], dtype=float)
    need(file_sha256(config.r0_oof_prediction_path) == config.r0_oof_prediction_sha256, "R0 checksum differs")
    r0_by_id = {}
    by_id = {row.source_id: row for row in embedded}
    with Path(config.r0_oof_prediction_path).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line); source_id = item["source_id"]
            need(set(item) == {"source_id", "fold", "continuous_prediction", "half_up_integer_prediction", "reference_score"}
                 and source_id in by_id and source_id not in r0_by_id and item["fold"] == by_id[source_id].oof_fold, "R0 row/fold differs")
            need(isinstance(item["continuous_prediction"], Mapping) and isinstance(item["reference_score"], Mapping)
                 and set(item["continuous_prediction"]) == set(item["reference_score"]) == set(AXES)
                 and all(type(item["continuous_prediction"][axis]) in {int, float}
                         and type(item["reference_score"][axis]) in {int, float} for axis in AXES), "R0 axis schema differs")
            pred = tuple(float(item["continuous_prediction"][axis]) for axis in AXES)
            ref = tuple(float(item["reference_score"][axis]) for axis in AXES)
            need(pred == by_id[source_id].base_predictions and all(abs(ref[i] - raw[source_id][i]) <= 1e-9 for i in range(3)), "R0 axis/raw binding differs")
            r0_by_id[source_id] = pred
    need(set(r0_by_id) == set(ids), "R0 coverage differs")
    predictions: dict[str, np.ndarray] = {"exact_r0": np.asarray([r0_by_id[source_id] for source_id in ids])}
    for spec in config.sources:
        _validate_upstream_aggregate(spec)
        joined = {}
        for binding in spec.fold_files:
            expected = {ids[index] for index in np.flatnonzero(folds == binding.outer_fold)}
            joined.update(_load_source_fold(spec, binding, expected, raw))
        need(set(joined) == set(ids), "source full coverage differs")
        predictions[spec.identifier] = np.asarray([joined[source_id] for source_id in ids])
    return ids, truth, folds, predictions


def promotion_gate(truth: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, ids: Sequence[str], config: CombinerConfig) -> Mapping[str, Any]:
    base, cand = compute_iterative_tail_metrics(truth, baseline), compute_iterative_tail_metrics(truth, candidate)
    delta = metric_improvements(base, cand); gate = config.promotion_gate
    bootstrap = paired_bootstrap_delta_ci(truth, baseline, candidate, document_ids=ids,
                                          n_resamples=int(gate["paired_bootstrap_resamples"]), seed=config.seed)
    tail_support = all((base["axes"][axis]["bands"]["1"]["count"] + base["axes"][axis]["bands"]["2"]["count"] > 0)
                       and base["axes"][axis]["bands"]["5"]["count"] > 0 for axis in AXES)
    gates = {"macro_rmse": delta["rmse"] is not None and delta["rmse"] >= float(gate["minimum_macro_rmse_improvement"]),
             "axis_rmse": all(value >= -float(gate["maximum_axis_rmse_worsening"]) for value in delta["axis_rmse"].values()),
             "gold_3_4_ba": delta["gold_3_4_balanced_accuracy"] is not None and delta["gold_3_4_balanced_accuracy"] >= -float(gate["maximum_gold_3_4_balanced_accuracy_drop"]),
             "spearman": delta["spearman"] is not None and delta["spearman"] >= -float(gate["maximum_spearman_drop"]),
             "low_tail": delta["low_tail_rmse"] is not None and delta["low_tail_rmse"] >= 0,
             "high_tail": delta["high_tail_rmse"] is not None and delta["high_tail_rmse"] >= 0,
             "tail_support_every_axis": tail_support,
             "bootstrap": bootstrap["intervals"]["rmse"]["lower"] is not None and bootstrap["intervals"]["rmse"]["lower"] > 0}
    return {"eligible": all(gates.values()), "gates": gates, "improvements": delta, "paired_bootstrap": bootstrap,
            "exact_r0_metrics": base, "candidate_metrics": cand}


def _private_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    need(not path.exists(), "refusing to overwrite restricted combiner output")
    path.parent.mkdir(parents=True, exist_ok=True); os.chmod(path.parent, 0o770)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows: stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary, 0o660); os.replace(temporary, path); os.chmod(path, 0o660)
    except Exception:
        Path(temporary).unlink(missing_ok=True); raise
    need(path.stat().st_mode & 0o007 == 0 and path.stat().st_mode & 0o111 == 0, "restricted output permissions differ")
    return file_sha256(path)


def _public_json(path: Path, value: Mapping[str, Any]) -> str:
    need(not path.exists(), "refusing to overwrite public combiner output")
    _validate_public(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path); return file_sha256(path)


def _validate_public(value: Any) -> None:
    forbidden = {"source_id", "raw_gold", "candidate_prediction", "protected_prediction",
                 "exact_r0_prediction", "essay", "prompt", "embedding"}
    if isinstance(value, Mapping):
        need(not (set(value) & forbidden), "restricted row material entered public output")
        for child in value.values(): _validate_public(child)
    elif isinstance(value, (list, tuple)):
        for child in value: _validate_public(child)


def run(config: CombinerConfig | str | Path, *, validate_only: bool = False) -> Mapping[str, Any]:
    value = CombinerConfig.from_json(config) if isinstance(config, (str, Path)) else config
    ids, truth, folds, predictions = load_inputs(value)
    if validate_only:
        return {"status": "validated", "records": len(ids), "sources": list(predictions),
                "validation_rows_loaded": False, "average_target_used": False}
    partner = predictions[value.fixed_partner_source_id]
    candidate = fixed_candidate(predictions["exact_r0"], partner, value.fixed_partner_weight)
    gate = promotion_gate(truth, predictions["exact_r0"], candidate, ids, value)
    protected = candidate if gate["eligible"] else predictions["exact_r0"]
    restricted = Path(value.restricted_output_root) / value.run_id / "oof_predictions.jsonl"
    restricted_sha = _private_jsonl(restricted, ({"source_id": source_id, "outer_fold": int(folds[index]),
        "raw_gold": {axis: float(truth[index, j]) for j, axis in enumerate(AXES)},
        "exact_r0_prediction": {axis: float(predictions["exact_r0"][index, j]) for j, axis in enumerate(AXES)},
        "candidate_prediction": {axis: float(candidate[index, j]) for j, axis in enumerate(AXES)},
        "protected_prediction": {axis: float(protected[index, j]) for j, axis in enumerate(AXES)}} for index, source_id in enumerate(ids)))
    result = {"schema_version": SCHEMA_VERSION, "status": "completed", "mode": "full_oof",
              "run_id": value.run_id, "records": 2000, "members_max": 2, "nonnegative_sum_to_one": True,
              "combination_mode": value.combination_mode,
              "fixed_candidate": {"r0_weight": 0.8, "partner_weight": 0.2,
                                  "partner_source_id": value.fixed_partner_source_id,
                                  "partner_method_id": value.fixed_partner_method_id,
                                  "calibration": "identity"},
              "calibration_status": value.calibration_status,
              "selection_optimism": "fixed_before_upstream_results; no learned source, calibrator, or weight selection from standard OOF",
              "promotion_gate": gate,
              "protected_output": "candidate" if gate["eligible"] else "exact_r0_identity",
              "restricted_predictions_sha256": restricted_sha, "config_sha256": value.config_sha256,
              "train_sha256": value.train_sha256, "fold_rows_sha256": value.fold_rows_sha256,
              "fold_manifest_sha256": value.fold_manifest_sha256,
              "r0_oof_prediction_sha256": value.r0_oof_prediction_sha256,
              "source_bindings": [{"id": source.identifier, "kind": source.kind, "provenance": source.provenance,
                                   "upstream_run_id": source.upstream_run_id,
                                   "upstream_config_sha256": source.upstream_config_sha256,
                                   "upstream_method_id": source.upstream_method_id,
                                   "upstream_method_inventory": list(source.upstream_method_inventory),
                                   "aggregate_sha256": source.aggregate_sha256,
                                   "folds": [{"outer_fold": item.outer_fold, "public_sha256": item.public_sha256,
                                              "restricted_sha256": item.restricted_sha256} for item in source.fold_files]}
                                  for source in value.sources],
              "validation_rows_loaded": False, "average_target_used": False,
              "privacy": "aggregate_only_public_row_predictions_restricted"}
    _public_json(Path(value.output_root) / value.run_id / "aggregate.json", result)
    return result


__all__ = ["CALIBRATION_STATUS", "CombinerConfig", "ConservativeCombinerError", "FoldFile",
           "SourceSpec", "fit_outer_combiner", "fixed_candidate", "load_inputs",
           "promotion_gate", "run"]
