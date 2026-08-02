"""Post-result, preregistered promotion gate for Stage3 ``coral-natural``.

The postprocessor reads checksum-bound restricted OOF rows but emits aggregates
only.  It is deliberately unable to select another Stage3 method: RPS remains
descriptive and is never eligible for promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .iterative_tail_metrics import AXES, compute_iterative_tail_metrics, metric_improvements, paired_bootstrap_delta_ci
from .kure_ordinal_oof import KUREOrdinalOOFConfig, config_sha256, load_recommended_methods


SCHEMA_VERSION = "mal2026-stage3-coral-promotion-v1"
STAGE6_PREREG_PATH = "configs/stage6_submission_prereg.v1.json"
STAGE6_PREREG_SHA256 = "7616e038dd0dcb8a10a15c09780ca178ff43700c132fa941ba4e050e2a8176e1"
STAGE6_PREREG_COMMIT = "32b0a43eda5612284d5bd718c5afbce2be182eff"
STAGE3_CONFIG_PATH = "configs/kure_ordinal_oof.v1.json"
STAGE3_CONFIG_FILE_SHA256 = "5108e37c490482fbbf45c043f2ad4d0154048d432b4a4a163b3d102f8c6d756e"
STAGE3_RUN_ID = "kure-ordinal-oof-v1-20260803-001"
CORAL_METHOD = "coral-natural"


class Stage3CoralPromotionError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Stage3CoralPromotionError(message)


def file_sha256(path: str | Path) -> str:
    location = Path(path)
    need(location.is_file() and not location.is_symlink(), "checksum input must be an ordinary file")
    digest = sha256()
    with location.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class OuterBinding:
    outer_fold: int
    public_path: str
    public_sha256: str
    coral_restricted_path: str
    coral_restricted_sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OuterBinding":
        expected = {"outer_fold", "public_path", "public_sha256", "coral_restricted_path", "coral_restricted_sha256"}
        need(isinstance(raw, Mapping) and set(raw) == expected, "outer binding fields differ")
        value = cls(int(raw["outer_fold"]), str(raw["public_path"]), str(raw["public_sha256"]),
                    str(raw["coral_restricted_path"]), str(raw["coral_restricted_sha256"]))
        need(0 <= value.outer_fold < 5
             and len(value.public_sha256) == len(value.coral_restricted_sha256) == 64,
             "outer binding differs")
        return value


@dataclass(frozen=True)
class PromotionConfig:
    schema_version: str
    run_id: str
    stage6_preregistration_path: str
    stage6_preregistration_sha256: str
    stage6_preregistration_commit: str
    stage3_config_path: str
    stage3_config_file_sha256: str
    stage3_report_config_sha256: str
    stage3_aggregate_path: str
    stage3_aggregate_sha256: str
    outer_bindings: tuple[OuterBinding, ...]
    output_path: str
    config_sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, digest: str | None = None) -> "PromotionConfig":
        need(isinstance(raw, Mapping) and "validation" not in json.dumps(raw, sort_keys=True).lower(),
             "validation paths/content are forbidden")
        normalized = dict(raw)
        need(isinstance(normalized.get("outer_bindings"), list), "outer_bindings must be a list")
        normalized["outer_bindings"] = tuple(OuterBinding.from_mapping(item) for item in normalized["outer_bindings"])
        normalized["config_sha256"] = digest or sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        need(set(normalized) == set(cls.__dataclass_fields__), "promotion config fields differ")
        value = cls(**normalized)
        value.validate()
        return value

    @classmethod
    def from_json(cls, path: str | Path) -> "PromotionConfig":
        location = Path(path)
        try:
            raw = json.loads(location.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Stage3CoralPromotionError("promotion config is unreadable") from exc
        return cls.from_mapping(raw, digest=file_sha256(location))

    def validate(self) -> None:
        need(self.schema_version == SCHEMA_VERSION and bool(self.run_id), "promotion identity differs")
        need(self.stage6_preregistration_path == STAGE6_PREREG_PATH
             and self.stage6_preregistration_sha256 == STAGE6_PREREG_SHA256
             and self.stage6_preregistration_commit == STAGE6_PREREG_COMMIT,
             "Stage6 preregistration binding differs")
        need(self.stage3_config_path == STAGE3_CONFIG_PATH
             and self.stage3_config_file_sha256 == STAGE3_CONFIG_FILE_SHA256,
             "Stage3 config file binding differs")
        need(len(self.stage3_report_config_sha256) == len(self.stage3_aggregate_sha256) == len(self.config_sha256) == 64,
             "checksum format differs")
        need(tuple(item.outer_fold for item in self.outer_bindings) == tuple(range(5)),
             "exact ordered five-fold bindings are required")
        _load_bound_contract(self)


def _load_bound_contract(config: PromotionConfig) -> tuple[Mapping[str, Any], KUREOrdinalOOFConfig]:
    need(file_sha256(config.stage6_preregistration_path) == STAGE6_PREREG_SHA256,
         "Stage6 preregistration checksum differs")
    prereg = json.loads(Path(config.stage6_preregistration_path).read_text(encoding="utf-8"))
    gate = prereg.get("common_stage3_promotion_gate")
    need(prereg.get("schema_version") == "mal2026-stage6-submission-prereg-v1"
         and prereg.get("status") == "preregistered_non_runnable_pending_upstream_artifacts"
         and prereg.get("upstream_preregistration", {}).get("stage3", {}).get("run_id") == STAGE3_RUN_ID
         and prereg.get("upstream_preregistration", {}).get("stage3", {}).get("config_file_sha256") == STAGE3_CONFIG_FILE_SHA256
         and prereg.get("upstream_preregistration", {}).get("stage3", {}).get("allowed_h1_method_id") == CORAL_METHOD
         and prereg.get("upstream_preregistration", {}).get("stage3", {}).get("rps_role")
             == "descriptive_only_not_eligible_for_H1_H2_or_full_refit",
         "Stage6 Stage3 preregistration contract differs")
    need(gate == {
        "reference": "H0_exact_R0_continuous_train_OOF", "candidate": "stage3_coral-natural_only", "operator": "AND",
        "macro_rmse_min_improvement": 0.005, "maximum_axis_rmse_worsening": 0.01,
        "maximum_gold_3_4_balanced_accuracy_drop": 0.01, "maximum_macro_spearman_drop": 0.005,
        "low_tail_1_2_noninferior": True, "high_tail_5_noninferior": True,
        "require_nonzero_low_1_2_and_high_5_support_every_axis": True,
        "require_all_five_outer_folds": True, "require_finite_metrics": True,
        "paired_bootstrap": {"resamples": 10000, "unit": "source_id_with_all_three_axes_clustered",
                             "improvement_quantity": "H0_macro_RMSE_minus_candidate_macro_RMSE",
                             "required_lower_bound_strictly_gt": 0.0}},
         "Stage6 promotion gate differs")

    need(file_sha256(config.stage3_config_path) == STAGE3_CONFIG_FILE_SHA256,
         "Stage3 config checksum differs")
    stage3 = KUREOrdinalOOFConfig.from_json(config.stage3_config_path, require_dependencies=True)
    need(stage3.run_id == STAGE3_RUN_ID and config.stage3_report_config_sha256 == config_sha256(stage3),
         "Stage3 run/report-config binding differs")
    for binding in config.outer_bindings:
        need(binding.public_path == str(Path(stage3.output_root) / f"outer-{binding.outer_fold:02d}.json")
             and binding.coral_restricted_path == str(Path(stage3.restricted_output_root) / f"outer-{binding.outer_fold:02d}"
                                                       / CORAL_METHOD / "predictions.jsonl"),
             "Stage3 deterministic outer paths differ")
    need(config.stage3_aggregate_path == str(Path(stage3.output_root) / "aggregate.json"),
         "Stage3 deterministic aggregate path differs")
    return prereg, stage3


def _load_canonical(stage3: KUREOrdinalOOFConfig) -> tuple[
    list[str], Mapping[str, tuple[float, float, float]], Mapping[str, str], Mapping[str, tuple[float, float, float]]
]:
    need(file_sha256(stage3.train_path) == stage3.train_sha256, "canonical train checksum differs")
    ordered: list[str] = []
    truth: dict[str, tuple[float, float, float]] = {}
    with Path(stage3.train_path).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            need(set(item) == {"id", "document_id", "prompt_num", "prompt", "essay", "score"}, "canonical row schema differs")
            source_id, scores = item["id"], item["score"]
            need(isinstance(source_id, str) and source_id not in truth and isinstance(scores, Mapping)
                 and set(scores) == {*AXES, "average"}, "canonical ID/axis schema differs")
            values = tuple(float(scores[axis]) for axis in AXES)
            need(all(math.isfinite(value) and 1.0 <= value <= 5.0 for value in values), "canonical raw axis differs")
            ordered.append(source_id); truth[source_id] = values  # type: ignore[assignment]
    need(len(ordered) == 2000, "canonical population differs")

    need(file_sha256(stage3.fold_manifest_path) == stage3.fold_manifest_sha256
         and file_sha256(stage3.fold_rows_path) == stage3.fold_rows_sha256,
         "fold artifact checksum differs")
    manifest = json.loads(Path(stage3.fold_manifest_path).read_text(encoding="utf-8"))
    need(manifest.get("schema_version") == "r0_ordinal_residual_embedding_v1"
         and manifest.get("split_role") == "train" and manifest.get("base_prediction_origin") == "oof"
         and manifest.get("fold_count") == 5 and manifest.get("contains_average_target") is False
         and manifest.get("rows_sha256") == stage3.fold_rows_sha256,
         "fold manifest contract differs")
    folds: dict[str, int] = {}
    fold_gold: dict[str, tuple[float, float, float]] = {}
    fold_base: dict[str, tuple[float, float, float]] = {}
    with Path(stage3.fold_rows_path).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            need(set(item) == {"source_id", "group_id", "shared_embedding", "base_continuous_prediction",
                               "raw_continuous_gold", "oof_fold"}, "fold row schema differs")
            source_id = item["source_id"]
            need(source_id in truth and source_id not in folds and item["oof_fold"] in range(5), "fold row ID/fold differs")
            for field in ("base_continuous_prediction", "raw_continuous_gold"):
                need(isinstance(item[field], Mapping) and set(item[field]) == set(AXES), "fold row axes differ")
            folds[source_id] = int(item["oof_fold"])
            fold_gold[source_id] = tuple(float(item["raw_continuous_gold"][axis]) for axis in AXES)  # type: ignore[assignment]
            fold_base[source_id] = tuple(float(item["base_continuous_prediction"][axis]) for axis in AXES)  # type: ignore[assignment]
    need(set(folds) == set(truth) and all(sum(value == fold for value in folds.values()) == 400 for fold in range(5))
         and all(fold_gold[source_id] == truth[source_id] for source_id in truth), "fold population/raw binding differs")

    need(file_sha256(stage3.r0_oof_prediction_path) == stage3.r0_oof_prediction_sha256, "exact R0 checksum differs")
    r0: dict[str, tuple[float, float, float]] = {}
    with Path(stage3.r0_oof_prediction_path).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line); source_id = item.get("source_id")
            need(set(item) == {"source_id", "fold", "continuous_prediction", "half_up_integer_prediction", "reference_score"}
                 and source_id in folds and source_id not in r0 and item["fold"] == folds[source_id], "exact R0 row/fold schema differs")
            need(all(isinstance(item[field], Mapping) and set(item[field]) == set(AXES)
                     for field in ("continuous_prediction", "half_up_integer_prediction", "reference_score")),
                 "exact R0 axes differ")
            values = tuple(float(item["continuous_prediction"][axis]) for axis in AXES)
            references = tuple(float(item["reference_score"][axis]) for axis in AXES)
            need(values == fold_base[source_id] and references == truth[source_id], "exact R0 prediction/raw binding differs")
            r0[source_id] = values  # type: ignore[assignment]
    need(set(r0) == set(truth), "exact R0 population differs")
    return ordered, truth, {source_id: str(folds[source_id]) for source_id in ordered}, r0  # type: ignore[return-value]


def _load_coral_fold(binding: OuterBinding, stage3: KUREOrdinalOOFConfig, expected_ids: set[str],
                     method_inventory: Sequence[str]) -> tuple[Mapping[str, tuple[float, float, float]], Mapping[str, Any]]:
    need(file_sha256(binding.public_path) == binding.public_sha256
         and file_sha256(binding.coral_restricted_path) == binding.coral_restricted_sha256,
         "outer public/restricted checksum differs")
    public = json.loads(Path(binding.public_path).read_text(encoding="utf-8"))
    need(public.get("schema_version") == "mal2026-kure-ordinal-oof-outer-v1"
         and public.get("status") == "completed" and public.get("mode") == "outer_fold"
         and public.get("run_id") == stage3.run_id and public.get("config_sha256") == config_sha256(stage3)
         and public.get("outer_fold") == binding.outer_fold and public.get("records") == 400
         and public.get("fold_manifest_sha256") == stage3.fold_manifest_sha256
         and public.get("fold_rows_sha256") == stage3.fold_rows_sha256
         and public.get("validation_rows_loaded") is False and public.get("average_target_used") is False,
         "outer report binding differs")
    methods = public.get("methods")
    need(isinstance(methods, list) and tuple(item.get("method") for item in methods) == tuple(method_inventory),
         "outer method inventory differs")
    coral = next((item for item in methods if item.get("method") == CORAL_METHOD), None)
    need(coral is not None and coral.get("family") == "coral"
         and coral.get("restricted_prediction_sha256") == binding.coral_restricted_sha256
         and isinstance(coral.get("axis_bindings"), list)
         and [item.get("axis") for item in coral["axis_bindings"]] == list(AXES),
         "outer coral method/axis binding differs")
    predictions: dict[str, tuple[float, float, float]] = {}
    with Path(binding.coral_restricted_path).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            need(set(item) == {"source_id", "outer_fold", "prediction"}, "coral restricted row schema differs")
            source_id, prediction = item["source_id"], item["prediction"]
            need(source_id in expected_ids and source_id not in predictions and item["outer_fold"] == binding.outer_fold
                 and isinstance(prediction, Mapping) and set(prediction) == set(AXES), "coral restricted ID/fold/axis differs")
            values = tuple(float(prediction[axis]) for axis in AXES)
            need(all(math.isfinite(value) and 1.0 <= value <= 5.0 for value in values), "coral prediction value differs")
            predictions[source_id] = values  # type: ignore[assignment]
    need(set(predictions) == expected_ids and len(predictions) == 400, "coral restricted population differs")
    return predictions, public


def _validate_aggregate(config: PromotionConfig, stage3: KUREOrdinalOOFConfig, method_inventory: Sequence[str]) -> Mapping[str, Any]:
    need(file_sha256(config.stage3_aggregate_path) == config.stage3_aggregate_sha256, "Stage3 aggregate checksum differs")
    aggregate = json.loads(Path(config.stage3_aggregate_path).read_text(encoding="utf-8"))
    need(aggregate.get("schema_version") == "mal2026-kure-ordinal-oof-aggregate-v1"
         and aggregate.get("status") == "completed" and aggregate.get("run_id") == stage3.run_id
         and aggregate.get("config_sha256") == config_sha256(stage3)
         and aggregate.get("records") == 2000 and aggregate.get("folds") == 5
         and aggregate.get("fold_manifest_sha256") == stage3.fold_manifest_sha256
         and aggregate.get("fold_rows_sha256") == stage3.fold_rows_sha256
         and aggregate.get("r0_oof_prediction_sha256") == stage3.r0_oof_prediction_sha256
         and aggregate.get("protected_output") == "exact_r0"
         and aggregate.get("validation_rows_loaded") is False and aggregate.get("average_target_used") is False,
         "Stage3 aggregate binding differs")
    methods = aggregate.get("methods")
    need(isinstance(methods, list) and tuple(item.get("method") for item in methods) == tuple(method_inventory),
         "Stage3 aggregate method inventory differs")
    coral = next((item for item in methods if item.get("method") == CORAL_METHOD), None)
    need(coral is not None and coral.get("family") == "coral", "Stage3 aggregate coral method differs")
    bindings = coral.get("fold_bindings")
    need(isinstance(bindings, list) and len(bindings) == 5
         and all(item.get("outer_fold") == configured.outer_fold
                 and item.get("public_sha256") == configured.public_sha256
                 and item.get("restricted_prediction_sha256") == configured.coral_restricted_sha256
                 for item, configured in zip(bindings, config.outer_bindings)),
         "Stage3 aggregate coral fold bindings differ")
    return aggregate


def promotion_gate(truth: np.ndarray, baseline: np.ndarray, coral: np.ndarray, source_ids: Sequence[str], *, seed: int) -> Mapping[str, Any]:
    baseline_metrics = compute_iterative_tail_metrics(truth, baseline)
    coral_metrics = compute_iterative_tail_metrics(truth, coral)
    improvements = metric_improvements(baseline_metrics, coral_metrics)
    bootstrap = dict(paired_bootstrap_delta_ci(truth, baseline, coral, document_ids=source_ids,
                                               n_resamples=10000, seed=seed))
    bootstrap["unit"] = "source_id_with_all_three_axes_clustered"
    support = all(baseline_metrics["axes"][axis]["bands"]["1"]["count"]
                  + baseline_metrics["axes"][axis]["bands"]["2"]["count"] > 0
                  and baseline_metrics["axes"][axis]["bands"]["5"]["count"] > 0 for axis in AXES)
    finite = all(value is not None and math.isfinite(float(value)) for value in (
        improvements["rmse"], improvements["gold_3_4_balanced_accuracy"], improvements["spearman"],
        improvements["low_tail_rmse"], improvements["high_tail_rmse"], *improvements["axis_rmse"].values()))
    gates = {
        "macro_rmse": improvements["rmse"] is not None and improvements["rmse"] >= 0.005,
        "axis_rmse": all(value >= -0.01 for value in improvements["axis_rmse"].values()),
        "gold_3_4_balanced_accuracy": improvements["gold_3_4_balanced_accuracy"] is not None
                                       and improvements["gold_3_4_balanced_accuracy"] >= -0.01,
        "macro_spearman": improvements["spearman"] is not None and improvements["spearman"] >= -0.005,
        "low_tail": improvements["low_tail_rmse"] is not None and improvements["low_tail_rmse"] >= 0.0,
        "high_tail": improvements["high_tail_rmse"] is not None and improvements["high_tail_rmse"] >= 0.0,
        "tail_support_every_axis": support,
        "all_five_outer_folds": True,
        "finite_metrics": finite,
        "paired_bootstrap": bootstrap["intervals"]["rmse"]["lower"] is not None
                            and bootstrap["intervals"]["rmse"]["lower"] > 0.0,
    }
    return {"eligible": all(gates.values()), "gates": gates, "improvements": improvements,
            "paired_bootstrap": bootstrap, "exact_r0_metrics": baseline_metrics, "coral_natural_metrics": coral_metrics}


def _write_public_fresh(path: Path, payload: Mapping[str, Any]) -> None:
    need(not path.exists(), "refusing to overwrite promotion output")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def run(config: PromotionConfig | str | Path, *, validate_only: bool = False) -> Mapping[str, Any]:
    value = PromotionConfig.from_json(config) if isinstance(config, (str, Path)) else config
    value.validate()
    _, stage3 = _load_bound_contract(value)
    methods = load_recommended_methods(stage3)
    method_inventory = tuple(method.identifier for method in methods)
    need(CORAL_METHOD in method_inventory and any(method.startswith("rps-") for method in method_inventory),
         "Stage3 coral/RPS inventory differs")
    ordered_ids, truth_by_id, folds_as_text, r0_by_id = _load_canonical(stage3)
    coral_by_id: dict[str, tuple[float, float, float]] = {}
    for binding in value.outer_bindings:
        expected_ids = {source_id for source_id, fold in folds_as_text.items() if fold == str(binding.outer_fold)}
        predictions, _ = _load_coral_fold(binding, stage3, expected_ids, method_inventory)
        need(not (set(coral_by_id) & set(predictions)), "coral outer predictions overlap")
        coral_by_id.update(predictions)
    need(set(coral_by_id) == set(ordered_ids), "coral full OOF population differs")
    _validate_aggregate(value, stage3, method_inventory)
    if validate_only:
        return {"schema_version": SCHEMA_VERSION, "status": "validated", "records": 2000, "folds": 5,
                "method": CORAL_METHOD, "rps_eligible": False,
                "validation_rows_loaded": False, "average_target_used": False}

    truth = np.asarray([truth_by_id[source_id] for source_id in ordered_ids], dtype=float)
    baseline = np.asarray([r0_by_id[source_id] for source_id in ordered_ids], dtype=float)
    coral = np.asarray([coral_by_id[source_id] for source_id in ordered_ids], dtype=float)
    decision = promotion_gate(truth, baseline, coral, ordered_ids, seed=stage3.seed)
    result = {"schema_version": SCHEMA_VERSION, "status": "completed", "run_id": value.run_id,
              "records": 2000, "folds": 5, "axes": list(AXES), "method": CORAL_METHOD,
              "eligible": decision["eligible"],
              "rps_eligible": False, "rps_role": "descriptive_only_never_eligible",
              "promotion_gate": decision, "bootstrap_seed": stage3.seed,
              "stage6_preregistration_sha256": value.stage6_preregistration_sha256,
              "stage6_preregistration_commit": value.stage6_preregistration_commit,
              "stage3_config_file_sha256": value.stage3_config_file_sha256,
              "stage3_report_config_sha256": value.stage3_report_config_sha256,
              "stage3_aggregate_sha256": value.stage3_aggregate_sha256,
              "canonical_train_sha256": stage3.train_sha256,
              "fold_manifest_sha256": stage3.fold_manifest_sha256,
              "fold_rows_sha256": stage3.fold_rows_sha256,
              "r0_oof_prediction_sha256": stage3.r0_oof_prediction_sha256,
              "outer_bindings": [{"outer_fold": item.outer_fold, "public_sha256": item.public_sha256,
                                    "coral_restricted_sha256": item.coral_restricted_sha256}
                                   for item in value.outer_bindings],
              "config_sha256": value.config_sha256, "validation_rows_loaded": False,
              "average_target_used": False,
              "privacy": "aggregate_only_no_rows_ids_text_embeddings_or_predictions"}
    _write_public_fresh(Path(value.output_path), result)
    return result


__all__ = ["CORAL_METHOD", "OuterBinding", "PromotionConfig", "SCHEMA_VERSION",
           "Stage3CoralPromotionError", "promotion_gate", "run"]
