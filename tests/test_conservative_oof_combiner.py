from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mal2026.conservative_oof_combiner import (
    CALIBRATION_STATUS, PREREGISTRATION_SHA256, CombinerConfig, ConservativeCombinerError, FoldFile, SourceSpec,
    _load_source_fold, _validate_upstream_aggregate, fit_outer_combiner, fixed_candidate, promotion_gate,
)
from mal2026.iterative_tail_metrics import AXES


def config_mapping() -> dict:
    return {
        "schema_version": "mal2026-conservative-oof-combiner-v1", "run_id": "unit-combiner",
        "train_path": "train.jsonl", "train_sha256": "a" * 64,
        "fold_manifest_path": "manifest.json", "fold_manifest_sha256": "b" * 64,
        "fold_rows_path": "rows.jsonl", "fold_rows_sha256": "c" * 64,
        "r0_oof_prediction_path": "r0.jsonl", "r0_oof_prediction_sha256": "d" * 64,
        "preregistration_path": "configs/conservative_oof_combiner.prereg.v1.json",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "sources": [{"id": "stage3-coral", "kind": "stage3_kure", "provenance": "standard_5fold_oof",
            "upstream_run_id": "kure-ordinal-oof-v1-20260803-001",
            "upstream_config_sha256": "28e6ba7465b91ba1fcc306f97f10a8e213d4a6e0069ba499fd92199d827a15aa",
            "upstream_outer_schema": "kure-outer-v1", "upstream_aggregate_schema": "kure-aggregate-v1",
            "upstream_method_id": "coral-natural", "upstream_method_inventory": ["coral-natural"],
            "aggregate_path": "outputs/kure-ordinal-oof-v1/kure-ordinal-oof-v1-20260803-001/aggregate.json",
            "aggregate_sha256": "f" * 64,
            "fold_files": [
            {"outer_fold": fold,
             "public_path": f"outputs/kure-ordinal-oof-v1/kure-ordinal-oof-v1-20260803-001/outer-{fold:02d}.json",
             "public_sha256": str(fold) * 64,
             "restricted_path": f"data/processed/restricted/kure_ordinal_oof_v1/kure-ordinal-oof-v1-20260803-001/outer-{fold:02d}/coral-natural/predictions.jsonl",
             "restricted_sha256": str(fold + 1) * 64}
            for fold in range(5)]}],
        "output_root": "outputs/combiner", "restricted_output_root": "data/processed/restricted/combiner",
        "seed": 17, "axes": list(AXES), "average_target_forbidden": True,
        "combination_mode": "preregistered_fixed_standard_oof", "fixed_partner_source_id": "stage3-coral",
        "fixed_partner_method_id": "coral-natural", "fixed_partner_weight": 0.2,
        "calibration_status": CALIBRATION_STATUS,
        "promotion_gate": {"minimum_macro_rmse_improvement": .005, "maximum_axis_rmse_worsening": .01,
            "maximum_gold_3_4_balanced_accuracy_drop": .01, "maximum_spearman_drop": .005,
            "low_tail_noninferior": True,
            "high_tail_noninferior": True, "paired_bootstrap_resamples": 10000,
            "paired_bootstrap_lower_bound_above_zero": True},
    }


class ConservativeCombinerTests(unittest.TestCase):
    def test_config_forbids_validation_and_average(self) -> None:
        CombinerConfig.from_mapping(config_mapping())
        invalid = dict(config_mapping(), note="validation rows")
        with self.assertRaisesRegex(ConservativeCombinerError, "validation"):
            CombinerConfig.from_mapping(invalid)
        invalid = dict(config_mapping(), axes=[*AXES, "average"])
        with self.assertRaisesRegex(ConservativeCombinerError, "average"):
            CombinerConfig.from_mapping(invalid)
        invalid = config_mapping()
        invalid["promotion_gate"]["maximum_spearman_drop"] = 0.006
        with self.assertRaisesRegex(ConservativeCombinerError, "promotion gate"):
            CombinerConfig.from_mapping(invalid)

    def test_preregistration_tamper_and_stage3_lineage_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tampered = Path(temporary) / "prereg.json"
            original = Path("configs/conservative_oof_combiner.prereg.v1.json").read_bytes()
            tampered.write_bytes(original + b"\n")
            invalid = config_mapping()
            invalid["preregistration_path"] = str(tampered)
            invalid["preregistration_sha256"] = hashlib.sha256(tampered.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ConservativeCombinerError, "preregistration checksum"):
                CombinerConfig.from_mapping(invalid)

        invalid = config_mapping()
        invalid["sources"][0]["upstream_config_sha256"] = "9" * 64
        with self.assertRaisesRegex(ConservativeCombinerError, "Stage3 run/config/report"):
            CombinerConfig.from_mapping(invalid)

    def test_standard_oof_learned_fitting_rejected_and_fixed_math_held_safe(self) -> None:
        with self.assertRaisesRegex(ConservativeCombinerError, "outer_nested"):
            fit_outer_combiner()
        r0 = np.asarray([[2.0, 3.0, 4.0], [4.0, 2.0, 1.0]])
        partner = np.asarray([[3.0, 4.0, 5.0], [5.0, 3.0, 2.0]])
        expected = .8 * r0 + .2 * partner
        self.assertTrue(np.allclose(fixed_candidate(r0, partner), expected))

    def test_source_fold_rejects_swapped_id_and_tampered_axis(self) -> None:
        expected = {f"id-{index}" for index in range(400)}
        raw = {identifier: (3.0, 3.0, 3.0) for identifier in expected}
        with tempfile.TemporaryDirectory() as temporary:
            valid = Path(temporary) / "valid.jsonl"
            rows = [{"source_id": identifier, "outer_fold": 0,
                     "prediction": {axis: 3.0 for axis in AXES}} for identifier in sorted(expected)]
            valid.write_text("".join(json.dumps(item) + "\n" for item in rows))
            public = Path(temporary) / "public.json"
            restricted_sha = hashlib.sha256(valid.read_bytes()).hexdigest()
            public.write_text(json.dumps({"schema_version": "kure-outer-v1", "status": "completed", "mode": "outer_fold",
                "run_id": "kure-run", "config_sha256": "e" * 64, "outer_fold": 0, "records": 400,
                "methods": [{"method": "coral-natural", "restricted_prediction_sha256": restricted_sha}]}))
            binding = FoldFile(0, str(public), hashlib.sha256(public.read_bytes()).hexdigest(), str(valid), restricted_sha)
            spec = SourceSpec("kure", "stage3_kure", "standard_5fold_oof", "kure-run", "e" * 64,
                              "kure-outer-v1", "kure-aggregate-v1", "coral-natural", ("coral-natural",), "aggregate.json", "f" * 64,
                              tuple([binding] * 5))
            self.assertEqual(set(_load_source_fold(spec, binding, expected, raw)), expected)
            tampered_public = json.loads(public.read_text()); tampered_public["config_sha256"] = "0" * 64
            public.write_text(json.dumps(tampered_public))
            tampered_binding = FoldFile(0, str(public), hashlib.sha256(public.read_bytes()).hexdigest(), str(valid), restricted_sha)
            with self.assertRaisesRegex(ConservativeCombinerError, "outer report"):
                _load_source_fold(spec, tampered_binding, expected, raw)
            rows[0]["source_id"] = "wrong-fold-id"
            valid.write_text("".join(json.dumps(item) + "\n" for item in rows))
            restricted_sha = hashlib.sha256(valid.read_bytes()).hexdigest()
            public.write_text(json.dumps({"schema_version": "kure-outer-v1", "status": "completed", "mode": "outer_fold",
                "run_id": "kure-run", "config_sha256": "e" * 64, "outer_fold": 0, "records": 400,
                "methods": [{"method": "coral-natural", "restricted_prediction_sha256": restricted_sha}]}))
            binding = FoldFile(0, str(public), hashlib.sha256(public.read_bytes()).hexdigest(), str(valid), restricted_sha)
            with self.assertRaisesRegex(ConservativeCombinerError, "ID/fold"):
                _load_source_fold(spec, binding, expected, raw)
            rows[0] = {"source_id": "id-0", "outer_fold": 0, "prediction": {"content": 3.0, "organization": 3.0}}
            valid.write_text("".join(json.dumps(item) + "\n" for item in rows))
            restricted_sha = hashlib.sha256(valid.read_bytes()).hexdigest()
            public.write_text(json.dumps({"schema_version": "kure-outer-v1", "status": "completed", "mode": "outer_fold",
                "run_id": "kure-run", "config_sha256": "e" * 64, "outer_fold": 0, "records": 400,
                "methods": [{"method": "coral-natural", "restricted_prediction_sha256": restricted_sha}]}))
            binding = FoldFile(0, str(public), hashlib.sha256(public.read_bytes()).hexdigest(), str(valid), restricted_sha)
            with self.assertRaisesRegex(ConservativeCombinerError, "axes"):
                _load_source_fold(spec, binding, expected, raw)

    def test_upstream_public_tamper_and_provenance_rejected(self) -> None:
        raw = config_mapping(); raw["sources"][0]["provenance"] = "ordinary_in_sample"
        with self.assertRaisesRegex(ConservativeCombinerError, "provenance"):
            CombinerConfig.from_mapping(raw)

    def test_stage3_aggregate_requires_exact_configured_fold_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            aggregate_path = Path(temporary) / "aggregate.json"
            folds = tuple(FoldFile(fold, f"public-{fold}.json", str(fold) * 64,
                                   f"restricted-{fold}.jsonl", str(fold + 1) * 64)
                          for fold in range(5))
            fold_bindings = [{"outer_fold": item.outer_fold, "public_sha256": item.public_sha256,
                              "restricted_prediction_sha256": item.restricted_sha256,
                              "axis_bindings": {"ignored": True}, "environment": {"ignored": True}}
                             for item in folds]
            aggregate = {"schema_version": "kure-aggregate-v1", "status": "completed", "run_id": "kure-run",
                         "config_sha256": "e" * 64, "records": 2000, "folds": 5,
                         "validation_rows_loaded": False, "average_target_used": False,
                         "methods": [{"method": "coral-natural", "fold_bindings": fold_bindings}]}
            aggregate_path.write_text(json.dumps(aggregate))

            def source() -> SourceSpec:
                return SourceSpec("kure", "stage3_kure", "standard_5fold_oof", "kure-run", "e" * 64,
                                  "kure-outer-v1", "kure-aggregate-v1", "coral-natural", ("coral-natural",),
                                  str(aggregate_path), hashlib.sha256(aggregate_path.read_bytes()).hexdigest(), folds)

            _validate_upstream_aggregate(source())
            aggregate["methods"][0]["fold_bindings"][2]["restricted_prediction_sha256"] = "9" * 64
            aggregate_path.write_text(json.dumps(aggregate))
            with self.assertRaisesRegex(ConservativeCombinerError, "fold bindings"):
                _validate_upstream_aggregate(source())

    def test_promotion_fails_without_every_axis_tail_support(self) -> None:
        config = CombinerConfig.from_mapping(config_mapping())
        truth = np.full((20, 3), 3.0); baseline = truth + .2; candidate = truth
        bootstrap = {"intervals": {"rmse": {"lower": 0.1}}}
        with patch("mal2026.conservative_oof_combiner.paired_bootstrap_delta_ci", return_value=bootstrap):
            result = promotion_gate(truth, baseline, candidate, [str(i) for i in range(20)], config)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["gates"]["tail_support_every_axis"])

    def test_promotion_rejects_spearman_drop_over_preregistered_limit(self) -> None:
        config = CombinerConfig.from_mapping(config_mapping())
        scores = np.tile(np.arange(1.0, 6.0), 4)
        truth = np.column_stack([scores, scores, scores])
        baseline = truth.copy()
        candidate = 6.0 - truth
        bootstrap = {"intervals": {"rmse": {"lower": 0.1}}}
        with patch("mal2026.conservative_oof_combiner.paired_bootstrap_delta_ci", return_value=bootstrap):
            result = promotion_gate(truth, baseline, candidate, [str(i) for i in range(20)], config)
        self.assertFalse(result["gates"]["spearman"])


if __name__ == "__main__":
    unittest.main()
