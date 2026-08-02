from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mal2026.iterative_tail_metrics import AXES
from mal2026.kure_ordinal_oof import KUREOrdinalOOFConfig, config_sha256
from mal2026.stage3_coral_promotion import (
    CORAL_METHOD, OuterBinding, PromotionConfig, Stage3CoralPromotionError,
    _load_coral_fold, promotion_gate, run,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_mapping(output_path: str = "outputs/stage3-coral-promotion/unit.json") -> dict:
    stage3 = KUREOrdinalOOFConfig.from_json("configs/kure_ordinal_oof.v1.json")
    return {
        "schema_version": "mal2026-stage3-coral-promotion-v1",
        "run_id": "stage3-coral-promotion-unit",
        "stage6_preregistration_path": "configs/stage6_submission_prereg.v1.json",
        "stage6_preregistration_sha256": "7616e038dd0dcb8a10a15c09780ca178ff43700c132fa941ba4e050e2a8176e1",
        "stage6_preregistration_commit": "32b0a43eda5612284d5bd718c5afbce2be182eff",
        "stage3_config_path": "configs/kure_ordinal_oof.v1.json",
        "stage3_config_file_sha256": "5108e37c490482fbbf45c043f2ad4d0154048d432b4a4a163b3d102f8c6d756e",
        "stage3_report_config_sha256": config_sha256(stage3),
        "stage3_aggregate_path": f"{stage3.output_root}/aggregate.json",
        "stage3_aggregate_sha256": "a" * 64,
        "outer_bindings": [{
            "outer_fold": fold,
            "public_path": f"{stage3.output_root}/outer-{fold:02d}.json",
            "public_sha256": str(fold) * 64,
            "coral_restricted_path": f"{stage3.restricted_output_root}/outer-{fold:02d}/{CORAL_METHOD}/predictions.jsonl",
            "coral_restricted_sha256": str(fold + 1) * 64,
        } for fold in range(5)],
        "output_path": output_path,
    }


class Stage3CoralPromotionTests(unittest.TestCase):
    def test_config_binds_committed_prereg_and_rejects_mismatch(self) -> None:
        PromotionConfig.from_mapping(config_mapping())
        invalid = config_mapping()
        invalid["stage6_preregistration_commit"] = "0" * 40
        with self.assertRaisesRegex(Stage3CoralPromotionError, "preregistration binding"):
            PromotionConfig.from_mapping(invalid)
        invalid = config_mapping()
        invalid["output_path"] = "outputs/validation/result.json"
        with self.assertRaisesRegex(Stage3CoralPromotionError, "validation"):
            PromotionConfig.from_mapping(invalid)

    def test_exact_gate_passes_improvement_and_fails_identity(self) -> None:
        scores = np.tile(np.arange(1.0, 6.0), 8)
        truth = np.column_stack([scores, scores, scores])
        baseline = np.clip(truth + 0.2, 1.0, 5.0)
        bootstrap = {"intervals": {"rmse": {"lower": 0.1}}}
        with patch("mal2026.stage3_coral_promotion.paired_bootstrap_delta_ci", return_value=bootstrap):
            passing = promotion_gate(truth, baseline, truth, [f"id-{i}" for i in range(len(truth))], seed=17)
            failing = promotion_gate(truth, baseline, baseline, [f"id-{i}" for i in range(len(truth))], seed=17)
        self.assertTrue(passing["eligible"])
        self.assertTrue(all(passing["gates"].values()))
        self.assertEqual("source_id_with_all_three_axes_clustered", passing["paired_bootstrap"]["unit"])
        self.assertFalse(failing["eligible"])
        self.assertFalse(failing["gates"]["macro_rmse"])

    def test_outer_loader_rejects_restricted_tamper(self) -> None:
        stage3 = KUREOrdinalOOFConfig.from_json("configs/kure_ordinal_oof.v1.json")
        expected = {f"id-{index:03d}" for index in range(400)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            restricted = root / "predictions.jsonl"
            restricted.write_text("".join(json.dumps({"source_id": source_id, "outer_fold": 0,
                "prediction": {axis: 3.0 for axis in AXES}}) + "\n" for source_id in sorted(expected)))
            restricted_sha = digest(restricted)
            public = root / "outer.json"
            public.write_text(json.dumps({
                "schema_version": "mal2026-kure-ordinal-oof-outer-v1", "status": "completed", "mode": "outer_fold",
                "run_id": stage3.run_id, "config_sha256": config_sha256(stage3), "outer_fold": 0, "records": 400,
                "fold_manifest_sha256": stage3.fold_manifest_sha256, "fold_rows_sha256": stage3.fold_rows_sha256,
                "validation_rows_loaded": False, "average_target_used": False,
                "methods": [
                    {"method": CORAL_METHOD, "family": "coral", "restricted_prediction_sha256": restricted_sha,
                     "axis_bindings": [{"axis": axis} for axis in AXES]},
                    {"method": "rps-natural", "family": "rps", "restricted_prediction_sha256": "9" * 64,
                     "axis_bindings": [{"axis": axis} for axis in AXES]},
                ],
            }))
            binding = OuterBinding(0, str(public), digest(public), str(restricted), restricted_sha)
            loaded, _ = _load_coral_fold(binding, stage3, expected, (CORAL_METHOD, "rps-natural"))
            self.assertEqual(expected, set(loaded))
            restricted.write_text(restricted.read_text() + "\n")
            with self.assertRaisesRegex(Stage3CoralPromotionError, "checksum"):
                _load_coral_fold(binding, stage3, expected, (CORAL_METHOD, "rps-natural"))

    def test_validate_only_checks_all_five_synthetic_folds_without_output(self) -> None:
        config = PromotionConfig.from_mapping(config_mapping())
        stage3 = KUREOrdinalOOFConfig.from_json("configs/kure_ordinal_oof.v1.json")
        identifiers = [f"id-{index:04d}" for index in range(2000)]
        truth = {source_id: (3.0, 3.0, 3.0) for source_id in identifiers}
        r0 = dict(truth)
        folds = {source_id: str(index // 400) for index, source_id in enumerate(identifiers)}

        def fold_loader(binding: OuterBinding, *_: object) -> tuple[dict, dict]:
            rows = {source_id: (3.0, 3.0, 3.0) for source_id in identifiers
                    if folds[source_id] == str(binding.outer_fold)}
            return rows, {}

        methods = (SimpleNamespace(identifier=CORAL_METHOD), SimpleNamespace(identifier="rps-natural"))
        with patch("mal2026.stage3_coral_promotion._load_bound_contract", return_value=({}, stage3)), \
             patch("mal2026.stage3_coral_promotion.load_recommended_methods", return_value=methods), \
             patch("mal2026.stage3_coral_promotion._load_canonical", return_value=(identifiers, truth, folds, r0)), \
             patch("mal2026.stage3_coral_promotion._load_coral_fold", side_effect=fold_loader) as load_fold, \
             patch("mal2026.stage3_coral_promotion._validate_aggregate", return_value={}):
            result = run(config, validate_only=True)
        self.assertEqual("validated", result["status"])
        self.assertEqual(5, load_fold.call_count)
        self.assertFalse(result["rps_eligible"])
        self.assertFalse(Path(config.output_path).exists())

    def test_public_result_is_aggregate_only_and_rps_never_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "aggregate.json"
            config = PromotionConfig.from_mapping(config_mapping(str(output)))
            stage3 = KUREOrdinalOOFConfig.from_json("configs/kure_ordinal_oof.v1.json")
            identifiers = [f"id-{index:04d}" for index in range(2000)]
            truth = {source_id: (3.0, 3.0, 3.0) for source_id in identifiers}
            folds = {source_id: str(index // 400) for index, source_id in enumerate(identifiers)}
            methods = (SimpleNamespace(identifier=CORAL_METHOD), SimpleNamespace(identifier="rps-natural"))

            def fold_loader(binding: OuterBinding, *_: object) -> tuple[dict, dict]:
                return ({source_id: (3.0, 3.0, 3.0) for source_id in identifiers
                         if folds[source_id] == str(binding.outer_fold)}, {})

            decision = {"eligible": False, "gates": {"macro_rmse": False}, "improvements": {},
                        "paired_bootstrap": {}, "exact_r0_metrics": {}, "coral_natural_metrics": {}}
            with patch("mal2026.stage3_coral_promotion._load_bound_contract", return_value=({}, stage3)), \
                 patch("mal2026.stage3_coral_promotion.load_recommended_methods", return_value=methods), \
                 patch("mal2026.stage3_coral_promotion._load_canonical", return_value=(identifiers, truth, folds, truth)), \
                 patch("mal2026.stage3_coral_promotion._load_coral_fold", side_effect=fold_loader), \
                 patch("mal2026.stage3_coral_promotion._validate_aggregate", return_value={}), \
                 patch("mal2026.stage3_coral_promotion.promotion_gate", return_value=decision):
                result = run(config)
            public = json.loads(output.read_text())
            self.assertEqual(result, public)
            self.assertFalse(public["rps_eligible"])
            def keys(value: object) -> set[str]:
                if isinstance(value, dict):
                    return set(value) | set().union(*(keys(child) for child in value.values()))
                if isinstance(value, list):
                    return set().union(*(keys(child) for child in value)) if value else set()
                return set()
            self.assertTrue({"source_id", "prediction", "essay", "raw_gold"}.isdisjoint(keys(public)))
            self.assertFalse(public["average_target_used"])


if __name__ == "__main__":
    unittest.main()
