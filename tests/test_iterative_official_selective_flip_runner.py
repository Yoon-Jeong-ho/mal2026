from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mal2026.iterative_official_agent_stack_models import FEATURE_DIM
from mal2026.iterative_official_selective_flip_models import SelectiveFlipResult, candidate_specs
from mal2026.iterative_official_selective_flip_runner import (
    OfficialSelectiveFlipData,
    _candidate_oof,
    run_outer_fold,
)
from mal2026.iterative_tail_runner import ExperimentData


def bundle_fixture() -> OfficialSelectiveFlipData:
    folds = np.repeat(np.arange(5), 2)
    base = np.full((10, 3), 3.0, dtype=np.float32)
    features = np.zeros((10, FEATURE_DIM), dtype=np.float64)
    features[:, 0] = folds
    data = ExperimentData(
        tuple(f"s{i}" for i in range(10)), tuple(f"d{i}" for i in range(10)),
        tuple("p" for _ in range(10)), np.zeros((10, 1), dtype=np.float32),
        base, base.copy(), folds, None,
    )
    return OfficialSelectiveFlipData(
        data, features, {"records": 10, "feature_dimensions": FEATURE_DIM},
        {"candidates": 30, "human_or_reference_score_read_or_prompted": False},
    )


class OfficialSelectiveFlipRunnerTest(unittest.TestCase):
    def test_candidate_oof_exact_coverage_and_fold_exclusion(self):
        bundle = bundle_fixture()
        calls = []

        def fake(spec, train_features, train_base, train_targets, predict_features, predict_base, *, device):
            calls.append((set(train_features[:, 0].astype(int)), set(predict_features[:, 0].astype(int))))
            return SelectiveFlipResult(np.asarray(predict_base, dtype=np.float64), {"fresh_initialization": True})

        with patch("mal2026.iterative_official_selective_flip_runner.fit_predict_selective_flip", side_effect=fake):
            prediction, audit = _candidate_oof(bundle, candidate_specs()[0], 0, device="cuda:0")
        self.assertEqual(4, len(calls))
        self.assertEqual(4, len(audit))
        self.assertTrue(np.isnan(prediction[bundle.experiment.folds == 0]).all())
        self.assertTrue(np.isfinite(prediction[bundle.experiment.folds != 0]).all())
        for train_folds, predict_folds in calls:
            self.assertNotIn(0, train_folds | predict_folds)
            self.assertEqual(3, len(train_folds))
            self.assertEqual(1, len(predict_folds))
            self.assertTrue(train_folds.isdisjoint(predict_folds))

    def test_outer_shard_completes_all_three_before_selection_and_writes_restricted_rows(self):
        bundle = bundle_fixture()
        specs = candidate_specs()
        completed = []

        def fake_oof(_bundle, spec, outer_fold, *, device):
            completed.append(spec.variant_id)
            prediction = np.full_like(bundle.experiment.targets, np.nan, dtype=np.float64)
            prediction[bundle.experiment.folds != outer_fold] = 3.0
            return prediction, [{"variant_id": spec.variant_id}]

        def fake_select(given_specs, metrics_by_id, baseline, config):
            self.assertEqual([spec.variant_id for spec in specs], completed)
            self.assertEqual(set(completed), set(metrics_by_id))
            return {
                "selected_id": "baseline", "fell_back_to_baseline": True,
                "inventory_valid": True,
                "decisions": [{"variant_id": spec.variant_id, "eligible": False} for spec in given_specs],
            }

        metric = {"macro": {"rmse": .5}}
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch("mal2026.iterative_official_selective_flip_runner.PUBLIC_ROOT", Path(temp) / "public"),
                patch("mal2026.iterative_official_selective_flip_runner.RESTRICTED_ROOT", Path(temp) / "restricted"),
                patch("mal2026.iterative_official_selective_flip_runner.validate_bound_inputs"),
                patch("mal2026.iterative_official_selective_flip_runner.load_official_selective_flip_data", return_value=bundle),
                patch("mal2026.iterative_official_selective_flip_runner._candidate_oof", side_effect=fake_oof),
                patch("mal2026.iterative_official_selective_flip_runner.select_candidate", side_effect=fake_select),
                patch("mal2026.iterative_official_selective_flip_runner.compute_iterative_tail_metrics", return_value=metric),
            ):
                result = run_outer_fold(0, device="cuda:0")
                public = Path(temp) / "public/outer-0/result.json"
                restricted = Path(temp) / "restricted/outer-0/predictions.jsonl"
                self.assertTrue(public.is_file())
                self.assertTrue(restricted.is_file())
                self.assertEqual(2, len(restricted.read_text().splitlines()))
        self.assertEqual([spec.variant_id for spec in specs], completed)
        self.assertEqual(3, result["candidate_count"])
        self.assertTrue(result["fell_back_to_baseline"])


if __name__ == "__main__":
    unittest.main()
