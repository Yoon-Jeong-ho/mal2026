import unittest
from unittest.mock import patch

import numpy as np

from mal2026.iterative_official_agent_stack_models import FEATURE_DIM
from mal2026.iterative_official_boundary_models import OfficialBoundaryResult, candidate_specs
from mal2026.iterative_official_boundary_runner import OfficialBoundaryData, _candidate_oof
from mal2026.iterative_tail_runner import ExperimentData


class OfficialBoundaryRunnerTest(unittest.TestCase):
    def test_candidate_oof_exact_coverage_and_fold_exclusion(self):
        folds = np.repeat(np.arange(5), 2)
        base = np.full((10, 3), 3.0, dtype=np.float32)
        features = np.zeros((10, FEATURE_DIM), dtype=np.float64); features[:, 0] = folds
        data = ExperimentData(
            tuple(f"s{i}" for i in range(10)), tuple(f"d{i}" for i in range(10)),
            tuple("p" for _ in range(10)), np.zeros((10, 1), dtype=np.float32),
            base, base.copy(), folds, None,
        )
        bundle = OfficialBoundaryData(data, features, {"records": 10}, {"candidates": 30})
        calls = []

        def fake(spec, train_features, train_base, train_targets, predict_features, predict_base, *, device):
            calls.append((set(train_features[:, 0].astype(int)), set(predict_features[:, 0].astype(int))))
            return OfficialBoundaryResult(np.asarray(predict_base, dtype=np.float64), {"fresh_initialization": True})

        with patch("mal2026.iterative_official_boundary_runner.fit_predict_official_boundary", side_effect=fake):
            prediction, audit = _candidate_oof(bundle, candidate_specs()[0], 0, device="cuda:0")
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(audit), 4)
        self.assertTrue(np.isnan(prediction[folds == 0]).all())
        self.assertTrue(np.isfinite(prediction[folds != 0]).all())
        for train_folds, predict_folds in calls:
            self.assertNotIn(0, train_folds | predict_folds)
            self.assertEqual(len(train_folds), 3)
            self.assertEqual(len(predict_folds), 1)
            self.assertTrue(train_folds.isdisjoint(predict_folds))


if __name__ == "__main__":
    unittest.main()
