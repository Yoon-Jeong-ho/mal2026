from __future__ import annotations

import unittest

import numpy as np

from mal2026.iterative_tail_runner import (
    _paired_bootstrap_macro_rmse,
    _threshold_predict,
    load_experiment_data,
    prepare_score_blind_cache,
    variants_for_round,
)


class IterativeTailRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare_score_blind_cache()
        cls.data = load_experiment_data()

    def test_fixed_round_variant_inventory(self):
        self.assertEqual([], list(variants_for_round(1)))
        self.assertEqual(5, len(variants_for_round(2)))
        self.assertEqual(2, len(variants_for_round(3)))
        self.assertEqual(4, len(variants_for_round(6)))
        self.assertEqual(3, len(variants_for_round(7)))
        self.assertEqual(3, len(variants_for_round(8)))
        self.assertTrue(all(variants_for_round(number) for number in range(2, 19)))
        self.assertEqual([], list(variants_for_round(19)))
        self.assertEqual([], list(variants_for_round(20)))

    def test_real_train_cache_is_aligned_and_score_blind(self):
        data = self.data
        self.assertEqual((2000, 4096), data.embeddings.shape)
        self.assertEqual((2000, 3), data.base.shape)
        self.assertEqual((2000, 3), data.targets.shape)
        self.assertEqual({0, 1, 2, 3, 4}, set(data.folds.tolist()))
        self.assertEqual((2000, 18), data.evidence.view("content_structured").shape)
        self.assertEqual((2000, 36), data.evidence.view("org_expression_structured").shape)
        self.assertEqual((2000, 177), data.evidence.view("consensus_disagreement").shape)
        self.assertEqual((2000, 576), data.evidence.view("evidence_hash").shape)
        self.assertEqual((2000, 969), data.evidence.view("full_fusion").shape)

    def test_threshold_head_is_bounded_and_fresh(self):
        rng = np.random.default_rng(11)
        base = rng.uniform(1, 5, size=(32, 3)).astype(np.float32)
        target = np.clip(base + 0.1, 1, 5)
        prediction, initial, final = _threshold_predict(base, target, base[:7], device="cpu")
        self.assertEqual((7, 3), prediction.shape)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(((prediction >= 1) & (prediction <= 5)).all())
        self.assertEqual(3, len(initial))
        self.assertTrue(all(left != right for left, right in zip(initial, final, strict=True)))

    def test_vectorized_final_bootstrap_sign(self):
        truth = np.tile(np.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]]), (4, 3))
        baseline = np.clip(truth + 0.5, 1, 5)
        candidate = truth.copy()
        result = _paired_bootstrap_macro_rmse(truth, baseline, candidate, n_resamples=100, seed=3)
        interval = result["intervals"]["rmse"]
        self.assertGreater(interval["estimate"], 0)
        self.assertGreater(interval["lower"], 0)


if __name__ == "__main__":
    unittest.main()
