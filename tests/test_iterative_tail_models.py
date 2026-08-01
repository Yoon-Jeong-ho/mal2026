from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from mal2026.iterative_tail_models import (
    CandidateSpec,
    FAMILIES,
    adjacent_score_supervised_contrastive_loss,
    apply_auxiliary_3v4_correction,
    effective_number_weights,
    equal_band_replay_weights,
    fit_predict,
)


def synthetic_fold():
    rng = np.random.default_rng(71)
    train_embeddings = rng.normal(size=(12, 8)).astype(np.float32)
    predict_embeddings = rng.normal(size=(5, 8)).astype(np.float32)
    train_base = rng.uniform(1.4, 4.6, size=(12, 3)).astype(np.float32)
    predict_base = rng.uniform(1.4, 4.6, size=(5, 3)).astype(np.float32)
    targets = np.clip(train_base + 0.25 * np.tanh(train_embeddings[:, :3]), 1, 5).astype(np.float32)
    return train_embeddings, train_base, targets, predict_embeddings, predict_base


class IterativeTailModelTests(unittest.TestCase):
    def test_every_family_smokes(self):
        train_embeddings, train_base, targets, predict_embeddings, predict_base = synthetic_fold()
        for family in FAMILIES:
            with self.subTest(family=family):
                result = fit_predict(
                    CandidateSpec(family=family, epochs=2, hidden_dim=7),
                    train_embeddings, train_base, targets, predict_embeddings, predict_base,
                )
                self.assertEqual((5, 3), result.predictions.shape)
                self.assertTrue(np.isfinite(result.predictions).all())
                self.assertTrue(((result.predictions >= 1) & (result.predictions <= 5)).all())

    def test_deterministic_fresh_rounds(self):
        train_embeddings, train_base, targets, predict_embeddings, predict_base = synthetic_fold()
        spec = CandidateSpec(family="joint_huber_ordinal", rounds=2, epochs=3, hidden_dim=9, seed=123)
        first = fit_predict(spec, train_embeddings, train_base, targets, predict_embeddings, predict_base)
        second = fit_predict(spec, train_embeddings, train_base, targets, predict_embeddings, predict_base)
        np.testing.assert_array_equal(first.predictions, second.predictions)
        self.assertEqual(first.initial_state_hashes, second.initial_state_hashes)
        self.assertEqual(2, len(set(first.initial_state_hashes)))
        other = fit_predict(replace(spec, seed=124), train_embeddings, train_base, targets, predict_embeddings, predict_base)
        self.assertNotEqual(first.initial_state_hashes, other.initial_state_hashes)

    def test_tail_weights_and_auxiliary_correction(self):
        targets = torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
        low = effective_number_weights(targets, mode="low", strength=2)
        high = effective_number_weights(targets, mode="high", strength=2)
        self.assertGreater(float(low[0]), float(low[-1]))
        self.assertGreater(float(high[-1]), float(high[0]))
        self.assertTrue(torch.isclose(equal_band_replay_weights(targets).mean(), torch.tensor(1.0)))
        predictions = torch.tensor([[3.5, 3.5, 3.5], [1.5, 5.0, 2.0]])
        low_logits, high_logits = torch.full_like(predictions, -10), torch.full_like(predictions, 10)
        corrected_low = apply_auxiliary_3v4_correction(predictions, low_logits)
        corrected_high = apply_auxiliary_3v4_correction(predictions, high_logits)
        self.assertTrue(torch.all(corrected_low[0] < corrected_high[0]))
        self.assertTrue(torch.equal(corrected_low[1], corrected_high[1]))

    def test_contrastive_and_contract(self):
        representations = torch.randn(4, 6, requires_grad=True)
        targets = torch.tensor([[2.0] * 3, [3.0] * 3, [5.0] * 3, [4.0] * 3])
        loss = adjacent_score_supervised_contrastive_loss(representations, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        train_embeddings, train_base, y, predict_embeddings, predict_base = synthetic_fold()
        with self.assertRaisesRegex(ValueError, "3 columns"):
            fit_predict(
                CandidateSpec(family="ridge_residual"), train_embeddings, train_base,
                np.column_stack((y, y.mean(1))), predict_embeddings, predict_base,
            )


if __name__ == "__main__":
    unittest.main()
