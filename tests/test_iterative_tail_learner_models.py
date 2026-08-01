from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

import numpy as np

from mal2026.iterative_tail_learner_models import (
    BATCH_SIZE,
    EPOCHS,
    FAMILIES,
    GRAD_CLIP,
    LEARNING_RATE,
    RUN_SEED,
    WEIGHT_DECAY,
    LearnerSpec,
    apply,
    candidate_specs,
    fit,
)


def _tiny() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41)
    train_embeddings = rng.normal(size=(12, 7)).astype(np.float32)
    train_base = rng.uniform(1.4, 4.6, size=(12, 3)).astype(np.float32)
    signal = .12 * np.tanh(train_embeddings[:, :3])
    train_targets = np.clip(train_base + signal, 1, 5).astype(np.float32)
    predict_embeddings = rng.normal(size=(5, 7)).astype(np.float32)
    predict_base = rng.uniform(1.4, 4.6, size=(5, 3)).astype(np.float32)
    return train_embeddings, train_base, train_targets, predict_embeddings, predict_base


class LearnerModelsTest(unittest.TestCase):
    def test_exact_inventory_and_common_training_contract(self) -> None:
        specs = candidate_specs()
        self.assertEqual(len(specs), 20)
        self.assertEqual([spec.cycle for spec in specs], list(range(1, 21)))
        self.assertEqual(tuple(dict.fromkeys(spec.family for spec in specs)), FAMILIES)
        self.assertEqual([sum(spec.family == family for spec in specs) for family in FAMILIES], [4] * 5)
        self.assertEqual({spec.seed for spec in specs}, {RUN_SEED})
        self.assertEqual((BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP),
                         (128, 40, 3e-4, 1e-4, 1.0))
        with self.assertRaises(ValueError):
            fit(replace(specs[0], parameters={**specs[0].parameters, "cap": .99}), *_tiny()[:3])

    def test_each_family_tiny_fit_apply_is_bounded_and_aggregate_safe(self) -> None:
        train_e, train_b, train_y, predict_e, predict_b = _tiny()
        for spec in candidate_specs()[::4]:
            with self.subTest(family=spec.family):
                fitted = fit(spec, train_e, train_b, train_y)
                result = apply(fitted, predict_e, predict_b)
                self.assertEqual(result.predictions.shape, (5, 3))
                self.assertTrue(np.isfinite(result.predictions).all())
                self.assertTrue(((result.predictions >= 1) & (result.predictions <= 5)).all())
                self.assertNotEqual(fitted.initial_state_hash, fitted.final_state_hash)
                self.assertNotIn("predictions", fitted.audit)
                self.assertNotIn("targets", fitted.audit)
                self.assertFalse(result.audit["gold_consumed"])
                if spec.family != "r0_anchored_distributional":
                    cap = float(spec.parameters["cap"])
                    self.assertLessEqual(float(np.max(np.abs(result.predictions - predict_b))), cap + 1e-5)
                else:
                    max_mix = float(spec.parameters["max_mix"])
                    self.assertLessEqual(float(np.max(np.abs(result.predictions - predict_b))), 4 * max_mix + 1e-5)

    def test_deterministic_fresh_fit_hashes_and_predictions(self) -> None:
        values = _tiny()
        first = fit(candidate_specs()[0], *values[:3])
        second = fit(candidate_specs()[0], *values[:3])
        self.assertIsNot(first.model, second.model)
        self.assertEqual(first.initial_state_hash, second.initial_state_hash)
        self.assertEqual(first.final_state_hash, second.final_state_hash)
        np.testing.assert_array_equal(apply(first, *values[3:]).predictions,
                                      apply(second, *values[3:]).predictions)

    def test_average_target_and_non_three_axis_scores_are_rejected(self) -> None:
        train_e, train_b, train_y, predict_e, predict_b = _tiny()
        spec = candidate_specs()[0]
        with self.assertRaisesRegex(ValueError, "average target is forbidden"):
            fit(spec, train_e, train_b, np.column_stack((train_y, train_y.mean(1))))
        fitted = fit(spec, train_e, train_b, train_y)
        with self.assertRaisesRegex(ValueError, "average target is forbidden"):
            apply(fitted, predict_e, np.column_stack((predict_b, predict_b.mean(1))))

    def test_apply_has_no_gold_parameter_and_zero_delta_is_identity(self) -> None:
        self.assertNotIn("gold", inspect.signature(apply).parameters)
        self.assertNotIn("target", inspect.signature(apply).parameters)
        train_e, train_b, train_y, predict_e, predict_b = _tiny()
        fitted = fit(candidate_specs()[0], train_e, train_b, train_y)
        for parameter in fitted.model.parameters():
            parameter.data.zero_()
        np.testing.assert_allclose(apply(fitted, predict_e, predict_b).predictions, predict_b,
                                   rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
