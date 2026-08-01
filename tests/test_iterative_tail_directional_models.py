from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

import numpy as np

from mal2026.iterative_tail_directional_models import (
    BATCH_SIZE,
    CAP_CENTER,
    CAP_HIGH,
    CAP_LOW,
    EPOCHS,
    FAMILY,
    LEARNING_RATE,
    RUN_SEED,
    WEIGHT_DECAY,
    apply,
    candidate_specs,
    fit,
)


def _tiny() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(67)
    features = rng.normal(size=(12, 64)).astype(np.float32)
    base = np.asarray([
        [2.2, 4.0, 3.2], [4.0, 2.1, 4.2], [3.2, 4.0, 2.0],
        [4.1, 3.2, 4.0], [2.0, 4.1, 3.1], [3.1, 2.2, 4.1],
        [4.0, 3.1, 2.1], [2.1, 4.0, 3.2], [3.2, 2.0, 4.0],
        [4.1, 3.2, 2.2], [2.2, 4.1, 3.1], [3.1, 2.1, 4.2],
    ], dtype=np.float32)
    target = np.clip(base + np.asarray([
        [-.4, .8, -.1], [.8, -.3, .7], [-.1, .8, -.4],
        [.8, -.1, -.2], [-.4, .8, .1], [.1, -.4, .8],
        [.8, -.1, -.4], [-.4, .8, .1], [-.1, -.4, .8],
        [.8, .1, -.4], [-.4, .8, -.1], [.1, -.4, .8],
    ], dtype=np.float32), 1, 5)
    folds = np.repeat(np.arange(3), 4)
    predict_features = rng.normal(size=(4, 64)).astype(np.float32)
    predict_base = rng.uniform(1.5, 4.4, size=(4, 3)).astype(np.float32)
    return features, base, target, folds, predict_features, predict_base


class DirectionalModelsTest(unittest.TestCase):
    def test_exact_three_candidate_inventory(self) -> None:
        specs = candidate_specs()
        self.assertEqual(len(specs), 3)
        self.assertEqual([spec.cycle for spec in specs], [1, 2, 3])
        self.assertEqual({spec.family for spec in specs}, {FAMILY})
        self.assertEqual([spec.variant_id.rsplit("-", 1)[-1] for spec in specs[:2]], ["primary", "conservative"])
        self.assertTrue(specs[0].parameters["nonlinear"])
        self.assertTrue(specs[1].parameters["nonlinear"])
        self.assertFalse(specs[2].parameters["nonlinear"])
        self.assertEqual([spec.parameters["benefit_margin"] for spec in specs], [.01, .02, .01])
        self.assertEqual([spec.parameters["identity_bias"] for spec in specs], [4.0, 4.5, 4.0])
        self.assertEqual({spec.seed for spec in specs}, {RUN_SEED})
        self.assertEqual((CAP_LOW, CAP_HIGH, CAP_CENTER), (.40, .80, .08))
        self.assertEqual((BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY), (128, 30, 3e-4, 1e-3))
        with self.assertRaises(ValueError):
            fit(replace(specs[0], parameters={**specs[0].parameters, "benefit_margin": .03}), *_tiny()[:4])

    def test_all_candidates_crossfit_exact_coverage_excludes_heldout_fold(self) -> None:
        values = _tiny()
        for spec in candidate_specs():
            with self.subTest(variant=spec.variant_id):
                fitted = fit(spec, *values[:4])
                self.assertEqual(fitted.audit["internal_fold_count"], 3)
                self.assertEqual((fitted.audit["crossfit_coverage_min"], fitted.audit["crossfit_coverage_max"]), (1, 1))
                for split in fitted.audit["internal_crossfit"]:
                    self.assertNotIn(split["heldout_fold"], split["train_folds"])
                    self.assertEqual(set(split["train_folds"]), {0, 1, 2} - {split["heldout_fold"]})
                    self.assertEqual((split["train_records"], split["heldout_records"]), (8, 4))
                result = apply(fitted, *values[4:])
                self.assertEqual(result.predictions.shape, (4, 3))
                self.assertTrue(((result.predictions >= 1) & (result.predictions <= 5)).all())
                self.assertLessEqual(float(np.max(np.abs(result.predictions - values[5]))), CAP_HIGH + 1e-5)
                self.assertGreater(result.audit["mean_identity_weight"], .5)
                self.assertNotIn("targets", fitted.audit)
                self.assertFalse(result.audit["gold_consumed"])

    def test_deterministic_fresh_models_hashes_and_predictions(self) -> None:
        values = _tiny()
        first = fit(candidate_specs()[2], *values[:4])
        second = fit(candidate_specs()[2], *values[:4])
        self.assertIsNot(first.model, second.model)
        self.assertEqual(first.initial_state_hash, second.initial_state_hash)
        self.assertEqual(first.final_state_hash, second.final_state_hash)
        np.testing.assert_array_equal(apply(first, *values[4:]).predictions,
                                      apply(second, *values[4:]).predictions)

    def test_apply_is_gold_free_and_high_expert_can_cross_4_5(self) -> None:
        self.assertNotIn("gold", inspect.signature(apply).parameters)
        self.assertNotIn("target", inspect.signature(apply).parameters)
        values = _tiny()
        fitted = fit(candidate_specs()[2], *values[:4])
        # Force the high expert and demonstrate that its registered .80 cap,
        # unlike V5's global .30 cap, can cross the official 4.5 boundary.
        fitted.model.expert.head.weight.data.zero_()
        fitted.model.expert.head.bias.data.view(3, 3)[:, 1] = 20.0
        fitted.model.gate.head.weight.data.zero_()
        gate_bias = fitted.model.gate.head.bias.data.view(3, 4)
        gate_bias.fill_(-20.0)
        gate_bias[:, 2] = 20.0
        base = np.full((1, 3), 3.8, dtype=np.float32)
        prediction = apply(fitted, np.zeros((1, 64), dtype=np.float32), base).predictions
        self.assertTrue((prediction > 4.5).all())
        self.assertTrue((prediction <= base + CAP_HIGH + 1e-6).all())

    def test_identity_default_shapes_fold_contract_and_average_rejection(self) -> None:
        values = _tiny()
        spec = candidate_specs()[0]
        with self.assertRaisesRegex(ValueError, "average target is forbidden"):
            fit(spec, values[0], values[1], np.column_stack((values[2], values[2].mean(1))), values[3])
        with self.assertRaisesRegex(ValueError, "N x 64"):
            fit(spec, np.column_stack((values[0], np.ones(len(values[0])))), values[1], values[2], values[3])
        with self.assertRaisesRegex(ValueError, "at least two"):
            fit(spec, values[0], values[1], values[2], np.zeros(len(values[0]), dtype=int))
        fitted = fit(spec, *values[:4])
        self.assertGreater(apply(fitted, *values[4:]).audit["mean_identity_weight"], .5)
        with self.assertRaisesRegex(ValueError, "average target is forbidden"):
            apply(fitted, values[4], np.column_stack((values[5], values[5].mean(1))))


if __name__ == "__main__":
    unittest.main()
