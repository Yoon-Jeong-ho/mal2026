import unittest
from unittest.mock import patch

import numpy as np

from mal2026.iterative_official_agent_stack_models import AgentStackFitResult, FEATURE_DIM
from mal2026.iterative_official_selective_flip_models import (
    candidate_specs,
    fit_predict_selective_flip,
)


class OfficialSelectiveFlipModelsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.x = rng.normal(size=(80, FEATURE_DIM))
        self.z = rng.normal(size=(12, FEATURE_DIM))
        self.base = np.full((80, 3), 3.5)
        classes = np.tile(np.array([2.0, 3.0, 4.0, 5.0]), 20)
        self.targets = np.column_stack((classes, np.roll(classes, 1), np.roll(classes, 2)))
        self.predict_base = np.full((12, 3), 3.5)

    def test_exact_inventory(self):
        specs = candidate_specs()
        self.assertEqual([spec.cycle for spec in specs], [1, 2, 3])
        self.assertEqual(
            [spec.head_kind for spec in specs],
            ["adjacent_3v4", "adjacent_3v4", "dual_average"],
        )
        self.assertEqual([spec.confidence for spec in specs], [0.60, 0.65, 0.60])
        self.assertEqual([spec.window for spec in specs], [0.20, 0.15, 0.20])
        self.assertEqual(len({spec.variant_id for spec in specs}), 3)

    def test_each_candidate_is_fresh_deterministic_and_bounded(self):
        for spec in candidate_specs():
            first = fit_predict_selective_flip(
                spec, self.x, self.base, self.targets, self.z, self.predict_base, device="cpu"
            )
            second = fit_predict_selective_flip(
                spec, self.x, self.base, self.targets, self.z, self.predict_base, device="cpu"
            )
            np.testing.assert_allclose(first.predictions, second.predictions, rtol=0, atol=1e-10)
            self.assertLessEqual(first.audit["max_abs_flip_correction"], spec.window + spec.epsilon + 1e-12)
            self.assertTrue(first.audit["fresh_initialization"])
            self.assertFalse(first.audit["checkpoint_reused"])
            self.assertEqual(first.audit["prediction_cells"], first.predictions.size)
            for head in first.audit["heads"].values():
                self.assertTrue(head["fresh_zero_initialization"])
                self.assertFalse(head["checkpoint_reused"])
                self.assertEqual(len(head["axis_coefficient_sha256"]), 3)

    def test_flip_requires_both_window_and_high_confidence_disagreement(self):
        spec = candidate_specs()[0]
        primary = np.array(
            [[3.31, 3.29, 3.40], [3.69, 3.71, 3.60], [3.40, 3.60, 3.50]],
            dtype=np.float64,
        )
        probabilities = np.array(
            [[0.70, 0.70, 0.59], [0.30, 0.30, 0.41], [0.59, 0.41, 0.39]],
            dtype=np.float64,
        )
        residual = AgentStackFitResult(
            primary,
            {"coefficient_sha256": "a" * 64, "fresh_closed_form_solve": True},
        )
        head_audit = {
            "fresh_zero_initialization": True,
            "checkpoint_reused": False,
            "axis_coefficient_sha256": ["b" * 64] * 3,
        }
        with patch(
            "mal2026.iterative_official_selective_flip_models.fit_predict_agent_stack",
            return_value=residual,
        ), patch(
            "mal2026.iterative_official_selective_flip_models._fit_logistic_heads",
            return_value=(probabilities, head_audit),
        ):
            result = fit_predict_selective_flip(
                spec,
                np.zeros((4, FEATURE_DIM)),
                np.full((4, 3), 3.5),
                np.full((4, 3), 3.0),
                np.zeros((3, FEATURE_DIM)),
                np.full((3, 3), 3.5),
                device="cpu",
            )
        expected = primary.copy()
        expected[0, 0] = 3.501  # within lower window and confident class 4
        expected[1, 0] = 3.499  # within upper window and confident class 3
        expected[2, 2] = 3.499  # exact boundary is treated as upper side
        np.testing.assert_allclose(result.predictions, expected, rtol=0, atol=1e-12)
        self.assertEqual(result.audit["upward_flip_cells"], 1)
        self.assertEqual(result.audit["downward_flip_cells"], 2)
        self.assertEqual(result.audit["total_flip_cells"], 3)


if __name__ == "__main__":
    unittest.main()
