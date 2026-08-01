import unittest

import numpy as np

from mal2026.iterative_official_balanced_boundary_models import (
    candidate_specs,
    fit_predict_balanced_boundary,
)
from mal2026.iterative_official_dual_agent_models import FEATURE_DIM


class OfficialBalancedBoundaryModelsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.x = rng.normal(size=(80, FEATURE_DIM))
        self.z = rng.normal(size=(12, FEATURE_DIM))
        self.base = np.full((80, 3), 3.5)
        classes = np.tile(np.array([2.0, 3.0, 4.0, 5.0]), 20)
        self.targets = np.column_stack((classes, np.roll(classes, 1), np.roll(classes, 2)))
        self.pbase = np.full((12, 3), 3.5)

    def test_exact_inventory(self):
        specs = candidate_specs()
        self.assertEqual([spec.cycle for spec in specs], [1, 2, 3])
        self.assertEqual([spec.l2 for spec in specs], [0.01, 0.10, 0.01])
        self.assertEqual([spec.confidence for spec in specs], [0.50, 0.50, 0.55])
        self.assertEqual([spec.window for spec in specs], [0.25, 0.25, 0.20])
        self.assertEqual(len({spec.variant_id for spec in specs}), 3)

    def test_each_candidate_is_fresh_deterministic_bounded_and_class_balanced(self):
        for spec in candidate_specs():
            first = fit_predict_balanced_boundary(
                spec, self.x, self.base, self.targets, self.z, self.pbase, device="cpu"
            )
            second = fit_predict_balanced_boundary(
                spec, self.x, self.base, self.targets, self.z, self.pbase, device="cpu"
            )
            np.testing.assert_allclose(first.predictions, second.predictions, rtol=0, atol=1e-10)
            self.assertLessEqual(first.audit["max_abs_flip_correction"], spec.window + spec.epsilon + 1e-12)
            self.assertTrue(first.audit["fresh_initialization"])
            self.assertFalse(first.audit["checkpoint_reused"])
            self.assertEqual(len(first.audit["residual"]["coefficient_sha256"]), 64)
            head = first.audit["heads"]["class_balanced_adjacent_3v4"]
            self.assertTrue(head["equal_total_class_weight"])
            self.assertTrue(head["fresh_zero_initialization"])
            self.assertEqual(len(head["axis_coefficient_sha256"]), 3)
            for weight in head["axis_class_weight_audit"]:
                self.assertAlmostEqual(weight["negative_total"], weight["positive_total"], places=10)


if __name__ == "__main__":
    unittest.main()
