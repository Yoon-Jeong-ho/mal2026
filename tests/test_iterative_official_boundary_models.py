import unittest

import numpy as np

from mal2026.iterative_official_agent_stack_models import FEATURE_DIM
from mal2026.iterative_official_boundary_models import candidate_specs, fit_predict_official_boundary


class OfficialBoundaryModelsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.x = rng.normal(size=(80, FEATURE_DIM))
        self.z = rng.normal(size=(12, FEATURE_DIM))
        self.base = np.full((80, 3), 3.5)
        # Guaranteed bands 3 and 4 plus low/high examples for threshold heads.
        classes = np.tile(np.array([2.0, 3.0, 4.0, 5.0]), 20)
        self.targets = np.column_stack((classes, np.roll(classes, 1), np.roll(classes, 2)))
        self.predict_base = np.full((12, 3), 3.5)

    def test_exact_inventory(self):
        specs = candidate_specs()
        self.assertEqual([spec.cycle for spec in specs], [1, 2, 3])
        self.assertEqual([spec.head_kind for spec in specs], ["adjacent_3v4", "threshold_ge4", "dual_average"])
        self.assertEqual(len({spec.variant_id for spec in specs}), 3)

    def test_each_head_is_fresh_deterministic_and_bounded(self):
        for spec in candidate_specs():
            first = fit_predict_official_boundary(spec, self.x, self.base, self.targets, self.z, self.predict_base, device="cpu")
            second = fit_predict_official_boundary(spec, self.x, self.base, self.targets, self.z, self.predict_base, device="cpu")
            np.testing.assert_allclose(first.predictions, second.predictions, rtol=0, atol=1e-10)
            self.assertLessEqual(first.audit["max_abs_boundary_correction"], spec.nudge + 1e-12)
            self.assertTrue(first.audit["fresh_initialization"])
            self.assertFalse(first.audit["checkpoint_reused"])
            for head in first.audit["heads"].values():
                self.assertTrue(head["fresh_zero_initialization"])
                self.assertFalse(head["checkpoint_reused"])
                self.assertEqual(len(head["axis_coefficient_sha256"]), 3)

    def test_adjacent_and_threshold_training_populations_differ(self):
        adjacent = fit_predict_official_boundary(candidate_specs()[0], self.x, self.base, self.targets, self.z, self.predict_base, device="cpu")
        threshold = fit_predict_official_boundary(candidate_specs()[1], self.x, self.base, self.targets, self.z, self.predict_base, device="cpu")
        adjacent_count = adjacent.audit["heads"]["adjacent_3v4"]["axis_label_counts"][0]["records"]
        threshold_count = threshold.audit["heads"]["threshold_ge4"]["axis_label_counts"][0]["records"]
        self.assertEqual(adjacent_count, 40)
        self.assertEqual(threshold_count, 80)


if __name__ == "__main__":
    unittest.main()
