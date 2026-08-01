import unittest

import numpy as np

from mal2026.iterative_official_dual_agent_models import (
    FEATURE_DIM,
    build_dual_agent_features,
    candidate_specs,
    fit_predict_dual_agent,
)
from mal2026.official_rationale_data import OfficialCandidate


def candidates(source_ids, offset=0):
    result = []
    axes = ("content", "organization", "expression")
    for row, source_id in enumerate(source_ids):
        for number in (1, 2, 3):
            values = {
                axis: 1 + ((row + number + axis_index + offset) % 5)
                for axis_index, axis in enumerate(axes)
            }
            result.append(OfficialCandidate(source_id, number, values, {axis: "근거" for axis in axes}))
    return result


class OfficialDualAgentModelsTest(unittest.TestCase):
    def test_exact_inventory(self):
        specs = candidate_specs()
        self.assertEqual([spec.cycle for spec in specs], [1, 2, 3])
        self.assertEqual([spec.head_kind for spec in specs], ["identity", "adjacent_3v4", "dual_average"])
        self.assertEqual(len({spec.variant_id for spec in specs}), 3)

    def test_fixed_dual_feature_shape_hash_and_source_order(self):
        source_ids = tuple(f"s{i}" for i in range(8))
        base = np.full((8, 3), 3.0)
        terra = candidates(source_ids, 0)
        luna = candidates(source_ids, 1)
        first, audit = build_dual_agent_features(base, source_ids, terra, luna)
        second, second_audit = build_dual_agent_features(base, source_ids, terra, luna)
        self.assertEqual(first.shape, (8, FEATURE_DIM))
        self.assertEqual(FEATURE_DIM, 96)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(audit["feature_matrix_sha256"], second_audit["feature_matrix_sha256"])
        swapped, _ = build_dual_agent_features(base, source_ids, luna, terra)
        self.assertFalse(np.array_equal(first, swapped))
        self.assertFalse(audit["human_or_reference_score_read_or_prompted"])

    def test_fresh_candidates_are_deterministic_and_flip_bounded(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(80, FEATURE_DIM))
        z = rng.normal(size=(12, FEATURE_DIM))
        base = np.full((80, 3), 3.5)
        classes = np.tile(np.array([2.0, 3.0, 4.0, 5.0]), 20)
        targets = np.column_stack((classes, np.roll(classes, 1), np.roll(classes, 2)))
        pbase = np.full((12, 3), 3.5)
        for spec in candidate_specs():
            first = fit_predict_dual_agent(spec, x, base, targets, z, pbase, device="cpu")
            second = fit_predict_dual_agent(spec, x, base, targets, z, pbase, device="cpu")
            np.testing.assert_allclose(first.predictions, second.predictions, rtol=0, atol=1e-10)
            self.assertEqual(first.predictions.shape, (12, 3))
            self.assertTrue(np.isfinite(first.predictions).all())
            self.assertTrue(first.audit["fresh_initialization"])
            self.assertFalse(first.audit["checkpoint_reused"])
            self.assertEqual(len(first.audit["residual"]["coefficient_sha256"]), 64)
            self.assertLessEqual(first.audit["max_abs_flip_correction"], spec.window + spec.epsilon + 1e-12)
            if spec.head_kind == "identity":
                self.assertEqual(first.audit["total_flip_cells"], 0)
                self.assertEqual(first.audit["heads"], {})
            else:
                for head in first.audit["heads"].values():
                    self.assertEqual(len(head["axis_coefficient_sha256"]), 3)
                    self.assertTrue(head["fresh_zero_initialization"])


if __name__ == "__main__":
    unittest.main()
