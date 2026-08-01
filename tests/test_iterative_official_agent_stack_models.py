import unittest
from dataclasses import dataclass

import numpy as np

from mal2026.iterative_official_agent_stack_models import (
    AXES, FEATURE_DIM, build_agent_score_features, candidate_specs, fit_predict_agent_stack,
)


@dataclass(frozen=True)
class Candidate:
    source_id: str
    candidate_number: int
    scores: dict[str, int]


def candidates(ids):
    return [
        Candidate(source_id, number, {axis: 2 + ((row + number + column) % 3) for column, axis in enumerate(AXES)})
        for row, source_id in enumerate(ids) for number in (1, 2, 3)
    ]


class OfficialAgentStackModelsTest(unittest.TestCase):
    def test_exact_three_candidate_inventory(self):
        specs = candidate_specs()
        self.assertEqual([spec.cycle for spec in specs], [1, 2, 3])
        self.assertEqual(len({spec.variant_id for spec in specs}), 3)
        self.assertEqual(specs[0].ridge_alpha, 10.0)
        self.assertEqual(specs[0].max_correction, 0.5)

    def test_fixed_feature_shape_order_and_determinism(self):
        ids = tuple(f"s{row}" for row in range(5))
        base = np.full((5, 3), 3.5)
        first, audit1 = build_agent_score_features(base, ids, candidates(ids))
        second, audit2 = build_agent_score_features(base, ids, list(reversed(candidates(ids))))
        self.assertEqual(first.shape, (5, FEATURE_DIM))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(audit1, audit2)
        self.assertFalse(audit1["human_or_reference_score_read_or_prompted"])

    def test_feature_builder_rejects_missing_or_average(self):
        ids = ("a", "b")
        base = np.full((2, 3), 3.0)
        with self.assertRaises(ValueError):
            build_agent_score_features(base, ids, candidates(ids)[:-1])
        bad = candidates(ids)
        bad[0] = Candidate("a", 1, {**bad[0].scores, "average": 3})
        with self.assertRaises(ValueError):
            build_agent_score_features(base, ids, bad)

    def test_fresh_ridge_is_deterministic_and_bounded(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(30, FEATURE_DIM))
        z = rng.normal(size=(8, FEATURE_DIM))
        base = np.full((30, 3), 3.0)
        target = np.clip(base + rng.normal(scale=0.4, size=(30, 3)), 1, 5)
        predict_base = np.full((8, 3), 3.0)
        spec = candidate_specs()[2]
        first = fit_predict_agent_stack(spec, x, base, target, z, predict_base, device="cpu")
        second = fit_predict_agent_stack(spec, x, base, target, z, predict_base, device="cpu")
        np.testing.assert_array_equal(first.predictions, second.predictions)
        self.assertLessEqual(float(np.max(np.abs(first.predictions - predict_base))), spec.max_correction + 1e-12)
        self.assertEqual(first.audit["coefficient_sha256"], second.audit["coefficient_sha256"])
        self.assertFalse(first.audit["checkpoint_reused"])

    def test_invalid_shapes_fail_closed(self):
        spec = candidate_specs()[0]
        with self.assertRaises(ValueError):
            fit_predict_agent_stack(spec, np.zeros((4, 38)), np.ones((4, 3)), np.ones((4, 3)), np.zeros((2, 38)), np.ones((2, 3)), device="cpu")


if __name__ == "__main__":
    unittest.main()
