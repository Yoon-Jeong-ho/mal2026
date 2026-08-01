from __future__ import annotations

import unittest

import numpy as np

from mal2026.iterative_official_rationale_semantic_models import (
    FUSION_DIM,
    SEMANTIC_DIM,
    STRUCTURED_DIM,
    candidate_specs,
    fit_predict_rationale_semantic,
)


class RationaleSemanticModelsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1212)
        self.semantic = rng.normal(size=(80, SEMANTIC_DIM))
        self.structured = rng.normal(size=(80, STRUCTURED_DIM))
        self.psemantic = rng.normal(size=(12, SEMANTIC_DIM))
        self.pstructured = rng.normal(size=(12, STRUCTURED_DIM))
        self.base = rng.uniform(1.2, 4.8, size=(80, 3))
        self.targets = np.clip(self.base + .12 * np.tanh(self.semantic[:, :3]), 1, 5)
        self.pbase = rng.uniform(1.2, 4.8, size=(12, 3))

    def fit(self, spec, psemantic=None, pstructured=None, pbase=None):
        return fit_predict_rationale_semantic(
            spec, self.semantic, self.structured, self.base, self.targets,
            self.psemantic if psemantic is None else psemantic,
            self.pstructured if pstructured is None else pstructured,
            self.pbase if pbase is None else pbase,
            device="cpu",
        )

    def test_exact_inventory_and_dimensions(self):
        specs = candidate_specs()
        self.assertEqual([1, 2, 3], [spec.cycle for spec in specs])
        self.assertEqual(["semantic201", "fusion297", "fusion297"], [spec.feature_kind for spec in specs])
        self.assertEqual(["identity", "identity", "balanced_adjacent_3v4"], [spec.head_kind for spec in specs])
        self.assertEqual(297, FUSION_DIM)
        self.assertEqual(3, len({spec.variant_id for spec in specs}))

    def test_all_candidates_are_fresh_deterministic_float64_and_bounded(self):
        for spec in candidate_specs():
            with self.subTest(cycle=spec.cycle):
                first, second = self.fit(spec), self.fit(spec)
                np.testing.assert_allclose(first.predictions, second.predictions, rtol=0, atol=1e-10)
                self.assertEqual(np.float64, first.predictions.dtype)
                self.assertTrue(((first.predictions >= 1) & (first.predictions <= 5)).all())
                self.assertTrue(first.audit["fresh_initialization"])
                self.assertFalse(first.audit["checkpoint_reused"])
                self.assertTrue(first.audit["residual"]["fresh_closed_form_solve"])
                expected = SEMANTIC_DIM if spec.cycle == 1 else FUSION_DIM
                self.assertEqual(expected, first.audit["residual"]["feature_dimensions"])

    def test_predict_distribution_cannot_change_fit_only_standardization(self):
        spec = candidate_specs()[1]
        single = self.fit(spec, self.psemantic[:1], self.pstructured[:1], self.pbase[:1])
        extreme_semantic = np.vstack((self.psemantic[:1], np.full((1, SEMANTIC_DIM), 1e6)))
        extreme_structured = np.vstack((self.pstructured[:1], np.full((1, STRUCTURED_DIM), -1e6)))
        extreme_base = np.vstack((self.pbase[:1], [[3, 3, 3]]))
        batched = self.fit(spec, extreme_semantic, extreme_structured, extreme_base)
        np.testing.assert_allclose(single.predictions[0], batched.predictions[0], rtol=0, atol=1e-12)
        self.assertEqual("fit_partition_only", batched.audit["residual"]["normalization_fit_scope"])

    def test_balanced_head_equalizes_classes_and_hard_flips_exactly(self):
        rng = np.random.default_rng(7)
        n = 80
        labels = np.tile(np.asarray([3.0, 4.0]), n // 2)
        semantic = rng.normal(scale=.01, size=(n, SEMANTIC_DIM))
        structured = rng.normal(scale=.01, size=(n, STRUCTURED_DIM))
        structured[:, 0] = np.where(labels == 4, 4.0, -4.0)
        targets = np.tile(labels[:, None], (1, 3))
        base = targets.copy()
        psemantic = np.zeros((2, SEMANTIC_DIM))
        pstructured = np.zeros((2, STRUCTURED_DIM)); pstructured[:, 0] = (4.0, -4.0)
        pbase = np.asarray([[3.49] * 3, [3.51] * 3])
        result = fit_predict_rationale_semantic(
            candidate_specs()[2], semantic, structured, base, targets,
            psemantic, pstructured, pbase, device="cpu",
        )
        np.testing.assert_array_equal(result.predictions[0], np.asarray([3.501] * 3))
        np.testing.assert_array_equal(result.predictions[1], np.asarray([3.499] * 3))
        head = result.audit["heads"]["class_balanced_adjacent_3v4"]
        self.assertTrue(head["equal_total_class_weight"])
        self.assertTrue(head["fresh_zero_initialization"])
        for audit in head["axis_class_weight_audit"]:
            self.assertAlmostEqual(audit["negative_total"], audit["positive_total"], places=12)

    def test_shapes_nonfinite_and_average_column_are_rejected(self):
        spec = candidate_specs()[0]
        with self.assertRaisesRegex(ValueError, "train targets"):
            fit_predict_rationale_semantic(
                spec, self.semantic, self.structured, self.base,
                np.column_stack((self.targets, self.targets.mean(1))),
                self.psemantic, self.pstructured, self.pbase, device="cpu",
            )
        broken = self.semantic.copy(); broken[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_predict_rationale_semantic(
                spec, broken, self.structured, self.base, self.targets,
                self.psemantic, self.pstructured, self.pbase, device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
