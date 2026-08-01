from __future__ import annotations

import unittest

import numpy as np

from mal2026.iterative_tail_remediation_models import (
    FAMILIES,
    PREDECLARED_GRIDS,
    RemediationSpec,
    apply_score_conditional_gate,
    apply_tail_boundary_adjustment,
    fit_predict,
    gold_band_equal_weights,
    weighted_pava,
)


def synthetic() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(20260801)
    base = np.tile(np.linspace(1.2, 4.8, 20)[:, None], (1, 3))
    gold = np.clip(base + np.column_stack((
        0.08 * np.sin(base[:, 0]),
        0.10 * np.cos(base[:, 1]),
        0.06 * np.sin(2 * base[:, 2]),
    )), 1.0, 5.0)
    challenger = np.clip(base + np.where(base <= 2.5, 0.25, 0.08), 1.0, 5.0)
    test_base = rng.uniform(1.0, 5.0, size=(7, 3))
    test_challenger = np.clip(test_base + np.where(test_base <= 2.5, 0.25, 0.08), 1.0, 5.0)
    return tuple(np.asarray(value, dtype=np.float64) for value in (gold, base, challenger, test_base, test_challenger))


class RemediationModelTests(unittest.TestCase):
    def test_every_family_smokes_deterministically(self) -> None:
        values = synthetic()
        for family in FAMILIES:
            with self.subTest(family=family):
                spec = RemediationSpec(family=family, equal_gold_band_weights=True)
                first = fit_predict(spec, *values)
                second = fit_predict(spec, *values)
                np.testing.assert_array_equal(first.predictions, second.predictions)
                self.assertEqual((7, 3), first.predictions.shape)
                self.assertEqual(np.float32, first.predictions.dtype)
                self.assertTrue(np.isfinite(first.predictions).all())
                self.assertTrue(((first.predictions >= 1.0) & (first.predictions <= 5.0)).all())
                self.assertEqual(3, len(first.selected_parameters))
                self.assertGreaterEqual(first.train_objective, 0.0)

    def test_gate_can_make_low_scores_exact_identity(self) -> None:
        base = np.asarray([1.5, 2.5, 3.0, 4.0])
        challenger = np.asarray([2.0, 3.0, 3.5, 4.5])
        output = apply_score_conditional_gate(
            base, challenger, kind="sigmoid", threshold=3.0, temperature=0.5,
            weight=1.0, low_identity_threshold=2.5,
        )
        np.testing.assert_array_equal(output[:2], base[:2])
        self.assertGreater(output[2], base[2])
        self.assertGreater(output[3], base[3])

    def test_weighted_pava_and_calibrators_are_monotonic(self) -> None:
        projected = weighted_pava([3.0, 2.0, 1.0, 4.0], [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(projected, [2.0, 2.0, 2.0, 4.0])
        gold, base, challenger, _, _ = synthetic()
        ordered = np.linspace(1.0, 5.0, 41)
        test = np.tile(ordered[:, None], (1, 3))
        for family in ("weighted_isotonic", "piecewise_5knot"):
            result = fit_predict(RemediationSpec(family=family), gold, base, challenger, test, test)
            self.assertTrue(np.all(np.diff(result.predictions, axis=0) >= -1e-7))
            for axis_parameters in result.selected_parameters:
                self.assertTrue(np.all(np.diff(axis_parameters["y_knots"]) >= -1e-12))

    def test_piecewise_solver_recovers_noninteger_linear_interpolation(self) -> None:
        x = np.linspace(1.0, 5.0, 81)
        true_knots = np.asarray([1.1, 1.7, 2.8, 4.4, 4.9])
        y = np.interp(x, np.arange(1.0, 6.0), true_knots)
        matrix_x = np.tile(x[:, None], (1, 3))
        matrix_y = np.tile(y[:, None], (1, 3))
        result = fit_predict(
            RemediationSpec(family="piecewise_5knot"),
            matrix_y, matrix_x, matrix_x, matrix_x, matrix_x,
        )
        np.testing.assert_allclose(result.predictions[:, 0], y, atol=1e-6)
        for parameters in result.selected_parameters:
            np.testing.assert_allclose(parameters["y_knots"], true_knots, atol=1e-6)

    def test_tail_offsets_and_boundary_nudge_are_bounded(self) -> None:
        score = np.asarray([1.1, 2.2, 3.25, 3.75, 4.8])
        adjusted = apply_tail_boundary_adjustment(
            score, low_offset=-0.2, high_offset=0.2, boundary_nudge=0.1,
        )
        self.assertLessEqual(float(np.max(np.abs(adjusted - score))), 0.2 + 1e-12)
        self.assertLess(adjusted[2], score[2])
        self.assertGreater(adjusted[3], score[3])
        self.assertTrue(((adjusted >= 1.0) & (adjusted <= 5.0)).all())

    def test_convex_blend_selects_identity_when_challenger_is_worse(self) -> None:
        gold, base, _, test_base, _ = synthetic()
        bad = np.clip(base + 1.0, 1.0, 5.0)
        test_bad = np.clip(test_base + 1.0, 1.0, 5.0)
        result = fit_predict(RemediationSpec(family="convex_blend"), gold, base, bad, test_base, test_bad)
        for selected in result.selected_parameters:
            self.assertIn(selected["weight"], PREDECLARED_GRIDS["blend_weight"])
            self.assertLessEqual(selected["weight"], 0.1)

    def test_gold_band_weights_equalize_observed_band_mass(self) -> None:
        gold = np.asarray([[2.0, 2.0, 2.0]] * 3 + [[4.0, 4.0, 4.0]])
        weights = gold_band_equal_weights(gold)
        for axis in range(3):
            self.assertAlmostEqual(weights[:3, axis].sum(), weights[3:, axis].sum())

    def test_four_column_average_target_is_rejected(self) -> None:
        gold, base, challenger, test_base, test_challenger = synthetic()
        forbidden = np.column_stack((gold, gold.mean(axis=1)))
        with self.assertRaisesRegex(ValueError, "average is forbidden"):
            fit_predict(
                RemediationSpec(family="gated_delta"),
                forbidden, base, challenger, test_base, test_challenger,
            )


if __name__ == "__main__":
    unittest.main()
