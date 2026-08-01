from __future__ import annotations

import unittest

import numpy as np

from mal2026.iterative_tail_router_models import FAMILIES, apply_route, fit_route, router_specs


def synthetic() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(8128)
    truth = np.tile(np.linspace(1.0, 5.0, 15)[:, None], (1, 3))
    gold = np.clip(truth + rng.normal(0, .04, truth.shape), 1, 5)
    base = np.clip(truth + rng.normal(0, .20, truth.shape), 1, 5)
    r17 = np.clip(base + np.where(base >= 3.5, .10, .04), 1, 5)
    direct = np.clip(base + rng.normal(0, .03, base.shape), 1, 5)
    hurdle = np.clip(base + np.where(base < 2.5, -.06, np.where(base > 4.5, .08, 0)), 1, 5)
    soft = np.clip(base + np.where(base < 2.5, -.02, np.where(base > 4.5, .03, 0)), 1, 5)
    return tuple(np.asarray(value, dtype=np.float64) for value in (gold, base, r17, direct, hurdle, soft))


class RouterModelTests(unittest.TestCase):
    def test_exact_inventory(self) -> None:
        specs = router_specs()
        self.assertEqual(20, len(specs))
        self.assertEqual(list(range(1, 21)), [spec.cycle for spec in specs])
        self.assertEqual(20, len({spec.variant_id for spec in specs}))
        for index, family in enumerate(FAMILIES):
            self.assertEqual([family] * 4, [spec.family for spec in specs[index * 4 : index * 4 + 4]])

    def test_all_twenty_are_deterministic_and_fit_apply_consistent(self) -> None:
        gold, base, r17, direct, hurdle, soft = synthetic()
        for spec in router_specs():
            with self.subTest(cycle=spec.cycle):
                first = fit_route(spec, gold, base, r17, direct, hurdle, soft)
                second = fit_route(spec, gold, base, r17, direct, hurdle, soft)
                np.testing.assert_array_equal(first.train_predictions, second.train_predictions)
                self.assertEqual(dict(first.selected_parameters), dict(second.selected_parameters))
                applied = apply_route(spec, first.selected_parameters, base, r17, direct, hurdle, soft)
                np.testing.assert_array_equal(first.train_predictions, applied)
                self.assertEqual((15, 3), applied.shape)
                self.assertEqual(np.float32, applied.dtype)
                self.assertTrue(((applied >= 1) & (applied <= 5)).all())

    def test_low_protected_family_is_identity_at_or_below_two(self) -> None:
        gold, base, r17, direct, hurdle, soft = synthetic()
        base[:3] = np.asarray(([1.1] * 3, [1.7] * 3, [2.0] * 3))
        r17[:3] = direct[:3] = hurdle[:3] = soft[:3] = 4.0
        result = fit_route(router_specs()[3], gold, base, r17, direct, hurdle, soft)
        np.testing.assert_array_equal(result.train_predictions[:3], base[:3].astype(np.float32))

    def test_formal_gate_none_falls_back_to_identity(self) -> None:
        _, base, _, _, _, _ = synthetic()
        gold = base.copy()
        bad = np.clip(base + .8, 1, 5)
        result = fit_route(router_specs()[16], gold, base, bad, bad, bad, bad)
        self.assertTrue(result.audit["identity_fallback"])
        self.assertFalse(result.audit["eligible"])
        np.testing.assert_array_equal(result.train_predictions, base.astype(np.float32))

    def test_average_columns_are_rejected_by_fit_and_apply(self) -> None:
        gold, base, r17, direct, hurdle, soft = synthetic()
        forbidden = np.column_stack((base, base.mean(axis=1)))
        with self.assertRaisesRegex(ValueError, "average is forbidden"):
            fit_route(router_specs()[0], np.column_stack((gold, gold.mean(axis=1))), base, r17, direct, hurdle, soft)
        result = fit_route(router_specs()[0], gold, base, r17, direct, hurdle, soft)
        with self.assertRaisesRegex(ValueError, "average is forbidden"):
            apply_route(router_specs()[0], result.selected_parameters, forbidden, r17, direct, hurdle, soft)


if __name__ == "__main__":
    unittest.main()
