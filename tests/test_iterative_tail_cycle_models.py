from __future__ import annotations

import unittest

import numpy as np

from mal2026.iterative_tail_cycle_models import FAMILIES, CycleSpec, cycle_specs, fit_predict


def synthetic() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(31415)
    score = np.tile(np.linspace(1.0, 5.0, 25)[:, None], (1, 3))
    base = np.clip(score + rng.normal(0.0, 0.20, size=score.shape), 1.0, 5.0)
    gold = np.clip(score + rng.normal(0.0, 0.06, size=score.shape), 1.0, 5.0)
    r17 = np.clip(base + np.where(base > 3.5, 0.10, 0.04), 1.0, 5.0)
    direct = np.clip(base + rng.normal(0.0, 0.04, size=base.shape), 1.0, 5.0)
    evidence = rng.normal(size=(len(base), 12))
    test_base = rng.uniform(1.0, 5.0, size=(7, 3))
    test_r17 = np.clip(test_base + np.where(test_base > 3.5, 0.10, 0.04), 1.0, 5.0)
    test_direct = np.clip(test_base + rng.normal(0.0, 0.04, size=test_base.shape), 1.0, 5.0)
    test_evidence = rng.normal(size=(len(test_base), 12))
    return tuple(np.asarray(value, dtype=np.float64) for value in (
        gold, base, r17, direct, evidence, test_base, test_r17, test_direct, test_evidence,
    ))


class CycleModelTests(unittest.TestCase):
    def test_exact_twenty_cycle_inventory(self) -> None:
        specs = cycle_specs()
        self.assertEqual(20, len(specs))
        self.assertEqual(list(range(1, 21)), [spec.cycle for spec in specs])
        self.assertEqual(20, len({spec.variant_id for spec in specs}))
        for family_index, family in enumerate(FAMILIES):
            family_specs = specs[family_index * 4 : family_index * 4 + 4]
            self.assertEqual([family] * 4, [spec.family for spec in family_specs])

    def test_all_families_smoke_deterministically(self) -> None:
        values = synthetic()
        for spec in cycle_specs()[::4]:
            with self.subTest(family=spec.family):
                first = fit_predict(spec, *values)
                second = fit_predict(spec, *values)
                np.testing.assert_array_equal(first.predictions, second.predictions)
                self.assertEqual((7, 3), first.predictions.shape)
                self.assertEqual(np.float32, first.predictions.dtype)
                self.assertTrue(np.isfinite(first.predictions).all())
                self.assertTrue(((first.predictions >= 1.0) & (first.predictions <= 5.0)).all())
                self.assertTrue(first.audit["fresh_initialization"])
                self.assertFalse(first.audit["checkpoint_reused"])
                self.assertFalse(first.audit["average_target_used"])
                self.assertEqual(dict(first.audit), dict(second.audit))

    def test_pareto_infeasible_falls_back_to_identity(self) -> None:
        gold, base, _, _, evidence, test_base, _, _, test_evidence = synthetic()
        gold = base.copy()
        bad_train = np.clip(base + 1.0, 1.0, 5.0)
        bad_test = np.clip(test_base + 1.0, 1.0, 5.0)
        spec = cycle_specs()[4]
        result = fit_predict(
            spec, gold, base, bad_train, bad_train, evidence,
            test_base, bad_test, bad_test, test_evidence,
        )
        self.assertFalse(result.audit["pareto_feasible"])
        self.assertTrue(result.audit["fallback_identity"])
        np.testing.assert_array_equal(result.predictions, test_base.astype(np.float32))

    def test_average_column_is_rejected(self) -> None:
        values = list(synthetic())
        values[0] = np.column_stack((values[0], values[0].mean(axis=1)))
        with self.assertRaisesRegex(ValueError, "average is forbidden"):
            fit_predict(cycle_specs()[0], *values)

    def test_unregistered_or_modified_spec_is_rejected(self) -> None:
        values = synthetic()
        registered = cycle_specs()[0]
        changed = CycleSpec(registered.cycle, registered.family, registered.variant_id, {**registered.parameters, "cap": 0.9})
        with self.assertRaisesRegex(ValueError, "fixed inventory"):
            fit_predict(changed, *values)


if __name__ == "__main__":
    unittest.main()
