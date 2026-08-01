"""Synthetic-only unittest checks for iterative tail metrics."""
from __future__ import annotations

import math
import unittest

from mal2026.iterative_tail_metrics import (
    IterativeMetricError,
    compute_iterative_tail_metrics,
    half_up_band,
    paired_bootstrap_delta_ci,
    promotion_decision,
)


def _repeat_axis(values):
    return [[value, value, value] for value in values]


class IterativeTailMetricTests(unittest.TestCase):
    def test_half_up_bands_and_no_average_target(self):
        self.assertEqual([1, 1, 2, 3, 4, 5, 5], [half_up_band(value) for value in (0, 1.49, 1.5, 2.5, 4.49, 4.5, 6)])
        with self.assertRaisesRegex(IterativeMetricError, "average is forbidden"):
            compute_iterative_tail_metrics(
                {"content": [1], "organization": [1], "expression": [1], "average": [1]},
                {"content": [1], "organization": [1], "expression": [1]},
            )

    def test_axis_band_tail_equal_group_and_transition_metrics(self):
        result = compute_iterative_tail_metrics(
            _repeat_axis([1, 2, 3, 4, 5]), _repeat_axis([1.4, 3, 4, 3, 4])
        )
        axis = result["axes"]["content"]
        self.assertEqual(5, result["record_count"])
        self.assertAlmostEqual(0.4, axis["bands"]["1"]["rmse"])
        self.assertEqual(1.0, axis["bands"]["1"]["recall"])
        self.assertEqual(0.0, axis["bands"]["2"]["recall"])
        expected_low = math.sqrt((0.4**2 + 1) / 2)
        self.assertAlmostEqual(expected_low, axis["low_tail_rmse"])
        self.assertAlmostEqual((expected_low + 1 + 1 + 1) / 4, axis["equal_group_rmse"])
        self.assertEqual(0.0, axis["gold_3_4_balanced_accuracy"])
        self.assertEqual(1.0, axis["rate_3_to_4"])
        self.assertEqual(1.0, axis["rate_4_to_3"])

    def test_promotion_gates_and_axis_regression(self):
        truth = _repeat_axis([1, 2, 3, 3, 4, 4, 5])
        baseline = _repeat_axis([2, 3, 4, 4, 3, 3, 4])
        candidate = _repeat_axis([1.2, 2.2, 3, 3, 4, 4, 4.8])
        decision = promotion_decision(
            compute_iterative_tail_metrics(truth, baseline),
            compute_iterative_tail_metrics(truth, candidate),
        )
        self.assertTrue(decision["promote"])
        self.assertFalse(decision["score1_used_for_promotion"])

        truth = _repeat_axis([1, 2, 3, 4, 5])
        baseline = _repeat_axis([1.4, 2.4, 3.4, 4.4, 4.4])
        candidate = [[1, 1, 1.5], [2, 2, 2.5], [3, 3, 3.5], [4, 4, 4.5], [5, 5, 4.5]]
        decision = promotion_decision(
            compute_iterative_tail_metrics(truth, baseline),
            compute_iterative_tail_metrics(truth, candidate),
        )
        self.assertFalse(decision["gates"]["no_axis_rmse_worsens_more_than_0_01"])

    def test_paired_bootstrap_is_deterministic(self):
        truth = _repeat_axis([1, 2, 3, 4, 5, 3, 4, 5])
        baseline = _repeat_axis([2, 3, 4, 3, 4, 4, 3, 4])
        candidate = truth
        first = paired_bootstrap_delta_ci(truth, baseline, candidate, n_resamples=50, seed=17)
        second = paired_bootstrap_delta_ci(truth, baseline, candidate, n_resamples=50, seed=17)
        self.assertEqual(first, second)
        self.assertGreater(first["intervals"]["rmse"]["estimate"], 0)
        clustered = paired_bootstrap_delta_ci(
            truth, baseline, candidate,
            document_ids=["a", "a", "b", "b", "c", "c", "d", "d"],
            n_resamples=50, seed=17,
        )
        self.assertEqual("document", clustered["unit"])
        self.assertEqual(4, clustered["cluster_count"])


if __name__ == "__main__":
    unittest.main()
