from __future__ import annotations

from copy import deepcopy
import unittest

from mal2026.iterative_tail_cycle_protocol import CycleProtocol, load_protocol
from mal2026.iterative_tail_cycle_runner import (
    DIRECT_ALPHA,
    IterativeTailCycleRunError,
    _macro_band,
    _promotion_decision,
    _protocol_direct_alpha,
)


def _metrics(rmse, equal, low, high, ba, spearman, axis_rmse=None):
    axis_value = rmse if axis_rmse is None else axis_rmse
    return {
        "axes": {
            axis: {
                "rmse": axis_value,
                "bands": {
                    str(band): {"rmse": float(band), "recall": band / 10.0, "one_off": 1.0}
                    for band in range(1, 6)
                },
            }
            for axis in ("content", "organization", "expression")
        },
        "macro": {
            "rmse": rmse,
            "equal_group_rmse": equal,
            "low_tail_rmse": low,
            "high_tail_rmse": high,
            "gold_3_4_balanced_accuracy": ba,
            "spearman": spearman,
        },
    }


class IterativeTailCycleRunnerTests(unittest.TestCase):
    def test_all_seven_gates_are_conjunctive_and_score1_is_descriptive(self):
        protocol = load_protocol()
        baseline = _metrics(.60, .80, 1.0, 1.0, .50, .50)
        candidate = _metrics(.59, .78, .99, .99, .52, .50)
        decision = _promotion_decision(protocol, baseline, candidate)
        self.assertTrue(decision["eligible"])
        self.assertEqual(7, len(decision["gates"]))
        self.assertTrue(all(decision["gates"].values()))
        self.assertFalse(decision["score1_used_for_promotion"])

        misses_equal = _metrics(.59, .795, .99, .99, .52, .50)
        rejected = _promotion_decision(protocol, baseline, misses_equal)
        self.assertFalse(rejected["eligible"])
        self.assertFalse(rejected["gates"]["equal_group_rmse_improvement"])

    def test_direct_alpha_is_exactly_bound_to_protocol(self):
        protocol = load_protocol()
        self.assertEqual(100.0, DIRECT_ALPHA)
        self.assertEqual(DIRECT_ALPHA, _protocol_direct_alpha(protocol))
        raw = deepcopy(protocol.raw)
        raw["fold_protocol"]["direct_evidence_ridge_challenger"]["ridge_alpha"] = 10.0
        altered = CycleProtocol(protocol.path, raw)
        with self.assertRaisesRegex(IterativeTailCycleRunError, "alpha differs"):
            _protocol_direct_alpha(altered)

    def test_macro_band_averages_only_the_three_axes(self):
        metrics = _metrics(.60, .80, 1.0, 1.0, .50, .50)
        self.assertEqual(2.0, _macro_band(metrics, 2, "rmse"))
        self.assertAlmostEqual(.2, _macro_band(metrics, 2, "recall"))


if __name__ == "__main__":
    unittest.main()
