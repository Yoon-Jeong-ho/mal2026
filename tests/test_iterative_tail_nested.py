"""Synthetic, row-content-free tests for leakage-guarded nested evaluation."""
from __future__ import annotations

import unittest

import numpy as np

from mal2026.iterative_tail_nested import (
    Candidate,
    NestedEvaluationError,
    make_validation_folds,
    nested_tail_evaluation,
)


def _truth() -> np.ndarray:
    values = np.tile(np.arange(1.0, 6.0), 10)
    return np.column_stack((values, np.roll(values, 1), np.roll(values, 2)))


class IterativeTailNestedTests(unittest.TestCase):
    def test_nested_selection_never_passes_outer_indices_to_inner_callbacks(self) -> None:
        truth = _truth()
        outer = make_validation_folds(np.arange(len(truth)), 5, 7)
        calls: list[tuple[str, set[int], set[int]]] = []

        def callback(name: str, error: float):
            def fit_predict(fit_idx: np.ndarray, predict_idx: np.ndarray) -> np.ndarray:
                calls.append((name, set(fit_idx.tolist()), set(predict_idx.tolist())))
                return truth[predict_idx] + error
            return fit_predict

        result = nested_tail_evaluation(
            truth,
            Candidate("identity-baseline", callback("baseline", 1.0)),
            [Candidate("strict-improvement", callback("candidate", 0.0))],
            outer_validation_folds=outer,
            seed=11,
            bootstrap_resamples=50,
        )
        self.assertTrue(all(fold["selected_candidate"] == "strict-improvement" for fold in result["folds"]))
        self.assertTrue(all(fold["outer_gold_used_for_inner_selection"] is False for fold in result["folds"]))
        self.assertTrue(result["final_gate_pass"])
        self.assertEqual({"estimate": -1.0, "lower": -1.0, "upper": -1.0}, result["candidate_minus_baseline_rmse_ci"])
        self.assertEqual(50, result["paired_row_bootstrap"]["n_resamples"])
        self.assertFalse(result["predictions_returned"])

        inner_calls = [call for call in calls if len(call[1]) == 30 and len(call[2]) == 10]
        self.assertEqual(5 * 4 * 2, len(inner_calls))
        outer_sets = [set(fold.tolist()) for fold in outer]
        for _, fit_indices, predict_indices in inner_calls:
            self.assertTrue(fit_indices.isdisjoint(predict_indices))
            excluded = set(range(len(truth))) - fit_indices - predict_indices
            self.assertIn(excluded, outer_sets)

    def test_empty_candidates_fail_closed_to_identity_baseline(self) -> None:
        truth = _truth()
        callback_calls = 0

        def baseline(fit_idx: np.ndarray, predict_idx: np.ndarray) -> np.ndarray:
            nonlocal callback_calls
            callback_calls += 1
            return truth[predict_idx] + 0.5

        result = nested_tail_evaluation(
            truth,
            Candidate("identity-baseline", baseline),
            [],
            seed=3,
            bootstrap_resamples=20,
        )
        self.assertEqual(5, callback_calls)
        self.assertTrue(all(fold["fell_back_to_identity_baseline"] for fold in result["folds"]))
        self.assertEqual(result["baseline_metrics"], result["selected_metrics"])
        self.assertEqual({"estimate": -0.0, "lower": -0.0, "upper": -0.0}, result["candidate_minus_baseline_rmse_ci"])
        self.assertFalse(result["final_gate_pass"])

    def test_rejects_inner_partition_containing_outer_gold(self) -> None:
        truth = _truth()
        outer = make_validation_folds(np.arange(len(truth)), 5, 5)
        inner = [
            list(make_validation_folds(np.setdiff1d(np.arange(len(truth)), fold), 4, 20 + index))
            for index, fold in enumerate(outer)
        ]
        inner[0][0] = np.append(inner[0][0][1:], outer[0][0])

        def predictor(fit_idx: np.ndarray, predict_idx: np.ndarray) -> np.ndarray:
            return truth[predict_idx]

        with self.assertRaisesRegex(NestedEvaluationError, "partition only their declared universe"):
            nested_tail_evaluation(
                truth,
                Candidate("identity-baseline", predictor),
                [Candidate("candidate", predictor)],
                outer_validation_folds=outer,
                inner_validation_folds=inner,
                bootstrap_resamples=5,
            )


if __name__ == "__main__":
    unittest.main()
