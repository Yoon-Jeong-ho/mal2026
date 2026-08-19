from __future__ import annotations

import unittest

from mal2026.rationale_aware_encoder import ContinuousScoreRow
from scripts.train_rationale_pipeline_score_encoder import (
    balanced_smoke_subset,
    exact_balance_weights,
)


class RationalePipelineScoreBalanceTests(unittest.TestCase):
    def test_regression_weights_equalize_every_axis_integer_band(self) -> None:
        labels = (
            [[1.1, 1.0, 1.2]] * 1
            + [[2.1, 2.0, 2.2]] * 2
            + [[3.1, 3.0, 3.2]] * 3
            + [[4.1, 4.0, 4.2]] * 4
            + [[5.0, 5.0, 5.0]] * 5
        )
        weights, audit = exact_balance_weights(labels, "bounded_regression")
        self.assertEqual(len(weights), len(labels))
        self.assertTrue(audit["all_axis_score_cells_equal"])
        expected = len(labels) / 5
        for axis in audit["weighted_mass"].values():
            for mass in axis.values():
                self.assertAlmostEqual(mass, expected)

    def test_classification_weights_use_zero_based_training_labels(self) -> None:
        labels = [[score - 1, score - 1, score - 1] for score in range(1, 6) for _ in range(score)]
        _, audit = exact_balance_weights(labels, "categorical_5class")
        self.assertEqual(audit["counts"]["content"], {str(score): score for score in range(1, 6)})
        self.assertTrue(audit["all_axis_score_cells_equal"])

    def test_balanced_smoke_subset_covers_all_fifteen_cells(self) -> None:
        rows = [
            ContinuousScoreRow(
                identifier=str(score), document_id=str(score), prompt_num="1", prompt="p", essay="e",
                labels=(float(score), float((score % 5) + 1), float(((score + 1) % 5) + 1)),
            )
            for score in range(1, 6)
        ]
        chosen = balanced_smoke_subset(rows)
        cells = {
            (axis, int(value))
            for row in chosen
            for axis, value in enumerate(row.labels)
        }
        self.assertEqual(cells, {(axis, score) for axis in range(3) for score in range(1, 6)})


if __name__ == "__main__":
    unittest.main()
