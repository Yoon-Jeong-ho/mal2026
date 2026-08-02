from __future__ import annotations

import unittest

import numpy as np

from mal2026.ordinal_tail_diagnostics import _prediction_diagnostics


class OrdinalTailDiagnosticsTest(unittest.TestCase):
    def test_prediction_diagnostics_preserve_three_axis_contract(self) -> None:
        gold = np.asarray([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [4., 5., 1.], [5., 1., 2.]])
        result = _prediction_diagnostics(gold, gold)
        self.assertEqual(set(result["axes"]), {"content", "organization", "expression"})
        for axis in result["axes"].values():
            self.assertEqual(sum(axis["prediction_band_histogram"].values()), 5)
            self.assertEqual(sum(map(sum, axis["confusion_rows_gold_columns_prediction_1_to_5"])), 5)


if __name__ == "__main__":
    unittest.main()
