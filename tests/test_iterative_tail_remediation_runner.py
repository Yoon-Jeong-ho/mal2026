from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mal2026.iterative_tail_remediation_protocol import load_protocol
from mal2026.iterative_tail_remediation_runner import (
    CandidateOOF,
    _public_subvariant,
    _remediation_subvariants,
    _select_candidate,
    regenerate_inner_selection_teacher,
)
from mal2026.iterative_tail_runner import ExperimentData


class _DummyEvidence:
    def __init__(self, rows: int):
        self.rows = rows

    def view(self, name: str):
        if name in {"consensus_disagreement", "evidence_hash"}:
            return np.zeros((self.rows, 2), dtype=np.float32)
        return None


def _metric(rmse: float, equal: float, low: float, high: float, ba: float, spearman: float):
    axes = {
        axis: {"rmse": rmse}
        for axis in ("content", "organization", "expression")
    }
    return {
        "axes": axes,
        "macro": {
            "rmse": rmse,
            "equal_group_rmse": equal,
            "low_tail_rmse": low,
            "high_tail_rmse": high,
            "gold_3_4_balanced_accuracy": ba,
            "spearman": spearman,
        },
    }


class IterativeTailRemediationRunnerTests(unittest.TestCase):
    def test_inner_teacher_excludes_outer_and_inner_validation_gold(self):
        folds = np.repeat(np.arange(5), 2)
        rows = len(folds)
        data = ExperimentData(
            source_ids=tuple(f"row-{index}" for index in range(rows)),
            document_ids=tuple(f"doc-{index}" for index in range(rows)),
            prompt_nums=tuple("1" for _ in range(rows)),
            embeddings=np.zeros((rows, 4), dtype=np.float32),
            base=np.full((rows, 3), 3.0, dtype=np.float32),
            targets=np.column_stack((folds + 1.0, folds + 1.0, folds + 1.0)).astype(np.float32),
            folds=folds,
            evidence=_DummyEvidence(rows),
        )
        calls = []

        def fake_fit(spec, train_embeddings, train_base, train_targets, predict_embeddings, predict_base, **kwargs):
            del spec, train_embeddings, train_base, predict_embeddings, kwargs
            train_fold_values = set((train_targets[:, 0] - 1).astype(int).tolist())
            predict_count = len(predict_base)
            calls.append((train_fold_values, predict_count))
            value = float(np.mean(train_targets))
            return SimpleNamespace(
                predictions=np.full((predict_count, 3), value, dtype=np.float32),
                initial_state_hashes=("i", "i", "i"),
                final_state_hashes=("f", "f", "f"),
            )

        with patch("mal2026.iterative_tail_remediation_runner.fit_frozen_candidate", side_effect=fake_fit):
            teacher, audit = regenerate_inner_selection_teacher(
                data, outer_fold=4, inner_validation_fold=3, device="cpu",
            )
        self.assertEqual(3, len(calls))
        self.assertEqual(3, len(audit))
        for train_folds, predict_count in calls:
            self.assertTrue(train_folds <= {0, 1, 2})
            self.assertNotIn(3, train_folds)
            self.assertNotIn(4, train_folds)
            self.assertEqual(2, len(train_folds))
            self.assertEqual(2, predict_count)
        self.assertTrue(np.isfinite(teacher[np.isin(folds, (0, 1, 2))]).all())
        self.assertTrue(np.isnan(teacher[np.isin(folds, (3, 4))]).all())

    def test_selection_is_global_baseline_relative_not_sequential(self):
        protocol = load_protocol()
        predictions = np.zeros((4, 3), dtype=np.float64)
        baseline = CandidateOOF("base-identity", "identity", predictions, _metric(.80, .90, 1.0, 1.0, .50, .50), {})
        first = CandidateOOF("first", "family-a", predictions, _metric(.78, .88, .99, .99, .52, .50), {})
        # This is eligible against baseline and has the best RMSE, but its BA
        # is below `first`; a sequential tournament would reject it.
        best = CandidateOOF("best", "family-b", predictions, _metric(.76, .87, .98, .98, .515, .50), {})
        selected, decisions = _select_candidate(protocol, baseline, (first, best), None)
        self.assertEqual("best", selected.key)
        self.assertTrue(all(record["comparison_reference"] == "base-identity" for record in decisions))
        self.assertTrue(all(record["promote"] for record in decisions))
        self.assertEqual([False, True], [record["selected"] for record in decisions])

    def test_public_metadata_scrubs_row_derived_knots(self):
        raw = {
            "spec": "weighted-isotonic-unweighted",
            "split_parameters": [{"selected_parameters": [{"x_knots": [1.1, 2.2], "y_knots": [1.2, 2.3]}]}],
        }
        public = _public_subvariant(raw)
        encoded = str(public)
        self.assertNotIn("x_knots", encoded)
        self.assertNotIn("y_knots", encoded)
        self.assertEqual(1, public["split_parameter_fits"])
        self.assertEqual(64, len(public["split_parameters_digest"]))

    def test_standalone_convex_blend_is_not_a_registered_candidate(self):
        names = [name for name, _ in _remediation_subvariants()]
        self.assertNotIn("convex-blend", names)


if __name__ == "__main__":
    unittest.main()
