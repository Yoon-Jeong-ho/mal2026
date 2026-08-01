from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mal2026.iterative_tail_learner_models import LearnerResult, candidate_specs
from mal2026.iterative_tail_learner_protocol import load_protocol
from mal2026.iterative_tail_learner_runner import (
    _candidate_inner_oof,
    _fit_and_select_candidates,
    _fresh_outer_prediction,
    _learner_features,
    _outer_train_folds,
    gate_decision,
    run_outer_fold,
)


class Evidence:
    def __init__(self, rows: int):
        self.values = np.zeros((rows, 576), dtype=np.float32)

    def view(self, name: str):
        return self.values if name == "evidence_hash" else None


def data_fixture() -> SimpleNamespace:
    rows = 2000
    return SimpleNamespace(
        source_ids=tuple(f"id-{index}" for index in range(rows)),
        embeddings=np.zeros((rows, 8), dtype=np.float32),
        base=np.full((rows, 3), 3.0, dtype=np.float32),
        targets=np.full((rows, 3), 3.0, dtype=np.float32),
        folds=np.repeat(np.arange(5), 400),
        evidence=Evidence(rows),
    )


def metrics(*, rmse=.60, equal=.75, low=.95, high=.90, ba=.60, spearman=.60):
    return {
        "macro": {"rmse": rmse, "equal_group_rmse": equal, "low_tail_rmse": low,
                  "high_tail_rmse": high, "gold_3_4_balanced_accuracy": ba, "spearman": spearman},
        "axes": {axis: {"rmse": .60} for axis in ("content", "organization", "expression")},
    }


class IterativeTailLearnerRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_features_are_only_frozen_embedding_plus_score_blind_hash(self):
        data = data_fixture()
        features = _learner_features(data)
        self.assertEqual((2000, 584), features.shape)
        np.testing.assert_array_equal(features[:, :8], data.embeddings)
        np.testing.assert_array_equal(features[:, 8:], data.evidence.values)

    def test_each_candidate_is_fresh_fit_S_to_D_with_O_sealed(self):
        data = data_fixture()
        features = _learner_features(data)
        fit_rows, predict_rows, fitted_ids = [], [], []

        def fake_fit(spec, train_features, _base, _targets):
            fitted = SimpleNamespace(spec=spec, audit={"fit_id": len(fit_rows)})
            fit_rows.append(set(np.flatnonzero(np.isin(features[:, 0], train_features[:, 0]))))
            fitted_ids.append(id(fitted))
            return fitted

        # Use row-index feature to prove exact membership passed to mocked fit/apply.
        features[:, 0] = np.arange(2000)

        def fake_apply(fitted, predict_features, _base):
            indices = predict_features[:, 0].astype(int)
            predict_rows.append(set(indices.tolist()))
            return LearnerResult(np.full((len(indices), 3), 3.0, dtype=np.float32), {"fit_id": fitted.audit["fit_id"]})

        with (
            patch("mal2026.iterative_tail_learner_runner.fit", side_effect=fake_fit),
            patch("mal2026.iterative_tail_learner_runner.apply", side_effect=fake_apply),
        ):
            oof = _candidate_inner_oof(data, features, 0, candidate_specs()[0])
        self.assertEqual((1600, 3), oof.predictions.shape)
        self.assertEqual(4, len(fitted_ids))
        self.assertEqual(4, len(set(fitted_ids)))
        for inner, (train, predict) in zip((1, 2, 3, 4), zip(fit_rows, predict_rows, strict=True), strict=True):
            self.assertEqual(1200, len(train)); self.assertEqual(400, len(predict))
            self.assertFalse(train & predict)
            self.assertTrue(all(data.folds[index] not in (0, inner) for index in train))
            self.assertTrue(all(data.folds[index] == inner for index in predict))

    def test_all_twenty_complete_before_selection_and_none_falls_back(self):
        data = data_fixture(); features = _learner_features(data); calls = []

        def fake_oof(_data, _features, _outer, spec):
            calls.append(spec.cycle)
            return SimpleNamespace(predictions=np.full((1600, 3), spec.cycle, dtype=np.float32), audit=())

        def fake_metrics(_gold, prediction):
            return {"macro": {"rmse": .60 if np.allclose(prediction, 3) else .50 + float(prediction[0, 0]) / 1000}}

        with (
            patch("mal2026.iterative_tail_learner_runner._candidate_inner_oof", side_effect=fake_oof),
            patch("mal2026.iterative_tail_learner_runner.compute_iterative_tail_metrics", side_effect=fake_metrics),
            patch("mal2026.iterative_tail_learner_runner.gate_decision", return_value={"eligible": False}),
        ):
            selected, records = _fit_and_select_candidates(self.protocol, data, features, 0, device="cpu")
        self.assertEqual(list(range(1, 21)), calls)
        self.assertEqual(20, len(records))
        self.assertIsNone(selected)

    def test_selected_spec_is_frozen_and_only_one_fresh_outer_refit_occurs(self):
        data = data_fixture(); features = _learner_features(data); spec = candidate_specs()[6]
        fits, applies = [], []

        def fake_fit(given, x, _base, _targets):
            fits.append((given, x[:, 0].copy()))
            return SimpleNamespace(spec=given, audit={})

        features[:, 0] = np.arange(2000)

        def fake_apply(fitted, x, _base):
            applies.append((fitted.spec, x[:, 0].copy()))
            return LearnerResult(np.full((len(x), 3), 3.0, dtype=np.float32), {})

        with patch("mal2026.iterative_tail_learner_runner.fit", side_effect=fake_fit), patch(
            "mal2026.iterative_tail_learner_runner.apply", side_effect=fake_apply
        ):
            prediction, audit = _fresh_outer_prediction(data, features, 2, spec)
        self.assertEqual((400, 3), prediction.shape)
        self.assertEqual(1, len(fits)); self.assertEqual(1, len(applies))
        self.assertIs(fits[0][0], spec); self.assertIs(applies[0][0], spec)
        self.assertTrue(all(data.folds[int(index)] != 2 for index in fits[0][1]))
        self.assertTrue(all(data.folds[int(index)] == 2 for index in applies[0][1]))
        self.assertTrue(audit["selection_frozen_before_refit"])

    def test_public_result_has_no_row_ids_while_restricted_rows_do(self):
        data = data_fixture(); spec = candidate_specs()[0]
        with tempfile.TemporaryDirectory() as temp:
            public, restricted = Path(temp) / "public", Path(temp) / "restricted"
            with (
                patch("mal2026.iterative_tail_learner_runner.PUBLIC_ROOT", public),
                patch("mal2026.iterative_tail_learner_runner.RESTRICTED_ROOT", restricted),
                patch("mal2026.iterative_tail_learner_runner.validate_bound_inputs"),
                patch("mal2026.iterative_tail_learner_runner.validate_model_inventory"),
                patch("mal2026.iterative_tail_learner_runner.load_experiment_data", return_value=data),
                patch("mal2026.iterative_tail_learner_runner._fit_and_select_candidates", return_value=(spec, [{"cycle": 1}])),
                patch("mal2026.iterative_tail_learner_runner._fresh_outer_prediction", return_value=(np.full((400, 3), 3.0), {"selection_frozen_before_refit": True})),
                patch("mal2026.iterative_tail_learner_runner.compute_iterative_tail_metrics", return_value=metrics()),
            ):
                run_outer_fold(0, device="cpu", protocol=self.protocol)
            public_text = (public / "outer-0/result.json").read_text()
            restricted_text = (restricted / "outer-0/predictions.jsonl").read_text()
            self.assertNotIn("source_id", public_text)
            self.assertNotIn("id-0", public_text)
            self.assertIn('"source_id": "id-0"', restricted_text)
            self.assertEqual(400, len(restricted_text.splitlines()))

    def test_final_gate_uses_candidate_minus_baseline_ci_upper(self):
        baseline = metrics()
        candidate = metrics(rmse=.589, equal=.74, low=.94, high=.89, ba=.611, spearman=.596)
        passed = gate_decision(self.protocol.raw["final_evaluation"], baseline, candidate, final=True,
                               bootstrap={"candidate_minus_baseline_ci": {"upper": -.0001}})
        failed = gate_decision(self.protocol.raw["final_evaluation"], baseline, candidate, final=True,
                               bootstrap={"candidate_minus_baseline_ci": {"upper": .0001}})
        self.assertTrue(passed["pass"]); self.assertFalse(failed["pass"])


if __name__ == "__main__":
    unittest.main()
