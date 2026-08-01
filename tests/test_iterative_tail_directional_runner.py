from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mal2026.iterative_tail_directional_models import DirectionalResult, candidate_specs
from mal2026.iterative_tail_directional_protocol import load_protocol
from mal2026.iterative_tail_directional_runner import (
    _candidate_inner_oof,
    _directional_features,
    _fit_and_select_candidates,
    _fresh_outer_prediction,
    projection_audit,
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
        embeddings=np.zeros((rows, 4096), dtype=np.float32),
        base=np.full((rows, 3), 3.0, dtype=np.float32),
        targets=np.full((rows, 3), 3.0, dtype=np.float32),
        folds=np.repeat(np.arange(5), 400),
        evidence=Evidence(rows),
    )


def metrics(*, rmse=.60, equal=.75, low=.95, high=.90, ba=.60, spearman=.60, recall=.0):
    return {
        "macro": {"rmse": rmse, "equal_group_rmse": equal, "low_tail_rmse": low,
                  "high_tail_rmse": high, "gold_3_4_balanced_accuracy": ba, "spearman": spearman},
        "axes": {
            axis: {"rmse": .60, "bands": {"5": {"recall": recall}}}
            for axis in ("content", "organization", "expression")
        },
    }


class IterativeTailDirectionalRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_projection_is_exact_deterministic_gold_free_4672_to_64(self):
        data = data_fixture()
        first = _directional_features(data, self.protocol)
        second = _directional_features(data, self.protocol)
        self.assertEqual((2000, 64), first.shape)
        np.testing.assert_array_equal(first, second)
        audit = projection_audit(self.protocol)
        self.assertEqual("1ef94e113b8c32e77017f9afbc020fdfd886426fc199ef4d6fc44f8e2f8f493b", audit["matrix_sha256"])
        self.assertFalse(audit["fit_to_data"] or audit["gold_used"])

    def test_candidate_fresh_fit_S_to_D_and_internal_fold_ids_exclude_D_O(self):
        data = data_fixture()
        features = np.zeros((2000, 64), dtype=np.float32)
        features[:, 0] = np.arange(2000)
        fits, predicts, objects = [], [], []

        def fake_fit(spec, train_features, _base, _targets, fold_ids):
            fitted = SimpleNamespace(spec=spec, audit={"folds": sorted(set(fold_ids.tolist()))})
            fits.append((set(train_features[:, 0].astype(int).tolist()), set(fold_ids.tolist())))
            objects.append(id(fitted))
            return fitted

        def fake_apply(fitted, predict_features, _base):
            indices = predict_features[:, 0].astype(int)
            predicts.append(set(indices.tolist()))
            return DirectionalResult(np.full((len(indices), 3), 3.0, dtype=np.float32), {"fit": id(fitted)})

        with (
            patch("mal2026.iterative_tail_directional_runner.fit", side_effect=fake_fit),
            patch("mal2026.iterative_tail_directional_runner.apply", side_effect=fake_apply),
        ):
            oof = _candidate_inner_oof(data, features, 0, candidate_specs()[0])
        self.assertEqual((1600, 3), oof.predictions.shape)
        self.assertEqual(4, len(set(objects)))
        for inner, ((train_rows, fold_ids), predict_rows) in enumerate(zip(fits, predicts, strict=True), start=1):
            self.assertEqual(set(range(1, 5)) - {inner}, fold_ids)
            self.assertEqual(1200, len(train_rows)); self.assertEqual(400, len(predict_rows))
            self.assertFalse(train_rows & predict_rows)
            self.assertTrue(all(data.folds[index] not in (0, inner) for index in train_rows))
            self.assertTrue(all(data.folds[index] == inner for index in predict_rows))

    def test_all_three_finish_before_selection_and_none_falls_back(self):
        data = data_fixture(); features = np.zeros((2000, 64), dtype=np.float32); calls = []

        def fake_oof(_data, _features, _outer, spec):
            calls.append(spec.cycle)
            return SimpleNamespace(predictions=np.full((1600, 3), 3.0, dtype=np.float32), audit=())

        with (
            patch("mal2026.iterative_tail_directional_runner._candidate_inner_oof", side_effect=fake_oof),
            patch("mal2026.iterative_tail_directional_runner.compute_iterative_tail_metrics", return_value=metrics()),
        ):
            selected, records = _fit_and_select_candidates(self.protocol, data, features, 0, device="cpu")
        self.assertEqual([1, 2, 3], calls)
        self.assertEqual(3, len(records))
        self.assertTrue(all("baseline_relative_decision" in record for record in records))
        self.assertIsNone(selected)

    def test_selected_spec_is_frozen_for_one_outer_refit_with_four_internal_folds(self):
        data = data_fixture(); features = np.zeros((2000, 64), dtype=np.float32); features[:, 0] = np.arange(2000)
        spec = candidate_specs()[0]; fits, applies = [], []

        def fake_fit(given, x, _base, _targets, folds):
            fits.append((given, x[:, 0].copy(), set(folds.tolist())))
            return SimpleNamespace(spec=given, audit={})

        def fake_apply(fitted, x, _base):
            applies.append((fitted.spec, x[:, 0].copy()))
            return DirectionalResult(np.full((len(x), 3), 3.0, dtype=np.float32), {})

        with (
            patch("mal2026.iterative_tail_directional_runner.fit", side_effect=fake_fit),
            patch("mal2026.iterative_tail_directional_runner.apply", side_effect=fake_apply),
        ):
            prediction, audit = _fresh_outer_prediction(data, features, 2, spec)
        self.assertEqual((400, 3), prediction.shape)
        self.assertEqual(1, len(fits)); self.assertEqual(1, len(applies))
        self.assertIs(fits[0][0], spec); self.assertEqual({0, 1, 3, 4}, fits[0][2])
        self.assertTrue(all(data.folds[int(index)] != 2 for index in fits[0][1]))
        self.assertTrue(all(data.folds[int(index)] == 2 for index in applies[0][1]))
        self.assertTrue(audit["selection_frozen_before_refit"])

    def test_public_result_has_no_row_ids_while_restricted_has_exact_400(self):
        data = data_fixture(); records = [{"variant_id": spec.variant_id} for spec in candidate_specs()]
        with tempfile.TemporaryDirectory() as temp:
            public, restricted = Path(temp) / "public", Path(temp) / "restricted"
            with (
                patch("mal2026.iterative_tail_directional_runner.PUBLIC_ROOT", public),
                patch("mal2026.iterative_tail_directional_runner.RESTRICTED_ROOT", restricted),
                patch("mal2026.iterative_tail_directional_runner.validate_bound_inputs"),
                patch("mal2026.iterative_tail_directional_runner.validate_model_inventory"),
                patch("mal2026.iterative_tail_directional_runner.load_experiment_data", return_value=data),
                patch("mal2026.iterative_tail_directional_runner._directional_features", return_value=np.zeros((2000, 64), dtype=np.float32)),
                patch("mal2026.iterative_tail_directional_runner._fit_and_select_candidates", return_value=(None, records)),
                patch("mal2026.iterative_tail_directional_runner.compute_iterative_tail_metrics", return_value=metrics()),
            ):
                run_outer_fold(0, device="cpu", protocol=self.protocol)
            public_text = (public / "outer-0/result.json").read_text()
            restricted_text = (restricted / "outer-0/predictions.jsonl").read_text()
            self.assertNotIn("source_id", public_text); self.assertNotIn("id-0", public_text)
            self.assertIn('"source_id": "id-0"', restricted_text)
            self.assertEqual(400, len(restricted_text.splitlines()))

    def test_protocol_gate_key_integrates_with_actual_selection_helper(self):
        from mal2026.iterative_tail_directional_selection import inner_gate
        baseline = metrics()
        candidate = metrics(rmse=.594, equal=.739, low=.94, high=.89, ba=.611, spearman=.596, recall=.011)
        decision = inner_gate(self.protocol.raw["inner_promotion_gate"], baseline, candidate)
        self.assertTrue(decision["config_valid"])
        self.assertTrue(decision["eligible"])
        self.assertEqual(8, len(decision["gates"]))


if __name__ == "__main__":
    unittest.main()
