from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mal2026.iterative_tail_router_models import RouterResult, router_specs
from mal2026.iterative_tail_router_protocol import load_protocol
from mal2026.iterative_tail_router_runner import (
    ComponentBank,
    _fit_and_select_routes,
    _outer_train_folds,
    gate_decision,
)


def metrics(
    *, rmse: float, equal: float, low: float, high: float, ba: float, spearman: float,
    axes: tuple[float, float, float] = (.60, .60, .60),
) -> dict:
    return {
        "macro": {
            "rmse": rmse,
            "equal_group_rmse": equal,
            "low_tail_rmse": low,
            "high_tail_rmse": high,
            "gold_3_4_balanced_accuracy": ba,
            "spearman": spearman,
        },
        "axes": {
            axis: {"rmse": value}
            for axis, value in zip(("content", "organization", "expression"), axes, strict=True)
        },
    }


class IterativeTailRouterRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol()

    def test_outer_inner_fold_partition_excludes_outer(self) -> None:
        self.assertEqual((1, 2, 3, 4), _outer_train_folds(0))
        self.assertEqual((0, 1, 2, 3), _outer_train_folds(4))
        for outer in range(5):
            folds = _outer_train_folds(outer)
            self.assertEqual(4, len(folds))
            self.assertNotIn(outer, folds)
            for inner in folds:
                training = tuple(fold for fold in folds if fold != inner)
                self.assertEqual(3, len(training))
                self.assertNotIn(outer, training)
                self.assertNotIn(inner, training)

    def test_inner_gate_is_strict_conjunction_and_score1_is_descriptive(self) -> None:
        baseline = metrics(rmse=.60, equal=.75, low=.95, high=.90, ba=.60, spearman=.60)
        candidate = metrics(
            rmse=.594, equal=.739, low=.94, high=.89, ba=.611, spearman=.596,
            axes=(.605, .595, .596),
        )
        decision = gate_decision(self.protocol.raw["inner_promotion_gate"], baseline, candidate)
        self.assertTrue(decision["eligible"])
        self.assertTrue(all(decision["gates"].values()))
        self.assertFalse(decision["score1_used_for_promotion"])
        too_small_macro = metrics(rmse=.596, equal=.739, low=.94, high=.89, ba=.611, spearman=.596)
        rejected = gate_decision(self.protocol.raw["inner_promotion_gate"], baseline, too_small_macro)
        self.assertFalse(rejected["eligible"])
        self.assertFalse(rejected["gates"]["macro_rmse_improvement"])

    def test_all_twenty_finish_before_baseline_relative_selection(self) -> None:
        specs = router_specs()
        rows = 8
        data = SimpleNamespace(targets=np.full((rows, 3), 3.0), base=np.full((rows, 3), 3.0))
        bank = ComponentBank(*(np.full((rows, 3), 3.0) for _ in range(4)), audit=())
        calls: list[int] = []

        def fake_fit(spec, *args):
            calls.append(spec.cycle)
            prediction = np.full((rows, 3), 1.0 + spec.cycle / 100.0, dtype=np.float32)
            return RouterResult(prediction, {"cycle": spec.cycle}, {"cycle": spec.cycle})

        def fake_metrics(_gold, prediction):
            # Baseline is 3.0; route macro values are unique and route 7 is best.
            if np.allclose(prediction, 3.0):
                value = .60
            else:
                cycle = int(round((float(prediction[0, 0]) - 1.0) * 100))
                value = .55 + abs(cycle - 7) / 1000.0
            return {"macro": {"rmse": value}}

        def fake_gate(_config, _baseline, candidate, **_kwargs):
            return {"eligible": candidate["macro"]["rmse"] < .57}

        with (
            patch("mal2026.iterative_tail_router_runner.fit_route", side_effect=fake_fit),
            patch("mal2026.iterative_tail_router_runner.compute_iterative_tail_metrics", side_effect=fake_metrics),
            patch("mal2026.iterative_tail_router_runner.gate_decision", side_effect=fake_gate),
        ):
            selected, parameters, records = _fit_and_select_routes(
                self.protocol, data, np.arange(rows), bank,
            )
        self.assertEqual(list(range(1, 21)), calls)
        self.assertEqual(20, len(records))
        self.assertIsNotNone(selected)
        self.assertEqual(specs[6].variant_id, selected.variant_id)
        self.assertEqual({"cycle": 7}, parameters)

    def test_no_eligible_route_fails_closed_to_baseline(self) -> None:
        rows = 4
        data = SimpleNamespace(targets=np.full((rows, 3), 3.0), base=np.full((rows, 3), 3.0))
        bank = ComponentBank(*(np.full((rows, 3), 3.0) for _ in range(4)), audit=())

        def fake_fit(spec, *args):
            return RouterResult(data.base.astype(np.float32), {"cycle": spec.cycle}, {"cycle": spec.cycle})

        with (
            patch("mal2026.iterative_tail_router_runner.fit_route", side_effect=fake_fit),
            patch("mal2026.iterative_tail_router_runner.gate_decision", return_value={"eligible": False}),
        ):
            selected, parameters, records = _fit_and_select_routes(
                self.protocol, data, np.arange(rows), bank,
            )
        self.assertIsNone(selected)
        self.assertIsNone(parameters)
        self.assertEqual(20, len(records))

    def test_final_gate_uses_candidate_minus_baseline_ci_direction(self) -> None:
        baseline = metrics(rmse=.60, equal=.75, low=.95, high=.90, ba=.60, spearman=.60)
        candidate = metrics(
            rmse=.589, equal=.74, low=.94, high=.89, ba=.611, spearman=.596,
            axes=(.605, .590, .590),
        )
        bootstrap = {"candidate_minus_baseline_ci": {"upper": -.0001}}
        passed = gate_decision(
            self.protocol.raw["final_evaluation"], baseline, candidate,
            final=True, bootstrap=bootstrap,
        )
        self.assertTrue(passed["pass"])
        crossing = {"candidate_minus_baseline_ci": {"upper": .0001}}
        failed = gate_decision(
            self.protocol.raw["final_evaluation"], baseline, candidate,
            final=True, bootstrap=crossing,
        )
        self.assertFalse(failed["pass"])
        self.assertFalse(failed["gates"]["candidate_minus_baseline_rmse_ci_upper_below_bound"])


if __name__ == "__main__":
    unittest.main()
