from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

from mal2026.solar_consensus_pilot import (
    calibrated_quarter_threshold,
    modal_label,
    requires_two_more_draws,
    stratified_fold_assignments,
    stratified_sources,
    visible_draw_seed,
)
from mal2026.solar_target_augmentation import SourceRow


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_solar_consensus_pilot.py"
SPEC = importlib.util.spec_from_file_location("solar_consensus_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
RUNNER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER_MODULE)


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SELECTION_MODULE = load_script("solar_consensus_selection", "build_solar_consensus_pool.py")
QWEN36_MODULE = load_script(
    "solar_consensus_qwen36_control", "run_solar_consensus_qwen36_control.py"
)


def row(index: int, score: tuple[float, float, float]) -> SourceRow:
    return SourceRow(
        identifier=f"source-{index}", document_id=f"document-{index}", prompt="논제",
        essay=("문장을 작성한다. " * (index + 2)).strip(), score=score,
    )


class SolarConsensusPilotTests(unittest.TestCase):
    def test_stratified_source_selection_is_order_independent(self) -> None:
        rows = [row(index, (1 + index % 5, 1 + (index * 2) % 5, 1 + (index * 3) % 5))
                for index in range(30)]
        forward = stratified_sources(rows, 20, 20260730)
        backward = stratified_sources(list(reversed(rows)), 20, 20260730)
        self.assertEqual([item.identifier for item in forward],
                         [item.identifier for item in backward])
        self.assertEqual(len({item.identifier for item in forward}), 20)

    def test_judge_seed_is_draw_specific_and_target_blind(self) -> None:
        messages = [{"role": "system", "content": "rubric"},
                    {"role": "user", "content": "prompt and essay"}]
        seeds = [visible_draw_seed(messages, index, 20260730) for index in range(5)]
        self.assertEqual(len(set(seeds)), 5)
        self.assertEqual(seeds[0], visible_draw_seed(messages, 0, 20260730))

    def test_adaptive_three_then_five_joint_triplet(self) -> None:
        same = [{"content": 2, "organization": 3, "expression": 4}] * 3
        self.assertFalse(requires_two_more_draws(same))
        label, support, distribution = modal_label(same)
        self.assertEqual(label, same[0])
        self.assertEqual(support, 3)
        self.assertEqual(distribution, {"2/3/4": 3})

        split = [
            {"content": 2, "organization": 3, "expression": 4},
            {"content": 2, "organization": 3, "expression": 4},
            {"content": 3, "organization": 3, "expression": 4},
        ]
        self.assertTrue(requires_two_more_draws(split))
        unstable, support, _ = modal_label(split + [
            {"content": 3, "organization": 3, "expression": 4},
            {"content": 4, "organization": 3, "expression": 4},
        ])
        self.assertIsNone(unstable)
        self.assertEqual(support, 2)

    def test_runner_mode_populations_and_no_full_mode(self) -> None:
        rows = [row(index, (1 + index % 5, 1 + (index * 2) % 5, 1 + (index * 3) % 5))
                for index in range(30)]
        smoke_sources, smoke_tasks, smoke_families = RUNNER_MODULE.mode_matrix("smoke", rows)
        self.assertEqual((len(smoke_sources), len(smoke_tasks), smoke_families), (5, 15, 2))
        pilot_sources, pilot_tasks, pilot_families = RUNNER_MODULE.mode_matrix("pilot", rows)
        self.assertEqual((len(pilot_sources), len(pilot_tasks), pilot_families), (20, 300, 8))
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn('choices=("smoke", "pilot", "full")', source)
        self.assertIn('"requested_target_is_label": False', source)
        self.assertIn('"official_evaluation_txt_prompt_exact": True', source)

    def test_stratified_fold_assignments_are_balanced_and_order_independent(self) -> None:
        rows = [row(index, (1 + index % 5, 1 + (index * 2) % 5, 1 + (index * 3) % 5))
                for index in range(31)]
        # The production ContinuousScoreRow uses `labels`; SourceRow's `score`
        # exercises the helper's equivalent sampling-only path.
        forward = stratified_fold_assignments(rows, 5, 20260730)
        backward = stratified_fold_assignments(list(reversed(rows)), 5, 20260730)
        self.assertEqual(forward, backward)
        counts = [sum(value == fold for value in forward.values()) for fold in range(5)]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_calibration_threshold_uses_train_oof_upper_quantile(self) -> None:
        self.assertEqual(
            calibrated_quarter_threshold([0.1, 0.2, 0.3, 0.6, 0.9], 0.8),
            0.75,
        )
        self.assertEqual(calibrated_quarter_threshold([0.1] * 10), 0.5)

    def test_selection_oof_metric_uses_three_nested_axis_violation_cells(self) -> None:
        prediction = {
            "reference_score": {axis: 3 for axis in SELECTION_MODULE.AXES},
            "continuous_prediction": {axis: 3.0 for axis in SELECTION_MODULE.AXES},
            "integer_prediction": {axis: 3 for axis in SELECTION_MODULE.AXES},
        }
        metrics = SELECTION_MODULE.original_oof_metrics({"record": prediction})
        self.assertEqual(metrics["macro_continuous_rmse"], 0.0)

    def test_qwen36_control_parser_and_agreement_metric(self) -> None:
        value = {
            axis: {"score": index + 1, "rationale": "판단 근거"}
            for index, axis in enumerate(QWEN36_MODULE.AXES)
        }
        import json
        self.assertEqual(
            QWEN36_MODULE.parse_output(json.dumps(value, ensure_ascii=False)), value
        )
        row = {
            "solar_modal_score": {axis: 3 for axis in QWEN36_MODULE.AXES},
            "qwen36_score": {axis: 3 for axis in QWEN36_MODULE.AXES},
        }
        self.assertEqual(
            QWEN36_MODULE.agreement_summary([row])["macro_continuous_rmse"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
