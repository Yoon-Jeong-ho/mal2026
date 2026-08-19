from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mal2026.qwen_rationale_oof import (
    AXES, Row, SourceBalancedDataset, emit_integers, fit_cutpoints,
    metric_bundle, mild_tail_weights, selection_key,
)
from mal2026.rationale_pipeline_prompts import rationale_to_score_text, routing


class FakeConfig(dict):
    def __getitem__(self, key):
        return super().__getitem__(key)


class QwenRationaleOOFTests(unittest.TestCase):
    def test_current_rationale_prompt_routing_and_rendering_are_bound(self) -> None:
        self.assertEqual(routing()["rationale_to_score_encoder"]["source_file"], "rationale_to_score.txt")
        text = rationale_to_score_text(
            "논제", "학생 글", {axis: f"{axis} 근거" for axis in AXES}
        )
        self.assertIn('"evaluation_rationales"', text)
        self.assertNotIn("ROUND_HALF_UP", text)

    def test_mild_tail_weights_are_capped_and_axis_mean_one(self) -> None:
        labels = []
        for score, count in ((1, 2), (2, 5), (3, 30), (4, 50), (5, 13)):
            labels.extend([[float(score), float(score), float(score)]] * count)
        weights, audit = mild_tail_weights(labels, 2.5)
        self.assertEqual(audit["mode"], "inverse_sqrt_frequency_capped_and_axis_mean_normalized")
        for axis in range(3):
            self.assertAlmostEqual(sum(row[axis] for row in weights) / len(weights), 1.0, places=8)
            self.assertLessEqual(max(row[axis] for row in weights), 2.5 + 1e-8)

    def test_emit_cutpoints_is_monotonic_and_half_up_default(self) -> None:
        values = [[1.49, 2.5, 4.5], [1.5, 2.49, 4.49]]
        self.assertEqual(emit_integers(values), [[1, 3, 5], [2, 2, 4]])
        cutpoints = {axis: [1.4, 2.4, 3.4, 4.4] for axis in AXES}
        self.assertEqual(emit_integers(values, cutpoints), [[2, 3, 5], [2, 3, 5]])

    def test_fit_cutpoints_can_correct_shifted_predictions(self) -> None:
        values = [1.9] * 20 + [2.9] * 20 + [3.9] * 20 + [4.9] * 20 + [5.0] * 20
        gold = [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20 + [5] * 20
        thresholds = fit_cutpoints(values, gold)
        emitted = [1 + sum(value >= threshold for threshold in thresholds) for value in values]
        self.assertEqual(emitted, gold)
        self.assertTrue(all(thresholds[i] <= thresholds[i + 1] for i in range(3)))

    def test_metric_and_selection_use_rmse_first(self) -> None:
        labels = [[1.0, 2.0, 5.0], [2.0, 5.0, 1.0]]
        perfect = metric_bundle(labels, labels)
        worse = metric_bundle(labels, [[2.0, 2.0, 5.0], [2.0, 5.0, 1.0]])
        self.assertLess(selection_key("perfect", perfect), selection_key("worse", worse))

    def test_source_balanced_dataset_length_is_unique_sources(self) -> None:
        # Rendering is not invoked in this invariant-only check.
        rows = [Row("a", "p", "e", (1.0, 2.0, 3.0)), Row("b", "p", "e", (3.0, 4.0, 5.0))]
        pools = {identifier: [{axis: "근거" for axis in AXES}, {axis: "다른 근거" for axis in AXES}] for identifier in ("a", "b")}
        config = FakeConfig(seed=7, tail_weight_cap=2.5, rationale_dropout_probability=0.5)
        dataset = SourceBalancedDataset(rows, pools, config, "rationale", loss_weighting="natural", training=True)
        self.assertEqual(len(dataset), 2)
        dataset.set_epoch(3)
        self.assertEqual(dataset.epoch, 3)


if __name__ == "__main__":
    unittest.main()
