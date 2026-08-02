from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from mal2026.kure_axis_contrastive import (
    AxisContrastiveConfig,
    KUREAxisContrastiveError,
    METHODS,
    ScoreBalancedBatchSampler,
    importance_by_score,
    inference_methods,
    interpolated_centroids,
    label_centroids,
    ordinal_contrastive_loss,
    ordinal_pair_target,
    score_axis,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kure_axis_contrastive.base.content.v1.json"


class KUREAxisContrastiveTests(unittest.TestCase):
    def test_all_six_configs_validate_without_dependencies(self) -> None:
        paths = sorted((ROOT / "configs").glob("kure_axis_contrastive.*.v1.json"))
        self.assertEqual(len(paths), 6)
        identities = set()
        for path in paths:
            value = AxisContrastiveConfig.from_json(path, require_dependencies=False)
            identities.add((value.arm, value.axis))
        self.assertEqual(len(identities), 6)

    def test_unknown_config_field_fails_closed(self) -> None:
        raw = json.loads(CONFIG.read_text())
        raw["extra"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(KUREAxisContrastiveError, "fields differ"):
                AxisContrastiveConfig.from_json(path, require_dependencies=False)

    def test_sampler_has_exact_quota_and_no_duplicates(self) -> None:
        labels = [1] * 4 + [2] * 20 + [3] * 30 + [4] * 30 + [5] * 20
        sampler = ScoreBalancedBatchSampler(labels, 17)
        first = list(sampler)
        second = list(ScoreBalancedBatchSampler(labels, 17))
        self.assertEqual(first, second)
        for batch in first:
            self.assertEqual(len(batch), len(set(batch)))
            self.assertEqual([sum(labels[index] == score for index in batch) for score in range(1, 6)], [2, 4, 5, 5, 4])

    def test_importance_recovers_original_class_mass(self) -> None:
        labels = [1] * 4 + [2] * 20 + [3] * 30 + [4] * 30 + [5] * 20
        weight = importance_by_score(labels)
        quota = [2, 4, 5, 5, 4]
        recovered = [quota[i] / 20 * weight[i + 1] for i in range(5)]
        recovered = [value / sum(recovered) for value in recovered]
        expected = [labels.count(score) / len(labels) for score in range(1, 6)]
        for left, right in zip(recovered, expected, strict=True):
            self.assertAlmostEqual(left, right)

    def test_pair_target_preserves_ordinal_geometry(self) -> None:
        target = ordinal_pair_target(torch.arange(1, 6, dtype=torch.float32))
        self.assertAlmostEqual(float(target[0, 0]), 1.0)
        self.assertGreater(float(target[0, 1]), float(target[0, 2]))
        self.assertGreater(float(target[0, 2]), float(target[0, 3]))
        self.assertAlmostEqual(float(target[0, 4]), -1.0, places=6)

    def test_well_ordered_embeddings_have_lower_pair_loss(self) -> None:
        config = AxisContrastiveConfig.from_json(CONFIG, require_dependencies=False)
        labels = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=torch.float32)
        angles = (labels - 1) * torch.pi / 4
        ordered = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
        generator = torch.Generator().manual_seed(9)
        random = torch.nn.functional.normalize(torch.randn(10, 2, generator=generator), dim=1)
        ordered_loss, _, _ = ordinal_contrastive_loss(ordered, labels, config)
        random_loss, _, _ = ordinal_contrastive_loss(random, labels, config)
        self.assertLess(float(ordered_loss), float(random_loss))

    def test_interpolated_center_inventory(self) -> None:
        centroids = torch.eye(5)
        half, half_values = interpolated_centroids(centroids, 0.5)
        tenth, tenth_values = interpolated_centroids(centroids, 0.1)
        self.assertEqual((len(half), len(half_values)), (9, 9))
        self.assertEqual((len(tenth), len(tenth_values)), (41, 41))
        self.assertTrue(torch.allclose(half.norm(dim=1), torch.ones(9)))

    def test_all_inference_methods_return_finite_scores(self) -> None:
        config = AxisContrastiveConfig.from_json(CONFIG, require_dependencies=False)
        generator = torch.Generator().manual_seed(3)
        labels = torch.tensor([score for score in range(1, 6) for _ in range(4)], dtype=torch.float32)
        embeddings = torch.nn.functional.normalize(torch.randn(20, 16, generator=generator), dim=1)
        eval_embeddings = torch.nn.functional.normalize(torch.randn(7, 16, generator=generator), dim=1)
        methods, diagnostic = inference_methods(embeddings, labels, eval_embeddings, torch.full((7,), 3.0), config)
        self.assertEqual(set(methods), set(METHODS))
        self.assertTrue(all(torch.isfinite(value).all() and ((1 <= value) & (value <= 5)).all() for value in methods.values()))
        self.assertEqual(diagnostic["prototype_support"], {str(score): 4 for score in range(1, 6)})

    def test_axis_metrics_include_tail_and_boundary(self) -> None:
        gold = torch.tensor([1, 2, 3, 3, 4, 4, 5], dtype=torch.float32)
        metrics = score_axis(gold, gold.clone())
        self.assertEqual(metrics["continuous_rmse"], 0.0)
        self.assertEqual(metrics["gold34_balanced_accuracy"], 1.0)
        self.assertEqual(metrics["low_1_2_rmse"], 0.0)
        self.assertEqual(metrics["score5_rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
