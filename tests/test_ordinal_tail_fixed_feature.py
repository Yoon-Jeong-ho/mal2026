from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from mal2026.ordinal_tail_fixed_feature import (
    CandidateSpec, EXPECTED_IDS, FixedFeatureConfig, OrdinalTailFixedFeatureError,
    _features, _validate_public_payload, build_axis_model, coral_pmf, corn_pmf,
    corn_targets, effective_number_weights, natural_prior, natural_prior_correction,
    nested_indices, rps_loss, sampling_prior, slace_components, slace_loss,
)
from mal2026.r0_ordinal_residual import ResidualRow


class FixedFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = FixedFeatureConfig.from_json("configs/ordinal_tail_program.v1.json")

    def test_candidate_inventory_is_exact_and_average_is_forbidden(self) -> None:
        self.assertEqual(tuple(x.identifier for x in self.config.candidates), EXPECTED_IDS)
        raw = json.loads(Path("configs/ordinal_tail_program.v1.json").read_text())
        raw["axes"] = ["content", "organization", "expression", "average"]
        with self.assertRaisesRegex(OrdinalTailFixedFeatureError, "average contract"):
            FixedFeatureConfig.from_mapping(raw)

    def test_rps(self) -> None:
        perfect = torch.eye(5)
        labels = torch.arange(1, 6)
        self.assertEqual(float(rps_loss(perfect, labels)), 0.0)
        self.assertGreater(float(rps_loss(perfect.flip(1), labels)), 0.0)

    def test_coral_monotonicity_and_pmf(self) -> None:
        logits = torch.tensor([[3.0, 2.0, 1.0, -1.0]])
        pmf = coral_pmf(logits)
        self.assertTrue(bool((pmf >= 0).all()))
        self.assertTrue(torch.allclose(pmf.sum(1), torch.ones(1)))
        model = build_axis_model(7, CandidateSpec("x", "coral", "natural"))
        output = model(torch.randn(3, 7))
        self.assertTrue(bool((output["logits"][:, 1:] <= output["logits"][:, :-1]).all()))

    def test_corn_mask_and_inference(self) -> None:
        target, mask = corn_targets(torch.tensor([1, 3, 5]))
        self.assertEqual(mask.sum(1).tolist(), [1, 3, 4])
        self.assertEqual(target[1].tolist(), [1.0, 1.0, 0.0, 0.0])
        pmf = corn_pmf(torch.zeros(2, 4))
        self.assertTrue(bool((pmf >= 0).all()))
        self.assertTrue(torch.allclose(pmf.sum(1), torch.ones(2)))

    def test_fold_fit_priors_and_corrections(self) -> None:
        labels = torch.tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5])
        prior = natural_prior(labels)
        self.assertTrue(torch.allclose(prior, torch.tensor([1, 2, 3, 4, 5]) / 15))
        weights = effective_number_weights(labels, 0.99)
        self.assertTrue(bool(torch.isfinite(weights).all()))
        spec = CandidateSpec("x", "softmax_ce", "sqrt_sampler")
        observed = sampling_prior(labels, spec)
        corrected = natural_prior_correction(observed[None, :], prior, observed)
        self.assertTrue(torch.allclose(corrected[0], prior))

    def test_slace_components_and_loss_are_finite(self) -> None:
        prior = torch.tensor([0.05, 0.15, 0.4, 0.3, 0.1])
        distance, soft, mask = slace_components(prior, 1.0)
        self.assertEqual((distance.shape, soft.shape, mask.shape), ((5, 5), (5, 5), (5, 5, 5)))
        self.assertTrue(torch.allclose(soft.sum(1), torch.ones(5)))
        loss = slace_loss(torch.randn(5, 5), torch.arange(1, 6), prior, 1.0)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_slace_is_count_aware_directed_and_matches_toy_formula(self) -> None:
        uniform = torch.full((5,), 0.2)
        skewed = torch.tensor([0.05, 0.10, 0.20, 0.25, 0.40])
        uniform_distance, uniform_soft, uniform_mask = slace_components(uniform, 1.0)
        skewed_distance, skewed_soft, skewed_mask = slace_components(skewed, 1.0)
        self.assertAlmostEqual(float(uniform_distance[2, 2]), -float(torch.log(torch.tensor(0.1))), places=6)
        self.assertAlmostEqual(float(uniform_distance[2, 1]), -float(torch.log(torch.tensor(0.3))), places=6)
        self.assertFalse(torch.allclose(uniform_distance, skewed_distance))
        self.assertFalse(torch.allclose(uniform_soft, skewed_soft))
        self.assertFalse(torch.equal(uniform_mask, skewed_mask))
        self.assertNotEqual(float(skewed_distance[0, 4]), float(skewed_distance[4, 0]))

    def test_nested_folds_are_isolated_and_use_embedding_only(self) -> None:
        rows = tuple(
            ResidualRow(str(i), f"g{i}", (float(i), 1.0), (9.0, 9.0, 9.0),
                        (3.0, 3.0, 3.0), (3, 3, 3), i % 5)
            for i in range(2000)
        )
        outer, inner = nested_indices(rows, 2)
        self.assertEqual(len(outer), 400)
        self.assertTrue(all(set(outer).isdisjoint(train) and set(outer).isdisjoint(dev)
                            for train, dev in inner.values()))
        self.assertEqual(_features(rows, [0, 1], 0).shape, (2, 2))
        self.assertNotIn(9.0, _features(rows, [0, 1], 0).flatten().tolist())

    def test_independent_axis_models(self) -> None:
        spec = CandidateSpec("x", "softmax_ce", "natural")
        first, second = build_axis_model(3, spec), build_axis_model(3, spec)
        self.assertIsNot(next(first.parameters()), next(second.parameters()))

    def test_public_privacy_rejects_row_content(self) -> None:
        _validate_public_payload({"metrics": {"rmse": 0.5}, "records": 2000})
        for key in ("source_id", "essay", "raw_gold", "candidate_prediction", "shared_embedding"):
            with self.assertRaisesRegex(OrdinalTailFixedFeatureError, "restricted row"):
                _validate_public_payload({"nested": [{key: "secret"}]})


if __name__ == "__main__":
    unittest.main()
