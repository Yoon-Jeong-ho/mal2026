from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mal2026.kure_crt_recovery import (
    KURECRTRecoveryConfig, KURECRTRecoveryError, _assert_private_file, _atomic_private_jsonl,
    _load_fit_and_held_text, cRT_update_budget, initialize_crt_head, prior_sanity_gate, run,
    train_cached_crt_head,
)
from mal2026.kure_ordinal_oof import KUREOrdinalOOFConfig


class KURECRTRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = json.loads(Path("configs/kure_crt_recovery.v1.json").read_text(encoding="utf-8"))
        cls.stage3 = json.loads(Path("configs/kure_ordinal_oof.v1.json").read_text(encoding="utf-8"))

    def config(self) -> KURECRTRecoveryConfig:
        return KURECRTRecoveryConfig.from_mapping(self.raw)

    def test_fixed_replay_and_exact_update_budget(self) -> None:
        config = self.config()
        self.assertEqual((config.phase1_epochs, config.phase1_learning_rate, config.phase1_weight_decay), (6, 5e-5, .01))
        self.assertEqual((config.batch_size, config.gradient_accumulation_steps, config.crt_epochs, config.crt_updates), (20, 2, 20, 800))
        self.assertEqual(cRT_update_budget(1600, 20, 2, 20), 800)
        with self.assertRaises(KURECRTRecoveryError):
            cRT_update_budget(1599, 20, 2, 20)

    def test_source_fold_hashes_and_backbone_replay_are_bound(self) -> None:
        config = self.config()
        for key in ("train_sha256", "fold_manifest_sha256", "fold_rows_sha256", "r0_oof_prediction_sha256", "backbone", "seed", "batch_size", "gradient_accumulation_steps", "max_length"):
            value = asdict(getattr(config, key)) if key == "backbone" else getattr(config, key)
            self.assertEqual(value, self.stage3[key])
        self.assertEqual(config.source_stage3_config_sha256, __import__("hashlib").sha256(Path("configs/kure_ordinal_oof.v1.json").read_bytes()).hexdigest())

    def test_head_is_zero_weight_and_empirical_log_prior_bias(self) -> None:
        head = torch.nn.Linear(1024, 5)
        labels = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5]
        q, mean = initialize_crt_head(head, labels)
        expected_q = np.bincount(labels, minlength=6)[1:] / len(labels)
        self.assertTrue(torch.equal(head.weight, torch.zeros_like(head.weight)))
        self.assertTrue(np.allclose(q, expected_q))
        self.assertTrue(torch.allclose(head.bias.detach().cpu(), torch.tensor(np.log(expected_q), dtype=head.bias.dtype)))
        self.assertAlmostEqual(mean, float(np.dot(expected_q, np.arange(1, 6))))

    def test_fit_only_bias_prior_smoke(self) -> None:
        labels = [1] * 2 + [2] * 20 + [3] * 80 + [4] * 50 + [5] * 8
        result = prior_sanity_gate(labels, steps=160, learning_rate=.05, weight_decay=.01)
        self.assertLessEqual(result["pmf_max_abs_error"], .02)
        self.assertLessEqual(result["expected_ordinal_label_mean_error"], .02)
        self.assertLessEqual(result["cross_entropy"], result["empirical_entropy"] + .005)
        self.assertEqual((result["learning_rate"], result["weight_decay"]), (.05, .01))

    def test_cached_head_counts_tiny_update_budget(self) -> None:
        config = self.config()
        labels = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5] * 4
        features = torch.randn(40, 1024)
        head, metadata = train_cached_crt_head(features, labels, [float(x) for x in labels], config, seed=7, device=torch.device("cpu"), max_updates=1)
        self.assertEqual(head.weight.shape, (5, 1024))
        self.assertEqual(metadata["updates"], 1)
        self.assertEqual((metadata["configured_epochs"], metadata["completed_epochs"]), (20, 1))
        self.assertEqual(metadata["scheduler"], "constant_no_warmup")
        with self.assertRaisesRegex(KURECRTRecoveryError, "fill every"):
            train_cached_crt_head(torch.randn(10, 1024), [1, 2, 3, 4, 5] * 2,
                                  [1., 2., 3., 4., 5.] * 2, config,
                                  seed=7, device=torch.device("cpu"), max_updates=1)

    def test_no_held_input_reaches_fit_only_prior_gate(self) -> None:
        source = inspect.getsource(run)
        self.assertIn("_load_fit_and_held_text(source, outer_fold)", source)
        self.assertNotIn("load_train_and_folds", source)
        self.assertIn("sanity = prior_sanity_gate(labels", source)
        self.assertLess(source.index("sanity = prior_sanity_gate(labels"), source.index("held_features = cache_cls_l2_features"))
        self.assertNotIn("held,", inspect.getsource(prior_sanity_gate))

    def test_held_loader_never_materializes_labels(self) -> None:
        source = KUREOrdinalOOFConfig.from_json("configs/kure_ordinal_oof.v1.json", require_dependencies=True)
        fit, held = _load_fit_and_held_text(source, 0)
        self.assertEqual((len(fit), len(held)), (1600, 400))
        self.assertTrue(all(hasattr(row, "labels") for row in fit))
        self.assertTrue(all(not hasattr(row, "labels") for row in held))
        self.assertFalse({row.identifier for row in fit} & {row.identifier for row in held})

    def test_acl_accepts_project_0770_but_rejects_world_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "restricted" / "rows.jsonl"
            _atomic_private_jsonl(path, [{"source_id": "private"}])
            self.assertEqual(path.stat().st_mode & 0o777, 0o660)
            os.chmod(path, 0o770); _assert_private_file(path)
            os.chmod(path, 0o774)
            with self.assertRaisesRegex(KURECRTRecoveryError, "project-private"):
                _assert_private_file(path)

    def test_validation_and_average_are_forbidden_and_smoke_is_nonselectable(self) -> None:
        bad = dict(self.raw); bad["note"] = "validation is forbidden"
        with self.assertRaisesRegex(KURECRTRecoveryError, "validation"):
            KURECRTRecoveryConfig.from_mapping(bad)
        bad = dict(self.raw, axes=["content", "organization", "expression", "average"])
        with self.assertRaisesRegex(KURECRTRecoveryError, "average"):
            KURECRTRecoveryConfig.from_mapping(bad)
        with self.assertRaisesRegex(KURECRTRecoveryError, "outer fold 0"):
            run(self.config(), outer_fold=1, validate_only=True, smoke=True)

    def test_aggregate_reuses_frozen_common_gate(self) -> None:
        source = inspect.getsource(__import__("mal2026.kure_crt_recovery", fromlist=["aggregate"]).aggregate)
        self.assertIn("decision = promotion_gate(", source)
        self.assertIn('"common_stage3_promotion_gate": decision', source)
        self.assertIn('"automatic_stage6_deployment_eligible": False', source)


if __name__ == "__main__":
    unittest.main()
