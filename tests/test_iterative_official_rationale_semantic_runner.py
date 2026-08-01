from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mal2026.iterative_official_rationale_semantic_models import (
    SEMANTIC_DIM,
    STRUCTURED_DIM,
    RationaleSemanticResult,
    candidate_specs,
)
from mal2026.iterative_official_rationale_semantic_runner import (
    OfficialRationaleSemanticData,
    _candidate_oof,
    load_official_rationale_semantic_data,
    run_outer_fold,
)
from mal2026.iterative_tail_runner import ExperimentData


def bundle_fixture() -> OfficialRationaleSemanticData:
    folds = np.repeat(np.arange(5), 2)
    base = np.full((10, 3), 3.0, dtype=np.float32)
    semantic = np.zeros((10, SEMANTIC_DIM), dtype=np.float64); semantic[:, 0] = folds
    structured = np.zeros((10, STRUCTURED_DIM), dtype=np.float64)
    data = ExperimentData(
        tuple(f"s{i}" for i in range(10)), tuple(f"d{i}" for i in range(10)),
        tuple("p" for _ in range(10)), np.zeros((10, 1), dtype=np.float32),
        base, base.copy(), folds, None,
    )
    return OfficialRationaleSemanticData(
        data, semantic, structured,
        {"records": 10, "semantic_dimensions": SEMANTIC_DIM, "structured_dimensions": STRUCTURED_DIM},
        {"candidate_count": 60, "source_count": 2, "row_content_in_provenance": False},
    )


class OfficialRationaleSemanticRunnerTest(unittest.TestCase):
    def test_score_free_canonical_hash_map_and_agent_source_split(self):
        source_ids = tuple(f"s{i}" for i in range(2000))
        experiment = SimpleNamespace(source_ids=source_ids, base=np.full((2000, 3), 3.0))
        writings = [SimpleNamespace(identifier=source_id, essay=f"essay {index}") for index, source_id in enumerate(source_ids)]
        dual_rows = [SimpleNamespace(agent_source=source) for source in ("terra", "luna") for _ in range(6000)]
        protocol = SimpleNamespace(raw={"lineage": {
            "terra_candidate_manifest_path": "terra-manifest", "terra_candidate_rows_path": "terra-rows",
            "luna_candidate_manifest_path": "luna-manifest", "luna_candidate_rows_path": "luna-rows",
            "generated_feature_manifest_path": "semantic-manifest",
            "generated_feature_rows_path": "semantic-rows",
        }})
        captured = {}

        def fake_load(*paths, essay_sha256_by_source):
            captured["paths"] = paths; captured["hashes"] = essay_sha256_by_source
            return dual_rows, {"candidate_count": 12000, "source_count": 2, "row_content_in_provenance": False}

        def fake_features(base, source_ids, terra, luna):
            captured["terra"] = terra; captured["luna"] = luna
            return np.zeros((2000, STRUCTURED_DIM)), {
                "records": 2000, "dimensions": STRUCTURED_DIM,
                "human_or_reference_score_read_or_prompted": False,
            }

        semantic_manifest = {
            "records": 2000, "feature_dim": SEMANTIC_DIM,
            "feature_matrix_sha256": "1" * 64, "feature_rows_sha256": "2" * 64,
            "model_id": "Qwen/Qwen3-Embedding-8B", "model_revision": "r",
            "projection_matrix_sha256": "3" * 64,
            "candidate_score_in_embedding_text": False, "validation_loaded": False,
        }
        semantic_rows = tuple(
            SimpleNamespace(features=tuple([0.0] * SEMANTIC_DIM)) for _ in source_ids
        )

        with (
            patch("mal2026.iterative_official_rationale_semantic_runner.load_protocol", return_value=protocol),
            patch("mal2026.iterative_official_rationale_semantic_runner.validate_bound_inputs"),
            patch("mal2026.iterative_official_rationale_semantic_runner.load_experiment_data", return_value=experiment),
            patch("mal2026.iterative_official_rationale_semantic_runner.load_writing_rows", return_value=writings) as score_free,
            patch("mal2026.iterative_official_rationale_semantic_runner.load_dual_candidates", side_effect=fake_load),
            patch("mal2026.iterative_official_rationale_semantic_runner.build_dual_agent_features", side_effect=fake_features),
            patch("mal2026.iterative_official_rationale_semantic_runner.load_feature_artifact", return_value=(semantic_manifest, semantic_rows)),
        ):
            result = load_official_rationale_semantic_data()
        score_free.assert_called_once_with("train", include_scores=False)
        self.assertEqual(sha256("essay 0".encode()).hexdigest(), captured["hashes"]["s0"])
        self.assertEqual(6000, len(captured["terra"])); self.assertEqual(6000, len(captured["luna"]))
        self.assertEqual((2000, STRUCTURED_DIM), result.structured_features.shape)
        self.assertEqual((2000, SEMANTIC_DIM), result.semantic_features.shape)

    def test_candidate_oof_exact_coverage_and_fold_exclusion(self):
        bundle = bundle_fixture(); calls = []

        def fake(spec, train_semantic, train_structured, train_base, train_targets,
                 predict_semantic, predict_structured, predict_base, *, device):
            calls.append((set(train_semantic[:, 0].astype(int)), set(predict_semantic[:, 0].astype(int))))
            return RationaleSemanticResult(np.asarray(predict_base, dtype=np.float64), {"fresh_initialization": True})

        with patch("mal2026.iterative_official_rationale_semantic_runner.fit_predict_rationale_semantic", side_effect=fake):
            prediction, audit = _candidate_oof(bundle, candidate_specs()[0], 0, device="cuda:0")
        self.assertEqual(4, len(calls)); self.assertEqual(4, len(audit))
        self.assertTrue(np.isnan(prediction[bundle.experiment.folds == 0]).all())
        self.assertTrue(np.isfinite(prediction[bundle.experiment.folds != 0]).all())
        for train_folds, predict_folds in calls:
            self.assertNotIn(0, train_folds | predict_folds)
            self.assertEqual(3, len(train_folds)); self.assertEqual(1, len(predict_folds))
            self.assertTrue(train_folds.isdisjoint(predict_folds))

    def test_outer_shard_completes_all_three_before_baseline_fallback(self):
        bundle = bundle_fixture(); specs = candidate_specs(); completed = []

        def fake_oof(_bundle, spec, outer_fold, *, device):
            completed.append(spec.variant_id)
            prediction = np.full_like(bundle.experiment.targets, np.nan, dtype=np.float64)
            prediction[bundle.experiment.folds != outer_fold] = 3.0
            return prediction, [{"variant_id": spec.variant_id}]

        def fake_select(given_specs, metrics_by_id, baseline, config):
            self.assertEqual([spec.variant_id for spec in specs], completed)
            self.assertEqual(set(completed), set(metrics_by_id))
            return {"selected_id": "baseline", "fell_back_to_baseline": True, "inventory_valid": True,
                    "decisions": [{"variant_id": spec.variant_id, "eligible": False} for spec in given_specs]}

        with tempfile.TemporaryDirectory() as temp:
            with (
                patch("mal2026.iterative_official_rationale_semantic_runner.PUBLIC_ROOT", Path(temp) / "public"),
                patch("mal2026.iterative_official_rationale_semantic_runner.RESTRICTED_ROOT", Path(temp) / "restricted"),
                patch("mal2026.iterative_official_rationale_semantic_runner.load_protocol", return_value=SimpleNamespace(raw={})),
                patch("mal2026.iterative_official_rationale_semantic_runner.validate_bound_inputs"),
                patch("mal2026.iterative_official_rationale_semantic_runner.load_official_rationale_semantic_data", return_value=bundle),
                patch("mal2026.iterative_official_rationale_semantic_runner._candidate_oof", side_effect=fake_oof),
                patch("mal2026.iterative_official_rationale_semantic_runner.select_candidate", side_effect=fake_select),
                patch("mal2026.iterative_official_rationale_semantic_runner.compute_iterative_tail_metrics", return_value={"macro": {"rmse": .5}}),
            ):
                result = run_outer_fold(0, device="cuda:0")
                self.assertEqual(2, len((Path(temp) / "restricted/outer-0/predictions.jsonl").read_text().splitlines()))
        self.assertEqual([spec.variant_id for spec in specs], completed)
        self.assertTrue(result["fell_back_to_baseline"])

    def test_outer_shard_refits_only_frozen_selected_candidate(self):
        bundle = bundle_fixture(); specs = candidate_specs(); completed = []

        def fake_oof(_bundle, spec, outer_fold, *, device):
            completed.append(spec.variant_id)
            prediction = np.full_like(bundle.experiment.targets, np.nan, dtype=np.float64)
            prediction[bundle.experiment.folds != outer_fold] = 3.0
            return prediction, [{"variant_id": spec.variant_id}]

        def fake_select(given_specs, metrics_by_id, baseline, config):
            self.assertEqual([spec.variant_id for spec in specs], completed)
            return {
                "selected_id": specs[1].variant_id,
                "fell_back_to_baseline": False,
                "inventory_valid": True,
                "decisions": [
                    {"variant_id": spec.variant_id, "eligible": spec == specs[1]}
                    for spec in given_specs
                ],
            }

        with tempfile.TemporaryDirectory() as temp:
            with (
                patch("mal2026.iterative_official_rationale_semantic_runner.PUBLIC_ROOT", Path(temp) / "public"),
                patch("mal2026.iterative_official_rationale_semantic_runner.RESTRICTED_ROOT", Path(temp) / "restricted"),
                patch("mal2026.iterative_official_rationale_semantic_runner.load_protocol", return_value=SimpleNamespace(raw={})),
                patch("mal2026.iterative_official_rationale_semantic_runner.validate_bound_inputs"),
                patch("mal2026.iterative_official_rationale_semantic_runner.load_official_rationale_semantic_data", return_value=bundle),
                patch("mal2026.iterative_official_rationale_semantic_runner._candidate_oof", side_effect=fake_oof),
                patch("mal2026.iterative_official_rationale_semantic_runner.select_candidate", side_effect=fake_select),
                patch("mal2026.iterative_official_rationale_semantic_runner._outer_refit", return_value=(np.full((2, 3), 3.25), {"fresh_initialization": True})) as refit,
                patch("mal2026.iterative_official_rationale_semantic_runner.compute_iterative_tail_metrics", return_value={"macro": {"rmse": .5}}),
            ):
                result = run_outer_fold(0, device="cuda:0")
        refit.assert_called_once_with(bundle, specs[1], 0, device="cuda:0")
        self.assertEqual(specs[1].variant_id, result["selected_candidate"])
        self.assertFalse(result["fell_back_to_baseline"])


if __name__ == "__main__":
    unittest.main()
