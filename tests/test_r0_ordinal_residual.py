from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mal2026.r0_ordinal_residual import (
    AXES,
    BasePredictionContract,
    EMBEDDING_SCHEMA_VERSION,
    EmbeddingArtifactManifest,
    GOLD_LABEL_POLICY,
    R0OrdinalResidualContractError,
    ResidualRunConfig,
    _temperature_probabilities,
    aggregate_predictions,
    axis_class_labels,
    blend_axis_posteriors,
    build_ordinal_residual_model,
    group_selection_split,
    load_embedding_artifact,
    rounded_axis_label,
    validate_split_contracts,
    write_public_aggregate,
)


def contract(role: str, **changes):
    value = {
        "split_role": role,
        "base_prediction_origin": "oof" if role == "train" else "held_out",
        "base_model_fit_excludes_split": True,
        "evaluation_only": role == "validation",
        "gold_label_policy": GOLD_LABEL_POLICY,
        "contains_average_target": False,
    }
    value.update(changes)
    return BasePredictionContract.from_mapping(value)


class TargetAndLeakageContractTests(unittest.TestCase):
    def test_raw_axes_are_rounded_half_up_and_average_is_forbidden(self):
        self.assertEqual(3, rounded_axis_label(2.5))
        self.assertEqual((2, 3, 5), axis_class_labels({"content": 1.5, "organization": 3.49, "expression": 4.5}))
        with self.assertRaisesRegex(R0OrdinalResidualContractError, "average"):
            axis_class_labels({"content": 2.0, "organization": 3.0, "expression": 4.0, "average": 3.0})

    def test_training_predictions_must_be_oof(self):
        with self.assertRaisesRegex(R0OrdinalResidualContractError, "OOF"):
            contract("train", base_prediction_origin="full_fit")
        validate_split_contracts(contract("train"), contract("validation"))

    def test_validation_is_held_out_and_evaluation_only(self):
        for change in ({"base_model_fit_excludes_split": False}, {"evaluation_only": False}, {"base_prediction_origin": "oof"}):
            with self.subTest(change=change), self.assertRaises(R0OrdinalResidualContractError):
                contract("validation", **change)


class ModelAndDecodeTests(unittest.TestCase):
    def test_posthoc_temperature_changes_confidence_without_changing_shape(self):
        import torch

        logits = torch.tensor([[[0.0, 0.0, 2.0, 0.0, 0.0]] * 3])
        cold = _temperature_probabilities(logits, 0.5)
        warm = _temperature_probabilities(logits, 2.0)
        self.assertEqual((1, 3, 5), (len(cold), len(cold[0]), len(cold[0][0])))
        self.assertGreater(cold[0][0][2], warm[0][0][2])

    def test_five_way_residual_model_returns_trainer_loss_and_all_probabilities(self):
        import torch

        torch.manual_seed(7)
        model = build_ordinal_residual_model(embedding_dim=4, hidden_dim=8, dropout=0.0)
        output = model(
            shared_embedding=torch.randn(2, 4),
            base_predictions=torch.tensor([[2.2, 3.4, 4.1], [1.8, 2.9, 3.7]]),
            labels=torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.long),
            raw_labels=torch.tensor([[2.2, 3.1, 4.2], [1.8, 2.8, 3.9]]),
        )
        self.assertEqual((2, len(AXES), 5), tuple(output["logits"].shape))
        self.assertEqual((2, len(AXES), 5), tuple(output["probabilities"].shape))
        self.assertTrue(torch.allclose(output["probabilities"].sum(-1), torch.ones(2, 3)))
        self.assertEqual(0, output["loss"].ndim)
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_model_rejects_float_average_style_labels(self):
        import torch

        model = build_ordinal_residual_model(2, hidden_dim=4, dropout=0.0)
        with self.assertRaisesRegex(R0OrdinalResidualContractError, "integer classes"):
            model(torch.zeros(1, 2), torch.full((1, 3), 3.0), labels=torch.full((1, 3), 3.2))

    def test_decode_is_soft_blend_not_hard_cascade(self):
        probabilities = ((0.05, 0.10, 0.60, 0.20, 0.05),) * 3
        decoded = blend_axis_posteriors((2.7, 3.2, 4.0), probabilities, strategy="expected_risk")
        self.assertEqual(3, len(decoded))
        for result in decoded:
            self.assertEqual(5, len(result["probabilities_1_to_5"]))
            self.assertGreater(result["posterior_weight"], 0.0)
            self.assertLess(result["posterior_weight"], 1.0)
            self.assertIn(result["blended_integer_class"], range(1, 6))


class PublicAggregateAndRunnerTests(unittest.TestCase):
    def test_validation_rows_are_parsed_only_after_train_only_selection(self):
        source = Path("src/mal2026/r0_ordinal_residual.py").read_text(encoding="utf-8")
        run_source = source[source.index("def run_residual_experiment"):]
        selection = run_source.index("_, selected_seed, selected_temperature")
        refit_model = run_source.index("refit_model = build_ordinal_residual_model")
        refit_train = run_source.index("final_trainer.train()")
        validation_load = run_source.index("_, validation_rows = load_embedding_artifact")
        validation_predict = run_source.index("validation_output = final_trainer.predict")
        self.assertLess(selection, refit_model)
        self.assertLess(refit_model, refit_train)
        self.assertLess(refit_train, validation_load)
        self.assertLess(validation_load, validation_predict)
        self.assertEqual(1, run_source.count('metric_key_prefix="validation_final_once"'))

    def test_all_model_seeds_share_one_fixed_selection_split(self):
        source = Path("src/mal2026/r0_ordinal_residual.py").read_text(encoding="utf-8")
        run_source = source[source.index("def run_residual_experiment"):]
        split_call = run_source.index("selection_train, selection_dev = group_selection_split")
        seed_loop = run_source.index("for seed in config.seeds:")
        self.assertLess(split_call, seed_loop)
        self.assertEqual(1, run_source.count("selection_train, selection_dev = group_selection_split"))
        self.assertIn("seed=config.selection_split_seed", run_source[split_call:seed_loop])

    def test_public_writer_accepts_only_numeric_aggregates(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "aggregate.json"
            write_public_aggregate(output, {"metrics": {"count": 3, "rate": 0.5}})
            self.assertEqual(3, json.loads(output.read_text(encoding="utf-8"))["metrics"]["count"])
            with self.assertRaises(R0OrdinalResidualContractError):
                write_public_aggregate(Path(directory) / "bad.json", {"rows": [1, 2, 3]})

    def test_embedding_artifact_schema_enforces_exact_oof_and_group_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "train.jsonl"
            rows = []
            for fold in range(5):
                rows.append({
                    "source_id": f"restricted-{fold}", "group_id": f"group-{fold}",
                    "shared_embedding": [0.1, 0.2],
                    "base_continuous_prediction": {axis: 3.0 for axis in AXES},
                    "raw_continuous_gold": {"content": 2.5, "organization": 3.2, "expression": 4.4},
                    "oof_fold": fold,
                })
            rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({
                "schema_version": EMBEDDING_SCHEMA_VERSION, "split_role": "train",
                "base_prediction_origin": "oof", "base_model_fit_excludes_split": True,
                "evaluation_only": False, "gold_label_policy": GOLD_LABEL_POLICY,
                "contains_average_target": False, "embedding_model_id": "Qwen/frozen",
                "embedding_model_revision": "a" * 40, "embedding_source": "aihub_warm",
                "embedding_frozen": True, "embedding_dim": 2, "fold_count": 5,
                "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            manifest, loaded = load_embedding_artifact(manifest_path, rows_path)
            self.assertIsInstance(manifest, EmbeddingArtifactManifest)
            self.assertEqual(5, len(loaded))
            selection_train, selection_dev = group_selection_split(loaded, seed=2026, dev_ratio=0.2)
            self.assertFalse({row.group_id for row in selection_train} & {row.group_id for row in selection_dev})

    def test_aggregate_selection_metrics_preserve_raw_rmse(self):
        probabilities = [((0.0, 0.1, 0.8, 0.1, 0.0),) * 3]
        metrics = aggregate_predictions(
            [(2.8, 3.1, 3.2)], [(3.0, 3.0, 3.0)], probabilities,
            blend_weight=1.0, thresholds=(1.5, 2.5, 3.5, 4.5),
        )
        self.assertEqual(3, metrics["overall"]["count"])
        self.assertAlmostEqual(0.0, metrics["overall"]["residual_raw_rmse"])
        self.assertGreater(metrics["overall"]["base_raw_rmse"], 0.0)
        self.assertGreaterEqual(metrics["overall"]["posterior_nll"], 0.0)
        self.assertGreaterEqual(metrics["overall"]["posterior_multiclass_brier"], 0.0)
        self.assertGreaterEqual(metrics["overall"]["posterior_confidence_ece_10bin"], 0.0)
        self.assertIn("predicted_3_conditional_accuracy", metrics["overall"])
        self.assertIn("predicted_4_coverage", metrics["overall"])
        self.assertIn("posterior_predicted_3_conditional_accuracy", metrics["overall"])
        self.assertIn("posterior_predicted_4_coverage", metrics["overall"])
        self.assertEqual(10, len(metrics["overall"]["confidence_bins"]))
        self.assertEqual(set(AXES), set(metrics["by_axis"]))
        for axis in AXES:
            self.assertEqual(1, metrics["by_axis"][axis]["count"])
            self.assertIn("base_raw_mae", metrics["by_axis"][axis])

    def test_old_config_without_temperature_candidates_gets_safe_default(self):
        config = ResidualRunConfig(
            run_id="backward", train_manifest="train-manifest", train_rows="train-rows",
            validation_manifest="validation-manifest", validation_rows="validation-rows",
            output_dir="model-output", public_aggregate_path="aggregate.json",
        )
        raw = asdict(config)
        raw.pop("posterior_temperature_candidates")
        raw.pop("selection_split_seed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            loaded = ResidualRunConfig.from_json(path)
        self.assertEqual((1.0,), loaded.posterior_temperature_candidates)
        self.assertEqual(1729, loaded.selection_split_seed)

    def test_runner_cli_exposes_config_only_without_running_training(self):
        completed = subprocess.run(
            [sys.executable, "scripts/run_r0_ordinal_residual.py", "--help"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("--config", completed.stdout)
        self.assertNotIn("--validation-contract", completed.stdout)


if __name__ == "__main__":
    unittest.main()
    ResidualRunConfig,
    aggregate_predictions,
