"""CPU-only contract tests for the exact R0 OOF single-fold runner."""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[1]


def _runner():
    path = ROOT / "scripts" / "run_r0_exact_oof_fold.py"
    spec = importlib.util.spec_from_file_location("run_r0_exact_oof_fold", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _population(module):
    examples = []
    lineage = []
    for index in range(module.TRAIN_RECORDS):
        identifier = f"source-{index:04d}"
        examples.append({
            "source_id": identifier,
            "text": f"private-{index}",
            "labels": [float(index % 5 + 1), float((index // 5) % 5 + 1), 3.0],
        })
        lineage.append(SimpleNamespace(
            identifier=identifier,
            document_id=f"document-{index:04d}",
            prompt_num=f"prompt-{index % 4}",
        ))
    return examples, lineage


class R0ExactOOFFoldTests(unittest.TestCase):
    def test_protocol_is_exact_r0_without_average_target(self) -> None:
        runner = _runner()
        protocol = runner.exact_protocol()
        self.assertEqual(protocol["arm"], "qwen3_aihub_warmstart")
        self.assertEqual(protocol["source_key"], "rank2_ax4_random1")
        self.assertEqual(protocol["score_fields"], ["content", "organization", "expression"])
        self.assertNotIn("average", protocol["score_fields"])
        self.assertEqual(runner.EPOCHS, (1, 2, 3, 4))
        self.assertEqual(runner.GRADIENT_ACCUMULATION_STEPS, 16)
        self.assertEqual(
            runner.PER_DEVICE_TRAIN_BATCH_SIZE * runner.GRADIENT_ACCUMULATION_STEPS, 64
        )
        self.assertEqual(
            runner.expected_checkpoint_steps(), {1: 25, 2: 50, 3: 75, 4: 100}
        )

    def test_each_fold_is_balanced_complete_and_source_document_disjoint(self) -> None:
        runner = _runner()
        examples, lineage = _population(runner)
        observed = set()
        fingerprint = None
        for fold in range(runner.FOLDS):
            train, heldout, assignments, gate = runner.prepare_fold_population(
                examples, lineage, fold
            )
            self.assertEqual(len(train), 1600)
            self.assertEqual(len(heldout), 400)
            self.assertEqual(gate, {
                "source_id_disjoint": True,
                "document_id_disjoint": True,
                "complete_train_2000_coverage": True,
                "train_records": 1600,
                "heldout_records": 400,
            })
            heldout_ids = {row["source_id"] for row in heldout}
            self.assertTrue(observed.isdisjoint(heldout_ids))
            observed.update(heldout_ids)
            current = runner.assignment_fingerprint(assignments)
            fingerprint = current if fingerprint is None else fingerprint
            self.assertEqual(current, fingerprint)
        self.assertEqual(len(observed), 2000)

    def test_epoch_prediction_mean_and_half_up_projection_are_exact(self) -> None:
        runner = _runner()
        epochs = [
            [[2.0, 2.0, 4.0]],
            [[2.0, 3.0, 4.0]],
            [[3.0, 3.0, 5.0]],
            [[3.0, 4.0, 5.0]],
        ]
        mean = runner.uniform_epoch_mean(epochs)
        self.assertEqual(mean, [[2.5, 3.0, 4.5]])
        rows = runner.row_predictions(
            [{"source_id": "restricted-id", "labels": [2.0, 3.0, 5.0]}], mean, 2
        )
        self.assertEqual(rows[0]["continuous_prediction"], {
            "content": 2.5, "organization": 3.0, "expression": 4.5,
        })
        self.assertEqual(rows[0]["half_up_integer_prediction"], {
            "content": 3, "organization": 3, "expression": 5,
        })

    def test_cuda_visibility_and_fresh_output_gate(self) -> None:
        runner = _runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(run_id="r0-exact-oof-test", fold=0, physical_gpu=2)
            with patch.object(runner, "OUTPUT_ROOT", root / "outputs"), patch.object(
                runner, "RESTRICTED_ROOT", root / "restricted"
            ), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}):
                output, restricted = runner.validate_runtime_args(args)
                self.assertEqual(output.name, "fold-00")
                self.assertEqual(restricted.name, "fold-00")
                output.mkdir(parents=True)
                with self.assertRaisesRegex(runner.R0ExactOOFError, "fresh"):
                    runner.validate_runtime_args(args)
            with patch.object(runner, "OUTPUT_ROOT", root / "outputs-2"), patch.object(
                runner, "RESTRICTED_ROOT", root / "restricted-2"
            ), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
                with self.assertRaisesRegex(runner.R0ExactOOFError, "visibility"):
                    runner.validate_runtime_args(
                        argparse.Namespace(run_id="another-run", fold=0, physical_gpu=2)
                    )

    def test_runner_never_loads_or_scores_validation(self) -> None:
        runner = _runner()
        source = inspect.getsource(runner)
        self.assertIn('_examples("train", TRAIN_RECORDS)', source)
        self.assertNotIn('_examples("validation"', source)
        self.assertNotIn('load_writing_rows("validation"', source)
        self.assertIn('trainer.predict(heldout_dataset)', source)
        self.assertNotIn('trainer.predict(train_dataset)', source)
        self.assertIn('"validation_rows_loaded": False', source)


if __name__ == "__main__":
    unittest.main()
