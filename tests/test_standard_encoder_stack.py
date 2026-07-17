from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest

from mal2026.standard_decoder_data import FEEDBACK_FIELDS, RestrictedRow
from mal2026.standard_encoder_data import encoder_input
from mal2026.standard_encoder_model import (
    EncoderModelSpec,
    NV_EMBEDDING_DIM,
    NVReview,
    StandardEncoderContractError,
    _nv_sentence_embeddings,
    verify_nv_snapshot,
)
from mal2026.standard_encoder_train import (
    StandardEncoderConfig,
    StandardEncoderTrainingError,
    _broadcast_rank_zero_finalization,
    _rank_zero_finalize,
    _selection_metrics_at_best_step,
    _training_arguments_kwargs,
    _validate_encoder_metric_health,
)

REVISION = "a" * 40


def qwen_mapping(**changes):
    value = {
        "backbone": "qwen3_embedding", "model_id": "Qwen/Qwen3-Embedding-8B", "revision": REVISION,
        "tokenizer_revision": REVISION, "model_path": "/private/qwen", "pooling": "last_nonpad",
        "normalize_embeddings": True, "lora_target_modules": ["q_proj", "v_proj"], "lora_r": 16,
        "lora_alpha": 32, "lora_dropout": 0.05, "nv_snapshot_dir": None, "nv_review": None,
    }
    value.update(changes)
    return value


class StandardEncoderModelTests(unittest.TestCase):
    def test_encoder_input_never_contains_id_or_gold_score(self):
        row = RestrictedRow(
            "private-id", "private prompt", "private essay",
            {"content": 1.0, "organization": 2.0, "expression": 3.0, "average": 4.0},
            {field: "private feedback" for field in FEEDBACK_FIELDS},
        )
        rendered = encoder_input(row)
        self.assertIn("private prompt", rendered)
        self.assertIn("private essay", rendered)
        self.assertNotIn("private-id", rendered)
        self.assertNotIn("4.0", rendered)

    def test_qwen_spec_requires_explicit_secure_architecture(self):
        self.assertEqual("qwen3_embedding", EncoderModelSpec.from_mapping(qwen_mapping()).backbone)
        for change in ({"revision": "main"}, {"normalize_embeddings": False}, {"pooling": "mean"}, {"lora_target_modules": ["latent_attention"]}):
            with self.subTest(change=change):
                with self.assertRaises(StandardEncoderContractError):
                    EncoderModelSpec.from_mapping(qwen_mapping(**change))

    def test_nv_snapshot_requires_complete_reviewed_python_inventory(self):
        source = b"# reviewed\n"
        digest = hashlib.sha256(source).hexdigest()
        review = NVReview.from_mapping({
            "model_id": "nvidia/NV-Embed-v2", "revision": REVISION, "license_acknowledged": True,
            "use_case": "research_noncommercial", "reviewer": "reviewer", "outcome": "approved",
            "reviewed_files": {"modeling_nvembed.py": digest},
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / REVISION
            root.mkdir()
            (root / "modeling_nvembed.py").write_bytes(source)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            self.assertEqual(root.resolve(), verify_nv_snapshot(root, review))
            (root / "extra.py").write_text("# not reviewed", encoding="utf-8")
            with self.assertRaises(StandardEncoderContractError):
                verify_nv_snapshot(root, review)

    def test_nv_pooling_has_no_generic_fallback(self):
        class Embedding:
            ndim = 2
            shape = (2, NV_EMBEDDING_DIM)
        self.assertIsInstance(_nv_sentence_embeddings({"sentence_embeddings": Embedding()}, 2), Embedding)
        with self.assertRaises(StandardEncoderContractError):
            _nv_sentence_embeddings({"sentence_embedding": Embedding()}, 2)


class StandardEncoderWandbTests(unittest.TestCase):
    def test_wandb_routing_is_explicit_and_sets_entity(self):
        import os
        from mal2026.standard_encoder_train import StandardEncoderConfig, _configure_wandb
        config = StandardEncoderConfig(
            run_id="encoder-run", phase="selection", backbone="qwen3_embedding", model_id="Qwen/Qwen3-Embedding-8B",
            model_revision=REVISION, tokenizer_revision=REVISION, model_path="/private/qwen", prepared_manifest="/manifest",
            output_dir="/out", nv_snapshot_dir=None, nv_review=None, wandb_project="expected-project", wandb_entity="expected-entity",
        )
        old = dict(os.environ)
        try:
            _configure_wandb(config)
            self.assertEqual("false", os.environ["WANDB_LOG_MODEL"])
            self.assertEqual("expected-project", os.environ["WANDB_PROJECT"])
            self.assertEqual("encoder-run", os.environ["WANDB_RUN_NAME"])
            self.assertEqual("expected-entity", os.environ["WANDB_ENTITY"])
        finally:
            os.environ.clear(); os.environ.update(old)


class StandardEncoderConfigTests(unittest.TestCase):
    def test_training_argument_keywords_bind_to_installed_transformers_signature(self):
        """Catch unsupported Trainer kwargs before a multi-GPU encoder launch."""
        from transformers import TrainingArguments

        config = StandardEncoderConfig(
            run_id="trainer-argument-contract", phase="selection", backbone="qwen3_embedding",
            model_id="Qwen/Qwen3-Embedding-8B", model_revision=REVISION, tokenizer_revision=REVISION,
            model_path="/private/qwen", prepared_manifest="/manifest", output_dir="/out",
            nv_snapshot_dir=None, nv_review=None,
        )
        kwargs = _training_arguments_kwargs(config, selected_steps=None)
        self.assertNotIn("overwrite_output_dir", kwargs)
        # ``bind`` uses the installed Transformers API but does not initialize
        # a Trainer, model, distributed process group, or GPU.  Re-adding an
        # unsupported keyword such as ``overwrite_output_dir`` fails here.
        inspect.signature(TrainingArguments).bind(**kwargs)

    def test_rejects_noncanonical_output_or_legacy_manual_settings(self):
        config = StandardEncoderConfig(
            run_id="test", phase="selection", backbone="qwen3_embedding", model_id="Qwen/Qwen3-Embedding-8B",
            model_revision=REVISION, tokenizer_revision=REVISION, model_path="/private/qwen",
            prepared_manifest=str(Path("data/manifests/aihub_human_feedback_v1.json").resolve()),
            output_dir="/tmp/not-standard", nv_snapshot_dir=None, nv_review=None,
        )
        with self.assertRaises(StandardEncoderTrainingError):
            config.validate()

    def test_source_never_imports_accelerate_or_a_custom_optimizer_loop(self):
        source = Path("src/mal2026/standard_encoder_train.py").read_text(encoding="utf-8")
        self.assertIn("from transformers import EarlyStoppingCallback, Trainer, TrainingArguments", source)
        self.assertNotIn("Accelerator", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("DistributedSampler", source)
        self.assertIn("trainer.train()", source)
        self.assertIn("trainer.save_model", source)

    def test_post_training_publication_is_rank_zero_only_and_does_not_reenter_distributed_eval(self):
        source = Path("src/mal2026/standard_encoder_train.py").read_text(encoding="utf-8")
        self.assertIn("trainer.accelerator.wait_for_everyone()", source)
        self.assertIn("if trainer.is_world_process_zero():", source)
        self.assertIn("_rank_zero_finalize(trainer, config, output, train_result.metrics)", source)
        self.assertIn("_broadcast_rank_zero_finalization(torch, trainer, payload, failed)", source)
        self.assertNotIn("trainer.evaluate()", source)

    def test_best_selection_metrics_come_from_the_matching_distributed_event(self):
        class State:
            log_history = [
                {"step": 4, "eval_primary_macro_mae": 0.5},
                {"step": 8, "eval_primary_macro_mae": 0.25, "eval_content_mae": 0.2},
            ]
        class Trainer:
            state = State()
        metrics = _selection_metrics_at_best_step(Trainer(), 8)
        self.assertEqual(0.25, metrics["eval_primary_macro_mae"])
        self.assertEqual(0.2, metrics["eval_content_mae"])
        with self.assertRaises(StandardEncoderTrainingError):
            _selection_metrics_at_best_step(Trainer(), 6)

    def test_rank_zero_finalization_failure_is_propagated_without_an_error_payload(self):
        class Distributed:
            @staticmethod
            def is_available(): return False
            @staticmethod
            def is_initialized(): return False
        class Torch:
            distributed = Distributed()
        with self.assertRaises(StandardEncoderTrainingError):
            _broadcast_rank_zero_finalization(Torch(), object(), None, True)
        payload = {"status": "completed", "aggregate": 1.0}
        self.assertEqual(payload, _broadcast_rank_zero_finalization(Torch(), object(), payload, False))


class StandardEncoderMetricHealthTests(unittest.TestCase):
    class State:
        def __init__(self, *, best_metric=0.25, event_metric=0.25, event_extra=0.2):
            self.global_step = 8
            self.best_global_step = 8
            self.best_model_checkpoint = "/ignored/checkpoint-8"
            self.best_metric = best_metric
            self.log_history = [
                {"loss": 1.0},
                {
                    "step": 8,
                    "eval_primary_macro_mae": event_metric,
                    "eval_content_mae": event_extra,
                },
            ]

    def test_selection_health_accepts_finite_matching_best_monitor(self):
        train, step, selection, best = _validate_encoder_metric_health(
            "selection", self.State(), {"train_loss": 0.8, "train_runtime": 1.2}
        )
        self.assertEqual({"train_loss": 0.8, "train_runtime": 1.2}, train)
        self.assertEqual(8, step)
        self.assertEqual(0.25, selection["eval_primary_macro_mae"])
        self.assertEqual(0.25, best)

    def test_metric_health_rejects_nan_or_infinity_in_all_persisted_sources(self):
        cases = [
            ("selection", self.State(), {"train_loss": float("nan")}),
            ("selection", self.State(), {"train_loss": 0.8, "train_runtime": float("inf")}),
            ("selection", self.State(event_metric=float("nan")), {"train_loss": 0.8}),
            ("selection", self.State(event_extra=float("inf")), {"train_loss": 0.8}),
            ("selection", self.State(best_metric=float("nan")), {"train_loss": 0.8}),
            ("selection", self.State(best_metric=float("inf")), {"train_loss": 0.8}),
            ("refit", type("RefitState", (), {"global_step": 1})(), {"train_loss": float("inf")}),
        ]
        for phase, state, metrics in cases:
            with self.subTest(phase=phase, metrics=metrics):
                with self.assertRaises(StandardEncoderTrainingError):
                    _validate_encoder_metric_health(phase, state, metrics)

    def test_rank_zero_gate_runs_before_save_or_completion_write(self):
        class Trainer:
            def __init__(self):
                self.state = type("State", (), {"global_step": 1})()
                self.save_called = False

            def save_model(self, _):
                self.save_called = True
                raise AssertionError("save_model must not run after a failed metric gate")

        config = StandardEncoderConfig(
            run_id="health-failure", phase="refit", backbone="qwen3_embedding", model_id="Qwen/Qwen3-Embedding-8B",
            model_revision=REVISION, tokenizer_revision=REVISION, model_path="/private/qwen", prepared_manifest="/manifest",
            output_dir="/out", nv_snapshot_dir=None, nv_review=None, selection_metadata_path="/selection/standard_encoder_training_complete.json",
        )
        trainer = Trainer()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(StandardEncoderTrainingError):
                _rank_zero_finalize(trainer, config, output, {"train_loss": math.nan})
            self.assertFalse(trainer.save_called)
            self.assertFalse((output / "standard_encoder_training_complete.json").exists())

    def test_static_gate_precedes_model_export_and_completion_write(self):
        source = Path("src/mal2026/standard_encoder_train.py").read_text(encoding="utf-8")
        finalizer = source[source.index("def _rank_zero_finalize"):source.index("def _broadcast_rank_zero_finalization")]
        self.assertLess(finalizer.index("_validate_encoder_metric_health"), finalizer.index("trainer.save_model"))
        self.assertLess(finalizer.index("_validate_encoder_metric_health"), finalizer.index("_write_complete(output, payload)"))


if __name__ == "__main__":
    unittest.main()
