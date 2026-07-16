from __future__ import annotations

import hashlib
import json
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
from mal2026.standard_encoder_train import StandardEncoderConfig, StandardEncoderTrainingError

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


if __name__ == "__main__":
    unittest.main()
