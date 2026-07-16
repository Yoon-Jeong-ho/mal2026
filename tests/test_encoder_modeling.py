"""Static, dependency-free contract tests for encoder experimentation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.encoder_modeling import (  # noqa: E402
    EncoderContractError,
    EncoderModelSpec,
    NVRemoteCodeReview,
    NV_EMBEDDING_DIMENSION,
    _remote_sentence_embeddings,
    enforce_nv_offline_runtime,
    verify_nv_snapshot,
)


REVISION = "a" * 40


def qwen_spec(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "backbone": "qwen3_embedding",
        "model_id": "Qwen/Qwen3-Embedding-8B",
        "revision": REVISION,
        "tokenizer_revision": REVISION,
        "pooling": "last_nonpad",
        "normalize_embeddings": True,
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "regression_loss": "mse",
        "loss_reduction": "mean",
    }
    value.update(updates)
    return value


class EncoderSpecTest(unittest.TestCase):
    def test_qwen_requires_pinned_revision_and_explicit_architecture(self) -> None:
        spec = EncoderModelSpec.from_mapping(qwen_spec())
        self.assertEqual(spec.pooling, "last_nonpad")
        for key, invalid in (("revision", "main"), ("normalize_embeddings", False), ("pooling", "mean")):
            with self.subTest(key=key):
                with self.assertRaises(EncoderContractError):
                    EncoderModelSpec.from_mapping(qwen_spec(**{key: invalid}))

    def test_qwen_rejects_implicit_lora_and_loss_policy(self) -> None:
        with self.assertRaises(EncoderContractError):
            EncoderModelSpec.from_mapping(qwen_spec(lora_target_modules=[]))
        with self.assertRaises(EncoderContractError):
            EncoderModelSpec.from_mapping(qwen_spec(regression_loss="huber"))
        with self.assertRaises(EncoderContractError):
            EncoderModelSpec.from_mapping(qwen_spec(loss_reduction="sum"))


class NVReviewTest(unittest.TestCase):
    def _review(self, digest: str) -> NVRemoteCodeReview:
        return NVRemoteCodeReview.from_mapping(
            {
                "model_id": "nvidia/NV-Embed-v2",
                "revision": REVISION,
                "license_acknowledged": True,
                "use_case": "research_noncommercial",
                "reviewer": "research-code-review",
                "outcome": "approved",
                "reviewed_files": {"modeling_nvembed.py": digest},
            }
        )

    def test_nv_snapshot_requires_exact_reviewed_python_file_hashes(self) -> None:
        content = b"# reviewed model source\n"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / REVISION
            root.mkdir()
            (root / "modeling_nvembed.py").write_bytes(content)
            verify_nv_snapshot(root, self._review(digest))
            (root / "extra.py").write_text("# unreviewed", encoding="utf-8")
            with self.assertRaises(EncoderContractError):
                verify_nv_snapshot(root, self._review(digest))

    def test_nv_review_rejects_license_or_reviewer_gate_failures(self) -> None:
        digest = "b" * 64
        for key, invalid in (("license_acknowledged", False), ("outcome", "pending"), ("use_case", "commercial")):
            payload = {
                "model_id": "nvidia/NV-Embed-v2",
                "revision": REVISION,
                "license_acknowledged": True,
                "use_case": "research_noncommercial",
                "reviewer": "reviewer",
                "outcome": "approved",
                "reviewed_files": {"modeling_nvembed.py": digest},
            }
            payload[key] = invalid
            with self.subTest(key=key):
                with self.assertRaises(EncoderContractError):
                    NVRemoteCodeReview.from_mapping(payload)

    def test_nv_spec_cannot_omit_snapshot_or_review(self) -> None:
        value = qwen_spec(
            backbone="nv_embed_v2",
            model_id="nvidia/NV-Embed-v2",
            pooling="remote_sentence_embedding",
        )
        with self.assertRaises(EncoderContractError):
            EncoderModelSpec.from_mapping(value)
        value["nv_snapshot_dir"] = f"/local/{REVISION}"
        value["nv_remote_code_review"] = {
            "model_id": "nvidia/NV-Embed-v2", "revision": REVISION,
            "license_acknowledged": True, "use_case": "research_noncommercial",
            "reviewer": "reviewer", "outcome": "approved", "reviewed_files": {"modeling.py": "a" * 64},
        }
        self.assertEqual(EncoderModelSpec.from_mapping(value).backbone, "nv_embed_v2")

    def test_nv_review_rejects_string_false_without_bool_coercion(self) -> None:
        payload = {
            "model_id": "nvidia/NV-Embed-v2", "revision": REVISION,
            "license_acknowledged": "false", "use_case": "research_noncommercial",
            "reviewer": "reviewer", "outcome": "approved", "reviewed_files": {"modeling.py": "a" * 64},
        }
        with self.assertRaises(EncoderContractError):
            NVRemoteCodeReview.from_mapping(payload)

    def test_nv_review_rejects_unknown_fields(self) -> None:
        payload = {
            "model_id": "nvidia/NV-Embed-v2", "revision": REVISION,
            "license_acknowledged": True, "use_case": "research_noncommercial",
            "reviewer": "reviewer", "outcome": "approved", "reviewed_files": {"modeling.py": "a" * 64}, "extra": True,
        }
        with self.assertRaises(EncoderContractError):
            NVRemoteCodeReview.from_mapping(payload)


class NVInterfaceTest(unittest.TestCase):
    class _Embedding:
        def __init__(self, batch: int, width: int, rank: int = 2) -> None:
            self.shape = (batch, width) if rank == 2 else (batch, 1, width)
            self.ndim = rank

    def test_nv_accepts_only_plural_rank_b_4096_embeddings(self) -> None:
        valid = self._Embedding(batch=2, width=NV_EMBEDDING_DIMENSION)
        self.assertIs(_remote_sentence_embeddings({"sentence_embeddings": valid}, batch_size=2), valid)
        for output in (
            {"sentence_embedding": valid},
            {"sentence_embeddings": self._Embedding(batch=2, width=1024)},
            {"sentence_embeddings": self._Embedding(batch=1, width=NV_EMBEDDING_DIMENSION)},
            {"sentence_embeddings": self._Embedding(batch=2, width=NV_EMBEDDING_DIMENSION, rank=3)},
        ):
            with self.subTest(output=tuple(output)):
                with self.assertRaises(EncoderContractError):
                    _remote_sentence_embeddings(output, batch_size=2)

    def test_nv_offline_gate_requires_local_config_and_tokenizer_before_remote_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / REVISION
            root.mkdir()
            with self.assertRaises(EncoderContractError):
                enforce_nv_offline_runtime(root)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            old_hf, old_transformers = os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE")
            try:
                enforce_nv_offline_runtime(root)
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            finally:
                for name, old in (("HF_HUB_OFFLINE", old_hf), ("TRANSFORMERS_OFFLINE", old_transformers)):
                    if old is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = old

    def test_lora_rejects_unreviewed_latent_target(self) -> None:
        with self.assertRaises(EncoderContractError):
            EncoderModelSpec.from_mapping(qwen_spec(lora_target_modules=["q_proj", "latent_attention"]))


class RunnerStaticSafetyTest(unittest.TestCase):
    @staticmethod
    def _runner_module():
        spec = importlib.util.spec_from_file_location("train_encoder_static", ROOT / "scripts" / "train_encoder.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_run_root_and_canonical_dataset_admission_reject_escapes(self) -> None:
        runner = self._runner_module()
        with self.assertRaises(runner.TrainingContractError):
            runner._resolve_run_dir("/tmp/not-a-run", "safe-run")
        with self.assertRaises(runner.TrainingContractError):
            runner._require_prior_run_dir("/tmp/not-a-run")
        # Check the immutable checksum gate before any data path is resolved/read.
        with self.assertRaises(runner.TrainingContractError):
            runner._require_canonical_dataset("/tmp/look-alike.jsonl", "0" * 64, "train")
        with self.assertRaises(runner.TrainingContractError):
            runner._require_canonical_dataset("eval/train.jsonl", runner.CANONICAL_TRAIN_SHA256, "train")

    def test_single_accelerate_shard_has_rank_coverage_and_post_prepare_updates(self) -> None:
        runner = self._runner_module()
        coverage = runner._rank_batch_coverage(record_count=23, batch_size=2, world_size=3)
        self.assertEqual(sorted(index for rank in coverage for index in rank), list(range(23)))
        for left, first in enumerate(coverage):
            for second in coverage[left + 1:]:
                self.assertFalse(set(first) & set(second))
        self.assertEqual(runner._prepared_update_count(prepared_batches=4, accumulation_steps=2), 2)
        self.assertEqual(runner._prepared_update_count(prepared_batches=5, accumulation_steps=2), 3)

    def test_runner_does_not_use_device_map_auto_or_raw_wandb_tables(self) -> None:
        source = (ROOT / "scripts" / "train_encoder.py").read_text(encoding="utf-8")
        self.assertNotIn("device_map", source)
        self.assertNotIn("wandb.Table", source)
        self.assertIn("aggregate", source)
        self.assertNotIn("from torch.utils.data.distributed", source)
        self.assertIn("_prepared_update_count(len(train_loader)", source)
        self.assertIn("_require_canonical_dataset", source)
        self.assertIn("_require_prior_run_artifact", source)
        self.assertIn("selection_metadata", source)
        self.assertIn("refit_adapter", source)
        modeling = (ROOT / "src" / "mal2026" / "encoder_modeling.py").read_text(encoding="utf-8")
        self.assertIn("pool_mask=attention_mask", modeling)
        self.assertIn("sentence_embeddings", modeling)
        self.assertNotIn('outputs.get("sentence_embedding")', modeling)


if __name__ == "__main__":
    unittest.main()
