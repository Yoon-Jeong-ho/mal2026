from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mal2026.iterative_official_rationale_embedding_data import (
    AXES,
    EMBEDDING_DIM,
    FEATURE_DIM,
    MODEL_ID,
    MODEL_REVISION,
    PROJECTION_DIM,
    PROJECTION_SEED,
    RUN_ID,
    SCHEMA_VERSION,
    OfficialRationaleEmbeddingError,
    build_rationale_features,
    file_sha256,
    load_feature_artifact,
    matrix_sha256,
    rademacher_projection,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/build_iterative_official_rationale_embeddings.py"
SPEC = importlib.util.spec_from_file_location("build_v12_rationale_embeddings", SCRIPT_PATH)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class OfficialRationaleEmbeddingTests(unittest.TestCase):
    def test_fixed_rademacher_projection_contract(self) -> None:
        first = rademacher_projection()
        second = rademacher_projection()
        self.assertEqual(first.shape, (EMBEDDING_DIM, PROJECTION_DIM))
        self.assertEqual(first.dtype, np.float32)
        self.assertFalse(first.flags.writeable)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(set(np.unique(first)), {-np.float32(1 / np.sqrt(PROJECTION_DIM)),
                                                np.float32(1 / np.sqrt(PROJECTION_DIM))})
        self.assertEqual(matrix_sha256(first), "8095c0e0cfcae32002b7e0d0f1c690336612cc7012ef18e0b6b13fa1c2dfd234")

    def test_feature_layout_is_axis_local_and_201_dimensional(self) -> None:
        values = np.zeros((2, 3, 3, EMBEDDING_DIM), dtype=np.float32)
        for source in range(2):
            for axis in range(3):
                for candidate in range(3):
                    values[source, axis, candidate, source * 100 + axis * 10 + candidate] = 1.0
        projection = np.arange(EMBEDDING_DIM * PROJECTION_DIM, dtype=np.float32).reshape(
            EMBEDDING_DIM, PROJECTION_DIM
        ) / 100_000.0
        result = build_rationale_features(values, projection)
        self.assertEqual(result.shape, (FEATURE_DIM,))
        for axis in range(3):
            block = result[axis * 67:(axis + 1) * 67]
            terra = values[0, axis].mean(0); terra /= np.linalg.norm(terra)
            luna = values[1, axis].mean(0); luna /= np.linalg.norm(luna)
            np.testing.assert_allclose(block[:32], ((terra + luna) * 0.5) @ projection)
            np.testing.assert_allclose(block[32:64], (terra - luna) @ projection)
            np.testing.assert_allclose(block[64:], [0.0, 0.0, 0.0])

    def test_renderer_returns_rationale_alone_in_registered_order(self) -> None:
        source_ids = ["essay-b", "essay-a"]
        grouped = {
            source: {
                source_id: {
                    candidate: {
                        axis: f" rationale:{source}:{source_id}:{candidate}:{axis} "
                        for axis in AXES
                    }
                    for candidate in (1, 2, 3)
                }
                for source_id in source_ids
            }
            for source in ("terra", "luna")
        }
        texts = BUILDER.ordered_rationale_texts(grouped, source_ids)
        self.assertEqual(len(texts), 36)
        self.assertEqual(texts[0], "rationale:terra:essay-b:1:content")
        self.assertEqual(texts[8], "rationale:terra:essay-b:3:expression")
        self.assertEqual(texts[18], "rationale:luna:essay-b:1:content")
        self.assertEqual(texts[-1], "rationale:luna:essay-a:3:expression")
        self.assertTrue(all(text.startswith("rationale:") for text in texts))
        for key in ("participant_score_included", "essay_included", "prompt_included", "gold_included"):
            self.assertFalse(BUILDER.RENDER_CONTRACT[key])

    def test_shards_are_fixed_contiguous_train_quarters(self) -> None:
        self.assertEqual([BUILDER.shard_bounds(2000, shard) for shard in range(4)], [
            (0, 500), (500, 1000), (1000, 1500), (1500, 2000)
        ])
        with self.assertRaises(BUILDER.BuildError):
            BUILDER.shard_bounds(1999, 0)

    def _write_artifact(self, tmp_path: Path) -> tuple[Path, Path, list[str]]:
        rows_path = tmp_path / "rows.jsonl"
        source_ids = [f"source-{index:04d}" for index in range(2000)]
        feature = [0.0] * FEATURE_DIM
        with rows_path.open("w", encoding="utf-8") as stream:
            for source_id in source_ids:
                stream.write(json.dumps({"source_id": source_id, "features": feature}) + "\n")
        manifest = {
            "schema_version": SCHEMA_VERSION, "status": "completed", "run_id": RUN_ID,
            "split_role": "train", "records": 2000, "feature_dim": FEATURE_DIM,
            "embedding_dim": EMBEDDING_DIM, "projection_dim": PROJECTION_DIM,
            "projection_seed": PROJECTION_SEED,
            "projection_matrix_sha256": matrix_sha256(rademacher_projection()),
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "validation_loaded": False, "candidate_score_in_embedding_text": False,
            "feature_rows_sha256": file_sha256(rows_path),
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, rows_path, source_ids

    def test_restricted_artifact_loader_binds_checksum_shape_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, rows_path, source_ids = self._write_artifact(Path(directory))
            manifest, rows = load_feature_artifact(
                manifest_path, rows_path, expected_source_ids=source_ids
            )
            self.assertEqual(manifest["records"], len(rows))
            self.assertEqual(len(rows), 2000)
            self.assertEqual(rows[0].source_id, source_ids[0])
            self.assertEqual(len(rows[0].features), FEATURE_DIM)
            with self.assertRaisesRegex(OfficialRationaleEmbeddingError, "source order"):
                load_feature_artifact(manifest_path, rows_path, expected_source_ids=list(reversed(source_ids)))
            with rows_path.open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(OfficialRationaleEmbeddingError, "checksum"):
                load_feature_artifact(manifest_path, rows_path)

    def test_last_nonpad_pooling_uses_attention_mask_and_float32(self) -> None:
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"torch unavailable: {exc}")
        output = torch.zeros((2, 4, 3), dtype=torch.bfloat16)
        output[0, 1] = torch.tensor([3.0, 4.0, 0.0])
        output[1, 3] = torch.tensor([0.0, 0.0, 2.0])
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        pooled = BUILDER._last_nonpad(output, mask)
        self.assertEqual(pooled.dtype, torch.float32)
        torch.testing.assert_close(pooled, torch.tensor([[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]))

    def test_cli_exposes_all_registered_actions(self) -> None:
        parser = BUILDER.parser()
        self.assertTrue(parser.parse_args(["--smoke"]).smoke)
        self.assertEqual(parser.parse_args(["--shard", "2"]).shard, 2)
        self.assertTrue(parser.parse_args(["--launch"]).launch)
        self.assertTrue(parser.parse_args(["--merge"]).merge)
        self.assertTrue(parser.parse_args(["--progress"]).progress)


if __name__ == "__main__":
    unittest.main()
