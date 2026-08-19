"""CPU-only contracts for the frozen residual embedding artifact builder."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[1]


def module():
    path = ROOT / "scripts" / "build_r0_residual_embeddings.py"
    spec = importlib.util.spec_from_file_location("build_r0_residual_embeddings", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def canonical_row(index: int, fold: int) -> dict:
    return {
        "id": f"source-{index}", "document_id": f"document-{index}",
        "prompt_num": "1", "prompt": f"prompt-{index}", "essay": f"essay-{index}",
        "score": {"content": 2.5, "organization": 3.25,
                  "expression": 4.0, "average": 3.25},
        "fold": fold,
    }


class FrozenResidualEmbeddingTests(unittest.TestCase):
    def test_exact_join_uses_document_group_and_never_average(self) -> None:
        m = module()
        sources = [m.SourceRow("a", "document-a", "p", "e",
                               {axis: float(i + 2) for i, axis in enumerate(m.AXES)})]
        rationale = {"a": {axis: f"r-{axis}" for axis in m.AXES}}
        base = [m.BaseRow("a", {axis: 3.0 for axis in m.AXES}, None)]
        joined = m.join_inputs("validation", sources, rationale, base)
        self.assertEqual("document-a", joined[0]["group_id"])
        self.assertNotIn("average", joined[0]["raw_continuous_gold"])

    def test_train_join_rejects_incomplete_fold_population(self) -> None:
        m = module()
        sources, rationales, bases = [], {}, []
        for fold in range(5):
            source_id = f"s{fold}"
            sources.append(m.SourceRow(source_id, f"d{fold}", "p", "e",
                                       {axis: 3.0 for axis in m.AXES}))
            rationales[source_id] = {axis: "r" for axis in m.AXES}
            bases.append(m.BaseRow(source_id, {axis: 3.0 for axis in m.AXES}, fold,
                                   {axis: 3.0 for axis in m.AXES}))
        rows = m.join_inputs("train", sources, rationales, bases)
        self.assertEqual(set(range(5)), {row["oof_fold"] for row in rows})
        self.assertTrue(all("average" not in row["raw_continuous_gold"] for row in rows))
        with self.assertRaisesRegex(m.FrozenEmbeddingArtifactError, "exactly match"):
            m.join_inputs("train", sources, {**rationales, "extra": rationales["s0"]}, bases)

    def test_canonical_parser_keeps_document_id_but_drops_average_target(self) -> None:
        m = module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            raw = canonical_row(0, 0)
            raw.pop("fold")
            path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            rows = m.load_canonical_rows("train", path=path, enforce_project_binding=False)
            self.assertEqual("document-0", rows[0].document_id)
            self.assertEqual(set(m.AXES), set(rows[0].raw_gold))
            self.assertNotIn("average", rows[0].raw_gold)

    def test_pooling_selects_true_last_nonpad_and_normalizes(self) -> None:
        import torch

        m = module()
        hidden = torch.tensor([
            [[3.0, 4.0], [5.0, 12.0], [99.0, 99.0]],
            [[99.0, 99.0], [8.0, 15.0], [7.0, 24.0]],
        ])
        mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
        pooled = m.last_nonpad_normalized(hidden, mask)
        self.assertTrue(torch.allclose(pooled[0], torch.tensor([5 / 13, 12 / 13])))
        self.assertTrue(torch.allclose(pooled[1], torch.tensor([7 / 25, 24 / 25])))
        self.assertTrue(torch.allclose(pooled.norm(dim=1), torch.ones(2)))

    def test_shard_bounds_cover_population_once(self) -> None:
        m = module()
        bounds = [m.shard_bounds(11, index) for index in range(4)]
        self.assertEqual([(0, 2), (2, 5), (5, 8), (8, 11)], bounds)
        covered = [item for start, stop in bounds for item in range(start, stop)]
        self.assertEqual(list(range(11)), covered)
        with self.assertRaisesRegex(m.FrozenEmbeddingArtifactError, "four shards"):
            m.shard_bounds(11, 0, 3)

    def test_cpu_merge_emits_exact_residual_manifest_and_round_trips(self) -> None:
        m = module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "restricted"
            run_id = "frozen-embedding-test"
            split_root = root / run_id / "train"
            sources = [
                m.SourceRow(f"s{i}", f"d{i}", "p", "e",
                            {axis: 3.0 for axis in m.AXES})
                for i in range(20)
            ]
            common = {
                "schema_version": "mal2026-r0-frozen-embedding-shard-v1",
                "status": "completed", "artifact_run_id": run_id,
                "split_role": "train", "shard_count": 4, "embedding_dim": 2,
                "embedding_model_id": m.MODEL_ID,
                "embedding_model_revision": m.MODEL_REVISION,
                "embedding_model_path": str(m.MODEL_PATH.resolve()),
                "embedding_model_config_sha256": "a" * 64,
                "embedding_source": "public", "embedding_frozen": True,
                "pooling": "last_nonpad_l2_normalized", "max_length": 2048,
                "canonical_source_sha256": "b" * 64,
                "rationale_rows_sha256": "c" * 64,
                "base_prediction_provenance": {
                    "result_path": "/restricted/result.json",
                    "result_sha256": "d" * 64, "prediction_sha256": "e" * 64,
                },
                "contains_average_target": False,
            }
            for shard in range(4):
                start, stop = m.shard_bounds(len(sources), shard)
                target = split_root / "shards" / f"shard-{shard:02d}"
                target.mkdir(parents=True)
                rows = [{
                    "source_id": sources[i].source_id,
                    "group_id": sources[i].document_id,
                    "shared_embedding": [0.6, 0.8],
                    "base_continuous_prediction": {axis: 3.0 for axis in m.AXES},
                    "raw_continuous_gold": {axis: 3.0 for axis in m.AXES},
                    "oof_fold": i % 5,
                } for i in range(start, stop)]
                rows_path = target / "rows.jsonl"
                rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                                     encoding="utf-8")
                metadata = {
                    **common, "shard_index": shard, "range_start": start,
                    "range_stop": stop, "records": stop - start,
                    "rows_sha256": m.file_sha256(rows_path),
                }
                (target / "metadata.json").write_text(json.dumps(metadata) + "\n",
                                                       encoding="utf-8")
            args = argparse.Namespace(
                artifact_run_id=run_id, split="train", shard_count=4,
            )
            with patch.object(m, "RESTRICTED_ROOT", root), patch.object(
                m, "load_canonical_rows", return_value=sources
            ):
                result = m.merge_shards(args)
            self.assertEqual(20, result["records"])
            manifest_path = split_root / "merged" / "manifest.json"
            rows_path = split_root / "merged" / "rows.jsonl"
            manifest, rows = m.load_embedding_artifact(manifest_path, rows_path)
            self.assertEqual(m.EMBEDDING_SCHEMA_VERSION, manifest.schema_version)
            self.assertEqual(5, manifest.fold_count)
            self.assertEqual(20, len(rows))
            self.assertTrue(all(row.group_id == f"d{i}" for i, row in enumerate(rows)))

    def test_source_declares_frozen_no_grad_and_exact_runtime_contract(self) -> None:
        source = (ROOT / "scripts" / "build_r0_residual_embeddings.py").read_text(encoding="utf-8")
        self.assertIn("torch.inference_mode()", source)
        self.assertIn("parameter.requires_grad_(False)", source)
        self.assertIn('model.eval()', source)
        self.assertIn("MAX_LENGTH = 2048", source)
        self.assertIn('"group_id": source.document_id', source)
        self.assertNotIn('"average": row', source)

    def test_cli_exposes_single_gpu_shard_and_cpu_merge(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/build_r0_residual_embeddings.py", "shard", "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        for flag in ("--split", "--shard-index", "--shard-count", "--physical-gpu"):
            self.assertIn(flag, completed.stdout)
        merged = subprocess.run(
            [sys.executable, "scripts/build_r0_residual_embeddings.py", "merge", "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(0, merged.returncode, merged.stderr)
        self.assertNotIn("--physical-gpu", merged.stdout)


if __name__ == "__main__":
    unittest.main()
