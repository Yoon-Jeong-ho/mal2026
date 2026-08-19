"""CPU-only tests for exact-R0 five-fold OOF aggregation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).parents[1]


def _module():
    path = ROOT / "scripts" / "aggregate_r0_exact_oof.py"
    spec = importlib.util.spec_from_file_location("aggregate_r0_exact_oof", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(module, root: Path, *, fingerprint_by_fold=None, duplicate=False):
    output = root / "outputs"
    restricted = root / "restricted"
    run_id = "r0-exact-oof-test"
    source_ids = [f"source-{index:04d}" for index in range(2000)]
    common_provenance = {
        "canonical_train_sha256": module.SOURCE_SHA256["train"],
        "rationale_generation_sha256": "a" * 64,
        "script_sha256": "b" * 64,
        "git_sha": "c" * 40,
        "warmstart": {"model_state_sha256": "d" * 64},
        "historical_r0_protocol": {"seed": 2026072601},
    }
    for fold in range(5):
        fold_output = output / run_id / f"fold-{fold:02d}"
        fold_private = restricted / run_id / f"fold-{fold:02d}"
        fold_output.mkdir(parents=True)
        fold_private.mkdir(parents=True)
        fingerprint = (
            fingerprint_by_fold[fold] if fingerprint_by_fold else "e" * 64
        )
        rows = []
        for offset in range(400):
            index = fold * 400 + offset
            identifier = source_ids[index]
            if duplicate and fold == 4 and offset == 399:
                identifier = source_ids[0]
            content = float(index % 5 + 1)
            organization = float((index // 5) % 5 + 1)
            expression = float((index // 25) % 5 + 1)
            continuous = {
                "content": content,
                "organization": organization,
                "expression": expression,
            }
            rows.append({
                "source_id": identifier,
                "fold": fold,
                "continuous_prediction": continuous,
                "half_up_integer_prediction": {
                    "content": int(content),
                    "organization": int(organization),
                    "expression": int(expression),
                },
                "reference_score": {
                    "content": content,
                    "organization": organization,
                    "expression": expression,
                },
            })
        prediction_path = fold_private / "oof_predictions.jsonl"
        prediction_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        aggregate = {
            "schema_version": "mal2026-r0-exact-oof-fold-aggregate-v1",
            "fold": fold, "folds": 5, "heldout_records": 400,
            "average_target_used": False, "validation_rows_loaded": False,
        }
        aggregate_path = fold_output / "aggregate_metrics.json"
        _write_json(aggregate_path, aggregate)
        checkpoints = []
        for epoch in range(1, 5):
            checkpoint = fold_output / "epoch_checkpoints" / f"epoch-{epoch:02d}"
            checkpoint.mkdir(parents=True)
            state_path = checkpoint / "trainable_model.safetensors"
            state_path.write_bytes(f"fold={fold};epoch={epoch}".encode())
            state_sha = module.file_sha256(state_path)
            metadata = {
                "schema_version": "mal2026-r0-exact-oof-checkpoint-v1",
                "run_id": run_id, "fold": fold, "epoch": epoch,
                "global_step": epoch * 25,
                "average_target_used": False,
                "source_train_sha256": module.SOURCE_SHA256["train"],
                "rationale_sha256": common_provenance["rationale_generation_sha256"],
                "fold_assignment_fingerprint": fingerprint,
                "trainable_state_sha256": state_sha,
            }
            metadata_path = checkpoint / "checkpoint_metadata.json"
            _write_json(metadata_path, metadata)
            checkpoints.append({
                "epoch": epoch,
                "global_step": epoch * 25,
                "trainable_state_path": str(state_path.resolve()),
                "trainable_state_sha256": state_sha,
                "checkpoint_metadata_sha256": module.file_sha256(metadata_path),
            })
        result = {
            "schema_version": "mal2026-r0-exact-oof-fold-result-v1",
            "status": "completed", "run_id": run_id, "fold": fold, "folds": 5,
            "score_fields": list(module.AXES), "average_target_used": False,
            "validation_rows_loaded": False, "validation_rows_directly_scored": False,
            "train_records": 1600, "heldout_records": 400,
            "fold_assignment_fingerprint": fingerprint,
            "leakage_gate": {
                "source_id_disjoint": True, "document_id_disjoint": True,
                "complete_train_2000_coverage": True,
            },
            "ensemble": {"epochs": [1, 2, 3, 4], "predictions_per_heldout_row": 4},
            "artifacts": {
                "restricted_oof_path": str(prediction_path.resolve()),
                "restricted_oof_sha256": module.file_sha256(prediction_path),
                "aggregate_path": str(aggregate_path.resolve()),
                "aggregate_sha256": module.file_sha256(aggregate_path),
            },
            "checkpoints": checkpoints,
            "provenance": common_provenance,
            "model_id": "Qwen/Qwen3-Embedding-8B", "model_revision": "revision",
            "rationale_source": "rank2_ax4_random1",
            "initialization": {"initialization": "aihub_48016_warmstart"},
        }
        _write_json(fold_output / "result.json", result)
    return run_id, output, restricted, source_ids


class AggregateR0ExactOOFTests(unittest.TestCase):
    def test_successful_merge_has_exact_2000_once_and_residual_oof_contract(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            run_id, output, restricted, ids = _fixture(module, Path(directory))
            rows, bindings, fingerprint = module.collect_verified_rows(
                run_id, ids, output_root=output, restricted_root=restricted
            )
            self.assertEqual(2000, len(rows))
            self.assertEqual(ids, [row["source_id"] for row in rows])
            self.assertEqual(5, len(bindings))
            self.assertEqual("e" * 64, fingerprint)
            contract = module.train_residual_contract()
            self.assertEqual("oof", contract["base_prediction_origin"])
            self.assertTrue(contract["base_model_fit_excludes_split"])
            self.assertFalse(contract["contains_average_target"])
            aggregate = module.public_aggregate(rows)
            self.assertEqual(2000, aggregate["unique_source_ids"])
            self.assertEqual(1, aggregate["predictions_per_source"])

    def test_duplicate_source_id_is_rejected(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            run_id, output, restricted, ids = _fixture(
                module, Path(directory), duplicate=True
            )
            with self.assertRaisesRegex(module.R0ExactOOFAggregateError, "2,000 unique"):
                module.collect_verified_rows(
                    run_id, ids, output_root=output, restricted_root=restricted
                )

    def test_mismatched_fold_fingerprint_is_rejected(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            fingerprints = ["e" * 64] * 4 + ["f" * 64]
            run_id, output, restricted, ids = _fixture(
                module, Path(directory), fingerprint_by_fold=fingerprints
            )
            with self.assertRaisesRegex(module.R0ExactOOFAggregateError, "fingerprints"):
                module.collect_verified_rows(
                    run_id, ids, output_root=output, restricted_root=restricted
                )

    def test_tampered_prediction_checksum_is_rejected(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            run_id, output, restricted, ids = _fixture(module, Path(directory))
            path = restricted / run_id / "fold-02" / "oof_predictions.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(module.R0ExactOOFAggregateError, "checksum"):
                module.collect_verified_rows(
                    run_id, ids, output_root=output, restricted_root=restricted
                )


if __name__ == "__main__":
    unittest.main()
