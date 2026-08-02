from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mal2026.kure_ordinal_oof import (
    BackboneSpec, KUREOrdinalOOFConfig, KUREOrdinalOOFError, _save_checkpoint_fresh,
    _load_private_fold, _secure_dir, _validate_outer_report, _validate_public_payload, _write_json_fresh,
    _write_jsonl_fresh, config_sha256, derived_seed, load_raw_axis_gold, load_recommended_methods,
    hybrid_loss, load_exact_r0, ordinal_loss, outer_split, run, validate_backbone_without_validation,
)
from mal2026.iterative_tail_metrics import compute_iterative_tail_metrics
from mal2026.kure_axis_contrastive import MODEL_CONFIG_SHA256, KUREAxisContrastiveError, token_length_audit
from mal2026.r0_ordinal_residual import ResidualRow
from mal2026.official_score_matrix import AXES, ScoreRow, file_sha256
from mal2026.ordinal_tail_fixed_feature import CandidateSpec


def config_mapping(aggregate: Path, aggregate_sha: str) -> dict:
    return {
        "schema_version": "mal2026-kure-ordinal-oof-v1", "run_id": "unit",
        "train_path": "eval/train.jsonl", "train_sha256": "a" * 64,
        "fold_manifest_path": "manifest.json", "fold_manifest_sha256": "f" * 64,
        "fold_rows_path": "rows.jsonl", "fold_rows_sha256": "b" * 64,
        "r0_oof_prediction_path": "r0.jsonl", "r0_oof_prediction_sha256": "9" * 64,
        "stage2_aggregate_path": str(aggregate), "stage2_aggregate_sha256": aggregate_sha,
        "stage2_config_sha256": "c" * 64,
        "backbone": {
            "arm": "aihub_full_backbone", "model_id": "nlpai-lab/KURE-v1",
            "model_revision": "d14c8a9423946e268a0c9952fecf3a7aabd73bd9", "model_path": "outputs/model",
            "model_config_sha256": "852d42e020c7f989c2acaf30fc683b7f768e8c6d1ab17166e835442162bd825d",
            "warmstart_completion_path": "outputs/warm/complete.json", "warmstart_completion_sha256": "d" * 64,
            "warmstart_artifact_path": "outputs/warm/model.safetensors", "warmstart_artifact_sha256": "e" * 64,
            "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        },
        "output_root": "outputs/public", "restricted_output_root": "data/processed/restricted/kure",
        "seed": 7, "epochs": 2, "crt_epochs": 1, "learning_rate": 5e-5, "weight_decay": 0.01,
        "batch_size": 4, "gradient_accumulation_steps": 2, "max_length": 1536,
        "raw_rmse_auxiliary_weight": 0.25,
        "axes": list(AXES), "average_target_forbidden": True,
    }


class KUREOrdinalOOFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.aggregate = Path(self.temp.name) / "aggregate.json"
        self.aggregate.write_text(json.dumps({
            "status": "completed", "config_sha256": "c" * 64,
            "phase2_recommended_distinct_families": [
                {"candidate_id": "rps-natural", "family": "rps"},
                {"candidate_id": "slace-a1", "family": "slace"},
            ],
        }))
        self.raw = config_mapping(self.aggregate, file_sha256(self.aggregate))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_config_forbids_validation_and_average(self) -> None:
        KUREOrdinalOOFConfig.from_mapping(self.raw)
        invalid = dict(self.raw, validation_path="eval/validation.jsonl")
        with self.assertRaisesRegex(KUREOrdinalOOFError, "validation"):
            KUREOrdinalOOFConfig.from_mapping(invalid)
        invalid = dict(self.raw, axes=[*AXES, "average"])
        with self.assertRaisesRegex(KUREOrdinalOOFError, "average"):
            KUREOrdinalOOFConfig.from_mapping(invalid)

    def test_stage2_binding_and_two_distinct_families(self) -> None:
        config = KUREOrdinalOOFConfig.from_mapping(self.raw)
        methods = load_recommended_methods(config)
        self.assertEqual([(x.identifier, x.family) for x in methods], [("rps-natural", "rps"), ("slace-a1", "slace")])
        bad = dict(self.raw, stage2_aggregate_sha256="0" * 64)
        with self.assertRaisesRegex(KUREOrdinalOOFError, "binding"):
            load_recommended_methods(KUREOrdinalOOFConfig.from_mapping(bad))

    def test_exact_outer_split_never_enters_fit(self) -> None:
        rows = [ScoreRow(str(i), f"d{i}", "p", "prompt", "essay", (3, 3, 3)) for i in range(2000)]
        folds = {row.identifier: index % 5 for index, row in enumerate(rows)}
        fit, held = outer_split(rows, folds, 3)
        self.assertEqual((len(fit), len(held)), (1600, 400))
        self.assertTrue({row.identifier for row in fit}.isdisjoint(row.identifier for row in held))

    def test_all_supported_ordinal_losses_are_finite(self) -> None:
        labels = torch.tensor([1, 2, 3, 4, 5] * 2)
        specs_logits = [
            (CandidateSpec("ce-natural", "softmax_ce", "natural"), torch.randn(10, 5)),
            (CandidateSpec("rps-natural", "rps", "natural"), torch.randn(10, 5)),
            (CandidateSpec("coral-natural", "coral", "natural"), torch.randn(10, 4)),
            (CandidateSpec("corn-natural", "corn", "natural"), torch.randn(10, 4)),
            (CandidateSpec("slace-a1", "slace", "slace_internal", alpha=1.0), torch.randn(10, 5)),
            (CandidateSpec("ce-effective-b0.99", "softmax_ce", "effective_number", beta=0.99), torch.randn(10, 5)),
        ]
        for spec, logits in specs_logits:
            self.assertTrue(bool(torch.isfinite(ordinal_loss(logits, labels, spec, labels))), spec.identifier)

    def test_fractional_raw_gold_changes_hybrid_loss(self) -> None:
        spec = CandidateSpec("ce-natural", "softmax_ce", "natural")
        logits = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0]])
        labels = torch.tensor([3])
        rounded = hybrid_loss(logits, labels, torch.tensor([3.0]), spec, labels)
        fractional = hybrid_loss(logits, labels, torch.tensor([3.25]), spec, labels)
        self.assertNotEqual(float(rounded), float(fractional))

    def test_axis_models_are_declared_independently(self) -> None:
        config = KUREOrdinalOOFConfig.from_mapping(self.raw)
        self.assertEqual(config.axes, AXES)
        self.assertNotIn("validation", json.dumps(self.raw).lower())

    def test_backbone_validation_never_touches_validation(self) -> None:
        root = Path(self.temp.name) / "model"; root.mkdir()
        (root / "config.json").write_text("config")
        completion = Path(self.temp.name) / "complete.json"; completion.write_text("done")
        artifact = Path(self.temp.name) / "model.safetensors"; artifact.write_text("weights")
        spec = BackboneSpec("aihub_full_backbone", "nlpai-lab/KURE-v1", "revision", str(root),
                            MODEL_CONFIG_SHA256, str(completion), "d" * 64, str(artifact), "e" * 64,
                            16, 32, 0.05)
        def digest(path):
            return MODEL_CONFIG_SHA256 if Path(path).name == "config.json" else "d" * 64 if Path(path) == completion else "e" * 64
        with patch("mal2026.kure_ordinal_oof.file_sha256", side_effect=digest) as mocked:
            validate_backbone_without_validation(spec)
        self.assertTrue(mocked.call_args_list)
        self.assertFalse(any("validation" in str(call).lower() for call in mocked.call_args_list))

    def test_derived_seeds_are_deterministic_and_separated(self) -> None:
        first = derived_seed(7, 2, "rps-natural", "content", "phase1")
        self.assertEqual(first, derived_seed(7, 2, "rps-natural", "content", "phase1"))
        self.assertEqual(len({first, derived_seed(7, 2, "rps-natural", "content", "crt"),
                              derived_seed(7, 2, "rps-natural", "organization", "phase1")}), 3)

    def test_restricted_writers_enforce_permissions(self) -> None:
        root = Path(self.temp.name) / "restricted"
        _secure_dir(root)
        manifest = root / "manifest.json"
        rows = root / "rows.jsonl"
        _write_json_fresh(manifest, {"safe": True}, private=True)
        _write_jsonl_fresh(rows, [{"source_id": "restricted"}])
        checkpoint = root / "state.safetensors"
        checkpoint_sha = _save_checkpoint_fresh(checkpoint, {"head.weight": torch.ones(1, 1)})
        self.assertEqual(root.stat().st_mode & 0o777, 0o700)
        self.assertEqual(manifest.stat().st_mode & 0o777, 0o600)
        self.assertEqual(rows.stat().st_mode & 0o777, 0o600)
        self.assertEqual(checkpoint.stat().st_mode & 0o777, 0o600)
        self.assertEqual(checkpoint_sha, file_sha256(checkpoint))

    def test_public_payload_rejects_restricted_content(self) -> None:
        _validate_public_payload({"metrics": {"rmse": 0.5}, "checkpoint_sha256": "a" * 64})
        with self.assertRaisesRegex(KUREOrdinalOOFError, "restricted content"):
            _validate_public_payload({"methods": [{"source_id": "secret"}]})

    def test_exact_r0_matches_fold_and_axis_values(self) -> None:
        r0 = Path(self.temp.name) / "r0.jsonl"
        r0.write_text(json.dumps({"source_id": "id", "fold": 2,
                                  "continuous_prediction": dict(zip(AXES, (2.0, 3.0, 4.0)))}) + "\n")
        raw = dict(self.raw, r0_oof_prediction_path=str(r0), r0_oof_prediction_sha256=file_sha256(r0))
        config = KUREOrdinalOOFConfig.from_mapping(raw)
        row = ResidualRow("id", "g", (0.0,), (2.0, 3.0, 4.0), (2.0, 3.0, 4.0), (2, 3, 4), 2)
        with patch("mal2026.kure_ordinal_oof.load_embedding_artifact", return_value=(object(), (row,))):
            self.assertEqual(load_exact_r0(config)["id"], (2.0, 3.0, 4.0))

    def test_smoke_is_gpu0_outer_fold_gate(self) -> None:
        config = KUREOrdinalOOFConfig.from_mapping(self.raw)
        with self.assertRaisesRegex(KUREOrdinalOOFError, "outer fold 0"):
            run(config, outer_fold=1, validate_only=True, smoke=True)

    def test_aggregate_rejects_smoke_stale_or_mixed_outer_report(self) -> None:
        config = KUREOrdinalOOFConfig.from_mapping(self.raw)
        methods = load_recommended_methods(config)
        axis_bindings = [{"axis": axis, "phase1_seed": index + 1, "crt_seed": index + 11,
                          "checkpoint_sha256": str(index) * 64, "lineage": {"pooling": "cls_l2"}}
                         for index, axis in enumerate(AXES, 1)]
        public = {
            "schema_version": "mal2026-kure-ordinal-oof-outer-v1", "status": "completed",
            "mode": "outer_fold", "run_id": config.run_id, "config_sha256": config_sha256(config),
            "outer_fold": 2, "records": 400, "stage2_aggregate_sha256": config.stage2_aggregate_sha256,
            "fold_manifest_sha256": config.fold_manifest_sha256, "fold_rows_sha256": config.fold_rows_sha256,
            "kure_model_revision": config.backbone.model_revision,
            "aihub_completion_sha256": config.backbone.warmstart_completion_sha256,
            "aihub_backbone_sha256": config.backbone.warmstart_artifact_sha256,
            "validation_rows_loaded": False, "average_target_used": False,
            "methods": [{"method": method.identifier, "family": method.family,
                         "restricted_prediction_sha256": "a" * 64, "axis_bindings": axis_bindings,
                         "objective": {"phase1": "family_ordinal_loss_plus_raw_expected_score_mse",
                                       "crt": "natural_ce_plus_raw_expected_score_mse",
                                       "raw_rmse_auxiliary_weight": 0.25}}
                        for method in methods],
        }
        _validate_outer_report(public, config, 2, methods)
        for key, bad_value in (("mode", "smoke"), ("run_id", "stale"), ("outer_fold", 3)):
            invalid = dict(public, **{key: bad_value})
            with self.assertRaisesRegex(KUREOrdinalOOFError, "outer report binding"):
                _validate_outer_report(invalid, config, 2, methods)
        mixed = dict(public, methods=list(reversed(public["methods"])))
        with self.assertRaisesRegex(KUREOrdinalOOFError, "method inventory"):
            _validate_outer_report(mixed, config, 2, methods)

    def test_private_fold_requires_exact_assigned_ids_schema_and_values(self) -> None:
        expected = {f"id-{index}" for index in range(400)}
        valid_path = Path(self.temp.name) / "private" / "valid.jsonl"
        valid_rows = [{"source_id": source_id, "outer_fold": 2,
                       "prediction": {axis: 3.25 for axis in AXES}} for source_id in sorted(expected)]
        _write_jsonl_fresh(valid_path, valid_rows)
        self.assertEqual(set(_load_private_fold(valid_path, 2, expected)), expected)

        swapped_path = Path(self.temp.name) / "private" / "swapped.jsonl"
        swapped = list(valid_rows)
        swapped[0] = {**swapped[0], "source_id": "id-from-another-fold"}
        _write_jsonl_fresh(swapped_path, swapped)
        with self.assertRaisesRegex(KUREOrdinalOOFError, "row/fold"):
            _load_private_fold(swapped_path, 2, expected)

        invalid_path = Path(self.temp.name) / "private" / "invalid.jsonl"
        invalid = list(valid_rows)
        invalid[0] = {**invalid[0], "prediction": {"content": 6.0, "organization": 3.0}}
        _write_jsonl_fresh(invalid_path, invalid)
        with self.assertRaisesRegex(KUREOrdinalOOFError, "prediction axes"):
            _load_private_fold(invalid_path, 2, expected)

    def test_token_audit_fails_before_truncation(self) -> None:
        rows = [ScoreRow("1", "d", "p", "prompt", "essay", (3, 3, 3))]
        class TooLong:
            def __call__(self, texts, **kwargs):
                return {"input_ids": [[1] * 1537 for _ in texts]}
        with self.assertRaisesRegex(KUREAxisContrastiveError, "truncated"):
            token_length_audit(rows, TooLong(), 1536)

    def test_raw_fractional_gold_is_preserved_and_average_not_read(self) -> None:
        path = Path(self.temp.name) / "train.jsonl"
        with path.open("w") as stream:
            for index in range(2000):
                score = {"content": 3.25 if index == 0 else 3.0,
                         "organization": 3.5 if index == 0 else 3.0,
                         "expression": 4.75 if index == 0 else 3.0,
                         "average": "not-a-numeric-input"}
                stream.write(json.dumps({"id": str(index), "document_id": str(index), "prompt_num": "p",
                                         "prompt": "p", "essay": "e", "score": score}) + "\n")
        raw = load_raw_axis_gold(path, file_sha256(path))
        self.assertEqual(raw["0"], (3.25, 3.5, 4.75))
        prediction = [[3.0, 3.0, 3.0]]
        raw_rmse = compute_iterative_tail_metrics([raw["0"]], prediction)["macro"]["rmse"]
        integer_rmse = compute_iterative_tail_metrics([[3.0, 4.0, 5.0]], prediction)["macro"]["rmse"]
        self.assertNotEqual(raw_rmse, integer_rmse)


if __name__ == "__main__":
    unittest.main()
