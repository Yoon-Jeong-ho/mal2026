from pathlib import Path
import json
import tempfile
import unittest

import torch

from mal2026.official_score_matrix import (
    AXES,
    MatrixConfig,
    ScoreRow,
    artifact_inventory_sha256,
    arm_names,
    decode_logits,
    deterministic_internal_split,
    file_sha256,
    load_score_rows,
    official_half_up,
    ordinal_targets,
    score_head_initial_sha256,
    score_metrics,
    select_bootstrap_candidate,
    select_epoch,
    write_integer_scores,
)


class OfficialScoreMatrixTests(unittest.TestCase):
    def test_config_dry_validation_and_eight_clean_arms(self) -> None:
        config = MatrixConfig.from_json(Path("configs/official_score_matrix.v1.json"), require_dependencies=False)
        self.assertEqual(config.score_fields, AXES)
        self.assertEqual(len(arm_names()), 8)
        self.assertEqual(len(set(arm_names())), 8)
        self.assertTrue(all("average" not in arm for arm in arm_names()))
        # Stage A public essay arms are runnable before either rationales or
        # the unrelated AI-Hub head artifact exists.
        config.validate_dependencies("bootstrap", head="bounded_regression", require_aihub=False)
        with self.assertRaisesRegex(Exception, "bounded_regression integer AI-Hub warmstate is unavailable"):
            config.validate_dependencies("bootstrap")

    def test_half_up_and_bounded_head(self) -> None:
        self.assertEqual([official_half_up(x) for x in (1.0, 1.49, 1.5, 4.5, 5.0)], [1, 1, 2, 5, 5])
        continuous, integer, violations = decode_logits(torch.zeros(2, 3), "bounded_regression")
        self.assertEqual(continuous.tolist(), [[3.0] * 3] * 2)
        self.assertEqual(integer.tolist(), [[3] * 3] * 2)
        self.assertFalse(bool(violations.any()))

    def test_ordinal_projection_targets_and_violation_metrics(self) -> None:
        labels = torch.tensor([[1, 3, 5]], dtype=torch.float32)
        self.assertEqual(ordinal_targets(labels).tolist(), [[[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]])
        logits = torch.tensor([[0.0, 2.0, -2.0, -3.0] * 3])
        continuous, integer, violations = decode_logits(logits, "ordinal_cumulative")
        self.assertEqual(tuple(violations.shape), (1, 3, 3))
        self.assertEqual(int(violations.sum()), 3)
        self.assertGreaterEqual(float(continuous.min()), 1)
        self.assertLessEqual(float(continuous.max()), 5)
        metrics = score_metrics([[3, 3, 3]], continuous.tolist(), integer.tolist(), violations.tolist())
        self.assertEqual(metrics["ordinal_monotonic_violation_count"], 3)
        self.assertIs(metrics["ordinal_monotonic_projection_applied"], True)

    def test_both_head_losses_backpropagate_on_cpu(self) -> None:
        labels = torch.tensor([[1, 3, 5], [2, 4, 2]], dtype=torch.float32)
        regression_logits = torch.zeros(2, 3, requires_grad=True)
        continuous, _, _ = decode_logits(regression_logits, "bounded_regression")
        regression_loss = torch.nn.functional.mse_loss(continuous, labels)
        regression_loss.backward()
        self.assertTrue(bool(torch.isfinite(regression_logits.grad).all()))
        ordinal_logits = torch.zeros(2, 12, requires_grad=True)
        ordinal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            ordinal_logits.reshape(-1, 3, 4), ordinal_targets(labels)
        )
        ordinal_loss.backward()
        self.assertTrue(bool(torch.isfinite(ordinal_logits.grad).all()))

    def test_integer_primary_selection_and_refit_head_hash(self) -> None:
        rows = [
            {"epoch": 1, "macro_integer_rmse": 0.7, "macro_integer_spearman": 0.9, "macro_continuous_rmse": 0.1},
            {"epoch": 2, "macro_integer_rmse": 0.6, "macro_integer_spearman": 0.1, "macro_continuous_rmse": 0.9},
            {"epoch": 3, "macro_integer_rmse": 0.6, "macro_integer_spearman": 0.2, "macro_continuous_rmse": 0.8},
            {"epoch": 4, "macro_integer_rmse": 0.6, "macro_integer_spearman": 0.2, "macro_continuous_rmse": 0.7},
            {"epoch": 5, "macro_integer_rmse": 0.6, "macro_integer_spearman": 0.2, "macro_continuous_rmse": 0.7},
        ]
        self.assertEqual(select_epoch(rows)["epoch"], 4)

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.score_head = torch.nn.Linear(5, 3)

        torch.manual_seed(2026072701)
        selection_model = TinyModel()
        selection_hash = score_head_initial_sha256(selection_model)
        _ = torch.rand(100)
        torch.manual_seed(2026072701)
        refit_model = TinyModel()
        self.assertEqual(selection_hash, score_head_initial_sha256(refit_model))
        self.assertEqual(len(score_head_initial_sha256(TinyModel().to(torch.bfloat16))), 64)

    def test_bootstrap_selection_uses_only_internal_metrics(self) -> None:
        candidates = [
            {"arm": "a", "macro_integer_rmse": 0.5, "macro_integer_spearman": 0.4, "macro_continuous_rmse": 0.5, "canonical_validation_rmse": 99.0},
            {"arm": "b", "macro_integer_rmse": 0.6, "macro_integer_spearman": 0.9, "macro_continuous_rmse": 0.1, "canonical_validation_rmse": 0.0},
        ]
        self.assertEqual(select_bootstrap_candidate(candidates)["arm"], "a")

    def test_restricted_integer_score_emission(self) -> None:
        rows = [ScoreRow("x", "d", "p", "prompt", "essay", (1, 2, 3))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            digest = write_integer_scores(path, rows, [[1, 3, 5]], "arm", "train")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(digest, file_sha256(path))
            self.assertEqual(payload["scores"], {"content": 1, "organization": 3, "expression": 5})
            self.assertNotIn("average", payload["scores"])

    def test_full_artifact_inventory_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "model.safetensors"
            state.write_bytes(b"synthetic-state")
            inventory = [{"path": state.name, "size": state.stat().st_size, "sha256": file_sha256(state)}]
            expected = __import__("hashlib").sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            self.assertEqual(artifact_inventory_sha256(root, inventory), expected)
            state.write_bytes(b"changed")
            with self.assertRaisesRegex(Exception, "inventory file differs"):
                artifact_inventory_sha256(root, inventory)

    def test_internal_split_is_exact_deterministic_and_group_safe(self) -> None:
        rows = [ScoreRow(str(i), str(i), str(i % 9), "p", "e", (1, 2, 3)) for i in range(2000)]
        train_a, dev_a, fingerprint_a = deterministic_internal_split(rows, 2026072701)
        _, dev_b, fingerprint_b = deterministic_internal_split(list(reversed(rows)), 2026072701)
        self.assertEqual((len(train_a), len(dev_a)), (1600, 400))
        self.assertEqual({row.identifier for row in dev_a}, {row.identifier for row in dev_b})
        self.assertEqual(fingerprint_a, fingerprint_b)
        self.assertFalse({(row.prompt_num, row.document_id) for row in train_a} & {(row.prompt_num, row.document_id) for row in dev_a})
        for prompt in {row.prompt_num for row in rows}:
            total = sum(row.prompt_num == prompt for row in rows)
            held_out = sum(row.prompt_num == prompt for row in dev_a)
            self.assertLessEqual(abs(held_out - total * 0.2), 1.0)

    def test_average_value_is_not_read_or_validated(self) -> None:
        row = {
            "id": "x", "document_id": "d", "prompt_num": "p", "prompt": "과제", "essay": "글",
            "score": {"content": 1.5, "organization": 2.5, "expression": 4.49, "average": {"forbidden": "sentinel"}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            loaded = load_score_rows(path, file_sha256(path), 1)
        self.assertEqual(loaded[0].labels, (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
