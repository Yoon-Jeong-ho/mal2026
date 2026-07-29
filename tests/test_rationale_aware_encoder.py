from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from mal2026.rationale_aware_encoder import (
    ContinuousScoreRow,
    RationaleEncoderConfig,
    load_continuous_rows,
    render_input,
    select_epoch,
    shuffled_rationales,
)


ROOT = Path(__file__).resolve().parents[1]


class RationaleAwareEncoderTests(unittest.TestCase):
    def test_qwen_config_binds_bundle_continuous_three_axis_contract(self) -> None:
        config = RationaleEncoderConfig.from_json(
            ROOT / "configs/rationale_aware_qwen3_embedding_8b_aihub_mal.v1.json",
            require_dependencies=False,
        )
        self.assertEqual(config.model_key, "qwen3_embedding_8b")
        self.assertEqual(config.score_fields, ("content", "organization", "expression"))
        self.assertFalse(config.average_target_used)
        self.assertEqual(config.target_projection, "none_preserve_raw_continuous")
        self.assertEqual(
            (config.per_device_train_batch_size, config.per_device_eval_batch_size, config.gradient_accumulation_steps),
            (4, 8, 2),
        )

    def test_kure_config_uses_numerically_stable_float32_mal_tuning(self) -> None:
        config = RationaleEncoderConfig.from_json(
            ROOT / "configs/rationale_aware_kure_v1_aihub_mal.v1.json",
            require_dependencies=False,
        )
        self.assertEqual(config.model_key, "kure_v1")
        self.assertEqual(config.training_dtype, "float32")
        self.assertEqual(config.score_fields, ("content", "organization", "expression"))
        self.assertFalse(config.average_target_used)

    def test_continuous_loader_preserves_fractional_axes(self) -> None:
        row = {
            "id": "x", "document_id": "d", "prompt_num": "p",
            "prompt": "주제", "essay": "학생 글",
            "score": {"content": 3.5, "organization": 4.25, "expression": 3.25, "average": 99},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            digest = sha256(path.read_bytes()).hexdigest()
            loaded = load_continuous_rows(path, digest, 1)
        self.assertEqual(loaded[0].labels, (3.5, 4.25, 3.25))

    def test_render_is_one_bundle_and_has_no_target(self) -> None:
        row = ContinuousScoreRow("x", "d", "p", "주제", "학생 글", (3.5, 4.25, 3.25))
        rationales = {axis: f"{axis} 설명" for axis in ("content", "organization", "expression")}
        text = render_input(row, rationales)
        self.assertIn('"evaluation_rationales"', text)
        self.assertNotIn('"score"', text)
        self.assertNotIn("4.25", text)

    def test_selection_is_continuous_rmse_primary(self) -> None:
        events = [
            {"epoch": 1, "macro_continuous_rmse": 0.60, "macro_continuous_spearman": 0.9, "macro_integer_rmse": 0.5},
            {"epoch": 2, "macro_continuous_rmse": 0.55, "macro_continuous_spearman": 0.1, "macro_integer_rmse": 0.8},
        ]
        self.assertEqual(select_epoch(events)["epoch"], 2)

    def test_selection_rejects_non_finite_metrics(self) -> None:
        events = [{
            "epoch": 1,
            "macro_continuous_rmse": float("nan"),
            "macro_continuous_spearman": 0.1,
            "macro_integer_rmse": 1.0,
        }]
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            select_epoch(events)

    def test_shuffled_bundle_never_retains_own_bundle(self) -> None:
        rows = [ContinuousScoreRow(str(i), str(i), "p", "q", "e", (3.0, 3.0, 3.0)) for i in range(4)]
        aligned = {row.identifier: {axis: f"{row.identifier}-{axis}" for axis in ("content", "organization", "expression")} for row in rows}
        shuffled = shuffled_rationales(rows, aligned, 7)
        for row in rows:
            self.assertIsNot(shuffled[row.identifier], aligned[row.identifier])


if __name__ == "__main__":
    unittest.main()
