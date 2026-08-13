from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from mal2026.decoder_fewshot_validation import (
    AXES,
    FewshotConfig,
    DecoderFewshotError,
    condition_metrics,
    parse_response,
    response_schema,
    rotate,
    rotation_assignments,
    round_half_up,
    select_shots,
)


class DecoderFewshotValidationTests(unittest.TestCase):
    def test_config_contract(self) -> None:
        config = FewshotConfig.from_json(Path("configs/decoder_fewshot_validation.v1.json"), require_models=True)
        self.assertEqual(5, len(config.models))
        self.assertEqual(("balanced5", "central5"), config.conditions)

    def test_half_up(self) -> None:
        self.assertEqual([1, 2, 3, 4, 5], [round_half_up(x) for x in (1.49, 1.5, 2.5, 3.5, 4.5)])

    def test_schema_and_parser(self) -> None:
        schema = response_schema()
        self.assertEqual(list(AXES), schema["required"])
        payload = {axis: {"score": index + 1, "rationale": "근거"} for index, axis in enumerate(AXES)}
        self.assertEqual(payload, parse_response(json.dumps(payload, ensure_ascii=False)))
        with self.assertRaises(DecoderFewshotError):
            parse_response("```json\n{}\n```")

    def test_rotation_is_balanced(self) -> None:
        ids = [f"id-{index}" for index in range(400)]
        assigned = rotation_assignments(ids, 2026073104)
        self.assertEqual({index: 80 for index in range(5)}, {index: list(assigned.values()).count(index) for index in range(5)})

    def test_rotate(self) -> None:
        self.assertEqual((2, 3, 4, 5, 1), rotate((1, 2, 3, 4, 5), 1))

    def test_real_train_only_shot_contract(self) -> None:
        selected = select_shots()
        self.assertEqual(5, len(selected["balanced5"]))
        for index in range(3):
            self.assertEqual([1, 2, 3, 4, 5], sorted(row.scores[index] for row in selected["balanced5"]))
            self.assertEqual([3, 3, 3, 4, 4], sorted(row.scores[index] for row in selected["central5"]))

    def test_metrics(self) -> None:
        rows = []
        for index in range(400):
            score = index % 5 + 1
            values = {axis: score for axis in AXES}
            rows.append({"parse_valid": True, "prediction": values, "gold_raw": values, "gold_integer": values})
        metrics = condition_metrics(rows)
        self.assertEqual(0.0, metrics["macro_raw_rmse"])
        self.assertEqual(1.0, metrics["macro_raw_spearman"])
        self.assertEqual(0.2, metrics["macro_score_3_rate"])

        partial = condition_metrics(rows[:-1], expected_count=399, total_count=400)
        self.assertEqual(399 / 400, partial["parse_success_rate"])


if __name__ == "__main__":
    unittest.main()
