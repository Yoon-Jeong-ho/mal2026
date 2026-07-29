from __future__ import annotations

import json
from pathlib import Path
import unittest

from mal2026.solar_axis_augmentation import (
    AXES,
    SolarAxisAugmentationError,
    load_train_rows,
    parse_output,
    render_messages,
    requested_drop,
    task_count,
)


class SolarAxisAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_train_rows()

    def test_exactly_three_train_only_variants(self) -> None:
        self.assertEqual(task_count(self.rows), 6000)
        self.assertEqual(set(AXES), {"content", "organization", "expression"})

    def test_prompt_has_no_average_and_binds_target(self) -> None:
        messages = render_messages(self.rows[0], "content")
        self.assertNotIn('"average"', messages[1]["content"])
        self.assertIn('"target_axis":"content"', messages[1]["content"])

    def test_parser_enforces_target_drop_and_non_target_preservation(self) -> None:
        row = self.rows[0]
        axis = "content"
        baseline = dict(zip(AXES, row.score, strict=True))
        upper = max(1.0, baseline[axis] - requested_drop(row.identifier, axis))
        score = {
            "content": int(upper * 4) / 4,
            "organization": round(baseline["organization"] * 4) / 4,
            "expression": round(baseline["expression"] * 4) / 4,
        }
        content = json.dumps({"augmented_essay": row.essay + " 문장 표현을 추가로 바꾸었다.", "score": score}, ensure_ascii=False)
        parsed = parse_output(content, row, axis)
        self.assertLessEqual(parsed["score"][axis], upper)

    def test_parser_rejects_weak_degradation(self) -> None:
        row = self.rows[0]
        score = {axis: round(value * 4) / 4 for axis, value in zip(AXES, row.score, strict=True)}
        content = json.dumps({"augmented_essay": row.essay + " 다른 문장이다.", "score": score}, ensure_ascii=False)
        with self.assertRaises(SolarAxisAugmentationError):
            parse_output(content, row, "content")


if __name__ == "__main__":
    unittest.main()
