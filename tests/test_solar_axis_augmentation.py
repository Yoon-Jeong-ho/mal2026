from __future__ import annotations

import json
import importlib.util
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

    def test_runner_uses_official_docker_without_implicit_pull(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "scripts/run_solar_axis_augmentation.py"
        spec = importlib.util.spec_from_file_location("solar_runner_contract", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        command = module.server_command(19420)
        self.assertEqual(command[:2], ["docker", "run"])
        self.assertNotIn("pull", command)
        self.assertEqual(command[command.index("--gpus") + 1], '"device=0,1,2,3"')
        self.assertIn("upstage/vllm-solar-open2:latest", command)
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "4")
        self.assertIn("--enable-expert-parallel", command)
        self.assertIn("--logits-processors", command)

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
