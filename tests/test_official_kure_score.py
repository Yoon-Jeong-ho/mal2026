from __future__ import annotations

from pathlib import Path
import unittest

from mal2026.official_kure_score import KUREScoreConfig, render_input
from mal2026.official_score_matrix import ScoreRow


ROOT = Path(__file__).resolve().parents[1]


class OfficialKUREScoreTests(unittest.TestCase):
    def test_template_binds_exact_evaluation_prompt_and_kure_snapshot(self) -> None:
        config = KUREScoreConfig.from_json(
            ROOT / "configs/official_kure_score.evaluation_prompt.v1.json",
            require_rationales=False,
        )
        self.assertEqual(config.model_id, "nlpai-lab/KURE-v1")
        self.assertEqual(config.score_prompt_kind, "user_supplied_evaluation_txt_v1")
        self.assertEqual(config.selection_epochs, tuple(range(1, 13)))

    def test_rationale_is_input_without_score_or_average_leakage(self) -> None:
        row = ScoreRow("id", "doc", "1", "실제 주제", "실제 학생 글", (1, 3, 5))
        rationales = {axis: f"{axis} 설명" for axis in ("content", "organization", "expression")}
        text = render_input(row, "rationale", rationales)
        self.assertIn("실제 주제", text)
        self.assertIn("실제 학생 글", text)
        self.assertIn("[evaluation_rationales]", text)
        self.assertNotIn("(1, 3, 5)", text)
        self.assertNotIn("reference_score", text)
        self.assertNotIn("average_target", text)


if __name__ == "__main__":
    unittest.main()
