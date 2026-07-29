from __future__ import annotations

from pathlib import Path
import unittest

from mal2026.evaluation_prompt_matrix import SCORE_DIRECT, SCORE_RATIONALE_AWARE
from mal2026.evaluation_prompt_score_encoder import EvaluationPromptScoreEncoderConfig


ROOT = Path(__file__).resolve().parents[1]


class EvaluationPromptScoreEncoderTests(unittest.TestCase):
    def test_direct_configs_are_strict_and_score_only(self) -> None:
        for name in ("qwen3_embedding_8b", "kure_v1"):
            config = EvaluationPromptScoreEncoderConfig.from_json(
                ROOT / f"configs/evaluation_prompt_score_encoder.{name}.direct.v1.json",
                require_dependencies=False,
            )
            self.assertEqual(config.input_kind, "direct")
            self.assertEqual(config.score_prompt_kind, SCORE_DIRECT)
            self.assertIsNone(config.rationale_variant)
            self.assertIsNone(config.rationale_key)
            self.assertFalse(config.average_target_used)

    def test_rationale_aware_configs_bind_variant_and_non_truncating_length(self) -> None:
        for name, expected_length in (("qwen3_embedding_8b", 2560), ("kure_v1", 2048)):
            for variant in ("score_blind", "score_conditioned"):
                config = EvaluationPromptScoreEncoderConfig.from_json(
                    ROOT / f"configs/evaluation_prompt_score_encoder.{name}.rationale_aware.{variant}.v1.json",
                    require_dependencies=False,
                )
                self.assertEqual(config.input_kind, "rationale_aware")
                self.assertEqual(config.rationale_variant, variant)
                self.assertEqual(config.score_prompt_kind, SCORE_RATIONALE_AWARE)
                self.assertEqual(config.max_length, expected_length)
                self.assertFalse(config.average_target_used)


if __name__ == "__main__":
    unittest.main()
