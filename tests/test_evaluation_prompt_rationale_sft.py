from __future__ import annotations

from pathlib import Path
import unittest

from mal2026.evaluation_prompt_matrix import RATIONALE_SCORE_BLIND, RATIONALE_SCORE_CONDITIONED
from mal2026.evaluation_prompt_rationale_sft import EvaluationPromptRationaleSFTConfig, sft_examples


ROOT = Path(__file__).resolve().parents[1]


class EvaluationPromptRationaleSFTTests(unittest.TestCase):
    def test_configs_are_strict(self) -> None:
        for name in ("score_blind", "score_conditioned"):
            config = EvaluationPromptRationaleSFTConfig.from_json(ROOT / f"configs/evaluation_prompt_rationale_sft.{name}.v2.json")
            self.assertIn(config.prompt_kind, (RATIONALE_SCORE_BLIND, RATIONALE_SCORE_CONDITIONED))
            legacy = EvaluationPromptRationaleSFTConfig.from_json(ROOT / f"configs/evaluation_prompt_rationale_sft.{name}.v1.json")
            self.assertTrue(legacy.prompt_kind.endswith("_v1"))

    def test_score_blind_and_conditioned_examples_differ_only_by_score_input(self) -> None:
        blind = sft_examples(RATIONALE_SCORE_BLIND, 1)[0]
        conditioned = sft_examples(RATIONALE_SCORE_CONDITIONED, 1)[0]
        self.assertEqual(blind["completion"], conditioned["completion"])
        self.assertNotIn("predicted_score", blind["prompt"][1]["content"])
        self.assertIn("predicted_score", conditioned["prompt"][1]["content"])


if __name__ == "__main__":
    unittest.main()
