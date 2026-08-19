from dataclasses import replace
from pathlib import Path
import unittest

from scripts.train_rationale_pipeline_score_encoder import Config


ROOT = Path(__file__).resolve().parents[1]


class RationalePipelineScoreFixedFullTests(unittest.TestCase):
    def base_config(self) -> Config:
        return Config.load(
            ROOT
            / "configs/rationale-pipeline-score-encoder-1to2-qwen3-embedding-8b-bounded-regression-base-20260808-006.json"
        )

    def test_existing_configs_remain_select_then_refit(self) -> None:
        config = self.base_config()
        self.assertEqual(config.training_protocol, "select_then_refit")
        self.assertIsNone(config.fixed_epochs)
        self.assertIsNone(config.fixed_epoch_source)

    def test_fixed_full_requires_predeclared_epoch_and_source(self) -> None:
        config = replace(
            self.base_config(),
            run_id="fixed-full-test",
            training_protocol="fixed_full_train",
            fixed_epochs=7,
            fixed_epoch_source="frozen from a completed natural arm",
        )
        config.validate()

    def test_fixed_full_rejects_out_of_range_epoch(self) -> None:
        config = replace(
            self.base_config(),
            run_id="fixed-full-invalid-test",
            training_protocol="fixed_full_train",
            fixed_epochs=9,
            fixed_epoch_source="frozen evidence",
        )
        with self.assertRaisesRegex(RuntimeError, "fixed-full epoch differs"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
