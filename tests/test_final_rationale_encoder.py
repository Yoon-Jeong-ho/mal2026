from __future__ import annotations

from pathlib import Path
import unittest

from mal2026.final_rationale_encoder import FinalEncoderConfig, select_candidate
from mal2026.solar_axis_augmentation import AXES


ROOT = Path(__file__).resolve().parents[1]


class FinalRationaleEncoderTests(unittest.TestCase):
    def test_config_compares_all_four_arms_and_never_uses_average(self) -> None:
        config = FinalEncoderConfig.from_json(
            ROOT / "configs/final_rationale_aware_score_encoder.v1.json",
            require_dependencies=False,
        )
        self.assertEqual(len(config.candidates), 4)
        self.assertEqual(config.score_fields, AXES)
        self.assertFalse(config.average_target_used)
        self.assertTrue(config.train_plus_validation)
        self.assertFalse(config.final_evaluation_performed)
        self.assertEqual(
            {(row["method"], row["model_key"]) for row in config.candidates},
            {
                ("original_bundle_rationale", "qwen3_embedding_8b"),
                ("original_bundle_rationale", "kure_v1"),
                ("original_plus_solar_augmented_bundle_rationale", "qwen3_embedding_8b"),
                ("original_plus_solar_augmented_bundle_rationale", "kure_v1"),
            },
        )

    def test_selection_uses_rmse_then_spearman_then_fixed_order(self) -> None:
        table = [
            {"order": 0, "macro_continuous_rmse": 0.55, "macro_continuous_spearman": 0.7},
            {"order": 1, "macro_continuous_rmse": 0.50, "macro_continuous_spearman": 0.5},
            {"order": 2, "macro_continuous_rmse": 0.50, "macro_continuous_spearman": 0.6},
            {"order": 3, "macro_continuous_rmse": 0.50, "macro_continuous_spearman": 0.6},
        ]
        self.assertEqual(select_candidate(table)["order"], 2)

    def test_final_runner_has_training_but_no_validation_prediction(self) -> None:
        source = (ROOT / "src/mal2026/final_rationale_encoder.py").read_text(encoding="utf-8")
        self.assertIn("validation_role\": \"training_data_after_final_method_selection", source)
        self.assertNotIn("predict_metrics(", source)
        self.assertIn('"final_evaluation_performed": False', source)


if __name__ == "__main__":
    unittest.main()
