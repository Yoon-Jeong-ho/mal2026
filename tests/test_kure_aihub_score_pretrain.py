from __future__ import annotations

from pathlib import Path
import unittest

from mal2026.kure_aihub_score_pretrain import KUREAIHubConfig, select_epoch


ROOT = Path(__file__).resolve().parents[1]


class KUREAIHubScorePretrainTests(unittest.TestCase):
    def test_config_is_full_parameter_three_axis_without_average(self) -> None:
        config = KUREAIHubConfig.from_json(
            ROOT / "configs/official_kure_aihub_score_full_pretrain.v1.json",
            require_dependencies=False,
        )
        self.assertEqual(config.score_fields, ("content", "organization", "expression"))
        self.assertFalse(config.average_target_used)
        self.assertEqual(config.training_dtype, "float32")
        self.assertEqual(config.per_device_train_batch_size, 16)

    def test_selection_uses_continuous_rmse_before_integer_projection(self) -> None:
        events = [
            {"epoch": 1, "macro_continuous_rmse": 0.8, "macro_continuous_spearman": 0.5, "macro_integer_rmse": 0.7},
            {"epoch": 2, "macro_continuous_rmse": 0.7, "macro_continuous_spearman": 0.1, "macro_integer_rmse": 0.9},
        ]
        self.assertEqual(select_epoch(events)["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
