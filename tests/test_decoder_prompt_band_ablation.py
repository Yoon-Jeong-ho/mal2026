from __future__ import annotations

from pathlib import Path
import unittest

from mal2026.decoder_prompt_band_ablation import (
    AblationConfig,
    PROMPT_ARMS,
    _extra_metrics,
    messages_for,
    request_records,
    train_probe,
)
from mal2026.official_score_prompt import EVALUATION_PROMPT_SHA256


class DecoderPromptBandAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = AblationConfig.from_json(Path("configs/decoder_prompt_band_ablation.v1.json"))

    def test_locked_config_and_prompt_contract(self) -> None:
        self.assertEqual(("official_p0", "axis_band_p1"), self.config.prompt_arms)
        self.assertEqual(EVALUATION_PROMPT_SHA256, self.config.official_prompt_sha256)
        revised = Path(self.config.revised_prompt_path).read_text(encoding="utf-8")
        self.assertEqual(1, revised.count("{주제 지문}"))
        self.assertEqual(1, revised.count("{논증적 글 본문}"))
        self.assertNotIn("49.1%", revised)
        self.assertIn("3점을 기본값으로 삼지 마라", revised)

    def test_train_probe_is_deterministic_and_train_only(self) -> None:
        first = train_probe(self.config)
        second = train_probe(self.config)
        self.assertEqual(400, len(first))
        self.assertEqual([row.identifier for row in first], [row.identifier for row in second])
        self.assertEqual(400, len({row.identifier for row in first}))

    def test_request_population_is_zero_shot_and_label_free_in_messages(self) -> None:
        rows = request_records(self.config)
        self.assertEqual(1600, len(rows))
        self.assertEqual(1600, len({(row["source_id"], row["split"], row["arm"]) for row in rows}))
        self.assertEqual({"official_p0", "axis_band_p1"}, {row["arm"] for row in rows})
        for row in (rows[0], rows[399], rows[800], rows[-1]):
            self.assertEqual(["system", "user"], [message["role"] for message in row["messages"]])
            self.assertNotIn("gold_raw", row["messages"])
            self.assertNotIn("gold_integer", row["messages"])

    def test_prompt_arms_only_change_instruction(self) -> None:
        official = messages_for(self.config, PROMPT_ARMS[0], "주제", "본문")
        revised = messages_for(self.config, PROMPT_ARMS[1], "주제", "본문")
        self.assertNotEqual(official[0]["content"], revised[0]["content"])
        self.assertIn("주제", official[1]["content"])
        self.assertIn("본문", official[1]["content"])
        self.assertIn("주제", revised[1]["content"])
        self.assertIn("본문", revised[1]["content"])

    def test_extra_metrics_tail_and_calibration(self) -> None:
        rows = []
        for gold, pred in ((1, 2), (2, 2), (3, 3), (4, 4), (5, 4)):
            rows.append({
                "parse_valid": True,
                "gold_raw": {axis: float(gold) for axis in ("content", "organization", "expression")},
                "gold_integer": {axis: gold for axis in ("content", "organization", "expression")},
                "prediction": {axis: pred for axis in ("content", "organization", "expression")},
            })
        metrics = _extra_metrics(rows, total_count=5)
        content = metrics["by_axis"]["content"]
        self.assertEqual(0.0, content["tail_recall"]["1"]["exact_recall"])
        self.assertEqual(1.0, content["tail_recall"]["2"]["exact_recall"])
        self.assertEqual(0.0, content["tail_recall"]["5"]["exact_recall"])
        self.assertAlmostEqual(0.5, content["low_tail_1_2_exact_recall"])
        self.assertAlmostEqual(0.0, content["mean_bias"])


if __name__ == "__main__":
    unittest.main()
