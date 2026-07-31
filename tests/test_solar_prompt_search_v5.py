import unittest
from pathlib import Path

from mal2026.solar_prompt_search_v5 import SearchConfigV5, base, evaluate, request_specs, train_splits


class SolarPromptSearchV5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SearchConfigV5.from_json(Path("configs/solar_prompt_search.v5.json"))

    def test_pairwise_order_swap_and_correction(self):
        row = train_splits(self.config)["discovery"][0]
        specs = request_specs(self.config, row)
        self.assertEqual(len(specs), 6)
        for axis in ("content", "organization", "expression"):
            pair = [spec for spec in specs if spec["axis"] == axis]
            self.assertEqual([spec["order"] for spec in pair], ["A", "B"])
            self.assertEqual(pair[0]["anchor_source_ids"], list(reversed(pair[1]["anchor_source_ids"])))
            prompt = pair[0]["messages"][-1]["content"]
            self.assertIn("base보다 낮은 기준 글", prompt)
            self.assertIn("base보다 높은 기준 글", prompt)
        base_prediction = base(self.config, row.identifier)
        synthetic = [{
            "parse_valid": True,
            "base_prediction": base_prediction,
            "gold_raw": {axis: min(5.0, base_prediction[axis] + 0.5) for axis in base_prediction},
            "final_verdict": {axis: "higher" for axis in base_prediction},
            "agreement": {axis: True for axis in base_prediction},
            "judgments": {axis: {"A": "higher", "B": "higher"} for axis in base_prediction},
        }]
        result = evaluate(synthetic, 0.5, 1)
        self.assertEqual(result["macro_consistency_rate"], 1.0)
        self.assertEqual(result["macro_direction_balanced_accuracy"], 1.0)
        self.assertEqual(result["per_axis"]["content"]["position_bias"]["signed_mean_a_minus_b"], 0.0)


if __name__ == "__main__":
    unittest.main()
