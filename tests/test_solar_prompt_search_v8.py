import unittest
from pathlib import Path

from mal2026.solar_prompt_search_v8 import SearchConfigV8, add_predictions, bracket_value, request_specs, select_anchors, train_splits


class SolarPromptSearchV8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SearchConfigV8.from_json(Path("configs/solar_prompt_search.v8.json"))

    def test_direct_pairs_are_swapped_and_hide_all_external_values(self):
        row = train_splits(self.config)["discovery"][0]
        specs = request_specs(self.config, row)
        self.assertEqual(len(specs), 12)
        for axis in ("content", "organization", "expression"):
            low, high = select_anchors(self.config, row.identifier, axis)
            self.assertLessEqual(float(low.scores[axis]), float(high.scores[axis]))
            for slot in ("low", "high"):
                pair = [spec for spec in specs if spec["axis"] == axis and spec["anchor_slot"] == slot]
                self.assertEqual([spec["order"] for spec in pair], ["target_first", "anchor_first"])
                forward = pair[0]["messages"][-1]["content"]
                reverse = pair[1]["messages"][-1]["content"]
                self.assertIn(row.essay, forward)
                self.assertIn(row.essay, reverse)
                for forbidden in ("OOF", "base", "낮은 기준", "높은 기준", str(pair[0]["anchor_score"])):
                    self.assertNotIn(forbidden, "\n".join(message["content"] for message in pair[0]["messages"]))

    def test_bracket_rules_and_blend_are_external(self):
        self.assertEqual(bracket_value(3.0, 2.5, 3.5, "higher", "lower"), 3.0)
        self.assertEqual(bracket_value(3.0, 2.5, 3.5, "lower", "lower"), 2.25)
        self.assertEqual(bracket_value(3.0, 2.5, 3.5, "higher", "higher"), 3.75)
        self.assertEqual(bracket_value(3.0, 2.5, 3.5, "tie", "lower"), 2.5)
        self.assertEqual(bracket_value(3.0, 2.5, 3.5, "unknown", "lower"), 3.0)
        row = {"base_prediction": {axis: 3.0 for axis in ("content", "organization", "expression")}, "anchor_scores": {axis: {"low": 2.0, "high": 4.0} for axis in ("content", "organization", "expression")}, "relations": {axis: {"low": "higher", "high": "higher"} for axis in ("content", "organization", "expression")}}
        predicted = add_predictions(row, 0.5)
        self.assertEqual(predicted["bracket_prediction"]["content"], 4.25)
        self.assertEqual(predicted["prediction"]["content"], 3.625)


if __name__ == "__main__": unittest.main()
