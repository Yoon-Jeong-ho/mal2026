import unittest
from pathlib import Path
from mal2026.solar_prompt_search_v3 import SearchConfigV3, _survival_expected, score_grid, train_splits

class SolarPromptSearchV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.config = SearchConfigV3.from_json(Path("configs/solar_prompt_search.v3.json"))

    def test_same_topic_grid_excludes_heldout(self):
        splits = train_splits(self.config)
        row = splits["discovery"][0]
        held = {item.identifier for values in splits.values() for item in values}
        grid = score_grid(self.config, row.identifier, "organization", 7)
        self.assertEqual(len(grid), 7)
        self.assertTrue(all(item.prompt == row.prompt and item.identifier not in held for item in grid))

    def test_survival_expected(self):
        score = _survival_expected([1, 2, 3, 4, 5], [1, 1, 0.5, 0, 0])
        self.assertTrue(2.5 <= score <= 3.5)

if __name__ == "__main__": unittest.main()
