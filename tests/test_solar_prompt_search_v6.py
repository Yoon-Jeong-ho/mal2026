import unittest
from pathlib import Path

from mal2026.solar_prompt_search_v6 import SearchConfigV6, _score_schema, base, train_splits


class SolarPromptSearchV6Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SearchConfigV6.from_json(Path("configs/solar_prompt_search.v6.json"))

    def test_split_base_and_schemas(self):
        splits = train_splits(self.config)
        self.assertEqual((len(splits["discovery"]), len(splits["confirmation"])), (160, 400))
        row = splits["discovery"][0]
        self.assertEqual(set(base(self.config, row.identifier)), {"content", "organization", "expression"})
        self.assertEqual(_score_schema("evidence_base_ternary")["properties"]["direction"]["enum"], ["lower", "same", "higher"])
        self.assertEqual(_score_schema("evidence_axis_integer")["properties"]["score"]["type"], "integer")


if __name__ == "__main__":
    unittest.main()
