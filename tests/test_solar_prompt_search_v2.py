import json
import unittest
from pathlib import Path

from mal2026.solar_prompt_search_v2 import SearchConfigV2, nearest, request_specs, train_splits


class SolarPromptSearchV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SearchConfigV2.from_json(Path("configs/solar_prompt_search.v2.json"))

    def test_split_and_pool_retrieval(self):
        splits = train_splits(self.config)
        row = splits["discovery"][0]
        examples = nearest(self.config, row.identifier, 8)
        held = {item.identifier for values in splits.values() for item in values}
        self.assertEqual(len(examples), 8)
        self.assertTrue(all(example.identifier not in held for example in examples))

    def test_request_shapes(self):
        row = train_splits(self.config)["discovery"][0]
        self.assertEqual(len(request_specs(self.config, "retrieval8_joint_continuous", row)), 1)
        self.assertEqual(len(request_specs(self.config, "retrieval8_axis_continuous", row)), 3)


if __name__ == "__main__":
    unittest.main()
