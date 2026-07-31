import json
import math
import tempfile
import unittest
from pathlib import Path

from mal2026.solar_prompt_search import (
    AXES,
    SearchConfig,
    _parse_spec,
    metrics,
    request_specs,
)


class SolarPromptSearchTest(unittest.TestCase):
    def test_config_contract(self):
        config = SearchConfig.from_json(Path("configs/solar_prompt_search.v1.json"))
        self.assertEqual(config.gpu_scope, (0, 1, 2, 3))
        self.assertEqual(config.target_raw_rmse, 0.4)

    def test_continuous_parsers(self):
        spec = {"axis": "content"}
        self.assertAlmostEqual(_parse_spec("axis_distribution_expected", spec, json.dumps({"probabilities": [0, 0, 0.25, 0.75, 0], "rationale": "x"}))["content"], 3.75)
        threshold = _parse_spec("axis_threshold_expected", spec, json.dumps({"at_least_2": 0.9, "at_least_3": 0.8, "at_least_4": 0.7, "at_least_5": 0.1, "rationale": "x"}))["content"]
        self.assertAlmostEqual(threshold, 3.5)
        self.assertAlmostEqual(_parse_spec("axis_position_continuous", spec, json.dumps({"position": 63, "rationale": "x"}))["content"], 3.52)

    def test_metrics(self):
        rows = []
        for prediction, gold in [(2.0, 2.5), (4.0, 3.5)]:
            rows.append({"parse_valid": True, "prediction": {axis: prediction for axis in AXES}, "gold_raw": {axis: gold for axis in AXES}})
        result = metrics(rows, 2)
        self.assertAlmostEqual(result["macro_raw_rmse"], 0.5)
        self.assertEqual(result["parse_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
