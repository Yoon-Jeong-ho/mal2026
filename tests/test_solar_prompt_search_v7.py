import unittest
from pathlib import Path

import numpy as np

from mal2026.solar_prompt_search_v7 import (
    ALPHAS,
    ATOMIC_DIMENSIONS,
    SearchConfigV7,
    _feature_schema,
    _fit_ridge,
    _messages,
    _predict_ridge,
    base,
    train_splits,
)


class SolarPromptSearchV7Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SearchConfigV7.from_json(Path("configs/solar_prompt_search.v7.json"))

    def test_split_base_and_atomic_schemas(self):
        splits = train_splits(self.config)
        self.assertEqual((len(splits["discovery"]), len(splits["confirmation"])), (160, 400))
        row = splits["discovery"][0]
        self.assertEqual(set(base(self.config, row.identifier)), {"content", "organization", "expression"})
        self.assertEqual(self.config.alphas, ALPHAS)
        for axis, dimensions in ATOMIC_DIMENSIONS.items():
            schema = _feature_schema(axis)
            self.assertEqual(list(schema["properties"]), [key for key, _ in dimensions])
            self.assertTrue(all(value == {"type": "integer", "minimum": 0, "maximum": 2} for value in schema["properties"].values()))

    def test_prompt_requests_atomic_levels_not_holistic_score(self):
        row = train_splits(self.config)["discovery"][0]
        for axis in ATOMIC_DIMENSIONS:
            text = "\n".join(message["content"] for message in _messages(row, axis))
            self.assertIn("0, 1, 2", text)
            self.assertIn("1~5 점수", text)
            self.assertIn("만들지 않는다", text)

    def test_ridge_standardizes_training_features_and_predicts_finite(self):
        x = np.asarray([[0, 0], [1, 10], [2, 20], [3, 30]], dtype=np.float64)
        residual = np.asarray([0.0, 0.5, 1.0, 1.5])
        model = _fit_ridge(x, residual, 1.0)
        prediction = _predict_ridge(model, x)
        self.assertEqual(prediction.shape, (4,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertAlmostEqual(float(prediction.mean()), float(residual.mean()))


if __name__ == "__main__":
    unittest.main()
