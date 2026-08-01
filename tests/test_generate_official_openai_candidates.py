import unittest

from scripts.generate_official_openai_candidates import (
    ALLOWED_MODELS,
    DIVERSITY,
    response_body,
)


class OfficialOpenAICandidateGenerationTest(unittest.TestCase):
    def test_allowed_models_and_exact_candidate_inventory(self):
        self.assertEqual(ALLOWED_MODELS, {"gpt-5.6-terra", "gpt-5.6-luna"})
        self.assertEqual(set(DIVERSITY), {1, 2, 3})

    def test_luna_request_remains_score_blind_and_strict(self):
        row = {"prompt": "쓰기 과제", "essay": "학생 글", "score": {"content": 1}}
        body = response_body(row, 2, "gpt-5.6-luna")
        self.assertEqual(body["model"], "gpt-5.6-luna")
        self.assertEqual(body["reasoning"], {"effort": "none"})
        self.assertFalse(body["store"])
        self.assertTrue(body["text"]["format"]["strict"])
        serialized = repr(body)
        self.assertNotIn("content': 1", serialized)
        self.assertNotIn("reference_score", serialized)


if __name__ == "__main__":
    unittest.main()
