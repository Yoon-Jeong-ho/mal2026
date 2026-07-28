import json
import unittest

from mal2026.official_writing_contract import (
    AXES,
    JUDGE_DIMENSIONS,
    OfficialContractError,
    integerize_score,
    integerize_scores,
    judge_messages,
    parse_judge_output,
    parse_participant_output,
)
from scripts.evaluate_official_q4_judge import request_body


def _candidate():
    return {axis: {"score": index + 2, "rationale": f"{axis} 근거"} for index, axis in enumerate(AXES)}


class OfficialWritingContractTests(unittest.TestCase):
    def test_half_up_projection_is_deterministic(self):
        self.assertEqual(
            [integerize_score(value) for value in (-9, 1.49, 1.5, 2.5, 4.5, 99)],
            [1, 1, 2, 3, 5, 5],
        )
        self.assertEqual(
            integerize_scores([1.5, 2.5, 3.5]),
            {"content": 2, "organization": 3, "expression": 4},
        )

    def test_participant_parser_is_strict(self):
        self.assertEqual(parse_participant_output(_candidate())["content"]["score"], 2)
        invalid = _candidate()
        invalid["content"]["score"] = 2.0
        with self.assertRaises(OfficialContractError):
            parse_participant_output(invalid)

    def test_judge_prompt_contains_candidate_score_not_reference_score(self):
        messages = judge_messages("주제", "학생 글", _candidate())
        self.assertIn("candidate_predicted_score_and_rationale", messages[1]["content"])
        self.assertIn('"score":2', messages[1]["content"])
        self.assertNotIn("human_score", messages[1]["content"])
        self.assertNotIn("reference_score", messages[1]["content"])

    def test_custom_system_prompt_preserves_exact_text_and_candidate_scores(self):
        custom = "사용자 제공 judge system prompt\n"
        body = request_body("judge", "주제", "학생 글", _candidate(), system_prompt=custom)
        self.assertEqual(body["messages"][0]["content"], custom)
        user = body["messages"][1]["content"]
        marker = "[candidate_predicted_score_and_rationale]\n"
        self.assertIn(marker, user)
        candidate = json.loads(user.split(marker, 1)[1])
        self.assertEqual(candidate, _candidate())
        self.assertNotIn("reference_score", user)

    def test_judge_parser_requires_all_twelve_cells(self):
        output = {
            axis: {dimension: {"evidence": "판정 근거", "score": 3} for dimension in JUDGE_DIMENSIONS}
            for axis in AXES
        }
        self.assertEqual(len(parse_judge_output(output)), 3)
        del output["content"]["specificity"]
        with self.assertRaises(OfficialContractError):
            parse_judge_output(output)


if __name__ == "__main__":
    unittest.main()
