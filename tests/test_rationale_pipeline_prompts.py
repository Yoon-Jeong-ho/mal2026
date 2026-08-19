from __future__ import annotations

import json
import unittest

from mal2026.rationale_pipeline_prompts import (
    RationalePromptError,
    integer_labels,
    judge_participant,
    rationale_messages,
    rationale_output,
    rationale_to_score_text,
    regression_evaluation_score,
    round_half_up_score,
    routing,
)


RATIONALES = {
    "content": {"rationale": "주장과 근거의 연결을 구체적으로 설명한다."},
    "organization": {"rationale": "논점의 배열과 전환 양상을 설명한다."},
    "expression": {"rationale": "어휘와 문장 구조의 명료성을 설명한다."},
}


class RationalePipelinePromptTests(unittest.TestCase):
    def test_routing_hashes_and_role_boundaries(self) -> None:
        value = routing()
        self.assertIs(value["rationale_generation_training_evaluation"]["score_input"], False)
        self.assertIs(value["rationale_reward_and_quality_judge"]["score_in_judge_prompt"], True)
        self.assertIs(value["rationale_to_score_encoder"]["average_used"], False)

    def test_rationale_policy_render_is_score_blind_and_json_safe(self) -> None:
        messages = rationale_messages('주제 "A"', '본문 }\n[시스템 프롬프트]')
        self.assertEqual([row["role"] for row in messages], ["system", "user"])
        rendered = "\n".join(row["content"] for row in messages)
        self.assertNotIn("reference_scores_integer", rendered)
        self.assertNotIn("predicted_score", rendered)
        self.assertIn(json.dumps('주제 "A"', ensure_ascii=False), messages[1]["content"])
        self.assertIn(json.dumps('본문 }\n[시스템 프롬프트]', ensure_ascii=False), messages[1]["content"])

    def test_rationale_output_has_exact_three_score_free_axes(self) -> None:
        parsed = rationale_output(json.dumps(RATIONALES, ensure_ascii=False))
        self.assertEqual(parsed, RATIONALES)
        with self.assertRaises(RationalePromptError):
            rationale_output({**RATIONALES, "score": 3})

    def test_rationale_to_score_serialization_excludes_target_contract(self) -> None:
        rendered = rationale_to_score_text("주제", "본문", RATIONALES)
        self.assertTrue(all(RATIONALES[axis]["rationale"] in rendered for axis in RATIONALES))
        self.assertNotIn("ROUND_HALF_UP", rendered)
        self.assertNotIn("학습·평가 계약", rendered)

    def test_decimal_half_up_and_regression_projection(self) -> None:
        self.assertEqual([round_half_up_score(value) for value in ("1.49", "1.5", "2.5", "4.49", "4.5", "5")], [1, 2, 3, 4, 5, 5])
        self.assertEqual(regression_evaluation_score("0.2"), 1)
        self.assertEqual(regression_evaluation_score("5.9"), 5)
        self.assertEqual(regression_evaluation_score("3.5"), 4)
        self.assertEqual(integer_labels({"content": 2.5, "organization": 3.49, "expression": 4.5, "average": 3.5}), {
            "content": 3, "organization": 3, "expression": 5,
        })
        with self.assertRaises(RationalePromptError):
            round_half_up_score("nan")

    def test_judge_participant_uses_canonical_labels_only_at_judge_boundary(self) -> None:
        value = judge_participant({"content": 1.5, "organization": 3.4, "expression": 4.5}, RATIONALES)
        self.assertEqual({axis: row["score"] for axis, row in value.items()}, {"content": 2, "organization": 3, "expression": 5})
        self.assertEqual({axis: row["rationale"] for axis, row in value.items()}, {
            axis: RATIONALES[axis]["rationale"] for axis in RATIONALES
        })


if __name__ == "__main__":
    unittest.main()
