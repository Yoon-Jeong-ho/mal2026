from __future__ import annotations

from hashlib import sha256
import json
import unittest

from mal2026.evaluation_prompt_matrix import (
    EVALUATION_SHA256,
    JUDGE_SHA256,
    RATIONALE_SCORE_BLIND,
    RATIONALE_SCORE_CONDITIONED,
    SCORE_DIRECT,
    SCORE_RATIONALE_AWARE,
    EvaluationPromptMatrixError,
    evaluation_sections,
    judge_messages_exact,
    judge_system_prompt,
    prompt_provenance,
    rationale_messages,
    rationale_schema,
    rationale_system_prompt,
    score_embedding_input,
    score_system_prompt,
)


PROMPT = "학교에서 휴대전화 사용을 허용해야 하는가?"
ESSAY = "나는 허용해야 한다고 생각한다. 수업 자료를 확인할 수 있기 때문이다."
SCORES = {"content": 3, "organization": 2, "expression": 4}
RATIONALES = {
    "content": "휴대전화 사용 허용이라는 쟁점에 답하고 이유를 제시했다.",
    "organization": "주장 뒤에 이유가 이어지지만 결론이 분명하지 않다.",
    "expression": "두 문장이 자연스럽고 의미가 명확하다.",
}


class EvaluationPromptMatrixTests(unittest.TestCase):
    def test_canonical_files_are_hash_bound_and_sectioned(self) -> None:
        system, user = evaluation_sections()
        self.assertIn("[평가 기준 정의]", system)
        self.assertIn("[출력 규칙]", system)
        self.assertIn("{주제 지문}", user)
        self.assertIn("{논증적 글 본문}", user)
        self.assertEqual(EVALUATION_SHA256, "1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145")
        self.assertEqual(sha256(judge_system_prompt().encode()).hexdigest(), JUDGE_SHA256)

    def test_score_blind_rationale_contains_no_score_payload(self) -> None:
        messages = rationale_messages(PROMPT, ESSAY, RATIONALE_SCORE_BLIND)
        wire = json.dumps(messages, ensure_ascii=False)
        self.assertNotIn("predicted_score", wire)
        self.assertNotIn('"score"', wire)
        self.assertTrue(all(axis in messages[0]["content"] for axis in SCORES))
        self.assertTrue(all(axis in messages[0]["content"] for axis in RATIONALES))
        with self.assertRaises(EvaluationPromptMatrixError):
            rationale_messages(PROMPT, ESSAY, RATIONALE_SCORE_BLIND, SCORES)

    def test_score_conditioned_rationale_receives_only_emitted_scores(self) -> None:
        messages = rationale_messages(PROMPT, ESSAY, RATIONALE_SCORE_CONDITIONED, SCORES)
        self.assertIn("predicted_score", messages[1]["content"])
        self.assertIn('"content":3', messages[1]["content"])
        self.assertIn("바꾸거나 다시 채점하지", rationale_system_prompt(RATIONALE_SCORE_CONDITIONED))
        self.assertIn("점수 필드는 출력하지", messages[0]["content"])

    def test_direct_and_rationale_aware_score_inputs_are_separate(self) -> None:
        direct = score_embedding_input(PROMPT, ESSAY, SCORE_DIRECT)
        aware = score_embedding_input(PROMPT, ESSAY, SCORE_RATIONALE_AWARE, RATIONALES)
        self.assertNotIn("[evaluation_rationales]", direct)
        self.assertIn("[evaluation_rationales]", aware)
        self.assertTrue(all(text in aware for text in RATIONALES.values()))
        self.assertIn("average나 rationale을 출력하지", score_system_prompt(SCORE_DIRECT))
        self.assertIn("essay_text가 최종 근거", score_system_prompt(SCORE_RATIONALE_AWARE))
        with self.assertRaises(EvaluationPromptMatrixError):
            score_embedding_input(PROMPT, ESSAY, SCORE_DIRECT, RATIONALES)

    def test_exact_judge_gets_prompt_essay_and_full_candidate_only(self) -> None:
        candidate = {axis: {"score": SCORES[axis], "rationale": RATIONALES[axis]} for axis in SCORES}
        messages = judge_messages_exact(PROMPT, ESSAY, candidate)
        self.assertEqual(messages[0]["content"], judge_system_prompt())
        user = messages[1]["content"]
        self.assertIn(PROMPT, user)
        self.assertIn(ESSAY, user)
        self.assertIn("candidate_predicted_score_and_rationale", user)
        self.assertTrue(all(RATIONALES[axis] in user for axis in RATIONALES))
        self.assertNotIn("human_score", user)
        self.assertNotIn("reference_score", user)

    def test_prompt_provenance_binds_source_and_derivation(self) -> None:
        for kind in (RATIONALE_SCORE_BLIND, RATIONALE_SCORE_CONDITIONED, SCORE_DIRECT, SCORE_RATIONALE_AWARE):
            provenance = prompt_provenance(kind)
            self.assertEqual(provenance["evaluation_txt_sha256"], EVALUATION_SHA256)
            self.assertEqual(len(provenance["derived_system_prompt_sha256"]), 64)
            self.assertEqual(len(provenance["contract_sha256"]), 64)

    def test_rationale_schema_does_not_clip_retained_target_ceiling(self) -> None:
        schema = rationale_schema()
        for axis in SCORES:
            spec = schema["properties"][axis]["properties"]["rationale"]
            self.assertEqual((spec["minLength"], spec["maxLength"]), (60, 420))


if __name__ == "__main__":
    unittest.main()
