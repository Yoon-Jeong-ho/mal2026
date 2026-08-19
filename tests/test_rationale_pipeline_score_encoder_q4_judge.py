from __future__ import annotations

import unittest

from scripts.run_rationale_pipeline_score_encoder_q4_judge import build_participant_rows


class ScoreEncoderQ4JudgeTests(unittest.TestCase):
    def test_uses_emitted_integer_scores_with_student_rationales(self) -> None:
        predictions = [{
            "source_id": "v1",
            "continuous_prediction": {"content": 1.1, "organization": 4.9, "expression": 2.4},
            "emitted_integer_prediction": {"content": 1, "organization": 5, "expression": 2},
        }]
        rationales = [{"source_id": "v1", "rationales": {
            "content": "내용 근거가 부족하다.", "organization": "구조가 명확하다.", "expression": "표현이 반복된다.",
        }}]
        rows = build_participant_rows(predictions, rationales)
        participant = rows[0]["participant_output"]
        self.assertEqual({axis: participant[axis]["score"] for axis in participant}, {"content": 1, "organization": 5, "expression": 2})
        self.assertEqual(participant["content"]["rationale"], "내용 근거가 부족하다.")

    def test_rejects_missing_score_rationale_linkage(self) -> None:
        predictions = [{
            "source_id": "v1", "continuous_prediction": {"content": 3.0, "organization": 3.0, "expression": 3.0},
            "emitted_integer_prediction": {"content": 3, "organization": 3, "expression": 3},
        }]
        rationales = [{"source_id": "different", "rationales": {"content": "c", "organization": "o", "expression": "e"}}]
        with self.assertRaisesRegex(RuntimeError, "linkage"):
            build_participant_rows(predictions, rationales)


if __name__ == "__main__":
    unittest.main()
