import unittest

from mal2026.official_aihub_rationale_data import (
    FEEDBACK_BY_AXIS,
    projected_rationales,
    projected_scores,
)
from mal2026.standard_decoder_data import FEEDBACK_FIELDS, RestrictedRow


def _row() -> RestrictedRow:
    feedback = {field: f"{field} 피드백" for field in FEEDBACK_FIELDS}
    return RestrictedRow(
        identifier="argumentative:test",
        prompt="과제",
        essay="학생 글",
        score={"content": 2.5, "organization": 3.49, "expression": 4.5, "average": 3.5},
        feedback=feedback,
    )


class OfficialAIHubRationaleDataTests(unittest.TestCase):
    def test_aihub_scores_are_projected_to_three_integers_only(self):
        self.assertEqual(
            projected_scores(_row()),
            {"content": 3, "organization": 3, "expression": 5},
        )

    def test_aihub_rationales_use_only_axis_analytic_feedback(self):
        result = projected_rationales(_row())
        self.assertEqual(set(result), set(FEEDBACK_BY_AXIS))
        self.assertNotIn("holistic", " ".join(result.values()))
        self.assertNotIn("task_1", " ".join(result.values()))
        for axis, fields in FEEDBACK_BY_AXIS.items():
            self.assertTrue(all(f"{field} 피드백" in result[axis] for field in fields))


if __name__ == "__main__":
    unittest.main()
