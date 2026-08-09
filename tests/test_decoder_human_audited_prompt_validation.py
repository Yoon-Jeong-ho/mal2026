from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from mal2026.decoder_human_audited_prompt_validation import (
    DerivationFilter,
    filter_reason_audits,
    messages_for,
    paired_rmse_bootstrap,
    prompt_sections,
)


ROOT = Path(__file__).resolve().parents[1]


class HumanAuditedPromptValidationTest(unittest.TestCase):
    def test_filter_is_train_only_and_requires_score_and_audit_agreement(self) -> None:
        responses = [
            {
                "item_index": 0, "user_name": "가", "source_id": "train-a", "split": "train",
                "content_score": 2, "target_content_band": 1, "content_reason": "구체적 이유",
                "organization_score": 5, "target_organization_band": 1, "organization_reason": "차이가 큼",
                "expression_score": 2, "target_expression_band": 2, "expression_reason": "감사 불일치",
            },
            {
                "item_index": 1, "user_name": "가", "source_id": "validation-a", "split": "validation",
                "content_score": 1, "target_content_band": 1, "content_reason": "검증 누출 방지",
                "organization_score": 1, "target_organization_band": 1, "organization_reason": "",
                "expression_score": 1, "target_expression_band": 1, "expression_reason": "",
            },
        ]
        audit = {"score_reason_audits": [
            {"item_number": 1, "user_name": "가", "axis": "content", "label": "partial", "note": "일부 정합"},
            {"item_number": 1, "user_name": "가", "axis": "organization", "label": "match", "note": "점수 차이로 제외"},
            {"item_number": 1, "user_name": "가", "axis": "expression", "label": "mismatch", "note": "감사로 제외"},
            {"item_number": 2, "user_name": "가", "axis": "content", "label": "match", "note": "검증에서 제외"},
        ]}
        filt = DerivationFilter("train", 1, ("match", "partial"), 1, 1)
        selected, validation_excluded = filter_reason_audits(responses, audit, filt)
        self.assertEqual([(x["source_id"], x["axis"]) for x in selected], [("train-a", "content")])
        self.assertEqual([(x["source_id"], x["axis"]) for x in validation_excluded], [("validation-a", "content")])

    def test_tracked_prompts_route_and_human_arm_contains_no_example(self) -> None:
        p1 = ROOT / "configs/public_spec_score_band_prompt.v1.txt"
        p2 = ROOT / "configs/human_audited_score_prompt.v1.txt"
        for path in (p1, p2):
            system, user = prompt_sections(path)
            self.assertIn("content", system)
            self.assertEqual(user.count("{주제 지문}"), 1)
            self.assertEqual(user.count("{논증적 글 본문}"), 1)
        config = SimpleNamespace(public_band_prompt_path=str(p1), human_audit_prompt_path=str(p2))
        messages = messages_for(config, "human_audit_p2", "주제", "학생 글")
        self.assertEqual([x["role"] for x in messages], ["system", "user"])
        self.assertIn("주제", messages[1]["content"])
        self.assertIn("학생 글", messages[1]["content"])
        self.assertNotIn("{주제 지문}", messages[1]["content"])

    def test_paired_bootstrap_reports_right_minus_left(self) -> None:
        rows = []
        for index in range(8):
            gold = {axis: 3.0 for axis in ("content", "organization", "expression")}
            rows.extend((
                {"source_id": str(index), "arm": "left", "prediction": {axis: 1 for axis in gold}, "gold_raw": gold},
                {"source_id": str(index), "arm": "right", "prediction": {axis: 3 for axis in gold}, "gold_raw": gold},
            ))
        result = paired_rmse_bootstrap(rows, "left", "right", replicates=100, seed=7)
        self.assertAlmostEqual(result["delta_right_minus_left"], -2.0)
        self.assertEqual(result["interval_95"], [-2.0, -2.0])


if __name__ == "__main__":
    unittest.main()
