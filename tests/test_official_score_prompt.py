from __future__ import annotations

import json
from pathlib import Path
import unittest

from mal2026.official_aihub_score_pretrain import PretrainConfig, IntegerScoreRow, render_input
from mal2026.official_decoder_aihub_pretrain import DecoderAIHubConfig
from mal2026.official_decoder_score import DecoderScoreConfig
from mal2026.official_score_matrix import MatrixConfig
from mal2026.official_score_prompt import (
    EVALUATION_PROMPT_SHA256,
    LEGACY_COMPACT,
    PUBLIC_SPEC_SCORE_ONLY,
    USER_SUPPLIED_EVALUATION,
    instruction,
    prompt_sha256,
    provenance,
    query_text,
    system_prompt,
)


ROOT = Path(__file__).resolve().parents[1]


class OfficialScorePromptTests(unittest.TestCase):
    def test_public_score_prompt_preserves_axes_rubric_and_integer_scale(self) -> None:
        prompt = system_prompt(PUBLIC_SPEC_SCORE_ONLY)
        for text in (
            "content", "주장", "근거", "논리적 연결",
            "organization", "서론", "문단 간 연결", "논리 전개",
            "expression", "어휘", "맞춤법", "문법", "주술 호응",
            "1 매우 미흡", "3 보통", "5 매우 우수",
        ):
            self.assertIn(text, prompt)
        self.assertIn('{"content":1,"organization":1,"expression":1}', prompt)
        self.assertIn("rationale과 average는 출력하지 마라", prompt)

    def test_public_and_legacy_prompts_have_distinct_hash_bound_lineages(self) -> None:
        self.assertNotEqual(prompt_sha256(LEGACY_COMPACT), prompt_sha256(PUBLIC_SPEC_SCORE_ONLY))
        public = provenance(PUBLIC_SPEC_SCORE_ONLY)
        self.assertEqual(public["score_prompt_sha256"], prompt_sha256(PUBLIC_SPEC_SCORE_ONLY))
        self.assertIn("not_verbatim_organizer_prompt", public["prompt_provenance"])

    def test_embedding_render_uses_public_rubric_without_target_leakage(self) -> None:
        text = render_input(IntegerScoreRow("과제", "학생 글", (1, 3, 5)), PUBLIC_SPEC_SCORE_ONLY)
        self.assertIn(instruction(PUBLIC_SPEC_SCORE_ONLY), text)
        self.assertIn("<student_essay>\n학생 글\n</student_essay>", text)
        self.assertNotIn("(1, 3, 5)", text)

    def test_user_evaluation_prompt_routes_system_and_user_sections_exactly(self) -> None:
        system = system_prompt(USER_SUPPLIED_EVALUATION)
        user = query_text("주제 지문 실제값", "학생 글 실제값", kind=USER_SUPPLIED_EVALUATION)
        self.assertTrue(system.startswith("[역할]"))
        self.assertNotIn("[시스템 프롬프트]", system)
        self.assertNotIn("[유저 프롬프트]", system)
        self.assertTrue(user.startswith("[prompt_text]"))
        self.assertIn("주제 지문 실제값", user)
        self.assertIn("학생 글 실제값", user)
        self.assertNotIn("{주제 지문}", user)
        self.assertNotIn("{논증적 글 본문}", user)
        self.assertEqual(prompt_sha256(USER_SUPPLIED_EVALUATION), EVALUATION_PROMPT_SHA256)

    def test_user_evaluation_rationale_arm_appends_only_rationale_input(self) -> None:
        rationales = {axis: f"{axis} 설명" for axis in ("content", "organization", "expression")}
        user = query_text("주제", "학생 글", rationales, USER_SUPPLIED_EVALUATION)
        self.assertIn("[evaluation_rationales]", user)
        self.assertIn("<content>content 설명</content>", user)
        self.assertNotIn("reference_score", user)
        self.assertNotIn("average", user)

    def test_new_configs_bind_the_public_score_prompt(self) -> None:
        embedding = PretrainConfig.from_json(
            ROOT / "configs/official_aihub_integer_score_pretrain.public_spec_score_prompt.v1.json",
            require_dependencies=False,
        )
        embedding_eval4 = PretrainConfig.from_json(
            ROOT / "configs/official_aihub_integer_score_pretrain.public_spec_score_prompt_eval4.v1.json",
            require_dependencies=False,
        )
        decoder_pretrain = DecoderAIHubConfig.from_json(
            ROOT / "configs/official_decoder_aihub_integer_score_pretrain.public_spec_score_prompt.v1.json",
            require_dependencies=False,
        )
        decoder_matrix = DecoderScoreConfig.from_json(
            ROOT / "configs/official_decoder_score_matrix.public_spec_score_prompt.v1.json",
            require_dependencies=False,
        )
        embedding_matrix = MatrixConfig.from_json(
            ROOT / "configs/official_score_matrix.public_spec_score_prompt.v1.json",
            require_dependencies=False,
        )
        self.assertEqual(
            {
                embedding.score_prompt_kind,
                embedding_eval4.score_prompt_kind,
                decoder_pretrain.score_prompt_kind,
                decoder_matrix.score_prompt_kind,
                embedding_matrix.score_prompt_kind,
            },
            {PUBLIC_SPEC_SCORE_ONLY},
        )
        self.assertEqual(
            (embedding_eval4.per_device_train_batch_size, embedding_eval4.per_device_eval_batch_size, embedding_eval4.gradient_accumulation_steps),
            (1, 4, 8),
        )
        self.assertEqual(decoder_pretrain.per_device_eval_batch_size, 4)
        self.assertEqual(decoder_matrix.per_device_eval_batch_size, 4)
        self.assertEqual(json.loads(json.dumps(provenance(PUBLIC_SPEC_SCORE_ONLY))), provenance(PUBLIC_SPEC_SCORE_ONLY))


if __name__ == "__main__":
    unittest.main()
