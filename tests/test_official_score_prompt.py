from __future__ import annotations

import json
from pathlib import Path
import unittest

from mal2026.official_aihub_score_pretrain import PretrainConfig, IntegerScoreRow, render_input
from mal2026.official_decoder_aihub_pretrain import DecoderAIHubConfig
from mal2026.official_decoder_score import DecoderScoreConfig
from mal2026.official_score_prompt import (
    LEGACY_COMPACT,
    PUBLIC_SPEC_SCORE_ONLY,
    instruction,
    prompt_sha256,
    provenance,
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

    def test_new_configs_bind_the_public_score_prompt(self) -> None:
        embedding = PretrainConfig.from_json(
            ROOT / "configs/official_aihub_integer_score_pretrain.public_spec_score_prompt.v1.json",
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
        self.assertEqual(
            {embedding.score_prompt_kind, decoder_pretrain.score_prompt_kind, decoder_matrix.score_prompt_kind},
            {PUBLIC_SPEC_SCORE_ONLY},
        )
        self.assertEqual(json.loads(json.dumps(provenance(PUBLIC_SPEC_SCORE_ONLY))), provenance(PUBLIC_SPEC_SCORE_ONLY))


if __name__ == "__main__":
    unittest.main()
