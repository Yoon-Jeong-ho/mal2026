from __future__ import annotations

import json
import re
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deployment" / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from mal2026_submission.contracts import (  # noqa: E402
    SubmissionContractError,
    compact_participant_json,
    extract_prompt_essay,
    parse_participant_output,
)
from mal2026_submission.pipeline import Completion  # noqa: E402
from mal2026_submission import pipeline as pipeline_module  # noqa: E402
from mal2026_submission.production import (  # noqa: E402
    _blind_messages,
    _evaluation_contract,
    _rationale_schema,
    _score_input,
)
from mal2026_submission.server import create_app  # noqa: E402
from mal2026_submission.production_r0 import (  # noqa: E402
    _legacy_messages,
    _latest_score_blind_messages,
    _score_input as r0_score_input,
    _tokenize_score_input,
)
from mal2026.api_rationale_data import AXES as DATA_AXES, WritingRow, decoder_messages  # noqa: E402
from mal2026.evaluation_prompt_matrix import (  # noqa: E402
    RATIONALE_SCORE_BLIND,
    SCORE_RATIONALE_AWARE,
    rationale_messages,
    rubric_prefix,
    score_embedding_input,
)
from mal2026.rlaif_top3_encoder import _input_text as historical_r0_score_input  # noqa: E402
from mal2026.rationale_pipeline_prompts import rationale_messages as latest_rationale_messages  # noqa: E402


class FakePipeline:
    served_model_name = "mal2026-test"

    def complete(self, messages, **kwargs):
        fields = extract_prompt_essay(messages)
        if fields is None:
            return Completion(content="연결 확인", completion_tokens=2)
        output = {
            "content": {"score": 3, "rationale": "주장을 제시했으나 근거가 충분히 구체적이지 않다."},
            "organization": {"score": 4, "rationale": "도입과 결론이 구분되고 문단의 전개 순서가 비교적 자연스럽다."},
            "expression": {"score": 3, "rationale": "의미는 전달되지만 일부 표현이 반복되어 문장이 단조롭다."},
        }
        return Completion(content=compact_participant_json(output), prompt_tokens=10, completion_tokens=30)


class SubmissionContractTest(unittest.TestCase):
    @staticmethod
    def _announced_first_json(text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for index, character in enumerate(text[start:], start):
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        return None

    @classmethod
    def _announced_parse(cls, raw: str):
        text = re.sub(r"```(?:json)?", "", raw.strip()).replace("```", "").strip()
        json_text = cls._announced_first_json(text)
        if json_text is None:
            return None
        try:
            parsed = json.loads(json_text)
        except Exception:
            return None
        required = {"content", "organization", "expression"}
        if required.issubset(parsed) and all(
            isinstance(parsed[axis], dict) and "score" in parsed[axis]
            for axis in required
        ):
            return parsed
        return None

    def test_production_prompts_match_trained_exact_prompt_contract(self) -> None:
        rubric, user_template = _evaluation_contract(ROOT / "evaluation.txt")
        rationales = {
            "content": "내용 근거",
            "organization": "구성 근거",
            "expression": "표현 근거",
        }
        self.assertEqual(rubric, rubric_prefix())
        self.assertEqual(
            _blind_messages(rubric, user_template, "주제", "본문"),
            rationale_messages("주제", "본문", RATIONALE_SCORE_BLIND),
        )
        self.assertEqual(
            _score_input(rubric, user_template, "주제", "본문", rationales),
            score_embedding_input("주제", "본문", SCORE_RATIONALE_AWARE, rationales),
        )

    def test_r0_prompts_match_historical_training_contract(self) -> None:
        row = WritingRow(identifier="x", prompt="주제", essay="본문", scores=None)
        rationales = {
            "content": "내용 근거",
            "organization": "구성 근거",
            "expression": "표현 근거",
        }
        self.assertEqual(_legacy_messages("주제", "본문"), decoder_messages(row, DATA_AXES))
        self.assertEqual(
            r0_score_input("주제", "본문", rationales),
            historical_r0_score_input("주제", "본문", rationales),
        )

    def test_latest_rationale_prompt_matches_frozen_training_contract(self) -> None:
        prompt_source = (ROOT / "Rationale_evaluation_training.txt").read_text(encoding="utf-8")
        prompt_text = '주제에 "인용"과 {중괄호}가 있음'
        essay_text = "본문에는\n여러 줄과 [essay_text] 표기가 있음"
        observed = _latest_score_blind_messages(prompt_source, prompt_text, essay_text)
        self.assertEqual(observed, latest_rationale_messages(prompt_text, essay_text))
        rendered = "\n".join(message["content"] for message in observed)
        self.assertNotIn("reference_scores_integer", rendered)
        self.assertNotIn("predicted_score", rendered)

    def test_latest_rationale_manifest_routes_to_r0_pipeline(self) -> None:
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "manifest.json").write_text(json.dumps({
                "pipeline_kind": "legacy_r0_prediction_ensemble_to_latest_score_blind_rationale",
            }), encoding="utf-8")
            with (
                patch.dict("os.environ", {"MAL2026_BUNDLE_ROOT": directory, "MAL2026_BACKEND": "production"}),
                patch(
                    "mal2026_submission.production_r0.R0EnsemblePipeline.from_environment",
                    return_value=sentinel,
                ) as factory,
            ):
                self.assertIs(pipeline_module.load_pipeline(), sentinel)
                factory.assert_called_once_with()

    def test_r0_score_tokenization_matches_frozen_truncation_contract(self) -> None:
        observed = {}

        def tokenizer(text, **kwargs):
            observed["text"] = text
            observed.update(kwargs)
            return {"input_ids": "encoded"}

        result = _tokenize_score_input(tokenizer, "긴 입력", 2048)
        self.assertEqual(result, {"input_ids": "encoded"})
        self.assertEqual(observed, {
            "text": "긴 입력",
            "return_tensors": "pt",
            "add_special_tokens": True,
            "truncation": True,
            "max_length": 2048,
        })

    def test_rationale_schema_is_strict_three_axis_json(self) -> None:
        schema = _rationale_schema(1, 384)
        self.assertEqual(schema["required"], ["content", "organization", "expression"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["content"]["properties"]["rationale"]["maxLength"], 384)

    def test_extracts_latest_official_fields(self) -> None:
        messages = [
            {"role": "system", "content": "rubric"},
            {"role": "user", "content": "[prompt_text]\n주제\n\n[essay_text]\n본문"},
        ]
        self.assertEqual(extract_prompt_essay(messages), ("주제", "본문"))

    def test_extracts_announced_single_user_role_payload(self) -> None:
        messages = [{
            "role": "user",
            "content": (
                "[채점 지시사항 전문]\nJSON 객체 하나만 출력하라.\n\n"
                "[prompt_text]\n주제\n\n[essay_text]\n본문"
            ),
        }]
        self.assertEqual(extract_prompt_essay(messages), ("주제", "본문"))

    def test_compact_output_survives_announced_naive_brace_parser(self) -> None:
        candidate = {
            "content": {"score": 3, "rationale": "학생 글의 단독 } 기호를 근거로 언급한다."},
            "organization": {"score": 4, "rationale": "문단에 { 표시가 있어도 구조는 유지된다."},
            "expression": {"score": 2, "rationale": "표현 근거가 구체적이다."},
        }
        compact = compact_participant_json(candidate)
        self.assertTrue(compact.startswith("{"))
        self.assertNotIn("```", compact)
        parsed = self._announced_parse(compact)
        self.assertIsNotNone(parsed)
        self.assertEqual(parse_participant_output(parsed), json.loads(compact))
        for axis in ("content", "organization", "expression"):
            self.assertNotIn("{", parsed[axis]["rationale"])
            self.assertNotIn("}", parsed[axis]["rationale"])

    def test_generic_message_is_not_task_input(self) -> None:
        self.assertIsNone(extract_prompt_essay([{"role": "user", "content": "안녕하세요"}]))

    def test_blank_task_field_fails_closed(self) -> None:
        with self.assertRaises(SubmissionContractError):
            extract_prompt_essay([{"role": "user", "content": "[prompt_text]\n\n[essay_text]\n본문"}])


class SubmissionServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client_context = TestClient(create_app(pipeline=FakePipeline()))
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def test_required_endpoints_and_openai_shape(self) -> None:
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        models = self.client.get("/v1/models")
        self.assertEqual(models.status_code, 200)
        self.assertEqual(models.json()["data"][0]["id"], "mal2026-test")
        response = self.client.post("/v1/chat/completions", json={
            "model": "mal2026-test",
            "messages": [{"role": "user", "content": "[prompt_text]\n주제\n\n[essay_text]\n본문"}],
            "max_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "stop": ["Q:", "User:"],
        })
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        content = json.loads(payload["choices"][0]["message"]["content"])
        self.assertEqual(set(content), {"content", "organization", "expression"})
        self.assertEqual(payload["model"], "mal2026-test")
        self.assertEqual(payload["usage"]["total_tokens"], 40)

    def test_announced_single_user_message_and_parser_contract(self) -> None:
        response = self.client.post("/v1/chat/completions", json={
            "model": "mal2026-test",
            "messages": [{
                "role": "user",
                "content": (
                    "[채점 지시사항 전문]\n세 영역을 채점하라.\n\n"
                    "[prompt_text]\n주제\n\n[essay_text]\n본문"
                ),
            }],
            "max_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        })
        self.assertEqual(response.status_code, 200, response.text)
        raw = response.json()["choices"][0]["message"]["content"]
        parsed = SubmissionContractTest._announced_parse(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(set(parsed), {"content", "organization", "expression"})

    def test_docker_guide_generic_chat_example_is_accepted(self) -> None:
        response = self.client.post("/v1/chat/completions", json={
            "model": "mal2026-test",
            "messages": [{"role": "user", "content": "안녕하세요. 한 줄로 자기소개해 주세요."}],
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "연결 확인")

    def test_unknown_model_is_rejected(self) -> None:
        response = self.client.post("/v1/chat/completions", json={
            "model": "wrong",
            "messages": [{"role": "user", "content": "hello"}],
        })
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
