from __future__ import annotations

import unittest

from mal2026.decoder import (
    ContractError,
    SCORE_KEYS,
    direct_target,
    parse_decoder_output,
    rationale_target,
    require_immutable_revision,
    validate_lora_targets,
    validate_rationale,
)
from mal2026.decoder_eval import aggregate_metrics, quadratic_weighted_kappa
from mal2026.decoder_train import build_sft_example, head_tail_truncate


SCORE = {"content": 1.0, "organization": 2.5, "expression": 3.1, "average": 4.0}
FALLBACK = {key: 3.0 for key in SCORE_KEYS}


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return "PREFIX:"

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


class DecoderContractTests(unittest.TestCase):
    def test_direct_target_is_number_json_and_parser_is_strict(self):
        target = direct_target(SCORE)
        self.assertEqual(target, '{"content":1.00,"organization":2.50,"expression":3.10,"average":4.00}')
        parsed = parse_decoder_output(target, "direct", FALLBACK)
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.scores["expression"], 3.1)
        self.assertFalse(parse_decoder_output("prefix " + target, "direct", FALLBACK).valid)
        self.assertFalse(parse_decoder_output('{"average":4.00,"content":1.00,"organization":2.50,"expression":3.10}', "direct", FALLBACK).valid)
        self.assertFalse(parse_decoder_output('{"content":1.000,"organization":2.50,"expression":3.10,"average":4.00}', "direct", FALLBACK).valid)
        self.assertEqual(parse_decoder_output("not json", "direct", FALLBACK).scores, FALLBACK)

    def test_rationale_offsets_and_extra_or_score_like_generated_fields_are_rejected(self):
        essay = "첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다."
        rationale = [
            {"criterion": "CONTENT", "quote": "첫 문장", "start": 0, "end": 4, "observation": "글의 주장을 제시한다"},
            {"criterion": "ORGANIZATION", "quote": "둘째 문장", "start": 9, "end": 14, "observation": "앞 문장 뒤에 내용을 잇는다"},
            {"criterion": "EXPRESSION", "quote": "셋째 문장", "start": 19, "end": 24, "observation": "문장을 사용한다"},
        ]
        target = rationale_target(rationale, SCORE, essay)
        self.assertTrue(parse_decoder_output(target, "rationale", FALLBACK, essay).valid)
        malformed = [dict(item) for item in rationale]
        malformed[0]["explanation"] = "우수"
        with self.assertRaises(ContractError):
            validate_rationale(malformed, essay)
        mismatched = [dict(item) for item in rationale]
        mismatched[1]["end"] = 13
        with self.assertRaises(ContractError):
            validate_rationale(mismatched, essay)

    def test_assistant_only_mask_and_fixed_head_tail_truncation(self):
        record = {"id": "private-id-not-emitted", "prompt": "주제", "essay": "글", "score": SCORE}
        example = build_sft_example(FakeTokenizer(), record, "direct", None, 100)
        target_length = len(direct_target(SCORE))
        self.assertEqual(example["labels"][: len(example["labels"]) - target_length - 1], [-100] * (len(example["labels"]) - target_length - 1))
        ids, labels, dropped = head_tail_truncate(list(range(100)), [7, 8], 20, 99)
        self.assertEqual(dropped, 83)
        self.assertEqual(ids[:12], list(range(12)))
        self.assertEqual(ids[12:17], list(range(95, 100)))
        self.assertEqual(labels[:17], [-100] * 17)
        self.assertEqual(labels[-3:], [7, 8, 99])

    def test_lora_and_revision_fail_closed(self):
        names = [f"model.layers.0.{key}" for key in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")]
        expected = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        self.assertEqual(validate_lora_targets(names, expected), expected)
        with self.assertRaises(ContractError):
            validate_lora_targets(names[:-1], expected)
        with self.assertRaises(ContractError):
            require_immutable_revision("main")
        self.assertEqual(require_immutable_revision("a" * 40), "a" * 40)

    def test_metric_golden_cases(self):
        self.assertEqual(quadratic_weighted_kappa([1, 2, 3], [1, 2, 3]), 1.0)
        metrics = aggregate_metrics([[1.0, 1.0, 1.0, 1.0], [5.0, 5.0, 5.0, 5.0]], [[1.0, 1.0, 1.0, 1.0], [5.0, 5.0, 5.0, 5.0]], [True, True])
        self.assertEqual(metrics["primary/average_mae"], 0.0)
        self.assertEqual(metrics["decoder/parse_failure_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
