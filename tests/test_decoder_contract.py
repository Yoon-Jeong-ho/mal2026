from __future__ import annotations

import sys
import types
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
from mal2026.decoder_rationale_generate import TEACHER_TEMPLATE_SHA256, _parse_teacher_output, teacher_request
from mal2026.decoder_train import (
    _SelectionGenerationDataset,
    _selection_generation_collator,
    accelerator_batch_assignment,
    build_sft_example,
    head_tail_truncate,
    score_mean,
    updates_for_prepared_loader,
)
from mal2026.data_contract import DatasetRecord, ScoreVector
from mal2026.rationale import RationaleValidationError, validate_rationale_payload


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

    def test_selection_collator_keeps_local_essays_for_rationale_validation(self):
        tokenizer = FakeTokenizer()
        records = [{"id": "synthetic-a", "prompt": "주제", "essay": "인공 글", "score": SCORE}]
        dataset = _SelectionGenerationDataset(tokenizer, records, 100)

        class FakeTorch:
            long = "long"
            float32 = "float32"

            @staticmethod
            def tensor(value, dtype=None):
                return value

        with patch.dict(sys.modules, {"torch": FakeTorch}):
            batch = _selection_generation_collator(tokenizer)([dataset[0]])
        self.assertEqual(batch["essays"], ["인공 글"])
        self.assertEqual(batch["scores"], [[1.0, 2.5, 3.1, 4.0]])

    def test_output_root_rejects_symlink_escape(self):
        from mal2026.decoder import ContractError, resolve_run_output_dir

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            outside = root / "outside"
            outside.mkdir()
            (root / "outputs" / "runs").symlink_to(outside, target_is_directory=True)
            with patch("mal2026.decoder.project_root", return_value=root):
                with self.assertRaises(ContractError):
                    resolve_run_output_dir("safe-run", root / "outputs" / "runs" / "safe-run")

    def test_score_blind_teacher_request_and_duplicate_criterion_rejection(self):
        record = DatasetRecord(
            id="private-id", document_id="private-document", prompt_num="private-prompt-num",
            prompt="인공 과제", essay="가나다라마바사", scores=ScoreVector(3.0, 3.0, 3.0, 3.0),
        )
        request = teacher_request(record)
        flattened = "\n".join(message["content"] for message in request)
        self.assertNotIn(record.id, flattened)
        self.assertNotIn(record.document_id, flattened)
        self.assertNotIn(record.prompt_num, flattened)
        self.assertNotIn("3.0", flattened)
        self.assertEqual(len(TEACHER_TEMPLATE_SHA256), 64)
        payload = {"rationale": [
            {"criterion": "CONTENT", "quote": "가나", "start": 0, "end": 2, "observation": "주제를 다룬다"},
            {"criterion": "CONTENT", "quote": "다라", "start": 2, "end": 4, "observation": "내용을 잇는다"},
            {"criterion": "EXPRESSION", "quote": "마바", "start": 4, "end": 6, "observation": "표현을 쓴다"},
        ]}
        with self.assertRaises(RationaleValidationError):
            validate_rationale_payload(payload, record.essay)
        with self.assertRaises(RationaleValidationError):
            _parse_teacher_output(__import__("json").dumps(payload, ensure_ascii=False), record.essay)

    def test_one_accelerate_shard_rank_coverage_and_post_prepare_update_count(self):
        assignments = accelerator_batch_assignment(batch_count=11, world_size=3)
        self.assertEqual(assignments, ((0, 3, 6, 9), (1, 4, 7, 10), (2, 5, 8)))
        self.assertEqual(sorted(index for assignment in assignments for index in assignment), list(range(11)))
        self.assertEqual(updates_for_prepared_loader(4, 8), 1)
        self.assertEqual(updates_for_prepared_loader(17, 8), 3)
        with self.assertRaises(ContractError):
            updates_for_prepared_loader(0, 8)

    def test_loaded_chat_template_hash_is_checked(self):
        from mal2026.decoder import require_tokenizer_chat_template, tokenizer_chat_template_sha256

        tokenizer = FakeTokenizer()
        tokenizer.chat_template = "{{ messages }}"
        digest = tokenizer_chat_template_sha256(tokenizer)
        self.assertEqual(require_tokenizer_chat_template(tokenizer, digest), digest)
        with self.assertRaises(ContractError):
            require_tokenizer_chat_template(tokenizer, "0" * 64)

    def test_selection_fallback_mean_is_partition_only_not_refit_all_records(self):
        optimization_partition = [
            {"score": {"content": 1.0, "organization": 1.0, "expression": 1.0, "average": 1.0}},
            {"score": {"content": 3.0, "organization": 3.0, "expression": 3.0, "average": 3.0}},
        ]
        refit_all_records = optimization_partition + [
            {"score": {"content": 5.0, "organization": 5.0, "expression": 5.0, "average": 5.0}},
        ]
        selection_mean = score_mean(optimization_partition)
        self.assertEqual(selection_mean, {key: 2.0 for key in SCORE_KEYS})
        self.assertNotEqual(selection_mean, score_mean(refit_all_records))

    def test_deterministic_generation_sanitizes_sampling_only_values(self):
        from mal2026.decoder import sanitized_deterministic_generation_config

        config = types.SimpleNamespace(
            do_sample=True, temperature=0.2, top_p=0.8, min_p=0.2,
            typical_p=0.7, top_k=12, epsilon_cutoff=0.1, eta_cutoff=0.1,
        )
        sanitized = sanitized_deterministic_generation_config(config)
        self.assertFalse(sanitized.do_sample)
        self.assertEqual((sanitized.temperature, sanitized.top_p, sanitized.top_k), (1.0, 1.0, 50))
        self.assertEqual((sanitized.min_p, sanitized.typical_p, sanitized.epsilon_cutoff, sanitized.eta_cutoff), (None, 1.0, 0.0, 0.0))
        self.assertTrue(config.do_sample)  # copied; do not mutate a loaded model config

    def test_orderly_distributed_shutdown_barriers_then_destroys_only_if_initialized(self):
        from mal2026.decoder import orderly_distributed_shutdown

        calls: list[str] = []
        fake_dist = types.ModuleType("torch.distributed")
        fake_dist.is_available = lambda: True
        fake_dist.is_initialized = lambda: True
        fake_dist.barrier = lambda: calls.append("barrier")
        fake_dist.destroy_process_group = lambda: calls.append("destroy")
        fake_torch = types.ModuleType("torch")
        fake_torch.distributed = fake_dist
        with patch.dict(sys.modules, {"torch": fake_torch, "torch.distributed": fake_dist}):
            orderly_distributed_shutdown()
        self.assertEqual(calls, ["barrier", "destroy"])


if __name__ == "__main__":
    unittest.main()
