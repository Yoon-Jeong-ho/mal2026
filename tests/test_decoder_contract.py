from __future__ import annotations

import sys
import copy
import json
import types
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mal2026.decoder import (
    ContractError,
    HUMAN_FEEDBACK_KEYS,
    SCORE_KEYS,
    direct_target,
    human_feedback_target,
    parse_decoder_output,
    require_immutable_revision,
    validate_lora_targets,
)
from mal2026.decoder_eval import aggregate_metrics, quadratic_weighted_kappa
from mal2026.decoder_train import (
    _SelectionGenerationDataset,
    _load_restricted_rows,
    _manifest_path,
    _prepared_data_dir,
    _validate_prepared_manifest,
    accelerator_batch_assignment,
    build_sft_example,
    head_tail_truncate,
    score_mean,
    updates_for_prepared_loader,
)

SCORE = {"content": 1.0, "organization": 2.5, "expression": 3.1, "average": 4.0}
FALLBACK = {key: 3.0 for key in SCORE_KEYS}
FEEDBACK = {
    "holistic": "글의 핵심 주장이 드러난다.", "content_1": "내용이 명료하다.",
    "content_2": "근거를 구체적으로 제시한다.", "content_3": "주제와 관련된 내용을 쓴다.",
    "organization_1": "문단이 자연스럽게 연결된다.", "organization_2": "중심 내용이 유지된다.",
    "expression_1": "어휘와 문장이 적절하다.", "expression_2": "어법 오류가 드물다.",
    "task_1": "과제 요구를 충실히 수행한다.",
}


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
    def test_direct_target_and_parser_are_strict(self):
        target = direct_target(SCORE)
        self.assertEqual(target, '{"content":1.00,"organization":2.50,"expression":3.10,"average":4.00}')
        self.assertTrue(parse_decoder_output(target, "direct", FALLBACK).valid)
        self.assertFalse(parse_decoder_output("prefix " + target, "direct", FALLBACK).valid)
        self.assertFalse(parse_decoder_output('{"average":4.00,"content":1.00,"organization":2.50,"expression":3.10}', "direct", FALLBACK).valid)
        self.assertFalse(parse_decoder_output('{"content":1.000,"organization":2.50,"expression":3.10,"average":4.00}', "direct", FALLBACK).valid)
        self.assertEqual(parse_decoder_output("not json", "direct", FALLBACK).scores, FALLBACK)

    def test_human_feedback_target_order_and_parser_are_strict(self):
        target = human_feedback_target(FEEDBACK, SCORE)
        self.assertTrue(parse_decoder_output(target, "human_feedback", FALLBACK).valid)
        self.assertEqual(tuple(__import__("json").loads(target)["feedback"]), HUMAN_FEEDBACK_KEYS)
        reordered = '{"feedback":{"content_1":"a","holistic":"b","content_2":"c","content_3":"d","organization_1":"e","organization_2":"f","expression_1":"g","expression_2":"h","task_1":"i"},"scores":{"content":1.00,"organization":2.50,"expression":3.10,"average":4.00}}'
        self.assertFalse(parse_decoder_output(reordered, "human_feedback", FALLBACK).valid)
        missing = dict(FEEDBACK); missing["task_1"] = ""
        with self.assertRaises(ContractError):
            human_feedback_target(missing, SCORE)
        duplicate = target.replace('"holistic"', '"holistic":"x","holistic"', 1)
        self.assertFalse(parse_decoder_output(duplicate, "human_feedback", FALLBACK).valid)

    def test_assistant_only_mask_and_fixed_head_tail_truncation(self):
        record = {"id": "private-id-not-emitted", "prompt": "주제", "essay": "글", "score": SCORE, "feedback": FEEDBACK}
        example = build_sft_example(FakeTokenizer(), record, "direct", 100)
        target_length = len(direct_target(SCORE))
        self.assertEqual(example["labels"][: len(example["labels"]) - target_length - 1], [-100] * (len(example["labels"]) - target_length - 1))
        ids, labels, dropped = head_tail_truncate(list(range(100)), [7, 8], 20, 99)
        self.assertEqual(dropped, 83)
        self.assertEqual(ids[:12], list(range(12)))
        self.assertEqual(ids[12:17], list(range(95, 100)))
        self.assertEqual(labels[:17], [-100] * 17)
        self.assertEqual(labels[-3:], [7, 8, 99])

    def test_human_feedback_is_never_used_to_build_generation_input(self):
        tokenizer = FakeTokenizer()
        first = {"id": "a", "prompt": "주제", "essay": "인공 글", "score": SCORE, "feedback": FEEDBACK}
        changed = dict(first); changed["feedback"] = {key: "완전히 다른 사람 피드백" for key in HUMAN_FEEDBACK_KEYS}
        self.assertEqual(_SelectionGenerationDataset(tokenizer, [first], 100)[0]["input_ids"], _SelectionGenerationDataset(tokenizer, [changed], 100)[0]["input_ids"])

    def test_prepared_row_rejects_duplicate_order_blank_and_count_errors(self):
        row = '{"id":"a","prompt":"문제","essay":"글","score":{"content":1.00,"organization":2.50,"expression":3.10,"average":4.00},"feedback":' + __import__("json").dumps(FEEDBACK, ensure_ascii=False, separators=(",", ":")) + "}\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"; path.write_text(row, encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(len(_load_restricted_rows(path, digest, 1)), 1)
            path.write_text(row.replace('"id":"a",', '"id":"a","id":"b",'), encoding="utf-8")
            with self.assertRaises(ContractError): _load_restricted_rows(path, hashlib.sha256(path.read_bytes()).hexdigest(), 1)

    def test_decoder_prepared_paths_are_exactly_canonical(self):
        self.assertEqual(_prepared_data_dir("data/processed/aihub_human_feedback_v1").name, "aihub_human_feedback_v1")
        self.assertEqual(_manifest_path("data/manifests/aihub_human_feedback_v1.json").name, "aihub_human_feedback_v1.json")
        with self.assertRaises(ContractError):
            _prepared_data_dir("data/processed/another-direct-child")
        with self.assertRaises(ContractError):
            _manifest_path("data/manifests/another-direct-child.json")

    def test_manifest_requires_training_only_common_eligibility_and_frozen_split(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "data" / "manifests" / "aihub_human_feedback_v1.json").read_text(encoding="utf-8"))
        _validate_prepared_manifest(manifest)
        invalid_source = copy.deepcopy(manifest)
        invalid_source["source"]["included_split"] = "AI-Hub upstream Validation"
        with self.assertRaises(ContractError): _validate_prepared_manifest(invalid_source)
        invalid_gate = copy.deepcopy(manifest)
        invalid_gate["eligibility"]["common_to_all_four_experiments"] = False
        with self.assertRaises(ContractError): _validate_prepared_manifest(invalid_gate)
        invalid_split = copy.deepcopy(manifest)
        invalid_split["split"]["requested_dev_fraction"] = "0.10"
        with self.assertRaises(ContractError): _validate_prepared_manifest(invalid_split)

    def test_lora_and_revision_fail_closed(self):
        names = [f"model.layers.0.{key}" for key in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")]
        expected = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        self.assertEqual(validate_lora_targets(names, expected), expected)
        with self.assertRaises(ContractError): validate_lora_targets(names[:-1], expected)
        with self.assertRaises(ContractError): require_immutable_revision("main")
        self.assertEqual(require_immutable_revision("a" * 40), "a" * 40)

    def test_metric_golden_cases(self):
        self.assertEqual(quadratic_weighted_kappa([1, 2, 3], [1, 2, 3]), 1.0)
        metrics = aggregate_metrics([[1.0] * 4, [5.0] * 4], [[1.0] * 4, [5.0] * 4], [True, True])
        self.assertEqual(metrics["primary/macro_mae"], 0.0)
        self.assertEqual(metrics["decoder/parse_failure_rate"], 0.0)

    def test_one_accelerate_shard_rank_coverage_and_post_prepare_update_count(self):
        assignments = accelerator_batch_assignment(batch_count=11, world_size=3)
        self.assertEqual(assignments, ((0, 3, 6, 9), (1, 4, 7, 10), (2, 5, 8)))
        self.assertEqual(sorted(index for assignment in assignments for index in assignment), list(range(11)))
        self.assertEqual(updates_for_prepared_loader(17, 8), 3)
        with self.assertRaises(ContractError): updates_for_prepared_loader(0, 8)

    def test_selection_fallback_mean_is_partition_only_not_refit_all_records(self):
        train = [{"score": {key: 1.0 for key in SCORE_KEYS}}, {"score": {key: 3.0 for key in SCORE_KEYS}}]
        self.assertEqual(score_mean(train), {key: 2.0 for key in SCORE_KEYS})

    def test_deterministic_generation_sanitizes_sampling_only_values(self):
        from mal2026.decoder import sanitized_deterministic_generation_config
        config = types.SimpleNamespace(do_sample=True, temperature=0.2, top_p=0.8, min_p=0.2, typical_p=0.7, top_k=12, epsilon_cutoff=0.1, eta_cutoff=0.1)
        sanitized = sanitized_deterministic_generation_config(config)
        self.assertFalse(sanitized.do_sample); self.assertEqual((sanitized.temperature, sanitized.top_p, sanitized.top_k), (1.0, 1.0, 50)); self.assertTrue(config.do_sample)

    def test_output_root_rejects_symlink_escape(self):
        from mal2026.decoder import resolve_run_output_dir
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "outputs").mkdir(); outside = root / "outside"; outside.mkdir(); (root / "outputs" / "runs").symlink_to(outside, target_is_directory=True)
            with patch("mal2026.decoder.project_root", return_value=root):
                with self.assertRaises(ContractError): resolve_run_output_dir("safe-run", root / "outputs" / "runs" / "safe-run")


if __name__ == "__main__": unittest.main()
