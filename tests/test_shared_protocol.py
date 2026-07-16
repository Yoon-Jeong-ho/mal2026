from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mal2026.config import ConfigError, validate_experiment_config
from mal2026.data_contract import DataContractError, ScoreVector, load_and_validate_jsonl, split_prompt_groups
from mal2026.formatting import (
    assert_teacher_input_safe,
    format_decoder_input,
    format_teacher_rationale_input,
    parse_score_json,
    render_score_target,
)
from mal2026.metrics import aggregate_prediction_rows, compute_regression_metrics, quadratic_weighted_kappa
from mal2026.provenance import TelemetrySafetyError, aggregate_only_payload
from mal2026.rationale import RationaleValidationError, validate_rationale_payload


def row(identifier: str, prompt_num: str, prompt: str, essay: str, average: float = 3.5) -> dict:
    return {
        "id": identifier,
        "document_id": "doc-" + identifier,
        "prompt_num": prompt_num,
        "prompt": prompt,
        "essay": essay,
        "score": {"content": 3.0, "organization": 3.25, "expression": 3.75, "average": average},
    }


class SharedProtocolTests(unittest.TestCase):
    def records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            # Fixtures are artificial and contain no restricted writing.
            path.write_text("\n".join(json.dumps(item) for item in [
                row("record-unique-a", "p1", "Prompt one", "Alpha text"),
                row("record-unique-b", "p1", "Prompt one", "Beta text"),
                row("record-unique-c", "p2", "Prompt two", "Gamma text"),
                row("record-unique-d", "p2", "Prompt two", "Delta text"),
            ]) + "\n", encoding="utf-8")
            return load_and_validate_jsonl(path)

    def test_group_split_is_deterministic_and_manifest_is_aggregate_only(self):
        records = self.records()
        first = split_prompt_groups(records, 0.5)
        second = split_prompt_groups(records, 0.5)
        self.assertEqual(first.manifest, second.manifest)
        self.assertEqual(2, len(first.development))
        self.assertEqual(2, len(first.optimization_train))
        self.assertNotIn("Prompt one", json.dumps(first.manifest))
        self.assertNotIn("Alpha text", json.dumps(first.manifest))

    def test_schema_rejects_duplicate_ids_and_prompt_num_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.jsonl"
            duplicate.write_text(json.dumps(row("x", "p1", "A", "x")) + "\n" + json.dumps(row("x", "p2", "B", "y")), encoding="utf-8")
            with self.assertRaises(DataContractError):
                load_and_validate_jsonl(duplicate)
        conflicting = self.records()[:]
        bad = list(conflicting)
        from dataclasses import replace
        bad.append(replace(conflicting[0], id="new", prompt="Different prompt"))
        with self.assertRaises(DataContractError):
            split_prompt_groups(bad)

    def test_teacher_input_excludes_labels_ids_and_split(self):
        record = self.records()[0]
        teacher = format_teacher_rationale_input(record)
        assert_teacher_input_safe(teacher)
        self.assertNotIn(record.id, teacher)
        self.assertNotIn(record.document_id, teacher)
        self.assertNotIn(str(record.scores.average), teacher)
        self.assertIn("<student_essay>", teacher)
        self.assertIn("<writing_prompt>", format_decoder_input(record))

    def test_exact_score_render_and_parse(self):
        scores = ScoreVector(1.0, 2.345, 3.5, 5.0)
        rendered = render_score_target(scores)
        self.assertEqual('{"content":1.00,"organization":2.35,"expression":3.50,"average":5.00}', rendered)
        self.assertTrue(parse_score_json(rendered, scores).valid)
        self.assertFalse(parse_score_json("Here: " + rendered, scores).valid)
        out_of_range = parse_score_json('{"content":5.01,"organization":2.35,"expression":3.50,"average":5.00}', scores)
        self.assertFalse(out_of_range.valid)
        self.assertTrue(out_of_range.out_of_range)

    def test_rationale_requires_all_criteria_and_exact_offsets_without_score_cues(self):
        essay = "가나다라마바사"
        payload = {"rationale": [
            {"criterion": "CONTENT", "quote": "가나", "start": 0, "end": 2, "observation": "주제를 직접 다룬다"},
            {"criterion": "ORGANIZATION", "quote": "다라", "start": 2, "end": 4, "observation": "앞뒤 문장이 연결된다"},
            {"criterion": "EXPRESSION", "quote": "마바", "start": 4, "end": 6, "observation": "어휘가 구체적이다"},
        ]}
        self.assertTrue(validate_rationale_payload(payload, essay).nonempty_valid)
        invalid = {"rationale": [dict(item, observation="3점 수준이다") for item in payload["rationale"]]}
        with self.assertRaises(RationaleValidationError):
            validate_rationale_payload(invalid, essay)
        invalid_offset = {"rationale": [dict(item) for item in payload["rationale"]]}
        invalid_offset["rationale"][0]["end"] = 1
        with self.assertRaises(RationaleValidationError):
            validate_rationale_payload(invalid_offset, essay)
        self.assertFalse(validate_rationale_payload({"rationale": []}, essay).nonempty_valid)

    def test_metrics_and_telemetry_are_aggregate_only(self):
        target = [ScoreVector(1, 2, 3, 4), ScoreVector(2, 3, 4, 5)]
        result = compute_regression_metrics(target, target)
        self.assertEqual(0.0, result["per_target"]["average"]["mae"])
        self.assertEqual(1.0, quadratic_weighted_kappa([1, 2, 3], [1, 2, 3]))
        aggregate = aggregate_prediction_rows([{"target": target[0].as_dict(), "prediction": target[0].as_dict()}])
        self.assertEqual(1, aggregate["record_count"])
        with self.assertRaises(TelemetrySafetyError):
            aggregate_only_payload({"essay": "private"})

    def test_config_fails_closed_on_templates_and_accepts_frozen_config(self):
        template = json.loads(Path("configs/encoder-qwen3.template.json").read_text(encoding="utf-8"))
        with self.assertRaises(ConfigError):
            validate_experiment_config(template)
        template["model"]["revision"] = "a" * 40
        template["model"]["tokenizer_revision"] = "b" * 40
        template["adapter"]["target_modules"] = ["q_proj"]
        self.assertEqual("encoder-qwen3", validate_experiment_config(template)["run_kind"])
        for field, invalid in (("normalize_embeddings", False), ("regression_loss", "mae"), ("loss_reduction", "sum"), ("pooling", "last_token_mean")):
            altered = json.loads(json.dumps(template))
            altered["model"][field] = invalid
            with self.assertRaises(ConfigError):
                validate_experiment_config(altered)
        for field, invalid in (("weight_decay", -0.1), ("num_workers", -1), ("early_stopping_min_delta", -0.1), ("early_stopping_patience", 0)):
            altered = json.loads(json.dumps(template))
            altered["optimization"][field] = invalid
            with self.assertRaises(ConfigError):
                validate_experiment_config(altered)


if __name__ == "__main__":
    unittest.main()
