from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mal2026.data_contract import DataContractError, ScoreVector, load_and_validate_jsonl, split_prompt_groups
from mal2026.formatting import format_decoder_input, format_encoder_input, parse_score_json, render_score_target
from mal2026.metrics import aggregate_prediction_rows, compute_regression_metrics, quadratic_weighted_kappa
from mal2026.provenance import TelemetrySafetyError, aggregate_only_payload


def row(identifier: str, prompt_num: str, prompt: str, essay: str, average: float = 3.5) -> dict:
    return {
        "id": identifier,
        "document_id": "doc-" + identifier,
        "prompt_num": prompt_num,
        "prompt": prompt,
        "essay": essay,
        "score": {"content": 3.0, "organization": 3.25, "expression": 3.75, "average": average},
    }


class CoreContractTests(unittest.TestCase):
    def records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in [
                row("record-unique-a", "p1", "Prompt one", "Alpha text"),
                row("record-unique-b", "p1", "Prompt one", "Beta text"),
                row("record-unique-c", "p2", "Prompt two", "Gamma text"),
                row("record-unique-d", "p2", "Prompt two", "Delta text"),
            ]) + "\n", encoding="utf-8")
            return load_and_validate_jsonl(path)

    def test_group_split_is_deterministic_and_aggregate_only(self):
        first = split_prompt_groups(self.records(), 0.5)
        second = split_prompt_groups(self.records(), 0.5)
        self.assertEqual(first.manifest, second.manifest)
        self.assertEqual(2, len(first.development))
        self.assertNotIn("Prompt one", json.dumps(first.manifest))
        self.assertNotIn("Alpha text", json.dumps(first.manifest))

    def test_schema_rejects_duplicate_ids_and_prompt_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.jsonl"
            duplicate.write_text(json.dumps(row("x", "p1", "A", "x")) + "\n" + json.dumps(row("x", "p2", "B", "y")), encoding="utf-8")
            with self.assertRaises(DataContractError):
                load_and_validate_jsonl(duplicate)

    def test_model_inputs_exclude_identifiers_and_scores(self):
        record = self.records()[0]
        decoder = format_decoder_input(record)
        encoder = format_encoder_input(record)
        self.assertNotIn(record.id, decoder)
        self.assertNotIn(record.document_id, decoder)
        self.assertNotIn(str(record.scores.average), decoder)
        self.assertIn("<student_essay>", decoder)
        self.assertIn("[학생 글]", encoder)

    def test_exact_score_render_and_parse(self):
        scores = ScoreVector(1.0, 2.345, 3.5, 5.0)
        rendered = render_score_target(scores)
        self.assertEqual('{"content":1.00,"organization":2.35,"expression":3.50,"average":5.00}', rendered)
        self.assertTrue(parse_score_json(rendered, scores).valid)
        self.assertFalse(parse_score_json("Here: " + rendered, scores).valid)

    def test_metrics_and_telemetry_are_aggregate_only(self):
        target = [ScoreVector(1, 2, 3, 4), ScoreVector(2, 3, 4, 5)]
        result = compute_regression_metrics(target, target)
        self.assertEqual(0.0, result["per_target"]["average"]["mae"])
        self.assertEqual(1.0, quadratic_weighted_kappa([1, 2, 3], [1, 2, 3]))
        aggregate = aggregate_prediction_rows([{"target": target[0].as_dict(), "prediction": target[0].as_dict()}])
        self.assertEqual(1, aggregate["record_count"])
        with self.assertRaises(TelemetrySafetyError):
            aggregate_only_payload({"feedback": "restricted"})


if __name__ == "__main__":
    unittest.main()
