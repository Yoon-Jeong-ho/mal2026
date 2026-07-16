from __future__ import annotations

import unittest

from mal2026.standard_decoder_data import (
    FEEDBACK_FIELDS, RestrictedRow, StandardDecoderContractError, parse_decoder_scores,
    render_human_feedback_target, render_scores,
)
from mal2026.standard_decoder_vllm import aggregate_metrics


def row() -> RestrictedRow:
    return RestrictedRow(
        "private-id", "private prompt", "private essay",
        {"content": 1.0, "organization": 2.0, "expression": 3.0, "average": 4.0},
        {key: f"private feedback {key}" for key in FEEDBACK_FIELDS},
    )


class StandardDecoderStackTests(unittest.TestCase):
    def test_direct_is_exact_and_no_prose_fallback(self):
        target = render_scores(row().score)
        self.assertEqual('{"content":1.00,"organization":2.00,"expression":3.00,"average":4.00}', target)
        self.assertEqual(row().score, parse_decoder_scores(target, "direct"))
        self.assertIsNone(parse_decoder_scores("reason " + target, "direct"))
        self.assertIsNone(parse_decoder_scores(target.replace("1.00", "1"), "direct"))
        self.assertIsNone(parse_decoder_scores(target.replace("1.00", "5.99"), "direct"))

    def test_human_feedback_parser_requires_complete_canonical_json(self):
        target = render_human_feedback_target(row())
        self.assertEqual(row().score, parse_decoder_scores(target, "human_feedback"))
        self.assertIsNone(parse_decoder_scores(target + "\n", "human_feedback"))
        self.assertIsNone(parse_decoder_scores(target.replace('"scores"', '"other"'), "human_feedback"))

    def test_renderer_refuses_noncanonical_score_order(self):
        with self.assertRaises(StandardDecoderContractError):
            render_scores({"average": 4, "content": 1, "organization": 2, "expression": 3})

    def test_aggregate_metrics_contains_no_row_text(self):
        target = row().score
        result = aggregate_metrics([target], [target], [True])
        self.assertEqual(0, result["primary_macro_mae"])
        self.assertNotIn("prompt", str(result))
        self.assertNotIn("private", str(result))


if __name__ == "__main__":
    unittest.main()
