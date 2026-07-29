from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import unittest

from mal2026.augmented_bundle_rationale import (
    AugmentedRow,
    config,
    output_schema,
    parse_output,
    render_messages,
    validate_records,
)
from mal2026.solar_axis_augmentation import AXES, load_train_rows


class AugmentedBundleRationaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = load_train_rows()

    def test_contract_is_bundle_only_and_has_no_average_or_score_output(self) -> None:
        value = config()
        self.assertEqual(value["output_contract"]["structure"], "bundle")
        self.assertFalse(value["output_contract"]["axis_triplet_allowed"])
        self.assertFalse(value["output_contract"]["score_output_allowed"])
        self.assertFalse(value["input_contract"]["average_allowed"])
        schema = output_schema()
        self.assertEqual(set(schema["properties"]), set(AXES))

    def test_render_includes_quarter_step_pseudo_score_but_no_output_score(self) -> None:
        source = self.sources[0]
        row = AugmentedRow(
            source.identifier, "content", source.identifier + "::solar-degrade::content",
            source.prompt, source.essay + " 편집 문장", (2.25, 3.5, 4.0), 1,
        )
        messages = render_messages(row)
        payload = messages[1]["content"]
        self.assertIn('"pseudo_score":{"content":2.25,"organization":3.5,"expression":4.0}', payload)
        self.assertNotIn('"average"', payload)
        self.assertIn("점수, 새 점수, 개선 제안은 출력하지 마라", messages[0]["content"])

    def test_parser_accepts_exact_bundle_rationale_only(self) -> None:
        raw = {axis: {"rationale": f"{axis} 근거"} for axis in AXES}
        parsed = parse_output(json.dumps(raw, ensure_ascii=False))
        self.assertEqual(set(parsed), set(AXES))
        with self.assertRaises(Exception):
            parse_output(json.dumps({**raw, "score": 3}, ensure_ascii=False))

    def test_full_population_requires_three_variants_per_train_source(self) -> None:
        raw = []
        for source in self.sources:
            for axis in AXES:
                raw.append({
                    "source_id": source.identifier,
                    "target_axis": axis,
                    "augmented_id": f"{source.identifier}::solar-degrade::{axis}",
                    "prompt": source.prompt,
                    "essay": source.essay + f" {axis} 편집본",
                    "score": {name: 3.0 for name in AXES},
                    "attempts": 1,
                })
        rows = validate_records(raw, self.sources)
        self.assertEqual(len(rows), 6000)
        self.assertEqual({row.target_axis for row in rows}, set(AXES))

    def test_runner_request_is_single_bundle_with_no_gold_score_output(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "scripts/run_solar_augmented_bundle_rationales.py"
        spec = importlib.util.spec_from_file_location("augmented_rationale_runner_contract", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = self.sources[0]
        row = AugmentedRow(
            source.identifier, "organization", source.identifier + "::solar-degrade::organization",
            source.prompt, source.essay + " 조직 편집본", (3.0, 2.25, 3.5), 1,
        )
        body = module.request_body(row, 1)
        schema = body["response_format"]["json_schema"]["schema"]
        self.assertEqual(set(schema["properties"]), set(AXES))
        self.assertNotIn("score", schema["properties"])
        self.assertEqual(body["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
