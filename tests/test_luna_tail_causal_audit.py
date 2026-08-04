from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_luna_tail_causal_audit as audit  # noqa: E402


CONFIG = ROOT / "configs/luna_tail_causal_audit.v1.json"
RUN_ID = "luna-r0-tail-causal-audit-v1-20260804-002"


class LunaTailCausalAuditTests(unittest.TestCase):
    def test_exact_population_and_no_validation_or_average(self):
        config = audit.load_config(CONFIG)
        cases, report = audit.build_cases(config)
        self.assertEqual(len(cases), 752)
        self.assertEqual(report["counts"]["primary_low_to_central:content"], 144)
        self.assertEqual(report["counts"]["primary_low_to_central:organization"], 226)
        self.assertEqual(report["counts"]["primary_low_to_central:expression"], 56)
        serialized = json.dumps(config, ensure_ascii=False).lower()
        self.assertNotIn("validation.jsonl", serialized)
        self.assertTrue(config["authorization"]["average_target_used"] is False)

    def test_blind_and_revealed_prompt_boundaries(self):
        source = {"prompt": "주제", "essay": "본문"}
        case = {"axis": "content", "source_id": "x", "gold_raw": 2.0,
                "gold_band": 2, "pred_raw": 3.1, "pred_band": 3}
        rationales = {name: {"x": {axis: f"{name}-{axis}" for axis in audit.AXES}}
                      for name in ("r0_input", "official_blind", "official_conditioned", "final_dpo")}
        for condition in ("canonical_blind", "operational_blind"):
            body = audit.response_body("gpt-5.6-luna", condition, source, case, rationales, "shuffled")
            user = body["input"][1]["content"][0]["text"]
            self.assertNotIn("human_reference", user)
            self.assertNotIn("exact_r0", user)
            self.assertNotIn("rationale", user)
        revealed = audit.response_body("gpt-5.6-luna", "revealed_causal", source, case, rationales, "shuffled")
        user = revealed["input"][1]["content"][0]["text"]
        self.assertIn("human_reference", user)
        self.assertIn("exact_r0", user)
        self.assertIn("shuffled_r0_input", user)

    def test_structured_schemas_are_strict_and_complete(self):
        for schema in (audit.blind_schema(), audit.causal_schema()):
            queue = [schema]
            while queue:
                value = queue.pop()
                if isinstance(value, dict):
                    if value.get("type") == "object":
                        self.assertFalse(value.get("additionalProperties", True))
                        self.assertEqual(set(value.get("required", [])), set(value.get("properties", {})))
                    queue.extend(value.values())
                elif isinstance(value, list):
                    queue.extend(value)

    def test_public_results_have_no_row_material(self):
        forbidden = {"source_id", "document_id", "essay", "essay_text", "prompt_text",
                     "why_not_adjacent", "human_vs_r0_reason"}
        for path in (
            ROOT / "outputs/luna-tail-causal-audit-v1" / RUN_ID / "aggregate.json",
            ROOT / "outputs/luna-tail-causal-audit-v1" / RUN_ID / "case_analysis.json",
        ):
            payload = json.loads(path.read_text(encoding="utf-8"))
            queue = [payload]
            while queue:
                value = queue.pop()
                if isinstance(value, dict):
                    self.assertTrue(forbidden.isdisjoint(value))
                    queue.extend(value.values())
                elif isinstance(value, list):
                    queue.extend(value)


if __name__ == "__main__":
    unittest.main()
