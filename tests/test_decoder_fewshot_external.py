from __future__ import annotations

import json
from pathlib import Path
import unittest

from mal2026.decoder_fewshot_external import (
    ExternalConfig,
    openai_body,
    request_records,
    response_text,
)


class DecoderFewshotExternalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ExternalConfig.from_json(Path("configs/decoder_fewshot_external.v1.json"))

    def test_config_contract(self) -> None:
        self.assertEqual(("gpt-5.6-terra", "gpt-5.6-luna"), self.config.api_models)
        self.assertEqual((0, 1, 2, 3), self.config.solar.gpu_scope)

    def test_request_population_and_role_contract(self) -> None:
        records = request_records(self.config)
        self.assertEqual(800, len(records))
        self.assertEqual(800, len({(row["source_id"], row["condition"]) for row in records}))
        messages = records[0]["messages"]
        self.assertEqual(12, len(messages))
        self.assertEqual(5, sum(row["role"] == "assistant" for row in messages))
        self.assertEqual("user", messages[-1]["role"])

    def test_openai_body(self) -> None:
        messages = request_records(self.config)[0]["messages"]
        body = openai_body("gpt-5.6-terra", messages, 1800)
        self.assertEqual("none", body["reasoning"]["effort"])
        self.assertFalse(body["store"])
        self.assertEqual("json_schema", body["text"]["format"]["type"])
        self.assertEqual([row["role"] for row in messages], [row["role"] for row in body["input"]])
        self.assertEqual(
            ["output_text" if row["role"] == "assistant" else "input_text" for row in messages],
            [row["content"][0]["type"] for row in body["input"]],
        )

    def test_response_text(self) -> None:
        payload = {"output": [{"content": [{"type": "output_text", "text": json.dumps({"ok": True})}]}]}
        self.assertEqual('{"ok": true}', response_text(payload))


if __name__ == "__main__":
    unittest.main()
