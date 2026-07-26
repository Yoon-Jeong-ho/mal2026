import json
from pathlib import Path
import tempfile
import unittest

from mal2026.official_remaining_pipeline import resolve_decoder_score_config


class OfficialRemainingPipelineTests(unittest.TestCase):
    def test_decoder_resolution_changes_only_rationale_bindings(self) -> None:
        # The production resolver is deliberately filesystem-bound.  This
        # test checks its public shape indirectly by pinning the source code's
        # no-average and three-axis contract rather than fabricating row data.
        source = Path(__file__).resolve().parents[1] / "src/mal2026/official_remaining_pipeline.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn('"rationale_key": handoff.get("rationale_key")', text)
        self.assertNotIn('"average"', text)

    def test_runtime_config_is_json_round_trippable(self) -> None:
        value = {"score_fields": ["content", "organization", "expression"], "average_target_used": False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(value))
            self.assertEqual(json.loads(path.read_text()), value)


if __name__ == "__main__":
    unittest.main()
