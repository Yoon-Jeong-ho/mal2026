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

class StandardDecoderRefitProvenanceTests(unittest.TestCase):
    def _refit_config(self, root, summary_path, *, learning_rate=2e-5):
        from mal2026.standard_decoder_train import StandardSFTConfig
        return StandardSFTConfig(
            run_id="refit-run", phase="refit", mode="direct", model_path="/models/qwen",
            tokenizer_path="/models/qwen", model_revision="a" * 40, tokenizer_revision="a" * 40,
            prepared_manifest="/manifest.json", output_dir=str(root / "outputs" / "standard-runs" / "refit-run"),
            seed=2026, max_length=2048, learning_rate=learning_rate, num_train_epochs=99,
            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
            eval_steps=0, save_steps=0, logging_steps=2, early_stopping_patience=1,
            lora_r=32, lora_alpha=64, lora_dropout=0.05,
            selection_summary_path=str(summary_path), selected_global_step=12,
        )

    def test_refit_requires_matching_selection_run_and_identity(self):
        import json
        import tempfile
        from pathlib import Path
        from mal2026 import standard_decoder_train as module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "outputs" / "standard-runs" / "selection-run"
            run_dir.mkdir(parents=True)
            summary_path = run_dir / "selected_checkpoint.json"
            old_root = module.ROOT
            module.ROOT = root
            try:
                config = self._refit_config(root, summary_path)
                identity = module._config_identity(config)
                (run_dir / "standard_training_complete.json").write_text(json.dumps({
                    "status": "completed", "phase": "selection", "run_id": "selection-run",
                    "mode": "direct", "model_revision": "a" * 40, "tokenizer_revision": "a" * 40,
                    "identity": identity,
                }), encoding="utf-8")
                summary_path.write_text(json.dumps({
                    "status": "completed", "phase": "selection", "selection_run_id": "selection-run",
                    "mode": "direct", "model_revision": "a" * 40, "tokenizer_revision": "a" * 40, "selected_global_step": 12,
                }), encoding="utf-8")
                self.assertEqual(12, module._verify_refit_selection(config)["selected_global_step"])
                mismatched = self._refit_config(root, summary_path, learning_rate=3e-5)
                with self.assertRaises(StandardDecoderContractError):
                    module._verify_refit_selection(mismatched)
            finally:
                module.ROOT = old_root
