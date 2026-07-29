from __future__ import annotations

from pathlib import Path
import unittest

from mal2026.augmented_bundle_rationale import AugmentedRow
from mal2026.augmented_rationale_encoder import AugmentedEncoderConfig, augmented_score_rows
from mal2026.rationale_aware_encoder import deterministic_split, load_continuous_rows
from mal2026.solar_axis_augmentation import AXES, load_train_rows


ROOT = Path(__file__).resolve().parents[1]


class AugmentedRationaleEncoderTests(unittest.TestCase):
    def test_both_configs_are_three_axis_continuous_and_source_disjoint(self) -> None:
        for name, model_key in (
            ("augmented_rationale_aware_qwen3_embedding_8b.v1.json", "qwen3_embedding_8b"),
            ("augmented_rationale_aware_kure_v1.v1.json", "kure_v1"),
        ):
            config = AugmentedEncoderConfig.from_json(ROOT / "configs" / name, require_dependencies=False)
            self.assertEqual(config.model_key, model_key)
            self.assertEqual(config.score_fields, AXES)
            self.assertFalse(config.average_target_used)
            self.assertEqual(config.selection_epochs, (1, 2, 3, 4))
            self.assertIn("source_disjoint", config.selection_dev_contract)

    def test_augmented_rows_preserve_fractional_three_axis_targets(self) -> None:
        sources = load_train_rows()
        raw = []
        for source in sources:
            for axis in AXES:
                raw.append(AugmentedRow(
                    source.identifier, axis, f"{source.identifier}::solar-degrade::{axis}",
                    source.prompt, source.essay + f" {axis} 편집본", (2.25, 3.5, 4.75), 1,
                ))
        converted = augmented_score_rows(raw)
        self.assertEqual(len(converted), 6000)
        self.assertEqual(converted[0].labels, (2.25, 3.5, 4.75))

    def test_internal_dev_source_augmentations_are_all_excluded(self) -> None:
        base = AugmentedEncoderConfig.from_json(
            ROOT / "configs/augmented_rationale_aware_qwen3_embedding_8b.v1.json",
            require_dependencies=False,
        )
        rows = load_continuous_rows(
            ROOT / "eval/train.jsonl",
            "b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737",
            2000,
        )
        train, dev, _ = deterministic_split(rows, base.seed)
        train_ids, dev_ids = {row.identifier for row in train}, {row.identifier for row in dev}
        sources = load_train_rows()
        train_augmented = sum(source.identifier in train_ids for source in sources) * 3
        dev_augmented = sum(source.identifier in dev_ids for source in sources) * 3
        self.assertEqual((len(train), train_augmented, len(dev), dev_augmented), (1600, 4800, 400, 1200))
        self.assertFalse(train_ids & dev_ids)

    def test_canonical_validation_load_is_after_refit(self) -> None:
        source = (ROOT / "src/mal2026/augmented_rationale_encoder.py").read_text(encoding="utf-8")
        self.assertGreater(source.index("validation = load_continuous_rows"), source.index("refit_result = refitter.train()"))
        self.assertIn('"validation_used_for_training_or_selection": False', source)


if __name__ == "__main__":
    unittest.main()
