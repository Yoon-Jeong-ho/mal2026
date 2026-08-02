from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mal2026.prompt_reference_npcr import (
    NPCRConfig, NPCRRow, PairSamplingSpec, PromptReferenceNPCRError, _validate_public_payload,
    _atomic_jsonl_private, bound_r0_predictions, build_prompt_pairs, build_utility_network,
    load_canonical_raw_rows, ordinal_band, outer_and_inner_indices, recover_absolute_scores,
    select_anchors, train_utility_network, utility_difference,
)


def row(identifier: int, prompt: str, score: float, fold: int = 0) -> NPCRRow:
    return NPCRRow(str(identifier), f"d{identifier}", prompt, tuple([float(identifier)] * 4096),
                   (score, score, score), fold)


class PromptReferenceNPCRTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = PairSamplingSpec("adjacent-skip", (1, 2), 2, 3)
        self.rows = (
            row(0, "p", 1.0), row(1, "p", 2.25), row(2, "p", 3.0), row(3, "p", 4.0),
            row(4, "q", 1.0), row(5, "q", 3.0), row(6, "q", 4.0),
        )

    def test_prompt_local_pair_sampling_and_document_exclusion(self) -> None:
        pairs = build_prompt_pairs(self.rows, tuple(range(len(self.rows))), 0, self.candidate, 19)
        self.assertTrue(pairs)
        for pair in pairs:
            self.assertEqual(self.rows[pair.query].prompt_num, self.rows[pair.reference].prompt_num)
            self.assertNotEqual(self.rows[pair.query].document_id, self.rows[pair.reference].document_id)
            self.assertIn(abs(ordinal_band(self.rows[pair.query].raw_scores[0]) - ordinal_band(self.rows[pair.reference].raw_scores[0])), (1, 2))

    def test_outer_inner_isolation(self) -> None:
        rows = tuple(row(index, f"p{index % 4}", 3.0, index % 5) for index in range(2000))
        outer, inner = outer_and_inner_indices(rows, 3)
        self.assertEqual(len(outer), 400)
        for fit, dev in inner.values():
            self.assertTrue(set(outer).isdisjoint(fit))
            self.assertTrue(set(outer).isdisjoint(dev))
            self.assertTrue(set(fit).isdisjoint(dev))

    def test_scalar_difference_is_antisymmetric(self) -> None:
        model = build_utility_network(4096, 4)
        left, right = torch.randn(3, 4096), torch.randn(3, 4096)
        self.assertTrue(torch.allclose(utility_difference(model, left, right), -utility_difference(model, right, left)))

    def test_anchor_recovery_uses_fit_only_and_fractional_anchor(self) -> None:
        rows = (row(0, "p", 2.25), row(1, "p", 3.75), row(2, "p", 99.0))
        # Query's intentionally invalid-looking held gold is never read by recovery.
        model = torch.nn.Linear(4096, 1, bias=False)
        with torch.no_grad(): model.weight.zero_()
        prediction = recover_absolute_scores(model, rows, (2,), (0, 1), 0, self.candidate, seed=7)
        self.assertAlmostEqual(float(prediction[0]), 3.0, places=6)
        anchors = select_anchors(rows, (0, 1), 2, 0, 3, 7)
        self.assertEqual(set(anchors), {0, 1})

    def test_raw_fractional_targets_and_average_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for index in range(2000):
                    stream.write(json.dumps({"id": str(index), "document_id": str(index), "prompt_num": "p", "prompt": "x", "essay": "y",
                                             "score": {"content": 2.25, "organization": 3.5, "expression": 4.75, "average": "forbidden"}}) + "\n")
            from mal2026.prompt_reference_npcr import file_sha256
            values = load_canonical_raw_rows(path, file_sha256(path))
            self.assertEqual(values["0"][2], (2.25, 3.5, 4.75))
        self.assertNotEqual(float(np.sqrt(np.mean((np.asarray([2.25]) - 2.0) ** 2))), 0.0)

    def test_public_payload_rejects_private_row_fields(self) -> None:
        _validate_public_payload({"metrics": {"rmse": 0.5}})
        for key in ("source_id", "raw_gold", "embedding", "prompt_num"):
            with self.assertRaises(PromptReferenceNPCRError):
                _validate_public_payload({"nested": [{key: "private"}]})

    def test_config_recursively_forbids_validation_text(self) -> None:
        raw = json.loads(Path("configs/prompt_reference_npcr.v1.json").read_text(encoding="utf-8"))
        raw["nested_note"] = {"forbidden": "validation must not be referenced"}
        with self.assertRaisesRegex(PromptReferenceNPCRError, "validation"):
            NPCRConfig.from_mapping(raw)

    def test_swapped_r0_fold_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "r0.jsonl"
            rows = (row(0, "p", 2.0, 0), row(1, "p", 3.0, 1))
            payload = []
            for item, fold in zip(rows, (1, 0), strict=True):
                payload.append({"source_id": item.source_id, "fold": fold,
                                "continuous_prediction": {axis: 3.0 for axis in ("content", "organization", "expression")},
                                "half_up_integer_prediction": {axis: 3 for axis in ("content", "organization", "expression")},
                                "reference_score": {axis: 3.0 for axis in ("content", "organization", "expression")}})
            path.write_text("".join(json.dumps(item) + "\n" for item in payload), encoding="utf-8")
            config = SimpleNamespace(r0_oof_prediction_path=str(path))
            with self.assertRaises(PromptReferenceNPCRError):
                bound_r0_predictions(config, rows)

    def test_pair_fit_is_deterministic(self) -> None:
        rows = tuple(row(index, "p", float(index + 1)) for index in range(5))
        config = SimpleNamespace(hidden_dim=4, learning_rate=0.01, weight_decay=0.0, epochs=1, batch_size=4)
        first, pairs = train_utility_network(rows, tuple(range(5)), 0, self.candidate, config, seed=41)
        second, _ = train_utility_network(rows, tuple(range(5)), 0, self.candidate, config, seed=41)
        self.assertTrue(pairs)
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(first.state_dict().values(), second.state_dict().values())))

    def test_private_writer_fsync_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "restricted" / "rows.jsonl"
            _atomic_jsonl_private(path, [{"source_id": "x"}])
            self.assertEqual(path.stat().st_mode & 0o777, 0o660)
            self.assertEqual(path.parent.stat().st_mode & 0o007, 0)


if __name__ == "__main__":
    unittest.main()
