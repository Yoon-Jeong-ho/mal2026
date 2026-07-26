from __future__ import annotations
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from mal2026.official_decoder_aihub_pretrain import (
    ARCHITECTURES, DecoderAIHubConfig, exported_tensor_contract, select_event,
)
from scripts.orchestrate_official_decoder_aihub_score_pretrain import plan


class DecoderAIHubPretrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path("configs/official_decoder_aihub_integer_score_pretrain.v1.json")
        self.config = DecoderAIHubConfig.from_json(self.path, require_dependencies=False)

    def test_full_parameter_integer_three_axis_contract(self) -> None:
        self.assertEqual(self.config.architectures, ARCHITECTURES)
        self.assertEqual(self.config.score_fields, ("content", "organization", "expression"))
        self.assertIs(self.config.integer_target_used, True)
        self.assertIs(self.config.average_target_used, False)
        self.assertEqual(self.config.training_method, "full_parameter")
        self.assertEqual(self.config.downstream_adaptation, "fresh_MAL_LoRA")
        serialized = self.path.read_text()
        self.assertNotIn("validation.jsonl", serialized)
        self.assertNotIn("lora_r", serialized.lower())

    def test_repaired_lineage_pins_fsdp1_for_adafactor(self) -> None:
        config = DecoderAIHubConfig.from_json(
            Path("configs/official_decoder_aihub_integer_score_pretrain.repair1.v1.json"),
            require_dependencies=False,
        )
        self.assertEqual(config.optimizer, "adafactor")
        self.assertEqual(config.fsdp_version, 1)

    def test_selection_identity_survives_json_round_trip(self) -> None:
        identity = self.config.identity("bounded_regression")
        self.assertEqual(json.loads(json.dumps(identity)), identity)
        self.assertIsInstance(identity["score_fields"], list)

    def test_selection_is_integer_primary_and_exact_step(self) -> None:
        events = [
            {"global_step": 100, "macro_integer_rmse": .7, "macro_integer_spearman": .9, "macro_continuous_rmse": .1},
            {"global_step": 200, "macro_integer_rmse": .6, "macro_integer_spearman": .1, "macro_continuous_rmse": .9},
            {"global_step": 300, "macro_integer_rmse": .6, "macro_integer_spearman": .2, "macro_continuous_rmse": 1.0},
        ]
        self.assertEqual(select_event(events)["global_step"], 300)
        source = Path("src/mal2026/official_decoder_aihub_pretrain.py").read_text()
        self.assertIn("StopAtSelectedStep", source)
        self.assertIn("scheduler_horizon_steps", source)
        self.assertIn("exact_selected_step_stop", source)
        self.assertIn("gradient_checkpointing_enable", source)

    def test_score_heads_match_live_fsdp_mixed_precision_dtype(self) -> None:
        for path in (
            "src/mal2026/official_decoder_aihub_pretrain.py",
            "src/mal2026/official_decoder_score.py",
        ):
            source = Path(path).read_text()
            self.assertIn("pooled.to(self.score_head.weight.dtype)", source)
            self.assertIn(".float()", source)
        pretrain = Path("src/mal2026/official_decoder_aihub_pretrain.py").read_text()
        self.assertIn("dtype=next(backbone.parameters()).dtype", pretrain)

    def test_gpu0_smoke_then_fsdp4_selection_refit_for_all_architectures(self) -> None:
        stages = plan(self.path, self.config)
        self.assertEqual(len(stages), 12)
        for offset in range(0, 12, 4):
            self.assertEqual([stage["gpus"] for stage in stages[offset:offset+4]], [[0], [0], [0,1,2,3], [0,1,2,3]])
            self.assertEqual([stage["phase"] for stage in stages[offset:offset+4]], ["selection", "refit", "selection", "refit"])
        source = Path("scripts/orchestrate_official_decoder_aihub_score_pretrain.py").read_text()
        self.assertIn("fsdp4_full_parameter", source)
        self.assertNotIn('"4,5,6,7"', source)

    def test_export_contract_distinguishes_generative_and_matched_heads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"model.layers.0.weight": torch.zeros(1)}, str(root / "model.safetensors"))
            generative = exported_tensor_contract(root, "generative")
            self.assertTrue(generative["complete_full_parameter_state"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({
                "backbone.layers.0.weight": torch.zeros(1),
                "score_head.weight": torch.zeros(3, 3584),
                "score_head.bias": torch.zeros(3),
            }, str(root / "model.safetensors"))
            bounded = exported_tensor_contract(root, "bounded_regression")
            self.assertEqual(bounded["score_head_tensor_shapes"]["score_head.weight"], [3, 3584])

    def test_generative_selection_is_free_running_not_teacher_forced_accuracy(self) -> None:
        source = inspect.getsource(__import__("mal2026.official_decoder_aihub_pretrain", fromlist=["GenerativeTrainer"]).run_training)
        self.assertIn("GenerativeTrainer", source)
        self.assertIn("_distributed_generative_metrics", source)
        self.assertNotIn("token_accuracy", source)


if __name__ == "__main__": unittest.main()
