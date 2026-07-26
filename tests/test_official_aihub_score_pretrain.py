from __future__ import annotations

from decimal import Decimal
import inspect
import json
from pathlib import Path
import tempfile
import unittest

from mal2026.official_aihub_score_pretrain import (
    AXES,
    CANONICAL_MANIFEST,
    HEADS,
    PretrainConfig,
    decode_logits,
    downstream_target_contract,
    exported_tensor_contract,
    initialization_contract_sha256,
    official_half_up,
    ordinal_targets,
    project_three_axes,
    select_event,
)


class AverageSentinel(dict):
    def __getitem__(self, key):
        if key == "average":
            raise AssertionError("average was read")
        return super().__getitem__(key)


class AIHubIntegerScorePretrainTests(unittest.TestCase):
    def test_half_up_integer_projection_boundaries(self) -> None:
        self.assertEqual(
            [official_half_up(value) for value in ("1.49", "1.50", Decimal("2.50"), 4.49, 4.5, 5)],
            [1, 2, 3, 4, 5, 5],
        )
        with self.assertRaises(ValueError):
            official_half_up("5.01")

    def test_projection_never_reads_average_sentinel(self) -> None:
        scores = AverageSentinel(content="2.50", organization="3.49", expression="4.50", average="SENTINEL")
        self.assertEqual(project_three_axes(scores), (3, 3, 5))
        source = inspect.getsource(project_three_axes)
        self.assertNotIn('["average"]', source)
        self.assertNotIn("['average']", source)

    def test_two_head_shapes_and_integer_decoding(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        labels = torch.tensor([[1, 3, 5], [2, 4, 1]])
        self.assertEqual(ordinal_targets(labels).shape, (2, 3, 4))
        for head, width in zip(HEADS, (3, 12), strict=True):
            continuous, integers, violations = decode_logits(torch.zeros(2, width), head)
            self.assertEqual(continuous.shape, (2, 3))
            self.assertEqual(integers.shape, (2, 3))
            self.assertEqual(integers.dtype, torch.int64)
            self.assertTrue(bool(((integers >= 1) & (integers <= 5)).all()))
            self.assertEqual(violations.shape[:2], (2, 3))

    def test_selection_rule_uses_integer_metrics_then_earlier_step(self) -> None:
        events = [
            {"global_step": 200, "macro_integer_rmse": 0.7, "macro_integer_spearman": 0.5, "macro_continuous_rmse": 0.6},
            {"global_step": 100, "macro_integer_rmse": 0.7, "macro_integer_spearman": 0.5, "macro_continuous_rmse": 0.6},
            {"global_step": 300, "macro_integer_rmse": 0.7, "macro_integer_spearman": 0.4, "macro_continuous_rmse": 0.5},
        ]
        self.assertEqual(select_event(events)["global_step"], 100)

    def test_config_is_three_axis_selection_only_and_historical_state_is_reference(self) -> None:
        config = PretrainConfig.from_json(Path("configs/official_aihub_integer_score_pretrain.v1.json"), require_dependencies=False)
        self.assertEqual(config.score_fields, AXES)
        self.assertIs(config.integer_target_used, True)
        self.assertEqual(config.target_projection, "official_half_up")
        self.assertIs(config.average_target_used, False)
        self.assertEqual(config.manifest_path, str(CANONICAL_MANIFEST))
        self.assertEqual(config.historical_reference_selected_step, 1900)
        self.assertEqual(config.historical_reference_classification, "reference_only_continuous_four_axis_not_loaded")
        self.assertEqual(config.training_method, "full_parameter")
        self.assertEqual(config.distributed_strategy, "fsdp_full_shard_auto_wrap")
        serialized = Path("configs/official_aihub_integer_score_pretrain.v1.json").read_text()
        self.assertNotIn("validation.jsonl", serialized)
        self.assertNotIn("warmstate_path", serialized)

    def test_selection_identity_survives_json_round_trip(self) -> None:
        config = PretrainConfig.from_json(Path("configs/official_aihub_integer_score_pretrain.v1.json"), require_dependencies=False)
        identity = config.identity("bounded_regression")
        self.assertEqual(json.loads(json.dumps(identity)), identity)
        self.assertIsInstance(identity["score_fields"], list)

    def test_downstream_contract_keys_are_explicit_at_every_artifact_layer(self) -> None:
        config = PretrainConfig.from_json(Path("configs/official_aihub_integer_score_pretrain.v1.json"), require_dependencies=False)
        self.assertEqual(downstream_target_contract(config), {
            "integer_target_used": True,
            "target_projection": "official_half_up",
            "score_fields": ["content", "organization", "expression"],
            "average_target_used": False,
        })
        orchestrator_source = Path("scripts/orchestrate_official_aihub_score_pretrain.py").read_text()
        for key in ("completion_path", "completion_sha256", "artifact_path", "artifact_sha256", "score_head_state_sha256"):
            self.assertIn(f'"{key}"', orchestrator_source)

    def test_required_arm_is_full_parameter_not_peft(self) -> None:
        source = Path("src/mal2026/official_aihub_score_pretrain.py").read_text()
        build = source[source.index("def build_model"):source.index("def _initial_head_sha256")]
        self.assertNotIn("get_peft_model", build)
        self.assertNotIn("LoraConfig", build)
        self.assertIn("all(parameter.requires_grad", build)
        self.assertIn("def gradient_checkpointing_enable", build)
        config = PretrainConfig.from_json(Path("configs/official_aihub_integer_score_pretrain.v1.json"), require_dependencies=False)
        digest = initialization_contract_sha256(config, "bounded_regression", "a" * 64)
        self.assertEqual(len(digest), 64)

    def test_runner_uses_fsdp4_and_gpu0_smoke(self) -> None:
        source = Path("scripts/orchestrate_official_aihub_score_pretrain.py").read_text()
        self.assertIn('"stage": "fsdp4_full_parameter"', source)
        self.assertIn('"gpus": [0]', source)
        self.assertIn('"gpus": [0, 1, 2, 3]', source)

    def test_export_contract_requires_full_backbone_and_matched_head(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("torch/safetensors unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({
                "backbone.layers.0.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
                "score_head.weight": torch.zeros(3, 2, dtype=torch.bfloat16),
            }, root / "model-00001.safetensors")
            save_file({"score_head.bias": torch.zeros(3, dtype=torch.bfloat16)}, root / "model-00002.safetensors")
            contract = exported_tensor_contract(root, "bounded_regression")
            self.assertEqual(contract["backbone_tensor_count"], 1)
            self.assertEqual(len(contract["score_head_state_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
