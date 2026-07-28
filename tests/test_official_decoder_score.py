from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from mal2026.official_decoder_score import (
    ARCHITECTURES, DecoderScoreConfig, arm_names, canonical_targets,
    experiment_contract, generative_metrics, parse_generated, render_target,
    rank_results, trainable_state, WarmArtifact, _load_full_warmstate,
)


class OfficialDecoderScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DecoderScoreConfig.from_json(Path("configs/official_decoder_score_matrix.v1.json"), require_dependencies=False)

    def test_twelve_arms_cover_architecture_initialization_and_input(self) -> None:
        arms = arm_names()
        self.assertEqual(len(arms), 12)
        self.assertEqual(len(set(arms)), 12)
        self.assertTrue(all("average" not in arm for arm in arms))
        self.assertEqual(set(ARCHITECTURES), {"generative", "bounded_regression", "ordinal_cumulative"})

    def test_generative_output_is_exactly_three_integer_scores(self) -> None:
        self.assertEqual(len(canonical_targets()), 125)
        self.assertEqual(len(set(canonical_targets())), 125)
        target = render_target((1, 3, 5))
        self.assertEqual(target, '{"content":1,"organization":3,"expression":5}')
        self.assertEqual(parse_generated(target), (1, 3, 5))
        for invalid in (
            '{"content":1,"organization":3,"expression":5,"average":3}',
            '{"content":1,"organization":3,"expression":5,"rationale":"x"}',
            '설명 {"content":1,"organization":3,"expression":5}',
            '{"content":1.0,"organization":3,"expression":5}',
        ):
            self.assertIsNone(parse_generated(invalid))

    def test_generative_metrics_are_integer_primary(self) -> None:
        metrics = generative_metrics([[1, 2, 3], [5, 4, 3]], [[1, 3, 3], [4, 4, 3]])
        self.assertIn("macro_integer_rmse", metrics)
        self.assertEqual(metrics["strict_parse_rate"], 1.0)
        self.assertNotIn("average", json.dumps(metrics))

    def test_contract_forbids_average_and_validation_selection(self) -> None:
        contract = experiment_contract(self.config)
        self.assertEqual(contract["score_fields"], ["content", "organization", "expression"])
        self.assertIs(contract["average_target_used"], False)
        self.assertIn("train-internal", contract["selection_source"])
        self.assertIn("single final", contract["canonical_validation_use"])
        self.assertEqual(contract["generative_output_space"]["canonical_outputs"], 125)
        source = inspect.getsource(DecoderScoreConfig.validate_warm_artifact)
        self.assertIn("matched_architecture", inspect.getsource(__import__("mal2026.official_decoder_score", fromlist=["build_model"]).build_model))
        self.assertIn("average_target_used", source)

    def test_dedicated_head_state_is_required_and_generative_has_none(self) -> None:
        class TinyHead(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lora_adapter = torch.nn.Parameter(torch.ones(1))
                self.score_head = torch.nn.Linear(2, 3)

        class TinyGenerative(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lora_adapter = torch.nn.Parameter(torch.ones(1))

        head_state = trainable_state(TinyHead(), "bounded_regression")
        self.assertEqual({key for key in head_state if key.startswith("score_head.")}, {"score_head.weight", "score_head.bias"})
        self.assertFalse(any(key.startswith("score_head.") for key in trainable_state(TinyGenerative(), "generative")))

    def test_orchestration_is_gpu0_gated_then_ddp4_and_never_names_other_gpus(self) -> None:
        from scripts.orchestrate_official_decoder_score_matrix import command_plan
        plan = command_plan(Path("configs/official_decoder_score_matrix.v1.json"))
        self.assertEqual(len(plan), 37)  # AI-Hub adds one distributed integration preflight; 12 target arms keep smoke/full.
        self.assertTrue(all(stage["gpus"] in ([0], [0, 1, 2, 3]) for stage in plan))
        self.assertEqual(sum(stage["gpus"] == [0] for stage in plan), 18)
        self.assertEqual(sum(stage["gpus"] == [0, 1, 2, 3] for stage in plan), 19)
        aihub_production = [stage for stage in plan[:13] if stage["stage"] != "fsdp4_one_update_preflight"]
        for offset in (0, 4, 8):
            self.assertEqual([stage["gpus"] for stage in aihub_production[offset:offset+4]], [[0], [0], [0,1,2,3], [0,1,2,3]])
        source = Path("scripts/orchestrate_official_decoder_score_matrix.py").read_text()
        self.assertNotIn('"4,5,6,7"', source)
        self.assertIn('"fsdp4_one_update_preflight"', source)

    def test_orchestration_can_run_six_essay_arms_before_rationales_exist(self) -> None:
        from scripts.orchestrate_official_decoder_score_matrix import command_plan, target_arms
        config = Path("configs/official_decoder_score_matrix.public_spec_score_prompt.v1.json")
        aihub = Path("configs/official_decoder_aihub_integer_score_pretrain.public_spec_score_prompt.v1.json")
        essay = target_arms("essay")
        rationale = target_arms("rationale")
        self.assertEqual(len(essay), 6)
        self.assertEqual(len(rationale), 6)
        self.assertTrue(all(arm.endswith("__essay") for arm in essay))
        self.assertTrue(all(arm.endswith("__rationale") for arm in rationale))
        plan = command_plan(config, aihub, "essay")
        target = [stage for stage in plan if str(stage["stage"]).startswith("target_")]
        self.assertEqual(len(target), 12)
        self.assertEqual({stage["arm"] for stage in target}, set(essay))

    def test_essay_aggregate_resolves_selected_epoch_from_event_history(self) -> None:
        from scripts.orchestrate_official_decoder_score_matrix import _write_essay_bootstrap_aggregate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(self.config, output_root=str(root))
            arm = "generative__public__essay"
            output = root / arm
            output.mkdir()
            completion = {
                "schema_version": "mal2026-official-decoder-integer-score-completion-v1",
                "status": "completed", "run_id": config.run_id,
                "architecture": "generative", "initialization": "public", "input_view": "essay",
                "score_prompt_kind": config.score_prompt_kind, "average_target_used": False,
                "selection": {
                    "selected_epoch": 2,
                    "events": [
                        {"epoch": 1, "macro_integer_rmse": 0.8, "macro_integer_spearman": 0.2, "macro_continuous_rmse": 0.7},
                        {"epoch": 2, "macro_integer_rmse": 0.7, "macro_integer_spearman": 0.3, "macro_continuous_rmse": 0.6},
                    ],
                },
                "canonical_validation": {"use": "single_final_descriptive_evaluation_not_selection", "metrics": {
                    "macro_integer_rmse": 0.75, "macro_integer_spearman": 0.25, "macro_continuous_rmse": 0.65,
                }},
            }
            (output / "training_complete.json").write_text(json.dumps(completion), encoding="utf-8")
            aggregate_path = _write_essay_bootstrap_aggregate(config, [arm])
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            self.assertEqual(aggregate["selected_arm"], arm)
            self.assertEqual(aggregate["candidates"][0]["epoch"], 2)
            self.assertEqual(aggregate["candidates"][0]["macro_integer_rmse"], 0.7)

    def test_final_ranking_is_integer_primary(self) -> None:
        def result(name, integer_rmse, integer_spearman, continuous_rmse):
            return {
                "status": "completed", "phase": "target_refit", "architecture": name,
                "initialization": "public", "input_view": "essay",
                "canonical_validation": {"metrics": {
                    "macro_integer_rmse": integer_rmse, "macro_integer_spearman": integer_spearman,
                    "macro_continuous_rmse": continuous_rmse,
                }},
            }
        ranked = rank_results([
            result("continuous_bait", 0.7, 0.9, 0.1),
            result("integer_winner", 0.6, 0.1, 0.9),
            result("rho_tiebreak", 0.6, 0.2, 1.0),
        ])
        self.assertEqual([row["architecture"] for row in ranked], ["rho_tiebreak", "integer_winner", "continuous_bait"])

    def test_full_aihub_backbone_and_matched_head_load_before_lora(self) -> None:
        from safetensors.torch import save_file
        class TinyBackbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {
                "backbone.layer.weight": torch.full((2, 2), 2.0),
                "backbone.layer.bias": torch.full((2,), 3.0),
                "score_head.weight": torch.zeros(3, 2),
                "score_head.bias": torch.zeros(3),
            }
            save_file(source, str(root / "model.safetensors"))
            artifact = WarmArtifact("unused", "a"*64, str(root), "b"*64, "unused", "c"*64)
            base = TinyBackbone()
            count, head = _load_full_warmstate(base, artifact, "bounded_regression")
            self.assertEqual(count, 2)
            self.assertEqual(set(head), {"score_head.weight", "score_head.bias"})
            self.assertTrue(torch.equal(base.layer.weight, source["backbone.layer.weight"]))


if __name__ == "__main__":
    unittest.main()
