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
    def test_vllm_entrypoint_import_is_safe_for_spawned_workers(self):
        """Importing the CLI module must not start the heavyweight evaluator."""
        import importlib.util
        import sys
        import types
        from pathlib import Path

        calls = []
        fake_runtime = types.ModuleType("mal2026.standard_decoder_vllm")

        class FakeConfig:
            @classmethod
            def from_json(cls, path):
                calls.append(("config", path))
                return "parsed-config"

        def fake_evaluate(config):
            calls.append(("evaluate", config))

        fake_runtime.VLLMEvalConfig = FakeConfig
        fake_runtime.run_vllm_evaluation = fake_evaluate
        original_runtime = sys.modules.get("mal2026.standard_decoder_vllm")
        module_name = "mal2026_test_vllm_entrypoint"
        try:
            sys.modules["mal2026.standard_decoder_vllm"] = fake_runtime
            spec = importlib.util.spec_from_file_location(
                module_name, Path("scripts/evaluate_standard_decoder_vllm.py")
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            entrypoint = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = entrypoint
            spec.loader.exec_module(entrypoint)
            self.assertEqual([], calls, "module import must not invoke vLLM")

            entrypoint.main(["--config", "/tmp/aggregate-only-config.json"])
            self.assertEqual(
                [("config", Path("/tmp/aggregate-only-config.json")), ("evaluate", "parsed-config")],
                calls,
            )
        finally:
            sys.modules.pop(module_name, None)
            if original_runtime is None:
                sys.modules.pop("mal2026.standard_decoder_vllm", None)
            else:
                sys.modules["mal2026.standard_decoder_vllm"] = original_runtime

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

    def test_standard_sft_disables_unused_parameter_search(self):
        from pathlib import Path
        source = Path("src/mal2026/standard_decoder_train.py").read_text(encoding="utf-8")
        self.assertIn("ddp_find_unused_parameters=False", source)

    def test_vllm_compatibility_contracts_are_required_and_forwarded(self):
        import json
        import tempfile
        from pathlib import Path
        from mal2026.standard_decoder_vllm import VLLMEvalConfig

        config = VLLMEvalConfig(
            run_id="eager-contract", mode="direct", model_path="/model", model_revision="a" * 40,
            adapter_path="/adapter", source="selection_dev", prepared_manifest="/manifest",
            validation_sha256="", output_dir="/out", enforce_eager=False,
        )
        self.assertTrue(VLLMEvalConfig(
            run_id="eager-default", mode="direct", model_path="/model", model_revision="a" * 40,
            adapter_path="/adapter", source="selection_dev", prepared_manifest="/manifest",
            validation_sha256="", output_dir="/out",
        ).enforce_eager)
        with self.assertRaises(StandardDecoderContractError):
            config.validate()
        flashinfer_config = VLLMEvalConfig(
            run_id="flashinfer-contract", mode="direct", model_path="/model", model_revision="a" * 40,
            adapter_path="/adapter", source="selection_dev", prepared_manifest="/manifest",
            validation_sha256="", output_dir="/out", disable_flashinfer_sampler=False,
        )
        self.assertTrue(VLLMEvalConfig(
            run_id="flashinfer-default", mode="direct", model_path="/model", model_revision="a" * 40,
            adapter_path="/adapter", source="selection_dev", prepared_manifest="/manifest",
            validation_sha256="", output_dir="/out",
        ).disable_flashinfer_sampler)
        with self.assertRaises(StandardDecoderContractError):
            flashinfer_config.validate()
        lora_rank_config = VLLMEvalConfig(
            run_id="lora-rank-contract", mode="direct", model_path="/model", model_revision="a" * 40,
            adapter_path="/adapter", source="selection_dev", prepared_manifest="/manifest",
            validation_sha256="", output_dir="/out", max_lora_rank=16,
        )
        self.assertEqual(32, VLLMEvalConfig(
            run_id="lora-rank-default", mode="direct", model_path="/model", model_revision="a" * 40,
            adapter_path="/adapter", source="selection_dev", prepared_manifest="/manifest",
            validation_sha256="", output_dir="/out",
        ).max_lora_rank)
        with self.assertRaises(StandardDecoderContractError):
            lora_rank_config.validate()
        with tempfile.TemporaryDirectory() as directory:
            false_path = Path(directory) / "false-eager.json"
            false_path.write_text(json.dumps({field: getattr(config, field) for field in config.__dataclass_fields__}), encoding="utf-8")
            with self.assertRaises(StandardDecoderContractError):
                VLLMEvalConfig.from_json(false_path)
            false_flashinfer_path = Path(directory) / "false-flashinfer.json"
            false_flashinfer_path.write_text(json.dumps({field: getattr(flashinfer_config, field) for field in flashinfer_config.__dataclass_fields__}), encoding="utf-8")
            with self.assertRaises(StandardDecoderContractError):
                VLLMEvalConfig.from_json(false_flashinfer_path)
            invalid_lora_rank_path = Path(directory) / "invalid-lora-rank.json"
            invalid_lora_rank_path.write_text(json.dumps({field: getattr(lora_rank_config, field) for field in lora_rank_config.__dataclass_fields__}), encoding="utf-8")
            with self.assertRaises(StandardDecoderContractError):
                VLLMEvalConfig.from_json(invalid_lora_rank_path)
            path = Path(directory) / "missing-lora-rank.json"
            missing_lora_rank = {field: getattr(VLLMEvalConfig(
                run_id="missing-flashinfer", mode="direct", model_path="/model", model_revision="a" * 40,
                adapter_path="/adapter", source="selection_dev", prepared_manifest="/manifest",
                validation_sha256="", output_dir="/out",
            ), field) for field in VLLMEvalConfig.__dataclass_fields__}
            missing_lora_rank.pop("max_lora_rank")
            path.write_text(json.dumps(missing_lora_rank), encoding="utf-8")
            with self.assertRaises(StandardDecoderContractError):
                VLLMEvalConfig.from_json(path)
        source = Path("src/mal2026/standard_decoder_vllm.py").read_text(encoding="utf-8")
        self.assertIn("enforce_eager=config.enforce_eager", source)
        self.assertIn("max_lora_rank=config.max_lora_rank", source)
        self.assertIn('"max_lora_rank": config.max_lora_rank', source)
        self.assertLess(source.index("adapter_config_path = adapter / \"adapter_config.json\""), source.index("from vllm import LLM"))
        self.assertLess(source.index("_configure_flashinfer_sampler_environment(config)"), source.index("from vllm import LLM"))
        matrix = Path("scripts/run_standard_experiment_matrix.sh").read_text(encoding="utf-8")
        self.assertIn('"enforce_eager":True', matrix)
        self.assertIn('"disable_flashinfer_sampler":True', matrix)
        self.assertIn('"max_lora_rank":32', matrix)

    def test_vllm_flashinfer_environment_requires_native_sampler_before_import(self):
        import os
        from mal2026.standard_decoder_vllm import (
            VLLMEvalConfig, _FLASHINFER_SAMPLER_ENV, _configure_flashinfer_sampler_environment,
        )

        config = VLLMEvalConfig(
            run_id="flashinfer-env", mode="direct", model_path="/model", model_revision="a" * 40,
            adapter_path="/adapter", source="selection_dev", prepared_manifest="/manifest",
            validation_sha256="", output_dir="/out",
        )
        previous = os.environ.get(_FLASHINFER_SAMPLER_ENV)
        try:
            os.environ.pop(_FLASHINFER_SAMPLER_ENV, None)
            _configure_flashinfer_sampler_environment(config)
            self.assertEqual("0", os.environ[_FLASHINFER_SAMPLER_ENV])
            os.environ[_FLASHINFER_SAMPLER_ENV] = "1"
            with self.assertRaises(StandardDecoderContractError):
                _configure_flashinfer_sampler_environment(config)
        finally:
            if previous is None:
                os.environ.pop(_FLASHINFER_SAMPLER_ENV, None)
            else:
                os.environ[_FLASHINFER_SAMPLER_ENV] = previous

    def test_vllm_preflight_rejects_invalid_or_oversized_adapter_rank_before_import(self):
        import json
        import tempfile
        from pathlib import Path
        from mal2026 import standard_decoder_vllm as module

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "data" / "manifests" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")
            adapter = root / "outputs" / "standard-runs" / "selection-run" / "checkpoint-12"
            adapter.mkdir(parents=True)
            run_root = adapter.parent
            (run_root / "standard_training_complete.json").write_text(json.dumps({
                "status": "completed", "phase": "selection", "mode": "direct",
                "model_revision": "a" * 40, "selection_candidate_steps": [12],
            }), encoding="utf-8")
            old_root, old_manifest = module.ROOT, module.DEFAULT_MANIFEST
            module.ROOT, module.DEFAULT_MANIFEST = root, manifest
            try:
                config = module.VLLMEvalConfig(
                    run_id="adapter-rank-preflight", mode="direct", model_path="/model", model_revision="a" * 40,
                    adapter_path=str(adapter), source="selection_dev", prepared_manifest=str(manifest),
                    validation_sha256="", output_dir=str(root / "outputs" / "standard-evals" / "eval"),
                    max_model_len=2048, max_new_tokens=256,
                )
                for rank in (None, True, 0, -1, 32.0, "32", 33):
                    with self.subTest(rank=rank):
                        (adapter / "adapter_config.json").write_text(json.dumps({"r": rank}), encoding="utf-8")
                        with self.assertRaises(StandardDecoderContractError):
                            config.validate()
                (adapter / "adapter_config.json").write_text(json.dumps({"r": 32}), encoding="utf-8")
                self.assertEqual(12, config.validate())
            finally:
                module.ROOT, module.DEFAULT_MANIFEST = old_root, old_manifest

    def test_vllm_template_exactly_matches_the_frozen_config_schema(self):
        import json
        from pathlib import Path
        from mal2026.standard_decoder_vllm import VLLMEvalConfig

        template = json.loads(Path("configs/standard-decoder-vllm-eval.template.json").read_text(encoding="utf-8"))
        self.assertEqual(set(VLLMEvalConfig.__dataclass_fields__), set(template))
        self.assertIs(template["enforce_eager"], True)
        self.assertIs(template["disable_flashinfer_sampler"], True)
        self.assertEqual(32, template["max_lora_rank"])

    def test_aggregate_metrics_contains_no_row_text(self):
        target = row().score
        result = aggregate_metrics([target], [target], [True])
        self.assertEqual(0, result["primary_macro_mae"])
        self.assertNotIn("prompt", str(result))
        self.assertNotIn("private", str(result))


if __name__ == "__main__":
    unittest.main()

class StandardDecoderWandbTests(unittest.TestCase):
    def test_wandb_routing_is_explicit_and_clears_stale_entity(self):
        import os
        from mal2026.standard_decoder_train import StandardSFTConfig, _configure_wandb
        config = StandardSFTConfig(
            run_id="decoder-run", phase="selection", mode="direct", model_path="/m", tokenizer_path="/t",
            model_revision="a" * 40, tokenizer_revision="a" * 40, prepared_manifest="/manifest", output_dir="/out",
            wandb_project="expected-project", wandb_entity=None,
        )
        old = dict(os.environ)
        try:
            os.environ["WANDB_ENTITY"] = "stale"
            _configure_wandb(config)
            self.assertEqual("false", os.environ["WANDB_LOG_MODEL"])
            self.assertEqual("expected-project", os.environ["WANDB_PROJECT"])
            self.assertEqual("decoder-run", os.environ["WANDB_RUN_NAME"])
            self.assertNotIn("WANDB_ENTITY", os.environ)
        finally:
            os.environ.clear(); os.environ.update(old)


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

class StandardDecoderHealthTests(unittest.TestCase):
    class State:
        def __init__(self, history, best_metric=None):
            self.log_history = history
            self.best_metric = best_metric

    def test_selection_health_accepts_finite_standard_trainer_metrics(self):
        from mal2026.standard_decoder_train import _validate_trainer_health
        _validate_trainer_health("selection", self.State([
            {"loss": 1.2, "grad_norm": 0.5}, {"eval_loss": 0.8},
        ], best_metric=0.8), {"train_loss": 1.0})

    def test_health_rejects_nonfinite_loss_grad_or_best_metric(self):
        from mal2026.standard_decoder_train import _validate_trainer_health
        cases = [
            ("selection", self.State([{"loss": 1.0}, {"eval_loss": float("nan")}], 0.5), {"train_loss": 1.0}),
            ("selection", self.State([{"loss": 1.0, "grad_norm": float("inf")}, {"eval_loss": 0.5}], 0.5), {"train_loss": 1.0}),
            ("selection", self.State([{"loss": 1.0}, {"eval_loss": 0.5}], float("nan")), {"train_loss": 1.0}),
            ("refit", self.State([{"loss": 1.0}], None), {"train_loss": float("inf")}),
            ("refit", self.State([{"loss": 1.0}], None), {"train_loss": 1.0, "train_runtime": float("nan")}),
        ]
        for phase, state, metrics in cases:
            with self.subTest(phase=phase, metrics=metrics):
                with self.assertRaises(StandardDecoderContractError):
                    _validate_trainer_health(phase, state, metrics)

class StandardDecoderDistributedLifecycleTests(unittest.TestCase):
    def test_rank_zero_outcome_is_broadcast_and_barriered(self):
        from mal2026.standard_decoder_train import _synchronize_rank_zero_success

        class Flag:
            def __init__(self, value): self.value = value
            def item(self): return self.value

        class Distributed:
            def __init__(self): self.barrier_called = False
            def is_available(self): return True
            def is_initialized(self): return True
            def broadcast(self, flag, src):
                self.source = src
                flag.value = 0  # Simulate rank zero reporting a failed health gate.
            def barrier(self): self.barrier_called = True

        class CUDA:
            def is_available(self): return False

        class Torch:
            int32 = object()
            cuda = CUDA()
            distributed = Distributed()
            def device(self, name, index=None): return (name, index)
            def tensor(self, value, **_): return Flag(value[0])

        torch = Torch()
        self.assertFalse(_synchronize_rank_zero_success(False, True, torch))
        self.assertEqual(0, torch.distributed.source)
        self.assertTrue(torch.distributed.barrier_called)

    def test_standard_sft_persists_only_on_world_process_zero(self):
        from pathlib import Path
        source = Path("src/mal2026/standard_decoder_train.py").read_text(encoding="utf-8")
        self.assertIn("is_world_process_zero = trainer.is_world_process_zero()", source)
        self.assertIn("if is_world_process_zero:\n        try:\n            assert serializable_train_metrics", source)
        self.assertIn("_write_completion_atomic", source)
