"""GPU-free contracts for the Qwen3-Embedding epoch checkpoint sweep."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.rlaif_qwen3_embedding import AXES, RATIONALE_SOURCE  # noqa: E402
from mal2026.rlaif_qwen3_epoch_sweep import (  # noqa: E402
    evaluation_config,
    expected_checkpoint_steps,
    training_config,
)


def _runner():
    path = ROOT / "scripts" / "run_rlaif_qwen3_embedding_epoch_sweep_v1.py"
    spec = importlib.util.spec_from_file_location("rlaif_qwen3_epoch_sweep_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_sweep_saves_exact_epoch_boundaries() -> None:
    assert expected_checkpoint_steps("gpu0_preflight") == {1: 1}
    assert expected_checkpoint_steps("full") == {epoch: epoch * 32 for epoch in range(1, 13)}
    full = training_config("full")
    assert full["arm"] == "qwen3_aihub_warmstart"
    assert full["source_key"] == RATIONALE_SOURCE
    assert full["score_fields"] == list(AXES)
    assert full["num_train_epochs"] == 12.0
    assert full["per_device_train_batch_size"] == 4
    assert full["gradient_accumulation_steps"] == 4
    assert "average" not in full["score_fields"]


def test_sweep_evaluates_all_checkpoints_on_fixed_population() -> None:
    preflight = evaluation_config("gpu0_preflight")
    full = evaluation_config("full")
    assert preflight["validation_record_limit"] == 4
    assert full["validation_record_limit"] == 400
    assert full["per_device_eval_batch_size"] == 8
    assert "full" in full["training_metadata_path"]


def test_runner_gates_gpu0_then_uses_ddp4_for_train_and_evaluation() -> None:
    runner = _runner()
    source = inspect.getsource(runner.main)
    assert "train-gpu0-preflight" in source and "evaluate-gpu0-preflight" in source
    assert source.count('"--nproc_per_node=4"') == 2
    assert "wait_idle([0, 1, 2, 3])" in source
