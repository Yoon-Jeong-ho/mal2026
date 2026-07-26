"""GPU-free contracts for the two-arm Qwen3-Embedding comparison."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

from safetensors import safe_open


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.rlaif_qwen3_embedding import (  # noqa: E402
    ARMS,
    AXES,
    RATIONALE_SOURCE,
    STANDARD_FIELDS,
    WARMSTART_STATE,
    evaluation_config,
    training_config,
    warmstart_provenance,
)


def _runner():
    path = ROOT / "scripts" / "run_rlaif_qwen3_embedding_comparison_v1.py"
    spec = importlib.util.spec_from_file_location("rlaif_qwen3_embedding_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_initializations_share_one_three_axis_protocol() -> None:
    assert ARMS == ("qwen3_base", "qwen3_aihub_warmstart")
    for arm in ARMS:
        preflight = training_config(arm, "gpu0_preflight")
        full = training_config(arm, "full")
        evaluation = evaluation_config(arm)
        assert preflight["score_fields"] == full["score_fields"] == list(AXES)
        assert preflight["source_key"] == full["source_key"] == RATIONALE_SOURCE
        assert full["per_device_train_batch_size"] == 4
        assert full["gradient_accumulation_steps"] == 4
        assert evaluation["per_device_eval_batch_size"] == 8
        assert "average" not in full["score_fields"]
    assert training_config("qwen3_base", "full")["initialization"] == "public_base"
    assert training_config("qwen3_aihub_warmstart", "full")["initialization"] == "aihub_48016_warmstart"


def test_aihub_warmstart_is_four_head_and_continues_only_first_three() -> None:
    provenance = warmstart_provenance()
    assert provenance["source_records"] == 48016
    assert provenance["loaded_score_fields"] == list(STANDARD_FIELDS)
    assert provenance["continued_score_fields"] == list(AXES)
    assert provenance["average_head_discarded_before_continuation"] is True
    with safe_open(WARMSTART_STATE, framework="pt", device="cpu") as handle:
        trainable = [name for name in handle.keys() if ".lora_A." in name or ".lora_B." in name or name.startswith("regression_head.")]
        assert len(trainable) == 506
        assert handle.get_slice("regression_head.weight").get_shape() == [4, 4096]
        assert handle.get_slice("regression_head.bias").get_shape() == [4]


def test_runner_gates_gpu0_then_uses_ddp4_without_average_target() -> None:
    runner = _runner()
    source = inspect.getsource(runner.main)
    assert "wait_idle([0])" in source
    assert '"--nproc_per_node=4"' in source
    assert "wait_idle([0, 1, 2, 3])" in source
    assert runner.ARMS == ARMS
    assert runner.AXES == AXES
