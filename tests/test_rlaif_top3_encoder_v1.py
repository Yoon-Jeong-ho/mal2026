"""GPU-free contracts for the independent top-three encoder protocol."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.rlaif_top3_encoder import (
    AXES,
    SELECTIONS,
    evaluation_config,
    generation_config,
    regression_config,
    selected_sources,
    three_axis_metrics,
)


def _runner():
    spec = importlib.util.spec_from_file_location("rlaif_top3_runner", ROOT / "scripts" / "run_rlaif_top3_encoder_v1.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exactly_three_complete_bundle_sources_are_fixed() -> None:
    assert selected_sources() == ("rank1_midm2_random1", "rank2_ax4_random1", "rank3_ax4_all5")
    assert len(SELECTIONS) == 3
    assert [SELECTIONS[source]["rank"] for source in selected_sources()] == [1, 2, 3]
    assert [SELECTIONS[source]["frozen_macro"] for source in selected_sources()] == [4.189100, 4.187033, 4.184067]
    assert all(SELECTIONS[source]["arm"] in {"random1", "all5"} for source in selected_sources())


def test_generation_sources_are_independent_and_full_bundle_shaped() -> None:
    paths = []
    for source in selected_sources():
        train = generation_config(source, "train", "full")
        validation = generation_config(source, "validation", "full")
        assert train["record_limit"] == 2000 and validation["record_limit"] == 400
        assert train["max_new_tokens"] == validation["max_new_tokens"] == 512
        assert train["restricted_output_dir"] != validation["restricted_output_dir"]
        paths.extend((train["restricted_output_dir"], validation["restricted_output_dir"]))
    assert len(paths) == len(set(paths)) == 6


def test_encoder_targets_and_metrics_are_exactly_three_axes() -> None:
    for source in selected_sources():
        training = regression_config(source, "full")
        evaluation = evaluation_config(source)
        assert training["score_fields"] == list(AXES)
        assert "average" not in training
        assert training["decoder_train_generation_dir"] != training["decoder_validation_generation_dir"]
        assert evaluation["source_key"] == source
    metrics = three_axis_metrics([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    assert set(metrics) == {*AXES, "three_axis_macro_rmse", "three_axis_macro_spearman"}
    assert metrics["three_axis_macro_rmse"] == 0.0


def test_runner_uses_gpu0_preflight_then_tp4_and_ddp4() -> None:
    runner = _runner()
    assert runner.selected_sources() == selected_sources()
    assert runner.RUN_BASE.name == "20260725-001"
    assert "--tensor-parallel-size" in inspect.getsource(runner.generation_server)
    assert "--nproc_per_node=4" in inspect.getsource(runner.run_evaluation)
