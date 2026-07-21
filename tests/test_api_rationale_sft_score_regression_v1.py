"""GPU-free contracts for the API-rationale SFT and score-regression matrix."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import AXES, parse_rationale_output, rationale_object
from mal2026.api_score_regression import regression_metrics


def load_runner():
    spec = importlib.util.spec_from_file_location("api_rationale_runner", ROOT / "scripts" / "run_api_rationale_sft_score_regression_v1.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def test_rationale_parser_is_score_free_and_order_independent() -> None:
    diagnoses = {axis: f"synthetic {axis} rationale" for axis in AXES}
    value = rationale_object(diagnoses, AXES)
    reordered = {"expression": value["expression"], "schema_version": value["schema_version"], "content": value["content"], "organization": value["organization"]}
    assert parse_rationale_output(json.dumps(reordered), AXES) == diagnoses
    contaminated = {**value, "score": 5}
    assert parse_rationale_output(json.dumps(contaminated), AXES) is None


def test_decoder_matrix_has_twelve_training_runs_and_six_all_axis_judge_systems() -> None:
    assert len(RUNNER.MODELS) == 3 and RUNNER.TASKS == ("bundle", "content", "organization", "expression")
    full = [RUNNER.sft_config(base, task, "full") for base in RUNNER.MODELS for task in RUNNER.TASKS]
    assert len(full) == 12 and all(item["train_limit"] == 6000 and item["max_steps"] == -1 for item in full)
    assert all(item["trust_remote_code"] is False and item["max_length"] == 3072 for item in full)
    judged = [RUNNER.judge_config(base, kind) for base in RUNNER.MODELS for kind in ("bundle", "axis_triplet")]
    assert len(judged) == 6 and all(item["client_max_inflight"] == 768 and item["source"] == "validation" and item["run_id"].endswith("-002") for item in judged)


def test_bundle_generation_repair_is_uniform_and_lineage_bound() -> None:
    """A preserved -001 length gate cannot mix budgets in model comparison."""
    bundled = [RUNNER.generation_config(base, "bundle", "validation") for base in RUNNER.MODELS]
    assert all(item["run_id"].endswith("-003") and item["max_new_tokens"] == 512 for item in bundled)
    axis = RUNNER.generation_config("ax4_light", "content", "validation")
    assert axis["run_id"].endswith("-002") and axis["max_new_tokens"] == 192
    assert RUNNER.judge_config("phi4_mini", "bundle")["generation_dir"].endswith("bundle-validation-003")
    assert RUNNER.judge_config("phi4_mini", "axis_triplet")["generation_dir"].endswith("axis_triplet-validation-002")


def test_score_regression_matrix_has_two_backbones_and_three_input_conditions() -> None:
    configs = [RUNNER.regression_config(backbone, condition, "phi4_mini") for backbone in RUNNER.ENCODERS for condition in ("direct", "api_rationale", "decoder_rationale")]
    assert len(configs) == 6
    assert all(item["run_id"].endswith("-003") if item["backbone_key"] == "qwen25_7b" else item["run_id"].endswith("-004") for item in configs)
    assert all(item["decoder_generation_dir"] is None for item in configs if item["input_condition"] != "decoder_rationale")
    assert all(isinstance(item["decoder_generation_dir"], str) for item in configs if item["input_condition"] == "decoder_rationale")
    assert all(item["max_length"] == 3072 and item["seed"] == 2026072108 for item in configs)


def test_tie_aware_spearman_and_rmse_are_reported_per_axis_and_macro() -> None:
    truth = [[1.0, 1.0, 4.0], [2.0, 2.0, 3.0], [2.0, 3.0, 2.0], [4.0, 4.0, 1.0]]
    prediction = [[1.0, 1.0, 4.0], [2.0, 2.0, 3.0], [2.0, 3.0, 2.0], [4.0, 4.0, 1.0]]
    result = regression_metrics(truth, prediction)
    assert set(result) == {*AXES, "macro_rmse", "macro_spearman"}
    assert result["macro_rmse"] == 0.0 and result["macro_spearman"] == 1.0
