"""GPU-free contract tests for the v4 aggregate-only pilot."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("repeat_v4", ROOT / "scripts/run_openai_explanation_repeat_distribution_v4.py")
assert SPEC and SPEC.loader
PILOT = module_from_spec(SPEC); SPEC.loader.exec_module(PILOT)


def test_config_locks_train_only_five_repeats_and_gpu_scope() -> None:
    cfg = PILOT.config()
    assert cfg["selection"]["split"] == "train"
    assert cfg["selection"]["max_essays"] <= 96
    assert cfg["runtime"]["physical_gpus"] == [4, 5, 6, 7]
    assert cfg["runtime"]["parallel_requests_per_server"] == 4
    assert cfg["sampling"]["deterministic"]["repeats"] == 5
    assert cfg["sampling"]["dispersion"]["repeats"] == 5
    assert cfg["protocol"]["selection_artifact_permitted"] is False


def test_scored_schema_is_candidate_isolated_and_rejects_invalid_shape() -> None:
    cfg = PILOT.config()
    prompt = PILOT.payload_layout({axis: 3.0 for axis in PILOT.AXES}, ["synthetic"], {"schema_version": "rationale-v3-sentence-id"}, list(PILOT.AXES), cfg["protocol"]["prompt_layouts"][0])
    assert "peer candidate" in prompt
    assert PILOT.normalize({"schema_version": PILOT.SCHEMA, "verdict": "scored", "scores": {axis: 3 for axis in PILOT.AXES}, "hard_gates": {axis: True for axis in PILOT.AXES}}) == ({axis: 3 for axis in PILOT.AXES}, True, True)
    assert PILOT.normalize({"verdict": "scored"}) == (None, False, False)


def test_uncertainty_rule_withholds_ties() -> None:
    metrics = {"candidates": {str(n): {"rubrics": {"overall": {"n": 10, "median": 3.0, "iqr": 1.0}}} for n in (1, 2, 3)}}
    result = PILOT.comparison(metrics, {"all": True})
    assert result["decision"] == "tie_or_withhold"
    assert result["selection_artifact_constructed"] is False
