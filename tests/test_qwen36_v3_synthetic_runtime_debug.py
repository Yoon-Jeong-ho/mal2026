import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("runtime_debug", Path(__file__).parents[1] / "scripts/run_qwen36_v3_synthetic_runtime_debug.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(RUNTIME)


def test_synthetic_prompts_and_schemas_are_fixed_and_data_free() -> None:
    pointwise = RUNTIME.pointwise_prompt(False)
    independent = RUNTIME.independent_prompt("abstain")
    assert "student essay" not in pointwise.lower()
    assert "validation" not in pointwise.lower()
    assert "eval/" not in pointwise.lower()
    assert "unavailable" in independent
    assert RUNTIME.valid_throughput({"schema_version": "mal2026-synthetic-throughput-v1", "status": "ok", "slots": [1, 2, 3]})


def test_independent_aggregation_is_label_free_and_fail_closed() -> None:
    assert RUNTIME.aggregate_independent({"axis_checks": {"content": "eligible", "organization": "eligible", "expression": "eligible"}}) == "eligible"
    assert RUNTIME.aggregate_independent({"axis_checks": {"content": "ineligible", "organization": "eligible", "expression": "eligible"}}) == "ineligible"
    assert RUNTIME.aggregate_independent({"axis_checks": {"content": "eligible", "organization": "abstain", "expression": "eligible"}}) == "abstain"
