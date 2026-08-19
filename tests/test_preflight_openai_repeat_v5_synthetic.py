import importlib.util
from pathlib import Path

import jsonschema


SPEC = importlib.util.spec_from_file_location("v5_preflight", Path(__file__).parents[1] / "scripts/preflight_openai_repeat_v5_synthetic.py")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(PREFLIGHT)


def good(verdict="scored", gates=None):
    return {"schema_version": PREFLIGHT.SCHEMA, "verdict": verdict,
            "scores": {axis: 3 for axis in PREFLIGHT.AXES},
            "hard_gates": gates or {axis: True for axis in PREFLIGHT.AXES}}


def test_v5_schema_requires_every_rubric_field_and_consistent_abstention() -> None:
    assert PREFLIGHT.normalize(good()) == ({axis: 3 for axis in PREFLIGHT.AXES}, None)
    assert PREFLIGHT.normalize(good("abstain", {"content": False, "organization": True, "expression": True})) == (None, None)
    assert PREFLIGHT.normalize({"schema_version": PREFLIGHT.SCHEMA, "verdict": "scored"})[1] == "schema_shape"
    assert PREFLIGHT.normalize(good("scored", {"content": False, "organization": True, "expression": True}))[1] == "semantic_scored_with_failed_gate"
    assert PREFLIGHT.normalize(good("abstain"))[1] == "semantic_abstain_without_failed_gate"


def test_v5_schema_enforces_the_same_verdict_to_gate_contract_as_normalizer() -> None:
    schema = PREFLIGHT.score_schema()
    for mask in range(8):
        gates = {axis: bool(mask & (1 << index)) for index, axis in enumerate(PREFLIGHT.AXES)}
        all_true = all(gates.values())
        for verdict in ("scored", "abstain"):
            value = good(verdict, gates)
            expected_valid = (verdict == "scored") == all_true
            if expected_valid:
                jsonschema.validate(value, schema)
            else:
                try:
                    jsonschema.validate(value, schema)
                except jsonschema.ValidationError:
                    continue
                raise AssertionError(f"schema accepted invalid {verdict=} {gates=}")


def test_retry_policy_is_bounded_and_never_retries_schema_failures() -> None:
    assert PREFLIGHT.retry_contract_test()


def test_fixed_prompts_are_data_free_and_contract_is_pinned() -> None:
    assert "student essay" in PREFLIGHT.prompt("boundary").lower()
    assert "eval/" not in PREFLIGHT.prompt("boundary")
    assert PREFLIGHT.GPUS == (4, 5, 6, 7)
    assert PREFLIGHT.PARALLEL == 1
    assert PREFLIGHT.CONTEXT_SIZE // PREFLIGHT.PARALLEL > PREFLIGHT.MAX_TOKENS
