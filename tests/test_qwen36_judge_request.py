"""Non-restricted contract checks for the local GGUF judge request."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("judge_feedback_candidates", ROOT / "scripts" / "judge_feedback_candidates.py")
assert SPEC and SPEC.loader
JUDGE = module_from_spec(SPEC)
SPEC.loader.exec_module(JUDGE)


def test_judge_request_disables_template_thinking_and_keeps_schema() -> None:
    body = JUDGE.judge_request_body(
        "qwen36-35b-a3b-q4_k_m",
        {"temperature": 0.0, "top_p": 1.0, "seed": 2026, "max_tokens": 384},
        {"chat_template_kwargs": {"enable_thinking": False}},
        "synthetic contract check",
    )
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["type"] == "json_object"
    assert body["response_format"]["schema"] == JUDGE.judge_response_schema()
