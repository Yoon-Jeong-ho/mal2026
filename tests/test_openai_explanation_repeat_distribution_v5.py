import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("repeat_v5", Path(__file__).parents[1] / "scripts/run_openai_explanation_repeat_distribution_v5.py")
V5 = importlib.util.module_from_spec(SPEC); assert SPEC and SPEC.loader; SPEC.loader.exec_module(V5)


def test_v5_binds_one_slot_context_and_transport_only_retry() -> None:
    cfg = V5.config()
    assert cfg["runtime"]["physical_gpus"] == [4, 5, 6, 7]
    assert cfg["runtime"]["parallel_requests_per_server"] == 1
    assert cfg["runtime"]["context_size"] == 4096
    assert cfg["request"]["max_tokens"] == 192
    assert cfg["retry"]["max_attempts"] == 2


def test_v5_body_uses_the_configured_schema_and_suppression() -> None:
    request = V5.body("qwen36-35b-a3b-q4_k_m", "fixed synthetic", 0.0, 1)
    assert request["max_tokens"] == 192
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["response_format"]["schema"]["properties"]["schema_version"]["const"] == V5.SCHEMA
