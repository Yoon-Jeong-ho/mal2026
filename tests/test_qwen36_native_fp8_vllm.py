"""GPU-free vLLM 0.25.1 request-envelope regression checks."""
import copy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load(name: str, filename: str):
    spec = spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYNTHETIC = load("native_fp8_synthetic", "preflight_qwen36_native_fp8_vllm_synthetic.py")
RUNNER = load("native_fp8_runner", "run_qwen36_native_fp8_vllm_v1.py")


def expected_schema_body(body: dict, schema_name: str) -> None:
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": body["response_format"]["json_schema"]["schema"],
        },
    }
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["schema_version"]["const"] == schema_name
    assert "allOf" not in schema


def test_synthetic_uses_vllm_json_schema_not_unconstrained_json_object() -> None:
    SYNTHETIC.WIRE.SCHEMA = "mal2026-qwen36-native-fp8-vllm-v1"
    inherited = SYNTHETIC.WIRE.body("fixed synthetic control")
    body = SYNTHETIC.vllm_json_schema_body("fixed synthetic control", "mal2026-qwen36-native-fp8-vllm-v1")
    expected_schema_body(body, "mal2026-qwen36-native-fp8-vllm-v1")
    expected = copy.deepcopy(inherited["response_format"]["schema"])
    expected.pop("allOf")
    assert body["response_format"]["json_schema"]["schema"] == expected


def test_train_only_adapter_preserves_the_frozen_schema_and_no_thinking_field() -> None:
    RUNNER.V5.BASE.SCHEMA = RUNNER.SCHEMA
    inherited = {
        "model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 2026072008,
        "max_tokens": 192,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "fixed synthetic control"}],
        "response_format": {"type": "json_object", "schema": RUNNER.V5.BASE.score_schema()},
    }
    body = RUNNER.vllm_json_schema_body(inherited)
    expected_schema_body(body, RUNNER.SCHEMA)


def test_grammar_projection_does_not_relax_the_fail_closed_semantic_normalizer() -> None:
    RUNNER.V5.WIRE.SCHEMA = RUNNER.SCHEMA
    invalid = {
        "schema_version": RUNNER.SCHEMA,
        "verdict": "scored",
        "scores": {axis: 3 for axis in RUNNER.V5.WIRE.AXES},
        "hard_gates": {"content": False, "organization": True, "expression": True},
    }
    assert RUNNER.V5.WIRE.normalize(invalid)[1] == "semantic_scored_with_failed_gate"


def test_native_config_preflights_all_inherited_prompt_and_control_fields() -> None:
    config = RUNNER.config(3)
    protocol = config["protocol"]
    assert protocol["controls"] == {
        "duplicate_identity": True,
        "invalid_evidence": True,
        "padded_verbosity": True,
        "repeats": 5,
    }
    assert len(protocol["prompt_layouts"]) == len(protocol["rubric_permutations"]) == 5


def test_tp4_runtime_lane_is_a_single_batched_server_on_only_project_gpus() -> None:
    import json
    config = json.loads((ROOT / "configs/qwen36_native_fp8_vllm.tp4.v1.json").read_text(encoding="utf-8"))
    runtime = config["runtime"]
    assert runtime["physical_gpus"] == [0, 1, 2, 3]
    assert (runtime["tensor_parallel_size"], runtime["data_parallel_size"]) == (4, 1)
    assert runtime["max_num_seqs"] == 64 and runtime["max_num_batched_tokens"] == 32768
    launcher = (ROOT / "scripts/run_qwen36_native_fp8_vllm_tp4_v1.sh").read_text(encoding="utf-8")
    assert "--tensor-parallel-size 4" in launcher
    assert "--max-num-seqs 64" in launcher
    assert "--max-num-batched-tokens 32768" in launcher


def test_tp4_eager_variant_changes_only_cuda_graph_startup_policy() -> None:
    import json
    base = json.loads((ROOT / "configs/qwen36_native_fp8_vllm.tp4.v1.json").read_text(encoding="utf-8"))
    eager = json.loads((ROOT / "configs/qwen36_native_fp8_vllm.tp4.eager.v2.json").read_text(encoding="utf-8"))
    for field in ("physical_gpus", "topology", "tensor_parallel_size", "data_parallel_size", "context_size", "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization", "gdn_prefill_backend"):
        assert eager["runtime"][field] == base["runtime"][field]
    assert eager["runtime"]["enforce_eager"] is True


def test_synthetic_tp4_gate_submits_independent_controls_to_vllm_concurrently() -> None:
    source = (ROOT / "scripts/preflight_qwen36_native_fp8_vllm_synthetic.py").read_text(encoding="utf-8")
    assert "ThreadPoolExecutor" in source
    assert "pool.map(invoke, jobs)" in source
    assert '"client_concurrency"' in source
    assert '"wall_total"' in source
