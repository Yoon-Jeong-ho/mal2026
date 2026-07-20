"""GPU-free contract checks for the 100-score DP4 collector."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load(name: str, filename: str):
    spec = spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = module_from_spec(spec); spec.loader.exec_module(module)
    return module


RUNNER = load("distribution100", "score_rationale_distribution_vllm_dp4.py")


def test_distribution100_config_is_single_endpoint_dp4_and_full_factorial() -> None:
    cfg = RUNNER.config(); runtime, factorial = cfg["runtime"], cfg["sampling"]["full_factorial"]
    assert runtime["physical_gpus"] == [0, 1, 2, 3]
    assert (runtime["tensor_parallel_size"], runtime["data_parallel_size"]) == (1, 4)
    assert runtime["client_max_inflight"] == runtime["max_num_seqs_per_dp_rank"] * 4 == 256
    assert factorial == {"prompt_layouts": 5, "rubric_permutations": 5, "seeds_per_cell": 4, "samples_per_candidate": 100}


def test_source_average_is_not_a_judge_axis() -> None:
    assert RUNNER.parse_scores({"content": 1.0, "organization": 2.0, "expression": 3.0, "average": 2.0}) == {"content": 1.0, "organization": 2.0, "expression": 3.0}


def test_one_candidate_has_exactly_100_unique_blinded_vllm_json_schema_requests() -> None:
    cfg = RUNNER.config()
    entry = {"custom_id": "synthetic-only-key", "candidate_number": 1, "scores": {axis: 3.0 for axis in RUNNER.AXES},
             "sentences": ["합성 통제 문장입니다."],
             "rationale": {"schema_version": "rationale-v3-sentence-id", **{axis: {"evidence_sentence_ids": [1], "diagnosis": "근거가 있다.", "next_step": "근거를 구체화하세요."} for axis in RUNNER.AXES}}}
    tasks = list(RUNNER.task_stream(cfg, "train", "qwen36-native-fp8-dist100-train-20260720-gpu0_smoke-999", "model", [entry], set()))
    assert len(tasks) == 100 == len({task["opaque_request_key"] for task in tasks})
    assert {task["sample_index"] for task in tasks} == set(range(100))
    assert all(task["body"]["temperature"] == 0.15 and task["body"]["top_p"] == 1.0 for task in tasks)
    response_format = tasks[0]["body"]["response_format"]
    assert response_format["type"] == "json_schema" and response_format["json_schema"]["name"] == RUNNER.SCHEMA
    assert "allOf" not in response_format["json_schema"]["schema"]


def test_essay_only_v2_prompt_has_no_reference_writing_score_or_score_conditioning() -> None:
    original_path, original_schema = RUNNER.CONFIG_PATH, RUNNER.SCHEMA
    try:
        RUNNER.CONFIG_PATH = ROOT / "configs/qwen36_native_fp8_vllm_distribution100_essay_only.v2.json"
        RUNNER.SCHEMA = "mal2026-qwen36-native-fp8-vllm-distribution100-essay-only-v2"
        cfg = RUNNER.config()
        entry = {"custom_id": "synthetic-only-key", "candidate_number": 1,
                 "sentences": ["합성 통제 문장입니다."],
                 "rationale": {"schema_version": "rationale-v3-sentence-id", **{axis: {"evidence_sentence_ids": [1], "diagnosis": "근거가 있다.", "next_step": "근거를 구체화하세요."} for axis in RUNNER.AXES}}}
        body = RUNNER.request_body(cfg, "model", entry, list(RUNNER.AXES), "rubric_then_essay", 2026072026)
        text = body["messages"][0]["content"]
        assert cfg["protocol"]["reference_score_in_prompt"] is False
        assert "frozen_score" not in text and "score conditioning" not in text
        assert "No human writing score, reference score, target label" in text
        assert "Do not assign a score to the student essay" in text
    finally:
        RUNNER.CONFIG_PATH, RUNNER.SCHEMA = original_path, original_schema


def test_essay_only_v3_has_five_prompt_types_with_ten_repeats_each() -> None:
    original_path, original_schema, original_token_count = RUNNER.CONFIG_PATH, RUNNER.SCHEMA, RUNNER.token_count
    try:
        RUNNER.CONFIG_PATH = ROOT / "configs/qwen36_native_fp8_vllm_essay_only_prompt5x10.v3.json"
        RUNNER.SCHEMA = "mal2026-qwen36-native-fp8-vllm-essay-only-prompt5x10-v3"
        cfg = RUNNER.config()
        entry = {"custom_id": "synthetic-only-key", "candidate_number": 1,
                 "sentences": ["합성 통제 문장입니다."],
                 "rationale": {"schema_version": "rationale-v3-sentence-id", **{axis: {"evidence_sentence_ids": [1], "diagnosis": "근거가 있다.", "next_step": "근거를 구체화하세요."} for axis in RUNNER.AXES}}}
        tasks = list(RUNNER.task_stream(cfg, "train", "qwen36-native-fp8-essay-only-prompt5x10-v3-train-20260720-gpu0_smoke-999", "model", [entry], set()))
        assert cfg["sampling"]["full_factorial"] == {"prompt_types": 5, "repeats_per_prompt_type": 10, "samples_per_candidate": 50}
        assert len(tasks) == 50 == len({task["opaque_request_key"] for task in tasks})
        assert {task["prompt_type_id"] for task in tasks} == {item["id"] for item in cfg["protocol"]["prompt_types"]}
        assert all(sum(task["prompt_type_id"] == prompt_type["id"] for task in tasks) == 10 for prompt_type in cfg["protocol"]["prompt_types"])
        assert all("frozen_score" not in task["body"]["messages"][0]["content"] for task in tasks)
        assert any(task["prompt_type_id"] == "communication_quality" and "natural Korean" in task["body"]["messages"][0]["content"] for task in tasks)
        RUNNER.token_count = lambda endpoint, content: 11
        assert RUNNER.prompt_budget("http://synthetic", cfg, "model", [entry]) == {"min_prompt_tokens": 11, "max_prompt_tokens": 11, "max_tokens": 192, "max_model_len": 4096}
    finally:
        RUNNER.CONFIG_PATH, RUNNER.SCHEMA, RUNNER.token_count = original_path, original_schema, original_token_count


def test_essay_only_v4_requires_scores_and_has_no_semantic_abstention_path() -> None:
    original_path, original_schema, original_token_count = RUNNER.CONFIG_PATH, RUNNER.SCHEMA, RUNNER.token_count
    try:
        RUNNER.CONFIG_PATH = ROOT / "configs/qwen36_native_fp8_vllm_essay_only_score5x10.v4.json"
        RUNNER.SCHEMA = "mal2026-qwen36-native-fp8-vllm-essay-only-score5x10-v4"
        cfg = RUNNER.config()
        entry = {"custom_id": "synthetic-only-key", "candidate_number": 1,
                 "sentences": ["합성 통제 문장입니다."],
                 "rationale": {"schema_version": "rationale-v3-sentence-id", **{axis: {"evidence_sentence_ids": [1], "diagnosis": "근거가 있다.", "next_step": "근거를 구체화하세요."} for axis in RUNNER.AXES}}}
        task = next(RUNNER.task_stream(cfg, "train", "qwen36-native-fp8-essay-only-score5x10-v4-train-20260720-gpu0_smoke-999", "model", [entry], set()))
        schema = task["body"]["response_format"]["json_schema"]["schema"]
        text = task["body"]["messages"][0]["content"]
        assert cfg["protocol"]["response_contract"] == "required_scores_only_v1"
        assert set(schema["required"]) == {"schema_version", "scores"}
        assert "verdict" not in schema["properties"] and "hard_gates" not in schema["properties"]
        assert "lowest appropriate score rather than withholding a score" in text
        assert "return verdict abstain" not in text
        assert RUNNER.normalize_judge_response({"schema_version": RUNNER.SCHEMA, "scores": {axis: 1 for axis in RUNNER.AXES}}, "required_scores_only_v1") == ({axis: 1 for axis in RUNNER.AXES}, None)
        RUNNER.token_count = lambda endpoint, content: 11
        assert RUNNER.prompt_budget("http://synthetic", cfg, "model", [entry])["max_prompt_tokens"] == 11
    finally:
        RUNNER.CONFIG_PATH, RUNNER.SCHEMA, RUNNER.token_count = original_path, original_schema, original_token_count


def test_rationale_only_v5_projects_out_candidate_score_ids_and_next_steps() -> None:
    original_path, original_schema = RUNNER.CONFIG_PATH, RUNNER.SCHEMA
    try:
        RUNNER.CONFIG_PATH = ROOT / "configs/qwen36_native_fp8_vllm_rationale_only_score5x10.v5.json"
        RUNNER.SCHEMA = "mal2026-qwen36-native-fp8-vllm-rationale-only-score5x10-v5"
        cfg = RUNNER.config()
        source_candidate = {"schema_version": "rationale-v3-sentence-id", **{axis: {"evidence_sentence_ids": [1], "diagnosis": f"{axis} 근거", "next_step": "수정 제안"} for axis in RUNNER.AXES}}
        projected = RUNNER.project_candidate(source_candidate, cfg)
        assert projected == {"schema_version": "rationale-only-v1", **{axis: {"rationale": f"{axis} 근거"} for axis in RUNNER.AXES}}
        entry = {"custom_id": "synthetic-only-key", "candidate_number": 1, "sentences": ["합성 통제 문장입니다."], "rationale": projected}
        body = RUNNER.request_body(cfg, "model", entry, list(RUNNER.AXES), "rubric_then_essay", 2026072050, cfg["protocol"]["prompt_types"][0]["review_emphasis"])
        text = body["messages"][0]["content"]
        assert '"next_step"' not in text and '"evidence_sentence_ids"' not in text and '"score"' not in text
        assert "no candidate writing score, sentence ID, or improvement proposal" in text
        assert cfg["protocol"]["response_contract"] == "required_scores_only_v1"
    finally:
        RUNNER.CONFIG_PATH, RUNNER.SCHEMA = original_path, original_schema


def test_rationale_only_v6_only_changes_runtime_throughput_controls() -> None:
    original_path, original_schema = RUNNER.CONFIG_PATH, RUNNER.SCHEMA
    try:
        RUNNER.CONFIG_PATH = ROOT / "configs/qwen36_native_fp8_vllm_rationale_only_score5x10.v6.json"
        RUNNER.SCHEMA = "mal2026-qwen36-native-fp8-vllm-rationale-only-score5x10-v6"
        cfg = RUNNER.config()
        runtime = cfg["runtime"]
        assert runtime["enforce_eager"] is False
        assert runtime["max_num_seqs_per_dp_rank"] == 192
        assert runtime["max_num_batched_tokens"] == 65536
        assert runtime["client_max_inflight"] == 768
        assert cfg["protocol"]["candidate_projection"] == "diagnosis_only_rationale_v1"
        assert cfg["protocol"]["response_contract"] == "required_scores_only_v1"
    finally:
        RUNNER.CONFIG_PATH, RUNNER.SCHEMA = original_path, original_schema


def test_validation_artifact_derivation_never_opens_train_source_text() -> None:
    source = (ROOT / "scripts" / "derive_validation_only_candidates.py").read_text(encoding="utf-8")
    assert "eval/train.jsonl" not in source
    assert '"train_source_text_opened": 0' in source
    assert '"split": "validation"' in source


def test_full_launcher_keeps_validation_after_train_and_uses_one_dp4_endpoint() -> None:
    source = (ROOT / "scripts" / "run_qwen36_native_fp8_vllm_distribution100_full.sh").read_text(encoding="utf-8")
    assert "--data-parallel-size 4" in source and "--tensor-parallel-size 1" in source
    assert source.index('execute --split train') < source.index('prepare --split validation') < source.index('execute --split validation')


def test_essay_only_full_launcher_uses_separate_score_blind_lineage() -> None:
    source = (ROOT / "scripts" / "run_qwen36_native_fp8_vllm_distribution100_essay_only_v2_full.sh").read_text(encoding="utf-8")
    assert "MAL2026_DIST100_CONFIG" in source and "essay-only-v2" in source
    assert "--data-parallel-size 4" in source and "--tensor-parallel-size 1" in source
    assert "judge_runs_essay_only_v2" in source and "frozen_validation_judge_runs_essay_only_v2" in source
