"""GPU-free contracts for the deterministic rationale-judge validity check."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).parents[1]


def load():
    spec = spec_from_file_location("contrastive_validity", ROOT / "scripts" / "evaluate_rationale_judge_contrastive_v1.py")
    assert spec and spec.loader
    module = module_from_spec(spec); spec.loader.exec_module(module)
    return module


CHECK = load()


def test_controls_are_rationale_only_and_all_four_conditions_are_independent() -> None:
    original = {"schema_version": "rationale-only-v1", **{axis: {"rationale": f"{axis} synthetic diagnosis"} for axis in CHECK.AXES}}
    record = {"base_key": "synthetic-key", "candidate_number": 1, "sentences": ["synthetic essay"], "foreign_sentences": ["different synthetic essay"], "rationale": original}
    result = CHECK.variants([record])
    assert [value["condition"] for value in result] == list(CHECK.CONDITIONS)
    assert all("score" not in json.dumps(value["rationale"], ensure_ascii=False) for value in result)
    rotated = next(value["rationale"] for value in result if value["condition"] == "axis_rotation")
    assert rotated["content"]["rationale"] == original["organization"]["rationale"]
    assert next(value["sentences"] for value in result if value["condition"] == "cross_essay") == record["foreign_sentences"]


def test_summary_requires_complete_unique_matched_controls_and_applies_predeclared_thresholds() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory)
        rows = []
        scores = {"original": 5, "generic": 1, "axis_rotation": 2, "cross_essay": 3}
        for condition, value in scores.items():
            rows.append({"opaque_request_key": f"synthetic-{condition}", "opaque_base_key": "synthetic-base", "prompt_type_id": "synthetic-prompt", "sampling_seed": 1,
                         "condition": condition, "scored": True, "schema_valid": True, "abstain": False, "failure_category": None,
                         "scores": {axis: value for axis in CHECK.AXES}})
        (path / "score_observations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        report = CHECK.summarize(path, 4)
    assert report["status"] == "completed" and all(report["hard_gates"].values())
    assert report["validity_gate"]["passed"] is True
    assert all(report["transforms"][condition]["all_axes_pass"] for condition in CHECK.CONDITIONS[1:])


def test_launcher_uses_gpu0_then_one_dp4_endpoint_without_eager_mode() -> None:
    source = (ROOT / "scripts" / "run_qwen36_native_fp8_vllm_contrastive_validity_v1.sh").read_text(encoding="utf-8")
    assert "MODE=\"$1\"" in source and "gpu0" in source and "full" in source
    assert "--data-parallel-size \"$DP\"" in source and "CUDA_VISIBLE_DEVICES=\"$CVD\"" in source
    assert "CVD=\"0,1,2,3\"; DP=4" in source
    assert "EAGER_ARGS=(); [[ \"$EAGER_FLAG\" == 1 ]] && EAGER_ARGS=(--enforce-eager)" in source
    assert "--enforce-eager \\\"${EAGER_ARGS" not in source
