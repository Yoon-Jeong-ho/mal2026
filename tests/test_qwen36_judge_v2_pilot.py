"""Synthetic-only contract checks for the train-only v2 judge pilot."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("judge_feedback_candidates_v2", ROOT / "scripts" / "judge_feedback_candidates_v2.py")
assert SPEC and SPEC.loader
JUDGE = module_from_spec(SPEC)
SPEC.loader.exec_module(JUDGE)


def test_v2_config_is_train_only_gpu0_and_selection_disabled() -> None:
    cfg = JUDGE.config()
    assert cfg["selection"]["split"] == "train"
    assert cfg["selection"]["max_essays"] <= 128
    assert cfg["runtime"]["gpu_allowlist"] == [0]
    assert cfg["protocol"]["selection_artifact_permitted"] is False
    assert len(cfg["protocol"]["rubric_permutations"]) == 6
    assert len(cfg["protocol"]["factorial_label_position_cells"]) == 4


def test_v2_request_keeps_non_thinking_and_schema() -> None:
    cfg = JUDGE.config()
    body = JUDGE.request_body("qwen36-35b-a3b-q4_k_m", cfg["sampling"], 7, "synthetic only", JUDGE.pairwise_schema())
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["schema"] == JUDGE.pairwise_schema()


def test_train_candidate_manifest_count_is_checked_before_deserialization() -> None:
    cfg = JUDGE.config()
    with TemporaryDirectory() as temporary:
        source_dir = Path(temporary)
        parent_candidates = source_dir / "candidates.jsonl"
        parent_candidates.write_bytes(b"parent\n")
        source_map = source_dir / "source_map.jsonl"
        source_map.write_bytes(b"map\n")
        aggregate = source_dir / "validation_aggregate.json"
        aggregate.write_bytes(b"aggregate\n")
        source_manifest = {"status": "validated", "splits": {"train": 1, "validation": 1}, "candidates_per_essay": 3,
                           "candidates_sha256": JUDGE.sha256(parent_candidates), "source_map_sha256": JUDGE.sha256(source_map)}
        (source_dir / "manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
        candidate_path = source_dir / cfg["selection"]["required_candidate_artifact"]
        candidate_path.parent.mkdir(parents=True)
        candidate_path.write_bytes(b"{}\n{}\n{}\n")
        manifest_path = source_dir / cfg["selection"]["required_candidate_manifest"]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": "rationale-v3-train-only-candidate-artifact-v1",
                    "status": "completed", "batch_run_id": "synthetic-train-batch", "split": "train",
                    "candidate_file": candidate_path.name, "candidate_file_sha256": JUDGE.sha256(candidate_path), "row_count": 3,
                    "parent_manifest_sha256": JUDGE.sha256(source_dir / "manifest.json"),
                    "parent_validation_aggregate_sha256": JUDGE.sha256(aggregate),
                    "parent_candidate_file_sha256": JUDGE.sha256(parent_candidates),
                    "parent_source_map_sha256": JUDGE.sha256(source_map), "parent_candidate_schema": JUDGE.CANDIDATE_SCHEMA,
                    "input_candidate_counts": {"train": 3, "validation": 3}, "output_candidate_counts": {"train": 3, "validation": 0},
                    "input_source_counts": {"train": 1, "validation": 1}, "output_source_counts": {"train": 1, "validation": 0},
                    "proof": {"candidate_custom_id_duplicates": 0, "source_candidate_duplicates": {"train": 0, "validation": 0},
                              "train_validation_source_id_overlap": 0, "train_validation_candidate_key_overlap": 0,
                              "unmapped_or_mismatched_candidates": 0, "validation_rows_in_new_artifact": 0,
                              "validation_requests_constructed": 0, "validation_source_text_opened": 0}}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        assert JUDGE.train_candidate_artifact(source_dir, "synthetic-train-batch", cfg, source_manifest) == candidate_path
        parent_candidates.write_bytes(b"tampered-parent\n")
        try:
            JUDGE.train_candidate_artifact(source_dir, "synthetic-train-batch", cfg, source_manifest)
        except RuntimeError as exc:
            assert str(exc) == "train-only candidate artifact manifest failed validation"
        else:
            raise AssertionError("tampered parent candidate lineage was accepted")
        parent_candidates.write_bytes(b"parent\n")
        manifest["row_count"] = 4
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            JUDGE.train_candidate_artifact(source_dir, "synthetic-train-batch", cfg, source_manifest)
        except RuntimeError as exc:
            assert str(exc) == "train-only candidate artifact manifest failed validation"
        else:
            raise AssertionError("mismatched train-only record count was accepted")


def test_malformed_outputs_are_schema_failures_not_silent_abstentions() -> None:
    assert JUDGE.normalized_pointwise({"verdict": "eligible"}) is None
    assert JUDGE.normalized_pairwise({"verdict": "A"}) is None
    pointwise = {"schema_version": JUDGE.JUDGE_SCHEMA, "verdict": "eligible",
                 "hard_gates": {"content": True, "organization": True, "expression": True}, "reason": "x" * 301}
    pairwise = {"schema_version": JUDGE.JUDGE_SCHEMA, "verdict": "A",
                "hard_gates": {"A": {"content": True, "organization": True, "expression": True},
                               "B": {"content": True, "organization": True, "expression": True}}, "reason": "x" * 301}
    assert JUDGE.normalized_pointwise(pointwise) is None
    assert JUDGE.normalized_pairwise(pairwise) is None


def test_factorial_cells_balance_label_and_position_for_a_fixed_winner() -> None:
    cfg = JUDGE.config()
    requests = []
    responses = []
    # Two pointwise groups, two lanes, and two exact repeats all pass.
    for candidate in (1, 2):
        group = f"point-{candidate}"
        for lane in range(2):
            for repeat in range(2):
                key = f"p-{candidate}-{lane}-{repeat}"
                requests.append({"opaque_request_key": key, "opaque_logical_key": f"p-{candidate}-{lane}", "opaque_group_key": group,
                                 "kind": "pointwise", "repeat": repeat, "candidate_number": candidate, "lane": lane})
                responses.append({"opaque_request_key": key, "resolved_verdict": "eligible", "transport_or_schema_failure": False})
    cells = cfg["protocol"]["factorial_label_position_cells"]
    for lane in range(2):
        for cell_index, cell in enumerate(cells):
            for repeat in range(2):
                key = f"r-{lane}-{cell_index}-{repeat}"
                label = cell["candidate_1_label"]
                requests.append({"opaque_request_key": key, "opaque_logical_key": f"r-{lane}-{cell_index}", "opaque_group_key": f"panel-{lane}",
                                 "kind": "pairwise", "repeat": repeat, "pair_key": "pair", "lane": lane, "cell": cell_index,
                                 "candidate_1_label": label, "display_order": cell["display_order"],
                                 "pointwise_group_candidate_1": "point-1", "pointwise_group_candidate_2": "point-2"})
                responses.append({"opaque_request_key": key, "resolved_verdict": label, "transport_or_schema_failure": False})
    metrics = JUDGE.aggregate(requests, responses, cfg)
    assert metrics["pairwise_repeat_stability"] == 1.0
    assert metrics["factorial_order_consistency"] == 1.0
    assert metrics["label_win_imbalance_abs"] == 0.0
    assert metrics["first_position_win_imbalance_abs"] == 0.0
    assert metrics["two_lane_consensus_rate"] == 1.0
