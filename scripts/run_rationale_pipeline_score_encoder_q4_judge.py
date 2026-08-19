#!/usr/bin/env python3
"""Judge score-encoder emitted scores together with student rationales.

The exact bytes of ``llm_as_judge.txt`` are used.  This is deliberately a
deployment-like consistency evaluation: the candidate score is the encoder's
actual emitted integer prediction and the rationale is the score-blind student
rationale used by the encoder on canonical validation.  Human/reference scores
are never substituted into the judge participant.  RMSE remains the measure of
score correctness; this judge measures rationale/score fidelity and grounding.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Mapping, Sequence

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mal2026.official_writing_contract import JUDGE_DIMENSIONS, parse_participant_output  # noqa: E402
from mal2026.rationale_pipeline_prompts import AXES, judge_participant, routing  # noqa: E402

from run_rationale_pipeline_sft_evaluation import (  # noqa: E402
    Q4_PROMPT_SHA,
    Q4_SERVER_SHA,
    Q4_MODEL_SHA,
    launch_q4,
    stop_owned,
    wait_released,
)


PYTHON = ROOT / ".venv-standard/bin/python"
EVALUATOR = ROOT / "scripts/evaluate_official_q4_judge.py"
JUDGE_PROMPT = ROOT / "llm_as_judge.txt"
OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-score-encoder-q4-judge-v1"
RESTRICTED_PARENT = ROOT / "data/processed/restricted/rationale_pipeline_v1/score_encoder_q4_judge"
Q4_AGGREGATE_PARENT = ROOT / "outputs/official-prompt-alignment-v1/q4-judge"
Q4_RESTRICTED_PARENT = ROOT / "data/processed/restricted/official_prompt_alignment_v1/q4_judge"
GPUS = (0, 1, 2, 3)
SMOKE_PORT = (19710,)
FULL_PORTS = (19720, 19721, 19722, 19723)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_participant_rows(
    predictions: Sequence[Mapping[str, Any]],
    rationale_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join emitted integer predictions to score-blind student rationales."""
    rationales: dict[str, Mapping[str, Any]] = {}
    for row in rationale_rows:
        need(set(row) >= {"source_id", "rationales"}, "student rationale row schema differs")
        source_id = str(row["source_id"])
        need(source_id not in rationales, "student validation rationale source duplicated")
        value = row["rationales"]
        need(isinstance(value, Mapping) and set(value) == set(AXES), "student rationale axes differ")
        rationales[source_id] = value

    participants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in predictions:
        need(set(row) == {"source_id", "continuous_prediction", "emitted_integer_prediction"}, "score prediction row schema differs")
        source_id = str(row["source_id"])
        need(source_id not in seen and source_id in rationales, "score prediction linkage differs")
        seen.add(source_id)
        emitted = row["emitted_integer_prediction"]
        need(isinstance(emitted, Mapping) and set(emitted) == set(AXES), "emitted score axes differ")
        need(all(type(emitted[axis]) is int and 1 <= int(emitted[axis]) <= 5 for axis in AXES), "emitted score is not integer 1-5")
        participant = parse_participant_output(judge_participant(emitted, rationales[source_id]))
        participants.append({"source_id": source_id, "participant_output": participant})
    need(len(participants) == len(rationales) and seen == set(rationales), "score/rationale validation coverage differs")
    return participants


def validate_result(path: Path) -> dict[str, Any]:
    result = read_json(path)
    need(result.get("status") == "completed" and result.get("model_key") in {"qwen3_embedding_8b", "kure_v1"}, "score result differs")
    need(result.get("objective") in {"bounded_regression", "categorical_5class"} and result.get("initialization") in {"base", "aihub"}, "score result arm differs")
    metrics = result.get("canonical_validation", {}).get("metrics", {})
    need(math.isfinite(float(metrics.get("macro_integer_rmse"))) and math.isfinite(float(metrics.get("macro_integer_spearman"))), "score result metrics differ")
    prediction_path = Path(str(result.get("validation_predictions_path", "")))
    need(prediction_path.is_file() and prediction_path.resolve().is_relative_to((ROOT / "data/processed/restricted").resolve()), "score prediction artifact unavailable")
    need(file_sha256(prediction_path) == result.get("validation_predictions_sha256"), "score prediction hash differs")
    config_path = ROOT / "configs" / f"{result['run_id']}.json"
    need(config_path.is_file() and file_sha256(config_path) == result.get("config_sha256"), "score result config differs")
    config = read_json(config_path)
    handoff_path = Path(str(config.get("rationale_handoff_path", "")))
    need(handoff_path.is_file() and file_sha256(handoff_path) == config.get("rationale_handoff_sha256") == result.get("rationale_handoff_sha256"), "score rationale handoff differs")
    handoff = read_json(handoff_path)
    need(handoff.get("validation_view") == "student_score_blind_single_only" and handoff.get("teacher_use") == "train_only_label_aware_augmentation_never_validation_or_selection_dev", "score validation rationale contract differs")
    rationale_path = Path(handoff["paths"]["student_validation_single"])
    need(rationale_path.is_file() and file_sha256(rationale_path) == handoff["sha256"]["student_validation_single"], "student validation rationale differs")
    return {"result": result, "result_path": path, "prediction_path": prediction_path, "rationale_path": rationale_path}


def write_participant(arm: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    predictions = read_jsonl(Path(arm["prediction_path"]))
    rationale_rows = read_jsonl(Path(arm["rationale_path"]))
    rows = build_participant_rows(predictions, rationale_rows)
    need(len(rows) == 400, "score judge validation population differs")
    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(destination, 0o600)
    return {"path": str(destination.resolve()), "sha256": file_sha256(destination), "records": len(rows)}


def telemetry(stop: threading.Event, destination: Path) -> None:
    while not stop.wait(1):
        try:
            raw = subprocess.check_output(["nvidia-smi", "--id=0,1,2,3", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
            rows = [{"index": int(a), "utilization_percent": int(b), "memory_used_mib": int(c)} for a, b, c in ([part.strip() for part in line.split(",")] for line in raw.splitlines())]
            with destination.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"at": now(), "gpus": rows}, separators=(",", ":")) + "\n")
        except Exception:
            pass


def judge_command(run_id: str, participant: Path, endpoints: Sequence[str], attestation: Path, expected: int) -> list[str]:
    command = [
        str(PYTHON), str(EVALUATOR), "--run-id", run_id,
        "--participant-file", str(participant), "--expected", str(expected),
        "--split", "validation", "--max-inflight", str(4 * len(endpoints)),
        "--server-attestation", str(attestation), "--system-prompt-file", str(JUDGE_PROMPT),
    ]
    for endpoint in endpoints:
        command.extend(("--endpoint", endpoint))
    return command


def run_judge(run_id: str, participant: Path, endpoints: Sequence[str], attestation: Path, expected: int, log: Path) -> Path:
    report = Q4_AGGREGATE_PARENT / run_id / "aggregate_judge_report.json"
    if report.is_file():
        value = read_json(report)
        need(value.get("status") == "completed" and value.get("counts", {}).get("valid") == expected and value.get("judge_system_prompt_sha256") == Q4_PROMPT_SHA, "completed score judge artifact differs")
        return report
    need(not (Q4_RESTRICTED_PARENT / run_id).exists() and not (Q4_AGGREGATE_PARENT / run_id).exists(), "partial score judge artifact requires recorded recovery")
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(judge_command(run_id, participant, endpoints, attestation, expected), cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    need(completed.returncode == 0 and report.is_file(), f"score encoder Q4 judge failed: {run_id}")
    value = read_json(report)
    need(value.get("status") == "completed" and value.get("counts", {}).get("valid") == expected, "score encoder Q4 judge gates failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--score-result", type=Path, action="append", required=True)
    args = parser.parse_args()
    setproctitle(f"mal2026:score-encoder-q4-judge:{args.run_id}"[:255])
    need(len(args.score_result) >= 1 and len({str(path.resolve()) for path in args.score_result}) == len(args.score_result), "score judge result inventory differs")
    need(file_sha256(JUDGE_PROMPT) == Q4_PROMPT_SHA, "exact llm_as_judge.txt differs")
    route = routing()
    need(route["rationale_reward_and_quality_judge"]["source_file_sha256"] == Q4_PROMPT_SHA, "judge prompt routing differs")

    output = OUTPUT_PARENT / args.run_id
    restricted = RESTRICTED_PARENT / args.run_id
    need(not output.exists() and not restricted.exists(), "score encoder judge output must be fresh")
    output.mkdir(parents=True); restricted.mkdir(parents=True, mode=0o700)
    arms = [validate_result(path) for path in args.score_result]
    need(len({arm["result"]["run_id"] for arm in arms}) == len(arms), "score judge run IDs duplicated")
    participant_inventory = {}
    for arm in arms:
        run_id = arm["result"]["run_id"]
        participant_inventory[run_id] = write_participant(arm, restricted / "participants" / f"{run_id}.jsonl")
    lineage = {
        "schema_version": "mal2026-rationale-pipeline-score-encoder-q4-judge-lineage-v1",
        "status": "running", "created_at": now(), "run_id": args.run_id,
        "user_authorization": "2026-08-11: add exact llm_as_judge.txt evaluation to scores emitted by the trained score models",
        "gpu_scope": list(GPUS), "judge_prompt_sha256": Q4_PROMPT_SHA,
        "q4_model_sha256": Q4_MODEL_SHA, "q4_server_sha256": Q4_SERVER_SHA,
        "candidate_score_kind": "actual_score_encoder_emitted_integer_prediction",
        "candidate_rationale_kind": "score_blind_student_validation_rationale",
        "human_or_reference_score_read_or_prompted": False,
        "interpretation": "judge fidelity/grounding is complementary to RMSE and does not independently establish score correctness",
        "score_results": [{"path": str(arm["result_path"].resolve()), "sha256": file_sha256(arm["result_path"])} for arm in arms],
        "participants": participant_inventory,
        "privacy": "aggregate lineage only; row artifacts remain restricted",
    }
    atomic_json(output / "lineage.json", lineage)

    smoke_participant = restricted / "smoke.jsonl"
    first_rows = read_jsonl(Path(next(iter(participant_inventory.values()))["path"]))
    with smoke_participant.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(first_rows[0], ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(smoke_participant, 0o600)

    smoke_processes = []
    try:
        smoke_processes, smoke_endpoints, smoke_attestation = launch_q4((0,), SMOKE_PORT, output / "runtime/smoke")
        run_judge(f"{args.run_id}-smoke", smoke_participant, smoke_endpoints, smoke_attestation, 1, output / "smoke.log")
    finally:
        if smoke_processes:
            stop_owned(smoke_processes); wait_released((0,))

    processes = []
    stop = threading.Event()
    thread: threading.Thread | None = None
    reports: list[dict[str, Any]] = []
    try:
        processes, endpoints, attestation = launch_q4(GPUS, FULL_PORTS, output / "runtime/full")
        thread = threading.Thread(target=telemetry, args=(stop, output / "telemetry.jsonl"), daemon=True); thread.start()
        total = len(arms)
        for index, arm in enumerate(arms, start=1):
            result = arm["result"]; encoder_run_id = result["run_id"]
            judge_run_id = f"{args.run_id}-{index:02d}"
            participant = Path(participant_inventory[encoder_run_id]["path"])
            report_path = run_judge(judge_run_id, participant, endpoints, attestation, 400, output / f"judge-{index:02d}.log")
            judge = read_json(report_path)
            metrics = result["canonical_validation"]["metrics"]
            reports.append({
                "encoder_run_id": encoder_run_id, "model_key": result["model_key"], "objective": result["objective"],
                "initialization": result["initialization"], "rationale_ratio": result["rationale_ratio"],
                "score_balance_mode": result.get("score_balance_mode", "none"), "training_protocol": result.get("training_protocol", "select_then_refit"),
                "macro_integer_rmse": metrics["macro_integer_rmse"], "macro_integer_spearman": metrics["macro_integer_spearman"],
                "judge_macro_mean": judge["macro_mean"], "judge_worst_cell_mean": judge["worst_cell_mean"],
                "judge_score_rationale_consistency": judge["dimension_means"]["score_rationale_consistency"],
                "judge_groundedness": judge["dimension_means"]["groundedness"], "judge_specificity": judge["dimension_means"]["specificity"],
                "judge_domain_match": judge["dimension_means"]["domain_match"],
                "judge_report_path": str(report_path.resolve()), "judge_report_sha256": file_sha256(report_path),
            })
            atomic_json(output / "state.json", {"status": "judging", "updated_at": now(), "completed": index, "total": total, "last_encoder_run_id": encoder_run_id})
    finally:
        stop.set()
        if thread is not None: thread.join(timeout=5)
        if processes:
            stop_owned(processes); wait_released(GPUS)

    need(len(reports) == len(arms), "score judge report inventory differs")
    best_rmse = min(reports, key=lambda row: (row["macro_integer_rmse"], -row["macro_integer_spearman"], row["encoder_run_id"]))
    best_judge = max(reports, key=lambda row: (row["judge_macro_mean"], row["judge_worst_cell_mean"], row["encoder_run_id"]))
    best_consistency = max(reports, key=lambda row: (row["judge_score_rationale_consistency"], row["judge_macro_mean"], row["encoder_run_id"]))
    aggregate = {
        "schema_version": "mal2026-rationale-pipeline-score-encoder-q4-judge-v1", "status": "completed", "completed_at": now(), "run_id": args.run_id,
        "candidate_score_kind": lineage["candidate_score_kind"], "candidate_rationale_kind": lineage["candidate_rationale_kind"],
        "human_or_reference_score_read_or_prompted": False, "judge_prompt_sha256": Q4_PROMPT_SHA,
        "candidates": reports, "best_by_rmse": best_rmse, "best_by_judge_macro": best_judge, "best_by_score_rationale_consistency": best_consistency,
        "interpretation": lineage["interpretation"], "average_used": False,
        "privacy": "aggregate only; no prompts, essays, rationales, identifiers, row scores, predictions, or judge evidence",
    }
    atomic_json(output / "aggregate.json", aggregate)
    lineage.update({"status": "completed", "completed_at": aggregate["completed_at"], "aggregate_sha256": file_sha256(output / "aggregate.json")})
    atomic_json(output / "lineage.json", lineage)
    atomic_json(output / "state.json", {"status": "completed", "completed_at": aggregate["completed_at"], "aggregate_path": str((output / "aggregate.json").resolve())})
    print(json.dumps({"status": "completed", "run_id": args.run_id, "candidates": len(reports), "best_by_rmse": best_rmse["encoder_run_id"], "best_by_judge": best_judge["encoder_run_id"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
