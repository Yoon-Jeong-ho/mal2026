#!/usr/bin/env python3
"""Compose authoritative encoder scores and rationale-only model outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import AXES, sha256_file  # noqa: E402
from mal2026.official_writing_contract import parse_participant_output  # noqa: E402


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_scores(path: Path, expected: int) -> dict[str, dict[str, int]]:
    restricted = (ROOT / "data/processed/restricted").resolve()
    need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(restricted), "score input must be restricted")
    result: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            allowed = ({"source_id", "emitted_integer_prediction"}, {"source_id", "continuous_prediction", "emitted_integer_prediction"})
            need(isinstance(raw, dict) and set(raw) in allowed, "score input schema differs")
            source_id, scores = raw["source_id"], raw["emitted_integer_prediction"]
            need(isinstance(source_id, str) and source_id not in result, "score source ID differs")
            need(isinstance(scores, dict) and set(scores) == set(AXES), "score axes differ")
            need(all(type(scores[axis]) is int and 1 <= scores[axis] <= 5 for axis in AXES), "score must be emitted integers")
            result[source_id] = {axis: int(scores[axis]) for axis in AXES}
    need(len(result) == expected, "score population differs")
    return result


def load_generation(path: Path, expected: int) -> tuple[str, dict[str, dict[str, str]], str]:
    report = json.loads((path / "aggregate_generation_report.json").read_text(encoding="utf-8"))
    need(report.get("status") == "completed" and all(report.get("hard_gates", {}).values()), "rationale generation is incomplete")
    records_path = path / "generated_rationales.jsonl"
    need(sha256_file(records_path) == report.get("generated_rationales_sha256"), "rationale generation checksum differs")
    task = str(report["task"])
    result: dict[str, dict[str, str]] = {}
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            need(set(raw) == {"source_id", "rationales", "failure_category", "attempts"} and raw["failure_category"] is None, "rationale generation row differs")
            result[str(raw["source_id"])] = {str(axis): str(text) for axis, text in raw["rationales"].items()}
    need(len(result) == expected, "rationale generation population differs")
    return task, result, sha256_file(records_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--aggregate-file", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    scores = load_scores(args.score_file, args.expected)
    generations = [load_generation(path.resolve(), args.expected) for path in args.generation_dir]
    tasks = [item[0] for item in generations]
    need(tasks == ["bundle"] or set(tasks) == set(AXES) and len(tasks) == 3, "participant composition requires bundle or exactly three axis generations")
    combined: dict[str, dict[str, str]] = {source_id: {} for source_id in scores}
    hashes: dict[str, str] = {}
    for task, rows, digest in generations:
        hashes[task] = digest
        for source_id, rationales in rows.items():
            need(source_id in combined and not (set(combined[source_id]) & set(rationales)), "rationale axes overlap or source differs")
            combined[source_id].update(rationales)
    need(all(set(value) == set(AXES) for value in combined.values()), "participant rationale axes are incomplete")
    output = args.output_file.resolve()
    aggregate = args.aggregate_file.resolve()
    restricted = (ROOT / "data/processed/restricted").resolve()
    need(output.is_relative_to(restricted) and not output.exists() and not aggregate.exists(), "participant outputs must be fresh/restricted")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for source_id in sorted(scores):
            participant = {axis: {"score": scores[source_id][axis], "rationale": combined[source_id][axis]} for axis in AXES}
            participant = parse_participant_output(participant)
            handle.write(json.dumps({"source_id": source_id, "participant_output": participant}, ensure_ascii=False, separators=(",", ":")) + "\n")
    payload = {
        "schema_version": "mal2026-official-participant-composition-v1",
        "status": "completed",
        "run_id": args.run_id,
        "records": args.expected,
        "score_file_sha256": sha256_file(args.score_file),
        "generation_sha256": hashes,
        "participant_file_sha256": sha256_file(output),
        "score_authority": "encoder_emitted_integer_prediction",
        "score_rationale_model_score_mismatch_count": 0,
        "strict_participant_parse_count": args.expected,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions_in_this_report",
    }
    aggregate.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "run_id": args.run_id, "records": args.expected}, sort_keys=True))


if __name__ == "__main__":
    main()
