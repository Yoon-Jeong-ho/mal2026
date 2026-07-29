#!/usr/bin/env python3
"""Compose restricted score predictions and bundled rationale handoffs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import sha256_file  # noqa: E402
from mal2026.official_writing_contract import AXES, parse_participant_output  # noqa: E402


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
            need(isinstance(raw, dict) and "source_id" in raw and "emitted_integer_prediction" in raw, "score row differs")
            source_id, scores = raw["source_id"], raw["emitted_integer_prediction"]
            need(isinstance(source_id, str) and source_id not in result, "score ID differs")
            need(isinstance(scores, dict) and set(scores) == set(AXES), "score axes differ")
            need(all(type(scores[axis]) is int and 1 <= scores[axis] <= 5 for axis in AXES), "score values differ")
            result[source_id] = {axis: scores[axis] for axis in AXES}
    need(len(result) == expected, "score population differs")
    return result


def load_rationales(path: Path, expected: int) -> dict[str, dict[str, str]]:
    restricted = (ROOT / "data/processed/restricted").resolve()
    need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(restricted), "rationale input must be restricted")
    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            need(isinstance(raw, dict) and set(raw) == {"source_id", "rationales"}, "rationale row differs")
            source_id, rationales = raw["source_id"], raw["rationales"]
            need(isinstance(source_id, str) and source_id not in result, "rationale ID differs")
            need(isinstance(rationales, dict) and set(rationales) == set(AXES), "rationale axes differ")
            need(all(isinstance(rationales[axis], str) and rationales[axis].strip() for axis in AXES), "rationale values differ")
            result[source_id] = {axis: rationales[axis].strip() for axis in AXES}
    need(len(result) == expected, "rationale population differs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--rationale-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--aggregate-file", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    scores = load_scores(args.score_file, args.expected)
    rationales = load_rationales(args.rationale_file, args.expected)
    need(set(scores) == set(rationales), "score/rationale IDs differ")
    restricted = (ROOT / "data/processed/restricted").resolve()
    output, aggregate = args.output_file.resolve(), args.aggregate_file.resolve()
    need(output.is_relative_to(restricted) and not output.exists() and not aggregate.exists(), "participant outputs must be fresh and restricted")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for source_id in sorted(scores):
            participant = parse_participant_output({axis: {"score": scores[source_id][axis], "rationale": rationales[source_id][axis]} for axis in AXES})
            handle.write(json.dumps({"source_id": source_id, "participant_output": participant}, ensure_ascii=False, separators=(",", ":")) + "\n")
    payload = {
        "schema_version": "mal2026-evaluation-prompt-participant-composition-v1",
        "status": "completed", "run_id": args.run_id, "records": args.expected,
        "score_file_sha256": sha256_file(args.score_file),
        "rationale_file_sha256": sha256_file(args.rationale_file),
        "participant_file_sha256": sha256_file(output),
        "score_authority": "encoder_actual_emitted_integer_prediction",
        "human_or_reference_score_read_or_prompted": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_or_predictions",
    }
    aggregate.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "run_id": args.run_id, "records": args.expected}, sort_keys=True))


if __name__ == "__main__":
    main()
