#!/usr/bin/env python3
"""Evaluate restricted prompt-injection cases with the unchanged official proxy prompt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_official_q4_judge as judge  # noqa: E402
from mal2026.api_rationale_data import SOURCE_SHA256, load_writing_rows, sha256_file  # noqa: E402
from mal2026.official_writing_contract import parse_participant_output  # noqa: E402


JUDGE_CONTRACT_SHA256 = "7b04149227a44852ca78bd65f5ec70245b284503256374debf2735f17ca69e50"


def load_cases(path: Path, expected: int) -> list[dict[str, Any]]:
    allowed = (ROOT / "data/processed/restricted").resolve()
    judge.need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(allowed), "injection input is unavailable or unrestricted")
    rows, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            judge.need(isinstance(raw, dict) and set(raw) == {"source_id", "participant_output", "essay_suffix"}, "injection row schema differs")
            judge.need(isinstance(raw["source_id"], str) and raw["source_id"] not in seen, "injection source ID differs")
            judge.need(isinstance(raw["essay_suffix"], str), "injection essay suffix differs")
            seen.add(raw["source_id"])
            rows.append({"source_id": raw["source_id"], "participant_output": parse_participant_output(raw["participant_output"]), "essay_suffix": raw["essay_suffix"]})
    judge.need(len(rows) == expected, "injection population differs")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True); parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", required=True); parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--max-inflight", type=int, default=16); parser.add_argument("--server-attestation", type=Path, required=True)
    parser.add_argument("--model", default="qwen36-35b-a3b-q4_k_m")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); judge.need(len(args.endpoint) in {1, 4} and args.max_inflight >= len(args.endpoint), "injection endpoint/concurrency differs")
    attestation = json.loads(args.server_attestation.read_text(encoding="utf-8"))
    judge.need(attestation.get("schema_version") == "mal2026-official-q4-judge-server-attestation-v1", "injection attestation schema differs")
    judge.need(attestation.get("model_sha256") == judge.MODEL_SHA256 and attestation.get("llama_revision") == judge.LLAMA_REVISION and attestation.get("llama_tag") == judge.LLAMA_TAG, "injection runtime provenance differs")
    judge.need(attestation.get("server_endpoints") == args.endpoint, "injection endpoints differ")
    cases = load_cases(args.case_file, args.expected)
    writings = {row.identifier: row for row in load_writing_rows("train", include_scores=False)}
    judge.need({row["source_id"] for row in cases} <= set(writings), "injection source differs from canonical train")
    record_root = judge.RESTRICTED_ROOT / args.run_id; aggregate_root = judge.AGGREGATE_ROOT / args.run_id
    judge.need(not record_root.exists() and not aggregate_root.exists(), "injection judge output must be fresh")
    record_root.mkdir(mode=0o700, parents=True); aggregate_root.mkdir(parents=True)
    manifest = {
        "schema_version": "mal2026-official-q4-judge-prompt-injection-manifest-v1", "status": "running",
        "run_id": args.run_id, "created_at": judge.now(), "case_file_sha256": sha256_file(args.case_file),
        "case_records": len(cases), "source_split": "train", "source_sha256": SOURCE_SHA256["train"],
        "model_sha256": judge.MODEL_SHA256, "llama_revision": judge.LLAMA_REVISION, "llama_tag": judge.LLAMA_TAG,
        "judge_contract_sha256": JUDGE_CONTRACT_SHA256, "temperature": 0.0, "top_p": 1.0, "seed": 42,
        "candidate_score_kind": "same_actual_emitted_integer_prediction_across_variants",
        "human_or_reference_score_read_or_prompted": False,
        "server_attestation_sha256": sha256_file(args.server_attestation),
    }
    judge.atomic_json(record_root / "manifest.json", manifest)
    work = []
    for index, row in enumerate(cases):
        writing = writings[row["source_id"]]
        work.append({"source_id": row["source_id"], "endpoint": args.endpoint[index % len(args.endpoint)], "body": judge.request_body(args.model, writing.prompt, writing.essay + row["essay_suffix"], row["participant_output"])})
    records = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
        futures = [(task, pool.submit(judge.call, task["endpoint"], task["body"])) for task in work]
        for task, future in futures:
            parsed, failure, attempts = future.result()
            records.append({"source_id": task["source_id"], "judge_output": parsed, "failure_category": failure, "attempts": attempts})
    records.sort(key=lambda row: str(row["source_id"]))
    record_path = record_root / "judge_records.jsonl"
    with record_path.open("x", encoding="utf-8") as handle:
        for row in records: handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = judge.aggregate(records, args.expected)
    report.update({"schema_version": "mal2026-official-q4-judge-prompt-injection-aggregate-v1", "run_id": args.run_id,
                   "judge_records_sha256": sha256_file(record_path), "case_file_sha256": sha256_file(args.case_file),
                   "source_split": "train", "source_sha256": SOURCE_SHA256["train"], "model_sha256": judge.MODEL_SHA256,
                   "llama_revision": judge.LLAMA_REVISION, "llama_tag": judge.LLAMA_TAG, "temperature": 0.0, "seed": 42,
                   "judge_prompt_modified": False, "candidate_score_kind": "same_actual_emitted_integer_prediction_across_variants",
                   "human_or_reference_score_read_or_prompted": False,
                   "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_injection_payloads_evidence_or_predictions"})
    judge.atomic_json(aggregate_root / "aggregate_judge_report.json", report)
    manifest.update({"status": report["status"], "completed_at": judge.now(),
                     "aggregate_report_sha256": sha256_file(aggregate_root / "aggregate_judge_report.json")})
    judge.atomic_json(record_root / "manifest.json", manifest)
    print(json.dumps({"status": report["status"], "run_id": args.run_id, "counts": report["counts"]}, sort_keys=True))
    if report["status"] != "completed": raise SystemExit(2)


if __name__ == "__main__": main()
