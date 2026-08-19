#!/usr/bin/env python3
"""Exact-Q4 target-score/rationale fidelity evaluation with honest provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from setproctitle import setproctitle

import evaluate_official_q4_judge as base


ROOT = Path(__file__).resolve().parents[1]
RESTRICTED_ROOT = ROOT / "data/processed/restricted/official_rationale_fidelity_v1/q4_judge"
AGGREGATE_ROOT = ROOT / "outputs/official-rationale-fidelity-v1/q4-judge"
SCORE_KIND = "canonical_human_reference_integerized_half_up_for_rationale_fidelity_only"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--participant-file", type=Path, required=True)
    parser.add_argument("--model", default="qwen36-35b-a3b-q4_k_m")
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--max-inflight", type=int, required=True)
    parser.add_argument("--server-attestation", type=Path, required=True)
    parser.add_argument("--system-prompt-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setproctitle(f"mal2026:q4-rationale-fidelity:{args.run_id}"[:255])
    base.need(1 <= args.expected <= base.EXPECTED_ESSAYS[args.split], "fidelity judge expected count differs")
    base.need(len(args.endpoint) in {1, 4} and args.max_inflight >= len(args.endpoint), "fidelity endpoint/concurrency differs")
    attestation = json.loads(args.server_attestation.read_text(encoding="utf-8"))
    base.need(attestation.get("schema_version") == "mal2026-official-q4-judge-server-attestation-v1", "fidelity attestation schema differs")
    base.need(attestation.get("model_sha256") == base.MODEL_SHA256 and attestation.get("llama_revision") == base.LLAMA_REVISION and attestation.get("llama_tag") == base.LLAMA_TAG, "fidelity runtime provenance differs")
    base.need(attestation.get("server_endpoints") == args.endpoint, "fidelity endpoints differ from attestation")
    participants = base.load_participants(args.participant_file, args.expected)

    prompt_path = args.system_prompt_file.resolve()
    base.need(prompt_path.is_file() and not prompt_path.is_symlink() and prompt_path.is_relative_to(ROOT.resolve()), "fidelity system prompt file differs")
    prompt_bytes = prompt_path.read_bytes()
    system_prompt = prompt_bytes.decode("utf-8")
    prompt_sha = sha256(prompt_bytes).hexdigest()
    base.need(bool(system_prompt.strip()) and attestation.get("judge_prompt_sha256") == prompt_sha, "fidelity prompt attestation differs")

    output = RESTRICTED_ROOT / args.run_id
    aggregate_output = AGGREGATE_ROOT / args.run_id
    base.need(not output.exists() and not aggregate_output.exists(), "fidelity judge output must be fresh")
    output.mkdir(mode=0o700, parents=True)
    aggregate_output.mkdir(parents=True)
    manifest = {
        "schema_version": "mal2026-official-q4-rationale-fidelity-v1",
        "status": "running",
        "run_id": args.run_id,
        "created_at": now(),
        "participant_sha256": base.sha256_file(args.participant_file),
        "participant_records": len(participants),
        "source_split": args.split,
        "source_sha256": base.SOURCE_SHA256[args.split],
        "model_sha256": base.MODEL_SHA256,
        "llama_revision": base.LLAMA_REVISION,
        "llama_tag": base.LLAMA_TAG,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "candidate_score_kind": SCORE_KIND,
        "human_or_reference_score_read_or_prompted": True,
        "deployment_like_emitted_score_evaluation": False,
        "judge_system_prompt_kind": "repository_file_exact_bytes",
        "judge_system_prompt_sha256": prompt_sha,
        "server_attestation_sha256": base.sha256_file(args.server_attestation),
    }
    base.atomic_json(output / "manifest.json", manifest)
    work = list(base.tasks(participants, args.model, args.endpoint, args.split, system_prompt=system_prompt))
    records: list[dict[str, object]] = []
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
        pending: dict[object, dict[str, object]] = {}
        iterator = iter(work)
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < args.max_inflight:
                try:
                    task = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(base.call, str(task["endpoint"]), task["body"])] = task
            if not pending:
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                parsed, failure, attempts = future.result()
                records.append({"source_id": task["source_id"], "judge_output": parsed, "failure_category": failure, "attempts": attempts})
    records.sort(key=lambda row: str(row["source_id"]))
    record_path = output / "judge_records.jsonl"
    with record_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = base.aggregate(records, args.expected)
    report.update({
        "schema_version": "mal2026-official-q4-rationale-fidelity-aggregate-v1",
        "run_id": args.run_id,
        "judge_records_sha256": base.sha256_file(record_path),
        "participant_sha256": manifest["participant_sha256"],
        "model_sha256": base.MODEL_SHA256,
        "llama_revision": base.LLAMA_REVISION,
        "llama_tag": base.LLAMA_TAG,
        "temperature": 0.0,
        "seed": 42,
        "candidate_score_kind": SCORE_KIND,
        "human_or_reference_score_read_or_prompted": True,
        "deployment_like_emitted_score_evaluation": False,
        "judge_system_prompt_kind": "repository_file_exact_bytes",
        "judge_system_prompt_sha256": prompt_sha,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_evidence_scores_or_predictions_in_this_report",
    })
    base.atomic_json(aggregate_output / "aggregate_judge_report.json", report)
    manifest.update({"status": report["status"], "completed_at": now(), "aggregate_report_sha256": base.sha256_file(aggregate_output / "aggregate_judge_report.json")})
    base.atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"status": report["status"], "run_id": args.run_id, "counts": report["counts"], "macro_mean": report["macro_mean"]}, sort_keys=True), flush=True)
    if report["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
