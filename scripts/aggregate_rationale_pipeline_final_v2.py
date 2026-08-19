#!/usr/bin/env python3
"""Attach emitted-score exact-Q4 judge evidence to the final pipeline report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

from setproctitle import setproctitle


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--score-judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setproctitle(f"mal2026:final-report-v2:{args.run_id}"[:255])
    need(args.base_report.is_file() and args.score_judge.is_file(), "final v2 input unavailable")
    base, judge = read(args.base_report), read(args.score_judge)
    need(base.get("status") == "completed" and base.get("schema_version") == "mal2026-rationale-pipeline-final-report-v1", "base final report differs")
    need(judge.get("status") == "completed" and judge.get("schema_version") == "mal2026-rationale-pipeline-score-encoder-q4-judge-v1", "score judge report differs")
    candidates = judge.get("candidates")
    need(isinstance(candidates, list) and len(candidates) >= 1, "score judge candidates unavailable")
    need(all(row.get("judge_report_sha256") and row.get("macro_integer_rmse") is not None for row in candidates), "score judge candidate evidence differs")
    value: Mapping[str, Any] = {
        "schema_version": "mal2026-rationale-pipeline-final-report-v2",
        "status": "completed", "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), "run_id": args.run_id,
        "base_report": {"path": str(args.base_report.resolve()), "sha256": digest(args.base_report), "best_score_system": base["best_score_system"]},
        "emitted_score_llm_as_judge": {
            "path": str(args.score_judge.resolve()), "sha256": digest(args.score_judge),
            "candidate_count": len(candidates), "best_by_rmse": judge["best_by_rmse"],
            "best_by_judge_macro": judge["best_by_judge_macro"],
            "best_by_score_rationale_consistency": judge["best_by_score_rationale_consistency"],
            "candidate_score_kind": judge["candidate_score_kind"], "candidate_rationale_kind": judge["candidate_rationale_kind"],
            "judge_prompt_sha256": judge["judge_prompt_sha256"],
        },
        "interpretation": {
            "score_correctness": "macro integer RMSE against Decimal ROUND_HALF_UP human labels",
            "score_rationale_fidelity": "exact llm_as_judge.txt on actual encoder-emitted integer scores plus score-blind student rationales",
            "non_substitution_rule": "judge fidelity does not replace RMSE and does not prove that the emitted score is correct",
        },
        "average_used": False,
        "privacy": "aggregate only; no prompts, essays, rationales, identifiers, row scores, predictions, judge evidence, or weights",
    }
    need(not args.output.exists(), "final v2 output must be fresh")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"status": "completed", "output": str(args.output), "candidate_count": len(candidates)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
