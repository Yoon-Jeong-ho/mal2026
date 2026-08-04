#!/usr/bin/env python3
"""Run the local MAL2026 human score/rationale validation web application."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from humaneval.core import ResponseStore, axis_band_counts, build_study  # noqa: E402
from humaneval.server import HumanValidationServer  # noqa: E402


DEFAULT_API = ROOT / "data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001/candidates.jsonl"
DEFAULT_MODEL_ROOT = ROOT / "data/processed/restricted/evaluation_prompt_rationale_v2/evaluation-prompt-rationale-generation-v2-score-blind-20260729-004"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "eval/train.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "eval/validation.jsonl")
    parser.add_argument("--rubric", type=Path, default=ROOT / "evaluation.txt")
    parser.add_argument("--judge-guide", type=Path, default=ROOT / "llm_as_judge.txt")
    parser.add_argument("--api-rationales", type=Path, action="append", default=None)
    parser.add_argument("--model-rationales", type=Path, action="append", default=None)
    parser.add_argument("--api-candidate", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--database", type=Path, default=ROOT / "outputs/humaneval/responses.sqlite3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print aggregate selection evidence only")
    parser.add_argument("--export-jsonl", type=Path, help="export saved responses and exit")
    return parser.parse_args()


def require_ignored_result_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    ignored_root = (ROOT / "outputs").resolve()
    if not resolved.is_relative_to(ignored_root):
        raise SystemExit(f"result artifacts must stay under ignored outputs/: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    api_paths = args.api_rationales or [DEFAULT_API]
    model_paths = args.model_rationales or [
        DEFAULT_MODEL_ROOT / "rationales.train.jsonl",
        DEFAULT_MODEL_ROOT / "rationales.validation.jsonl",
    ]
    study = build_study(
        split_paths={"train": args.train, "validation": args.validation},
        rubric_path=args.rubric,
        judge_guide_path=args.judge_guide,
        api_rationale_paths=api_paths,
        model_rationale_paths=model_paths,
        seed=args.seed,
        api_candidate=args.api_candidate,
    )
    counts = axis_band_counts(study.items)
    print(f"study fingerprint: {study.fingerprint}")
    print(f"selected items: {len(study.items)}; hidden per-axis target-band counts: {counts}")
    print(f"common prompt notices: 1; reviewer names: 4")
    if args.dry_run:
        print("preflight passed")
        return

    database = require_ignored_result_path(args.database)
    store = ResponseStore(database, study)
    if args.export_jsonl is not None:
        output = require_ignored_result_path(args.export_jsonl)
        count = store.export_jsonl(output)
        print(f"exported {count} response rows to {output}")
        return

    static_root = ROOT / "humaneval/web"
    server = HumanValidationServer((args.host, args.port), store, static_root)
    host, port = server.server_address[:2]
    print(f"response database: {database}")
    print(f"listening on http://{host}:{port}")
    print("Use SSH port forwarding for remote reviewers; bind publicly only on an access-controlled network.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
