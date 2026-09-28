#!/usr/bin/env python3
"""Create an ignored D1 seed containing only the frozen 20-item study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from humaneval.core import build_study  # noqa: E402


DEFAULT_API = ROOT / "data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001/candidates.jsonl"
DEFAULT_MODEL_ROOT = ROOT / "data/processed/restricted/evaluation_prompt_rationale_v2/evaluation-prompt-rationale-generation-v2-score-blind-20260729-004"
DEFAULT_OUTPUT = ROOT / "outputs/humaneval/cloudflare-study-seed.sql"


def sql_text(value: str) -> str:
    if "\x00" in value:
        raise ValueError("D1 seed text must not contain NUL bytes")
    return "'" + value.replace("'", "''") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    ignored_root = (ROOT / "outputs").resolve()
    if not output.is_relative_to(ignored_root):
        raise SystemExit(f"seed must stay under git-ignored outputs/: {output}")
    if output.exists() and not args.replace:
        raise SystemExit(f"seed already exists; pass --replace to overwrite: {output}")

    study = build_study(
        split_paths={"train": ROOT / "eval/train.jsonl", "validation": ROOT / "eval/validation.jsonl"},
        rubric_path=ROOT / "evaluation.txt",
        judge_guide_path=ROOT / "llm_as_judge.txt",
        api_rationale_paths=[DEFAULT_API],
        model_rationale_paths=[
            DEFAULT_MODEL_ROOT / "rationales.train.jsonl",
            DEFAULT_MODEL_ROOT / "rationales.validation.jsonl",
        ],
    )
    meta = {
        "schema": "mal2026-human-validation-study-v1",
        "fingerprint": study.fingerprint,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "common_notice": study.common_notice,
        "rubric_json": json.dumps(study.rubric.as_public_dict(), ensure_ascii=False, separators=(",", ":")),
        "judge_guide_json": json.dumps(study.judge_guide.as_public_dict(), ensure_ascii=False, separators=(",", ":")),
    }
    lines = ["-- Restricted deployment artifact. Never commit this file."]
    for key, value in meta.items():
        lines.append(
            f"INSERT INTO study_meta(key, value) VALUES ({sql_text(key)}, {sql_text(value)});"
        )
    for item in study.items:
        values = [
            str(item.index),
            sql_text(item.source_id),
            sql_text(item.split),
            str(item.score_bands["content"]),
            str(item.score_bands["organization"]),
            str(item.score_bands["expression"]),
            sql_text(item.topic_prompt),
            sql_text(item.essay),
            sql_text(json.dumps(item.api_rationale, ensure_ascii=False, separators=(",", ":"))),
            sql_text(json.dumps(item.model_rationale, ensure_ascii=False, separators=(",", ":"))),
            sql_text(item.first_source),
        ]
        lines.append(
            "INSERT INTO study_items("
            "item_index,source_id,split,target_content_band,target_organization_band,"
            "target_expression_band,topic_prompt,essay,api_rationale_json,"
            "model_rationale_json,first_source) VALUES ("
            + ",".join(values)
            + ");"
        )

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output.chmod(0o600)
    print(f"study fingerprint: {study.fingerprint}")
    print(f"selected items: {len(study.items)}")
    print(f"ignored D1 seed: {output} ({output.stat().st_size} bytes, mode 0600)")


if __name__ == "__main__":
    main()
