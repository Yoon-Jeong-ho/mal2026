#!/usr/bin/env python3
"""Synthetic, score-schema health gate for a local train-only Qwen reward server."""
from __future__ import annotations

import argparse
from pathlib import Path

from mal2026.rlaif_grpo import JUDGE, RLAIFSettings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    settings = RLAIFSettings.from_json(); template = settings.fixed_prompt_template(); prompt_type = template["protocol"]["prompt_types"][0]
    rationale = {
        "schema_version": "rationale-only-v1",
        "content": {"rationale": "주장과 근거가 연결되어 내용 전개가 구체적이다."},
        "organization": {"rationale": "서론의 주장과 결론의 정리가 자연스럽게 이어진다."},
        "expression": {"rationale": "문장이 명확하고 어휘 선택이 자연스럽다."},
    }
    entry = {"sentences": ["학생은 자신의 주장을 근거와 함께 설명했다.", "결론에서 앞선 내용을 정리했다."], "rationale": rationale}
    body = JUDGE.request_body(template, settings.judge["model_id"], entry, ["content", "organization", "expression"], prompt_type["layout"], settings.judge["training_seed_by_prompt_index"][0], prompt_type["review_emphasis"])
    result = JUDGE.call(args.endpoint, {"opaque_request_key": "synthetic", "response_contract": "required_scores_only_v1", "body": body})
    if not result.get("scored") or not result.get("schema_valid") or result.get("failure_category") is not None:
        raise SystemExit("synthetic reward-server score-schema gate failed")


if __name__ == "__main__":
    main()
