#!/usr/bin/env python3
"""Bind one immutable AI-Hub score-pretrain aggregate into a matrix template."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_score_matrix import HEADS, MatrixConfig, file_sha256  # noqa: E402
from mal2026.official_score_prompt import provenance as score_prompt_provenance  # noqa: E402


def resolve(template_path: Path, aggregate_path: Path, output_path: Path, *, validate_artifacts: bool) -> MatrixConfig:
    raw = json.loads(template_path.read_text(encoding="utf-8"))
    template = MatrixConfig.from_json(template_path, require_dependencies=False)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    expected_prompt = score_prompt_provenance(template.score_prompt_kind)
    if (
        aggregate.get("schema_version") != "mal2026-aihub-integer-score-pretrain-aggregate-v2"
        or aggregate.get("status") != "completed"
        or aggregate.get("score_prompt_kind") != expected_prompt["score_prompt_kind"]
        or aggregate.get("score_prompt_sha256") != expected_prompt["score_prompt_sha256"]
    ):
        raise RuntimeError("AI-Hub score pretrain aggregate identity or prompt differs")
    results = aggregate.get("results")
    if not isinstance(results, list) or {item.get("head") for item in results if isinstance(item, dict)} != set(HEADS):
        raise RuntimeError("AI-Hub score pretrain head coverage differs")
    by_head = {item["head"]: item for item in results}
    for head, prefix in (("bounded_regression", "aihub_bounded"), ("ordinal_cumulative", "aihub_ordinal")):
        result = by_head[head]
        completion = Path(result["completion_path"])
        artifact = Path(result["artifact_path"])
        if completion.resolve() != Path(raw[f"{prefix}_completion_path"]).resolve():
            raise RuntimeError(f"{head} completion path differs from the frozen template")
        if artifact.resolve() != Path(raw[f"{prefix}_artifact_path"]).resolve():
            raise RuntimeError(f"{head} artifact path differs from the frozen template")
        if file_sha256(completion) != result.get("completion_sha256"):
            raise RuntimeError(f"{head} completion checksum differs")
        payload = json.loads(completion.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "completed"
            or payload.get("phase") != "refit"
            or payload.get("head") != head
            or payload.get("score_prompt_sha256") != expected_prompt["score_prompt_sha256"]
            or payload.get("state", {}).get("artifact_sha256") != result.get("artifact_sha256")
        ):
            raise RuntimeError(f"{head} completion binding differs")
        raw[f"{prefix}_completion_sha256"] = result["completion_sha256"]
        raw[f"{prefix}_artifact_sha256"] = result["artifact_sha256"]

    allowed_root = (ROOT / "outputs" / "official-score-matrix-config-resolution-v1" / template.run_id).resolve()
    if output_path.resolve().parent != allowed_root:
        raise RuntimeError("resolved config output is outside its ignored run root")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise RuntimeError("refusing to replace resolved score matrix config")
    output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resolved = MatrixConfig.from_json(output_path, require_dependencies=False)
    if validate_artifacts:
        resolved.validate_dependencies("bootstrap")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-artifact-validation", action="store_true")
    args = parser.parse_args()
    config = resolve(
        args.template,
        args.aggregate,
        args.output,
        validate_artifacts=not args.skip_artifact_validation,
    )
    print(json.dumps({
        "status": "resolved",
        "run_id": config.run_id,
        "score_prompt_kind": config.score_prompt_kind,
        "output": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
        "artifacts_validated": not args.skip_artifact_validation,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
