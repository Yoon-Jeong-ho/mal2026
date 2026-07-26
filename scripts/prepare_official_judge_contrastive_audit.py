#!/usr/bin/env python3
"""Create restricted train-only variants for the frozen-judge sanity gate."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_writing_contract import AXES, parse_participant_output  # noqa: E402


SOURCE = ROOT / "data/processed/restricted/official_openai_candidates_v1/official-openai-candidates-v1-train3-20260727-001/candidates.train.jsonl"
RUN_ID = "official-judge-contrastive-train32-001"
OUTPUT = ROOT / "data/processed/restricted/official_prompt_alignment_v1/judge_contrastive" / RUN_ID
REPORT = ROOT / "outputs/official-prompt-alignment-v1/judge-contrastive" / RUN_ID / "preparation_report.json"
N = 32


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def changed_score(score: int) -> int:
    value = score + 2 if score <= 3 else score - 2
    need(1 <= value <= 5 and abs(value - score) == 2, "score perturbation differs")
    return value


def variants(participant: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    base = parse_participant_output(participant)
    rotated = {"content": "organization", "organization": "expression", "expression": "content"}
    swapped = {axis: {"score": base[axis]["score"], "rationale": base[rotated[axis]]["rationale"]} for axis in AXES}
    perturbed = {axis: {"score": changed_score(base[axis]["score"]), "rationale": base[axis]["rationale"]} for axis in AXES}
    unsupported = {
        axis: {
            "score": base[axis]["score"],
            "rationale": "이 설명은 원문의 구체적인 문장이나 근거를 확인하지 않고 전반적으로 적절하다고만 판단한다.",
        }
        for axis in AXES
    }
    return {name: parse_participant_output(value) for name, value in {"base": base, "axis_swapped": swapped, "score_perturbed": perturbed, "unsupported": unsupported}.items()}


def main() -> None:
    need(SOURCE.is_file() and not OUTPUT.exists() and not REPORT.exists(), "contrastive preparation must be fresh")
    selected: list[dict[str, Any]] = []
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("candidate") == 1:
                selected.append(row)
    selected.sort(key=lambda row: str(row["source_id"])); selected = selected[:N]
    need(len(selected) == N and len({row["source_id"] for row in selected}) == N, "contrastive source population differs")
    OUTPUT.mkdir(mode=0o700, parents=True); REPORT.parent.mkdir(parents=True)
    files = {name: OUTPUT / f"{name}.jsonl" for name in ("base", "axis_swapped", "score_perturbed", "unsupported")}
    handles = {name: path.open("x", encoding="utf-8") for name, path in files.items()}
    try:
        for row in selected:
            for name, participant in variants(row["participant_output"]).items():
                handles[name].write(json.dumps({"source_id": row["source_id"], "participant_output": participant}, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        for handle in handles.values(): handle.close()
    payload = {
        "schema_version": "mal2026-official-judge-contrastive-preparation-v1", "status": "completed", "run_id": RUN_ID,
        "records_per_variant": N, "source_split": "train", "source_candidate_index": 1,
        "variants": {name: {"sha256": file_sha(path), "records": N} for name, path in files.items()},
        "manipulations": {"axis_swapped": "cyclic rationale-only axis swap; scores fixed", "score_perturbed": "every emitted integer score shifted exactly two points; rationales fixed", "unsupported": "all rationales replaced by a fixed explicitly ungrounded generic sentence; scores fixed"},
        "human_or_reference_score_read_or_prompted": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_or_API_response_metadata_in_report",
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "records_per_variant": N, "variants": sorted(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
