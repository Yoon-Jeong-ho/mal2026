#!/usr/bin/env python3
"""Prepare restricted, train-only clean and prompt-injection judge inputs."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_judge_injection_gate import make_variant, need  # noqa: E402


CONFIG = ROOT / "configs/official_q4_judge_prompt_injection_gate.v1.json"
SOURCE = ROOT / "data/processed/restricted/official_openai_candidates_v1/official-openai-candidates-v1-train3-20260727-001/candidates.train.jsonl"


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cfg: dict[str, Any] = json.loads(CONFIG.read_text(encoding="utf-8"))
    run_id = cfg["run_id"]
    count = int(cfg["source"]["records"])
    output = ROOT / "data/processed/restricted/official_prompt_alignment_v1/judge_prompt_injection" / run_id
    report = ROOT / "outputs/official-prompt-alignment-v1/judge-prompt-injection" / run_id / "preparation_report.json"
    need(SOURCE.is_file() and not output.exists() and not report.exists(), "injection preparation must be fresh")
    selected = []
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("candidate") == cfg["source"]["candidate_index"]:
                selected.append(row)
    selected.sort(key=lambda row: str(row["source_id"]))
    selected = selected[:count]
    need(len(selected) == count and len({row["source_id"] for row in selected}) == count, "injection population differs")
    output.mkdir(mode=0o700, parents=True)
    report.parent.mkdir(parents=True)
    files = {name: output / f"{name}.jsonl" for name in cfg["variants"]}
    handles = {name: path.open("x", encoding="utf-8") for name, path in files.items()}
    try:
        for row in selected:
            base_scores = [row["participant_output"][axis]["score"] for axis in ("content", "organization", "expression")]
            for name, spec in cfg["variants"].items():
                participant, essay_suffix = make_variant(row["participant_output"], **spec)
                scores = [participant[axis]["score"] for axis in ("content", "organization", "expression")]
                need(scores == base_scores, "variant changed predicted scores")
                handles[name].write(json.dumps({
                    "source_id": row["source_id"],
                    "participant_output": participant,
                    "essay_suffix": essay_suffix,
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    payload = {
        "schema_version": "mal2026-official-judge-prompt-injection-preparation-v1",
        "status": "completed", "run_id": run_id, "source_split": "train",
        "source_candidate_index": cfg["source"]["candidate_index"], "records_per_variant": count,
        "config_sha256": file_sha(CONFIG),
        "variants": {name: {"records": count, "sha256": file_sha(path)} for name, path in files.items()},
        "same_actual_predicted_integer_scores_across_variants": True,
        "judge_prompt_modified": False, "human_or_reference_score_read_or_prompted": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_or_injection_payloads_in_report",
    }
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "records_per_variant": count, "variants": sorted(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
