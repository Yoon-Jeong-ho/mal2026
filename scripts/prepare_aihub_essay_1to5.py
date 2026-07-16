#!/usr/bin/env python3
"""Create a derived, split-cleaned AI-Hub essay-label dataset for modeling.

Raw AI-Hub files are never changed.  The derived JSONL output is intentionally
ignored by Git because it contains writing responses and prompts.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET = ROOT / "data" / "raw" / "aihub" / "024_essay_writing_evaluation"
OUTPUT_DIR = ROOT / "data" / "processed" / "aihub_essay_1to5_v1"
MANIFEST_PATH = ROOT / "data" / "manifests" / "aihub_essay_1to5_v1.json"
SCORE_MIN = 0.0
SCORE_MAX = 30.0


def normalized_text_hash(record: dict) -> str:
    paragraphs = record.get("paragraph", [])
    text = "\n".join(
        str(paragraph.get("paragraph_txt", "")) for paragraph in paragraphs if isinstance(paragraph, dict)
    ).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_score(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Non-numeric essay score: {value!r}")
    if not SCORE_MIN <= float(value) <= SCORE_MAX:
        raise ValueError(f"Essay score outside expected 0–30 range: {value!r}")
    return 1.0 + 4.0 * (float(value) - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


def iter_label_records(split: str):
    archives = sorted(
        archive
        for archive in RAW_DATASET.rglob("*.zip")
        if "라벨링데이터" in archive.as_posix() and (("Training" in archive.as_posix()) == (split == "train"))
    )
    for archive in archives:
        with ZipFile(archive) as zf:
            for member in sorted(name for name in zf.namelist() if name.endswith(".json")):
                yield json.loads(zf.read(member).decode("utf-8-sig"))


def with_derived_target(record: dict, source_split: str, derived_split: str) -> dict:
    score = record.get("score", {})
    average = score.get("essay_scoreT_avg")
    rater_scores = score.get("essay_scoreT")
    if not isinstance(rater_scores, list) or len(rater_scores) != 3:
        raise ValueError("Expected exactly three essay rater scores")
    record["derived_model_target"] = {
        "essay_score_1to5": normalized_score(average),
        "essay_rater_scores_1to5": [normalized_score(value) for value in rater_scores],
        "normalization": "1 + 4 * (raw_score / 30)",
        "raw_score_range": [SCORE_MIN, SCORE_MAX],
    }
    record["derived_split"] = derived_split
    record["source_split"] = source_split
    return record


def write_record(handle, digest, record: dict):
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    encoded = line.encode("utf-8")
    handle.write(encoded)
    digest.update(encoded)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing derived data: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    train_path = OUTPUT_DIR / "train.jsonl"
    validation_path = OUTPUT_DIR / "validation.jsonl"
    train_digest, validation_digest = hashlib.sha256(), hashlib.sha256()
    train_text_hashes: set[str] = set()
    train_count = validation_count = moved_count = 0
    source_counts = Counter()
    normalized_targets = []

    with train_path.open("xb") as train_file, validation_path.open("xb") as validation_file:
        for record in iter_label_records("train"):
            train_text_hashes.add(normalized_text_hash(record))
            record = with_derived_target(record, "train", "train")
            normalized_targets.append(record["derived_model_target"]["essay_score_1to5"])
            write_record(train_file, train_digest, record)
            train_count += 1
            source_counts["train"] += 1

        for record in iter_label_records("validation"):
            duplicate_with_train = normalized_text_hash(record) in train_text_hashes
            derived_split = "train" if duplicate_with_train else "validation"
            record = with_derived_target(record, "validation", derived_split)
            normalized_targets.append(record["derived_model_target"]["essay_score_1to5"])
            if duplicate_with_train:
                write_record(train_file, train_digest, record)
                train_count += 1
                moved_count += 1
            else:
                write_record(validation_file, validation_digest, record)
                validation_count += 1
            source_counts["validation"] += 1

    manifest = {
        "dataset": "024.에세이 글 평가 데이터",
        "derivation_version": "v1",
        "command": "python scripts/prepare_aihub_essay_1to5.py",
        "raw_input_root": "data/raw/aihub/024_essay_writing_evaluation",
        "derived_output_root": "data/processed/aihub_essay_1to5_v1",
        "split_rule": "Validation records with a normalized response-text hash present in raw training are reassigned to derived training.",
        "normalization": {
            "formula": "1 + 4 * (raw_score / 30)",
            "raw_range": [SCORE_MIN, SCORE_MAX],
            "output_range": [1.0, 5.0],
            "target_fields": ["derived_model_target.essay_score_1to5", "derived_model_target.essay_rater_scores_1to5"],
            "type": "continuous min-max normalization; original scores are retained unchanged",
        },
        "records": {
            "raw_train": source_counts["train"],
            "raw_validation": source_counts["validation"],
            "validation_reassigned_to_train": moved_count,
            "derived_train": train_count,
            "derived_validation": validation_count,
        },
        "derived_target_observed_range": [min(normalized_targets), max(normalized_targets)],
        "files": {
            "train.jsonl": {"sha256": train_digest.hexdigest(), "records": train_count},
            "validation.jsonl": {"sha256": validation_digest.hexdigest(), "records": validation_count},
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR.relative_to(ROOT)}")
    print(f"Wrote {MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
