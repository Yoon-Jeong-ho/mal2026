#!/usr/bin/env python3
"""Build prompt-disjoint 80:20 train/validation splits for all writing datasets.

The raw archives remain unchanged.  Derived JSONL files contain protected writing
content and are therefore placed under the Git-ignored data/processed/ directory.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "aihub"
OUTPUT_ROOT = ROOT / "data" / "processed" / "aihub_writing_group_split_v1"
MANIFEST_PATH = ROOT / "data" / "manifests" / "aihub_writing_group_split_v1.json"
SEED = "2026"
VALIDATION_GROUP_FRACTION = 0.20


DATASETS = {
    "024_essay_writing_evaluation": {
        "name": "024.에세이 글 평가 데이터",
        "group_path": ("info", "essay_main_subject"),
        "stratum_path": ("info", "essay_type"),
        "normalize_essay_score": True,
    },
    "025_descriptive_writing_evaluation": {
        "name": "25.서술형 글쓰기 평가 데이터",
        "group_path": ("essay_question", "id"),
        "stratum_path": ("essay_question", "subject"),
        "normalize_essay_score": False,
    },
    "026_argumentative_writing_evaluation": {
        "name": "26.논술형 글쓰기 평가 데이터",
        "group_path": ("essay_question", "id"),
        "stratum_path": ("essay_question", "subject"),
        "normalize_essay_score": False,
    },
}


def get_path(record: dict, path: tuple[str, ...]):
    value = record
    for key in path:
        value = value[key]
    return value


def group_hash(group_value) -> str:
    """Hash a group identifier to keep manifests free of prompt text/IDs."""
    return hashlib.sha256(str(group_value).encode("utf-8")).hexdigest()


def stable_order(value) -> str:
    return hashlib.sha256(f"{SEED}\0{value}".encode("utf-8")).hexdigest()


def iter_label_records(dataset_dir: Path):
    for archive in sorted(path for path in dataset_dir.rglob("*.zip") if "라벨링데이터" in path.as_posix()):
        source_split = "train" if "Training" in archive.as_posix() else "validation"
        with ZipFile(archive) as zf:
            for member in sorted(name for name in zf.namelist() if name.endswith(".json")):
                yield json.loads(zf.read(member).decode("utf-8-sig")), source_split


def select_validation_groups(group_counts: dict, group_strata: dict):
    """Select about 20% of prompt groups in each task/subject stratum."""
    by_stratum = defaultdict(list)
    for group, count in group_counts.items():
        by_stratum[group_strata[group]].append((group, count))

    selected = set()
    split_stats = {}
    for stratum, groups in sorted(by_stratum.items()):
        groups = sorted(groups, key=lambda item: stable_order(item[0]))
        validation_groups = max(1, round(len(groups) * VALIDATION_GROUP_FRACTION))
        for group, _ in groups[:validation_groups]:
            selected.add(group)
        split_stats[str(stratum)] = {
            "all_groups": len(groups),
            "validation_groups": validation_groups,
            "all_records": sum(count for _, count in groups),
            "validation_records": sum(count for _, count in groups[:validation_groups]),
        }
    return selected, split_stats


def add_essay_target(record: dict) -> None:
    score = record["score"]
    average = float(score["essay_scoreT_avg"])
    rater_scores = [float(value) for value in score["essay_scoreT"]]
    if not 0.0 <= average <= 30.0 or len(rater_scores) != 3 or any(not 0.0 <= value <= 30.0 for value in rater_scores):
        raise ValueError("Unexpected essay score shape or range")
    record["derived_model_target"] = {
        "essay_score_1to5": 1.0 + 4.0 * average / 30.0,
        "essay_rater_scores_1to5": [1.0 + 4.0 * value / 30.0 for value in rater_scores],
        "normalization": "1 + 4 * (raw_score / 30)",
        "raw_score_range": [0.0, 30.0],
    }


def write_jsonl(handle, digest, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    encoded = line.encode("utf-8")
    handle.write(encoded)
    digest.update(encoded)


def process_dataset(directory_name: str, config: dict) -> dict:
    dataset_dir = RAW_ROOT / directory_name
    group_counts = Counter()
    group_strata = {}
    for record, _ in iter_label_records(dataset_dir):
        group = str(get_path(record, config["group_path"]))
        stratum = str(get_path(record, config["stratum_path"]))
        group_counts[group] += 1
        previous = group_strata.setdefault(group, stratum)
        if previous != stratum:
            raise ValueError(f"Prompt group maps to more than one stratum in {directory_name}")

    validation_groups, strata = select_validation_groups(group_counts, group_strata)
    dataset_output = OUTPUT_ROOT / directory_name
    dataset_output.mkdir(parents=True)
    digests = {"train": hashlib.sha256(), "validation": hashlib.sha256()}
    records = Counter()
    source_split_counts = {"train": Counter(), "validation": Counter()}

    with (dataset_output / "train.jsonl").open("xb") as train_handle, (dataset_output / "validation.jsonl").open("xb") as validation_handle:
        handles = {"train": train_handle, "validation": validation_handle}
        for record, source_split in iter_label_records(dataset_dir):
            group = str(get_path(record, config["group_path"]))
            derived_split = "validation" if group in validation_groups else "train"
            if config["normalize_essay_score"]:
                add_essay_target(record)
            record["derived_split"] = derived_split
            record["source_split"] = source_split
            record["group_split"] = {
                "version": "v1",
                "group_hash": group_hash(group),
                "stratum": str(get_path(record, config["stratum_path"])),
            }
            write_jsonl(handles[derived_split], digests[derived_split], record)
            records[derived_split] += 1
            source_split_counts[derived_split][source_split] += 1

    return {
        "name": config["name"],
        "group_key": "prompt hash" if directory_name.startswith("024_") else "essay_question.id hash",
        "stratification": "essay_type" if directory_name.startswith("024_") else "essay_question.subject",
        "groups": {"total": len(group_counts), "validation": len(validation_groups), "train": len(group_counts) - len(validation_groups)},
        "records": {"train": records["train"], "validation": records["validation"], "validation_fraction": records["validation"] / sum(records.values())},
        "strata": strata,
        "derived_files": {
            "train.jsonl": {"records": records["train"], "sha256": digests["train"].hexdigest(), "source_split_counts": dict(source_split_counts["train"])},
            "validation.jsonl": {"records": records["validation"], "sha256": digests["validation"].hexdigest(), "source_split_counts": dict(source_split_counts["validation"])},
        },
    }


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise SystemExit(f"Refusing to overwrite existing derived data: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    result = {name: process_dataset(name, config) for name, config in DATASETS.items()}
    manifest = {
        "derivation_version": "v1",
        "command": "python scripts/prepare_aihub_group_split.py",
        "raw_input_root": "data/raw/aihub",
        "derived_output_root": "data/processed/aihub_writing_group_split_v1",
        "seed": SEED,
        "split": "prompt-disjoint, stratified 80:20 train/validation split after pooling the original AI-Hub train and validation labels",
        "datasets": result,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_ROOT.relative_to(ROOT)}")
    print(f"Wrote {MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
