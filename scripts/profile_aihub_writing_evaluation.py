#!/usr/bin/env python3
"""Profile the downloaded AI-Hub Korean writing-evaluation labels without extraction.

The script emits only aggregate statistics: it never writes response text, prompts,
feedback, identifiers, or other row-level data to its reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "aihub"
JSON_OUTPUT = ROOT / "data" / "reports" / "aihub_writing_evaluation_profile.json"
MARKDOWN_OUTPUT = ROOT / "docs" / "aihub_writing_evaluation_profile.md"


DATASETS = {
    "024_essay_writing_evaluation": {
        "name": "에세이 글 평가 데이터",
        "kind": "essay",
        "expected_archives": 20,
        "required": [
            ("info", "essay_id"),
            ("info", "essay_len"),
            ("info", "essay_level"),
            ("info", "essay_main_subject"),
            ("info", "essay_type"),
            ("paragraph",),
            ("score", "essay_scoreT"),
            ("score", "essay_scoreT_avg"),
        ],
        "categories": {
            "essay_type": ("info", "essay_type"),
            "essay_level": ("info", "essay_level"),
            "student_grade_group": ("student", "student_grade_group"),
        },
    },
    "025_descriptive_writing_evaluation": {
        "name": "서술형 글쓰기 평가 데이터",
        "kind": "school",
        "expected_archives": 80,
        "required": [
            ("essay_answer", "id"),
            ("essay_answer", "text"),
            ("essay_answer", "len_syllable"),
            ("essay_answer", "len_word"),
            ("essay_question", "id"),
            ("essay_question", "grade"),
            ("essay_question", "subject"),
            ("score", "personal", "holistic", "score"),
        ],
        "categories": {
            "grade": ("essay_question", "grade"),
            "subject": ("essay_question", "subject"),
            "question_type": ("essay_question", "type"),
            "question_level": ("essay_question", "level"),
            "answer_gender": ("essay_answer", "gender"),
            "answer_region": ("essay_answer", "region"),
        },
    },
    "026_argumentative_writing_evaluation": {
        "name": "논술형 글쓰기 평가 데이터",
        "kind": "school",
        "expected_archives": 48,
        "required": [
            ("essay_answer", "id"),
            ("essay_answer", "text"),
            ("essay_answer", "len_syllable"),
            ("essay_answer", "len_word"),
            ("essay_question", "id"),
            ("essay_question", "grade"),
            ("essay_question", "subject"),
            ("score", "personal", "holistic", "score"),
        ],
        "categories": {
            "grade": ("essay_question", "grade"),
            "subject": ("essay_question", "subject"),
            "question_type": ("essay_question", "type"),
            "question_level": ("essay_question", "level"),
            "answer_gender": ("essay_answer", "gender"),
            "answer_region": ("essay_answer", "region"),
        },
    },
}

MISSING = object()


def get_path(record: dict, path: tuple[str, ...]):
    value = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return value


def is_missing(value) -> bool:
    return value is MISSING or value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, (list, dict)) and not value
    )


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def q(p: float) -> float:
        index = (len(ordered) - 1) * p
        lo, hi = int(index), math.ceil(index)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {
        "count": len(ordered),
        "min": min(ordered),
        "p05": q(0.05),
        "p25": q(0.25),
        "median": q(0.50),
        "mean": statistics.fmean(ordered),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": max(ordered),
    }


def score_distribution(values: list[float], counter: Counter) -> dict:
    """Return a bounded distribution without exposing high-cardinality floats."""
    if len(counter) <= 20:
        return {"kind": "values", "counts": dict(sorted(counter.items(), key=lambda item: float(item[0])))}
    lower, upper = min(values), max(values)
    if lower == upper:
        return {"kind": "values", "counts": {str(lower): len(values)}}
    bins = 10
    width = (upper - lower) / bins
    counts = [0] * bins
    for value in values:
        index = min(int((value - lower) / width), bins - 1)
        counts[index] += 1
    return {
        "kind": "histogram",
        "bins": [
            {"lower": lower + index * width, "upper": lower + (index + 1) * width, "count": count}
            for index, count in enumerate(counts)
        ],
    }


def split_of(path: Path) -> str:
    return "train" if "Training" in path.as_posix() else "validation"


def member_stems(zf: ZipFile) -> set[str]:
    return {Path(name).stem for name in zf.namelist() if not name.endswith("/")}


@dataclass
class Profile:
    config: dict
    labels: Counter = field(default_factory=Counter)
    source_members: Counter = field(default_factory=Counter)
    pair_stems: dict = field(default_factory=lambda: defaultdict(lambda: {"label": set(), "source": set()}))
    parse_errors: int = 0
    missing: Counter = field(default_factory=Counter)
    category: dict = field(default_factory=lambda: defaultdict(Counter))
    ids: dict = field(default_factory=lambda: defaultdict(list))
    group_ids: dict = field(default_factory=lambda: defaultdict(list))
    text_hashes: dict = field(default_factory=lambda: defaultdict(list))
    length_values: dict = field(default_factory=lambda: defaultdict(list))
    score_values: dict = field(default_factory=lambda: defaultdict(list))
    score_counter: dict = field(default_factory=lambda: defaultdict(Counter))
    vector_length_errors: Counter = field(default_factory=Counter)
    records: Counter = field(default_factory=Counter)

    def inspect(self, record: dict, split: str):
        self.records[split] += 1
        for path in self.config["required"]:
            if is_missing(get_path(record, path)):
                self.missing[".".join(path)] += 1
        for name, path in self.config["categories"].items():
            value = get_path(record, path)
            if not is_missing(value):
                self.category[name][str(value).strip()] += 1

        if self.config["kind"] == "essay":
            identifier = get_path(record, ("info", "essay_id"))
            group_identifier = get_path(record, ("info", "essay_main_subject"))
            text = "\n".join(
                str(part.get("paragraph_txt", "")) for part in get_path(record, ("paragraph",)) if isinstance(part, dict)
            ) if isinstance(get_path(record, ("paragraph",)), list) else ""
            self.add_length("essay_len", get_path(record, ("info", "essay_len")))
            scores = get_path(record, ("score", "essay_scoreT"))
            average = get_path(record, ("score", "essay_scoreT_avg"))
            self.add_score("overall_mean", average)
            if isinstance(scores, list):
                if len(scores) != 3:
                    self.vector_length_errors["essay_scoreT"] += 1
                for score in scores:
                    self.add_score("rater_score", score)
                numeric_scores = [float(score) for score in scores if number(score)]
                if len(numeric_scores) > 1:
                    self.add_score("within_record_rater_sd", statistics.pstdev(numeric_scores))
            else:
                self.vector_length_errors["essay_scoreT_nonlist"] += 1
        else:
            identifier = get_path(record, ("essay_answer", "id"))
            group_identifier = get_path(record, ("essay_question", "id"))
            text = get_path(record, ("essay_answer", "text"))
            self.add_length("text_characters", len(text) if isinstance(text, str) else None)
            self.add_length("len_syllable", get_path(record, ("essay_answer", "len_syllable")))
            self.add_length("len_word", get_path(record, ("essay_answer", "len_word")))
            scores = get_path(record, ("score", "personal", "holistic", "score"))
            if isinstance(scores, list):
                if len(scores) != 2:
                    self.vector_length_errors["holistic_score"] += 1
                for score in scores:
                    self.add_score("holistic_rater_score", score)
                if len(scores) == 2 and all(number(score) for score in scores):
                    self.add_score("holistic_mean", statistics.fmean(scores))
                    self.add_score("holistic_rater_abs_difference", abs(scores[0] - scores[1]))
            else:
                self.vector_length_errors["holistic_score_nonlist"] += 1

        if not is_missing(identifier):
            self.ids[split].append(str(identifier))
        if isinstance(group_identifier, str) and group_identifier.strip():
            # Hash free-text essay prompts so reports remain aggregate-only.
            self.group_ids[split].append(hashlib.sha256(group_identifier.strip().encode("utf-8")).hexdigest())
        elif not is_missing(group_identifier):
            self.group_ids[split].append(str(group_identifier))
        if isinstance(text, str) and text.strip():
            self.text_hashes[split].append(hashlib.sha256(text.strip().encode("utf-8")).hexdigest())

    def add_length(self, name: str, value):
        if number(value):
            self.length_values[name].append(float(value))

    def add_score(self, name: str, value):
        if number(value):
            value = float(value)
            self.score_values[name].append(value)
            self.score_counter[name][str(value)] += 1


def profile_dataset(directory: Path, config: dict) -> dict:
    profile = Profile(config)
    archives = sorted(directory.rglob("*.zip"))
    if len(archives) != config["expected_archives"]:
        raise RuntimeError(f"{directory.name}: expected {config['expected_archives']} ZIPs, found {len(archives)}")

    for archive in archives:
        split = split_of(archive)
        is_label = "라벨링데이터" in archive.as_posix()
        with ZipFile(archive) as zf:
            stems = member_stems(zf)
            if is_label:
                profile.pair_stems[split]["label"].update(stems)
                for member in (item for item in zf.namelist() if item.lower().endswith(".json")):
                    try:
                        record = json.loads(zf.read(member).decode("utf-8-sig"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        profile.parse_errors += 1
                        continue
                    if not isinstance(record, dict):
                        profile.parse_errors += 1
                        continue
                    profile.labels[split] += 1
                    profile.inspect(record, split)
            else:
                profile.pair_stems[split]["source"].update(stems)
                profile.source_members[split] += len(stems)

    def duplicate_stats(values: list[str]) -> dict:
        unique = set(values)
        return {"count": len(values), "distinct": len(unique), "duplicate_rows": len(values) - len(unique)}

    split_integrity = {}
    for split in ("train", "validation"):
        labels = profile.pair_stems[split]["label"]
        source = profile.pair_stems[split]["source"]
        split_integrity[split] = {
            "label_members": profile.labels[split],
            "source_members": profile.source_members[split],
            "label_source_stem_mismatch": len(labels.symmetric_difference(source)),
            "candidate_id": duplicate_stats(profile.ids[split]),
            "question_or_prompt_group": duplicate_stats(profile.group_ids[split]),
            "normalized_text": duplicate_stats(profile.text_hashes[split]),
        }
    split_integrity["cross_split"] = {
        "candidate_id_overlap": len(set(profile.ids["train"]).intersection(profile.ids["validation"])),
        "question_or_prompt_group_overlap": len(set(profile.group_ids["train"]).intersection(profile.group_ids["validation"])),
        "normalized_text_overlap": len(set(profile.text_hashes["train"]).intersection(profile.text_hashes["validation"])),
    }

    total = sum(profile.records.values())
    return {
        "display_name": config["name"],
        "zip_archives": len(archives),
        "records": dict(profile.records),
        "json_parse_errors": profile.parse_errors,
        "required_field_missing": {
            path: {"count": count, "rate": count / total if total else 0} for path, count in sorted(profile.missing.items())
        },
        "categories": {name: dict(sorted(counter.items())) for name, counter in profile.category.items()},
        "lengths": {name: summary(values) for name, values in profile.length_values.items()},
        "scores": {
            name: {"summary": summary(values), "distribution": score_distribution(values, counter)}
            for name, (values, counter) in ((name, (profile.score_values[name], profile.score_counter[name])) for name in profile.score_values)
        },
        "vector_length_errors": dict(profile.vector_length_errors),
        "integrity": split_integrity,
    }


def markdown(report: dict) -> str:
    lines = [
        "# AI-Hub Korean Writing Evaluation Data Profile",
        "",
        "## Scope and method",
        "",
        "This profile scans every label JSON directly inside the downloaded ZIP archives. It records aggregate structure, completeness, category, length, score, and split-integrity statistics only; no response text, prompts, feedback, or identifiers are emitted.",
        "",
        "## Dataset inventory",
        "",
        "| Dataset | Labels (train / validation) | ZIPs | JSON parse errors |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in report["datasets"].values():
        records = result["records"]
        lines.append(f"| {result['display_name']} | {records.get('train', 0):,} / {records.get('validation', 0):,} | {result['zip_archives']} | {result['json_parse_errors']} |")

    for result in report["datasets"].values():
        lines += ["", f"## {result['display_name']}", "", "### Completeness and split integrity", ""]
        lines.append("- Required-field missingness: " + ("none observed" if not result["required_field_missing"] else json.dumps(result["required_field_missing"], ensure_ascii=False)))
        integrity = result["integrity"]
        lines.append(f"- Train label/source filename mismatches: {integrity['train']['label_source_stem_mismatch']:,}; validation: {integrity['validation']['label_source_stem_mismatch']:,}.")
        lines.append(f"- Candidate-ID overlap across train/validation: {integrity['cross_split']['candidate_id_overlap']:,}; normalized-text overlap: {integrity['cross_split']['normalized_text_overlap']:,}.")
        lines.append(f"- Question/prompt-group overlap across train/validation: {integrity['cross_split']['question_or_prompt_group_overlap']:,} (prompt text is hashed for this check).")
        lines.append(f"- Candidate-ID duplicate rows (train/validation): {integrity['train']['candidate_id']['duplicate_rows']:,} / {integrity['validation']['candidate_id']['duplicate_rows']:,}.")
        lines.append(f"- Normalized-text duplicate rows (train/validation): {integrity['train']['normalized_text']['duplicate_rows']:,} / {integrity['validation']['normalized_text']['duplicate_rows']:,}.")

        lines += ["", "### Categorical composition", ""]
        for name, values in result["categories"].items():
            formatted = ", ".join(f"{key}: {value:,}" for key, value in values.items())
            lines.append(f"- **{name}** — {formatted}")

        lines += ["", "### Length and score distributions", ""]
        for name, values in result["lengths"].items():
            lines.append(f"- **{name}** — n={values['count']:,}, min/p05/median/p95/max = {values['min']:.2f}/{values['p05']:.2f}/{values['median']:.2f}/{values['p95']:.2f}/{values['max']:.2f}, mean={values['mean']:.2f}.")
        for name, values in result["scores"].items():
            summary_values = values["summary"]
            distribution = values["distribution"]
            if distribution["kind"] == "values":
                rendered = ", ".join(f"{score}: {count:,}" for score, count in distribution["counts"].items())
                suffix = f" values: {rendered}."
            else:
                rendered = ", ".join(
                    f"[{bin_['lower']:.2f}, {bin_['upper']:.2f}{']' if index == len(distribution['bins']) - 1 else ')'}: {bin_['count']:,}"
                    for index, bin_ in enumerate(distribution["bins"])
                )
                suffix = f" 10-bin histogram: {rendered}."
            lines.append(f"- **{name}** — n={summary_values['count']:,}, mean={summary_values['mean']:.3f}, median={summary_values['median']:.3f}, range={summary_values['min']:.3f}–{summary_values['max']:.3f};{suffix}")
        if result["vector_length_errors"]:
            lines.append(f"- Score-vector length anomalies: {json.dumps(result['vector_length_errors'], ensure_ascii=False)}")

    lines += ["", "## Caveats", "", "- Filename pairing and normalized-text checks are ingestion checks, not a semantic leakage audit. Verify IDs and any near-duplicate responses again after creating the modeling table.", "- Score meaning and aggregation should be confirmed against AI-Hub's rubric documentation before selecting a training target.", ""]
    return "\n".join(lines)


def main() -> None:
    results = {}
    for directory_name, config in DATASETS.items():
        results[directory_name] = profile_dataset(RAW_ROOT / directory_name, config)
    report = {"method": "ZIP-internal full-label profile", "datasets": results}
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_OUTPUT.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {JSON_OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {MARKDOWN_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
