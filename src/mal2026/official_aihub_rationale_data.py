"""Official-rationale adaptation of the closest existing AI-Hub feedback.

Only the argumentative corpus from the canonical, already prepared AI-Hub
Training data is admitted.  Human analytic feedback is mapped to the three
official axes; holistic/task feedback is excluded because it mixes domains or
acts as improvement advice.  Row data remains in memory and in ignored roots.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .api_rationale_data import AXES, ROOT
from .official_rationale_data import axes_for_task, messages, rationale_object
from .official_writing_contract import integerize_score
from .standard_decoder_data import DEFAULT_MANIFEST, RestrictedRow, load_prepared_split


MANIFEST = ROOT / "data/manifests/aihub_argumentative_official_rationale_v1.json"
EXPECTED_COUNTS = {"selection_train": 12_813, "selection_dev": 3_197, "refit_train": 16_010}
FEEDBACK_BY_AXIS = {
    "content": ("content_1", "content_2", "content_3"),
    "organization": ("organization_1", "organization_2"),
    "expression": ("expression_1", "expression_2"),
}


class OfficialAIHubRationaleError(ValueError):
    """Raised when the fixed AI-Hub rationale projection differs."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialAIHubRationaleError(message)


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance() -> dict[str, Any]:
    need(MANIFEST.is_file() and DEFAULT_MANIFEST.is_file(), "AI-Hub rationale manifest is unavailable")
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    need(
        value.get("schema_version") == 1
        and value.get("dataset_id") == "aihub_argumentative_official_rationale_v1"
        and value.get("source", {}).get("prepared_manifest_sha256") == file_sha(DEFAULT_MANIFEST)
        and value.get("selection", {}).get("identifier_namespace_prefix") == "argumentative:",
        "AI-Hub rationale provenance differs",
    )
    counts = {
        split: int(value.get("selection", {}).get("splits", {}).get(split, {}).get("record_count", -1))
        for split in EXPECTED_COUNTS
    }
    need(counts == EXPECTED_COUNTS, "AI-Hub rationale split counts differ")
    contract = value.get("rationale_contract", {})
    need(
        tuple(contract.get("content_feedback_fields", ())) == FEEDBACK_BY_AXIS["content"]
        and tuple(contract.get("organization_feedback_fields", ())) == FEEDBACK_BY_AXIS["organization"]
        and tuple(contract.get("expression_feedback_fields", ())) == FEEDBACK_BY_AXIS["expression"]
        and contract.get("excluded_feedback_fields") == ["holistic", "task_1"],
        "AI-Hub rationale feedback mapping differs",
    )
    return {
        "dataset_id": value["dataset_id"],
        "manifest_sha256": file_sha(MANIFEST),
        "prepared_manifest_sha256": file_sha(DEFAULT_MANIFEST),
        "counts": counts,
        "included_corpus": "argumentative",
        "upstream_split": "Training_only",
        "integer_score_projection": "clip_1_5_round_half_up",
        "average_excluded": True,
    }


def load_argumentative(split: str) -> list[RestrictedRow]:
    provenance()
    need(split in EXPECTED_COUNTS, "AI-Hub rationale split differs")
    rows = [row for row in load_prepared_split(split) if row.identifier.startswith("argumentative:")]
    need(len(rows) == EXPECTED_COUNTS[split] and len({row.identifier for row in rows}) == len(rows), "AI-Hub argumentative population differs")
    return rows


def _join_feedback(row: RestrictedRow, axis: str) -> str:
    need(row.feedback is not None and axis in FEEDBACK_BY_AXIS, "AI-Hub feedback is unavailable")
    selected: list[str] = []
    for field in FEEDBACK_BY_AXIS[axis]:
        text = row.feedback[field].strip()
        need(bool(text), "AI-Hub analytic feedback is blank")
        if text not in selected:
            selected.append(text)
    result = " ".join(selected)
    need(bool(result), "AI-Hub axis rationale is blank")
    return result


def projected_scores(row: RestrictedRow) -> dict[str, int]:
    need(set(row.score) >= set(AXES), "AI-Hub score axes differ")
    return {axis: integerize_score(row.score[axis]) for axis in AXES}


def projected_rationales(row: RestrictedRow) -> dict[str, str]:
    return {axis: _join_feedback(row, axis) for axis in AXES}


def sft_examples(task: str, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    axes = axes_for_task(task)
    rows = load_argumentative(split)
    if limit is not None:
        need(0 < limit <= len(rows), "AI-Hub rationale limit differs")
        rows = rows[:limit]
    examples: list[dict[str, Any]] = []
    for row in rows:
        scores = projected_scores(row)
        rationales = projected_rationales(row)
        examples.append({
            "prompt": messages(row.prompt, row.essay, scores, axes),
            "completion": [{"role": "assistant", "content": json.dumps(rationale_object(rationales, axes), ensure_ascii=False, separators=(",", ":"))}],
        })
    need(len(examples) == (limit if limit is not None else EXPECTED_COUNTS[split]), "AI-Hub rationale example count differs")
    return examples


def structure_sft_examples(structure: str, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Render AI-Hub pretraining examples for the selected deployment structure.

    The bundle structure uses one three-axis target per essay.  The axis-triplet
    structure uses all three analytic rationale groups as three single-axis
    examples per essay so full-parameter pretraining does not discard two
    thirds of the human feedback.
    """
    need(structure in {"bundle", "axis_triplet"}, "AI-Hub rationale structure differs")
    rows = load_argumentative(split)
    expected = len(rows) if structure == "bundle" else len(rows) * len(AXES)
    examples: list[dict[str, Any]] = []
    for row in rows:
        scores = projected_scores(row)
        rationales = projected_rationales(row)
        choices: tuple[tuple[str, ...], ...] = (AXES,) if structure == "bundle" else tuple((axis,) for axis in AXES)
        for axes in choices:
            examples.append({
                "prompt": messages(row.prompt, row.essay, scores, axes),
                "completion": [{"role": "assistant", "content": json.dumps(rationale_object(rationales, axes), ensure_ascii=False, separators=(",", ":"))}],
            })
            if limit is not None and len(examples) >= limit:
                need(0 < limit <= expected, "AI-Hub rationale structure limit differs")
                return examples
    need(len(examples) == expected and (limit is None or len(examples) == limit), "AI-Hub rationale structure example count differs")
    return examples
