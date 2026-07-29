"""Bundle rationale contract for Solar-generated train-only essays.

The selected DPO policy remains a single bundled three-axis rationale model.
Solar pseudo scores are continuous quarter-step synthetic supervision; no
human/reference score and no canonical validation row enters this stage.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .official_rationale_data import parse_rationale_output, rationale_schema
from .solar_axis_augmentation import AXES, SourceRow, file_sha256, load_train_rows


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/solar_augmented_bundle_rationale_prompt.v1.json"
EXPECTED_CONFIG_SHA256 = "56269ad840cd35b5932577c3c5944d069301e8f07bdd8d2dd31f66f3d816932b"
SOLAR_RUN_ID = "solar-open2-axis-degradation-train-v1-20260729-004"
SOLAR_RESULT = ROOT / "outputs/solar-axis-degradation-v1" / SOLAR_RUN_ID / "result.json"
RESTRICTED_ROOT = (ROOT / "data/processed/restricted").resolve()


class AugmentedBundleRationaleError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise AugmentedBundleRationaleError(message)


def config() -> dict[str, Any]:
    need(CONFIG_PATH.is_file() and not CONFIG_PATH.is_symlink(), "augmented rationale prompt is unavailable")
    need(file_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256, "augmented rationale prompt checksum differs")
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    need(raw.get("schema_version") == "mal2026-solar-augmented-bundle-rationale-prompt-v1", "prompt schema differs")
    provenance = raw.get("provenance", {})
    output = raw.get("output_contract", {})
    inputs = raw.get("input_contract", {})
    need(provenance.get("rationale_structure") == "bundle", "rationale structure differs")
    need(provenance.get("axis_triplet_allowed") is False, "axis-triplet is permitted")
    need(provenance.get("human_or_reference_score_allowed") is False, "protected score is permitted")
    need(output.get("structure") == "bundle" and output.get("axis_triplet_allowed") is False, "output is not bundle-only")
    need(output.get("score_output_allowed") is False, "rationale output permits a score")
    need(output.get("required_axes") == list(AXES), "rationale axes differ")
    need(inputs.get("allowed_split") == "train_only" and inputs.get("average_allowed") is False, "input split/average differs")
    need(inputs.get("pseudo_score_axes") == list(AXES), "pseudo-score axes differ")
    need(inputs.get("pseudo_score_step") == 0.25, "pseudo-score grid differs")
    return raw


@dataclass(frozen=True)
class AugmentedRow:
    source_id: str
    target_axis: str
    identifier: str
    prompt: str
    essay: str
    score: tuple[float, float, float]
    attempts: int


def _quarter_score(raw: Mapping[str, Any]) -> tuple[float, float, float]:
    need(set(raw) == set(AXES), "Solar pseudo-score axes differ")
    values: list[float] = []
    for axis in AXES:
        value = raw[axis]
        need(type(value) in {int, float} and not isinstance(value, bool), "Solar pseudo score is nonnumeric")
        value = float(value)
        need(math.isfinite(value) and 1 <= value <= 5 and abs(value * 4 - round(value * 4)) < 1e-8, "Solar pseudo-score grid differs")
        values.append(value)
    return tuple(values)  # type: ignore[return-value]


def validate_records(raw_rows: Iterable[Mapping[str, Any]], sources: Sequence[SourceRow]) -> list[AugmentedRow]:
    source_map = {row.identifier: row for row in sources}
    need(len(source_map) == 2000, "canonical source population differs")
    seen: set[str] = set()
    coverage: dict[str, set[str]] = {}
    rows: list[AugmentedRow] = []
    expected_fields = {"source_id", "target_axis", "augmented_id", "prompt", "essay", "score", "attempts"}
    for raw in raw_rows:
        need(set(raw) == expected_fields, "Solar augmented row schema differs")
        source_id, axis, identifier = raw["source_id"], raw["target_axis"], raw["augmented_id"]
        need(isinstance(source_id, str) and source_id in source_map, "Solar source linkage differs")
        need(axis in AXES, "Solar target axis differs")
        need(identifier == f"{source_id}::solar-degrade::{axis}" and identifier not in seen, "Solar augmented ID differs")
        need(raw["prompt"] == source_map[source_id].prompt, "Solar prompt linkage differs")
        essay = raw["essay"]
        need(isinstance(essay, str) and essay.strip() and essay.strip() != source_map[source_id].essay.strip(), "Solar augmented essay differs")
        attempts = raw["attempts"]
        need(type(attempts) is int and 1 <= attempts <= 3, "Solar generation attempts differ")
        score = raw["score"]
        need(isinstance(score, Mapping), "Solar pseudo score is not an object")
        seen.add(identifier)
        coverage.setdefault(source_id, set()).add(axis)
        rows.append(AugmentedRow(source_id, axis, identifier, raw["prompt"], essay.strip(), _quarter_score(score), attempts))
    need(len(rows) == 6000 and set(coverage) == set(source_map), "Solar augmented population differs")
    need(all(axes == set(AXES) for axes in coverage.values()), "Solar three-axis coverage differs")
    rows.sort(key=lambda row: (row.source_id, AXES.index(row.target_axis)))
    return rows


def load_completed_solar() -> tuple[list[AugmentedRow], dict[str, Any]]:
    need(SOLAR_RESULT.is_file() and not SOLAR_RESULT.is_symlink(), "completed Solar result is unavailable")
    result = json.loads(SOLAR_RESULT.read_text(encoding="utf-8"))
    need(isinstance(result, dict) and result.get("schema_version") == "mal2026-solar-axis-degradation-result-v1", "Solar result schema differs")
    need(result.get("status") == "completed" and result.get("run_id") == SOLAR_RUN_ID, "Solar result is not completed")
    need(result.get("records") == 6000 and result.get("source_records") == 2000 and result.get("variants_per_source") == 3, "Solar counts differ")
    path = Path(str(result.get("augmented_train_path", "")))
    need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(RESTRICTED_ROOT), "Solar augmented path differs")
    need(file_sha256(path) == result.get("augmented_train_sha256"), "Solar augmented checksum differs")
    with path.open(encoding="utf-8") as handle:
        raw_rows = [json.loads(line) for line in handle if line.strip()]
    rows = validate_records(raw_rows, load_train_rows())
    return rows, result


def render_messages(row: AugmentedRow) -> list[dict[str, str]]:
    prompt = config()
    scores = {axis: row.score[index] for index, axis in enumerate(AXES)}
    payload = {
        "augmented_rationale_input": {
            "source_id": row.identifier,
            "target_axis": row.target_axis,
            "prompt_text": row.prompt,
            "essay_text": row.essay,
            "pseudo_score": scores,
        }
    }
    return [
        {"role": "system", "content": prompt["messages"]["system"]},
        {"role": "user", "content": prompt["messages"]["user_preamble"] + "\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def output_schema() -> Mapping[str, Any]:
    limit = config()["output_contract"]["rationale_character_limit_per_axis"]
    return rationale_schema(AXES, character_limit=int(limit))


def parse_output(value: str) -> dict[str, str]:
    parsed = parse_rationale_output(value, AXES)
    limit = int(config()["output_contract"]["rationale_character_limit_per_axis"])
    need(all(len(parsed[axis]) <= limit for axis in AXES), "augmented rationale exceeds field limit")
    return parsed
