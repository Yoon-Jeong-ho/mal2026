"""Train-only Solar axis-degradation contract and strict output validation."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/solar_axis_degradation_prompt.v1.json"
EVALUATION_PATH = ROOT / "evaluation.txt"
TRAIN_PATH = ROOT / "eval/train.jsonl"
AXES = ("content", "organization", "expression")
EXPECTED_CONFIG_SHA256 = "5b1d436d4e31e194bf3a617d3489e8e2be3768fc760bbed5c45f26a81b58d422"
EXPECTED_EVALUATION_SHA256 = "1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145"
EXPECTED_TRAIN_SHA256 = "b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737"


class SolarAxisAugmentationError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SolarAxisAugmentationError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_config() -> dict[str, Any]:
    need(CONFIG_PATH.is_file() and not CONFIG_PATH.is_symlink(), "Solar prompt config is unavailable")
    need(file_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256, "Solar prompt config checksum differs")
    need(file_sha256(EVALUATION_PATH) == EXPECTED_EVALUATION_SHA256, "evaluation.txt checksum differs")
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    need(raw.get("schema_version") == "mal2026-solar-axis-degradation-prompt-v1", "Solar prompt schema differs")
    need(raw.get("input_contract", {}).get("allowed_split") == "train_only", "Solar augmentation escaped train")
    need(raw.get("input_contract", {}).get("target_axes") == list(AXES), "Solar target axes differ")
    need(raw.get("input_contract", {}).get("variants_per_source") == 3, "Solar variant count differs")
    need(raw.get("input_contract", {}).get("gold_average_allowed") is False, "Solar prompt permits average")
    return raw


@dataclass(frozen=True)
class SourceRow:
    identifier: str
    prompt: str
    essay: str
    score: tuple[float, float, float]


def load_train_rows() -> list[SourceRow]:
    need(TRAIN_PATH.is_file() and file_sha256(TRAIN_PATH) == EXPECTED_TRAIN_SHA256, "canonical train differs")
    rows: list[SourceRow] = []
    seen: set[str] = set()
    with TRAIN_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            identifier = raw.get("id")
            need(isinstance(identifier, str) and identifier and identifier not in seen, "train ID differs")
            seen.add(identifier)
            score = raw.get("score")
            need(isinstance(score, dict) and set(score) == {*AXES, "average"}, "train score schema differs")
            values = tuple(float(score[axis]) for axis in AXES)
            need(all(math.isfinite(value) and 1 <= value <= 5 for value in values), "train axis score differs")
            need(isinstance(raw.get("prompt"), str) and raw["prompt"].strip(), "train prompt is blank")
            need(isinstance(raw.get("essay"), str) and raw["essay"].strip(), "train essay is blank")
            rows.append(SourceRow(identifier, raw["prompt"], raw["essay"], values))
    need(len(rows) == 2000, "canonical train population differs")
    return rows


def requested_drop(source_id: str, axis: str) -> float:
    need(axis in AXES, "unknown degradation axis")
    choices = (0.75, 1.25, 1.75)
    index = int(sha256(f"2026072904\0{source_id}\0{axis}".encode()).hexdigest()[:8], 16) % len(choices)
    return choices[index]


def render_messages(row: SourceRow, axis: str) -> list[dict[str, str]]:
    config = prompt_config()
    drop = requested_drop(row.identifier, axis)
    baseline = {name: row.score[index] for index, name in enumerate(AXES)}
    payload = {
        "axis_degradation_input": {
            "source_id": row.identifier,
            "prompt_text": row.prompt,
            "essay_text": row.essay,
            "baseline_score": baseline,
            "target_axis": axis,
            "degradation_goal": {
                "target_score_upper_bound": max(1.0, baseline[axis] - drop),
                "non_target_max_absolute_change": config["quality_gates"]["non_target_max_absolute_change"],
            },
        }
    }
    return [
        {"role": "system", "content": config["messages"]["system"]},
        {
            "role": "user",
            "content": config["messages"]["user_preamble"] + "\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def parse_output(content: str, row: SourceRow, axis: str) -> dict[str, Any]:
    config = prompt_config()
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SolarAxisAugmentationError("Solar output is not JSON") from exc
    need(isinstance(raw, dict) and set(raw) == {"augmented_essay", "score"}, "Solar output keys differ")
    essay = raw["augmented_essay"]
    need(isinstance(essay, str) and len(essay.strip()) >= 20 and essay.strip() != row.essay.strip(), "Solar essay differs")
    ratio = len(essay.strip()) / len(row.essay.strip())
    gates = config["quality_gates"]
    need(gates["minimum_length_ratio"] <= ratio <= gates["maximum_length_ratio"], "Solar essay length ratio differs")
    score = raw["score"]
    need(isinstance(score, dict) and set(score) == set(AXES), "Solar score axes differ")
    parsed: dict[str, float] = {}
    for name in AXES:
        value = score[name]
        need(type(value) in {int, float} and not isinstance(value, bool), "Solar score is nonnumeric")
        value = float(value)
        need(math.isfinite(value) and 1 <= value <= 5 and abs(value * 4 - round(value * 4)) < 1e-8, "Solar score grid differs")
        parsed[name] = value
    baseline = {name: row.score[index] for index, name in enumerate(AXES)}
    upper = max(1.0, baseline[axis] - requested_drop(row.identifier, axis))
    need(parsed[axis] <= upper + 1e-8, "Solar target axis did not degrade enough")
    for name in AXES:
        if name != axis:
            need(abs(parsed[name] - baseline[name]) <= gates["non_target_max_absolute_change"] + 1e-8, "Solar non-target axis changed too much")
    return {"augmented_essay": essay.strip(), "score": parsed}


def output_schema() -> Mapping[str, Any]:
    return prompt_config()["output_contract"]


def task_count(rows: Sequence[SourceRow]) -> int:
    need(len(rows) == 2000, "Solar source population differs")
    return len(rows) * len(AXES)
