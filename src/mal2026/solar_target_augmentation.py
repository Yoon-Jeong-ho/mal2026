"""Version-2 train-only Solar target-score augmentation contracts.

The editor, score verifier, and source-fidelity auditor are deliberately
separate calls.  In particular, neither verifier sees the requested target.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from difflib import SequenceMatcher
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from .evaluation_prompt_matrix import evaluation_sections


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/solar_axis_target_score_prompt.v2.json"
EVALUATION_PATH = ROOT / "evaluation.txt"
TRAIN_PATH = ROOT / "eval/train.jsonl"
VALIDATION_PATH = ROOT / "eval/validation.jsonl"

AXES = ("content", "organization", "expression")
TARGET_SCORES = (1, 2, 3, 4, 5)
EXPECTED_CONFIG_SHA256 = "87427079e91909c899fe5ab5e60a32981775400dfe685475a3db652f05c91d9b"
EXPECTED_EVALUATION_SHA256 = "1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145"
EXPECTED_TRAIN_SHA256 = "b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737"
EXPECTED_VALIDATION_SHA256 = "0805445029328848164cf15f34b90b88fb5f7896d7b73f24f1717b733a9a00a4"


class SolarTargetAugmentationError(RuntimeError):
    """Raised when a v2 augmentation contract is violated."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise SolarTargetAugmentationError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_config() -> dict[str, Any]:
    _need(CONFIG_PATH.is_file() and not CONFIG_PATH.is_symlink(), "v2 prompt config is unavailable")
    _need(file_sha256(CONFIG_PATH) == EXPECTED_CONFIG_SHA256, "v2 prompt config checksum differs")
    _need(EVALUATION_PATH.is_file() and file_sha256(EVALUATION_PATH) == EXPECTED_EVALUATION_SHA256,
          "evaluation.txt checksum differs")
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    contract = raw.get("input_contract", {})
    provenance = raw.get("provenance", {})
    _need(raw.get("schema_version") == "mal2026-solar-axis-target-score-prompt-v2", "v2 schema differs")
    _need(provenance.get("rubric_source_sha256") == EXPECTED_EVALUATION_SHA256, "rubric binding differs")
    _need(contract.get("allowed_split") == "train_only", "augmentation escaped train")
    _need(contract.get("canonical_source_count") == 2000, "source count contract differs")
    _need(contract.get("target_axes") == list(AXES), "target axes differ")
    _need(contract.get("target_scores") == list(TARGET_SCORES), "target scores differ")
    _need(contract.get("variants_per_source") == 15, "variant count differs")
    _need(contract.get("expected_task_count") == 30000, "task count differs")
    _need(contract.get("gold_average_allowed") is False, "derived score is permitted")
    _need(set(raw.get("rubric", {})) == {*AXES, *(f"score_{score}" for score in TARGET_SCORES)},
          "editor rubric differs")
    boundaries = raw.get("axis_edit_boundaries", {})
    _need(isinstance(boundaries, dict) and set(boundaries) == set(AXES), "editor boundaries differ")
    _need(all(isinstance(boundaries[axis], dict) and set(boundaries[axis]) == {"allowed", "forbidden"}
              for axis in AXES), "editor boundary schema differs")
    families = raw.get("axis_operation_families", {})
    _need(isinstance(families, dict) and set(families) == set(AXES),
          "editor operation families differ")
    _need(all(isinstance(families[axis], list) and len(families[axis]) == 4 and
              all(isinstance(item, str) and item.strip() for item in families[axis])
              for axis in AXES), "editor operation family population differs")
    score_families = raw.get("axis_score_operation_families", {})
    _need(isinstance(score_families, dict) and
          set(score_families) == {"content:1", "content:5", "organization:5", "expression:5"} and
          all(isinstance(items, list) and len(items) == 4 and
              all(isinstance(item, str) and item.strip() for item in items)
              for items in score_families.values()),
          "editor axis-score operation families differ")
    _need(raw.get("generation", {}).get("independent_candidate_families_per_task") == 4,
          "independent candidate family count differs")
    return raw


@dataclass(frozen=True, slots=True)
class SourceRow:
    identifier: str
    document_id: str
    prompt: str
    essay: str
    score: tuple[float, float, float]

    @property
    def baseline(self) -> dict[str, float]:
        return dict(zip(AXES, self.score, strict=True))


@dataclass(frozen=True, slots=True)
class AugmentationTask:
    task_id: str
    source: SourceRow
    target_axis: str
    target_score: int


def _text(value: Any, label: str) -> str:
    _need(isinstance(value, str) and bool(value.strip()), f"{label} is blank")
    return value


def source_row_from_mapping(raw: Mapping[str, Any]) -> SourceRow:
    identifier = _text(raw.get("id"), "source id")
    document_id = _text(raw.get("document_id"), "document id")
    score = raw.get("score")
    _need(isinstance(score, Mapping), "source score schema differs")
    values: list[float] = []
    for axis in AXES:
        value = score.get(axis)
        _need(type(value) in {int, float}, "source axis score is nonnumeric")
        number = float(value)
        _need(math.isfinite(number) and 1 <= number <= 5, "source axis score is out of range")
        values.append(number)
    return SourceRow(
        identifier=identifier,
        document_id=document_id,
        prompt=_text(raw.get("prompt"), "source prompt"),
        essay=_text(raw.get("essay"), "source essay"),
        score=(values[0], values[1], values[2]),
    )


def _read_rows(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SolarTargetAugmentationError(f"invalid JSON on line {line_number}") from exc
            _need(isinstance(raw, Mapping), f"row {line_number} is not an object")
            rows.append(raw)
    return rows


def validate_train_validation_disjoint(
    train_rows: Sequence[SourceRow], validation_rows: Iterable[Mapping[str, Any] | SourceRow]
) -> None:
    train_ids = {row.identifier for row in train_rows}
    train_documents = {row.document_id for row in train_rows}
    _need(len(train_ids) == len(train_rows), "duplicate canonical train id")
    _need(len(train_documents) == len(train_rows), "duplicate canonical train document_id")
    validation_ids: set[str] = set()
    validation_documents: set[str] = set()
    for raw in validation_rows:
        if isinstance(raw, SourceRow):
            identifier, document_id = raw.identifier, raw.document_id
        else:
            identifier = _text(raw.get("id"), "validation id")
            document_id = _text(raw.get("document_id"), "validation document id")
        validation_ids.add(identifier)
        validation_documents.add(document_id)
    _need(not train_ids.intersection(validation_ids), "train/validation id overlap")
    _need(not train_documents.intersection(validation_documents), "train/validation document_id overlap")


def load_train_rows() -> list[SourceRow]:
    """Load the exact canonical train set after a read-only leakage gate."""
    prompt_config()
    _need(TRAIN_PATH.is_file() and file_sha256(TRAIN_PATH) == EXPECTED_TRAIN_SHA256,
          "canonical train checksum differs")
    _need(VALIDATION_PATH.is_file() and file_sha256(VALIDATION_PATH) == EXPECTED_VALIDATION_SHA256,
          "canonical validation checksum differs")
    rows = [source_row_from_mapping(raw) for raw in _read_rows(TRAIN_PATH)]
    _need(len(rows) == 2000, "canonical train population differs")
    validate_train_validation_disjoint(rows, _read_rows(VALIDATION_PATH))
    return rows


def make_task(row: SourceRow, axis: str, score: int) -> AugmentationTask:
    _need(axis in AXES, "unknown target axis")
    _need(type(score) is int and score in TARGET_SCORES, "unknown target score")
    task_id = f"{row.identifier}::solar-target::{axis}::{score}"
    return AugmentationTask(task_id, row, axis, score)


def build_tasks(rows: Sequence[SourceRow]) -> list[AugmentationTask]:
    _need(len(rows) == 2000, "canonical source population differs")
    _need(len({row.identifier for row in rows}) == len(rows), "duplicate canonical source id")
    tasks = [make_task(row, axis, score) for row in rows for axis in AXES for score in TARGET_SCORES]
    _need(len(tasks) == 30000 and len({task.task_id for task in tasks}) == 30000, "task matrix differs")
    return tasks


def task_count(rows: Sequence[SourceRow]) -> int:
    _need(len(rows) == 2000, "canonical source population differs")
    return len(rows) * len(AXES) * len(TARGET_SCORES)


def render_editor_messages(
    task: AugmentationTask,
    candidate_family: int = 0,
) -> list[dict[str, str]]:
    """Render only from the immutable canonical source attached to the task."""
    config = prompt_config()
    _need(type(candidate_family) is int and 0 <= candidate_family < 4,
          "candidate family differs")
    source_length = len(task.source.essay.strip())
    lower_ratio, upper_ratio = config["quality_gates"]["prompt_length_ratio_by_axis"][task.target_axis]
    lower_ratio, upper_ratio = config["quality_gates"].get(
        "prompt_length_ratio_by_axis_score", {}
    ).get(f"{task.target_axis}:{task.target_score}", [lower_ratio, upper_ratio])
    source_sentences, source_paragraph_breaks = _source_layout(task.source.essay)
    operation_family = config.get("axis_score_operation_families", {}).get(
        f"{task.target_axis}:{task.target_score}",
        config["axis_operation_families"][task.target_axis],
    )[candidate_family]
    if task.target_axis == "organization":
        output_contract = config["editor_output_contracts"][
            "organization_fixed_order" if task.target_score == 4 else "organization_reorder"
        ]
    elif task.target_axis == "content" and task.target_score == 5:
        output_contract = config["editor_output_contracts"]["content5_evidence_ledger"]
    else:
        output_contract = config["editor_output_contracts"]["content_expression"]
    canonical_source: dict[str, Any] = {
        "source_id": task.source.identifier,
        "prompt_text": task.source.prompt,
        "baseline_score": task.source.baseline,
        "target_axis": task.target_axis,
        "target_score": task.target_score,
        "evaluation_rubric": config["rubric"],
        "target_axis_edit_boundary": config["axis_edit_boundaries"][task.target_axis],
        "target_axis_score_recipe": config["axis_target_recipes"][task.target_axis][str(task.target_score)],
        "target_axis_typed_edit_contract":
            config["quality_gates"]["typed_edit_contracts"][task.target_axis],
        "operation_family_index": candidate_family,
        "operation_family": operation_family,
        "required_editor_output": output_contract,
        "output_length_chars": {
            "source": source_length,
            "minimum": max(20, math.ceil(source_length * float(lower_ratio))),
            "maximum": math.floor(source_length * float(upper_ratio)),
        },
    }
    if task.target_axis == "content" and task.target_score == 5:
        canonical_source.update({
            "source_sentence_count": len(source_sentences),
            "source_sentence_units": [
                {"index": index, "text": sentence}
                for index, sentence in enumerate(source_sentences)
            ],
            "source_paragraph_break_after": source_paragraph_breaks,
        })
    else:
        canonical_source.update({
            "source_sentence_count": len(source_sentences),
            "source_sentence_units": [
                {"index": index, "text": sentence}
                for index, sentence in enumerate(source_sentences)
            ],
            "source_paragraph_break_after": source_paragraph_breaks,
        })
    if task.target_axis in {"content", "expression"} and not (
        task.target_axis == "content" and task.target_score == 5
    ):
        minimum_edits, maximum_edits = _sentence_edit_bounds(
            task.target_axis, task.target_score, len(source_sentences)
        )
        canonical_source["sentence_edit_count"] = {
            "minimum": minimum_edits,
            "maximum": maximum_edits,
        }
    payload = {"canonical_source": canonical_source}
    messages = [
        {"role": "system", "content": config["messages"]["editor_system"]},
        {"role": "user", "content": config["messages"]["editor_user_preamble"] + "\n" +
         json.dumps(payload, ensure_ascii=False, separators=(",", ":")) +
         "\n\n[현재 작업의 필수 편집]\n" +
         config["axis_target_recipes"][task.target_axis][str(task.target_score)] +
         "\n\n[독립 operation family]\n" +
         config["messages"]["candidate_family_preamble"] + "\n" +
         operation_family},
    ]
    return messages


def render_messages(row: SourceRow, axis: str, score: int) -> list[dict[str, str]]:
    """Compatibility-friendly editor entry point for a canonical source row."""
    return render_editor_messages(make_task(row, axis, score))


def render_verifier_messages(prompt: str, augmented_essay: str) -> list[dict[str, str]]:
    """Build an exact-rubric score request without target or source information."""
    prompt_config()
    prompt_text = _text(prompt, "verifier prompt")
    essay_text = _text(augmented_essay, "verifier essay")
    system, user_template = evaluation_sections()
    user = user_template.replace("{주제 지문}", prompt_text).replace("{논증적 글 본문}", essay_text)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_fidelity_messages(source_essay: str, augmented_essay: str) -> list[dict[str, str]]:
    config = prompt_config()
    payload = {"source_essay": _text(source_essay, "fidelity source essay"),
               "augmented_essay": _text(augmented_essay, "fidelity augmented essay")}
    return [
        {"role": "system", "content": config["messages"]["fidelity_system"]},
        {"role": "user", "content": config["messages"]["fidelity_user_preamble"] + "\n" +
         json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def _paragraph_units(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"\n+", text.strip()) if item.strip()]


def _sentence_units(text: str) -> list[str]:
    return [
        item.strip() for item in re.findall(r"[^.!?\n]+(?:[.!?]+|$)", text.strip())
        if item.strip()
    ]


def _source_layout(text: str) -> tuple[list[str], list[int]]:
    sentences: list[str] = []
    paragraph_break_after: list[int] = []
    paragraphs = _paragraph_units(text)
    for paragraph_index, paragraph in enumerate(paragraphs):
        units = _sentence_units(paragraph)
        _need(bool(units), "source paragraph has no sentence units")
        sentences.extend(units)
        if paragraph_index < len(paragraphs) - 1:
            paragraph_break_after.append(len(sentences) - 1)
    _need(bool(sentences), "source has no sentence units")
    return sentences, paragraph_break_after


def _compose_sentence_units(sentences: Sequence[str], paragraph_break_after: Sequence[int]) -> str:
    breaks = set(paragraph_break_after)
    parts: list[str] = []
    for index, sentence in enumerate(sentences):
        parts.append(sentence.strip())
        if index < len(sentences) - 1:
            parts.append("\n\n" if index in breaks else " ")
    return "".join(parts).strip()


def _coerce_single_sentence(value: str) -> str:
    """Fold an editor's accidental sub-sentences back into one sentence slot."""
    units = _sentence_units(value)
    _need(bool(units), "edited sentence is blank")
    if len(units) == 1:
        return units[0]
    folded = [re.sub(r"[.!?]+$", "", item).strip() for item in units[:-1]]
    return ", ".join([*folded, units[-1].strip()])


def _sentence_edit_bounds(axis: str, target_score: int, sentence_count: int) -> tuple[int, int]:
    _need(axis in {"content", "expression"}, "sentence edit axis differs")
    fractions = prompt_config()["quality_gates"]["sentence_edit_fraction_by_axis_score"][axis][
        str(target_score)
    ]
    minimum = max(1, math.ceil(sentence_count * float(fractions[0])))
    maximum = max(minimum, min(sentence_count, math.ceil(sentence_count * float(fractions[1]))))
    return minimum, maximum


def _preserve_terminal_punctuation(source_sentence: str, replacement: str) -> str:
    source_match = re.search(r"[.!?]+$", source_sentence.strip())
    terminal = source_match.group(0) if source_match else ""
    stem = re.sub(r"[.!?]+$", "", replacement.strip()).strip()
    return stem + terminal


def _non_whitespace_similarity(left: str, right: str) -> float:
    normalized_left = re.sub(r"\s+", "", unicodedata.normalize("NFC", left))
    normalized_right = re.sub(r"\s+", "", unicodedata.normalize("NFC", right))
    return SequenceMatcher(None, normalized_left, normalized_right, autojunk=False).ratio()


def _organization_sentence_inventory(text: str, connectors: Sequence[str]) -> Counter[str]:
    alternatives = "|".join(re.escape(item) for item in sorted(connectors, key=len, reverse=True))
    prefix = re.compile(rf"^(?:(?:{alternatives})\s*[,，:]?\s*)+")
    normalized: list[str] = []
    for sentence in _sentence_units(text):
        value = re.sub(r"\s+", " ", unicodedata.normalize("NFC", sentence)).strip()
        normalized.append(prefix.sub("", value))
    return Counter(normalized)


def _strip_leading_connectors(sentence: str, connectors: Sequence[str]) -> str:
    alternatives = "|".join(re.escape(item) for item in sorted(connectors, key=len, reverse=True))
    prefix = re.compile(rf"^(?:(?:{alternatives})\s*[,，:]?\s*)+")
    return prefix.sub("", sentence.strip())


def _validate_typed_edit(
    source: SourceRow,
    essay: str,
    axis: str,
    target_score: int | None = None,
) -> None:
    gates = prompt_config()["quality_gates"]
    _need(axis in AXES, "typed edit axis differs")
    for marker in gates["forbidden_new_evaluation_meta"]:
        _need(marker not in essay or marker in source.essay, "new evaluation metadata was added")
    contract = gates["typed_edit_contracts"][axis]
    if contract.get("same_paragraph_count"):
        _need(len(_paragraph_units(essay)) == len(_paragraph_units(source.essay)),
              f"{axis} paragraph scaffold differs")
    same_sentence_count = contract.get("same_sentence_count")
    if target_score is not None:
        same_sentence_count = contract.get("same_sentence_count_by_score", {}).get(
            str(target_score), same_sentence_count
        )
    if same_sentence_count:
        _need(len(_sentence_units(essay)) == len(_sentence_units(source.essay)),
              f"{axis} sentence scaffold differs")
    minimum_similarity = contract.get("minimum_non_whitespace_character_similarity")
    score_specific = contract.get("minimum_non_whitespace_character_similarity_by_score", {})
    if target_score is not None and str(target_score) in score_specific:
        minimum_similarity = score_specific[str(target_score)]
    if minimum_similarity is not None:
        _need(_non_whitespace_similarity(source.essay, essay) >= float(minimum_similarity),
              f"{axis} lexical scaffold differs")
    numeric_tokens_must_match = contract.get("numeric_token_sequence_must_match")
    if target_score is not None:
        numeric_tokens_must_match = contract.get("numeric_token_sequence_must_match_by_score", {}).get(
            str(target_score), numeric_tokens_must_match
        )
    if numeric_tokens_must_match:
        numbers = re.compile(r"\d+(?:[.,]\d+)*%?")
        _need(numbers.findall(essay) == numbers.findall(source.essay),
              f"{axis} numeric token sequence differs")
    if contract.get("normalized_sentence_inventory_must_match"):
        connectors = contract["allowed_connector_edits"]
        _need(_organization_sentence_inventory(essay, connectors) ==
              _organization_sentence_inventory(source.essay, connectors),
              "organization sentence inventory differs")


def _validate_augmented_essay(
    source: SourceRow,
    essay: str,
    axis: str | None,
    target_score: int | None = None,
) -> str:
    essay = _text(essay, "augmented essay").strip()
    _need(len(essay) >= 20, "augmented essay is too short")
    _need(essay != source.essay.strip(), "exact source copy is forbidden")
    change_gates = prompt_config()["quality_gates"]
    minimum_change = float(change_gates["minimum_substantive_change_ratio"])
    score_key = f"{axis}:{target_score}"
    if score_key in change_gates.get("minimum_substantive_change_ratio_by_axis_score", {}):
        minimum_change = float(
            change_gates["minimum_substantive_change_ratio_by_axis_score"][score_key]
        )
    if axis == "organization":
        # Reordering or changing a paragraph boundary is itself the typed edit;
        # whitespace-insensitive similarity would incorrectly reject that edit.
        _need(essay != source.essay.strip(), "substantive source change is too small")
    elif axis == "content":
        _need(1.0 - _non_whitespace_similarity(source.essay, essay) >= minimum_change,
              "substantive source change is too small")
    elif axis == "expression":
        normalized_source = unicodedata.normalize("NFC", source.essay.strip())
        normalized_essay = unicodedata.normalize("NFC", essay)
        similarity = SequenceMatcher(None, normalized_source, normalized_essay, autojunk=False).ratio()
        _need(1.0 - similarity >= minimum_change,
              "substantive source change is too small")
    else:
        normalized_source = unicodedata.normalize("NFC", source.essay.strip())
        normalized_essay = unicodedata.normalize("NFC", essay)
        similarity = SequenceMatcher(None, normalized_source, normalized_essay, autojunk=False).ratio()
        _need(1.0 - similarity >= minimum_change, "substantive source change is too small")
    ratio = len(essay) / len(source.essay.strip())
    gates = prompt_config()["quality_gates"]
    lower_ratio, upper_ratio = gates["prompt_length_ratio_by_axis"].get(
        axis, [gates["minimum_length_ratio"], gates["maximum_length_ratio"]]
    )
    score_key = f"{axis}:{target_score}"
    lower_ratio, upper_ratio = gates.get("prompt_length_ratio_by_axis_score", {}).get(
        score_key, [lower_ratio, upper_ratio]
    )
    _need(float(lower_ratio) <= ratio <= float(upper_ratio),
          "augmented essay length ratio differs")
    if axis is not None:
        _validate_typed_edit(source, essay, axis, target_score)
    return essay


def parse_editor_output(
    content: str,
    source: SourceRow,
    axis: str | None = None,
    target_score: int | None = None,
) -> str:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SolarTargetAugmentationError("editor output is not JSON") from exc
    _need(isinstance(raw, dict), "editor output is not an object")
    if axis is None:
        _need(set(raw) == {"augmented_essay"}, "editor output keys differ")
        return _validate_augmented_essay(source, raw["augmented_essay"], None)

    source_sentences, source_breaks = _source_layout(source.essay)
    if axis == "content" and target_score == 5:
        _need(set(raw) == {"evidence_additions"}, "editor evidence output keys differ")
        additions = raw["evidence_additions"]
        _need(isinstance(additions, list) and 1 <= len(additions) <= len(source_sentences),
              "editor evidence addition population differs")
        allowed_types = {"causal_bridge", "consequence_explanation", "conclusion_synthesis"}
        by_index: dict[int, str] = {}
        for item in additions:
            _need(isinstance(item, dict) and
                  set(item) == {"source_sentence_index", "addition_type", "addition_text"} and
                  type(item["source_sentence_index"]) is int and
                  isinstance(item["addition_type"], str) and
                  isinstance(item["addition_text"], str),
                  "editor evidence addition item differs")
            index = item["source_sentence_index"]
            addition_type = item["addition_type"]
            addition_text = _text(item["addition_text"], "evidence addition").strip()
            _need(0 <= index < len(source_sentences) and index not in by_index and
                  addition_type in allowed_types and "\n" not in addition_text and
                  len(_sentence_units(addition_text)) == 1,
                  "editor evidence addition is invalid")
            by_index[index] = addition_text
        values: list[str] = []
        breaks: list[int] = []
        source_break_set = set(source_breaks)
        for index, sentence in enumerate(source_sentences):
            values.append(sentence)
            if index in by_index:
                values.append(by_index[index])
            if index in source_break_set:
                breaks.append(len(values) - 1)
        essay = _compose_sentence_units(values, breaks)
    elif axis in {"content", "expression"}:
        _need(set(raw) == {"sentence_actions"}, "editor sentence output keys differ")
        actions = raw["sentence_actions"]
        _need(isinstance(actions, list) and len(actions) == len(source_sentences),
              "editor sentence action population differs")
        score = 3 if target_score is None else target_score
        minimum_edits, maximum_edits = _sentence_edit_bounds(axis, score, len(source_sentences))
        normalized_edits: dict[int, str] = {}
        for index, item in enumerate(actions):
            _need(isinstance(item, dict) and set(item) == {"apply", "replacement"} and
                  type(item["apply"]) is bool and isinstance(item["replacement"], str),
                  "editor sentence action item differs")
            if not item["apply"]:
                _need(item["replacement"] == "", "inactive editor sentence replacement differs")
                continue
            replacement = _coerce_single_sentence(
                _text(item["replacement"], "edited sentence").replace("\n", " ")
            )
            replacement = _preserve_terminal_punctuation(source_sentences[index], replacement)
            source_comparison = unicodedata.normalize("NFC", source_sentences[index].strip())
            replacement_comparison = unicodedata.normalize("NFC", replacement.strip())
            if replacement_comparison == source_comparison:
                continue
            if axis == "content":
                _need(re.sub(r"\s+", "", replacement_comparison) !=
                      re.sub(r"\s+", "", source_comparison),
                      "content editor sentence changed only whitespace")
            normalized_edits[index] = replacement
        _need(minimum_edits <= len(normalized_edits) <= maximum_edits,
              "editor substantive sentence edit count differs")
        values = list(source_sentences)
        for index, replacement in normalized_edits.items():
            values[index] = replacement
        essay = _compose_sentence_units(values, source_breaks)
    elif axis == "organization":
        fixed_order = target_score == 4
        expected = (
            {"paragraph_break_after", "connector_actions"}
            if fixed_order else
            {"sentence_order", "paragraph_break_after", "connector_actions"}
        )
        _need(set(raw) == expected, "organization plan keys differ")
        breaks = raw["paragraph_break_after"]
        connector_actions = raw["connector_actions"]
        count = len(source_sentences)
        if fixed_order:
            order = list(range(count))
        else:
            requested_order = raw["sentence_order"]
            _need(isinstance(requested_order, list) and len(requested_order) == count and
                  all(type(item) is int and 0 <= item < count for item in requested_order),
                  "organization sentence order differs")
            _need(len(set(requested_order)) == count,
                  "organization sentence order is not a permutation")
            order = requested_order
        _need(isinstance(breaks, list) and all(type(item) is int for item in breaks) and
              all(0 <= item < count - 1 for item in breaks),
              "organization paragraph break plan differs")
        _need(len(set(breaks)) == len(breaks),
              "organization paragraph break plan has duplicates")
        allowed = prompt_config()["quality_gates"]["typed_edit_contracts"]["organization"][
            "allowed_connector_edits"
        ]
        normalized_breaks = sorted(set(breaks))
        connector_limit = {1: 0, 2: 1, 3: 1, 4: 1, 5: 0}.get(target_score, 2)
        paragraph_starts = {0, *(item + 1 for item in normalized_breaks)}
        _need(isinstance(connector_actions, list) and len(connector_actions) <= connector_limit,
              "organization connector plan exceeds limit")
        normalized_prefixes = [""] * count
        seen_positions: set[int] = set()
        used_connectors: set[str] = set()
        for item in connector_actions:
            _need(isinstance(item, dict) and set(item) == {"position", "connector"} and
                  type(item["position"]) is int and isinstance(item["connector"], str),
                  "organization connector action differs")
            position = item["position"]
            connector = item["connector"]
            _need(position in paragraph_starts and position not in seen_positions and
                  connector in allowed and connector not in used_connectors,
                  "organization connector action is invalid")
            normalized_prefixes[position] = connector
            seen_positions.add(position)
            used_connectors.add(connector)
        ordered = [
            (f"{normalized_prefixes[position]}, " if normalized_prefixes[position] else "") +
            (
                _strip_leading_connectors(source_sentences[source_index], allowed)
                if fixed_order else source_sentences[source_index]
            )
            for position, source_index in enumerate(order)
        ]
        essay = _compose_sentence_units(ordered, normalized_breaks)
    else:
        raise SolarTargetAugmentationError("typed edit axis differs")
    return _validate_augmented_essay(source, essay, axis, target_score)


def parse_verifier_output(content: str) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SolarTargetAugmentationError("verifier output is not JSON") from exc
    _need(isinstance(raw, dict) and set(raw) == set(AXES), "verifier axes differ")
    parsed: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        item = raw[axis]
        _need(isinstance(item, dict) and set(item) == {"score", "rationale"}, "verifier item differs")
        score = item["score"]
        _need(type(score) is int and score in TARGET_SCORES, "verifier score is not a 1-5 integer")
        rationale = _text(item["rationale"], "verifier rationale").strip()
        parsed[axis] = {"score": score, "rationale": rationale}
    return parsed


def parse_fidelity_output(content: str) -> dict[str, bool]:
    expected = {"source_based", "topic", "stance", "genre", "new_external_facts_added"}
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SolarTargetAugmentationError("fidelity output is not JSON") from exc
    _need(isinstance(raw, dict) and set(raw) == expected, "fidelity output keys differ")
    _need(all(type(raw[key]) is bool for key in expected), "fidelity output is non-boolean")
    return {key: raw[key] for key in expected}


def validate_candidate(
    task: AugmentationTask,
    augmented_essay: str,
    verifier: Mapping[str, Mapping[str, Any]],
    source_verifier: Mapping[str, Mapping[str, Any]],
    fidelity: Mapping[str, bool],
) -> dict[str, Any]:
    """Apply all hard gates after independent editor/verifier/auditor calls."""
    essay = _validate_augmented_essay(
        task.source, augmented_essay, task.target_axis, task.target_score
    )
    parsed_verifier = parse_verifier_output(json.dumps(verifier, ensure_ascii=False))
    parsed_source_verifier = parse_verifier_output(json.dumps(source_verifier, ensure_ascii=False))
    parsed_fidelity = parse_fidelity_output(json.dumps(fidelity, ensure_ascii=False))
    scores = {axis: parsed_verifier[axis]["score"] for axis in AXES}
    _need(scores[task.target_axis] == task.target_score, "target verifier score differs")
    for axis in AXES:
        if axis != task.target_axis:
            source_score = parsed_source_verifier[axis]["score"]
            _need(scores[axis] == source_score, "non-target verifier score changed too much")
    _need(all(parsed_fidelity[key] is True for key in ("source_based", "topic", "stance", "genre")),
          "source fidelity failed")
    _need(parsed_fidelity["new_external_facts_added"] is False, "new external facts were added")
    return {"task_id": task.task_id, "source_id": task.source.identifier, "target_axis": task.target_axis,
            "target_score": task.target_score, "augmented_essay": essay, "score": scores}


def select_smoke_sources(
    rows: Sequence[SourceRow], count: int = 5, seed: str | int | None = None
) -> list[SourceRow]:
    """Choose an order-independent deterministic SHA-256 pseudo-random sample."""
    _need(type(count) is int and count >= 5, "smoke source count must be at least five")
    _need(count <= len(rows), "smoke source count exceeds population")
    _need(len({row.identifier for row in rows}) == len(rows), "duplicate smoke source id")
    if seed is None:
        seed = prompt_config()["smoke"]["seed"]
    seed_text = str(seed)
    ranked = sorted(rows, key=lambda row: (sha256(f"{seed_text}\0{row.identifier}".encode()).digest(), row.identifier))
    return ranked[:count]


def editor_output_schema(task: AugmentationTask) -> Mapping[str, Any]:
    sentence_count = len(_source_layout(task.source.essay)[0])
    if task.target_axis == "content" and task.target_score == 5:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["evidence_additions"],
            "properties": {
                "evidence_additions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": sentence_count,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["source_sentence_index", "addition_type", "addition_text"],
                        "properties": {
                            "source_sentence_index": {
                                "type": "integer", "minimum": 0, "maximum": sentence_count - 1
                            },
                            "addition_type": {
                                "type": "string",
                                "enum": ["causal_bridge", "consequence_explanation",
                                         "conclusion_synthesis"],
                            },
                            "addition_text": {"type": "string"},
                        },
                    },
                }
            },
        }
    if task.target_axis in {"content", "expression"}:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["sentence_actions"],
            "properties": {
                "sentence_actions": {
                    "type": "array",
                    "minItems": sentence_count,
                    "maxItems": sentence_count,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["apply", "replacement"],
                        "properties": {
                            "apply": {"type": "boolean"},
                            "replacement": {"type": "string"},
                        },
                    },
                }
            },
        }
    connectors = prompt_config()["quality_gates"]["typed_edit_contracts"]["organization"][
        "allowed_connector_edits"
    ]
    maximum_breaks = {4: 3, 5: 4}.get(task.target_score, max(0, sentence_count - 1))
    connector_limit = {1: 0, 2: 1, 3: 1, 4: 1, 5: 0}[task.target_score]
    properties: dict[str, Any] = {
        "paragraph_break_after": {
            "type": "array", "maxItems": min(maximum_breaks, max(0, sentence_count - 1)),
            "items": {"type": "integer", "minimum": 0, "maximum": max(0, sentence_count - 2)},
        },
        "connector_actions": {
            "type": "array", "maxItems": connector_limit,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["position", "connector"],
                "properties": {
                    "position": {"type": "integer", "minimum": 0,
                                 "maximum": max(0, sentence_count - 1)},
                    "connector": {"type": "string", "enum": connectors},
                },
            },
        },
    }
    required = ["paragraph_break_after", "connector_actions"]
    if task.target_score != 4:
        properties["sentence_order"] = {
            "type": "array", "minItems": sentence_count, "maxItems": sentence_count,
            "items": {"type": "integer", "minimum": 0, "maximum": sentence_count - 1},
        }
        required.insert(0, "sentence_order")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }
