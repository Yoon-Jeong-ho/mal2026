"""Private in-memory contracts for API-rationale SFT and score regression.

The OpenAI/API candidate artifact is restricted.  This module deliberately
keeps essays, identifiers, diagnoses, and generated model output in memory;
callers may persist only aggregate metrics and provenance outside the
restricted root.  Crucially, the candidate artifact's own score field is
never read: score-regression labels come only from the canonical writing data.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "openai-rationale-terra-full-20260719-001"
RESTRICTED_ROOT = ROOT / "data" / "processed" / "restricted" / "openai_rationale_batches" / RUN_ID
AXES = ("content", "organization", "expression")
TRAIN_SOURCE = ROOT / "eval" / "train.jsonl"
VALIDATION_SOURCE = ROOT / "eval" / "validation.jsonl"
SOURCE_SHA256 = {
    "train": "b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737",
    "validation": "0805445029328848164cf15f34b90b88fb5f7896d7b73f24f1717b733a9a00a4",
}
CANDIDATE_PATH = {
    "train": RESTRICTED_ROOT / "derived" / "train-only-candidates-v1-20260719-001" / "candidates.train.jsonl",
    "validation": RESTRICTED_ROOT / "derived" / "validation-only-candidates-v1-20260720-001" / "candidates.validation.jsonl",
}
CANDIDATE_SHA256 = {
    "train": "d69dd9bc349c117a55b4bf652507135f084c9045d95d5939da3980223893aecf",
    "validation": "21c7b97e6faf6d8092b4a27e35b60083f9b9b60861493867061816fcb12f9d83",
}
EXPECTED_ESSAYS = {"train": 2000, "validation": 400}
EXPECTED_CANDIDATES = {"train": 6000, "validation": 1200}


class APIRationaleContractError(ValueError):
    """Raised before an ambiguous, leaky, or noncanonical private input is used."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise APIRationaleContractError(message)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    _need(path.is_file() and not path.is_symlink(), "private input must be an ordinary existing file")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise APIRationaleContractError(f"blank JSONL record at line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise APIRationaleContractError(f"invalid JSONL at line {line_number}") from exc
            _need(isinstance(value, dict), "private JSONL record must be an object")
            yield value


def _text(value: Any, field: str) -> str:
    _need(isinstance(value, str) and bool(value.strip()), f"{field} must be nonblank text")
    return value


def _score(value: Any, field: str) -> float:
    _need(type(value) in {int, float} and not isinstance(value, bool), f"{field} must be numeric")
    parsed = float(value)
    _need(1.0 <= parsed <= 5.0, f"{field} must be in [1,5]")
    return parsed


@dataclass(frozen=True)
class WritingRow:
    """One private writing record; identifiers/text never leave RAM."""

    identifier: str
    prompt: str
    essay: str
    scores: Mapping[str, float] | None


@dataclass(frozen=True)
class CandidateRow:
    """One API rationale candidate without its provider-side candidate score."""

    custom_id: str
    source_id: str
    candidate_number: int
    diagnoses: Mapping[str, str]


@dataclass(frozen=True)
class JoinedCandidate:
    """Private writing/candidate join, used only in memory."""

    writing: WritingRow
    candidate: CandidateRow


def _source_path(split: str) -> Path:
    _need(split in {"train", "validation"}, "split must be train or validation")
    return TRAIN_SOURCE if split == "train" else VALIDATION_SOURCE


def load_writing_rows(split: str, *, include_scores: bool = True) -> list[WritingRow]:
    """Load canonical writing rows; labels are opt-in for score regression only."""
    path = _source_path(split)
    _need(sha256_file(path) == SOURCE_SHA256[split], "writing source checksum changed")
    rows: list[WritingRow] = []
    seen: set[str] = set()
    expected = {"id", "document_id", "prompt_num", "prompt", "essay", "score"}
    for raw in _jsonl(path):
        _need(set(raw) == expected, "writing source schema differs")
        identifier = _text(raw["id"], "writing.id")
        _need(identifier not in seen, "writing source IDs must be unique")
        seen.add(identifier)
        if include_scores:
            score = raw["score"]
            _need(isinstance(score, dict) and tuple(score) == ("content", "organization", "expression", "average"), "writing score schema differs")
            scores: Mapping[str, float] | None = {axis: _score(score[axis], f"writing.score.{axis}") for axis in AXES}
        else:
            # The rationale-generator and judge stages do not read, prompt, or
            # use the writing-score field at all.  The source checksum and row
            # schema still pin the immutable source population.
            scores = None
        rows.append(WritingRow(
            identifier=identifier, prompt=_text(raw["prompt"], "writing.prompt"), essay=_text(raw["essay"], "writing.essay"),
            scores=scores,
        ))
    _need(len(rows) == EXPECTED_ESSAYS[split], "writing source count differs")
    return rows


def _candidate_diagnoses(rationale: Any) -> dict[str, str]:
    _need(isinstance(rationale, dict) and set(rationale) == {"schema_version", *AXES}, "candidate rationale axes differ")
    _need(rationale.get("schema_version") == "rationale-v3-sentence-id", "candidate rationale schema differs")
    result: dict[str, str] = {}
    for axis in AXES:
        value = rationale[axis]
        _need(isinstance(value, dict) and set(value) == {"evidence_sentence_ids", "diagnosis", "next_step"}, "candidate rationale fields differ")
        ids = value["evidence_sentence_ids"]
        _need(isinstance(ids, list) and ids and all(type(item) is int and item > 0 for item in ids), "candidate evidence IDs differ")
        result[axis] = _text(value["diagnosis"], f"candidate.{axis}.diagnosis")
        _text(value["next_step"], f"candidate.{axis}.next_step")
    return result


def load_candidates(split: str, writings: Sequence[WritingRow] | None = None) -> list[CandidateRow]:
    """Read every validated candidate while deliberately never reading `raw['score']`."""
    _need(split in {"train", "validation"}, "split must be train or validation")
    path = CANDIDATE_PATH[split]
    _need(sha256_file(path) == CANDIDATE_SHA256[split], "candidate artifact checksum changed")
    source_hashes = {
        row.identifier: sha256(row.essay.encode("utf-8")).hexdigest()
        for row in (writings if writings is not None else load_writing_rows(split, include_scores=False))
    }
    expected = {"api_response_id", "candidate", "custom_id", "essay_sha256", "model", "rationale", "schema_version", "score", "source_id", "split"}
    result: list[CandidateRow] = []
    seen: set[str] = set(); group_counts: dict[str, set[int]] = {}
    for raw in _jsonl(path):
        _need(set(raw) == expected and raw.get("split") == split, "candidate source schema/split differs")
        # Do not access raw["score"]: it is a model-generated candidate score
        # and is prohibited as either SFT target or regression feature/label.
        custom_id = _text(raw["custom_id"], "candidate.custom_id")
        source_id = _text(raw["source_id"], "candidate.source_id")
        _need(custom_id not in seen and source_id in source_hashes, "candidate linkage is invalid")
        seen.add(custom_id)
        _need(raw.get("essay_sha256") == source_hashes[source_id], "candidate essay linkage checksum differs")
        number = raw.get("candidate")
        _need(type(number) is int and number in {1, 2, 3}, "candidate number differs")
        group_counts.setdefault(source_id, set()).add(number)
        result.append(CandidateRow(custom_id=custom_id, source_id=source_id, candidate_number=number, diagnoses=_candidate_diagnoses(raw["rationale"])))
    _need(len(result) == EXPECTED_CANDIDATES[split] and len(group_counts) == EXPECTED_ESSAYS[split], "candidate population count differs")
    _need(all(numbers == {1, 2, 3} for numbers in group_counts.values()), "candidate population is incomplete")
    return result


def joined_candidates(split: str) -> list[JoinedCandidate]:
    """Return the validated full candidate population joined to its writing source."""
    writings = load_writing_rows(split, include_scores=False)
    by_id = {row.identifier: row for row in writings}
    result = [JoinedCandidate(writing=by_id[row.source_id], candidate=row) for row in load_candidates(split, writings)]
    _need(len(result) == EXPECTED_CANDIDATES[split], "joined candidate count differs")
    return result


def rationale_object(diagnoses: Mapping[str, str], axes: Sequence[str]) -> dict[str, Any]:
    """Create the sole rationale-only output contract shared by SFT and generation."""
    chosen = tuple(axes)
    _need(bool(chosen) and len(set(chosen)) == len(chosen) and set(chosen) <= set(AXES), "invalid rationale axes")
    return {"schema_version": "rationale-only-v1", **{axis: {"rationale": _text(diagnoses[axis], f"diagnosis.{axis}")} for axis in chosen}}


def render_rationale_target(diagnoses: Mapping[str, str], axes: Sequence[str]) -> str:
    return json.dumps(rationale_object(diagnoses, axes), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def parse_rationale_output(text: Any, axes: Sequence[str]) -> dict[str, str] | None:
    """Strictly parse model output without score extraction or text repair."""
    chosen = tuple(axes)
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {"schema_version", *chosen} or value.get("schema_version") != "rationale-only-v1":
        return None
    parsed: dict[str, str] = {}
    for axis in chosen:
        part = value.get(axis)
        if not isinstance(part, dict) or set(part) != {"rationale"} or not isinstance(part["rationale"], str) or not part["rationale"].strip():
            return None
        parsed[axis] = part["rationale"]
    return parsed


def decoder_messages(writing: WritingRow, axes: Sequence[str]) -> list[dict[str, str]]:
    """Score-blind native-chat messages for rationale generation."""
    chosen = tuple(axes)
    _need(bool(chosen) and len(set(chosen)) == len(chosen) and set(chosen) <= set(AXES), "invalid decoder axes")
    axes_text = ", ".join(chosen)
    contract = render_rationale_target({axis: f"<{axis}>" for axis in chosen}, chosen).replace("<", "[").replace(">", "]")
    system = (
        "당신은 한국어 글쓰기 평가의 근거 설명을 작성하는 평가자입니다. 학생 글과 과제만 근거로 "
        "내용(content), 구성(organization), 표현(expression) 중 요청된 축의 구체적 진단 설명을 작성하십시오. "
        "글 자체의 점수, 후보 점수, 문장 번호, 개선 제안은 포함하지 마십시오."
    )
    user = (
        f"요청 축: {axes_text}\n"
        f"반드시 다음 키 구조의 JSON만 반환하십시오: {contract}\n"
        f"<writing_prompt>\n{writing.prompt}\n</writing_prompt>\n"
        f"<student_essay>\n{writing.essay}\n</student_essay>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def candidate_sft_examples(task: str) -> list[dict[str, Any]]:
    """Build all 6,000 prompt/completion examples for one decoder task in memory."""
    axes = axes_for_task(task)
    result = []
    for joined in joined_candidates("train"):
        messages = decoder_messages(joined.writing, axes)
        result.append({"prompt": messages, "completion": [{"role": "assistant", "content": render_rationale_target(joined.candidate.diagnoses, axes)}]})
    _need(len(result) == EXPECTED_CANDIDATES["train"], "SFT example population differs")
    return result


def axes_for_task(task: str) -> tuple[str, ...]:
    if task == "bundle":
        return AXES
    _need(task in AXES, "task must be bundle or one canonical axis")
    return (task,)


def validation_writings() -> list[WritingRow]:
    return load_writing_rows("validation")


def train_writings() -> list[WritingRow]:
    return load_writing_rows("train")


def load_generated_rationales(generation_dir: Path, *, source: str, task: str) -> dict[str, dict[str, str]]:
    """Load a completed restricted decoder artifact for a downstream declared stage."""
    axes = AXES if task == "axis_triplet" else axes_for_task(task)
    root = generation_dir.resolve()
    _need(root.parent == (RESTRICTED_ROOT / "decoder_generation_v1").resolve() and root.is_dir() and not generation_dir.is_symlink(), "generated-rationale root differs")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        report = json.loads((root / "aggregate_generation_report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise APIRationaleContractError("generated-rationale provenance is unreadable") from exc
    _need(isinstance(manifest, dict) and isinstance(report, dict) and manifest.get("status") == "completed" and report.get("status") == "completed", "generated-rationale provenance is incomplete")
    _need(manifest.get("source") == source and manifest.get("task") == task and all(report.get("hard_gates", {}).values()), "generated-rationale provenance differs")
    records = root / "generated_rationales.jsonl"
    _need(records.is_file() and sha256_file(records) == report.get("generated_rationales_sha256"), "generated-rationale checksum differs")
    result: dict[str, dict[str, str]] = {}
    for raw in _jsonl(records):
        _need(set(raw) == {"attempts", "failure_category", "parse_valid", "rationale", "source_id"}, "generated-rationale row schema differs")
        identifier = _text(raw["source_id"], "generated.source_id")
        _need(identifier not in result and raw.get("parse_valid") is True and raw.get("failure_category") is None, "generated rationale must be uniquely valid")
        rationale = raw.get("rationale")
        _need(isinstance(rationale, dict) and set(rationale) == set(axes), "generated rationale axes differ")
        result[identifier] = {axis: _text(rationale[axis], f"generated.{axis}") for axis in axes}
    _need(len(result) == EXPECTED_ESSAYS[source], "generated-rationale record count differs")
    return result


def aggregate_input_provenance() -> dict[str, Any]:
    """Non-sensitive immutable data identity safe to store in tracked records."""
    return {
        "api_run_id": RUN_ID, "source_sha256": dict(SOURCE_SHA256), "candidate_sha256": dict(CANDIDATE_SHA256),
        "expected_essays": dict(EXPECTED_ESSAYS), "expected_candidates": dict(EXPECTED_CANDIDATES),
        "candidate_scores_read_or_prompted": False,
    }
