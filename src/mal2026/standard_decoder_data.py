"""Restricted-data contract and rendering for the standard decoder stack.

This module deliberately has no Torch, TRL, or vLLM dependency.  It reads
restricted rows only from the ignored prepared-data root and returns them in
memory.  Metrics/artifacts must be aggregate-only; callers must never write
row prompts, essays, feedback, identifiers, or model completions.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = ROOT / "data" / "processed"
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "aihub_human_feedback_v1.json"
VALIDATION_PATH = ROOT / "eval" / "validation.jsonl"
SCORE_FIELDS = ("content", "organization", "expression", "average")
FEEDBACK_FIELDS = (
    "holistic", "content_1", "content_2", "content_3", "organization_1",
    "organization_2", "expression_1", "expression_2", "task_1",
)
SYSTEM_MESSAGE = (
    "당신은 한국어 글쓰기 평가자입니다. 과제와 학생 글만 근거로 평가하십시오. "
    "지시된 JSON 형식 외의 문장을 출력하지 마십시오."
)
_DIRECT_RE = re.compile(
    r'\A\{"content":([1-5]\.[0-9]{2}),"organization":([1-5]\.[0-9]{2}),'
    r'"expression":([1-5]\.[0-9]{2}),"average":([1-5]\.[0-9]{2})\}\Z'
)


class StandardDecoderContractError(ValueError):
    """A privacy, schema, or frozen evaluation contract was violated."""


@dataclass(frozen=True)
class RestrictedRow:
    """Private in-memory row; never serialize it outside ignored storage."""

    identifier: str
    prompt: str
    essay: str
    score: dict[str, float]
    feedback: dict[str, str] | None


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StandardDecoderContractError(f"{field} must be a nonblank string")
    return value


def _score(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise StandardDecoderContractError(f"{field} must be numeric")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise StandardDecoderContractError(f"{field} is not numeric") from exc
    if not parsed.is_finite() or not Decimal("1") <= parsed <= Decimal("5"):
        raise StandardDecoderContractError(f"{field} is outside [1, 5]")
    return float(parsed)


def _object_pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StandardDecoderContractError("decoder output contains duplicate JSON keys")
        result[key] = value
    return result


def render_scores(score: Mapping[str, float]) -> str:
    if tuple(score) != SCORE_FIELDS:
        raise StandardDecoderContractError("score fields must use frozen ordering")
    values = [_score(score[name], f"score.{name}") for name in SCORE_FIELDS]
    return "{" + ",".join(f'"{name}":{value:.2f}' for name, value in zip(SCORE_FIELDS, values, strict=True)) + "}"


def render_human_feedback_target(row: RestrictedRow) -> str:
    if row.feedback is None or tuple(row.feedback) != FEEDBACK_FIELDS:
        raise StandardDecoderContractError("human-feedback target requires exactly nine ordered feedback fields")
    feedback = {key: _nonblank(row.feedback[key], f"feedback.{key}") for key in FEEDBACK_FIELDS}
    return '{"feedback":' + json.dumps(feedback, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ',"scores":' + render_scores(row.score) + "}"


def decoder_user_prompt(row: RestrictedRow, mode: str) -> str:
    if mode not in {"direct", "human_feedback"}:
        raise StandardDecoderContractError("mode must be direct or human_feedback")
    requested = (
        '정확히 {"content":x.xx,"organization":x.xx,"expression":x.xx,"average":x.xx} JSON만 반환하십시오.'
        if mode == "direct"
        else '먼저 9개 사람 채점 피드백을 JSON의 feedback 객체에 작성하고, 이어 scores 객체에 '
        'content, organization, expression, average 점수를 소수 둘째 자리까지 반환하십시오.'
    )
    return f"{requested}\n<writing_prompt>\n{row.prompt}\n</writing_prompt>\n<student_essay>\n{row.essay}\n</student_essay>"


def messages_for_sft(row: RestrictedRow, mode: str) -> list[dict[str, str]]:
    target = render_scores(row.score) if mode == "direct" else render_human_feedback_target(row)
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": decoder_user_prompt(row, mode)},
        {"role": "assistant", "content": target},
    ]


def messages_for_generation(row: RestrictedRow, mode: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": decoder_user_prompt(row, mode)},
    ]


def _validate_source_row(raw: Any, *, need_feedback: bool) -> RestrictedRow:
    if not isinstance(raw, dict):
        raise StandardDecoderContractError("prepared row must be an object")
    expected = {"id", "prompt", "essay", "score", "feedback"}
    if set(raw) != expected:
        raise StandardDecoderContractError("prepared row schema differs from frozen contract")
    score_raw, feedback_raw = raw["score"], raw["feedback"]
    if not isinstance(score_raw, dict) or tuple(score_raw) != SCORE_FIELDS:
        raise StandardDecoderContractError("prepared row score keys differ from frozen contract")
    if not isinstance(feedback_raw, dict) or tuple(feedback_raw) != FEEDBACK_FIELDS:
        raise StandardDecoderContractError("prepared row feedback keys differ from frozen contract")
    feedback = {key: _nonblank(feedback_raw[key], f"feedback.{key}") for key in FEEDBACK_FIELDS}
    return RestrictedRow(
        identifier=_nonblank(raw["id"], "id"), prompt=_nonblank(raw["prompt"], "prompt"), essay=_nonblank(raw["essay"], "essay"),
        score={key: _score(score_raw[key], f"score.{key}") for key in SCORE_FIELDS}, feedback=feedback if need_feedback else feedback,
    )


def _validate_validation_row(raw: Any) -> RestrictedRow:
    if not isinstance(raw, dict) or set(raw) != {"id", "document_id", "prompt_num", "prompt", "essay", "score"}:
        raise StandardDecoderContractError("frozen validation row schema differs from expected contract")
    score_raw = raw["score"]
    if not isinstance(score_raw, dict) or tuple(score_raw) != SCORE_FIELDS:
        raise StandardDecoderContractError("validation score keys differ from frozen contract")
    return RestrictedRow(
        identifier=_nonblank(raw["id"], "id"), prompt=_nonblank(raw["prompt"], "prompt"), essay=_nonblank(raw["essay"], "essay"),
        score={key: _score(score_raw[key], f"score.{key}") for key in SCORE_FIELDS}, feedback=None,
    )


def _safe_manifest_path(path: Path) -> Path:
    if path.resolve() != DEFAULT_MANIFEST.resolve():
        raise StandardDecoderContractError("only the canonical aggregate prepared manifest is accepted")
    return path.resolve()


def _safe_processed_file(filename: str) -> Path:
    candidate = (PROCESSED_ROOT / "aihub_human_feedback_v1" / filename)
    if candidate.parent.resolve() != (PROCESSED_ROOT / "aihub_human_feedback_v1").resolve() or candidate.is_symlink():
        raise StandardDecoderContractError("prepared data must be an ordinary file under canonical ignored root")
    if not candidate.is_file():
        raise StandardDecoderContractError("prepared data file is missing")
    return candidate


def load_prepared_split(split: str, manifest_path: Path = DEFAULT_MANIFEST) -> list[RestrictedRow]:
    """Read a hash-verified private prepared split into memory only."""
    if split not in {"selection_train", "selection_dev", "refit_train"}:
        raise StandardDecoderContractError("invalid prepared split")
    manifest = json.loads(_safe_manifest_path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("dataset_id") != "aihub_human_feedback_v1":
        raise StandardDecoderContractError("invalid prepared aggregate manifest")
    details = manifest.get("files", {}).get(split)
    if not isinstance(details, dict) or set(details) != {"filename", "sha256", "record_count"}:
        raise StandardDecoderContractError("prepared manifest split details are invalid")
    path = _safe_processed_file(details["filename"])
    if _sha256(path) != details["sha256"]:
        raise StandardDecoderContractError("prepared split SHA-256 mismatch")
    rows = [_validate_source_row(json.loads(line), need_feedback=True) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != details["record_count"] or not rows:
        raise StandardDecoderContractError("prepared split count mismatch")
    if len({row.identifier for row in rows}) != len(rows):
        raise StandardDecoderContractError("prepared split has duplicate identifiers")
    return rows


def load_frozen_validation(expected_sha256: str) -> list[RestrictedRow]:
    """Read only the designated validation file after its caller-pinned hash check."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise StandardDecoderContractError("validation SHA-256 must be a 64-character lowercase hex digest")
    if not VALIDATION_PATH.is_file() or _sha256(VALIDATION_PATH) != expected_sha256:
        raise StandardDecoderContractError("frozen validation path/hash contract failed")
    rows = [_validate_validation_row(json.loads(line)) for line in VALIDATION_PATH.read_text(encoding="utf-8").splitlines() if line]
    if not rows or len({row.identifier for row in rows}) != len(rows):
        raise StandardDecoderContractError("validation IDs are invalid")
    return rows


def parse_decoder_scores(text: str, mode: str) -> dict[str, float] | None:
    """Strictly parse a complete response; invalid output is never number-scraped."""
    if not isinstance(text, str) or mode not in {"direct", "human_feedback"}:
        return None
    if mode == "direct":
        matched = _DIRECT_RE.fullmatch(text)
        if not matched:
            return None
        return {field: float(value) for field, value in zip(SCORE_FIELDS, matched.groups(), strict=True)}
    try:
        parsed = json.loads(text, object_pairs_hook=_object_pairs_no_duplicates)
    except (json.JSONDecodeError, StandardDecoderContractError):
        return None
    if not isinstance(parsed, dict) or list(parsed) != ["feedback", "scores"]:
        return None
    feedback, scores = parsed["feedback"], parsed["scores"]
    if not isinstance(feedback, dict) or list(feedback) != list(FEEDBACK_FIELDS) or any(not isinstance(value, str) or not value.strip() for value in feedback.values()):
        return None
    if not isinstance(scores, dict) or list(scores) != list(SCORE_FIELDS):
        return None
    # Canonical serialization must use exactly two-decimal numeric literals, not strings or exponent notation.
    try:
        canonical = render_human_feedback_scores_only(feedback, scores)
    except StandardDecoderContractError:
        return None
    return {key: float(scores[key]) for key in SCORE_FIELDS} if canonical == text else None


def render_human_feedback_scores_only(feedback: Mapping[str, Any], scores: Mapping[str, Any]) -> str:
    """Canonicalization helper used only to validate generated JSON."""
    if tuple(feedback) != FEEDBACK_FIELDS or tuple(scores) != SCORE_FIELDS:
        raise StandardDecoderContractError("invalid generated field order")
    clean_feedback = {key: _nonblank(feedback[key], f"feedback.{key}") for key in FEEDBACK_FIELDS}
    clean_scores = {key: _score(scores[key], f"score.{key}") for key in SCORE_FIELDS}
    return '{"feedback":' + json.dumps(clean_feedback, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ',"scores":' + render_scores(clean_scores) + "}"


def score_mean(rows: Iterable[RestrictedRow]) -> dict[str, float]:
    materialized = list(rows)
    if not materialized:
        raise StandardDecoderContractError("cannot calculate empty fallback mean")
    return {field: sum(row.score[field] for row in materialized) / len(materialized) for field in SCORE_FIELDS}
