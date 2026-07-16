"""Safe model inputs and exact score-target serialization.

No function in this module includes IDs, gold scores, split labels, or metadata
in a teacher input. The only target-rendering functions are used after the
input has been constructed.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import re
from typing import Mapping

from .constants import SCORE_FIELDS, SCORE_MAX, SCORE_MIN
from .data_contract import DatasetRecord, ScoreVector

_SCORE_VALUE = r"(?:[1-5]\.[0-9]{2})"
_SCORE_JSON_RE = re.compile(
    rf'\A\s*\{{\s*"content"\s*:\s*({_SCORE_VALUE})\s*,\s*'
    rf'"organization"\s*:\s*({_SCORE_VALUE})\s*,\s*'
    rf'"expression"\s*:\s*({_SCORE_VALUE})\s*,\s*'
    rf'"average"\s*:\s*({_SCORE_VALUE})\s*\}}\s*\Z'
)


class ScoreParseError(ValueError):
    """A model did not produce the frozen exact score JSON target."""


@dataclass(frozen=True)
class ScoreParseResult:
    scores: ScoreVector
    valid: bool
    out_of_range: bool
    error: str | None


def _two_decimal(value: float) -> str:
    rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return format(rounded, ".2f")


def render_score_target(scores: ScoreVector | Mapping[str, float]) -> str:
    """Return the sole direct-SFT target representation, with ordered keys."""
    values = scores.as_dict() if isinstance(scores, ScoreVector) else dict(scores)
    if set(values) != set(SCORE_FIELDS):
        raise ValueError("scores must contain exactly the four protocol fields")
    return "{" + ",".join(f'"{name}":{_two_decimal(float(values[name]))}' for name in SCORE_FIELDS) + "}"


def _safe_prompt_and_essay(record: DatasetRecord) -> str:
    # Delimiters reduce accidental instruction mixing but do not claim to defeat
    # adversarial content. IDs, split membership, and scores are omitted.
    return (
        "<writing_prompt>\n" + record.prompt + "\n</writing_prompt>\n"
        "<student_essay>\n" + record.essay + "\n</student_essay>"
    )


def format_decoder_input(record: DatasetRecord) -> str:
    """Input for direct-score student SFT/inference; contains no gold labels."""
    return (
        "You are a Korean writing assessor. Read the assignment and essay. "
        "Return only the exact JSON score object requested by the system.\n"
        + _safe_prompt_and_essay(record)
    )


def format_encoder_input(record: DatasetRecord, max_chars: int | None = None) -> str:
    """Stable text representation for encoder regression, without labels/IDs."""
    text = "[과제]\n" + record.prompt + "\n[학생 글]\n" + record.essay
    if max_chars is not None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive when provided")
        return text[:max_chars]
    return text


def parse_score_json(text: str, fallback: ScoreVector) -> ScoreParseResult:
    """Strict parser: no prose/number-extraction fallback; invalid uses train mean."""
    if not isinstance(text, str):
        return ScoreParseResult(fallback, False, False, "output is not a string")
    matched = _SCORE_JSON_RE.fullmatch(text)
    if matched is None:
        return ScoreParseResult(fallback, False, False, "output does not match exact four-field JSON")
    values = {name: float(value) for name, value in zip(SCORE_FIELDS, matched.groups(), strict=True)}
    out_of_range = any(value < SCORE_MIN or value > SCORE_MAX for value in values.values())
    if out_of_range:
        return ScoreParseResult(fallback, False, True, "score is outside [1, 5]")
    return ScoreParseResult(ScoreVector(**values), True, False, None)
