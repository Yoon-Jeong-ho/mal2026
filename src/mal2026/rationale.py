"""Validation for train-only synthetic evidence rationales.

Synthetic rationales are never treated as human labels or faithful
explanations. This module fails closed on schema, leakage, or offset errors.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .constants import EVIDENCE_CRITERIA, PROHIBITED_RATIONALE_PATTERNS

_PROHIBITED = tuple(re.compile(pattern, re.IGNORECASE) for pattern in PROHIBITED_RATIONALE_PATTERNS)
_ENTRY_KEYS = frozenset({"criterion", "quote", "start", "end", "observation"})


class RationaleValidationError(ValueError):
    """A synthetic rationale violates the frozen no-leakage evidence contract."""


@dataclass(frozen=True)
class RationaleValidation:
    nonempty_valid: bool
    entry_count: int


def _assert_clean_prose(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RationaleValidationError("observation must be a nonblank string")
    for pattern in _PROHIBITED:
        if pattern.search(value):
            raise RationaleValidationError("observation contains a score, rating, or score proxy")


def validate_rationale_payload(payload: Mapping[str, Any], essay: str | None) -> RationaleValidation:
    """Validate exact schema and, when supplied, quote-to-offset equality.

    `rationale: []` is a deliberate schema-valid fallback after failed teacher
    retries. It is returned as nonempty_valid=False so the caller can enforce
    the 85% gate without dropping the training record.
    """
    if not isinstance(payload, Mapping) or set(payload) != {"rationale"}:
        raise RationaleValidationError("rationale payload must contain only 'rationale'")
    entries = payload["rationale"]
    if not isinstance(entries, list):
        raise RationaleValidationError("rationale must be a list")
    if not entries:
        return RationaleValidation(nonempty_valid=False, entry_count=0)
    if len(entries) != len(EVIDENCE_CRITERIA):
        raise RationaleValidationError("nonempty rationale must contain exactly one entry per criterion")
    seen_criteria: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != _ENTRY_KEYS:
            raise RationaleValidationError("each rationale entry must use the exact evidence schema")
        criterion = entry["criterion"]
        if criterion not in EVIDENCE_CRITERIA:
            raise RationaleValidationError("unknown rationale criterion")
        if criterion in seen_criteria:
            raise RationaleValidationError("each rationale criterion may occur only once")
        seen_criteria.add(criterion)
        quote = entry["quote"]
        start, end = entry["start"], entry["end"]
        if not isinstance(quote, str) or not quote:
            raise RationaleValidationError("quote must be a nonempty string")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise RationaleValidationError("offsets must be integer character positions")
        if start < 0 or end <= start:
            raise RationaleValidationError("offsets must satisfy 0 <= start < end")
        if essay is not None:
            if end > len(essay) or essay[start:end] != quote:
                raise RationaleValidationError("quote does not equal essay[start:end]")
        _assert_clean_prose(entry["observation"])
    if seen_criteria != set(EVIDENCE_CRITERIA):
        raise RationaleValidationError("nonempty rationale must cover exactly all three criteria")
    return RationaleValidation(nonempty_valid=True, entry_count=len(entries))
