"""Strict public request and participant-output contracts for submission."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


AXES = ("content", "organization", "expression")


class SubmissionContractError(ValueError):
    """Raised when an evaluator request or model output is not deployable."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise SubmissionContractError(message)


def extract_prompt_essay(messages: Sequence[Mapping[str, Any]]) -> tuple[str, str] | None:
    """Extract the official task fields from the latest matching message.

    The Docker guide also uses an ordinary chat request as a connectivity
    example.  Such a request has no task markers and returns ``None`` rather
    than being misclassified as a writing-evaluation row.
    """

    _need(isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)), "messages must be a sequence")
    for message in reversed(messages):
        _need(isinstance(message, Mapping), "each message must be an object")
        content = message.get("content")
        _need(isinstance(content, str), "message content must be text")
        prompt_marker = "[prompt_text]"
        essay_marker = "[essay_text]"
        prompt_start = content.find(prompt_marker)
        if prompt_start < 0:
            continue
        essay_start = content.find(essay_marker, prompt_start + len(prompt_marker))
        if essay_start < 0:
            continue
        prompt = content[prompt_start + len(prompt_marker):essay_start].strip()
        essay = content[essay_start + len(essay_marker):].strip()
        _need(bool(prompt), "prompt_text is blank")
        _need(bool(essay), "essay_text is blank")
        return prompt, essay
    return None


def parse_rationales(value: str | Mapping[str, Any]) -> dict[str, str]:
    try:
        raw = json.loads(value) if isinstance(value, str) else dict(value)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SubmissionContractError("rationale output is not one JSON object") from exc
    _need(set(raw) == set(AXES), "rationale axes differ")
    result: dict[str, str] = {}
    for axis in AXES:
        cell = raw[axis]
        _need(isinstance(cell, Mapping) and set(cell) == {"rationale"}, f"{axis} rationale shape differs")
        rationale = cell["rationale"]
        _need(isinstance(rationale, str) and bool(rationale.strip()), f"{axis} rationale is blank")
        result[axis] = rationale.strip()
    return result


def participant_output(scores: Mapping[str, int], rationales: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    _need(set(scores) == set(AXES), "score axes differ")
    _need(set(rationales) == set(AXES), "rationale axes differ")
    result: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        score = scores[axis]
        rationale = rationales[axis]
        _need(type(score) is int and 1 <= score <= 5, f"{axis} score must be an integer in [1,5]")
        _need(isinstance(rationale, str) and bool(rationale.strip()), f"{axis} rationale is blank")
        # The evaluator's announced first-JSON extractor counts braces without
        # recognizing JSON string boundaries.  Replace braces inside free-text
        # rationales so an essay quotation can never terminate that parser's
        # top-level object early.  Structural JSON braces are added afterward.
        parser_safe = rationale.strip().replace("{", "(").replace("}", ")")
        _need(bool(parser_safe), f"{axis} parser-safe rationale is blank")
        result[axis] = {"score": score, "rationale": parser_safe}
    return result


def parse_participant_output(value: str | Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(value) if isinstance(value, str) else dict(value)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SubmissionContractError("participant output is not one JSON object") from exc
    _need(set(raw) == set(AXES), "participant axes differ")
    scores: dict[str, int] = {}
    rationales: dict[str, str] = {}
    for axis in AXES:
        cell = raw[axis]
        _need(isinstance(cell, Mapping) and set(cell) == {"score", "rationale"}, f"{axis} participant shape differs")
        scores[axis] = cell["score"]  # type: ignore[assignment]
        rationales[axis] = cell["rationale"]  # type: ignore[assignment]
    return participant_output(scores, rationales)


def compact_participant_json(value: str | Mapping[str, Any]) -> str:
    parsed = parse_participant_output(value)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
