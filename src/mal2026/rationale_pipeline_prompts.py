"""Immutable prompt routing and score projection for the rationale pipeline."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
ROUTING = ROOT / "configs/rationale_pipeline_prompt_routing.v1.json"
AXES = ("content", "organization", "expression")
RATIONALE_SYSTEM_MARKER = "[시스템 프롬프트]"
RATIONALE_USER_MARKER = "[유저 프롬프트 템플릿]"
SCORE_INPUT_MARKER = "[인코더 입력 템플릿]"
SCORE_CONTRACT_MARKER = "[학습·평가 계약 - 인코더 입력에 포함하지 않음]"


class RationalePromptError(ValueError):
    """Raised when a prompt, rationale, or score violates the frozen contract."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RationalePromptError(message)


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def routing() -> dict[str, Any]:
    value = json.loads(ROUTING.read_text(encoding="utf-8"))
    need(value.get("schema_version") == "mal2026-rationale-pipeline-prompt-routing-v1", "rationale routing schema differs")
    bindings = (
        value["rationale_generation_training_evaluation"],
        value["rationale_reward_and_quality_judge"],
        value["rationale_to_score_encoder"],
    )
    for binding in bindings:
        path = ROOT / str(binding["source_file"])
        need(path.is_file() and not path.is_symlink(), "rationale prompt file is unavailable")
        need(sha256_file(path) == binding["source_file_sha256"], "rationale prompt hash differs")
    generation = value["rationale_generation_training_evaluation"]
    reward = value["rationale_reward_and_quality_judge"]
    encoder = value["rationale_to_score_encoder"]
    need(generation.get("score_input") is False and generation.get("score_output") is False, "rationale generator score boundary differs")
    need(reward.get("score_in_policy_prompt") is False and reward.get("score_in_judge_prompt") is True, "reward score routing differs")
    need(encoder.get("score_input") is False and encoder.get("average_used") is False, "encoder score boundary differs")
    return value


def _split_once(text: str, first: str, second: str) -> tuple[str, str]:
    need(text.count(first) == 1 and text.count(second) == 1, "prompt section marker differs")
    before, tail = text.split(first, 1)
    need(not before.strip(), "unexpected text precedes prompt system marker")
    system, template = tail.split(second, 1)
    need(bool(system.strip()) and bool(template.strip()), "prompt section is blank")
    return system.strip(), template.strip()


def _replace_once(template: str, placeholder: str, value: str) -> str:
    need(template.count(placeholder) == 1, f"prompt placeholder differs: {placeholder}")
    return template.replace(placeholder, json.dumps(value, ensure_ascii=False), 1)


def rationale_messages(prompt_text: str, essay_text: str) -> list[dict[str, str]]:
    routing()
    need(isinstance(prompt_text, str) and isinstance(essay_text, str), "rationale inputs must be strings")
    text = (ROOT / "Rationale_evaluation_training.txt").read_text(encoding="utf-8")
    system, template = _split_once(text, RATIONALE_SYSTEM_MARKER, RATIONALE_USER_MARKER)
    user = _replace_once(template, "{prompt_text_json_string}", prompt_text)
    user = _replace_once(user, "{essay_text_json_string}", essay_text)
    need("reference_scores_integer" not in system + user and "predicted_score" not in system + user, "score leaked into rationale policy prompt")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_rationales(value: Mapping[str, Any]) -> dict[str, str]:
    need(set(value) == set(AXES), "rationale axes differ")
    result: dict[str, str] = {}
    for axis in AXES:
        part = value[axis]
        text = part.get("rationale") if isinstance(part, Mapping) else part
        need(isinstance(text, str) and bool(text.strip()), f"{axis} rationale is blank")
        result[axis] = text.strip()
    return result


def rationale_output(value: str | Mapping[str, Any]) -> dict[str, dict[str, str]]:
    try:
        raw = json.loads(value) if isinstance(value, str) else dict(value)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RationalePromptError("rationale output is not JSON") from exc
    rationales = normalize_rationales(raw)
    return {axis: {"rationale": rationales[axis]} for axis in AXES}


def rationale_to_score_text(prompt_text: str, essay_text: str, rationales: Mapping[str, Any]) -> str:
    routing()
    need(isinstance(prompt_text, str) and isinstance(essay_text, str), "score encoder inputs must be strings")
    normalized = normalize_rationales(rationales)
    source = (ROOT / "rationale_to_score.txt").read_text(encoding="utf-8")
    need(source.count(RATIONALE_SYSTEM_MARKER) == source.count(SCORE_INPUT_MARKER) == source.count(SCORE_CONTRACT_MARKER) == 1, "score prompt marker differs")
    before_contract, contract = source.split(SCORE_CONTRACT_MARKER, 1)
    need(bool(contract.strip()), "score projection contract is blank")
    system, template = _split_once(before_contract, RATIONALE_SYSTEM_MARKER, SCORE_INPUT_MARKER)
    rendered = _replace_once(template, "{prompt_text_json_string}", prompt_text)
    rendered = _replace_once(rendered, "{essay_text_json_string}", essay_text)
    for axis in AXES:
        rendered = _replace_once(rendered, f"{{{axis}_rationale_json_string}}", normalized[axis])
    result = system + "\n\n" + rendered
    need(SCORE_CONTRACT_MARKER not in result and "ROUND_HALF_UP" not in result, "training target contract leaked into encoder input")
    return result


def finite_score(value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RationalePromptError("score is not numeric") from exc
    need(parsed.is_finite() and Decimal("1") <= parsed <= Decimal("5"), "score is outside finite [1,5]")
    return parsed


def round_half_up_score(value: Any) -> int:
    return int(finite_score(value).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def regression_evaluation_score(value: Any) -> int:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RationalePromptError("regression prediction is not numeric") from exc
    need(parsed.is_finite(), "regression prediction is not finite")
    clipped = min(Decimal("5"), max(Decimal("1"), parsed))
    return int(clipped.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def integer_labels(scores: Mapping[str, Any]) -> dict[str, int]:
    need(set(scores) >= set(AXES), "score axes differ")
    return {axis: round_half_up_score(scores[axis]) for axis in AXES}


def judge_participant(scores: Mapping[str, Any], rationales: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    labels = integer_labels(scores)
    normalized = normalize_rationales(rationales)
    return {axis: {"score": labels[axis], "rationale": normalized[axis]} for axis in AXES}
