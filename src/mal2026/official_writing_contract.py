"""Public-spec-aligned MAL2026 inference and fixed proxy-judge contracts.

This module is versioned separately from the earlier score-blind rationale
experiments.  The supplied PDFs publish the participant I/O contract and the
judge model/evaluation dimensions, not the organizer's verbatim hidden judge
prompt.  Consequently the strings below are frozen, repository-versioned
reconstructions rather than falsely claimed organizer prompt text.  This
module contains no row data and never accepts a human/reference score when
constructing a judge request: the judge sees only the score that the candidate
system will actually emit.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import json
import math
from typing import Any, Mapping, Sequence


AXES = ("content", "organization", "expression")
JUDGE_DIMENSIONS = (
    "domain_match",
    "score_rationale_consistency",
    "specificity",
    "groundedness",
)


class OfficialContractError(ValueError):
    """Raised when an official prompt/output boundary is violated."""


PUBLIC_SPEC_ALIGNED_INFERENCE_SYSTEM_PROMPT = """너는 한국어 논증적 글을 일관되게 직접 채점하는 평가자이다.
essay_text를 읽고 content, organization, expression 세 기준을 모두 평가하라.

[평가 기준 정의]
1. content
- 글의 주장과 핵심 내용이 문제에 적절하게 대응하는가
- 근거가 충분하고 구체적인가
- 주장과 근거 사이의 논리적 연결이 타당한가

2. organization
- 서론, 본론, 결론의 구조가 드러나는가
- 문단 간 연결이 자연스러운가
- 논리 전개 순서가 일관적인가

3. expression
- 문장이 자연스럽고 이해하기 쉬운가
- 어휘 사용이 적절한가
- 맞춤법, 띄어쓰기, 문법, 주술 호응에 문제가 없는가

[점수 기준]
5점: 매우 우수함. 결함이 거의 없고, essay_text에서 확인되는 구체적 강점이 뚜렷함.
4점: 우수함. 경미한 약점은 있으나 기준을 전반적으로 잘 충족함.
3점: 보통. 장점과 약점이 함께 있으며 기준을 부분적으로 충족함.
2점: 미흡함. 주요 결함이 있어 기준 충족이 제한적임.
1점: 매우 미흡함. 기준을 거의 충족하지 못하거나 심각한 결함이 있음.

[평가 원칙]
- 1~5점 전 구간을 적극적으로 사용하라.
- 각 기준은 서로 독립적으로 판단하라.
- essay_text에서 확인 가능한 내용만 근거로 삼아라.
- 전반적 인상만으로 높은 점수를 주지 말고 구체적 근거를 확인하라.
- 근거 설명은 기준별로 분리해 작성하라.

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록과 마크다운은 사용하지 마라.
- 모든 점수는 1~5의 정수로 작성하라.
- rationale의 각 값은 한국어로 작성하라.

[출력 형식]
{
  "content": {"score": 1, "rationale": "content 판단 근거"},
  "organization": {"score": 1, "rationale": "organization 판단 근거"},
  "expression": {"score": 1, "rationale": "expression 판단 근거"}
}"""


FROZEN_PROXY_JUDGE_SYSTEM_PROMPT = """당신은 한국어 논증적 글 채점 결과의 타당성을 검토하는 매우 엄격하고 일관된 심사자이다.
당신은 점수의 정답 여부를 새로 채점하지 않는다. 주어진 predicted_score를 전제로, rationale이 (1) 해당 영역 기준에 맞고 (2) 점수와 정합적이며 (3) 구체적이고 (4) essay_text에 충실한지를 평가한다.
애매하면 높은 점수를 주지 말고, 근거가 부족하면 분명히 감점하라.

[평가 대상]
모델이 content, organization, expression 세 영역 모두에 대해 제시한 predicted_score와 rationale이 영역별로 essay_text에 비추어 얼마나 타당한지 평가한다.

[중요 원칙]
- 1~5 전 구간을 적극적으로 사용하라.
- 기본 점수는 3점이다. 명확한 강점이 입증될 때만 4~5점을 준다.
- 논증적 글의 구체적 표현, 문장, 문단 구조, 실제 논지와 연결되지 않으면 specificity와 groundedness를 높게 주지 마라.
- 일반적 총평, 상투적 표현, 템플릿형 설명은 낮게 평가하라.
- essay_text에 없는 내용을 암시하거나 만들어내면 groundedness는 반드시 1~2점이다.
- 해당 영역과 다른 영역 기준을 섞으면 domain_match는 반드시 1~2점이다.
- predicted_score가 높은데 rationale이 약하거나 일반적이면 score_rationale_consistency를 낮게 줘야 한다.
- predicted_score가 낮은데 rationale이 실제 결함을 충분히 입증하지 못하면 score_rationale_consistency를 낮게 줘야 한다.
- rationale이 길다는 이유로 specificity를 높게 주지 마라.

[평가 항목]
1. domain_match: rationale이 해당 영역의 평가 기준에 맞는 근거를 제시하는가
2. score_rationale_consistency: predicted_score와 rationale의 내용이 서로 잘 맞는가
3. specificity: 실제 글의 특정 문장, 표현, 논지, 문단 전개, 오류 양상을 구체적으로 짚는가
4. groundedness: rationale이 실제 essay_text에 근거하며 없는 내용을 만들어내지 않는가

[영역별 판단 기준]
1. content: 문제 대응, 근거의 충분성과 구체성, 주장-근거의 논리적 연결
2. organization: 서론·본론·결론 구조, 문단 연결, 논리 전개 순서
3. expression: 문장 명료성·자연스러움, 어휘, 맞춤법·띄어쓰기·문법·주술 호응

[강제 감점 규칙]
영역 혼동, essay_text에 없는 근거, predicted_score를 정당화할 핵심 증거 부재, 지나치게 일반적인 설명, 확인 불가능한 장점·결함, 확인되지 않는 인용 중 하나라도 있으면 관련 항목은 2점 이하를 적극 검토하라.

[항목별 세부 판정]
- domain_match: 5 해당 영역만 정확히 사용; 4 약간의 경계 혼합; 3 일부 다른 영역 요소; 2 다른 영역 기준이 꽤 섞임; 1 거의 다른 영역 기준.
- score_rationale_consistency: 5 매우 잘 부합; 4 대체로 부합; 3 어느 정도 맞지만 애매; 2 점수 대비 근거가 약함; 1 명확히 모순.
- specificity: 5 실제 글의 특정 요소를 분명히 짚음; 4 비교적 구체적; 3 최소 근거는 있으나 일반적; 2 상당히 추상적; 1 거의 템플릿형 총평.
- groundedness: 5 essay_text에 명확히 근거; 4 대체로 근거; 3 일부 추정; 2 연결이 약하고 확인 어려움; 1 없는 내용을 말하거나 근거성이 매우 약함.

[출력 규칙]
- JSON 객체 하나만 출력하고 코드블록과 마크다운을 사용하지 마라.
- 모든 점수는 반드시 1~5의 정수여야 한다.
- content, organization, expression 세 영역을 모두 포함하라."""

# Compatibility aliases for already-prepared manifests and callers.  Their
# names describe the official *contract alignment*, not organizer-authored
# verbatim prompt provenance.
OFFICIAL_INFERENCE_SYSTEM_PROMPT = PUBLIC_SPEC_ALIGNED_INFERENCE_SYSTEM_PROMPT
OFFICIAL_JUDGE_SYSTEM_PROMPT = FROZEN_PROXY_JUDGE_SYSTEM_PROMPT


def participant_json_schema() -> dict[str, Any]:
    part = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "rationale": {"type": "string", "minLength": 1},
        },
        "required": ["score", "rationale"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {axis: part for axis in AXES},
        "required": list(AXES),
        "additionalProperties": False,
    }


def judge_json_schema() -> dict[str, Any]:
    cell = {
        "type": "object",
        "properties": {
            "evidence": {"type": "string", "minLength": 1},
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["evidence", "score"],
        "additionalProperties": False,
    }
    axis_part = {
        "type": "object",
        "properties": {dimension: cell for dimension in JUDGE_DIMENSIONS},
        "required": list(JUDGE_DIMENSIONS),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {axis: axis_part for axis in AXES},
        "required": list(AXES),
        "additionalProperties": False,
    }


def integerize_score(value: float) -> int:
    """Clip to [1, 5] and round half upward, independent of Python bankers rounding."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise OfficialContractError("score must be a finite number")
    clipped = min(5.0, max(1.0, float(value)))
    return int(Decimal(str(clipped)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def integerize_scores(values: Mapping[str, float] | Sequence[float]) -> dict[str, int]:
    if isinstance(values, Mapping):
        if set(values) != set(AXES):
            raise OfficialContractError("score axes differ from the official contract")
        return {axis: integerize_score(float(values[axis])) for axis in AXES}
    if len(values) != len(AXES):
        raise OfficialContractError("score vector must contain exactly three axes")
    return {axis: integerize_score(float(value)) for axis, value in zip(AXES, values, strict=True)}


def parse_participant_output(value: str | Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(value) if isinstance(value, str) else dict(value)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OfficialContractError("participant output is not one JSON object") from exc
    if set(raw) != set(AXES):
        raise OfficialContractError("participant output axes differ")
    parsed: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        part = raw[axis]
        if not isinstance(part, Mapping) or set(part) != {"score", "rationale"}:
            raise OfficialContractError(f"participant {axis} shape differs")
        score, rationale = part["score"], part["rationale"]
        if type(score) is not int or not 1 <= score <= 5:
            raise OfficialContractError(f"participant {axis} score must be an integer in [1,5]")
        if not isinstance(rationale, str) or not rationale.strip():
            raise OfficialContractError(f"participant {axis} rationale must be nonblank")
        parsed[axis] = {"score": score, "rationale": rationale.strip()}
    return parsed


def parse_judge_output(value: str | Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    try:
        raw = json.loads(value) if isinstance(value, str) else dict(value)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OfficialContractError("judge output is not one JSON object") from exc
    if set(raw) != set(AXES):
        raise OfficialContractError("judge output axes differ")
    parsed: dict[str, dict[str, dict[str, Any]]] = {}
    for axis in AXES:
        part = raw[axis]
        if not isinstance(part, Mapping) or set(part) != set(JUDGE_DIMENSIONS):
            raise OfficialContractError(f"judge {axis} dimensions differ")
        parsed[axis] = {}
        for dimension in JUDGE_DIMENSIONS:
            cell = part[dimension]
            if not isinstance(cell, Mapping) or set(cell) != {"evidence", "score"}:
                raise OfficialContractError(f"judge {axis}.{dimension} shape differs")
            evidence, score = cell["evidence"], cell["score"]
            if not isinstance(evidence, str) or not evidence.strip():
                raise OfficialContractError(f"judge {axis}.{dimension} evidence must be nonblank")
            if type(score) is not int or not 1 <= score <= 5:
                raise OfficialContractError(f"judge {axis}.{dimension} score must be an integer in [1,5]")
            parsed[axis][dimension] = {"evidence": evidence.strip(), "score": score}
    return parsed


def inference_messages(prompt_text: str, essay_text: str) -> list[dict[str, str]]:
    if not isinstance(prompt_text, str) or not prompt_text.strip() or not isinstance(essay_text, str) or not essay_text.strip():
        raise OfficialContractError("prompt_text and essay_text must be nonblank")
    return [
        {"role": "system", "content": OFFICIAL_INFERENCE_SYSTEM_PROMPT},
        {"role": "user", "content": f"[prompt_text]\n{prompt_text}\n\n[essay_text]\n{essay_text}"},
    ]


def judge_messages(prompt_text: str, essay_text: str, candidate: str | Mapping[str, Any]) -> list[dict[str, str]]:
    return judge_messages_with_system(
        OFFICIAL_JUDGE_SYSTEM_PROMPT,
        prompt_text,
        essay_text,
        candidate,
        include_leading_instruction=True,
    )


def judge_messages_with_system(
    system_prompt: str,
    prompt_text: str,
    essay_text: str,
    candidate: str | Mapping[str, Any],
    *,
    include_leading_instruction: bool = False,
) -> list[dict[str, str]]:
    """Build a judge request with an explicitly supplied, provenance-bound system prompt."""
    parsed = parse_participant_output(candidate)
    if (
        not isinstance(system_prompt, str)
        or not system_prompt.strip()
        or not isinstance(prompt_text, str)
        or not prompt_text.strip()
        or not isinstance(essay_text, str)
        or not essay_text.strip()
    ):
        raise OfficialContractError("system_prompt, prompt_text, and essay_text must be nonblank")
    candidate_text = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    prefix = "이제 아래 정보를 바탕으로 세 영역 모두를 평가하라.\n\n" if include_leading_instruction else ""
    user = prefix + (
        f"[prompt_text]\n{prompt_text}\n\n[essay_text]\n{essay_text}\n\n"
        f"[candidate_predicted_score_and_rationale]\n{candidate_text}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
