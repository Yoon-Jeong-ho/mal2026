"""Versioned score-only prompts derived from the public MAL2026 contract.

The task PDF specifies the three analytic axes and the required integer output
contract, but it does not disclose an organizer-authored system prompt.  The
``public_spec_score_only_v1`` prompt below therefore preserves the public
rubric while removing rationale generation instructions, as permitted for the
separate score-model component.  The legacy prompt remains available solely
to reproduce already completed historical runs.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Mapping


LEGACY_COMPACT = "legacy_compact_v1"
PUBLIC_SPEC_SCORE_ONLY = "public_spec_score_only_v1"
SCORE_PROMPT_KINDS = (LEGACY_COMPACT, PUBLIC_SPEC_SCORE_ONLY)


LEGACY_COMPACT_INSTRUCTION = (
    "Predict three integer Korean writing scores (content, organization, "
    "expression), each from 1 to 5."
)


PUBLIC_SPEC_SCORE_ONLY_INSTRUCTION = """한국어 논증적 글을 다음 세 영역에서 서로 독립적으로 평가하라.
- content: 문제에 대한 주장과 핵심 내용의 적절성, 근거의 충분성과 구체성, 주장과 근거의 논리적 연결
- organization: 서론·본론·결론 구조, 문단 간 연결, 일관된 논리 전개
- expression: 문장의 자연스러움과 명료성, 적절한 어휘, 맞춤법·띄어쓰기·문법·주술 호응
점수는 1 매우 미흡, 2 미흡, 3 보통, 4 우수, 5 매우 우수의 정수 척도를 사용하라.
과제와 학생 글에서 확인되는 내용만 근거로 판단하고 1~5 전 구간을 사용하라.
content, organization, expression의 정수 점수만 출력하며 rationale과 average는 출력하지 마라."""


DECODER_OUTPUT_RULE = (
    'JSON 객체 {"content":1,"organization":1,"expression":1} 형식 하나만 '
    "출력하라. 코드블록, 마크다운, 설명, 추가 키를 출력하지 마라."
)


def instruction(kind: str) -> str:
    if kind == LEGACY_COMPACT:
        return LEGACY_COMPACT_INSTRUCTION
    if kind == PUBLIC_SPEC_SCORE_ONLY:
        return PUBLIC_SPEC_SCORE_ONLY_INSTRUCTION
    raise ValueError("unknown score prompt kind")


def system_prompt(kind: str) -> str:
    if kind == LEGACY_COMPACT:
        return "당신은 한국어 글쓰기 평가자입니다. 과제와 학생 글만 근거로 세 정수 점수만 출력하십시오."
    return instruction(kind) + "\n" + DECODER_OUTPUT_RULE


def query_text(
    prompt_text: str,
    essay_text: str,
    rationales: Mapping[str, str] | None = None,
) -> str:
    text = (
        f"<writing_prompt>\n{prompt_text}\n</writing_prompt>\n"
        f"<student_essay>\n{essay_text}\n</student_essay>"
    )
    if rationales is not None:
        axes = ("content", "organization", "expression")
        if set(rationales) != set(axes) or not all(
            isinstance(rationales[axis], str) and rationales[axis].strip() for axis in axes
        ):
            raise ValueError("three nonblank rationale axes are required")
        text += "\n<evaluation_rationales>\n" + "\n".join(
            f"<{axis}>{rationales[axis]}</{axis}>" for axis in axes
        ) + "\n</evaluation_rationales>"
    return text


def embedding_input(
    prompt_text: str,
    essay_text: str,
    kind: str,
    rationales: Mapping[str, str] | None = None,
) -> str:
    return f"Instruct: {instruction(kind)}\nQuery:\n{query_text(prompt_text, essay_text, rationales)}"


def prompt_sha256(kind: str) -> str:
    payload = system_prompt(kind) + "\n" + DECODER_OUTPUT_RULE + "\n" + instruction(kind)
    return sha256(payload.encode("utf-8")).hexdigest()


def provenance(kind: str) -> dict[str, str]:
    return {
        "score_prompt_kind": kind,
        "score_prompt_sha256": prompt_sha256(kind),
        "prompt_provenance": (
            "legacy_compact_reproduction"
            if kind == LEGACY_COMPACT
            else "public_spec_aligned_score_only_reconstruction_not_verbatim_organizer_prompt"
        ),
    }
