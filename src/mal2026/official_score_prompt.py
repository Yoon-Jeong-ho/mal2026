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
import json
from pathlib import Path
from typing import Mapping


LEGACY_COMPACT = "legacy_compact_v1"
PUBLIC_SPEC_SCORE_ONLY = "public_spec_score_only_v1"
USER_SUPPLIED_EVALUATION = "user_supplied_evaluation_txt_v1"
RATIONALE_AWARE_ENCODER = "rationale_aware_encoder_score_v1"
SCORE_PROMPT_KINDS = (
    LEGACY_COMPACT,
    PUBLIC_SPEC_SCORE_ONLY,
    USER_SUPPLIED_EVALUATION,
    RATIONALE_AWARE_ENCODER,
)
ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PROMPT_PATH = ROOT / "evaluation.txt"
EVALUATION_PROMPT_SHA256 = "1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145"
RATIONALE_AWARE_PROMPT_PATH = ROOT / "configs/official_rationale_aware_score_prompt.v1.json"
RATIONALE_AWARE_PROMPT_SHA256 = "692da6e051ba9864d5699f5e8e11c143ce30784415c5050b89d63a3aedccc60c"


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


def _evaluation_sections() -> tuple[str, str]:
    """Return the exact system body and user template from evaluation.txt.

    The two marker lines are routing metadata, not model-message content.  All
    bytes inside each routed section (including Korean spacing) are otherwise
    retained.  The whole-file digest prevents silent prompt drift.
    """
    if not EVALUATION_PROMPT_PATH.is_file() or EVALUATION_PROMPT_PATH.is_symlink():
        raise ValueError("evaluation prompt file is unavailable")
    payload = EVALUATION_PROMPT_PATH.read_bytes()
    if sha256(payload).hexdigest() != EVALUATION_PROMPT_SHA256:
        raise ValueError("evaluation prompt file digest differs")
    text = payload.decode("utf-8")
    lines = text.splitlines(keepends=True)
    markers = {line.strip(): index for index, line in enumerate(lines) if line.strip() in {"[시스템 프롬프트]", "[유저 프롬프트]"}}
    if set(markers) != {"[시스템 프롬프트]", "[유저 프롬프트]"} or markers["[시스템 프롬프트]"] != 0 or markers["[유저 프롬프트]"] <= 1:
        raise ValueError("evaluation prompt section markers differ")
    split = markers["[유저 프롬프트]"]
    system = "".join(lines[1:split])
    user = "".join(lines[split + 1:])
    if user.count("{주제 지문}") != 1 or user.count("{논증적 글 본문}") != 1:
        raise ValueError("evaluation prompt placeholders differ")
    return system, user


def _rationale_aware_contract() -> dict[str, object]:
    if not RATIONALE_AWARE_PROMPT_PATH.is_file() or RATIONALE_AWARE_PROMPT_PATH.is_symlink():
        raise ValueError("rationale-aware score prompt is unavailable")
    payload = RATIONALE_AWARE_PROMPT_PATH.read_bytes()
    if sha256(payload).hexdigest() != RATIONALE_AWARE_PROMPT_SHA256:
        raise ValueError("rationale-aware score prompt digest differs")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("rationale-aware score prompt is invalid UTF-8 JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "mal2026-rationale-aware-encoder-score-prompt-v1":
        raise ValueError("rationale-aware score prompt schema differs")
    provenance = value.get("provenance")
    messages = value.get("messages")
    input_contract = value.get("input_contract")
    output_contract = value.get("model_output_contract")
    supervision = value.get("supervision")
    if not isinstance(provenance, dict) or provenance.get("source_file_sha256") != EVALUATION_PROMPT_SHA256 or provenance.get("verbatim_source_prompt") is not False:
        raise ValueError("rationale-aware score prompt provenance differs")
    if not isinstance(messages, dict) or not all(isinstance(messages.get(key), str) and messages[key].strip() for key in ("system", "user_preamble")):
        raise ValueError("rationale-aware score prompt messages differ")
    if not isinstance(input_contract, dict) or input_contract.get("required_rationale_structure") != "bundle" or input_contract.get("required_rationale_axes") != ["content", "organization", "expression"] or input_contract.get("gold_or_reference_score_allowed") is not False or input_contract.get("average_allowed") is not False:
        raise ValueError("rationale-aware score input contract differs")
    if not isinstance(output_contract, dict) or output_contract.get("head_order") != ["content", "organization", "expression"] or output_contract.get("value_type") != "continuous_number" or output_contract.get("range") != [1.0, 5.0] or output_contract.get("average_allowed") is not False or output_contract.get("rationale_output_allowed") is not False:
        raise ValueError("rationale-aware score output contract differs")
    if not isinstance(supervision, dict) or supervision.get("target_projection") != "none_preserve_raw_continuous" or supervision.get("score_average_read") is not False or supervision.get("score_average_target_used") is not False:
        raise ValueError("rationale-aware score supervision differs")
    return value


def instruction(kind: str) -> str:
    if kind == LEGACY_COMPACT:
        return LEGACY_COMPACT_INSTRUCTION
    if kind == PUBLIC_SPEC_SCORE_ONLY:
        return PUBLIC_SPEC_SCORE_ONLY_INSTRUCTION
    if kind == USER_SUPPLIED_EVALUATION:
        return _evaluation_sections()[0]
    if kind == RATIONALE_AWARE_ENCODER:
        return str(_rationale_aware_contract()["messages"]["system"])  # type: ignore[index]
    raise ValueError("unknown score prompt kind")


def system_prompt(kind: str) -> str:
    if kind == LEGACY_COMPACT:
        return "당신은 한국어 글쓰기 평가자입니다. 과제와 학생 글만 근거로 세 정수 점수만 출력하십시오."
    if kind == USER_SUPPLIED_EVALUATION:
        return _evaluation_sections()[0]
    if kind == RATIONALE_AWARE_ENCODER:
        return instruction(kind)
    return instruction(kind) + "\n" + DECODER_OUTPUT_RULE


def query_text(
    prompt_text: str,
    essay_text: str,
    rationales: Mapping[str, str] | None = None,
    kind: str = LEGACY_COMPACT,
) -> str:
    axes = ("content", "organization", "expression")
    if rationales is not None and (
        set(rationales) != set(axes)
        or not all(isinstance(rationales[axis], str) and rationales[axis].strip() for axis in axes)
    ):
        raise ValueError("three nonblank rationale axes are required")
    if kind == RATIONALE_AWARE_ENCODER:
        if rationales is None:
            raise ValueError("rationale-aware score prompt requires bundle rationales")
        contract = _rationale_aware_contract()
        messages = contract["messages"]
        input_contract = contract["input_contract"]
        payload = {
            str(input_contract["payload_key"]): {  # type: ignore[index]
                "prompt_text": prompt_text,
                "essay_text": essay_text,
                "evaluation_rationales": {axis: rationales[axis] for axis in axes},
            }
        }
        return str(messages["user_preamble"]) + "\n" + json.dumps(  # type: ignore[index]
            payload, ensure_ascii=False, separators=(",", ":")
        )
    if kind == USER_SUPPLIED_EVALUATION:
        template = _evaluation_sections()[1]
        text = template.replace("{주제 지문}", prompt_text).replace("{논증적 글 본문}", essay_text)
    elif kind in (LEGACY_COMPACT, PUBLIC_SPEC_SCORE_ONLY):
        text = (
            f"<writing_prompt>\n{prompt_text}\n</writing_prompt>\n"
            f"<student_essay>\n{essay_text}\n</student_essay>"
        )
    else:
        raise ValueError("unknown score prompt kind")
    if rationales is not None:
        rendered = "\n".join(f"<{axis}>{rationales[axis]}</{axis}>" for axis in axes)
        text += (
            f"\n\n[evaluation_rationales]\n{rendered}"
            if kind == USER_SUPPLIED_EVALUATION
            else f"\n<evaluation_rationales>\n{rendered}\n</evaluation_rationales>"
        )
    return text


def embedding_input(
    prompt_text: str,
    essay_text: str,
    kind: str,
    rationales: Mapping[str, str] | None = None,
) -> str:
    return f"Instruct: {instruction(kind)}\nQuery:\n{query_text(prompt_text, essay_text, rationales, kind)}"


def prompt_sha256(kind: str) -> str:
    if kind == USER_SUPPLIED_EVALUATION:
        _evaluation_sections()
        return EVALUATION_PROMPT_SHA256
    if kind == RATIONALE_AWARE_ENCODER:
        _rationale_aware_contract()
        return RATIONALE_AWARE_PROMPT_SHA256
    payload = system_prompt(kind) + "\n" + DECODER_OUTPUT_RULE + "\n" + instruction(kind)
    return sha256(payload.encode("utf-8")).hexdigest()


def provenance(kind: str) -> dict[str, str]:
    return {
        "score_prompt_kind": kind,
        "score_prompt_sha256": prompt_sha256(kind),
        "prompt_provenance": {
            LEGACY_COMPACT: "legacy_compact_reproduction",
            PUBLIC_SPEC_SCORE_ONLY: "public_spec_aligned_score_only_reconstruction_not_verbatim_organizer_prompt",
            USER_SUPPLIED_EVALUATION: "user_supplied_evaluation_txt_exact_section_routing",
            RATIONALE_AWARE_ENCODER: "derived_evaluation_txt_rationale_aware_continuous_three_axis_encoder_contract",
        }[kind],
    }
