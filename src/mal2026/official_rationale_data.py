"""Restricted official-API candidate data for score-conditioned rationale SFT."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .api_rationale_data import AXES, ROOT, SOURCE_SHA256, load_writing_rows, sha256_file
from .official_writing_contract import parse_participant_output


API_RUN_ID = "official-openai-candidates-v1-train3-20260727-001"
API_ROOT = ROOT / "data/processed/restricted/official_openai_candidates_v1" / API_RUN_ID
MANIFEST = API_ROOT / "manifest.json"
CANDIDATES = API_ROOT / "candidates.train.jsonl"
EXPECTED_ESSAYS = 2000
EXPECTED_CANDIDATES = 6000
TASKS = ("bundle", *AXES)


class OfficialRationaleDataError(ValueError):
    """Raised when private official candidate data differs from its contract."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialRationaleDataError(message)


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    need(path.is_file() and not path.is_symlink(), "restricted official candidate file is unavailable")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            need(bool(line.strip()), f"blank official candidate row at {line_number}")
            value = json.loads(line)
            need(isinstance(value, dict), "official candidate row must be an object")
            yield value


@dataclass(frozen=True)
class OfficialCandidate:
    source_id: str
    candidate_number: int
    scores: Mapping[str, int]
    rationales: Mapping[str, str]


def axes_for_task(task: str) -> tuple[str, ...]:
    need(task in TASKS, "official rationale task differs")
    return AXES if task == "bundle" else (task,)


def candidate_provenance() -> dict[str, Any]:
    need(MANIFEST.is_file() and not MANIFEST.is_symlink(), "official candidate manifest is unavailable")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "mal2026-official-openai-candidate-v1",
        "status": "validated",
        "run_id": API_RUN_ID,
        "model": "gpt-5.6-terra",
        "split": "train",
        "train_rows": EXPECTED_ESSAYS,
        "candidates_per_essay": 3,
        "requests": EXPECTED_CANDIDATES,
        "accepted": EXPECTED_CANDIDATES,
        "human_or_reference_score_read_or_prompted": False,
        "source_sha256": SOURCE_SHA256["train"],
    }
    need(isinstance(manifest, dict) and all(manifest.get(key) == value for key, value in expected.items()), "official candidate manifest differs")
    need(CANDIDATES.is_file() and sha256_file(CANDIDATES) == manifest.get("candidates_sha256"), "official candidate checksum differs")
    return {
        "run_id": API_RUN_ID,
        "model": manifest["model"],
        "source_sha256": manifest["source_sha256"],
        "candidate_sha256": manifest["candidates_sha256"],
        "official_system_prompt_sha256": manifest["official_system_prompt_sha256"],
        "essays": EXPECTED_ESSAYS,
        "candidates": EXPECTED_CANDIDATES,
        "human_or_reference_score_read_or_prompted": False,
    }


def load_candidates() -> list[OfficialCandidate]:
    candidate_provenance()
    writings = load_writing_rows("train", include_scores=False)
    essay_hashes = {row.identifier: sha256(row.essay.encode()).hexdigest() for row in writings}
    seen: set[str] = set()
    coverage: dict[str, set[int]] = {}
    result: list[OfficialCandidate] = []
    expected_fields = {"api_response_id", "candidate", "custom_id", "essay_sha256", "model", "participant_output", "schema_version", "source_id", "split"}
    for raw in jsonl(CANDIDATES):
        need(set(raw) == expected_fields and raw.get("schema_version") == "mal2026-official-openai-candidate-v1", "official candidate schema differs")
        need(raw.get("model") == "gpt-5.6-terra" and raw.get("split") == "train", "official candidate identity differs")
        custom_id, source_id, number = raw.get("custom_id"), raw.get("source_id"), raw.get("candidate")
        need(isinstance(custom_id, str) and custom_id not in seen, "official candidate custom ID differs")
        need(isinstance(source_id, str) and source_id in essay_hashes, "official candidate source ID differs")
        need(type(number) is int and number in {1, 2, 3}, "official candidate number differs")
        need(raw.get("essay_sha256") == essay_hashes[source_id], "official candidate essay linkage differs")
        parsed = parse_participant_output(raw.get("participant_output"))
        seen.add(custom_id)
        coverage.setdefault(source_id, set()).add(number)
        result.append(OfficialCandidate(
            source_id=source_id,
            candidate_number=number,
            scores={axis: int(parsed[axis]["score"]) for axis in AXES},
            rationales={axis: str(parsed[axis]["rationale"]) for axis in AXES},
        ))
    need(len(result) == EXPECTED_CANDIDATES and len(coverage) == EXPECTED_ESSAYS, "official candidate population differs")
    need(all(numbers == {1, 2, 3} for numbers in coverage.values()), "official candidate coverage differs")
    return result


RATIONALE_SYSTEM_PROMPT = """너는 한국어 논증적 글을 일관되게 평가하는 평가자이다.
주어진 predicted_score는 이미 다른 점수 모델이 결정한 최종 정수 점수이므로 바꾸거나 재채점하지 마라.
학생의 essay_text와 공식 영역 기준에 근거하여, 요청된 영역의 predicted_score를 타당하게 설명하는 한국어 rationale만 작성하라.

[공식 영역 기준]
- content: 문제에 대한 주장과 핵심 내용의 적절성, 근거의 충분성과 구체성, 주장과 근거의 논리적 연결
- organization: 서론·본론·결론 구조, 문단 간 연결, 일관된 논리 전개
- expression: 문장의 자연스러움과 명료성, 적절한 어휘, 맞춤법·띄어쓰기·문법·주술 호응

[원칙]
- essay_text에서 확인 가능한 구체적 문장, 표현, 논지, 문단 전개 또는 오류 양상을 근거로 삼아라.
- 영역을 서로 섞지 마라.
- 일반적이거나 템플릿 같은 총평을 피하라.
- 점수, 새 점수, 개선 제안은 출력하지 마라.
- 요청된 rationale JSON 객체 하나만 출력하고 코드블록이나 마크다운을 사용하지 마라."""


def rationale_object(rationales: Mapping[str, str], axes: Sequence[str]) -> dict[str, Any]:
    chosen = tuple(axes)
    need(bool(chosen) and set(chosen) <= set(AXES), "rationale output axes differ")
    return {axis: {"rationale": str(rationales[axis]).strip()} for axis in chosen}


def rationale_schema(axes: Sequence[str]) -> dict[str, Any]:
    chosen = tuple(axes)
    part = {"type": "object", "properties": {"rationale": {"type": "string", "minLength": 1}}, "required": ["rationale"], "additionalProperties": False}
    return {"type": "object", "properties": {axis: part for axis in chosen}, "required": list(chosen), "additionalProperties": False}


def parse_rationale_output(value: str | Mapping[str, Any], axes: Sequence[str]) -> dict[str, str]:
    chosen = tuple(axes)
    try:
        raw = json.loads(value) if isinstance(value, str) else dict(value)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OfficialRationaleDataError("rationale output is not JSON") from exc
    need(set(raw) == set(chosen), "rationale output axes differ")
    result: dict[str, str] = {}
    for axis in chosen:
        part = raw[axis]
        need(isinstance(part, Mapping) and set(part) == {"rationale"}, "rationale output shape differs")
        text = part["rationale"]
        need(isinstance(text, str) and bool(text.strip()), "rationale output must be nonblank")
        result[axis] = text.strip()
    return result


def messages(prompt: str, essay: str, scores: Mapping[str, int], axes: Sequence[str]) -> list[dict[str, str]]:
    chosen = tuple(axes)
    need(set(scores) == set(AXES) and all(type(scores[axis]) is int and 1 <= scores[axis] <= 5 for axis in AXES), "authoritative score vector differs")
    requested = ", ".join(chosen)
    score_text = json.dumps({axis: scores[axis] for axis in chosen}, ensure_ascii=False, separators=(",", ":"))
    shape = json.dumps(rationale_object({axis: f"[{axis} 판단 근거]" for axis in chosen}, chosen), ensure_ascii=False, separators=(",", ":"))
    user = (
        f"[요청 영역]\n{requested}\n\n[predicted_score]\n{score_text}\n\n"
        f"[출력 형식]\n{shape}\n\n[prompt_text]\n{prompt}\n\n[essay_text]\n{essay}"
    )
    return [{"role": "system", "content": RATIONALE_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def sft_examples(task: str, limit: int | None = None) -> list[dict[str, Any]]:
    axes = axes_for_task(task)
    writings = {row.identifier: row for row in load_writing_rows("train", include_scores=False)}
    candidates = load_candidates()
    if limit is not None:
        need(0 < limit <= len(candidates), "official SFT limit differs")
        candidates = candidates[:limit]
    result = []
    for candidate in candidates:
        writing = writings[candidate.source_id]
        result.append({
            "prompt": messages(writing.prompt, writing.essay, candidate.scores, axes),
            "completion": [{"role": "assistant", "content": json.dumps(rationale_object(candidate.rationales, axes), ensure_ascii=False, separators=(",", ":"))}],
        })
    need(len(result) == (limit if limit is not None else EXPECTED_CANDIDATES), "official SFT example count differs")
    return result
