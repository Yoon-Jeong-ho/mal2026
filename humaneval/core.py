"""Local, privacy-preserving human validation study support.

The web application deliberately keeps writing text and rationales in memory.
Only reviewer responses, opaque source identifiers, and study provenance are
written to the ignored SQLite result store.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence


AXES = ("content", "organization", "expression")
AXIS_LABELS = {"content": "내용", "organization": "구성", "expression": "표현"}
ALLOWED_USERS = ("명훈", "찬희", "정호", "지민")
AXIS_BAND_TARGET = {1: 4, 2: 4, 3: 2, 4: 2, 5: 8}
RATIONALE_VERDICTS = ("appropriate", "partial", "inappropriate")
COMMON_NOTICE_MARKER = "[유의 사항]"


class HumanValidationError(RuntimeError):
    """Raised when study inputs or state fail closed."""


class HumanValidationConflict(HumanValidationError):
    """Raised for stale or out-of-order reviewer submissions."""


@dataclass(frozen=True)
class EvaluationRow:
    source_id: str
    split: str
    prompt: str
    essay: str
    scores: Mapping[str, float]

    @property
    def score_bands(self) -> Mapping[str, int]:
        return {
            axis: max(
                1,
                min(5, int(Decimal(str(self.scores[axis])).quantize(Decimal("1"), rounding=ROUND_HALF_UP))),
            )
            for axis in AXES
        }


@dataclass(frozen=True)
class Rubric:
    criteria: Mapping[str, tuple[str, ...]]
    score_scale: Mapping[int, str]

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "criteria": {
                axis: {"label": AXIS_LABELS[axis], "points": list(self.criteria[axis])}
                for axis in AXES
            },
            "score_scale": {str(score): self.score_scale[score] for score in range(1, 6)},
        }


@dataclass(frozen=True)
class JudgeGuide:
    intro: str
    checks: tuple[tuple[str, str], ...]

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "intro": self.intro,
            "checks": [{"title": title, "description": description} for title, description in self.checks],
        }


@dataclass(frozen=True)
class StudyItem:
    index: int
    source_id: str
    split: str
    score_bands: Mapping[str, int]
    topic_prompt: str
    essay: str
    api_rationale: Mapping[str, str]
    model_rationale: Mapping[str, str]
    first_source: str

    def rationale_for(self, source: str) -> Mapping[str, str]:
        if source == "api":
            return self.api_rationale
        if source == "model":
            return self.model_rationale
        raise HumanValidationError(f"unknown rationale source: {source}")


@dataclass(frozen=True)
class Study:
    items: tuple[StudyItem, ...]
    common_notice: str
    rubric: Rubric
    judge_guide: JudgeGuide
    fingerprint: str
    axis_band_target: Mapping[int, int]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise HumanValidationError(f"JSONL input does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise HumanValidationError(f"blank JSONL row: {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HumanValidationError(f"invalid JSONL row: {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise HumanValidationError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise HumanValidationError(f"JSONL input is empty: {path}")
    return rows


def load_evaluation_rows(split_paths: Mapping[str, Path]) -> tuple[EvaluationRow, ...]:
    result: list[EvaluationRow] = []
    seen: set[str] = set()
    for split in ("train", "validation"):
        path = split_paths.get(split)
        if path is None:
            raise HumanValidationError(f"missing {split} evaluation path")
        for raw in _read_jsonl(path):
            source_id = raw.get("id")
            prompt, essay, scores = raw.get("prompt"), raw.get("essay"), raw.get("score")
            if not isinstance(source_id, str) or not source_id or source_id in seen:
                raise HumanValidationError("evaluation source IDs must be non-empty and unique")
            if not isinstance(prompt, str) or not prompt.strip() or not isinstance(essay, str) or not essay.strip():
                raise HumanValidationError(f"evaluation text is missing for source ID {source_id}")
            if not isinstance(scores, dict) or not all(axis in scores for axis in (*AXES, "average")):
                raise HumanValidationError(f"evaluation scores are malformed for source ID {source_id}")
            normalized_scores: dict[str, float] = {}
            for axis in (*AXES, "average"):
                value = scores[axis]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= float(value) <= 5:
                    raise HumanValidationError(f"evaluation score is outside 1..5 for source ID {source_id}")
                normalized_scores[axis] = float(value)
            seen.add(source_id)
            result.append(EvaluationRow(source_id, split, prompt.strip(), essay.strip(), normalized_scores))
    return tuple(result)


def separate_common_notice(rows: Sequence[EvaluationRow]) -> tuple[dict[str, str], str]:
    """Return per-source topic prompts and the one exact shared notice suffix."""
    topic_prompts: dict[str, str] = {}
    notices: set[str] = set()
    for row in rows:
        topic, marker, suffix = row.prompt.partition(COMMON_NOTICE_MARKER)
        if not marker or not topic.strip() or not suffix.strip():
            raise HumanValidationError(f"prompt lacks a separable common notice: {row.source_id}")
        topic_prompts[row.source_id] = topic.strip()
        notices.add((marker + suffix).strip())
    if len(notices) != 1:
        raise HumanValidationError(f"expected one common prompt notice, observed {len(notices)}")
    return topic_prompts, next(iter(notices))


def load_rationales(paths: Sequence[Path], *, preferred_candidate: int | None = None) -> dict[str, dict[str, str]]:
    """Load either API ``rationale`` or model ``rationales`` JSONL bundles."""
    result: dict[str, dict[str, str]] = {}
    for path in paths:
        for raw in _read_jsonl(path):
            if preferred_candidate is not None and "candidate" in raw and raw.get("candidate") != preferred_candidate:
                continue
            source_id = raw.get("source_id")
            bundle = raw.get("rationale", raw.get("rationales"))
            if bundle is None:
                bundle = raw.get("participant_output")
            if not isinstance(source_id, str) or not source_id or not isinstance(bundle, dict):
                raise HumanValidationError(f"rationale row schema differs in {path}")
            normalized: dict[str, str] = {}
            for axis in AXES:
                value = bundle.get(axis)
                if isinstance(value, dict):
                    if isinstance(value.get("rationale"), str):
                        value = value["rationale"]
                    elif isinstance(value.get("diagnosis"), str):
                        # The API source also carries ``next_step``. It is
                        # intentionally excluded so both blind sources expose
                        # evaluation evidence rather than an asymmetric editing
                        # suggestion contract.
                        value = value["diagnosis"]
                if not isinstance(value, str) or not value.strip():
                    raise HumanValidationError(f"missing {axis} rationale for source ID {source_id}")
                normalized[axis] = value.strip()
            if source_id in result:
                raise HumanValidationError(f"duplicate rationale source ID after filtering: {source_id}")
            result[source_id] = normalized
    if not result:
        raise HumanValidationError("rationale inputs produced no usable rows")
    return result


def parse_evaluation_rubric(path: Path) -> Rubric:
    if not path.is_file():
        raise HumanValidationError(f"rubric does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    criteria: dict[str, tuple[str, ...]] = {}
    for position, axis in enumerate(AXES, start=1):
        marker = f"{position}. {axis}"
        start = text.find(marker)
        if start < 0:
            raise HumanValidationError(f"rubric lacks criterion section: {axis}")
        body_start = start + len(marker)
        next_markers = [text.find(f"{position + 1}. {AXES[position]}", body_start)] if position < len(AXES) else []
        next_markers.append(text.find("[점수 기준]", body_start))
        ends = [value for value in next_markers if value >= 0]
        if not ends:
            raise HumanValidationError(f"rubric criterion section is unterminated: {axis}")
        points = tuple(
            line[2:].strip()
            for line in text[body_start:min(ends)].splitlines()
            if line.strip().startswith("- ")
        )
        if not points:
            raise HumanValidationError(f"rubric criterion has no points: {axis}")
        criteria[axis] = points

    score_scale: dict[int, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        for score in range(1, 6):
            prefix = f"- {score}점:"
            if stripped.startswith(prefix):
                score_scale[score] = stripped[len(prefix):].strip()
    if set(score_scale) != set(range(1, 6)):
        raise HumanValidationError("rubric must define every integer score from 1 through 5")
    return Rubric(criteria, score_scale)


def parse_judge_guide(path: Path) -> JudgeGuide:
    if not path.is_file():
        raise HumanValidationError(f"rationale judge guide does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    required = (
        "domain_match",
        "score_rationale_consistency",
        "specificity",
        "groundedness",
        "generic한 총평",
        "essay_text에 없는 내용",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise HumanValidationError(f"rationale judge guide contract differs: {missing}")
    return JudgeGuide(
        intro="점수를 다시 채점하는 것이 아니라, 각 영역의 평가 설명이 학생 글에 비추어 타당한지 판단해 주세요.",
        checks=(
            ("영역 적합성", "내용·구성·표현 중 지금 보는 영역의 기준에 맞는 근거인지 확인합니다."),
            ("구체성", "막연한 총평이 아니라 실제 주장, 문장, 문단 전개나 오류 양상을 구체적으로 짚는지 봅니다."),
            ("글 근거성", "설명이 학생 글에서 실제로 확인되며, 글에 없는 내용을 만들어내지 않았는지 확인합니다."),
            ("엄격한 판단", "상투적이거나 근거가 부족하거나 다른 영역의 기준이 섞이면 일부 적절함 또는 적절하지 않음을 선택합니다."),
        ),
    )


def _stable_key(seed: int, namespace: str, source_id: str) -> str:
    return sha256(f"{seed}:{namespace}:{source_id}".encode("utf-8")).hexdigest()


def _exact_vector_rows(
    rows: Sequence[EvaluationRow], available_ids: set[str], vector: tuple[int, int, int], count: int, seed: int,
) -> list[EvaluationRow]:
    eligible = [
        row for row in rows
        if row.source_id in available_ids and tuple(row.score_bands[axis] for axis in AXES) == vector
    ]
    eligible.sort(key=lambda row: _stable_key(seed, f"vector-{vector}", row.source_id))
    if len(eligible) < count:
        raise HumanValidationError(f"score vector {vector} has {len(eligible)} eligible rows; {count} required")
    return eligible[:count]


def _balanced_low_rows(
    rows: Sequence[EvaluationRow], available_ids: set[str], *, seed: int,
) -> list[EvaluationRow]:
    """Pick eight 1/2-only essays with four 1s and four 2s on every axis."""
    eligible = [
        row for row in rows
        if row.source_id in available_ids and all(row.score_bands[axis] in {1, 2} for axis in AXES)
    ]
    eligible.sort(key=lambda row: _stable_key(seed, "balanced-low", row.source_id))
    # State is (items, content_ones, organization_ones, expression_ones).
    # Keeping the first path after stable sorting makes the exact solution
    # deterministic without copying any writing text into the protocol.
    states: dict[tuple[int, int, int, int], tuple[EvaluationRow, ...]] = {(0, 0, 0, 0): ()}
    for row in eligible:
        increments = tuple(1 if row.score_bands[axis] == 1 else 0 for axis in AXES)
        updated = dict(states)
        for (count, content, organization, expression), chosen in states.items():
            next_state = (
                count + 1,
                content + increments[0],
                organization + increments[1],
                expression + increments[2],
            )
            if next_state[0] <= 8 and all(value <= 4 for value in next_state[1:]):
                updated.setdefault(next_state, (*chosen, row))
        states = updated
    selected = states.get((8, 4, 4, 4))
    if selected is None:
        raise HumanValidationError("cannot construct the required per-axis balanced 1/2 subset")
    return list(selected)


def select_rows(
    rows: Sequence[EvaluationRow], available_ids: set[str], *, seed: int,
) -> tuple[EvaluationRow, ...]:
    """Select 20 essays with the same explicit distribution on every axis.

    Each of content, organization, and expression contains four score-1 rows,
    four score-2 rows, two score-3 rows, two score-4 rows, and eight score-5
    rows after half-up rounding of that axis's source score.
    """
    selected = _balanced_low_rows(rows, available_ids, seed=seed)
    selected.extend(_exact_vector_rows(rows, available_ids, (3, 3, 3), 2, seed))
    selected.extend(_exact_vector_rows(rows, available_ids, (4, 4, 4), 2, seed))
    selected.extend(_exact_vector_rows(rows, available_ids, (5, 5, 5), 8, seed))
    if len({row.source_id for row in selected}) != 20:
        raise HumanValidationError("axis-balanced selection contains duplicate source IDs")
    selected.sort(key=lambda row: _stable_key(seed, "final-order", row.source_id))
    counts = axis_band_counts_from_rows(selected)
    expected = {axis: dict(AXIS_BAND_TARGET) for axis in AXES}
    if counts != expected:
        raise HumanValidationError(f"axis-balanced selection differs: {counts}")
    return tuple(selected)


def build_study(
    *,
    split_paths: Mapping[str, Path],
    rubric_path: Path,
    judge_guide_path: Path,
    api_rationale_paths: Sequence[Path],
    model_rationale_paths: Sequence[Path],
    seed: int = 20260805,
    api_candidate: int = 1,
) -> Study:
    rows = load_evaluation_rows(split_paths)
    topic_prompts, common_notice = separate_common_notice(rows)
    rubric = parse_evaluation_rubric(rubric_path)
    judge_guide = parse_judge_guide(judge_guide_path)
    api = load_rationales(api_rationale_paths, preferred_candidate=api_candidate)
    model = load_rationales(model_rationale_paths)
    available = set(api) & set(model)
    chosen = select_rows(rows, available, seed=seed)

    items: list[StudyItem] = []
    for index, row in enumerate(chosen):
        first_source = "api" if int(_stable_key(seed, "rationale-order", row.source_id), 16) % 2 == 0 else "model"
        items.append(StudyItem(
            index=index,
            source_id=row.source_id,
            split=row.split,
            score_bands=row.score_bands,
            topic_prompt=topic_prompts[row.source_id],
            essay=row.essay,
            api_rationale=api[row.source_id],
            model_rationale=model[row.source_id],
            first_source=first_source,
        ))

    fingerprint_payload = {
        "schema": "mal2026-human-validation-study-v1",
        "seed": seed,
        "axis_band_target": {str(key): AXIS_BAND_TARGET[key] for key in sorted(AXIS_BAND_TARGET)},
        "common_notice_sha256": sha256(common_notice.encode("utf-8")).hexdigest(),
        "rubric_sha256": sha256(rubric_path.read_bytes()).hexdigest(),
        "judge_guide_sha256": sha256(judge_guide_path.read_bytes()).hexdigest(),
        "items": [
            {
                "source_id": item.source_id,
                "split": item.split,
                "score_bands": dict(item.score_bands),
                "first_source": item.first_source,
                "prompt_sha256": sha256(item.topic_prompt.encode("utf-8")).hexdigest(),
                "essay_sha256": sha256(item.essay.encode("utf-8")).hexdigest(),
                "api_sha256": sha256(json.dumps(item.api_rationale, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
                "model_sha256": sha256(json.dumps(item.model_rationale, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
            }
            for item in items
        ],
    }
    fingerprint = sha256(json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return Study(tuple(items), common_notice, rubric, judge_guide, fingerprint, dict(AXIS_BAND_TARGET))


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ResponseStore:
    """Concurrency-safe SQLite response store with strict phase transitions."""

    def __init__(self, path: Path, study: Study):
        self.path = path
        self.study = study
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._initialize()
        try:
            path.chmod(0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS study_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS study_items (
                    item_index INTEGER PRIMARY KEY,
                    source_id TEXT NOT NULL UNIQUE,
                    split TEXT NOT NULL CHECK(split IN ('train', 'validation')),
                    target_content_band INTEGER NOT NULL CHECK(target_content_band BETWEEN 1 AND 5),
                    target_organization_band INTEGER NOT NULL CHECK(target_organization_band BETWEEN 1 AND 5),
                    target_expression_band INTEGER NOT NULL CHECK(target_expression_band BETWEEN 1 AND 5),
                    first_source TEXT NOT NULL CHECK(first_source IN ('api', 'model'))
                );
                CREATE TABLE IF NOT EXISTS responses (
                    user_name TEXT NOT NULL,
                    item_index INTEGER NOT NULL,
                    content_score INTEGER CHECK(content_score BETWEEN 1 AND 5),
                    organization_score INTEGER CHECK(organization_score BETWEEN 1 AND 5),
                    expression_score INTEGER CHECK(expression_score BETWEEN 1 AND 5),
                    content_reason TEXT,
                    organization_reason TEXT,
                    expression_reason TEXT,
                    score_submitted_at TEXT,
                    rationale_a_source TEXT CHECK(rationale_a_source IN ('api', 'model')),
                    rationale_a_content_verdict TEXT CHECK(rationale_a_content_verdict IN ('appropriate', 'partial', 'inappropriate')),
                    rationale_a_organization_verdict TEXT CHECK(rationale_a_organization_verdict IN ('appropriate', 'partial', 'inappropriate')),
                    rationale_a_expression_verdict TEXT CHECK(rationale_a_expression_verdict IN ('appropriate', 'partial', 'inappropriate')),
                    rationale_a_content_reason TEXT,
                    rationale_a_organization_reason TEXT,
                    rationale_a_expression_reason TEXT,
                    rationale_a_submitted_at TEXT,
                    rationale_b_source TEXT CHECK(rationale_b_source IN ('api', 'model')),
                    rationale_b_content_verdict TEXT CHECK(rationale_b_content_verdict IN ('appropriate', 'partial', 'inappropriate')),
                    rationale_b_organization_verdict TEXT CHECK(rationale_b_organization_verdict IN ('appropriate', 'partial', 'inappropriate')),
                    rationale_b_expression_verdict TEXT CHECK(rationale_b_expression_verdict IN ('appropriate', 'partial', 'inappropriate')),
                    rationale_b_content_reason TEXT,
                    rationale_b_organization_reason TEXT,
                    rationale_b_expression_reason TEXT,
                    rationale_b_submitted_at TEXT,
                    PRIMARY KEY(user_name, item_index),
                    FOREIGN KEY(item_index) REFERENCES study_items(item_index)
                );
            """)
            existing = connection.execute("SELECT value FROM study_meta WHERE key = 'fingerprint'").fetchone()
            if existing is None:
                connection.execute("INSERT INTO study_meta(key, value) VALUES ('fingerprint', ?)", (self.study.fingerprint,))
                connection.execute("INSERT INTO study_meta(key, value) VALUES ('created_at', ?)", (_now(),))
                connection.executemany(
                    """INSERT INTO study_items(
                        item_index, source_id, split, target_content_band,
                        target_organization_band, target_expression_band, first_source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [(
                        item.index, item.source_id, item.split,
                        item.score_bands["content"], item.score_bands["organization"],
                        item.score_bands["expression"], item.first_source,
                    ) for item in self.study.items],
                )
            else:
                if existing["value"] != self.study.fingerprint:
                    raise HumanValidationError("existing response database belongs to a different study fingerprint")
                persisted = connection.execute(
                    """SELECT item_index, source_id, split, target_content_band,
                        target_organization_band, target_expression_band, first_source
                        FROM study_items ORDER BY item_index"""
                ).fetchall()
                expected = [(
                    item.index, item.source_id, item.split,
                    item.score_bands["content"], item.score_bands["organization"],
                    item.score_bands["expression"], item.first_source,
                ) for item in self.study.items]
                observed = [tuple(row) for row in persisted]
                if observed != expected:
                    raise HumanValidationError("persisted study item order differs from current inputs")

    @staticmethod
    def _validate_user(user_name: str) -> None:
        if user_name not in ALLOWED_USERS:
            raise HumanValidationError("reviewer name is not allowed")

    @staticmethod
    def _clean_reason(reason: Any) -> str:
        if not isinstance(reason, str):
            raise HumanValidationError("reason must be text")
        value = reason.strip()
        if len(value) > 2000:
            raise HumanValidationError("reason must be at most 2,000 characters")
        return value

    def _response(self, connection: sqlite3.Connection, user_name: str, item_index: int) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM responses WHERE user_name = ? AND item_index = ?", (user_name, item_index)
        ).fetchone()

    def _phase(self, row: sqlite3.Row | None) -> str:
        if row is None or row["score_submitted_at"] is None:
            return "score"
        if row["rationale_a_submitted_at"] is None:
            return "rationale_a"
        if row["rationale_b_submitted_at"] is None:
            return "rationale_b"
        return "complete"

    def _next(self, connection: sqlite3.Connection, user_name: str) -> tuple[StudyItem | None, sqlite3.Row | None, str]:
        for item in self.study.items:
            row = self._response(connection, user_name, item.index)
            phase = self._phase(row)
            if phase != "complete":
                return item, row, phase
        return None, None, "finished"

    def state(self, user_name: str) -> dict[str, Any]:
        self._validate_user(user_name)
        with self._connect() as connection:
            return self._state(connection, user_name)

    def _state(self, connection: sqlite3.Connection, user_name: str) -> dict[str, Any]:
        completed = int(connection.execute(
            "SELECT COUNT(*) FROM responses WHERE user_name = ? AND rationale_b_submitted_at IS NOT NULL", (user_name,)
        ).fetchone()[0])
        item, row, phase = self._next(connection, user_name)
        base: dict[str, Any] = {
            "user": user_name,
            "progress": {"completed": completed, "total": len(self.study.items)},
            "phase": phase,
            "common_notice": self.study.common_notice,
            "rubric": self.study.rubric.as_public_dict(),
            "judge_guide": self.study.judge_guide.as_public_dict(),
        }
        if item is None:
            return base
        base["item"] = {
            "index": item.index,
            "number": item.index + 1,
            "topic_prompt": item.topic_prompt,
            "essay": item.essay,
        }
        if phase.startswith("rationale"):
            assert row is not None
            source = item.first_source if phase == "rationale_a" else ("model" if item.first_source == "api" else "api")
            label = "A" if phase == "rationale_a" else "B"
            base["submitted_scores"] = {axis: int(row[f"{axis}_score"]) for axis in AXES}
            base["rationale"] = {
                "label": label,
                "texts": dict(item.rationale_for(source)),
            }
        return base

    def record_scores(
        self,
        user_name: str,
        item_index: int,
        scores: Mapping[str, Any],
        reasons: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._validate_user(user_name)
        if set(scores) != set(AXES) or any(type(scores[axis]) is not int or not 1 <= scores[axis] <= 5 for axis in AXES):
            raise HumanValidationError("all three scores must be integers from 1 through 5")
        if set(reasons) != set(AXES):
            raise HumanValidationError("a separate reason field is required for all three criteria")
        cleaned_reasons = {axis: self._clean_reason(reasons[axis]) for axis in AXES}
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            item, row, phase = self._next(connection, user_name)
            if item is None:
                raise HumanValidationConflict("review is already complete")
            if item.index != item_index or phase != "score":
                if (
                    row is not None
                    and item.index == item_index
                    and all(row[f"{axis}_score"] == scores[axis] for axis in AXES)
                    and all((row[f"{axis}_reason"] or "") == cleaned_reasons[axis] for axis in AXES)
                ):
                    connection.commit()
                    return self._state(connection, user_name)
                raise HumanValidationConflict("score submission is stale or out of order")
            connection.execute(
                """INSERT INTO responses(
                    user_name, item_index, content_score, organization_score, expression_score,
                    content_reason, organization_reason, expression_reason, score_submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_name, item_index,
                    scores["content"], scores["organization"], scores["expression"],
                    cleaned_reasons["content"], cleaned_reasons["organization"],
                    cleaned_reasons["expression"], _now(),
                ),
            )
            connection.commit()
            return self._state(connection, user_name)

    def record_rationale(
        self,
        user_name: str,
        item_index: int,
        verdicts: Mapping[str, Any],
        reasons: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._validate_user(user_name)
        if set(verdicts) != set(AXES) or any(verdicts[axis] not in RATIONALE_VERDICTS for axis in AXES):
            raise HumanValidationError("all three rationale verdicts are required")
        if set(reasons) != set(AXES):
            raise HumanValidationError("a separate rationale reason field is required for all three criteria")
        cleaned_reasons = {axis: self._clean_reason(reasons[axis]) for axis in AXES}
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            item, row, phase = self._next(connection, user_name)
            if item is None or row is None:
                raise HumanValidationConflict("rationale submission is stale or the review is complete")
            if item.index != item_index or phase not in {"rationale_a", "rationale_b"}:
                raise HumanValidationConflict("rationale submission is stale or out of order")
            suffix = "a" if phase == "rationale_a" else "b"
            source = item.first_source if suffix == "a" else ("model" if item.first_source == "api" else "api")
            connection.execute(
                f"""UPDATE responses SET
                    rationale_{suffix}_source = ?,
                    rationale_{suffix}_content_verdict = ?,
                    rationale_{suffix}_organization_verdict = ?,
                    rationale_{suffix}_expression_verdict = ?,
                    rationale_{suffix}_content_reason = ?,
                    rationale_{suffix}_organization_reason = ?,
                    rationale_{suffix}_expression_reason = ?,
                    rationale_{suffix}_submitted_at = ?
                    WHERE user_name = ? AND item_index = ?""",
                (
                    source,
                    verdicts["content"], verdicts["organization"], verdicts["expression"],
                    cleaned_reasons["content"], cleaned_reasons["organization"], cleaned_reasons["expression"],
                    _now(), user_name, item_index,
                ),
            )
            connection.commit()
            return self._state(connection, user_name)

    def export_jsonl(self, path: Path) -> int:
        """Export response rows without copying essay, prompt, or rationale text."""
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._connect() as connection, path.open("w", encoding="utf-8") as handle:
            rows = connection.execute("""
                SELECT r.*, i.source_id, i.split, i.target_content_band,
                    i.target_organization_band, i.target_expression_band, i.first_source
                FROM responses AS r JOIN study_items AS i USING(item_index)
                ORDER BY r.user_name, r.item_index
            """).fetchall()
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return len(rows)


def axis_band_counts_from_rows(rows: Iterable[EvaluationRow]) -> dict[str, dict[int, int]]:
    counts = {axis: {score: 0 for score in range(1, 6)} for axis in AXES}
    for row in rows:
        for axis in AXES:
            counts[axis][row.score_bands[axis]] += 1
    return counts


def axis_band_counts(items: Iterable[StudyItem]) -> dict[str, dict[int, int]]:
    counts = {axis: {score: 0 for score in range(1, 6)} for axis in AXES}
    for item in items:
        for axis in AXES:
            counts[axis][item.score_bands[axis]] += 1
    return counts
