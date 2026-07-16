"""Restricted JSONL validation and deterministic prompt-group development split.

The functions in this module return in-memory records only.  Callers must not
serialize records or free-text fields to tracked files, W&B, or summaries.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from .constants import DEFAULT_DEV_FRACTION, SCORE_FIELDS, SCORE_MAX, SCORE_MIN

_WS = re.compile(r"\s+")
_REQUIRED_RECORD_FIELDS = frozenset({"id", "document_id", "prompt_num", "prompt", "essay", "score"})


class DataContractError(ValueError):
    """Raised when restricted data violate the frozen experiment contract."""


@dataclass(frozen=True)
class ScoreVector:
    content: float
    organization: float
    expression: float
    average: float

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in SCORE_FIELDS}

    def as_tuple(self) -> tuple[float, float, float, float]:
        return tuple(getattr(self, name) for name in SCORE_FIELDS)  # type: ignore[return-value]


@dataclass(frozen=True)
class DatasetRecord:
    id: str
    document_id: str
    prompt_num: str
    prompt: str
    essay: str
    scores: ScoreVector


@dataclass(frozen=True)
class PromptGroupSplit:
    """In-memory split plus a non-sensitive, trackable manifest."""

    optimization_train: tuple[DatasetRecord, ...]
    development: tuple[DatasetRecord, ...]
    manifest: dict[str, Any]


def normalize_prompt(value: str) -> str:
    """Freeze prompt normalization used only for grouping and hashing."""
    if not isinstance(value, str):
        raise DataContractError("prompt must be a string")
    normalized = _WS.sub(" ", unicodedata.normalize("NFKC", value)).strip()
    if not normalized:
        raise DataContractError("prompt must not be blank after normalization")
    return normalized


def stable_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _nonblank_string(value: Any, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(f"line {line_number}: {field} must be a nonblank string")
    return value


def _validate_scores(raw: Any, line_number: int) -> ScoreVector:
    if not isinstance(raw, Mapping) or set(raw) != set(SCORE_FIELDS):
        raise DataContractError(f"line {line_number}: score keys must be exactly {SCORE_FIELDS}")
    values: dict[str, float] = {}
    for field in SCORE_FIELDS:
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DataContractError(f"line {line_number}: score.{field} must be a number")
        parsed = float(value)
        if not isfinite(parsed) or not SCORE_MIN <= parsed <= SCORE_MAX:
            raise DataContractError(f"line {line_number}: score.{field} must be finite and in [1, 5]")
        values[field] = parsed
    return ScoreVector(**values)


def validate_record(raw: Any, line_number: int) -> DatasetRecord:
    if not isinstance(raw, Mapping):
        raise DataContractError(f"line {line_number}: record must be an object")
    missing = _REQUIRED_RECORD_FIELDS - set(raw)
    if missing:
        raise DataContractError(f"line {line_number}: missing required fields: {sorted(missing)}")
    return DatasetRecord(
        id=_nonblank_string(raw["id"], "id", line_number),
        document_id=_nonblank_string(raw["document_id"], "document_id", line_number),
        prompt_num=str(raw["prompt_num"]),
        prompt=_nonblank_string(raw["prompt"], "prompt", line_number),
        essay=_nonblank_string(raw["essay"], "essay", line_number),
        scores=_validate_scores(raw["score"], line_number),
    )


def load_and_validate_jsonl(path: str | Path, expected_sha256: str | None = None) -> list[DatasetRecord]:
    """Load a restricted JSONL file after schema, duplicate-ID, and hash checks."""
    resolved = Path(path)
    if not resolved.is_file():
        raise DataContractError(f"dataset file does not exist: {resolved}")
    observed_hash = file_sha256(resolved)
    if expected_sha256 is not None and observed_hash != expected_sha256:
        raise DataContractError("dataset SHA-256 mismatch; refusing to train or evaluate")

    rows: list[DatasetRecord] = []
    seen_ids: set[str] = set()
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise DataContractError(f"line {line_number}: blank JSONL line is not allowed")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataContractError(f"line {line_number}: invalid JSON") from exc
            row = validate_record(raw, line_number)
            if row.id in seen_ids:
                raise DataContractError(f"line {line_number}: duplicate id")
            seen_ids.add(row.id)
            rows.append(row)
    if not rows:
        raise DataContractError("dataset must contain at least one record")
    return rows


def _prompt_groups(records: Sequence[DatasetRecord]) -> dict[str, list[DatasetRecord]]:
    groups: dict[str, list[DatasetRecord]] = defaultdict(list)
    prompt_by_num: dict[str, str] = {}
    for record in records:
        normalized = normalize_prompt(record.prompt)
        previous = prompt_by_num.setdefault(record.prompt_num, normalized)
        if previous != normalized:
            raise DataContractError("one prompt_num maps to multiple normalized prompts")
        groups[stable_hash(normalized)].append(record)
    return dict(groups)


def split_prompt_groups(
    records: Sequence[DatasetRecord], dev_fraction: float = DEFAULT_DEV_FRACTION
) -> PromptGroupSplit:
    """Select exactly one frozen prompt group nearest the requested dev size.

    The selection is deterministic: compare absolute record-count distance to
    the requested fraction, then use the normalized prompt SHA-256 as the
    lexicographic tie-break. This intentionally creates a group-disjoint dev
    set without persisting free text or raw identifiers in the manifest.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise DataContractError("dev_fraction must be strictly between 0 and 1")
    if not records:
        raise DataContractError("cannot split an empty record collection")
    groups = _prompt_groups(records)
    if len(groups) < 2:
        raise DataContractError("at least two prompt groups are required for train/dev split")
    target_count = len(records) * dev_fraction
    selected_hash = min(groups, key=lambda key: (abs(len(groups[key]) - target_count), key))
    development = tuple(groups[selected_hash])
    optimization_train = tuple(record for key, group in groups.items() if key != selected_hash for record in group)
    if not optimization_train or not development:
        raise DataContractError("prompt split produced an empty partition")
    manifest = {
        "schema_version": 1,
        "selection_algorithm": "one_prompt_group_nearest_fraction_then_sha256",
        "requested_dev_fraction": dev_fraction,
        "total_records": len(records),
        "optimization_train_records": len(optimization_train),
        "development_records": len(development),
        "development_prompt_hashes": [selected_hash],
        "optimization_record_id_sha256": stable_hash("\n".join(sorted(record.id for record in optimization_train))),
        "development_record_id_sha256": stable_hash("\n".join(sorted(record.id for record in development))),
    }
    return PromptGroupSplit(optimization_train, development, manifest)
