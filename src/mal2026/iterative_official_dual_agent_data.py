"""Fail-closed loader for the restricted Terra/Luna official candidates.

Only validated, checksum-bound train artifacts are accepted.  Row-level
participant content is returned solely as restricted ``OfficialCandidate``
objects; provenance contains aggregate counts and hashes only.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from mal2026.official_writing_contract import AXES, parse_participant_output


SCHEMA_VERSION = "mal2026-official-openai-candidate-v1"
SOURCE_SHA256 = "b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737"
EXPECTED_ESSAYS = 2000
EXPECTED_CANDIDATES = 6000
CANDIDATES_PER_ESSAY = 3
SOURCE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("terra", "official-openai-candidates-v1-train3-20260727-001", "gpt-5.6-terra"),
    ("luna", "official-openai-candidates-luna-v1-train3-20260802-001", "gpt-5.6-luna"),
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_FIELDS = {
    "api_response_id", "candidate", "custom_id", "essay_sha256", "model",
    "participant_output", "schema_version", "source_id", "split",
}


class OfficialDualAgentDataError(ValueError):
    """Raised when either restricted source differs from the fixed contract."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialDualAgentDataError(message)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(value: str | Path, name: str) -> Path:
    path = Path(value)
    _need(path.is_file() and not path.is_symlink(), f"{name} must be an available regular file")
    return path


def _read_manifest(value: str | Path, *, source: str, run_id: str, model: str,
                   candidates_path: Path) -> tuple[Mapping[str, Any], str]:
    path = _regular_file(value, f"{source} manifest")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfficialDualAgentDataError(f"{source} manifest is not valid JSON") from exc
    expected = {
        "schema_version": SCHEMA_VERSION, "status": "validated", "run_id": run_id,
        "model": model, "split": "train", "train_rows": EXPECTED_ESSAYS,
        "candidates_per_essay": CANDIDATES_PER_ESSAY, "requests": EXPECTED_CANDIDATES,
        "accepted": EXPECTED_CANDIDATES, "source_sha256": SOURCE_SHA256,
        "human_or_reference_score_read_or_prompted": False,
    }
    _need(isinstance(raw, dict) and all(raw.get(key) == item for key, item in expected.items()),
          f"{source} manifest identity, validation, or population differs")
    candidate_sha = raw.get("candidates_sha256")
    _need(isinstance(candidate_sha, str) and _HEX64.fullmatch(candidate_sha) is not None,
          f"{source} candidate checksum is invalid")
    _need(_sha256_file(candidates_path) == candidate_sha, f"{source} candidate checksum differs")
    return MappingProxyType(raw), candidate_sha


def _essay_hashes(value: Mapping[str, str]) -> Mapping[str, str]:
    _need(isinstance(value, Mapping) and len(value) == EXPECTED_ESSAYS,
          "canonical essay SHA mapping must contain exactly 2,000 essays")
    result: dict[str, str] = {}
    for source_id, digest in value.items():
        _need(isinstance(source_id, str) and bool(source_id) and isinstance(digest, str)
              and _HEX64.fullmatch(digest) is not None, "canonical essay SHA mapping differs")
        result[source_id] = digest
    return MappingProxyType(result)


@dataclass(frozen=True)
class OfficialCandidate:
    agent_source: str
    run_id: str
    model: str
    source_id: str
    candidate_number: int
    essay_sha256: str
    scores: Mapping[str, int]
    rationales: Mapping[str, str]


def _load_source(*, source: str, run_id: str, model: str, manifest_path: str | Path,
                 candidates_path: str | Path, essay_hashes: Mapping[str, str]) -> tuple[list[OfficialCandidate], Mapping[str, Any]]:
    candidate_file = _regular_file(candidates_path, f"{source} candidates")
    manifest, candidate_sha = _read_manifest(
        manifest_path, source=source, run_id=run_id, model=model, candidates_path=candidate_file,
    )
    seen_custom: set[str] = set()
    coverage: dict[str, set[int]] = {}
    result: list[OfficialCandidate] = []
    try:
        handle = candidate_file.open(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise OfficialDualAgentDataError(f"{source} candidates are unreadable") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            _need(bool(line.strip()), f"{source} candidate row {line_number} is blank")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OfficialDualAgentDataError(f"{source} candidate row {line_number} is not JSON") from exc
            _need(isinstance(raw, dict) and set(raw) == _FIELDS and raw.get("schema_version") == SCHEMA_VERSION,
                  f"{source} candidate schema differs")
            _need(raw.get("model") == model and raw.get("split") == "train",
                  f"{source} candidate model or split differs")
            custom_id, source_id, number = raw.get("custom_id"), raw.get("source_id"), raw.get("candidate")
            _need(isinstance(raw.get("api_response_id"), str) and bool(raw["api_response_id"]),
                  f"{source} API response ID differs")
            _need(isinstance(source_id, str) and source_id in essay_hashes,
                  f"{source} candidate source ID differs")
            _need(type(number) is int and 1 <= number <= CANDIDATES_PER_ESSAY,
                  f"{source} candidate number differs")
            custom_parts = custom_id.split(":") if isinstance(custom_id, str) else []
            _need(len(custom_parts) == 4 and custom_parts[:3] == [run_id, "train", str(number)]
                  and bool(custom_parts[3]) and custom_id not in seen_custom,
                  f"{source} custom ID differs")
            _need(raw.get("essay_sha256") == essay_hashes[source_id],
                  f"{source} candidate essay SHA differs")
            try:
                participant = parse_participant_output(raw.get("participant_output"))
            except (TypeError, ValueError) as exc:
                raise OfficialDualAgentDataError(f"{source} participant schema differs") from exc
            seen_custom.add(custom_id)
            coverage.setdefault(source_id, set()).add(number)
            result.append(OfficialCandidate(
                agent_source=source, run_id=run_id, model=model, source_id=source_id,
                candidate_number=number, essay_sha256=essay_hashes[source_id],
                scores=MappingProxyType({axis: int(participant[axis]["score"]) for axis in AXES}),
                rationales=MappingProxyType({axis: str(participant[axis]["rationale"]) for axis in AXES}),
            ))
    _need(len(result) == EXPECTED_CANDIDATES and len(seen_custom) == EXPECTED_CANDIDATES,
          f"{source} candidate count differs from 6,000")
    _need(set(coverage) == set(essay_hashes), f"{source} essay population differs")
    _need(all(numbers == {1, 2, 3} for numbers in coverage.values()),
          f"{source} per-essay candidate coverage differs")
    provenance = {
        "agent_source": source, "run_id": run_id, "model": model,
        "source_sha256": manifest["source_sha256"], "candidates_sha256": candidate_sha,
        "essay_count": len(coverage), "candidate_count": len(result),
        "candidates_per_essay": CANDIDATES_PER_ESSAY,
    }
    return result, provenance


def load_dual_candidates(
    terra_manifest: str | Path,
    terra_candidates: str | Path,
    luna_manifest: str | Path,
    luna_candidates: str | Path,
    *,
    essay_sha256_by_source: Mapping[str, str],
) -> tuple[list[OfficialCandidate], Mapping[str, Any]]:
    """Load both fixed sources and return restricted rows plus aggregate provenance."""
    hashes = _essay_hashes(essay_sha256_by_source)
    paths = ((terra_manifest, terra_candidates), (luna_manifest, luna_candidates))
    rows: list[OfficialCandidate] = []
    provenance = []
    for (source, run_id, model), (manifest, candidates) in zip(SOURCE_SPECS, paths, strict=True):
        loaded, audit = _load_source(
            source=source, run_id=run_id, model=model, manifest_path=manifest,
            candidates_path=candidates, essay_hashes=hashes,
        )
        rows.extend(loaded)
        provenance.append(audit)
    _need(len(rows) == 2 * EXPECTED_CANDIDATES, "dual-source candidate population differs")
    aggregate = {
        "schema_version": "mal2026-iterative-official-dual-agent-data-v1",
        "source_count": 2, "essay_count": EXPECTED_ESSAYS,
        "candidate_count": len(rows), "candidates_per_essay_per_source": CANDIDATES_PER_ESSAY,
        "participant_axes": list(AXES), "sources": provenance,
        "row_content_in_provenance": False,
    }
    return rows, aggregate


__all__ = [
    "CANDIDATES_PER_ESSAY", "EXPECTED_CANDIDATES", "EXPECTED_ESSAYS", "OfficialCandidate",
    "OfficialDualAgentDataError", "SCHEMA_VERSION", "SOURCE_SHA256", "SOURCE_SPECS",
    "load_dual_candidates",
]
