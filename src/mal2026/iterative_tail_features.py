"""Score-blind rationale features for the iterative tail experiment.

The inputs are three pre-existing train-only rationale generations whose
manifests explicitly state that neither a human nor reference score was read
or prompted.  This module never reads validation and never emits rationale
text.  It converts each rationale to deterministic numeric features in memory;
row-level arrays remain restricted experiment material.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np


AXES = ("content", "organization", "expression")
STRUCTURED_DIM = 18
HASH_DIM = 96
RATIONALE_SOURCES: tuple[tuple[str, str, str], ...] = (
    (
        "evaluation_prompt_v1_agent_a",
        "data/processed/restricted/evaluation_prompt_rationale_v1/"
        "evaluation-prompt-rationale-generation-v1-score-blind-20260729-001/"
        "rationales.train.jsonl",
        "1a10524f79823e097e6f56f8c7ac3a499baf2d79b81cbb5b69184ddc88610223",
    ),
    (
        "evaluation_prompt_v1_agent_b",
        "data/processed/restricted/evaluation_prompt_rationale_v1/"
        "evaluation-prompt-rationale-generation-v1-score-blind-20260729-002/"
        "rationales.train.jsonl",
        "f5cce4058f56ea7dedc7de07bfb20b343345676054dd29bf541adeed3d594e3c",
    ),
    (
        "evaluation_prompt_v2_agent_c",
        "data/processed/restricted/evaluation_prompt_rationale_v2/"
        "evaluation-prompt-rationale-generation-v2-score-blind-20260729-004/"
        "rationales.train.jsonl",
        "d4a2be9a070c786728fde6f64f066ac9d462bc5f83305a2d9161b380abd88e55",
    ),
    (
        "official_dpo_score_blind_agent_d",
        "data/processed/restricted/official_prompt_alignment_v1/final_rationale_handoff/"
        "official-rationale-dpo-selected-handoff-exact-bundle-20260729-021/"
        "rationales.train.jsonl",
        "45dc9bfd05d60c75214221e34149ed7bff6dae0d571a90fde287ab193bb6f347",
    ),
)


class ScoreBlindFeatureError(ValueError):
    """Raised when a rationale artifact or requested feature view drifts."""


@dataclass(frozen=True)
class ScoreBlindFeatureBundle:
    source_ids: tuple[str, ...]
    structured: np.ndarray  # [rows, agents, axes, STRUCTURED_DIM]
    hashed: np.ndarray  # [rows, agents, axes, HASH_DIM]
    source_bindings: tuple[Mapping[str, str], ...]

    def view(self, name: str) -> np.ndarray | None:
        """Return a predeclared score-blind feature view.

        Views are deliberately fixed before observing experiment results.
        ``none`` returns no extra features.  No view includes a human score or
        the model's scoring error.
        """
        if name == "none":
            return None
        structured = self.structured
        hashed = self.hashed
        if name == "content_structured":
            # Round 14 is bound to the v2 rationale artifact only.
            result = structured[:, 2, 0, :]
        elif name == "org_expression_structured":
            # Round 15 is likewise the v2 artifact's other two axes.
            result = structured[:, 2, 1:, :].reshape(len(self.source_ids), -1)
        elif name == "consensus_disagreement":
            means = structured.mean(axis=1)
            stds = structured.std(axis=1)
            ranges = structured.max(axis=1) - structured.min(axis=1)
            similarities = _agent_cosine_summaries(hashed)
            result = np.concatenate(
                (means.reshape(len(self.source_ids), -1),
                 stds.reshape(len(self.source_ids), -1),
                 ranges.reshape(len(self.source_ids), -1), similarities), axis=1,
            )
        elif name == "evidence_hash":
            means = hashed.mean(axis=1)
            stds = hashed.std(axis=1)
            result = np.concatenate(
                (means.reshape(len(self.source_ids), -1), stds.reshape(len(self.source_ids), -1)), axis=1
            )
        elif name == "full_fusion":
            consensus = self.view("consensus_disagreement")
            evidence = self.view("evidence_hash")
            assert consensus is not None and evidence is not None
            result = np.concatenate((structured.reshape(len(self.source_ids), -1), consensus, evidence), axis=1)
        else:
            raise ScoreBlindFeatureError(f"unknown score-blind feature view: {name}")
        result = np.asarray(result, dtype=np.float32)
        if result.ndim != 2 or len(result) != len(self.source_ids) or not np.isfinite(result).all():
            raise ScoreBlindFeatureError("score-blind feature matrix is invalid")
        return result


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_WORD_GROUPS: tuple[tuple[str, ...], ...] = (
    ("명확", "분명", "구체", "적절", "타당", "충분"),
    ("부족", "미흡", "모호", "불분명", "제한", "단순"),
    ("근거", "사례", "예시", "통계", "인용", "자료"),
    ("논리", "주장", "논증", "이유", "설득", "내용"),
    ("구조", "서론", "본론", "결론", "문단", "연결"),
    ("표현", "문장", "어휘", "맞춤법", "문법", "반복"),
    ("개선", "보완", "필요", "강화", "수정", "다듬"),
    ("첫째", "둘째", "셋째", "우선", "다음", "마지막"),
)


def _structured_features(text: str) -> np.ndarray:
    length = max(1, len(text))
    tokens = re.findall(r"\S+", text)
    sentences = [part for part in re.split(r"[.!?。！？]+", text) if part.strip()]
    base = [
        math.log1p(length) / 8.0,
        math.log1p(len(tokens)) / 6.0,
        math.log1p(len(sentences)) / 4.0,
        text.count(",") / length * 100.0,
        text.count(";") / length * 100.0,
        sum(char.isdigit() for char in text) / length * 100.0,
        sum(char in "\"'‘’“”" for char in text) / length * 100.0,
        len(set(tokens)) / max(1, len(tokens)),
        text.count("하지만") + text.count("그러나") + text.count("반면"),
        text.count("때문") + text.count("따라서") + text.count("그러므로"),
    ]
    # Eight rubric/evidence lexicons complete the fixed 18-dimensional view.
    base.extend(sum(text.count(word) for word in group) / max(1, len(tokens)) for group in _WORD_GROUPS)
    result = np.asarray(base, dtype=np.float32)
    if result.shape != (STRUCTURED_DIM,) or not np.isfinite(result).all():
        raise ScoreBlindFeatureError("structured rationale features are invalid")
    return result


def _hashed_features(text: str) -> np.ndarray:
    """Signed character 2--4 gram hashing without a learned vocabulary."""
    normalized = re.sub(r"\s+", " ", text.strip())
    result = np.zeros(HASH_DIM, dtype=np.float32)
    for width in (2, 3, 4):
        for offset in range(max(0, len(normalized) - width + 1)):
            gram = normalized[offset : offset + width]
            digest = sha256(gram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % HASH_DIM
            sign = 1.0 if digest[4] & 1 else -1.0
            result[bucket] += sign
    norm = float(np.linalg.norm(result))
    if norm > 0:
        result /= norm
    return result


def _agent_cosine_summaries(hashed: np.ndarray) -> np.ndarray:
    # Input [rows, agents, axes, dim].  Vectors are already L2-normalized.
    pairs = ((0, 1), (0, 2), (1, 2))
    scores = [np.sum(hashed[:, left] * hashed[:, right], axis=-1) for left, right in pairs]
    stacked = np.stack(scores, axis=1)  # [rows, pairs, axes]
    return np.concatenate(
        (stacked.reshape(len(hashed), -1), stacked.mean(axis=1), stacked.std(axis=1)), axis=1
    ).astype(np.float32)


def _read_source(path: Path, expected_sha256: str) -> dict[str, dict[str, str]]:
    if not path.is_file() or path.is_symlink() or _file_sha256(path) != expected_sha256:
        raise ScoreBlindFeatureError(f"score-blind rationale binding differs: {path}")
    records: dict[str, dict[str, str]] = {}
    required = {"source_id", "rationales"}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict) or set(raw) != required:
                raise ScoreBlindFeatureError(f"rationale row schema differs at line {line_number}")
            source_id, rationales = raw["source_id"], raw["rationales"]
            if not isinstance(source_id, str) or source_id in records:
                raise ScoreBlindFeatureError("duplicate or invalid rationale source ID")
            if not isinstance(rationales, dict) or set(rationales) != set(AXES):
                raise ScoreBlindFeatureError("rationale axes differ")
            if not all(isinstance(rationales[axis], str) and rationales[axis].strip() for axis in AXES):
                raise ScoreBlindFeatureError("rationale text must be nonempty")
            records[source_id] = {axis: rationales[axis] for axis in AXES}
    if len(records) != 2000:
        raise ScoreBlindFeatureError("each score-blind rationale source must contain 2,000 train rows")
    return records


def load_score_blind_features(
    source_ids: Sequence[str], *, root: str | Path = ".",
) -> ScoreBlindFeatureBundle:
    """Load only score-blind train rationales and align them to frozen rows."""
    identifiers = tuple(source_ids)
    if len(identifiers) != 2000 or len(set(identifiers)) != 2000:
        raise ScoreBlindFeatureError("feature alignment requires exactly 2,000 unique train IDs")
    root_path = Path(root)
    sources = []
    bindings = []
    for name, relative, expected_sha in RATIONALE_SOURCES:
        path = root_path / relative
        records = _read_source(path, expected_sha)
        if set(records) != set(identifiers):
            raise ScoreBlindFeatureError("rationale/train ID populations differ")
        sources.append(records)
        bindings.append({"name": name, "path": relative, "sha256": expected_sha, "score_conditioning": "false"})
    structured = np.empty((len(identifiers), len(sources), len(AXES), STRUCTURED_DIM), dtype=np.float32)
    hashed = np.empty((len(identifiers), len(sources), len(AXES), HASH_DIM), dtype=np.float32)
    for row_index, source_id in enumerate(identifiers):
        for agent_index, records in enumerate(sources):
            for axis_index, axis in enumerate(AXES):
                text = records[source_id][axis]
                structured[row_index, agent_index, axis_index] = _structured_features(text)
                hashed[row_index, agent_index, axis_index] = _hashed_features(text)
    return ScoreBlindFeatureBundle(identifiers, structured, hashed, tuple(bindings))


def write_score_blind_feature_cache(
    bundle: ScoreBlindFeatureBundle, cache_path: str | Path, manifest_path: str | Path,
) -> Mapping[str, object]:
    """Write a restricted numeric cache and an aggregate-only binding manifest."""
    cache = Path(cache_path)
    manifest = Path(manifest_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        source_ids=np.asarray(bundle.source_ids, dtype=str),
        structured=bundle.structured,
        hashed=bundle.hashed,
    )
    payload: dict[str, object] = {
        "schema_version": "mal2026-score-blind-feature-cache-v1",
        "records": len(bundle.source_ids),
        "agents": int(bundle.structured.shape[1]),
        "axes": list(AXES),
        "structured_dim": STRUCTURED_DIM,
        "hash_dim": HASH_DIM,
        "cache_sha256": _file_sha256(cache),
        "score_conditioning": False,
        "validation_loaded": False,
        "source_bindings": list(bundle.source_bindings),
    }
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_score_blind_feature_cache(
    source_ids: Sequence[str], cache_path: str | Path, manifest_path: str | Path,
) -> ScoreBlindFeatureBundle:
    """Load a bound restricted cache and require exact row-ID order."""
    cache = Path(cache_path)
    manifest_file = Path(manifest_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreBlindFeatureError("score-blind cache manifest is unreadable") from exc
    if (
        manifest.get("schema_version") != "mal2026-score-blind-feature-cache-v1"
        or manifest.get("records") != 2000
        or manifest.get("agents") != len(RATIONALE_SOURCES)
        or manifest.get("axes") != list(AXES)
        or manifest.get("structured_dim") != STRUCTURED_DIM
        or manifest.get("hash_dim") != HASH_DIM
        or manifest.get("score_conditioning") is not False
        or manifest.get("validation_loaded") is not False
        or manifest.get("cache_sha256") != _file_sha256(cache)
    ):
        raise ScoreBlindFeatureError("score-blind cache binding differs")
    with np.load(cache, allow_pickle=False) as payload:
        cached_ids = tuple(str(value) for value in payload["source_ids"].tolist())
        structured = np.asarray(payload["structured"], dtype=np.float32)
        hashed = np.asarray(payload["hashed"], dtype=np.float32)
    if cached_ids != tuple(source_ids):
        raise ScoreBlindFeatureError("score-blind cache row alignment differs")
    expected_structured = (2000, len(RATIONALE_SOURCES), len(AXES), STRUCTURED_DIM)
    expected_hashed = (2000, len(RATIONALE_SOURCES), len(AXES), HASH_DIM)
    if structured.shape != expected_structured or hashed.shape != expected_hashed:
        raise ScoreBlindFeatureError("score-blind cache shape differs")
    if not np.isfinite(structured).all() or not np.isfinite(hashed).all():
        raise ScoreBlindFeatureError("score-blind cache contains non-finite values")
    bindings = tuple(manifest.get("source_bindings", ()))
    return ScoreBlindFeatureBundle(cached_ids, structured, hashed, bindings)


__all__ = [
    "AXES", "HASH_DIM", "RATIONALE_SOURCES", "STRUCTURED_DIM",
    "ScoreBlindFeatureBundle", "ScoreBlindFeatureError", "load_score_blind_features",
    "load_score_blind_feature_cache", "write_score_blind_feature_cache",
]
