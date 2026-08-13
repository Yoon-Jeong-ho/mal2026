"""Privacy-preserving aggregate surface profiles for the canonical train split."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AXES = ("content", "organization", "expression")
EXPECTED_RECORDS = 2_000
SENTENCE_BOUNDARY = re.compile(r"[.!?。！？]+")
PROXY_MARKERS: dict[str, tuple[str, ...]] = {
    "evidence_proxy_count": ("예를 들어", "예컨대", "실제로", "통계", "자료", "연구", "사례", "근거"),
    "causal_proxy_count": ("때문", "따라서", "그러므로", "그러니", "결과적으로", "원인", "결과"),
    "enumeration_proxy_count": (
        "첫째", "둘째", "셋째", "첫 번째", "두 번째", "세 번째", "먼저", "다음으로", "마지막으로"
    ),
    "transition_proxy_count": (
        "그러나", "하지만", "반면", "한편", "또한", "더불어", "따라서", "그러므로", "결론적으로", "요컨대"
    ),
}
FEATURES = (
    "character_count",
    "non_whitespace_character_count",
    "sentence_count",
    "mean_non_whitespace_characters_per_sentence",
    "serialized_line_count",
    *PROXY_MARKERS,
)


def score_band(raw_score: int | float | str | Decimal) -> int:
    """Map a 1--5 raw score to an integer band using decimal ROUND_HALF_UP."""
    value = Decimal(str(raw_score))
    if not Decimal("1") <= value <= Decimal("5"):
        raise ValueError(f"score outside [1, 5]: {value}")
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _count_markers(text: str, markers: Iterable[str]) -> int:
    return sum(text.count(marker) for marker in markers)


def surface_features(text: str) -> dict[str, float]:
    """Extract deliberately limited, interpretable surface features from one essay."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("essay must be a non-empty string")
    non_whitespace = sum(not char.isspace() for char in text)
    sentence_parts = [part for part in SENTENCE_BOUNDARY.split(text) if part.strip()]
    sentence_count = max(1, len(sentence_parts))
    serialized_lines = [line for line in text.splitlines() if line.strip()]
    features: dict[str, float] = {
        "character_count": float(len(text)),
        "non_whitespace_character_count": float(non_whitespace),
        "sentence_count": float(sentence_count),
        "mean_non_whitespace_characters_per_sentence": non_whitespace / sentence_count,
        "serialized_line_count": float(max(1, len(serialized_lines))),
    }
    features.update({name: float(_count_markers(text, markers)) for name, markers in PROXY_MARKERS.items()})
    return features


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return tie-aware Spearman rho, or None when either side is constant."""
    if len(left) != len(right) or not left:
        raise ValueError("paired non-empty sequences are required")
    x_rank, y_rank = _average_ranks(left), _average_ranks(right)
    x_mean, y_mean = statistics.fmean(x_rank), statistics.fmean(y_rank)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_rank, y_rank, strict=True))
    x_ss = sum((x - x_mean) ** 2 for x in x_rank)
    y_ss = sum((y - y_mean) ** 2 for y in y_rank)
    if x_ss == 0 or y_ss == 0:
        return None
    return numerator / math.sqrt(x_ss * y_ss)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_profile(records: Sequence[Mapping[str, Any]], *, source_sha256: str) -> dict[str, Any]:
    """Build an aggregate-only profile; no source text or identifier is retained."""
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"canonical train must contain exactly {EXPECTED_RECORDS} records")
    extracted: list[dict[str, float]] = []
    raw_scores = {axis: [] for axis in AXES}
    bands = {axis: [] for axis in AXES}
    for row in records:
        if set(AXES) - set(row.get("score", {})):
            raise ValueError("record is missing an axis score")
        extracted.append(surface_features(row.get("essay")))
        for axis in AXES:
            score = float(row["score"][axis])
            raw_scores[axis].append(score)
            bands[axis].append(score_band(score))

    axes: dict[str, Any] = {}
    for axis in AXES:
        counts = Counter(bands[axis])
        band_profiles: dict[str, Any] = {}
        for band in range(1, 6):
            indices = [index for index, value in enumerate(bands[axis]) if value == band]
            band_profiles[str(band)] = {
                "count": len(indices),
                "feature_means": {
                    feature: (round(statistics.fmean(extracted[index][feature] for index in indices), 6) if indices else None)
                    for feature in FEATURES
                },
            }
        axes[axis] = {
            "band_counts": {str(band): counts[band] for band in range(1, 6)},
            "bands": band_profiles,
            "feature_raw_score_spearman": {
                feature: _rounded_or_none(spearman_correlation([row[feature] for row in extracted], raw_scores[axis]))
                for feature in FEATURES
            },
        }

    single_line_count = sum(row["serialized_line_count"] == 1 for row in extracted)
    return {
        "schema_version": "mal2026-train-score-band-profile-v1",
        "privacy": {
            "aggregate_only": True,
            "source_text_included": False,
            "source_identifiers_included": False,
            "individual_examples_included": False,
        },
        "source": {
            "split": "train",
            "record_count": len(records),
            "sha256": source_sha256,
            "validation_records_read": 0,
        },
        "banding": {"method": "Decimal(str(score)).quantize(1, ROUND_HALF_UP)", "bands": [1, 2, 3, 4, 5]},
        "feature_definitions": {
            "character_count": "Python len(text), including whitespace",
            "non_whitespace_character_count": "characters for which str.isspace() is false",
            "sentence_count": "non-empty segments split on . ! ? and CJK equivalents; minimum 1",
            "mean_non_whitespace_characters_per_sentence": "non-whitespace characters divided by sentence_count",
            "serialized_line_count": "non-empty physical lines from str.splitlines(); minimum 1",
            **{name: {"operation": "sum of literal substring occurrence counts", "markers": list(markers)} for name, markers in PROXY_MARKERS.items()},
        },
        "overall_feature_means": {
            feature: round(statistics.fmean(row[feature] for row in extracted), 6) for feature in FEATURES
        },
        "serialization_diagnostic": {
            "single_serialized_line_count": single_line_count,
            "single_serialized_line_rate": round(single_line_count / len(records), 6),
        },
        "axes": axes,
        "limitations": [
            "Proxy counts use short fixed marker lists and cannot establish whether evidence, causality, enumeration, or transitions are valid or effective.",
            "Rare score bands yield unstable aggregate means; counts must be considered with every comparison.",
            "Physical line breaks may be lost or normalized during JSONL serialization, and line count does not measure discourse organization.",
            "Surface-feature correlations are descriptive associations with raw scores, not scoring rules or causal effects.",
        ],
    }


def _rounded_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 6)
