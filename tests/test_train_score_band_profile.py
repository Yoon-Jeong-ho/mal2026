"""Synthetic-only tests for the aggregate train score-band profiler."""
from decimal import Decimal
import math

from mal2026 import train_score_band_profile as profile


def test_score_band_uses_round_half_up() -> None:
    assert profile.score_band(Decimal("1.49")) == 1
    assert profile.score_band(Decimal("1.50")) == 2
    assert profile.score_band(2.5) == 3
    assert profile.score_band("4.5") == 5


def test_surface_features_are_limited_counts() -> None:
    features = profile.surface_features("먼저 근거가 있다. 따라서 결과가 생긴다!\n하지만 반면도 있다")
    assert features["sentence_count"] == 3
    assert features["serialized_line_count"] == 2
    assert features["evidence_proxy_count"] == 1
    assert features["causal_proxy_count"] == 2
    assert features["enumeration_proxy_count"] == 1
    assert features["transition_proxy_count"] == 3


def test_spearman_is_tie_aware_and_handles_constant_feature() -> None:
    assert math.isclose(profile.spearman_correlation([1, 1, 2, 3], [1, 2, 3, 4]), 0.948683298, rel_tol=1e-9)
    assert profile.spearman_correlation([1, 1], [2, 3]) is None


def test_profile_is_aggregate_only() -> None:
    original_expected = profile.EXPECTED_RECORDS
    profile.EXPECTED_RECORDS = 2
    records = [
        {"id": "forbidden-id-a", "essay": "먼저 근거다.", "score": {"content": 1.5, "organization": 2.5, "expression": 3.5}},
        {"id": "forbidden-id-b", "essay": "하지만 결과다.", "score": {"content": 4.5, "organization": 3.5, "expression": 2.5}},
    ]
    try:
        result = profile.build_profile(records, source_sha256="0" * 64)
    finally:
        profile.EXPECTED_RECORDS = original_expected
    serialized = __import__("json").dumps(result, ensure_ascii=False)
    assert "forbidden-id" not in serialized
    assert "먼저 근거다" not in serialized
    assert result["privacy"] == {
        "aggregate_only": True,
        "source_text_included": False,
        "source_identifiers_included": False,
        "individual_examples_included": False,
    }
    assert result["axes"]["content"]["band_counts"] == {"1": 0, "2": 1, "3": 0, "4": 0, "5": 1}
    assert result["source"]["validation_records_read"] == 0
