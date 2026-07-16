"""Frozen protocol constants. Values here must not be changed post-result."""

from __future__ import annotations

SCORE_FIELDS = ("content", "organization", "expression", "average")
SCORE_MIN = 1.0
SCORE_MAX = 5.0
DEFAULT_SEED = 20260716
DEFAULT_DEV_FRACTION = 0.10
