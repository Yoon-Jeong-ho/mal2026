"""Frozen protocol constants. Values here must not be changed post-result."""

from __future__ import annotations

SCORE_FIELDS = ("content", "organization", "expression", "average")
EVIDENCE_CRITERIA = ("CONTENT", "ORGANIZATION", "EXPRESSION")
SCORE_MIN = 1.0
SCORE_MAX = 5.0
DEFAULT_SEED = 20260716
DEFAULT_DEV_FRACTION = 0.10

# Applied only to synthetic rationale prose (not to quoted essay spans). The
# teacher may never receive a score field, and its prose may not carry a score
# or a rating proxy. Keep this lexicon versioned and test any extension.
PROHIBITED_RATIONALE_PATTERNS = (
    r"[0-9]+(?:\.[0-9]+)?\s*점",
    r"(?<![\w.])[0-9]+(?:\.[0-9]+)?(?![\w.])",  # score-like numeric prose
    r"점수", r"등급", r"평점", r"채점", r"평가", r"배점",
    r"우수", r"탁월", r"보통", r"미흡", r"부족", r"훌륭",
    r"높(?:다|은|게|음)", r"낮(?:다|은|게|음)", r"잘\s*(?:썼|작성|했)",
    r"좋(?:다|은|게|음)", r"나쁘(?:다|ㄴ|게|음)",
)
