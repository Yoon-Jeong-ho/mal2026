"""Pure protocol helpers for Solar generate/filter/consensus experiments."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

from .solar_target_augmentation import AXES, SourceRow


JUDGE_DRAWS_INITIAL = 3
JUDGE_DRAWS_MAXIMUM = 5


class SolarConsensusPilotError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise SolarConsensusPilotError(message)


def rounded_score(value: float) -> int:
    """Project only for source sampling strata, never for a training label."""
    return min(5, max(1, int(math.floor(float(value) + 0.5))))


def length_bin(length: int, ordered_lengths: Sequence[int]) -> int:
    need(length > 0 and bool(ordered_lengths), "source length stratum differs")
    return min(3, sum(length > ordered_lengths[int(q * (len(ordered_lengths) - 1))]
                      for q in (0.25, 0.5, 0.75)))


def stratified_sources(
    rows: Sequence[SourceRow], count: int, seed: int | str,
) -> list[SourceRow]:
    """Greedily cover axis-score and length strata with deterministic ties."""
    need(count > 0 and len(rows) >= count, "stratified source count differs")
    need(len({row.identifier for row in rows}) == len(rows), "source IDs are not unique")
    lengths = sorted(len(row.essay.strip()) for row in rows)
    features: dict[str, tuple[str, ...]] = {}
    ranks: dict[str, bytes] = {}
    for row in rows:
        values = tuple(
            f"axis-score:{axis}:{rounded_score(row.score[index])}"
            for index, axis in enumerate(AXES)
        ) + (f"length:{length_bin(len(row.essay.strip()), lengths)}",)
        features[row.identifier] = values
        ranks[row.identifier] = sha256(
            f"{seed}\0solar-consensus-source\0{row.identifier}".encode()
        ).digest()
    coverage: Counter[str] = Counter()
    remaining = {row.identifier: row for row in rows}
    selected: list[SourceRow] = []
    while len(selected) < count:
        winner = min(
            remaining.values(),
            key=lambda row: (
                -sum(1.0 / (1.0 + coverage[item]) for item in features[row.identifier]),
                ranks[row.identifier],
            ),
        )
        selected.append(winner)
        remaining.pop(winner.identifier)
        coverage.update(features[winner.identifier])
    return selected


def stratified_fold_assignments(
    rows: Sequence[Any], folds: int, seed: int | str,
) -> dict[str, int]:
    """Assign source rows to balanced, label-stratified document-safe folds."""
    need(folds >= 2 and len(rows) >= folds, "OOF fold population differs")
    need(len({row.identifier for row in rows}) == len(rows), "OOF source IDs differ")
    need(len({row.document_id for row in rows}) == len(rows),
         "OOF requires one canonical row per document")
    strata: dict[tuple[str, int, int, int], list[Any]] = {}
    for row in rows:
        labels = row.labels if hasattr(row, "labels") else row.score
        prompt_group = str(getattr(row, "prompt_num", "unknown"))
        key = (prompt_group, *(rounded_score(labels[index]) for index in range(3)))
        strata.setdefault(key, []).append(row)
    assignments: dict[str, int] = {}
    fold_sizes = [0] * folds
    for stratum in sorted(strata):
        ordered = sorted(
            strata[stratum],
            key=lambda row: sha256(
                f"{seed}\0solar-consensus-oof\0{row.document_id}".encode()
            ).digest(),
        )
        rotation = int.from_bytes(
            sha256(f"{seed}\0{stratum}".encode()).digest()[:4], "big"
        ) % folds
        for row in ordered:
            minimum = min(fold_sizes)
            candidates = [index for index, size in enumerate(fold_sizes) if size == minimum]
            fold = min(candidates, key=lambda index: (index - rotation) % folds)
            assignments[row.identifier] = fold
            fold_sizes[fold] += 1
    need(len(assignments) == len(rows) and max(fold_sizes) - min(fold_sizes) <= 1,
         "OOF fold balance differs")
    return assignments


def visible_draw_seed(
    messages: Sequence[Mapping[str, str]], draw_index: int, base_seed: int,
) -> int:
    """Vary judge draws using only target-blind visible request content."""
    need(0 <= draw_index < JUDGE_DRAWS_MAXIMUM, "judge draw index differs")
    visible = json.dumps(list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = sha256(f"{base_seed}\0solar-consensus-judge\0{draw_index}\0{visible}".encode()).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % 2_147_483_647


def triplet(score: Mapping[str, int]) -> tuple[int, int, int]:
    values = tuple(int(score[axis]) for axis in AXES)
    need(all(1 <= value <= 5 for value in values), "judge triplet differs")
    return values  # type: ignore[return-value]


def requires_two_more_draws(draws: Sequence[Mapping[str, int]]) -> bool:
    need(len(draws) == JUDGE_DRAWS_INITIAL, "initial judge draw population differs")
    return len({triplet(draw) for draw in draws}) != 1


def modal_label(
    draws: Sequence[Mapping[str, int]],
) -> tuple[dict[str, int] | None, int, dict[str, int]]:
    """Return a unique triplet supported by >=3 of exactly 3 or 5 draws."""
    need(len(draws) in {JUDGE_DRAWS_INITIAL, JUDGE_DRAWS_MAXIMUM},
         "judge draw population differs")
    counts = Counter(triplet(draw) for draw in draws)
    value, support = counts.most_common(1)[0]
    label = (
        {axis: int(value[index]) for index, axis in enumerate(AXES)}
        if support >= 3 else None
    )
    distribution = {
        "/".join(str(item) for item in key): int(count)
        for key, count in sorted(counts.items())
    }
    return label, int(support), distribution


def upper_quantile(values: Sequence[float], quantile: float) -> float:
    need(bool(values) and 0.0 <= quantile <= 1.0 and
         all(math.isfinite(float(value)) for value in values),
         "calibration quantile input differs")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def calibrated_quarter_threshold(
    values: Sequence[float], quantile: float = 0.8, minimum: float = 0.5,
) -> float:
    """Round an OOF-only error quantile upward to a 0.25 score boundary."""
    need(minimum >= 0.0, "calibration minimum differs")
    raw = max(float(minimum), upper_quantile(values, quantile))
    return math.ceil(raw * 4.0 - 1e-12) / 4.0
