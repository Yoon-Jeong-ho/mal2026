"""Decoder-only contracts for Korean writing score prediction.

This module deliberately has no module-level ML imports so its target, parsing,
and leakage safeguards are testable without a training environment.  The
training runner imports its optional Hugging Face dependencies at execution
only.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCORE_KEYS: tuple[str, ...] = ("content", "organization", "expression", "average")
SCORE_MIN = Decimal("1.00")
SCORE_MAX = Decimal("5.00")
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# These values identify the user-supplied restricted evaluation files.  They
# are intentionally constants rather than mutable config defaults: a run must
# fail if a look-alike file or a changed fixed split is supplied.
CANONICAL_TRAIN_SHA256 = "b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737"
CANONICAL_VALIDATION_SHA256 = "0805445029328848164cf15f34b90b88fb5f7896d7b73f24f1717b733a9a00a4"

# These must match the frozen protocol rather than accepting loosely formatted
# generation.  Exact ordered output makes assistant-only supervision auditable.
DIRECT_KEYS = SCORE_KEYS
RATIONALE_CRITERIA = frozenset({"CONTENT", "ORGANIZATION", "EXPRESSION"})
# Excludes numeric offsets: generated evidence prose itself must not carry
# scores, ratings, or score-proxy wording. Quotes are source text and not scanned.
SCORE_CUE_RE = re.compile(
    r"(?:[0-9０-９]+(?:[.,][0-9０-９]+)?\s*(?:점|점수|등급|grade|score|rating))"
    r"|(?:만점|최고점|최저점|우수|탁월|보통|미흡|낮은\s*점수|높은\s*점수)"
    r"|(?:\b(?:excellent|good|poor|weak|strong)\b)",
    re.IGNORECASE,
)


class ContractError(ValueError):
    """Raised when an input, target, output, or execution config is invalid."""


def project_root() -> Path:
    """Return the repository root without consulting caller-controlled CWD."""
    return Path(__file__).resolve().parents[2]


def canonical_dataset_path(split: str) -> Path:
    if split not in {"train", "validation"}:
        raise ContractError("dataset split must be train or validation")
    return project_root() / "eval" / f"{split}.jsonl"


def require_canonical_dataset(path: str | Path, split: str, digest: str) -> Path:
    """Bind a decoder run to its one approved local restricted input file."""
    expected_digest = CANONICAL_TRAIN_SHA256 if split == "train" else CANONICAL_VALIDATION_SHA256
    if digest != expected_digest:
        raise ContractError(f"{split}_sha256 must equal the frozen canonical checksum")
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
        expected = canonical_dataset_path(split).resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"canonical eval/{split}.jsonl must exist before launch") from exc
    if resolved != expected:
        raise ContractError(f"{split} path must be the canonical eval/{split}.jsonl")
    return expected


def _path_has_symlink(path: Path) -> bool:
    """Reject symlinks instead of attempting to reason about their target."""
    current = Path(path.anchor) if path.is_absolute() else Path()
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            return True
    return False


def resolve_run_output_dir(run_id: str, output_dir: str | Path, *, must_exist: bool = False) -> Path:
    """Allow only an unsymlinked ``outputs/runs/<run-id>`` directory.

    The lexical equality check prevents a path such as ``../`` from being
    accepted, and the symlink checks prevent an ignored output location from
    silently escaping to a tracked or external location.
    """
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ContractError("run_id must be a safe nonempty immutable run identifier")
    root = project_root()
    expected = root / "outputs" / "runs" / run_id
    supplied = Path(output_dir)
    supplied = supplied if supplied.is_absolute() else root / supplied
    if supplied.absolute() != expected.absolute():
        raise ContractError("output_dir must be exactly repository outputs/runs/<run-id>")
    for path in (root / "outputs", root / "outputs" / "runs", expected):
        if _path_has_symlink(path):
            raise ContractError("output path contains a symlink and is rejected")
    # resolve(strict=False) also detects a symlink introduced between checks.
    if supplied.resolve(strict=False) != expected.resolve(strict=False):
        raise ContractError("output path resolution escaped the approved run directory")
    if must_exist and not expected.is_dir():
        raise ContractError("required prior run output directory does not exist")
    return expected


def require_path_under_run(path: str | Path, run_id: str) -> Path:
    """Accept an existing non-symlinked artifact only from the named run."""
    run_dir = resolve_run_output_dir(run_id, project_root() / "outputs" / "runs" / run_id, must_exist=True)
    candidate = Path(path)
    if not candidate.exists() or _path_has_symlink(candidate):
        raise ContractError("referenced run artifact must exist and may not use symlinks")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(run_dir.resolve(strict=True))
    except ValueError as exc:
        raise ContractError("referenced artifact must be inside its named refit run") from exc
    return resolved


class _JsonNumber(str):
    """Preserve the JSON numeric lexeme so strict two-decimal parsing is possible."""


@dataclass(frozen=True)
class ParsedPrediction:
    """Strict decoder parse result; fallback is intentionally explicit."""

    scores: dict[str, float]
    valid: bool
    error: str | None = None
    rationale_valid: bool | None = None


def require_immutable_revision(revision: str, field: str = "revision") -> str:
    """Fail closed on branches/tags; reproducible runs require a commit SHA."""
    if not isinstance(revision, str) or not IMMUTABLE_REVISION_RE.fullmatch(revision):
        raise ContractError(f"{field} must be a 40-character lowercase git commit SHA")
    return revision


def format_score(value: float | Decimal) -> str:
    """Canonical two-decimal half-up representation for a source score."""
    try:
        number = Decimal(str(value))
    except Exception as exc:  # pragma: no cover - Decimal has detailed errors
        raise ContractError(f"score is not decimal: {value!r}") from exc
    if not number.is_finite() or not (SCORE_MIN <= number <= SCORE_MAX):
        raise ContractError(f"score must be finite and in [1, 5], got {value!r}")
    rounded = number.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded:.2f}"


def _ordered_score_object(score: Mapping[str, Any]) -> dict[str, str]:
    if set(score) != set(SCORE_KEYS):
        raise ContractError(f"score keys must be exactly {SCORE_KEYS}, got {tuple(score)}")
    return {key: format_score(score[key]) for key in SCORE_KEYS}


def _score_json(score: Mapping[str, Any]) -> str:
    ordered = _ordered_score_object(score)
    return "{" + ",".join(f'"{key}":{ordered[key]}' for key in SCORE_KEYS) + "}"


def direct_target(score: Mapping[str, Any]) -> str:
    """Create exact, ordered direct-SFT JSON (no prose and no markdown)."""
    return _score_json(score)


def validate_rationale(rationale: Any, essay: str) -> list[dict[str, Any]]:
    """Delegate evidence/leakage validation to the shared rationale contract.

    The decoder stores only the ``rationale`` list inside its SFT output, while
    the shared validator owns the versioned criterion, prose-cue, and offset
    policy. Quotes are source evidence and are never score-cue scanned.
    """
    if not isinstance(rationale, list) or not isinstance(essay, str):
        raise ContractError("rationale must be a list and essay must be a string")
    from .rationale import RationaleValidationError, validate_rationale_payload

    try:
        validate_rationale_payload({"rationale": rationale}, essay=essay)
    except RationaleValidationError as exc:
        raise ContractError(str(exc)) from exc
    # JSON-normalize only after the shared validator has checked every field.
    return json.loads(json.dumps(rationale, ensure_ascii=False))


def rationale_target(rationale: Any, score: Mapping[str, Any], essay: str) -> str:
    """Create exact rationale-then-score target after structural validation."""
    checked = validate_rationale(rationale, essay)
    # ``json.dumps`` cannot preserve two decimal places for floats, so score JSON
    # is rendered separately while the synthetic evidence remains ordinary JSON.
    rationale_json = json.dumps(checked, ensure_ascii=False, separators=(",", ":"))
    return '{"rationale":' + rationale_json + ',"scores":' + _score_json(score) + "}"


def _strict_score_object(obj: Any) -> dict[str, float]:
    if not isinstance(obj, dict) or tuple(obj) != SCORE_KEYS:
        raise ContractError("score JSON must have exactly ordered keys content, organization, expression, average")
    result: dict[str, float] = {}
    for key in SCORE_KEYS:
        value = obj[key]
        # Numeric lexemes are retained by json.loads. Strings, bools, exponent
        # notation, integers, and any precision other than exactly two decimals
        # are protocol violations, not candidates for numeric extraction.
        if not isinstance(value, _JsonNumber) or not re.fullmatch(r"[1-5]\.\d{2}", str(value)):
            raise ContractError(f"{key} must be a finite two-decimal JSON number in [1, 5]")
        decimal = Decimal(value)
        if not (SCORE_MIN <= decimal <= SCORE_MAX):
            raise ContractError(f"{key} is out of range")
        result[key] = float(decimal)
    return result


def parse_decoder_output(text: str, mode: str, fallback_mean: Mapping[str, float], essay: str | None = None) -> ParsedPrediction:
    """Parse only the protocol JSON; any deviation returns the fixed train mean.

    There is intentionally no prose stripping, partial parsing, or number
    extraction fallback. ``fallback_mean`` must originate from the optimization
    training partition and is supplied by the caller/manifest.
    """
    fallback = {key: float(fallback_mean[key]) for key in SCORE_KEYS}
    try:
        if mode not in {"direct", "rationale"}:
            raise ContractError("mode must be direct or rationale")
        if not isinstance(text, str):
            raise ContractError("decoder output must be text")
        parsed = json.loads(text, parse_float=_JsonNumber)
        if mode == "direct":
            return ParsedPrediction(scores=_strict_score_object(parsed), valid=True, rationale_valid=None)
        if not isinstance(parsed, dict) or tuple(parsed) != ("rationale", "scores"):
            raise ContractError("rationale output must contain exactly ordered rationale and scores")
        scores = _strict_score_object(parsed["scores"])
        if essay is None:
            raise ContractError("essay is required to validate rationale output")
        validate_rationale(parsed["rationale"], essay)
        return ParsedPrediction(scores=scores, valid=True, rationale_valid=bool(parsed["rationale"]))
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        return ParsedPrediction(scores=fallback, valid=False, error=str(exc), rationale_valid=False if mode == "rationale" else None)



def target_for_record(record: Mapping[str, Any], mode: str, rationale: Any | None = None) -> str:
    """Validate only required record fields and build a decoder assistant target."""
    score = record.get("score")
    if not isinstance(score, Mapping):
        raise ContractError("record.score must be an object")
    if mode == "direct":
        return direct_target(score)
    if mode != "rationale":
        raise ContractError("mode must be direct or rationale")
    essay = record.get("essay")
    if not isinstance(essay, str):
        raise ContractError("record.essay must be a string")
    if rationale is None:
        raise ContractError("rationale mode requires a train-only synthetic rationale")
    return rationale_target(rationale, score, essay)


def prompt_text(record: Mapping[str, Any]) -> str:
    """Build the frozen user message without leaking labels, ids, or split data."""
    prompt, essay = record.get("prompt"), record.get("essay")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(essay, str) or not essay.strip():
        raise ContractError("record must include nonempty prompt and essay strings")
    return (
        "다음 글을 채점하세요. 출력은 지시된 JSON 형식만 사용하세요.\n\n"
        f"[문제]\n{prompt}\n\n[학생 글]\n{essay}"
    )


def template_sha256(system_message: str, user_instruction: str) -> str:
    return hashlib.sha256((system_message + "\n" + user_instruction).encode("utf-8")).hexdigest()


def validate_lora_targets(module_names: Iterable[str], target_modules: Sequence[str]) -> tuple[str, ...]:
    """Require every configured LoRA target to exist before adapting a model."""
    targets = tuple(target_modules)
    required = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    if targets != required:
        raise ContractError(f"decoder LoRA targets must exactly be {required}")
    names = tuple(module_names)
    missing = [target for target in targets if not any(name.endswith(f".{target}") or name == target for name in names)]
    if missing:
        raise ContractError(f"configured LoRA target modules missing from model: {missing}")
    return targets


def assert_finite_loss(loss: Any) -> None:
    """Small dependency-light guard used by smoke/full training loops."""
    value = float(loss.detach().float().item()) if hasattr(loss, "detach") else float(loss)
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite training loss: {value}")
