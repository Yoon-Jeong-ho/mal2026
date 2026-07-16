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
from copy import deepcopy
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

# These values make assistant targets and generation parsing fully deterministic.
# `human_feedback` is supervised only from AI-Hub labels; it is never part of a
# model input and is unavailable in frozen final evaluation.
DIRECT_KEYS = SCORE_KEYS
HUMAN_FEEDBACK_KEYS: tuple[str, ...] = (
    "holistic", "content_1", "content_2", "content_3", "organization_1",
    "organization_2", "expression_1", "expression_2", "task_1",
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
    """Bind final decoder evaluation to its approved restricted input file."""
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
    """Allow only an unsymlinked ``outputs/runs/<run-id>`` directory."""
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
    feedback_valid: bool | None = None


def require_immutable_revision(revision: str, field: str = "revision") -> str:
    """Fail closed on branches/tags; reproducible runs require a commit SHA."""
    if not isinstance(revision, str) or not IMMUTABLE_REVISION_RE.fullmatch(revision):
        raise ContractError(f"{field} must be a 40-character lowercase git commit SHA")
    return revision


def format_score(value: float | Decimal) -> str:
    """Canonical two-decimal half-up representation for a source score."""
    try:
        number = Decimal(str(value))
    except Exception as exc:
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
    """Create exact, ordered score-only SFT JSON (no prose and no markdown)."""
    return _score_json(score)


def _ordered_feedback_object(feedback: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(feedback, Mapping) or tuple(feedback) != HUMAN_FEEDBACK_KEYS:
        raise ContractError("feedback JSON must have exactly ordered human-feedback keys")
    result: dict[str, str] = {}
    for key in HUMAN_FEEDBACK_KEYS:
        value = feedback[key]
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"feedback.{key} must be a nonblank string")
        result[key] = value
    return result


def human_feedback_target(feedback: Mapping[str, Any], score: Mapping[str, Any]) -> str:
    """Create the fixed ordered human-feedback-then-score assistant target."""
    checked = _ordered_feedback_object(feedback)
    feedback_json = json.dumps(checked, ensure_ascii=False, separators=(",", ":"))
    return '{"feedback":' + feedback_json + ',"scores":' + _score_json(score) + "}"


def _no_duplicate_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_score_object(obj: Any) -> dict[str, float]:
    if not isinstance(obj, dict) or tuple(obj) != SCORE_KEYS:
        raise ContractError("score JSON must have exactly ordered keys content, organization, expression, average")
    result: dict[str, float] = {}
    for key in SCORE_KEYS:
        value = obj[key]
        if not isinstance(value, _JsonNumber) or not re.fullmatch(r"[1-5]\.\d{2}", str(value)):
            raise ContractError(f"{key} must be a finite two-decimal JSON number in [1, 5]")
        decimal = Decimal(value)
        if not (SCORE_MIN <= decimal <= SCORE_MAX):
            raise ContractError(f"{key} is out of range")
        result[key] = float(decimal)
    return result


def _strict_feedback_object(obj: Any) -> dict[str, str]:
    return _ordered_feedback_object(obj)


def parse_decoder_output(text: str, mode: str, fallback_mean: Mapping[str, float]) -> ParsedPrediction:
    """Accept only the exact protocol JSON, else use the fixed train mean.

    No prose stripping, partial parsing, or numeric extraction is permitted.
    Generated feedback is schema-checked but never compared to unavailable
    frozen-evaluation feedback.
    """
    try:
        fallback = {key: float(fallback_mean[key]) for key in SCORE_KEYS}
        if set(fallback_mean) != set(SCORE_KEYS):
            raise ContractError("fallback_mean must have exactly four score keys")
        if mode not in {"direct", "human_feedback"}:
            raise ContractError("mode must be direct or human_feedback")
        if not isinstance(text, str):
            raise ContractError("decoder output must be text")
        parsed = json.loads(text, parse_float=_JsonNumber, object_pairs_hook=_no_duplicate_object)
        if mode == "direct":
            return ParsedPrediction(scores=_strict_score_object(parsed), valid=True, feedback_valid=None)
        if not isinstance(parsed, dict) or tuple(parsed) != ("feedback", "scores"):
            raise ContractError("human-feedback output must contain exactly ordered feedback and scores")
        _strict_feedback_object(parsed["feedback"])
        return ParsedPrediction(scores=_strict_score_object(parsed["scores"]), valid=True, feedback_valid=True)
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        fallback = {key: float(fallback_mean[key]) for key in SCORE_KEYS}
        return ParsedPrediction(scores=fallback, valid=False, error=str(exc), feedback_valid=False if mode == "human_feedback" else None)


def target_for_record(record: Mapping[str, Any], mode: str) -> str:
    """Build an assistant target from score-only or source human-feedback labels."""
    score = record.get("score")
    if not isinstance(score, Mapping):
        raise ContractError("record.score must be an object")
    if mode == "direct":
        return direct_target(score)
    if mode != "human_feedback":
        raise ContractError("mode must be direct or human_feedback")
    return human_feedback_target(record.get("feedback"), score)


def prompt_text(record: Mapping[str, Any]) -> str:
    """Build the user message without leaking labels, IDs, or split metadata."""
    prompt, essay = record.get("prompt"), record.get("essay")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(essay, str) or not essay.strip():
        raise ContractError("record must include nonempty prompt and essay strings")
    return (
        "다음 글을 채점하세요. 출력은 지시된 JSON 형식만 사용하세요.\n\n"
        f"[문제]\n{prompt}\n\n[학생 글]\n{essay}"
    )

def template_sha256(system_message: str, user_instruction: str) -> str:
    return hashlib.sha256((system_message + "\n" + user_instruction).encode("utf-8")).hexdigest()


def tokenizer_chat_template_sha256(tokenizer: Any) -> str:
    """Hash the *loaded* tokenizer chat template, not a caller declaration."""
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template.strip():
        raise ContractError("loaded tokenizer must expose a nonempty chat_template")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def require_tokenizer_chat_template(tokenizer: Any, expected_sha256: str) -> str:
    if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(expected_sha256):
        raise ContractError("expected chat template hash must be a SHA-256")
    observed = tokenizer_chat_template_sha256(tokenizer)
    if observed != expected_sha256:
        raise ContractError("loaded tokenizer chat template does not match frozen canonical config")
    return observed


def sanitized_deterministic_generation_config(generation_config: Any) -> Any:
    """Copy and neutralize sampling-only settings for greedy decoding.

    Some model repositories persist temperature/top-p/top-k values in their
    generation config. Transformers correctly warns when those values coexist
    with ``do_sample=False``. They are semantically inactive for greedy decode,
    so set their neutral defaults explicitly rather than suppressing warnings.
    """
    config = deepcopy(generation_config)
    neutral_values = {
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "min_p": None,
        "typical_p": 1.0,
        "top_k": 50,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
    }
    for name, value in neutral_values.items():
        if hasattr(config, name):
            setattr(config, name, value)
    return config


def orderly_distributed_shutdown() -> None:
    """Barrier and destroy an initialized process group; harmless if absent."""
    try:
        import torch.distributed as distributed
    except ImportError:
        return
    if not distributed.is_available() or not distributed.is_initialized():
        return
    try:
        distributed.barrier()
    finally:
        distributed.destroy_process_group()


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
