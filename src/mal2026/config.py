"""Fail-closed validation for non-secret experiment configuration files."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Mapping


class ConfigError(ValueError):
    """A configuration is incomplete, mutable, or outside the frozen contract."""


_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_KINDS = frozenset({"decoder-direct", "decoder-rationale-score", "encoder-qwen3", "encoder-nvembed"})


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be an object")
    return value


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.startswith("REQUIRED_"):
        raise ConfigError(f"{name} is required")
    return value


def _immutable_revision(value: Any, name: str) -> str:
    result = _required_text(value, name)
    if not _IMMUTABLE_REVISION.fullmatch(result):
        raise ConfigError(f"{name} must be a lowercase immutable 40--64 hex commit revision")
    return result


def _sha256(value: Any, name: str) -> str:
    result = _required_text(value, name)
    if not _SHA256.fullmatch(result):
        raise ConfigError(f"{name} must be a lowercase SHA-256")
    return result


def _require_adapter(raw: Mapping[str, Any]) -> None:
    targets = raw.get("target_modules")
    if not isinstance(targets, list) or not targets or not all(isinstance(item, str) and item and not item.startswith("REQUIRED_") for item in targets):
        raise ConfigError("adapter.target_modules must be a nonempty, architecture-validated list")
    for name in ("rank", "alpha"):
        if not isinstance(raw.get(name), int) or raw[name] <= 0:
            raise ConfigError(f"adapter.{name} must be a positive integer")
    if not isinstance(raw.get("dropout"), (int, float)) or not 0 <= float(raw["dropout"]) < 1:
        raise ConfigError("adapter.dropout must be in [0, 1)")


def validate_experiment_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-normalized config only when all launch-critical fields freeze."""
    raw = _mapping(raw, "config")
    kind = _required_text(raw.get("run_kind"), "run_kind")
    if kind not in _ALLOWED_KINDS:
        raise ConfigError("run_kind is not an approved experiment")
    model = _mapping(raw.get("model"), "model")
    _required_text(model.get("id"), "model.id")
    _immutable_revision(model.get("revision"), "model.revision")
    _immutable_revision(model.get("tokenizer_revision"), "model.tokenizer_revision")
    if kind.startswith("decoder"):
        _sha256(model.get("chat_template_sha256"), "model.chat_template_sha256")
    else:
        _required_text(model.get("pooling"), "model.pooling")
        _required_text(model.get("pooling_revision"), "model.pooling_revision")
    _require_adapter(_mapping(raw.get("adapter"), "adapter"))

    data = _mapping(raw.get("data"), "data")
    if not isinstance(data.get("max_sequence_length"), int) or data["max_sequence_length"] <= 0:
        raise ConfigError("data.max_sequence_length must be positive")
    fraction = data.get("dev_fraction")
    if not isinstance(fraction, (int, float)) or not 0 < float(fraction) < 1:
        raise ConfigError("data.dev_fraction must be between zero and one")

    optimization = _mapping(raw.get("optimization"), "optimization")
    for name in ("seed", "epochs", "per_device_batch_size", "gradient_accumulation_steps"):
        if not isinstance(optimization.get(name), int) or optimization[name] <= 0:
            raise ConfigError(f"optimization.{name} must be a positive integer")
    if not isinstance(optimization.get("learning_rate"), (int, float)) or float(optimization["learning_rate"]) <= 0:
        raise ConfigError("optimization.learning_rate must be positive")

    if kind == "decoder-rationale-score":
        teacher = _mapping(raw.get("teacher"), "teacher")
        _required_text(teacher.get("id"), "teacher.id")
        _immutable_revision(teacher.get("revision"), "teacher.revision")
        _sha256(teacher.get("prompt_template_sha256"), "teacher.prompt_template_sha256")
        for name in ("seed", "max_new_tokens", "max_retries"):
            if not isinstance(teacher.get(name), int) or teacher[name] < 0:
                raise ConfigError(f"teacher.{name} must be a nonnegative integer")
    if kind == "encoder-nvembed":
        review = _mapping(raw.get("remote_code_review"), "remote_code_review")
        acknowledgement = _required_text(review.get("license_acknowledgement"), "remote_code_review.license_acknowledgement")
        if "NONCOMMERCIAL" not in acknowledgement.upper():
            raise ConfigError("NV-Embed requires an explicit non-commercial acknowledgement")
        if _required_text(review.get("review_outcome"), "remote_code_review.review_outcome") != "APPROVED":
            raise ConfigError("NV-Embed remote code review outcome must be APPROVED")
        files = review.get("reviewed_file_sha256")
        if not isinstance(files, Mapping) or not files:
            raise ConfigError("NV-Embed reviewed_file_sha256 must be nonempty")
        for filename, digest in files.items():
            _required_text(filename, "remote-code filename")
            _sha256(digest, f"remote-code hash for {filename}")
    # JSON round-trip rejects non-serializable accidental runtime objects.
    return json.loads(json.dumps(raw, sort_keys=True))


def load_experiment_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Read JSON config, validate it, and return it with a stable config hash."""
    location = Path(path)
    try:
        raw = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"unable to read JSON config: {location}") from exc
    config = validate_experiment_config(raw)
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return config, sha256(canonical.encode("utf-8")).hexdigest()
