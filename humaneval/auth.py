"""Shared-password and session authentication helpers.

The generated authentication file contains only versioned scrypt hashes.  It
is still treated as a secret because a copied hash file permits offline
password guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
import base64
import getpass
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from .core import HumanValidationError


AUTH_SCHEMA = "mal2026-humaneval-auth-v2"
DEFAULT_SESSION_TTL_SECONDS = 12 * 60 * 60
DEFAULT_SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
MIN_PASSWORD_LENGTH = 6


class AuthConfigError(HumanValidationError):
    """Raised when authentication material is missing or malformed."""


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: Any, *, field: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise AuthConfigError(f"authentication {field} must be non-empty base64 text")
    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.b64decode(padded, altchars=b"-_", validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise AuthConfigError(f"authentication {field} is not valid base64") from exc


@dataclass(frozen=True)
class SecretHash:
    salt: bytes
    digest: bytes
    n: int = DEFAULT_SCRYPT_N
    r: int = SCRYPT_R
    p: int = SCRYPT_P

    @classmethod
    def create(cls, secret: str, *, n: int = DEFAULT_SCRYPT_N) -> "SecretHash":
        if not isinstance(secret, str) or not secret:
            raise AuthConfigError("cannot hash an empty authentication secret")
        if n < 2**10 or n > 2**18 or n & (n - 1):
            raise AuthConfigError("scrypt n must be a power of two from 2^10 through 2^18")
        salt = os.urandom(16)
        digest = hashlib.scrypt(
            secret.encode("utf-8"), salt=salt, n=n, r=SCRYPT_R, p=SCRYPT_P,
            dklen=SCRYPT_DKLEN, maxmem=256 * 1024 * 1024,
        )
        return cls(salt=salt, digest=digest, n=n)

    @classmethod
    def from_dict(cls, value: Any) -> "SecretHash":
        if not isinstance(value, dict) or set(value) != {
            "algorithm", "n", "r", "p", "salt", "digest"
        }:
            raise AuthConfigError("authentication hash object has an unexpected schema")
        if value["algorithm"] != "scrypt":
            raise AuthConfigError("only scrypt authentication hashes are supported")
        n, r, p = value["n"], value["r"], value["p"]
        if type(n) is not int or n < 2**10 or n > 2**18 or n & (n - 1):
            raise AuthConfigError("authentication scrypt n is outside the accepted range")
        if r != SCRYPT_R or p != SCRYPT_P:
            raise AuthConfigError("authentication scrypt r/p parameters differ from the contract")
        salt = _b64decode(value["salt"], field="salt")
        digest = _b64decode(value["digest"], field="digest")
        if len(salt) != 16 or len(digest) != SCRYPT_DKLEN:
            raise AuthConfigError("authentication salt or digest length is invalid")
        return cls(salt=salt, digest=digest, n=n, r=r, p=p)

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "scrypt",
            "n": self.n,
            "r": self.r,
            "p": self.p,
            "salt": _b64encode(self.salt),
            "digest": _b64encode(self.digest),
        }

    def verify(self, candidate: str) -> bool:
        if not isinstance(candidate, str):
            candidate = ""
        derived = hashlib.scrypt(
            candidate.encode("utf-8"), salt=self.salt, n=self.n, r=self.r,
            p=self.p, dklen=len(self.digest), maxmem=256 * 1024 * 1024,
        )
        return hmac.compare_digest(self.digest, derived)


@dataclass(frozen=True)
class AuthConfig:
    shared_password: SecretHash
    session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS

    def authenticate(self, shared_password: str) -> bool:
        return self.shared_password.verify(shared_password)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": AUTH_SCHEMA,
            "session_ttl_seconds": self.session_ttl_seconds,
            "shared_password": self.shared_password.as_dict(),
        }


def create_auth_config(
    shared_password: str,
    *,
    session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
    scrypt_n: int = DEFAULT_SCRYPT_N,
) -> AuthConfig:
    if not isinstance(shared_password, str) or len(shared_password) < MIN_PASSWORD_LENGTH:
        raise AuthConfigError(
            f"shared password must contain at least {MIN_PASSWORD_LENGTH} characters"
        )
    if type(session_ttl_seconds) is not int or not 15 * 60 <= session_ttl_seconds <= 24 * 60 * 60:
        raise AuthConfigError("session lifetime must be between 15 minutes and 24 hours")
    return AuthConfig(
        shared_password=SecretHash.create(shared_password, n=scrypt_n),
        session_ttl_seconds=session_ttl_seconds,
    )


def write_auth_config(path: Path, config: AuthConfig, *, replace: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and not replace:
        raise AuthConfigError(
            f"authentication file already exists; use the explicit replacement option: {resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        resolved.parent.chmod(0o700)
    except OSError:
        pass
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(config.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(resolved)
        resolved.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
    return resolved


def load_auth_config(path: Path) -> AuthConfig:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise AuthConfigError(
            f"authentication file does not exist; generate it before launch: {resolved}"
        )
    if resolved.stat().st_size > 64 * 1024:
        raise AuthConfigError("authentication file is unexpectedly large")
    if os.name == "posix" and resolved.stat().st_mode & 0o077:
        raise AuthConfigError(
            f"authentication file must not be readable by group or others (run chmod 600): {resolved}"
        )
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthConfigError("authentication file is not valid UTF-8 JSON") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "schema", "session_ttl_seconds", "shared_password"
    }:
        raise AuthConfigError("authentication file has an unexpected schema")
    if raw["schema"] != AUTH_SCHEMA:
        raise AuthConfigError("authentication file schema version is unsupported")
    ttl = raw["session_ttl_seconds"]
    if type(ttl) is not int or not 15 * 60 <= ttl <= 24 * 60 * 60:
        raise AuthConfigError("authentication session lifetime is outside the accepted range")
    return AuthConfig(
        shared_password=SecretHash.from_dict(raw["shared_password"]),
        session_ttl_seconds=ttl,
    )


def prompt_and_write_auth_config(path: Path, *, replace: bool = False) -> Path:
    """Prompt without terminal echo and write only hashes to ``path``."""
    print(f"Authentication hashes will be written to: {path.expanduser().resolve()}")
    shared = getpass.getpass(
        f"공통 입장 비밀번호 ({MIN_PASSWORD_LENGTH}자 이상, 입력 내용은 표시되지 않음): "
    )
    shared_confirmation = getpass.getpass("공통 입장 비밀번호 확인: ")
    if shared != shared_confirmation:
        raise AuthConfigError("shared password confirmation does not match")
    config = create_auth_config(shared)
    return write_auth_config(path, config, replace=replace)
