"""Dependency-free authenticated HTTP application for human validation."""
from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import ipaddress
import json
import mimetypes
from pathlib import Path
import secrets
from threading import BoundedSemaphore, Lock
import time
from typing import Any
from urllib.parse import urlparse

from .auth import AuthConfig
from .core import (
    ALLOWED_USERS,
    HumanValidationConflict,
    HumanValidationError,
    ResponseStore,
)


@dataclass(frozen=True)
class Session:
    token: str
    user_name: str
    csrf_token: str
    expires_at: float


class SessionStore:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()

    def create(self, user_name: str) -> Session:
        token = secrets.token_urlsafe(32)
        session = Session(
            token=token,
            user_name=user_name,
            csrf_token=secrets.token_urlsafe(32),
            expires_at=time.time() + self.ttl_seconds,
        )
        with self._lock:
            self._sessions[token] = session
        return session

    def resolve(self, token: str | None) -> Session | None:
        if token is None:
            return None
        with self._lock:
            session = self._sessions.get(token)
            if session is not None and session.expires_at <= time.time():
                self._sessions.pop(token, None)
                return None
            return session

    def remove(self, token: str | None) -> None:
        if token is None:
            return
        with self._lock:
            self._sessions.pop(token, None)


class LoginAttemptLimiter:
    """Small in-memory limiter for shared-password guesses."""

    def __init__(
        self,
        *,
        max_failures: int = 5,
        window_seconds: int = 10 * 60,
        block_seconds: int = 15 * 60,
    ) -> None:
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self.block_seconds = block_seconds
        self._failures: dict[str, list[float]] = {}
        self._blocked_until: dict[str, float] = {}
        self._lock = Lock()

    def allowed(self, key: str, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        with self._lock:
            blocked_until = self._blocked_until.get(key, 0.0)
            if blocked_until > current:
                return False
            self._blocked_until.pop(key, None)
            recent = [
                value for value in self._failures.get(key, ())
                if value > current - self.window_seconds
            ]
            if recent:
                self._failures[key] = recent
            else:
                self._failures.pop(key, None)
            return True

    def record_failure(self, key: str, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        with self._lock:
            recent = [
                value for value in self._failures.get(key, ())
                if value > current - self.window_seconds
            ]
            recent.append(current)
            self._failures[key] = recent
            if len(recent) >= self.max_failures:
                self._blocked_until[key] = current + self.block_seconds
                self._failures.pop(key, None)

    def record_success(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
            self._blocked_until.pop(key, None)


class HumanValidationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        store: ResponseStore,
        static_root: Path,
        *,
        auth_config: AuthConfig,
        allowed_origins: set[str],
        secure_cookie: bool = True,
        login_limiter: LoginAttemptLimiter | None = None,
        global_login_limiter: LoginAttemptLimiter | None = None,
    ):
        if not allowed_origins:
            raise ValueError("at least one allowed browser origin is required")
        self.response_store = store
        self.auth_config = auth_config
        self.session_store = SessionStore(auth_config.session_ttl_seconds)
        self.login_limiter = login_limiter or LoginAttemptLimiter()
        self.global_login_limiter = global_login_limiter or LoginAttemptLimiter(
            max_failures=50,
            window_seconds=10 * 60,
            block_seconds=15 * 60,
        )
        # Scrypt is intentionally memory-hard. Bound concurrent verifications
        # so a burst of new connections cannot multiply its memory use without
        # limit before the failed-attempt counters update.
        self.login_slots = BoundedSemaphore(4)
        self.allowed_origins = frozenset(
            origin.rstrip("/").lower() for origin in allowed_origins
        )
        self.secure_cookie = secure_cookie
        self.static_root = static_root.resolve()
        super().__init__(address, HumanValidationHandler)


class HumanValidationHandler(BaseHTTPRequestHandler):
    server: HumanValidationServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        super().log_message(format, *args)

    def _token(self) -> str | None:
        try:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
        except CookieError:
            return None
        morsel = cookie.get("hvsession")
        return None if morsel is None else morsel.value

    def _session(self) -> Session | None:
        return self.server.session_store.resolve(self._token())

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        cookie: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
        )
        if self.server.secure_cookie:
            self.send_header(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' https://challenges.cloudflare.com; "
            "style-src 'self'; img-src 'self'; connect-src 'self' https://challenges.cloudflare.com; "
            "frame-src https://challenges.cloudflare.com; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'",
        )
        if cookie is not None:
            self.send_header("Set-Cookie", cookie)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(
        self,
        status: int,
        value: Any,
        *,
        cookie: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(
            status,
            body,
            "application/json; charset=utf-8",
            cookie=cookie,
            headers=headers,
        )

    def _error(
        self, status: int, message: str, *, headers: dict[str, str] | None = None
    ) -> None:
        self._json(status, {"error": message}, headers=headers)

    def _cookie(self, token: str, *, max_age: int) -> str:
        value = (
            f"hvsession={token}; Path=/; Max-Age={max_age}; "
            "HttpOnly; SameSite=Strict"
        )
        if self.server.secure_cookie:
            value += "; Secure"
        return value

    def _state(self, session: Session, value: dict[str, Any] | None = None) -> dict[str, Any]:
        state = (
            self.server.response_store.state(session.user_name)
            if value is None else value
        )
        state["csrf_token"] = session.csrf_token
        return state

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return False
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
            return False
        normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}".rstrip("/")
        return normalized in self.server.allowed_origins

    def _csrf_allowed(self, session: Session) -> bool:
        supplied = self.headers.get("X-CSRF-Token", "")
        return hmac.compare_digest(
            session.csrf_token.encode("ascii"), supplied.encode("utf-8")
        )

    def _client_key(self) -> str:
        # The service is intentionally bound to loopback, so this header is
        # supplied by the local cloudflared process rather than a public peer.
        candidate = self.headers.get("CF-Connecting-IP")
        if candidate is not None:
            try:
                return str(ipaddress.ip_address(candidate.strip()))
            except ValueError:
                pass
        return str(self.client_address[0])

    def _read_json(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise HumanValidationError("request content type must be application/json")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise HumanValidationError("request body is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise HumanValidationError("invalid request length") from exc
        if length < 0 or length > 32_768:
            raise HumanValidationError("request body is too large")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HumanValidationError("request body is not valid JSON") from exc
        if not isinstance(value, dict):
            raise HumanValidationError("request JSON must be an object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/healthz":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        if path == "/api/config":
            self._json(HTTPStatus.OK, {"turnstile_site_key": None})
            return
        if path == "/api/state":
            session = self._session()
            if session is None:
                self._error(HTTPStatus.UNAUTHORIZED, "로그인이 필요합니다.")
                return
            self._json(HTTPStatus.OK, self._state(session))
            return
        static_names = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}
        name = static_names.get(path)
        if name is None:
            self._error(HTTPStatus.NOT_FOUND, "찾을 수 없습니다.")
            return
        target = (self.server.static_root / name).resolve()
        if not target.is_relative_to(self.server.static_root) or not target.is_file():
            self._error(HTTPStatus.NOT_FOUND, "정적 파일을 찾을 수 없습니다.")
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type += "; charset=utf-8"
        self._send_bytes(HTTPStatus.OK, target.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not self._origin_allowed():
            # The request body may contain credentials. Close the connection
            # after the denial so unread bytes can never be parsed/logged as a
            # second HTTP request line on this keep-alive connection.
            self.close_connection = True
            self._error(HTTPStatus.FORBIDDEN, "허용되지 않은 요청 출처입니다.")
            return
        try:
            payload = self._read_json()
            if path == "/api/login":
                name = payload.get("name")
                password = payload.get("password")
                name = name if isinstance(name, str) else ""
                password = password if isinstance(password, str) else ""
                if len(name) > 64:
                    name = ""
                if len(password) > 256:
                    password = ""
                client_key = self._client_key()
                if (
                    not self.server.login_limiter.allowed(client_key)
                    or not self.server.global_login_limiter.allowed("all")
                ):
                    self._error(
                        HTTPStatus.TOO_MANY_REQUESTS,
                        "로그인 시도가 너무 많습니다. 15분 뒤 다시 시도해 주세요.",
                        headers={"Retry-After": "900"},
                    )
                    return
                if not self.server.login_slots.acquire(blocking=False):
                    self._error(
                        HTTPStatus.TOO_MANY_REQUESTS,
                        "로그인 요청이 처리 중입니다. 잠시 뒤 다시 시도해 주세요.",
                        headers={"Retry-After": "5"},
                    )
                    return
                try:
                    password_ok = self.server.auth_config.authenticate(password)
                finally:
                    self.server.login_slots.release()
                if name not in ALLOWED_USERS or not password_ok:
                    self.server.login_limiter.record_failure(client_key)
                    self.server.global_login_limiter.record_failure("all")
                    self._error(
                        HTTPStatus.UNAUTHORIZED,
                        "평가자 이름 또는 공통 비밀번호가 올바르지 않습니다.",
                    )
                    return
                self.server.login_limiter.record_success(client_key)
                session = self.server.session_store.create(name)
                cookie = self._cookie(
                    session.token,
                    max_age=self.server.auth_config.session_ttl_seconds,
                )
                self._json(
                    HTTPStatus.OK,
                    self._state(session),
                    cookie=cookie,
                )
                return
            session = self._session()
            if session is None:
                self._error(HTTPStatus.UNAUTHORIZED, "로그인이 필요합니다.")
                return
            if not self._csrf_allowed(session):
                self._error(HTTPStatus.FORBIDDEN, "요청 확인 토큰이 올바르지 않습니다.")
                return
            if path == "/api/logout":
                self.server.session_store.remove(self._token())
                self._json(
                    HTTPStatus.OK,
                    {"ok": True},
                    cookie=self._cookie("", max_age=0),
                )
                return
            user = session.user_name
            item_index = payload.get("item_index")
            if type(item_index) is not int:
                raise HumanValidationError("문항 번호가 올바르지 않습니다.")
            if path == "/api/score":
                scores = payload.get("scores")
                reasons = payload.get("reasons")
                if not isinstance(scores, dict) or not isinstance(reasons, dict):
                    raise HumanValidationError("세 영역의 점수를 모두 선택해 주세요.")
                state = self.server.response_store.record_scores(user, item_index, scores, reasons)
                self._json(HTTPStatus.OK, self._state(session, state))
                return
            if path == "/api/rationale":
                verdicts = payload.get("verdicts")
                reasons = payload.get("reasons")
                if not isinstance(verdicts, dict) or not isinstance(reasons, dict):
                    raise HumanValidationError("세 영역의 적절성을 모두 판단해 주세요.")
                state = self.server.response_store.record_rationale(
                    user, item_index, verdicts, reasons
                )
                self._json(HTTPStatus.OK, self._state(session, state))
                return
            self._error(HTTPStatus.NOT_FOUND, "찾을 수 없습니다.")
        except HumanValidationConflict as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except HumanValidationError as exc:
            # Some request-shape failures happen before the body is consumed.
            # Closing prevents leftover credential bytes from becoming a new
            # request line on this HTTP/1.1 connection.
            self.close_connection = True
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "서버 처리 중 오류가 발생했습니다.")
