"""Dependency-free local HTTP application for the human validation study."""
from __future__ import annotations

from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from .core import (
    ALLOWED_USERS,
    HumanValidationConflict,
    HumanValidationError,
    ResponseStore,
)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, str] = {}
        self._lock = Lock()

    def create(self, user_name: str) -> str:
        if user_name not in ALLOWED_USERS:
            raise HumanValidationError("reviewer name is not allowed")
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[token] = user_name
        return token

    def resolve(self, token: str | None) -> str | None:
        if token is None:
            return None
        with self._lock:
            return self._sessions.get(token)

    def remove(self, token: str | None) -> None:
        if token is None:
            return
        with self._lock:
            self._sessions.pop(token, None)


class HumanValidationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: ResponseStore, static_root: Path):
        self.response_store = store
        self.session_store = SessionStore()
        self.static_root = static_root.resolve()
        super().__init__(address, HumanValidationHandler)


class HumanValidationHandler(BaseHTTPRequestHandler):
    server: HumanValidationServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        super().log_message(format, *args)

    def _token(self) -> str | None:
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        morsel = cookie.get("hvsession")
        return None if morsel is None else morsel.value

    def _user(self) -> str | None:
        return self.server.session_store.resolve(self._token())

    def _send_bytes(self, status: int, body: bytes, content_type: str, *, cookie: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )
        if cookie is not None:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: Any, *, cookie: str | None = None) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8", cookie=cookie)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _read_json(self) -> dict[str, Any]:
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
        if path == "/api/state":
            user = self._user()
            if user is None:
                self._error(HTTPStatus.UNAUTHORIZED, "로그인이 필요합니다.")
                return
            self._json(HTTPStatus.OK, self.server.response_store.state(user))
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
        try:
            payload = self._read_json()
            if path == "/api/login":
                name = payload.get("name")
                if not isinstance(name, str) or name not in ALLOWED_USERS:
                    raise HumanValidationError("등록된 이름을 선택해 주세요.")
                token = self.server.session_store.create(name)
                cookie = f"hvsession={token}; Path=/; HttpOnly; SameSite=Strict"
                self._json(HTTPStatus.OK, self.server.response_store.state(name), cookie=cookie)
                return
            if path == "/api/logout":
                self.server.session_store.remove(self._token())
                self._json(HTTPStatus.OK, {"ok": True}, cookie="hvsession=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict")
                return
            user = self._user()
            if user is None:
                self._error(HTTPStatus.UNAUTHORIZED, "로그인이 필요합니다.")
                return
            item_index = payload.get("item_index")
            if type(item_index) is not int:
                raise HumanValidationError("문항 번호가 올바르지 않습니다.")
            if path == "/api/score":
                scores = payload.get("scores")
                reasons = payload.get("reasons")
                if not isinstance(scores, dict) or not isinstance(reasons, dict):
                    raise HumanValidationError("세 영역의 점수를 모두 선택해 주세요.")
                state = self.server.response_store.record_scores(user, item_index, scores, reasons)
                self._json(HTTPStatus.OK, state)
                return
            if path == "/api/rationale":
                verdicts = payload.get("verdicts")
                reasons = payload.get("reasons")
                if not isinstance(verdicts, dict) or not isinstance(reasons, dict):
                    raise HumanValidationError("세 영역의 적절성을 모두 판단해 주세요.")
                state = self.server.response_store.record_rationale(
                    user, item_index, verdicts, reasons
                )
                self._json(HTTPStatus.OK, state)
                return
            self._error(HTTPStatus.NOT_FOUND, "찾을 수 없습니다.")
        except HumanValidationConflict as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except HumanValidationError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "서버 처리 중 오류가 발생했습니다.")
