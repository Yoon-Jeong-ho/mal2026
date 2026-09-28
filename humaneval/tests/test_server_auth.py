from __future__ import annotations

import http.client
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest

from humaneval.auth import create_auth_config
from humaneval.server import (
    HumanValidationServer,
    LoginAttemptLimiter,
)


ORIGIN = "https://human-eval.example.com"
SHARED_PASSWORD = "correct horse battery staple"


class FakeResponseStore:
    def state(self, user_name: str):
        return {
            "user": user_name,
            "phase": "score",
            "progress": {"completed": 0, "total": 20},
        }


class AuthenticatedServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        auth = create_auth_config(SHARED_PASSWORD, scrypt_n=2**10)
        self.server = HumanValidationServer(
            ("127.0.0.1", 0),
            FakeResponseStore(),
            root,
            auth_config=auth,
            allowed_origins={ORIGIN},
            secure_cookie=True,
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temporary.cleanup()

    def request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        origin: str | None = ORIGIN,
        cookie: str | None = None,
        csrf: str | None = None,
    ):
        headers = {}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if origin is not None:
            headers["Origin"] = origin
        if cookie is not None:
            headers["Cookie"] = cookie
        if csrf is not None:
            headers["X-CSRF-Token"] = csrf
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            value = json.loads(raw.decode("utf-8"))
            return response.status, dict(response.getheaders()), value
        finally:
            connection.close()

    def test_password_session_and_csrf_contract(self):
        status, _, _ = self.request("GET", "/api/state", origin=None)
        self.assertEqual(401, status)

        status, _, _ = self.request("POST", "/api/login", {
            "name": "정호", "password": "wrong password"
        })
        self.assertEqual(401, status)

        status, headers, state = self.request("POST", "/api/login", {
            "name": "정호", "password": SHARED_PASSWORD
        })
        self.assertEqual(200, status)
        self.assertEqual("정호", state["user"])
        self.assertIn("csrf_token", state)
        set_cookie = headers["Set-Cookie"]
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("SameSite=Strict", set_cookie)
        self.assertIn("Secure", set_cookie)
        cookie = set_cookie.split(";", 1)[0]

        status, _, resumed = self.request(
            "GET", "/api/state", origin=None, cookie=cookie
        )
        self.assertEqual(200, status)
        self.assertEqual("정호", resumed["user"])

        status, _, _ = self.request(
            "POST", "/api/logout", {}, cookie=cookie
        )
        self.assertEqual(403, status)
        status, _, _ = self.request(
            "POST", "/api/logout", {}, cookie=cookie, csrf=state["csrf_token"]
        )
        self.assertEqual(200, status)

    def test_cross_origin_login_is_rejected_before_authentication(self):
        status, _, value = self.request(
            "POST",
            "/api/login",
            {"name": "정호", "password": SHARED_PASSWORD},
            origin="https://attacker.example",
        )
        self.assertEqual(403, status)
        self.assertIn("출처", value["error"])

    def test_login_limiter_blocks_and_recovers(self):
        limiter = LoginAttemptLimiter(
            max_failures=2, window_seconds=10, block_seconds=20
        )
        self.assertTrue(limiter.allowed("client", now=100))
        limiter.record_failure("client", now=100)
        limiter.record_failure("client", now=101)
        self.assertFalse(limiter.allowed("client", now=102))
        self.assertTrue(limiter.allowed("client", now=122))


if __name__ == "__main__":
    unittest.main()
