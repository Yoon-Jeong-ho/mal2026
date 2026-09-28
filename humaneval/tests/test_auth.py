from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from humaneval.auth import (
    AuthConfigError,
    create_auth_config,
    load_auth_config,
    write_auth_config,
)
SHARED_PASSWORD = "correct horse battery staple"


class AuthenticationConfigTests(unittest.TestCase):
    def config(self):
        # A small test-only work factor keeps unit tests fast. The interactive
        # generator always uses the production work factor.
        return create_auth_config(SHARED_PASSWORD, scrypt_n=2**10)

    def test_round_trip_and_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            write_auth_config(path, self.config())
            loaded = load_auth_config(path)
            self.assertTrue(loaded.authenticate(SHARED_PASSWORD))
            self.assertFalse(loaded.authenticate("wrong password"))
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(SHARED_PASSWORD, text)
            if os.name == "posix":
                self.assertEqual(0o600, path.stat().st_mode & 0o777)

    def test_short_password_fails_closed(self):
        with self.assertRaises(AuthConfigError):
            create_auth_config("12345", scrypt_n=2**10)

    def test_existing_file_requires_explicit_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            write_auth_config(path, self.config())
            with self.assertRaises(AuthConfigError):
                write_auth_config(path, self.config())
            write_auth_config(path, self.config(), replace=True)
            self.assertTrue(load_auth_config(path).authenticate(SHARED_PASSWORD))


if __name__ == "__main__":
    unittest.main()
