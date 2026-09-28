from __future__ import annotations

from pathlib import Path
import stat
import tempfile
import unittest

from humaneval.deploy.quick_tunnel_supervisor import (
    atomic_write_url,
    extract_quick_tunnel_url,
)


class QuickTunnelSupervisorTests(unittest.TestCase):
    def test_extracts_only_trycloudflare_https_url(self) -> None:
        line = "Visit https://example-alpha-2.trycloudflare.com when ready"
        self.assertEqual(
            extract_quick_tunnel_url(line),
            "https://example-alpha-2.trycloudflare.com",
        )
        self.assertIsNone(extract_quick_tunnel_url("http://bad.trycloudflare.com"))
        self.assertIsNone(extract_quick_tunnel_url("https://example.com"))

    def test_atomic_url_file_is_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "public-url.txt"
            atomic_write_url(path, "https://example.trycloudflare.com")
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "https://example.trycloudflare.com\n",
            )
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
