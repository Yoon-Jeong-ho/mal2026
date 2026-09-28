"""Interactively configure the shared Worker password without echoing it."""

from __future__ import annotations

import getpass
import hashlib
import hmac
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys


MIN_PASSWORD_LENGTH = 6
ROOT = Path(__file__).resolve().parent


def derive_digest(password: str, pepper: str) -> str:
    return hmac.new(
        pepper.encode("utf-8"),
        password.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def put_secret(wrangler: Path, name: str, value: str) -> None:
    completed = subprocess.run(
        [str(wrangler), "secret", "put", name],
        cwd=ROOT,
        env={**os.environ, "WRANGLER_LOG_PATH": f"/private/tmp/mal2026-{name.lower()}.log"},
        input=value + "\n",
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Cloudflare secret registration failed for {name}")


def main() -> int:
    wrangler = ROOT / "node_modules" / ".bin" / "wrangler"
    if not wrangler.is_file() or shutil.which("node") is None:
        print("Wrangler or Node.js is unavailable in the current environment.", file=sys.stderr)
        return 1

    print("운영 공통 비밀번호를 Cloudflare에 등록합니다.")
    print("입력 내용은 화면, 명령행, 파일에 남지 않습니다.")
    password = getpass.getpass(f"공통 비밀번호 ({MIN_PASSWORD_LENGTH}자 이상): ")
    confirmation = getpass.getpass("공통 비밀번호 확인: ")
    if password != confirmation:
        print("두 비밀번호가 일치하지 않습니다.", file=sys.stderr)
        return 2
    if len(password) < MIN_PASSWORD_LENGTH:
        print(f"비밀번호는 {MIN_PASSWORD_LENGTH}자 이상이어야 합니다.", file=sys.stderr)
        return 2

    pepper = secrets.token_urlsafe(32)
    digest = derive_digest(password, pepper)
    try:
        put_secret(wrangler, "PASSWORD_PEPPER", pepper)
        put_secret(wrangler, "PASSWORD_DIGEST", digest)
    finally:
        password = ""
        confirmation = ""
        digest = ""
        pepper = ""
    print("운영 비밀번호 등록이 완료됐습니다. 이 터미널을 닫아도 됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
