#!/usr/bin/env python3
"""Run a temporary Cloudflare Quick Tunnel and the matching humaneval origin.

This is an account-free pilot launcher. The generated trycloudflare.com hostname
is written to an ignored file and may change whenever the tunnel restarts.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time


QUICK_TUNNEL_URL = re.compile(
    r"https://[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.trycloudflare\.com"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloudflared", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--url-file", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    return parser.parse_args()


def extract_quick_tunnel_url(line: str) -> str | None:
    match = QUICK_TUNNEL_URL.search(line)
    return match.group(0) if match else None


def atomic_write_url(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{url}\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def terminate(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    args = parse_args()
    repository = args.repository.expanduser().resolve()
    cloudflared = args.cloudflared.expanduser().resolve()
    python = args.python.expanduser().resolve()
    url_file = args.url_file.expanduser().resolve()

    for executable in (cloudflared, python):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise SystemExit(f"required executable is unavailable: {executable}")
    run_py = repository / "humaneval/run.py"
    if not run_py.is_file():
        raise SystemExit(f"humaneval launcher is unavailable: {run_py}")

    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    tunnel = subprocess.Popen(
        [
            str(cloudflared),
            "tunnel",
            "--url",
            f"http://127.0.0.1:{args.port}",
            "--no-autoupdate",
        ],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert tunnel.stdout is not None

    lines: queue.Queue[str | None] = queue.Queue()

    def read_tunnel_output() -> None:
        for line in tunnel.stdout:
            sys.stderr.write(line)
            sys.stderr.flush()
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_tunnel_output, daemon=True)
    reader.start()

    app: subprocess.Popen[str] | None = None
    deadline = time.monotonic() + args.startup_timeout
    public_origin: str | None = None
    try:
        while not stop_requested.is_set() and time.monotonic() < deadline:
            if tunnel.poll() is not None:
                print("cloudflared exited before publishing a URL", file=sys.stderr)
                return 1
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                return 1
            public_origin = extract_quick_tunnel_url(line)
            if public_origin:
                break

        if stop_requested.is_set():
            return 0
        if public_origin is None:
            print("timed out waiting for a Quick Tunnel URL", file=sys.stderr)
            return 1

        atomic_write_url(url_file, public_origin)
        print(f"public URL: {public_origin}", flush=True)
        app = subprocess.Popen(
            [
                str(python),
                str(run_py),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--public-origin",
                public_origin,
            ],
            cwd=repository,
            text=True,
        )

        while not stop_requested.wait(1.0):
            if tunnel.poll() is not None:
                print("cloudflared stopped; restarting the pair", file=sys.stderr)
                return 1
            if app.poll() is not None:
                print("humaneval stopped; restarting the pair", file=sys.stderr)
                return 1
        return 0
    finally:
        terminate(app)
        terminate(tunnel)


if __name__ == "__main__":
    raise SystemExit(main())
