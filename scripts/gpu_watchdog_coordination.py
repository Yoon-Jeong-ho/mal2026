#!/usr/bin/env python3
"""Aggregate-only ownership leases for the local GPU 0--3 watchdogs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

ALLOWED = {0, 1, 2, 3}


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def need(value: bool, message: str) -> None:
    if not value:
        raise SystemExit(message)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def coordination_dir(root: Path, requested: str | Path | None) -> Path:
    result = Path(requested).resolve() if requested else (root / "outputs" / "reservations" / "gpu0-3-watchdog-coordination-v1").resolve()
    need(result.is_relative_to(root / "outputs" / "reservations"), "coordination directory escapes reservations")
    return result


def gpu_has_compute_process(gpu: int) -> bool:
    """Return busy/idle only. Process IDs and commands are neither stored nor logged."""
    need(gpu in ALLOWED, "GPU is outside 0--3")
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GPU {gpu} process query failed")
    return any(line.strip() and line.strip() != "No running processes found" for line in result.stdout.splitlines())


@dataclass
class GpuLease:
    directory: Path
    gpu: int
    owner: str
    purpose: str
    kind: str
    token: str
    descriptor: int

    @property
    def owner_path(self) -> Path:
        return self.directory / f"gpu{self.gpu}-owner.json"

    @property
    def request_path(self) -> Path:
        return self.directory / f"gpu{self.gpu}-priority-request.json"

    @classmethod
    def acquire(cls, directory: Path, gpu: int, owner: str, purpose: str, kind: str) -> "GpuLease | None":
        need(gpu in ALLOWED, "GPU is outside 0--3")
        directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(directory / f"gpu{gpu}.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return None
        lease = cls(directory, gpu, owner, purpose, kind, uuid4().hex, descriptor)
        atomic_json(lease.owner_path, {"at": now(), "gpu": gpu, "owner": owner, "run_purpose": purpose, "kind": kind, "token": lease.token})
        return lease

    def priority_requested(self) -> bool:
        return self.kind == "backfill" and self.request_path.is_file()

    def release(self) -> None:
        try:
            try:
                if json.loads(self.owner_path.read_text(encoding="utf-8")).get("token") == self.token:
                    self.owner_path.unlink(missing_ok=True)
            except (OSError, json.JSONDecodeError):
                pass
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)


def request_priority(directory: Path, gpu: int, requester: str, kind: str) -> None:
    need(gpu in ALLOWED, "GPU is outside 0--3")
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / f"gpu{gpu}-priority-request.json", {"at": now(), "gpu": gpu, "requester": requester, "kind": kind})


def clear_priority_request(directory: Path, gpu: int, requester: str | None = None) -> None:
    path = directory / f"gpu{gpu}-priority-request.json"
    if requester is not None:
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("requester") != requester:
                return
        except (OSError, json.JSONDecodeError):
            return
    path.unlink(missing_ok=True)
