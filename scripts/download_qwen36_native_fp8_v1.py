#!/usr/bin/env python3
"""Resumably fetch and checksum the one authorized official native-FP8 repo."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen36_native_fp8_vllm.v1.json"
REPOSITORY = "Qwen/Qwen3.6-35B-A3B-FP8"
PUBLISHED_BYTES = 37493015668


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True); a = p.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg["model"] != {"repository": REPOSITORY, "revision": "resolved-at-download-and-pinned-in-manifest", "official_repository_bytes": PUBLISHED_BYTES, "format": "native-fp8-transformers"}:
        raise SystemExit("native-FP8 config provenance contract changed")
    if a.manifest.exists(): raise SystemExit("refusing to overwrite a download manifest")
    api = json.loads(urlopen("https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B-FP8?blobs=true", timeout=60).read().decode("utf-8"))
    revision = api.get("sha")
    siblings = api.get("siblings")
    if not isinstance(revision, str) or len(revision) != 40 or not isinstance(siblings, list):
        raise SystemExit("official Hugging Face metadata did not provide an immutable revision/manifest")
    official = []
    for item in siblings:
        name, size, lfs = item.get("rfilename"), item.get("size"), item.get("lfs", {})
        if not isinstance(name, str) or not isinstance(size, int) or size < 0: raise SystemExit("official file metadata is incomplete")
        official.append({"path": name, "bytes": size, "official_sha256": lfs.get("oid") if isinstance(lfs, dict) else None})
    if sum(item["bytes"] for item in official) != PUBLISHED_BYTES:
        raise SystemExit("official repository byte total differs from authorized scout evidence")
    a.destination.parent.mkdir(parents=True, exist_ok=True)
    # huggingface_hub resumes verified partial blobs in its cache/local-dir metadata;
    # no credential argument is passed, so an ambient token can never reach logs.
    command = [str(ROOT / ".venv-standard/bin/hf"), "download", REPOSITORY, "--revision", revision,
               "--local-dir", str(a.destination), "--max-workers", "4", "--quiet"]
    completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if completed.returncode:
        raise SystemExit("official hf download failed; resumable local cache was preserved")
    observed = []
    for item in official:
        path = a.destination / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]: raise SystemExit("downloaded official file is absent or byte-mismatched")
        actual = digest(path)
        if item["official_sha256"] and actual != item["official_sha256"]: raise SystemExit("downloaded official LFS checksum mismatch")
        observed.append({**item, "observed_sha256": actual})
    manifest = {"schema_version": "mal2026-native-fp8-checkpoint-manifest-v1", "repository": REPOSITORY,
                "revision": revision, "official_repository_bytes": PUBLISHED_BYTES,
                "observed_repository_bytes": sum(item["bytes"] for item in observed), "files": observed,
                "config_sha256": digest(CONFIG), "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "credentials_recorded": False, "download_tool": "huggingface_hub hf download (resumable local-dir/cache)"}
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "downloaded_and_verified", "revision": revision, "files": len(observed), "bytes": manifest["observed_repository_bytes"], "manifest": str(a.manifest)}, sort_keys=True))


if __name__ == "__main__": main()
