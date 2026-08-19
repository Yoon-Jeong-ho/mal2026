#!/usr/bin/env python3
"""Prepare and optionally publish the public R0 custom-weight repository."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "deployment/runtime_bundle_r0"
EXPORT = ROOT / "outputs/hf-exports/yoonLM--mal2026-r0-ensemble-v1"
REPO_ID = "yoonLM/mal2026-r0-ensemble-v1"


class PublishError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise PublishError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def env_token() -> str:
    path = ROOT / ".env"
    need(path.is_file(), ".env is unavailable")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    token = values.get("HF_TOKEN", "")
    need(bool(token), "HF_TOKEN is unavailable")
    return token


def copy_adapter(source: Path, destination: Path, base_model: str) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=False)
    model_source = source / "adapter_model.safetensors"
    config_source = source / "adapter_config.json"
    need(model_source.is_file() and config_source.is_file(), f"adapter is incomplete: {source}")
    os.link(model_source, destination / model_source.name)
    config = json.loads(config_source.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = base_model
    (destination / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {
        "adapter_model_sha256": file_sha256(destination / model_source.name),
        "adapter_config_sha256": file_sha256(destination / "adapter_config.json"),
    }


def prepare() -> dict[str, Any]:
    complete_path = BUNDLE / "bundle_complete.json"
    need(complete_path.is_file(), "R0 bundle completion is unavailable")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    need(complete.get("status") == "completed", "R0 bundle is incomplete")
    need(not EXPORT.exists(), "refusing to replace an existing HF export")
    EXPORT.mkdir(parents=True)
    shutil.copy2(ROOT / "deployment/hf_model_card.r0.md", EXPORT / "README.md")
    shutil.copy2(BUNDLE / "manifest.json", EXPORT / "runtime_manifest.json")
    shutil.copy2(complete_path, EXPORT / "bundle_complete.json")
    shutil.copy2(BUNDLE / "score/base/LICENSE", EXPORT / "LICENSE")

    inventory: dict[str, Any] = {
        "schema_version": "mal2026-r0-hf-artifact-v1",
        "repo_id": REPO_ID,
        "visibility": "public",
        "contains_base_model_weights": False,
        "contains_restricted_rows_text_ids_predictions_or_credentials": False,
        "score_base": {
            "model_id": "Qwen/Qwen3-Embedding-8B",
            "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        },
        "rationale_base": {
            "model_id": "skt/A.X-4.0-Light",
            "revision": "ba21c20ea1b31ded1ec3e2fb432335077dc4be98",
        },
        "score_epochs": {},
        "rationale_adapters": {},
        "metrics": {
            "continuous_macro_rmse": 0.5582937519204271,
            "continuous_macro_spearman": 0.6441959864775355,
            "integer_macro_rmse": 0.6158981882311673,
            "integer_macro_spearman": 0.5681974394968795,
        },
    }
    for epoch in range(1, 5):
        name = f"epoch_{epoch:02d}"
        adapter_info = copy_adapter(
            BUNDLE / "score/adapters" / name,
            EXPORT / "score/adapters" / name,
            "Qwen/Qwen3-Embedding-8B",
        )
        head_source = BUNDLE / "score/heads" / f"{name}.safetensors"
        head_destination = EXPORT / "score/heads" / head_source.name
        head_destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(head_source, head_destination)
        inventory["score_epochs"][name] = {
            **adapter_info,
            "head_sha256": file_sha256(head_destination),
        }
    inventory["rationale_adapters"]["rank2_ax4_random1"] = copy_adapter(
        BUNDLE / "rationale/adapters/rank2_ax4_random1",
        EXPORT / "rationale/adapters/rank2_ax4_random1",
        "skt/A.X-4.0-Light",
    )
    inventory["rationale_adapters"]["final_dpo"] = copy_adapter(
        BUNDLE / "rationale/adapters/final_dpo",
        EXPORT / "rationale/adapters/final_dpo",
        "skt/A.X-4.0-Light",
    )
    (EXPORT / "artifact_manifest.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {
        "status": "prepared", "repo_id": REPO_ID, "path": str(EXPORT),
        "bytes": sum(path.stat().st_size for path in EXPORT.rglob("*") if path.is_file()),
        "artifact_manifest_sha256": file_sha256(EXPORT / "artifact_manifest.json"),
    }


def publish() -> dict[str, Any]:
    from huggingface_hub import HfApi

    need(EXPORT.is_dir() and (EXPORT / "artifact_manifest.json").is_file(), "HF export is not prepared")
    token = env_token()
    api = HfApi(token=token)
    identity = api.whoami()
    need(identity.get("name") == "yoonLM", "HF identity differs")
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
    api.update_repo_settings(repo_id=REPO_ID, repo_type="model", private=False)
    api.upload_large_folder(
        repo_id=REPO_ID, repo_type="model", folder_path=str(EXPORT),
        num_workers=4, print_report=False, print_report_every=60,
    )
    info = api.model_info(REPO_ID, files_metadata=True)
    need(info.private is False, "HF repository is unexpectedly private")
    return {
        "status": "published", "repo_id": REPO_ID, "private": info.private,
        "revision": info.sha, "files": len(info.siblings or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    try:
        report = publish() if args.publish else prepare()
    except Exception as exc:
        print(json.dumps({"status": "failed", "type": type(exc).__name__, "message": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
