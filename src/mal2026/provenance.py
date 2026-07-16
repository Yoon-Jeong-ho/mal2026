"""Aggregate-only run provenance and rank-zero W&B safety guards."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping


class TelemetrySafetyError(ValueError):
    """Attempted telemetry contains restricted text, labels, or artifacts."""


_FORBIDDEN_KEYS = frozenset({
    "essay", "prompt", "text", "id", "document_id", "feedback", "rationale", "raw_output",
    "tokens", "input_ids", "labels", "dataset", "api_key", "token", "password",
})


def _assert_aggregate_only(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TelemetrySafetyError(f"{path} uses a non-string key")
            if key.casefold() in _FORBIDDEN_KEYS:
                raise TelemetrySafetyError(f"{path}.{key} is restricted telemetry")
            _assert_aggregate_only(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        # Aggregate telemetry must be scalar or named aggregate mappings. Lists
        # are commonly accidental data tables, so reject them categorically.
        raise TelemetrySafetyError(f"{path} contains a list/tuple and may expose rows")
    elif value is None or isinstance(value, (str, int, float, bool)):
        return
    else:
        raise TelemetrySafetyError(f"{path} has unsupported telemetry type")


def aggregate_only_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    _assert_aggregate_only(payload)
    # round-trip prevents callers retaining non-JSON scalar subclasses.
    return json.loads(json.dumps(payload, sort_keys=True))


def git_provenance() -> dict[str, Any]:
    """Collect code state only; never inspect or hash restricted dataset content."""
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {
        "git_sha": run("git", "rev-parse", "HEAD"),
        "git_dirty": run("git", "status", "--porcelain") != "",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_run_manifest(
    *, run_id: str, config_hash: str, data_contract: Mapping[str, Any], command: str,
    output_path: str, extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a local-run manifest with hashes/counts, never raw records/text."""
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "config_hash": config_hash,
        "data_contract": aggregate_only_payload(data_contract),
        "command": command,
        "output_path": output_path,
        "provenance": git_provenance(),
    }
    if extra:
        manifest["extra"] = aggregate_only_payload(extra)
    return manifest


def write_local_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Write only underneath ignored outputs/runs/<run-id>; never overwrite."""
    destination = Path(path)
    parts = destination.resolve().parts
    if "outputs" not in parts or "runs" not in parts:
        raise TelemetrySafetyError("run manifests must be written under ignored outputs/runs")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing run manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(aggregate_only_payload(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wandb_rank_zero_init(*, project: str, run_id: str, config: Mapping[str, Any], rank: int) -> Any | None:
    """Start W&B only on rank zero and never enable artifact/model uploads."""
    if rank != 0:
        return None
    safe_config = aggregate_only_payload(config)
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    try:
        import wandb
    except ImportError as exc:  # clear preflight failure rather than silent telemetry loss
        raise RuntimeError("wandb is required for experiment telemetry") from exc
    return wandb.init(project=project, id=run_id, resume="never", config=safe_config)


def wandb_log_aggregates(run: Any | None, metrics: Mapping[str, Any], step: int) -> None:
    if run is None:
        return
    safe_metrics = aggregate_only_payload(metrics)
    run.log(safe_metrics, step=step)
