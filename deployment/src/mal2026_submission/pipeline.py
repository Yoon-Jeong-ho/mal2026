"""Pipeline interface and production loader used by the HTTP server."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class Completion:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Pipeline(Protocol):
    served_model_name: str

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
        stop: str | list[str] | None,
    ) -> Completion: ...


def load_pipeline() -> Pipeline:
    backend = os.environ.get("MAL2026_BACKEND", "production")
    if backend != "production":
        raise RuntimeError("only the production backend is permitted in a submission image")
    root = Path(os.environ.get("MAL2026_BUNDLE_ROOT", "/opt/mal2026/models"))
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("runtime manifest is unreadable") from exc
    if manifest.get("pipeline_kind") in {
        "legacy_r0_prediction_ensemble_to_dpo_rationale",
        "legacy_r0_prediction_ensemble_to_latest_score_blind_rationale",
    }:
        from .production_r0 import R0EnsemblePipeline

        return R0EnsemblePipeline.from_environment()
    from .production import ProductionPipeline

    return ProductionPipeline.from_environment()
