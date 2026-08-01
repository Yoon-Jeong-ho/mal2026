"""Contracts for train-only Terra/Luna rationale semantic features."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ID = "iterative-official-rationale-embeddings-v12-20260802-001"
SCHEMA_VERSION = "mal2026-iterative-official-rationale-embeddings-v12"
MODEL_ID = "Qwen/Qwen3-Embedding-8B"
MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
EMBEDDING_DIM = 4096
PROJECTION_DIM = 32
FEATURE_DIM = 201
PROJECTION_SEED = 2026080212
MAX_LENGTH = 2048
AXES = ("content", "organization", "expression")
SOURCES = ("terra", "luna")


class OfficialRationaleEmbeddingError(ValueError):
    """Raised when a target-blind embedding artifact differs."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialRationaleEmbeddingError(message)


def file_sha256(path: str | Path) -> str:
    value = Path(path)
    _need(value.is_file() and not value.is_symlink(), "artifact must be an ordinary file")
    digest = sha256()
    with value.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rademacher_projection() -> np.ndarray:
    """Return the fixed data-independent 4096x32 projection."""
    generator = np.random.default_rng(PROJECTION_SEED)
    signs = generator.integers(0, 2, size=(EMBEDDING_DIM, PROJECTION_DIM), dtype=np.int8)
    result = (2.0 * signs.astype(np.float32) - 1.0) / math.sqrt(PROJECTION_DIM)
    result.setflags(write=False)
    return result


def matrix_sha256(matrix: Any) -> str:
    value = np.asarray(matrix, dtype="<f4")
    _need(value.shape == (EMBEDDING_DIM, PROJECTION_DIM), "projection matrix shape differs")
    return sha256(value.tobytes(order="C")).hexdigest()


def _normalized(value: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    _need(math.isfinite(norm) and norm > 1e-12, f"{name} has zero or non-finite norm")
    return value / norm


def build_rationale_features(embeddings: Any, projection: Any | None = None) -> np.ndarray:
    """Build 201 target-blind features from [source, axis, candidate, 4096]."""
    values = np.asarray(embeddings, dtype=np.float32)
    _need(values.shape == (2, 3, 3, EMBEDDING_DIM) and np.isfinite(values).all(),
          "rationale embeddings must have shape [2,3,3,4096]")
    norms = np.linalg.norm(values, axis=-1)
    _need(np.all(np.abs(norms - 1.0) <= 2e-4), "rationale embeddings must be L2 normalized")
    matrix = rademacher_projection() if projection is None else np.asarray(projection, dtype=np.float32)
    _need(matrix.shape == (EMBEDDING_DIM, PROJECTION_DIM) and np.isfinite(matrix).all(),
          "projection matrix differs")
    features: list[np.ndarray] = []
    pairs = ((0, 1), (0, 2), (1, 2))
    for axis in range(3):
        terra, luna = values[0, axis], values[1, axis]
        terra_centroid = _normalized(terra.mean(0), "Terra centroid")
        luna_centroid = _normalized(luna.mean(0), "Luna centroid")
        pooled = ((terra_centroid + luna_centroid) * .5) @ matrix
        difference = (terra_centroid - luna_centroid) @ matrix
        terra_within = np.mean([float(terra[left] @ terra[right]) for left, right in pairs])
        luna_within = np.mean([float(luna[left] @ luna[right]) for left, right in pairs])
        cross = float(terra_centroid @ luna_centroid)
        features.extend((pooled, difference, np.asarray((terra_within, luna_within, cross), dtype=np.float32)))
    output = np.concatenate(features).astype(np.float32, copy=False)
    _need(output.shape == (FEATURE_DIM,) and np.isfinite(output).all(), "final rationale features differ")
    return output


@dataclass(frozen=True)
class RationaleFeatureRow:
    source_id: str
    features: tuple[float, ...]


def load_feature_artifact(manifest_path: str | Path, rows_path: str | Path,
                          *, expected_source_ids: Sequence[str] | None = None) -> tuple[Mapping[str, Any], tuple[RationaleFeatureRow, ...]]:
    """Load a merged restricted artifact and validate its public manifest."""
    manifest_file, rows_file = Path(manifest_path), Path(rows_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialRationaleEmbeddingError("feature manifest is unreadable") from exc
    required = {
        "schema_version": SCHEMA_VERSION, "status": "completed", "run_id": RUN_ID,
        "split_role": "train", "records": 2000, "feature_dim": FEATURE_DIM,
        "embedding_dim": EMBEDDING_DIM, "projection_dim": PROJECTION_DIM,
        "projection_seed": PROJECTION_SEED, "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "validation_loaded": False,
        "candidate_score_in_embedding_text": False,
    }
    _need(isinstance(manifest, dict) and all(manifest.get(key) == value for key, value in required.items()),
          "feature manifest contract differs")
    _need(manifest.get("projection_matrix_sha256") == matrix_sha256(rademacher_projection()),
          "projection matrix binding differs")
    _need(rows_file.is_file() and not rows_file.is_symlink()
          and file_sha256(rows_file) == manifest.get("feature_rows_sha256"), "feature rows checksum differs")
    rows: list[RationaleFeatureRow] = []
    seen: set[str] = set()
    with rows_file.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            raw = json.loads(line)
            _need(isinstance(raw, dict) and set(raw) == {"source_id", "features"},
                  f"feature row schema differs at line {line_number}")
            source_id, raw_features = raw["source_id"], raw["features"]
            _need(isinstance(source_id, str) and source_id and source_id not in seen,
                  f"feature source ID differs at line {line_number}")
            _need(isinstance(raw_features, list) and len(raw_features) == FEATURE_DIM,
                  f"feature dimensions differ at line {line_number}")
            features = tuple(float(item) for item in raw_features)
            _need(all(math.isfinite(item) for item in features), f"non-finite feature at line {line_number}")
            seen.add(source_id); rows.append(RationaleFeatureRow(source_id, features))
    _need(len(rows) == 2000, "feature row count differs")
    if expected_source_ids is not None:
        _need(tuple(row.source_id for row in rows) == tuple(expected_source_ids),
              "feature source order differs")
    return manifest, tuple(rows)


__all__ = [
    "AXES", "EMBEDDING_DIM", "FEATURE_DIM", "MAX_LENGTH", "MODEL_ID", "MODEL_REVISION",
    "OfficialRationaleEmbeddingError", "PROJECTION_DIM", "PROJECTION_SEED", "RUN_ID",
    "RationaleFeatureRow", "SCHEMA_VERSION", "SOURCES", "build_rationale_features",
    "file_sha256", "load_feature_artifact", "matrix_sha256", "rademacher_projection",
]
