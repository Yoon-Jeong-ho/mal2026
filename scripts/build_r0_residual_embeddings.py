#!/usr/bin/env python3
"""Build restricted frozen Qwen3 embeddings for the R0 residual trainer.

The train artifact joins the exact five-fold OOF R0 predictions to the
canonical train essays and the fixed ``rank2_ax4_random1`` rationales.  The
validation artifact uses the already completed held-out R0 ensemble.  Shards
are deliberately single-GPU jobs; merging is CPU-only and verifies all input,
row, and shard bindings before emitting the exact ``EmbeddingArtifactManifest``
and JSONL row contract consumed by :mod:`mal2026.r0_ordinal_residual`.

Row-level outputs, including embeddings and identifiers, never leave the
ignored restricted data root.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import (  # noqa: E402
    EXPECTED_ESSAYS,
    SOURCE_SHA256,
    TRAIN_SOURCE,
    VALIDATION_SOURCE,
)
from mal2026.r0_ordinal_residual import (  # noqa: E402
    AXES,
    EMBEDDING_SCHEMA_VERSION,
    GOLD_LABEL_POLICY,
    EmbeddingArtifactManifest,
    load_embedding_artifact,
)
from mal2026.rlaif_qwen3_embedding import (  # noqa: E402
    MODEL_ID,
    MODEL_PATH,
    MODEL_REVISION,
    RATIONALE_SOURCE,
)
from mal2026.rlaif_top3_encoder import (  # noqa: E402
    _input_text,
    _load_generated_rationales,
    generation_dir,
)


RESTRICTED_ROOT = (
    ROOT / "data" / "processed" / "restricted" /
    "r0_ordinal_residual_embeddings_v1"
)
OOF_ROOT = ROOT / "data" / "processed" / "restricted" / "r0_exact_oof_v1"
OOF_PUBLIC_ROOT = ROOT / "outputs" / "r0-exact-oof-v1"
VALIDATION_PREDICTIONS = (
    ROOT / "data" / "processed" / "restricted" /
    "official_prompt_alignment_v1" / "score_predictions" /
    "official-score-r0-ensemble-full-20260727-002" /
    "r0_prediction_ensemble.jsonl"
)
VALIDATION_AGGREGATE = (
    ROOT / "outputs" / "official-prompt-alignment-v1" / "score-metrics" /
    "official-score-r0-ensemble-full-20260727-002" / "aggregate_metrics.json"
)
SHARD_COUNT = 4
MAX_LENGTH = 2048
RUN_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{2,127}")
FINAL_ROW_FIELDS = {
    "source_id", "group_id", "shared_embedding",
    "base_continuous_prediction", "raw_continuous_gold", "oof_fold",
}


class FrozenEmbeddingArtifactError(RuntimeError):
    """Raised before an ambiguous, leaky, or unbound artifact is emitted."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenEmbeddingArtifactError(message)


def file_sha256(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"ordinary file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenEmbeddingArtifactError(f"{label} is unreadable") from exc
    need(isinstance(value, dict), f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    need(isinstance(value, str) and bool(value.strip()), f"{label} must be nonblank")
    return value


def _axis_scores(value: Any, label: str) -> dict[str, float]:
    need(isinstance(value, dict) and set(value) == set(AXES), f"{label} axes differ")
    result: dict[str, float] = {}
    for axis in AXES:
        raw = value[axis]
        need(type(raw) in {int, float}, f"{label}.{axis} must be numeric")
        score = float(raw)
        need(math.isfinite(score) and 1.0 <= score <= 5.0,
             f"{label}.{axis} must be finite within [1,5]")
        result[axis] = score
    return result


@dataclass(frozen=True)
class SourceRow:
    source_id: str
    document_id: str
    prompt: str
    essay: str
    raw_gold: dict[str, float]


@dataclass(frozen=True)
class BaseRow:
    source_id: str
    continuous_prediction: dict[str, float]
    oof_fold: int | None
    reference_score: dict[str, float] | None = None


def canonical_path(split: str) -> Path:
    need(split in {"train", "validation"}, "split must be train or validation")
    return TRAIN_SOURCE if split == "train" else VALIDATION_SOURCE


def load_canonical_rows(split: str, *, path: Path | None = None,
                        enforce_project_binding: bool = True) -> list[SourceRow]:
    """Read canonical rows while retaining the true document grouping key."""
    source = path or canonical_path(split)
    if enforce_project_binding:
        need(source.resolve() == canonical_path(split).resolve(), "canonical path differs")
        need(file_sha256(source) == SOURCE_SHA256[split], "canonical checksum differs")
    result: list[SourceRow] = []
    seen_ids: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            need(isinstance(raw, dict) and set(raw) == {
                "id", "document_id", "prompt_num", "prompt", "essay", "score"
            }, f"canonical schema differs at line {line_number}")
            identifier = _text(raw["id"], "source_id")
            document_id = _text(raw["document_id"], "document_id")
            need(identifier not in seen_ids, "canonical source IDs must be unique")
            seen_ids.add(identifier)
            score = raw["score"]
            need(isinstance(score, dict) and tuple(score) == (*AXES, "average"),
                 "canonical score schema differs")
            # The essay-level average is checked structurally but deliberately
            # neither parsed nor copied into any residual target.
            gold = _axis_scores({axis: score[axis] for axis in AXES}, "raw gold")
            result.append(SourceRow(
                identifier, document_id, _text(raw["prompt"], "prompt"),
                _text(raw["essay"], "essay"), gold,
            ))
    need(bool(result), "canonical source is empty")
    if enforce_project_binding:
        need(len(result) == EXPECTED_ESSAYS[split], "canonical row count differs")
    return result


def load_base_rows(split: str, path: Path) -> list[BaseRow]:
    """Parse exact OOF train or held-out validation predictions only."""
    result: list[BaseRow] = []
    seen: set[str] = set()
    expected = (
        {"source_id", "fold", "continuous_prediction",
         "half_up_integer_prediction", "reference_score"}
        if split == "train" else
        {"source_id", "continuous_prediction", "emitted_integer_prediction"}
    )
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            need(isinstance(raw, dict) and set(raw) == expected,
                 f"base prediction schema differs at line {line_number}")
            identifier = _text(raw["source_id"], "base source_id")
            need(identifier not in seen, "base prediction source IDs must be unique")
            seen.add(identifier)
            prediction = _axis_scores(raw["continuous_prediction"], "base prediction")
            if split == "train":
                fold = raw["fold"]
                need(type(fold) is int and 0 <= fold < 5, "OOF fold must be 0..4")
                reference = _axis_scores(raw["reference_score"], "OOF reference score")
            else:
                fold = None
                reference = None
            result.append(BaseRow(identifier, prediction, fold, reference))
    need(bool(result), "base prediction artifact is empty")
    return result


def join_inputs(
    split: str,
    sources: Sequence[SourceRow],
    rationales: Mapping[str, Mapping[str, str]],
    predictions: Sequence[BaseRow],
) -> list[dict[str, Any]]:
    """Perform a one-to-one exact source join and expose no average target."""
    source_ids = [row.source_id for row in sources]
    need(len(source_ids) == len(set(source_ids)), "canonical IDs are not unique")
    need(set(source_ids) == set(rationales), "rationale IDs do not exactly match canonical IDs")
    by_prediction = {row.source_id: row for row in predictions}
    need(len(by_prediction) == len(predictions) and set(source_ids) == set(by_prediction),
         "base prediction IDs do not exactly match canonical IDs")
    result: list[dict[str, Any]] = []
    for source in sources:
        rationale = rationales[source.source_id]
        need(isinstance(rationale, Mapping) and set(rationale) == set(AXES),
             "rationale axes differ")
        need(all(isinstance(rationale[axis], str) and rationale[axis].strip() for axis in AXES),
             "rationale text must be nonblank")
        base = by_prediction[source.source_id]
        need((base.oof_fold is not None) == (split == "train"),
             "OOF fold/split contract differs")
        if split == "train":
            need(base.reference_score is not None and all(
                base.reference_score[axis] == source.raw_gold[axis] for axis in AXES
            ), "OOF reference scores do not exactly match canonical raw gold")
        result.append({
            "source_id": source.source_id,
            "group_id": source.document_id,
            "text": _input_text(source.prompt, source.essay, rationale),
            "base_continuous_prediction": dict(base.continuous_prediction),
            "raw_continuous_gold": dict(source.raw_gold),
            "oof_fold": base.oof_fold,
        })
    if split == "train":
        need({row["oof_fold"] for row in result} == set(range(5)),
             "joined train rows must contain all five folds")
    return result


def shard_bounds(total: int, index: int, count: int = SHARD_COUNT) -> tuple[int, int]:
    need(count == SHARD_COUNT, "the artifact contract requires exactly four shards")
    need(type(index) is int and 0 <= index < count, "shard index is out of range")
    need(type(total) is int and total >= count, "row count is too small for four shards")
    return total * index // count, total * (index + 1) // count


def last_nonpad_normalized(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Select the actual last non-pad token and L2-normalize in float32."""
    import torch
    import torch.nn.functional as functional

    need(last_hidden_state.ndim == 3 and attention_mask.ndim == 2,
         "pooling tensors have invalid rank")
    need(tuple(last_hidden_state.shape[:2]) == tuple(attention_mask.shape),
         "pooling tensor shapes differ")
    mask = attention_mask.to(dtype=torch.bool)
    need(bool(mask.any(dim=1).all().item()), "every sequence requires a non-pad token")
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    last = positions.masked_fill(~mask, -1).max(dim=1).values
    batch = torch.arange(mask.shape[0], device=mask.device)
    pooled = last_hidden_state[batch, last].float()
    return functional.normalize(pooled, p=2, dim=1)


def embed_texts(texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
    """Run the immutable local public encoder with every parameter frozen."""
    need(bool(texts) and batch_size > 0, "embedding batch is empty or invalid")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("embedding generation requires the existing project environment") from exc

    need(torch.cuda.is_available() and torch.cuda.device_count() == 1,
         "one and only one visible CUDA GPU is required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), revision=MODEL_REVISION, local_files_only=True,
        trust_remote_code=False, use_fast=True,
    )
    if tokenizer.pad_token is None:
        need(tokenizer.eos_token is not None, "Qwen3 tokenizer lacks pad/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        str(MODEL_PATH), revision=MODEL_REVISION, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embeddings: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                list(texts[start:start + batch_size]), padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            batch = {key: value.to("cuda:0") for key, value in batch.items()}
            output = model(**batch, return_dict=True)
            pooled = last_nonpad_normalized(output.last_hidden_state, batch["attention_mask"])
            embeddings.extend(pooled.cpu().tolist())
    need(len(embeddings) == len(texts), "embedding output count differs")
    return embeddings


def prediction_path(split: str, prediction_run_id: str | None) -> Path:
    if split == "train":
        need(isinstance(prediction_run_id, str) and RUN_PATTERN.fullmatch(prediction_run_id),
             "train requires a valid exact-OOF prediction run ID")
        return OOF_ROOT / prediction_run_id / "merged" / "oof_predictions.jsonl"
    need(prediction_run_id is None, "validation uses the fixed held-out R0 ensemble")
    return VALIDATION_PREDICTIONS


def validate_prediction_provenance(split: str, prediction_run_id: str | None,
                                   path: Path) -> dict[str, Any]:
    checksum = file_sha256(path)
    if split == "train":
        result_path = OOF_PUBLIC_ROOT / str(prediction_run_id) / "merged" / "result.json"
        result = read_json(result_path, "exact OOF merge result")
        need(result.get("schema_version") == "mal2026-r0-exact-oof-merge-result-v1" and
             result.get("status") == "completed" and result.get("folds") == 5 and
             result.get("records") == EXPECTED_ESSAYS["train"],
             "exact OOF merge result is incomplete")
        need(result.get("merged_restricted_oof_sha256") == checksum and
             result.get("base_prediction_origin_oof") == 1,
             "exact OOF prediction binding differs")
        return {"result_path": str(result_path.resolve()),
                "result_sha256": file_sha256(result_path), "prediction_sha256": checksum}
    aggregate = read_json(VALIDATION_AGGREGATE, "held-out R0 aggregate")
    candidates = aggregate.get("results")
    matching = [item for item in candidates if isinstance(item, dict) and
                item.get("candidate") == "r0_prediction_ensemble"] if isinstance(candidates, list) else []
    need(aggregate.get("status") == "completed" and aggregate.get("validation_rows") == EXPECTED_ESSAYS["validation"] and
         len(matching) == 1 and matching[0].get("prediction_sha256") == checksum,
         "held-out R0 prediction binding differs")
    return {"result_path": str(VALIDATION_AGGREGATE.resolve()),
            "result_sha256": file_sha256(VALIDATION_AGGREGATE),
            "prediction_sha256": checksum}


def artifact_split_root(artifact_run_id: str, split: str) -> Path:
    need(RUN_PATTERN.fullmatch(artifact_run_id) is not None, "artifact run ID differs")
    need(split in {"train", "validation"}, "split differs")
    path = RESTRICTED_ROOT / artifact_run_id / split
    need(path.resolve().is_relative_to(RESTRICTED_ROOT.resolve()),
         "artifact path escaped restricted storage")
    return path


def _write_json_fresh(path: Path, payload: Mapping[str, Any]) -> str:
    need(not path.exists(), f"fresh output required: {path.name}")
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
    os.chmod(path, 0o600)
    return file_sha256(path)


def _write_jsonl_fresh(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    need(not path.exists(), f"fresh output required: {path.name}")
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    return file_sha256(path)


def build_shard(args: argparse.Namespace) -> dict[str, Any]:
    need(args.shard_count == SHARD_COUNT, "exactly four shards are required")
    need(args.physical_gpu in {0, 1, 2, 3}, "physical GPU must be within 0..3")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.physical_gpu),
         "CUDA visibility must equal the declared physical GPU")
    need(MODEL_PATH.is_dir() and not MODEL_PATH.is_symlink(),
         "immutable local Qwen3 snapshot is unavailable")
    base_path = prediction_path(args.split, args.prediction_run_id)
    base_provenance = validate_prediction_provenance(
        args.split, args.prediction_run_id, base_path
    )
    rationale_root = generation_dir(RATIONALE_SOURCE, args.split, "full")
    rationale_path = rationale_root / "generated_rationales.jsonl"
    rationales = _load_generated_rationales(
        rationale_root, RATIONALE_SOURCE, args.split, "full",
        EXPECTED_ESSAYS[args.split],
    )
    sources = load_canonical_rows(args.split)
    bases = load_base_rows(args.split, base_path)
    joined = join_inputs(args.split, sources, rationales, bases)
    start, stop = shard_bounds(len(joined), args.shard_index, args.shard_count)
    selected = joined[start:stop]
    embeddings = embed_texts([row["text"] for row in selected], batch_size=args.batch_size)
    need(bool(embeddings) and all(len(value) == len(embeddings[0]) for value in embeddings),
         "embedding dimensions differ")
    rows = [{
        "source_id": row["source_id"],
        "group_id": row["group_id"],
        "shared_embedding": embedding,
        "base_continuous_prediction": row["base_continuous_prediction"],
        "raw_continuous_gold": row["raw_continuous_gold"],
        "oof_fold": row["oof_fold"],
    } for row, embedding in zip(selected, embeddings, strict=True)]
    output = artifact_split_root(args.artifact_run_id, args.split) / "shards" / f"shard-{args.shard_index:02d}"
    need(not output.exists(), "shard output must be fresh")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows_path = output / "rows.jsonl"
    rows_sha = _write_jsonl_fresh(rows_path, rows)
    metadata = {
        "schema_version": "mal2026-r0-frozen-embedding-shard-v1",
        "status": "completed", "artifact_run_id": args.artifact_run_id,
        "split_role": args.split, "shard_index": args.shard_index,
        "shard_count": args.shard_count, "range_start": start, "range_stop": stop,
        "records": len(rows), "embedding_dim": len(embeddings[0]),
        "embedding_model_id": MODEL_ID, "embedding_model_revision": MODEL_REVISION,
        "embedding_model_path": str(MODEL_PATH.resolve()),
        "embedding_model_config_sha256": file_sha256(MODEL_PATH / "config.json"),
        "embedding_source": "public", "embedding_frozen": True,
        "pooling": "last_nonpad_l2_normalized", "max_length": MAX_LENGTH,
        "canonical_source_sha256": SOURCE_SHA256[args.split],
        "rationale_rows_sha256": file_sha256(rationale_path),
        "base_prediction_provenance": base_provenance,
        "rows_sha256": rows_sha,
        "contains_average_target": False,
    }
    metadata_path = output / "metadata.json"
    metadata_sha = _write_json_fresh(metadata_path, metadata)
    return {"status": "completed", "split": args.split, "shard": args.shard_index,
            "records": len(rows), "rows_sha256": rows_sha,
            "metadata_sha256": metadata_sha}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            need(isinstance(raw, dict) and set(raw) == FINAL_ROW_FIELDS,
                 f"shard row schema differs at line {line_number}")
            rows.append(raw)
    return rows


def merge_shards(args: argparse.Namespace) -> dict[str, Any]:
    need(args.shard_count == SHARD_COUNT, "exactly four shards are required")
    split_root = artifact_split_root(args.artifact_run_id, args.split)
    sources = load_canonical_rows(args.split)
    expected_ids = [row.source_id for row in sources]
    expected_documents = {row.source_id: row.document_id for row in sources}
    common: dict[str, Any] | None = None
    all_rows: list[dict[str, Any]] = []
    shard_bindings: list[dict[str, Any]] = []
    common_keys = (
        "artifact_run_id", "split_role", "shard_count", "embedding_dim",
        "embedding_model_id", "embedding_model_revision", "embedding_model_path",
        "embedding_model_config_sha256", "embedding_source", "embedding_frozen",
        "pooling", "max_length", "canonical_source_sha256",
        "rationale_rows_sha256", "base_prediction_provenance",
        "contains_average_target",
    )
    for index in range(SHARD_COUNT):
        root = split_root / "shards" / f"shard-{index:02d}"
        metadata_path, rows_path = root / "metadata.json", root / "rows.jsonl"
        metadata = read_json(metadata_path, f"shard {index} metadata")
        need(metadata.get("schema_version") == "mal2026-r0-frozen-embedding-shard-v1" and
             metadata.get("status") == "completed" and metadata.get("shard_index") == index,
             f"shard {index} identity differs")
        need(metadata.get("rows_sha256") == file_sha256(rows_path),
             f"shard {index} row checksum differs")
        current = {key: metadata.get(key) for key in common_keys}
        common = current if common is None else common
        need(current == common, "shard provenance differs")
        start, stop = shard_bounds(len(sources), index, SHARD_COUNT)
        need((metadata.get("range_start"), metadata.get("range_stop"), metadata.get("records")) ==
             (start, stop, stop - start), f"shard {index} range differs")
        rows = _read_rows(rows_path)
        need(len(rows) == stop - start and [row["source_id"] for row in rows] == expected_ids[start:stop],
             f"shard {index} source population or order differs")
        all_rows.extend(rows)
        shard_bindings.append({"shard_index": index,
                               "metadata_sha256": file_sha256(metadata_path),
                               "rows_sha256": file_sha256(rows_path),
                               "records": len(rows)})
    need(common is not None and len(all_rows) == len(sources), "merged row count differs")
    seen: set[str] = set()
    for row in all_rows:
        identifier = row["source_id"]
        need(isinstance(identifier, str) and identifier not in seen, "merged source IDs differ")
        seen.add(identifier)
        need(row["group_id"] == expected_documents[identifier],
             "group_id must equal canonical document_id")
        embedding = row["shared_embedding"]
        need(isinstance(embedding, list) and len(embedding) == common["embedding_dim"] and
             all(type(value) in {int, float} and math.isfinite(float(value)) for value in embedding),
             "merged embedding differs")
        norm = math.sqrt(sum(float(value) ** 2 for value in embedding))
        need(abs(norm - 1.0) <= 2e-4, "embedding is not L2 normalized")
        _axis_scores(row["base_continuous_prediction"], "merged base prediction")
        _axis_scores(row["raw_continuous_gold"], "merged raw gold")
        need("average" not in row["raw_continuous_gold"], "average target is forbidden")
        if args.split == "train":
            need(type(row["oof_fold"]) is int and 0 <= row["oof_fold"] < 5,
                 "train fold differs")
        else:
            need(row["oof_fold"] is None, "validation fold must be null")
    if args.split == "train":
        need({row["oof_fold"] for row in all_rows} == set(range(5)),
             "merged train artifact lacks an OOF fold")

    merged = split_root / "merged"
    need(not merged.exists(), "merged output must be fresh")
    merged.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows_path = merged / "rows.jsonl"
    rows_sha = _write_jsonl_fresh(rows_path, all_rows)
    manifest_payload = {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "split_role": args.split,
        "base_prediction_origin": "oof" if args.split == "train" else "held_out",
        "base_model_fit_excludes_split": True,
        "evaluation_only": args.split == "validation",
        "gold_label_policy": GOLD_LABEL_POLICY,
        "contains_average_target": False,
        "embedding_model_id": MODEL_ID,
        "embedding_model_revision": MODEL_REVISION,
        "embedding_source": "public",
        "embedding_frozen": True,
        "embedding_dim": common["embedding_dim"],
        "fold_count": 5 if args.split == "train" else 0,
        "rows_sha256": rows_sha,
    }
    # Validate the exact manifest schema before and after persistence.
    EmbeddingArtifactManifest(**manifest_payload).validate()
    manifest_path = merged / "manifest.json"
    manifest_sha = _write_json_fresh(manifest_path, manifest_payload)
    loaded_manifest, loaded_rows = load_embedding_artifact(manifest_path, rows_path)
    need(loaded_manifest.rows_sha256 == rows_sha and len(loaded_rows) == len(all_rows),
         "residual trainer round-trip verification failed")
    provenance = {
        "schema_version": "mal2026-r0-frozen-embedding-provenance-v1",
        "status": "completed", "artifact_run_id": args.artifact_run_id,
        "split_role": args.split, "records": len(all_rows),
        "row_level_storage": "restricted_only", "shards": shard_bindings,
        "manifest_sha256": manifest_sha, "rows_sha256": rows_sha,
        "input_provenance": {key: common[key] for key in common_keys
                             if key not in {"artifact_run_id", "split_role", "shard_count", "embedding_dim"}},
        "script_sha256": file_sha256(Path(__file__).resolve()),
    }
    provenance_path = merged / "provenance.json"
    provenance_sha = _write_json_fresh(provenance_path, provenance)
    return {"status": "completed", "split": args.split, "records": len(all_rows),
            "embedding_dim": common["embedding_dim"], "rows_sha256": rows_sha,
            "manifest_sha256": manifest_sha, "provenance_sha256": provenance_sha}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create four frozen Qwen3 embedding shards or verify and merge them."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard", help="run one single-GPU embedding shard")
    shard.add_argument("--artifact-run-id", required=True)
    shard.add_argument("--split", required=True, choices=("train", "validation"))
    shard.add_argument("--prediction-run-id")
    shard.add_argument("--shard-index", required=True, type=int, choices=range(SHARD_COUNT))
    shard.add_argument("--shard-count", type=int, default=SHARD_COUNT, choices=(SHARD_COUNT,))
    shard.add_argument("--physical-gpu", required=True, type=int, choices=(0, 1, 2, 3))
    shard.add_argument("--batch-size", type=int, default=4)
    merge = subparsers.add_parser("merge", help="CPU-only four-shard verification and merge")
    merge.add_argument("--artifact-run-id", required=True)
    merge.add_argument("--split", required=True, choices=("train", "validation"))
    merge.add_argument("--shard-count", type=int, default=SHARD_COUNT, choices=(SHARD_COUNT,))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = build_shard(args) if args.command == "shard" else merge_shards(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
