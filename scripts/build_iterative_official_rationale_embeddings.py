#!/usr/bin/env python3
"""Build the restricted V12 Terra/Luna rationale-only feature artifact."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import SOURCE_SHA256, load_writing_rows  # noqa: E402
from mal2026.iterative_official_dual_agent_data import load_dual_candidates  # noqa: E402
from mal2026.iterative_official_rationale_embedding_data import (  # noqa: E402
    AXES, EMBEDDING_DIM, FEATURE_DIM, MAX_LENGTH, MODEL_ID, MODEL_REVISION,
    PROJECTION_DIM, PROJECTION_SEED, RUN_ID, SCHEMA_VERSION, SOURCES,
    build_rationale_features, file_sha256, matrix_sha256, rademacher_projection,
)


MODEL_PATH = ROOT / "outputs/model-cache" / f"Qwen--Qwen3-Embedding-8B-{MODEL_REVISION}"
TERRA_ROOT = ROOT / "data/processed/restricted/official_openai_candidates_v1/official-openai-candidates-v1-train3-20260727-001"
LUNA_ROOT = ROOT / "data/processed/restricted/official_openai_candidates_v1/official-openai-candidates-luna-v1-train3-20260802-001"
RESTRICTED_BASE = ROOT / "data/processed/restricted/iterative_official_rationale_embeddings_v12"
PUBLIC_BASE = ROOT / "outputs/iterative-official-rationale-embeddings-v12"
SHARD_COUNT = 4
SMOKE_RECORDS = 2
RENDER_CONTRACT = {
    "kind": "participant_axis_rationale_text_alone_v1", "source_order": list(SOURCES),
    "candidate_order": [1, 2, 3], "axis_order": list(AXES),
    "essay_included": False, "prompt_included": False, "participant_score_included": False,
    "gold_included": False,
}
AUTHORIZED_GPUS = (0, 1, 2, 3)


class BuildError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def _json_hash(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json_fresh(path: Path, value: Mapping[str, Any], *, restricted: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if restricted:
        os.chmod(path, 0o600)


def _write_jsonl_fresh(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    return file_sha256(path)


def _append_ledger(event: str, **fields: Any) -> None:
    path = PUBLIC_BASE / RUN_ID / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "schema_version": "mal2026-iterative-official-rationale-embeddings-ledger-event-v12",
            "run_id": RUN_ID, "at_epoch": time.time(), "event": event, **fields,
        }, sort_keys=True) + "\n")


def _gpu_preflight(gpus: Sequence[int]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    snapshots: list[Mapping[str, Any]] = []
    conflicts: list[Mapping[str, Any]] = []
    for gpu in gpus:
        need(gpu in AUTHORIZED_GPUS, "GPU preflight escaped authorized scope 0..3")
        state = subprocess.run(
            ["nvidia-smi", "-i", str(gpu),
             "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout.strip()
        processes = subprocess.run(
            ["nvidia-smi", "-i", str(gpu),
             "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout.strip()
        snapshot = {"gpu": gpu, "state": state, "compute_processes": processes or None}
        snapshots.append(snapshot)
        if processes:
            conflicts.append({"gpu": gpu, "compute_processes": processes})
    return snapshots, conflicts


def shard_bounds(total: int, shard: int) -> tuple[int, int]:
    need(total == 2000 and 0 <= shard < SHARD_COUNT, "V12 requires 2,000 rows and shard 0..3")
    return total * shard // SHARD_COUNT, total * (shard + 1) // SHARD_COUNT


def _manifest_metadata(path: Path, expected_model: str) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    need(raw.get("status") == "validated" and raw.get("model") == expected_model,
         "candidate manifest is not validated or model-bound")
    prompt_hash = raw.get("official_system_prompt_sha256")
    need(isinstance(prompt_hash, str) and len(prompt_hash) == 64, "official prompt hash is unavailable")
    return {"manifest_sha256": file_sha256(path), "official_system_prompt_sha256": prompt_hash}


def load_inputs(args: argparse.Namespace) -> tuple[
    list[str], Mapping[str, Mapping[str, Mapping[int, Mapping[str, str]]]], Mapping[str, Any]
]:
    writings = load_writing_rows("train", include_scores=False)
    source_ids = [row.identifier for row in writings]
    need(len(source_ids) == 2000 and len(set(source_ids)) == 2000, "canonical train population differs")
    essay_hashes = {row.identifier: sha256(row.essay.encode("utf-8")).hexdigest() for row in writings}
    candidates, provenance = load_dual_candidates(
        args.terra_manifest, args.terra_candidates, args.luna_manifest, args.luna_candidates,
        essay_sha256_by_source=essay_hashes,
    )
    grouped: dict[str, dict[str, dict[int, Mapping[str, str]]]] = {
        source: {source_id: {} for source_id in source_ids} for source in SOURCES
    }
    for candidate in candidates:
        grouped[candidate.agent_source][candidate.source_id][candidate.candidate_number] = candidate.rationales
    for source in SOURCES:
        need(all(set(grouped[source][source_id]) == {1, 2, 3} for source_id in source_ids),
             f"{source} rationale coverage differs")
    terra_meta = _manifest_metadata(Path(args.terra_manifest), "gpt-5.6-terra")
    luna_meta = _manifest_metadata(Path(args.luna_manifest), "gpt-5.6-luna")
    need(terra_meta["official_system_prompt_sha256"] == luna_meta["official_system_prompt_sha256"],
         "Terra/Luna official prompt hashes differ")
    bindings = {
        "canonical_train_sha256": SOURCE_SHA256["train"],
        "candidate_provenance": provenance,
        "terra_manifest_sha256": terra_meta["manifest_sha256"],
        "luna_manifest_sha256": luna_meta["manifest_sha256"],
        "official_system_prompt_sha256": terra_meta["official_system_prompt_sha256"],
        "render_contract_sha256": _json_hash(RENDER_CONTRACT),
    }
    return source_ids, grouped, bindings


def ordered_rationale_texts(grouped: Mapping[str, Mapping[str, Mapping[int, Mapping[str, str]]]],
                            source_ids: Sequence[str]) -> list[str]:
    """Render only rationale strings in the exact registered order."""
    texts = []
    for source in SOURCES:
        for source_id in source_ids:
            for candidate in (1, 2, 3):
                rationale = grouped[source][source_id][candidate]
                need(set(rationale) == set(AXES), "participant rationale axes differ")
                for axis in AXES:
                    text = rationale[axis]
                    need(isinstance(text, str) and bool(text.strip()), "participant rationale is blank")
                    texts.append(text.strip())
    return texts


def _load_encoder() -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    need(os.environ.get("CUDA_VISIBLE_DEVICES") is not None and torch.cuda.is_available()
         and torch.cuda.device_count() == 1, "exactly one declared visible GPU is required")
    need(MODEL_PATH.is_dir() and not MODEL_PATH.is_symlink(), "frozen Qwen snapshot is unavailable")
    config_path = MODEL_PATH / "config.json"
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), revision=MODEL_REVISION,
                                               local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        need(tokenizer.eos_token is not None, "tokenizer lacks pad/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(str(MODEL_PATH), revision=MODEL_REVISION, local_files_only=True,
                                      trust_remote_code=False, torch_dtype=torch.bfloat16,
                                      low_cpu_mem_usage=True, device_map={"": "cuda:0"})
    need(getattr(model.config, "hidden_size", None) == EMBEDDING_DIM, "Qwen hidden dimension differs")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model, file_sha256(config_path)


def _last_nonpad(output: Any, mask: Any) -> Any:
    import torch
    import torch.nn.functional as F
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    final = positions.masked_fill(~mask.bool(), -1).max(1).values
    need(bool((final >= 0).all().item()), "empty token sequence")
    pooled = output[torch.arange(len(mask), device=mask.device), final].float()
    return F.normalize(pooled, p=2, dim=1)


def embed_texts(texts: Sequence[str], tokenizer: Any, model: Any, *, batch_size: int) -> tuple[np.ndarray, Mapping[str, int]]:
    import torch
    need(bool(texts) and batch_size > 0, "embedding batch differs")
    arrays = []
    total_tokens = truncated = maximum = 0
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            lengths = tokenizer(chunk, add_special_tokens=True, truncation=False,
                                return_length=True)["length"]
            total_tokens += sum(int(value) for value in lengths)
            maximum = max(maximum, max(int(value) for value in lengths))
            truncated += sum(int(value) > MAX_LENGTH for value in lengths)
            batch = tokenizer(chunk, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            batch = {key: value.to("cuda:0") for key, value in batch.items()}
            output = model(**batch, return_dict=True).last_hidden_state
            arrays.append(_last_nonpad(output, batch["attention_mask"]).cpu().numpy().astype(np.float32))
    embeddings = np.concatenate(arrays)
    need(embeddings.shape == (len(texts), EMBEDDING_DIM) and np.isfinite(embeddings).all(),
         "embedding output differs")
    return embeddings, {"texts": len(texts), "total_tokens": total_tokens,
                        "max_tokens": maximum, "truncated_texts": truncated}


def _feature_rows(source_ids: Sequence[str], texts_embedding: np.ndarray) -> list[Mapping[str, Any]]:
    n = len(source_ids)
    expected = len(SOURCES) * n * 3 * len(AXES)
    need(texts_embedding.shape == (expected, EMBEDDING_DIM), "ordered embedding population differs")
    # Ordered source -> source_id -> candidate -> axis, then transpose to
    # [essay, source, axis, candidate, dimension].
    shaped = texts_embedding.reshape(2, n, 3, 3, EMBEDDING_DIM).transpose(1, 0, 3, 2, 4)
    projection = rademacher_projection()
    return [{"source_id": source_id, "features": build_rationale_features(shaped[index], projection).tolist()}
            for index, source_id in enumerate(source_ids)]


def _run_part(args: argparse.Namespace, *, smoke: bool) -> Mapping[str, Any]:
    physical = 0 if smoke else args.physical_gpu
    need(physical in range(4) and os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical),
         "CUDA_VISIBLE_DEVICES must equal the declared physical GPU")
    source_ids, grouped, bindings = load_inputs(args)
    if smoke:
        selected = source_ids[:SMOKE_RECORDS]
        output_root = RESTRICTED_BASE / f"{RUN_ID}-smoke"
        public_root = PUBLIC_BASE / f"{RUN_ID}-smoke"
        part = "smoke"
    else:
        start, stop = shard_bounds(len(source_ids), args.shard)
        selected = source_ids[start:stop]
        output_root = RESTRICTED_BASE / RUN_ID / "shards" / f"shard-{args.shard:02d}"
        public_root = PUBLIC_BASE / RUN_ID / "shards" / f"shard-{args.shard:02d}"
        part = f"shard-{args.shard:02d}"
    texts = ordered_rationale_texts(grouped, selected)
    tokenizer, model, config_sha = _load_encoder()
    embeddings, token_audit = embed_texts(texts, tokenizer, model, batch_size=args.batch_size)
    rows = _feature_rows(selected, embeddings)
    rows_path = output_root / "features.jsonl"
    rows_sha = _write_jsonl_fresh(rows_path, rows)
    matrix = np.asarray([row["features"] for row in rows], dtype="<f4")
    metadata = {
        "schema_version": SCHEMA_VERSION + "-part", "status": "completed", "run_id": RUN_ID,
        "part": part, "physical_gpu": physical, "records": len(rows), "feature_dim": FEATURE_DIM,
        "feature_rows_sha256": rows_sha, "feature_matrix_sha256": sha256(matrix.tobytes()).hexdigest(),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_config_sha256": config_sha, "embedding_dim": EMBEDDING_DIM,
        "max_length": MAX_LENGTH, "pooling": "last_nonpad_float32_l2",
        "projection_dim": PROJECTION_DIM, "projection_seed": PROJECTION_SEED,
        "projection_matrix_sha256": matrix_sha256(rademacher_projection()),
        "token_audit": token_audit, "bindings": bindings,
        "validation_loaded": False, "candidate_score_in_embedding_text": False,
        "essay_in_embedding_text": False, "prompt_in_embedding_text": False,
    }
    _write_json_fresh(output_root / "metadata.json", metadata, restricted=True)
    public = {key: value for key, value in metadata.items() if key not in {"feature_rows_sha256"}}
    public["restricted_feature_rows_sha256"] = rows_sha
    _write_json_fresh(public_root / "metadata.json", public)
    return metadata


def merge(args: argparse.Namespace) -> Mapping[str, Any]:
    source_ids, _, bindings = load_inputs(args)
    all_rows = []
    parts = []
    common = None
    token_totals = {"texts": 0, "total_tokens": 0, "truncated_texts": 0, "max_tokens": 0}
    for shard in range(SHARD_COUNT):
        root = RESTRICTED_BASE / RUN_ID / "shards" / f"shard-{shard:02d}"
        metadata_path, rows_path = root / "metadata.json", root / "features.jsonl"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        need(metadata.get("status") == "completed" and metadata.get("part") == f"shard-{shard:02d}"
             and metadata.get("feature_rows_sha256") == file_sha256(rows_path), "shard binding differs")
        keys = ("model_id", "model_revision", "model_config_sha256", "embedding_dim", "max_length",
                "pooling", "projection_dim", "projection_seed", "projection_matrix_sha256", "bindings")
        current = {key: metadata.get(key) for key in keys}
        common = current if common is None else common
        need(current == common, "shard provenance differs")
        start, stop = shard_bounds(2000, shard)
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
        need(len(rows) == stop - start and [row.get("source_id") for row in rows] == source_ids[start:stop],
             "shard source order differs")
        all_rows.extend(rows)
        token = metadata["token_audit"]
        for key in ("texts", "total_tokens", "truncated_texts"):
            token_totals[key] += int(token[key])
        token_totals["max_tokens"] = max(token_totals["max_tokens"], int(token["max_tokens"]))
        parts.append({"shard": shard, "records": len(rows), "metadata_sha256": file_sha256(metadata_path),
                      "feature_rows_sha256": metadata["feature_rows_sha256"]})
    need(len(all_rows) == 2000 and common is not None and common["bindings"] == bindings,
         "merged feature population or input binding differs")
    restricted_root, public_root = RESTRICTED_BASE / RUN_ID / "merged", PUBLIC_BASE / RUN_ID
    rows_path = restricted_root / "rows.jsonl"
    rows_sha = _write_jsonl_fresh(rows_path, all_rows)
    feature_matrix = np.asarray([row["features"] for row in all_rows], dtype="<f4")
    need(feature_matrix.shape == (2000, FEATURE_DIM) and np.isfinite(feature_matrix).all(),
         "merged feature matrix differs")
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "completed", "run_id": RUN_ID,
        "split_role": "train", "records": 2000, "feature_dim": FEATURE_DIM,
        "feature_rows_sha256": rows_sha,
        "feature_matrix_sha256": sha256(feature_matrix.tobytes(order="C")).hexdigest(),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_config_sha256": common["model_config_sha256"], "embedding_dim": EMBEDDING_DIM,
        "max_length": MAX_LENGTH, "pooling": "last_nonpad_float32_l2",
        "projection_dim": PROJECTION_DIM, "projection_seed": PROJECTION_SEED,
        "projection_matrix_sha256": common["projection_matrix_sha256"],
        "candidate_bindings": bindings, "render_contract": RENDER_CONTRACT,
        "render_contract_sha256": _json_hash(RENDER_CONTRACT), "token_audit": token_totals,
        "shards": parts, "row_level_storage": "restricted_only",
        "validation_loaded": False, "candidate_score_in_embedding_text": False,
        "essay_in_embedding_text": False, "prompt_in_embedding_text": False,
    }
    _write_json_fresh(restricted_root / "manifest.json", manifest, restricted=True)
    _write_json_fresh(public_root / "manifest.json", manifest)
    return manifest


def progress() -> Mapping[str, Any]:
    complete = []
    for shard in range(SHARD_COUNT):
        path = RESTRICTED_BASE / RUN_ID / "shards" / f"shard-{shard:02d}" / "metadata.json"
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if value.get("status") == "completed":
                complete.append(shard)
    merged = (PUBLIC_BASE / RUN_ID / "manifest.json").is_file()
    return {"run_id": RUN_ID, "completed_shards": complete, "shard_count": SHARD_COUNT,
            "percent": 100.0 * len(complete) / SHARD_COUNT, "merged": merged}


def launch(args: argparse.Namespace) -> Mapping[str, Any]:
    smoke = RESTRICTED_BASE / f"{RUN_ID}-smoke" / "metadata.json"
    need(smoke.is_file() and json.loads(smoke.read_text(encoding="utf-8")).get("status") == "completed",
         "passing GPU0 smoke is required before launch")
    snapshots, conflicts = _gpu_preflight(AUTHORIZED_GPUS)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    _append_ledger(
        "full_preflight", gpu_scope=list(AUTHORIZED_GPUS),
        gpu_authorization="user explicitly authorized GPUs 0,1,2,3 for this named iterative program",
        gpu_snapshot=snapshots, conflicts=conflicts, git_sha_at_launch=git_sha,
        exact_command="scripts/build_iterative_official_rationale_embeddings.sh launch --batch-size " + str(args.batch_size),
    )
    need(not conflicts, f"pre-existing compute process detected; feature launch refused: {conflicts}")
    processes = []
    for shard in range(SHARD_COUNT):
        command = [sys.executable, str(Path(__file__).resolve()), "--shard", str(shard),
                   "--physical-gpu", str(shard), "--batch-size", str(args.batch_size),
                   "--terra-manifest", str(args.terra_manifest), "--terra-candidates", str(args.terra_candidates),
                   "--luna-manifest", str(args.luna_manifest), "--luna-candidates", str(args.luna_candidates)]
        environment = dict(os.environ); environment["CUDA_VISIBLE_DEVICES"] = str(shard)
        processes.append(subprocess.Popen(command, cwd=ROOT, env=environment))
    codes = [process.wait() for process in processes]
    _append_ledger("full_shards_finished", exit_codes=codes, gpu_scope=list(AUTHORIZED_GPUS))
    need(codes == [0] * SHARD_COUNT, f"one or more shard processes failed: {codes}")
    return progress()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    actions = value.add_mutually_exclusive_group(required=True)
    actions.add_argument("--smoke", action="store_true")
    actions.add_argument("--shard", type=int)
    actions.add_argument("--launch", action="store_true")
    actions.add_argument("--merge", action="store_true")
    actions.add_argument("--progress", action="store_true")
    value.add_argument("--physical-gpu", type=int, default=0)
    value.add_argument("--batch-size", type=int, default=8)
    value.add_argument("--terra-manifest", type=Path, default=TERRA_ROOT / "manifest.json")
    value.add_argument("--terra-candidates", type=Path, default=TERRA_ROOT / "candidates.train.jsonl")
    value.add_argument("--luna-manifest", type=Path, default=LUNA_ROOT / "manifest.json")
    value.add_argument("--luna-candidates", type=Path, default=LUNA_ROOT / "candidates.train.jsonl")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.smoke:
        snapshots, conflicts = _gpu_preflight((0,))
        _append_ledger(
            "smoke_preflight", gpu_scope=[0],
            gpu_authorization="user explicitly authorized GPUs 0,1,2,3 for this named iterative program",
            gpu_snapshot=snapshots, conflicts=conflicts,
            exact_command="scripts/build_iterative_official_rationale_embeddings.sh smoke --batch-size " + str(args.batch_size),
        )
        need(not conflicts, f"pre-existing GPU0 compute process detected; smoke refused: {conflicts}")
        result = _run_part(args, smoke=True)
        _append_ledger("smoke_completed", gpu=0, records=result["records"])
    elif args.shard is not None:
        need(args.shard in range(SHARD_COUNT), "shard must be 0..3")
        result = _run_part(args, smoke=False)
    elif args.launch:
        result = launch(args)
    elif args.merge:
        result = merge(args)
        _append_ledger("merge_completed", records=result["records"], feature_matrix_sha256=result["feature_matrix_sha256"])
    else:
        result = progress()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
