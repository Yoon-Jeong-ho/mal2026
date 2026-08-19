#!/usr/bin/env python3
"""Preflight and stage the exact-prompt Docker model bundle.

This script never downloads a model or reads train/validation rows.  It uses
only completed local model artifacts and the public evaluation prompt.  The
large export is deliberately separate from Docker build/upload.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = ROOT / "deployment"
BUNDLE = DEPLOYMENT / "runtime_bundle"
RESULT = ROOT / (
    "outputs/evaluation-prompt-score-encoder-v1/"
    "evaluation-prompt-score-encoder-v1-qwen3-embedding-8b-rationale-aware-score-blind-20260729-001/"
    "result.json"
)
SCORE_STATE = RESULT.parent / "selected_refit_trainable.safetensors"
BLIND_ADAPTER = ROOT / (
    "outputs/evaluation-prompt-rationale-sft-v2/"
    "evaluation-prompt-rationale-sft-v2-ax4-score-blind-20260729-001/adapter"
)
FINAL_DPO_ADAPTER = ROOT / (
    "outputs/official-rationale-rl-v1/orchestration/"
    "official-rationale-rl-experiment-v1-exact-judge-20260729-019/models/"
    "dpo-official-bundle-ddp4-full-user-aligned-001/adapter"
)
RATIONALE_BASE = ROOT / "outputs/model-cache/skt--A.X-4.0-Light-ba21c20ea1b31ded1ec3e2fb432335077dc4be98"
EVALUATION = ROOT / "evaluation.txt"
EXPECTED = {
    "evaluation": "1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145",
    "score_state": "4bdf0fd0ea96ef229bbea087fcde4ffa8f1737c458e9f67471a7e65b577b75e9",
    "blind_adapter_model": "2d3f3c2ad0f773fcf1958e42f96fe93b66c4fa56899d1d89d3ca523f65a8c86d",
    "final_dpo_adapter_model": "887abf9d1bf07693251a17b7a0fb655fe8203fa6945e9c178a38bdc538ded826",
}


class BundleError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_result() -> dict[str, Any]:
    try:
        value = json.loads(RESULT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError("exact score result is unreadable") from exc
    need(isinstance(value, dict), "exact score result is not an object")
    need(value.get("schema_version") == "mal2026-evaluation-prompt-score-encoder-result-v1", "score result schema differs")
    need(value.get("input_kind") == "rationale_aware" and value.get("rationale_variant") == "score_blind", "score arm differs")
    need(value.get("evaluation_txt_sha256") == EXPECTED["evaluation"], "score prompt lineage differs")
    return value


def preflight(*, include_dpo: bool, hash_large_files: bool) -> dict[str, Any]:
    result = read_result()
    config = result.get("config")
    need(isinstance(config, dict), "score config is unavailable")
    score_base = Path(str(config.get("model_path", "")))
    warmstart = Path(str(config.get("warmstart_artifact_path", "")))
    paths = {
        "evaluation": EVALUATION,
        "score_state": SCORE_STATE,
        "score_base": score_base,
        "score_warmstart": warmstart,
        "rationale_base": RATIONALE_BASE,
        "blind_adapter": BLIND_ADAPTER,
    }
    if include_dpo:
        paths["final_dpo_adapter"] = FINAL_DPO_ADAPTER
    for label, path in paths.items():
        need(path.exists() and not path.is_symlink(), f"{label} is unavailable")

    checks = {
        "evaluation": file_sha256(EVALUATION),
        "score_state": file_sha256(SCORE_STATE),
        "blind_adapter_model": file_sha256(BLIND_ADAPTER / "adapter_model.safetensors"),
    }
    if include_dpo:
        checks["final_dpo_adapter_model"] = file_sha256(FINAL_DPO_ADAPTER / "adapter_model.safetensors")
    for label, digest in checks.items():
        need(digest == EXPECTED[label], f"{label} checksum differs")

    warmstart_marker = RESULT.parent / "warmstart_verified.json"
    marker = json.loads(warmstart_marker.read_text(encoding="utf-8"))
    need(marker.get("verified") is True and marker.get("artifact_sha256") == config.get("warmstart_artifact_sha256"), "warmstart verification differs")
    if hash_large_files:
        from mal2026.rationale_aware_encoder import verify_artifact_inventory

        class ConfigView:
            warmstart_completion_path = str(config["warmstart_completion_path"])
            warmstart_artifact_path = str(config["warmstart_artifact_path"])
            warmstart_artifact_sha256 = str(config["warmstart_artifact_sha256"])

        verified = verify_artifact_inventory(ConfigView())  # type: ignore[arg-type]
        need(verified.get("verified") is True, "warmstart inventory verification failed")

    usage = shutil.disk_usage(BUNDLE.parent)
    need(usage.free >= 40 * 1024**3, "less than 40 GiB is free for the staged bundle")
    return {
        "status": "passed",
        "candidate": "exact_score_dpo_rationale_v1" if include_dpo else "exact_contract_reuse_blind_v1",
        "external_download": False,
        "restricted_rows_read": False,
        "checksums": checks,
        "score_model_id": result.get("model_id"),
        "score_model_revision": result.get("model_revision"),
        "rationale_model_id": "skt/A.X-4.0-Light",
        "rationale_model_revision": "ba21c20ea1b31ded1ec3e2fb432335077dc4be98",
        "free_bytes": usage.free,
    }


def hardlink_tree(source: Path, destination: Path) -> None:
    need(source.is_dir() and not destination.exists(), f"cannot stage {source.name}")
    shutil.copytree(source, destination, copy_function=os.link, symlinks=False)


def export_score_backbone(result: dict[str, Any], destination: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    from mal2026.evaluation_prompt_score_encoder import EvaluationPromptScoreEncoderConfig
    from mal2026.rationale_aware_encoder import build_model

    raw = dict(result["config"])
    raw["score_fields"] = tuple(raw["score_fields"])
    raw["selection_epochs"] = tuple(raw["selection_epochs"])
    config = EvaluationPromptScoreEncoderConfig(**raw)
    config.validate(require_dependencies=False)
    model, lineage = build_model(config)  # type: ignore[arg-type]
    state = load_file(str(SCORE_STATE), device="cpu")
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    need(set(state) == expected, "score trainable-state tensor set differs")
    incompatible = model.load_state_dict(state, strict=False)
    need(not incompatible.unexpected_keys, "score state has unexpected tensors")

    destination.mkdir(parents=True, exist_ok=False)
    head = {
        "weight": model.score_head.weight.detach().to(dtype=torch.bfloat16, device="cpu").contiguous(),
        "bias": model.score_head.bias.detach().to(dtype=torch.bfloat16, device="cpu").contiguous(),
    }
    head_path = destination.parent / "score_head.safetensors"
    save_file(head, str(head_path), metadata={"schema_version": "mal2026-three-axis-bounded-regression-head-v1"})
    merged = model.backbone.merge_and_unload(safe_merge=True)
    merged.config.use_cache = False
    merged.save_pretrained(destination, safe_serialization=True, max_shard_size="4GB")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, revision=config.model_revision, local_files_only=True,
        trust_remote_code=False, use_fast=True,
    )
    tokenizer.save_pretrained(destination)
    return {
        "initialization": lineage,
        "source_trainable_state_sha256": EXPECTED["score_state"],
        "score_head_sha256": file_sha256(head_path),
    }


def stage(*, include_dpo: bool, hash_large_files: bool) -> dict[str, Any]:
    audit = preflight(include_dpo=include_dpo, hash_large_files=hash_large_files)
    existing = [path for path in BUNDLE.iterdir() if path.name != "README.md"]
    need(not existing, "runtime_bundle already contains staged or failed artifacts")
    started = time.time()
    try:
        score_root = BUNDLE / "score"
        rationale_root = BUNDLE / "rationale"
        score_root.mkdir(parents=True, exist_ok=False)
        rationale_root.mkdir(parents=True, exist_ok=False)
        result = read_result()
        export = export_score_backbone(result, score_root / "backbone")
        hardlink_tree(RATIONALE_BASE, rationale_root / "base")
        hardlink_tree(BLIND_ADAPTER, rationale_root / "adapters" / "score_blind_v2")
        if include_dpo:
            hardlink_tree(FINAL_DPO_ADAPTER, rationale_root / "adapters" / "final_dpo")
        shutil.copy2(EVALUATION, BUNDLE / "evaluation.txt")
        template = DEPLOYMENT / (
            "runtime_manifest.dpo.template.json" if include_dpo else "runtime_manifest.template.json"
        )
        shutil.copy2(template, BUNDLE / "manifest.json")
        complete = {
            **audit,
            "status": "completed",
            "duration_seconds": time.time() - started,
            "score_export": export,
        }
        (BUNDLE / "bundle_complete.json").write_text(
            json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        return complete
    except Exception as exc:
        failure = {
            "status": "failed",
            "type": type(exc).__name__,
            "message": str(exc),
            "duration_seconds": time.time() - started,
        }
        (BUNDLE / "bundle_failed.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true", help="export and stage the local runtime bundle")
    parser.add_argument("--include-dpo", action="store_true", help="stage the optional final DPO adapter")
    parser.add_argument("--hash-large-files", action="store_true", help="rehash the 30+ GiB warmstart inventory")
    args = parser.parse_args()
    try:
        report = (
            stage(include_dpo=args.include_dpo, hash_large_files=args.hash_large_files)
            if args.stage
            else preflight(include_dpo=args.include_dpo, hash_large_files=args.hash_large_files)
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "type": type(exc).__name__, "message": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
