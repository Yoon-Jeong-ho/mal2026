#!/usr/bin/env python3
"""Verify and stage the metric-first R0 P1--4 prediction ensemble bundle."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = ROOT / "deployment"
BUNDLE = DEPLOYMENT / "runtime_bundle_r0"
TRAIN_ROOT = ROOT / (
    "outputs/rlaif-qwen3-embedding-epoch-sweep-v1/"
    "rlaif-qwen3-embedding-epoch-sweep-v1-full-003"
)
TRAINING_COMPLETE = TRAIN_ROOT / "training_complete.json"
METRICS = ROOT / (
    "outputs/official-prompt-alignment-v1/score-metrics/"
    "official-score-r0-ensemble-full-20260727-002/aggregate_metrics.json"
)
QWEN_BASE = ROOT / "outputs/model-cache/Qwen--Qwen3-Embedding-8B-1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
AX_BASE = ROOT / "outputs/model-cache/skt--A.X-4.0-Light-ba21c20ea1b31ded1ec3e2fb432335077dc4be98"
BLIND_ADAPTER = ROOT / (
    "outputs/rlaif-grpo-prompt-ensemble-v8/"
    "rlaif-grpo-prompt-ensemble-v8-ax4_light-bundle-random1-full-023/adapter"
)
FINAL_ADAPTER = ROOT / (
    "outputs/official-rationale-rl-v1/orchestration/"
    "official-rationale-rl-experiment-v1-exact-judge-20260729-019/models/"
    "dpo-official-bundle-ddp4-full-user-aligned-001/adapter"
)
EXPECTED_CHECKPOINTS = {
    1: "fd317dd017133f2d6120d857a2d7f7d6caebaf547590cead155ca0ffda1a9c7c",
    2: "dd47ac7ea93dce46da5f9e9b44cf86039331a0973f5f7f21cee46b2f4b85b57d",
    3: "96f5897a27f600b52156dd992e41fa7aefb0d7c6948d3fbf71b0c002939d469d",
    4: "30b8c677973a2b0df8052cf67a2717ba37b6f64180bc4e48be1572c3e4b6c592",
}
EXPECTED_BLIND_SHA = "39c68bb5c98da25eaa466434ba1c6d4a47bedcec580c991f00335627382a3a73"
EXPECTED_FINAL_SHA = "887abf9d1bf07693251a17b7a0fb655fe8203fa6945e9c178a38bdc538ded826"
PEFT_SOURCE = ROOT / ".venv-standard/lib/python3.12/site-packages/peft"


class R0BundleError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise R0BundleError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R0BundleError(f"{label} is unreadable") from exc
    need(isinstance(value, dict), f"{label} must be an object")
    return value


def checkpoint_path(epoch: int) -> Path:
    return TRAIN_ROOT / "epoch_checkpoints" / f"epoch-{epoch:02d}" / "trainable_model.safetensors"


def preflight() -> dict[str, Any]:
    training = read_json(TRAINING_COMPLETE, "R0 training completion")
    need(training.get("status") == "completed", "R0 training is incomplete")
    need(training.get("run_id") == "rlaif-qwen3-embedding-epoch-sweep-v1-full-003", "R0 training identity differs")
    need(training.get("source_key") == "rank2_ax4_random1", "R0 rationale source differs")
    need(training.get("average_target_used") is False and training.get("score_fields") == ["content", "organization", "expression"], "R0 score axes differ")
    need(training.get("model_revision") == "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", "R0 model revision differs")
    checkpoints = training.get("checkpoints")
    need(isinstance(checkpoints, list) and len(checkpoints) >= 4, "R0 checkpoint inventory differs")
    observed: dict[str, str] = {}
    for epoch, expected in EXPECTED_CHECKPOINTS.items():
        path = checkpoint_path(epoch)
        need(path.is_file() and not path.is_symlink(), f"R0 epoch {epoch} checkpoint is unavailable")
        digest = file_sha256(path)
        need(digest == expected, f"R0 epoch {epoch} checksum differs")
        row = next((item for item in checkpoints if isinstance(item, dict) and item.get("epoch") == epoch), None)
        need(isinstance(row, dict) and row.get("trainable_state_sha256") == expected, f"R0 epoch {epoch} metadata differs")
        observed[f"epoch_{epoch:02d}"] = digest
    for label, path in {"Qwen base": QWEN_BASE, "A.X base": AX_BASE, "blind adapter": BLIND_ADAPTER, "final adapter": FINAL_ADAPTER}.items():
        need(path.is_dir() and not path.is_symlink(), f"{label} is unavailable")
    need(PEFT_SOURCE.is_dir() and (PEFT_SOURCE / "__init__.py").is_file(), "PEFT 0.19.1 source is unavailable")
    need('__version__ = "0.19.1"' in (PEFT_SOURCE / "__init__.py").read_text(encoding="utf-8"), "PEFT source version differs")
    blind_sha = file_sha256(BLIND_ADAPTER / "adapter_model.safetensors")
    final_sha = file_sha256(FINAL_ADAPTER / "adapter_model.safetensors")
    need(blind_sha == EXPECTED_BLIND_SHA and final_sha == EXPECTED_FINAL_SHA, "R0 rationale adapter checksum differs")
    metrics = read_json(METRICS, "R0 aggregate metrics")
    result = next((row for row in metrics.get("results", []) if row.get("candidate") == "r0_prediction_ensemble"), None)
    need(isinstance(result, dict), "R0 prediction ensemble result is absent")
    continuous = result.get("continuous_metrics")
    integer = result.get("official_integer_metrics")
    need(isinstance(continuous, dict) and isinstance(integer, dict), "R0 aggregate metrics differ")
    need(abs(float(continuous.get("three_axis_macro_rmse")) - 0.5582937519204271) < 1e-12, "R0 RMSE differs")
    need(abs(float(continuous.get("three_axis_macro_spearman")) - 0.6441959864775355) < 1e-12, "R0 Spearman differs")
    free = shutil.disk_usage(BUNDLE.parent).free
    need(free >= 40 * 1024**3, "less than 40 GiB is free for the R0 bundle")
    return {
        "status": "passed",
        "candidate": "historical_r0_prediction_ensemble_dpo_v1",
        "external_download": False,
        "restricted_rows_read": False,
        "checkpoint_sha256": observed,
        "blind_adapter_sha256": blind_sha,
        "final_adapter_sha256": final_sha,
        "validation_continuous_macro_rmse": float(continuous["three_axis_macro_rmse"]),
        "validation_continuous_macro_spearman": float(continuous["three_axis_macro_spearman"]),
        "validation_integer_macro_rmse": float(integer["three_axis_macro_rmse"]),
        "validation_integer_macro_spearman": float(integer["three_axis_macro_spearman"]),
        "free_bytes": free,
    }


def hardlink_tree(source: Path, destination: Path) -> None:
    need(source.is_dir() and not destination.exists(), f"cannot stage {destination}")
    shutil.copytree(source, destination, copy_function=os.link, symlinks=False)


def export_score_adapters(training: Mapping[str, Any], score_root: Path) -> list[dict[str, Any]]:
    from safetensors.torch import load_file, save_file

    from mal2026.rlaif_qwen3_epoch_sweep import EpochSweepTrainConfig
    from mal2026.rlaif_qwen3_embedding import build_model

    raw = dict(training["config"])
    raw["score_fields"] = tuple(raw["score_fields"])
    config = EpochSweepTrainConfig(**raw)
    config.validate(require_fresh_output=False)
    model, initialization = build_model(config)  # type: ignore[arg-type]
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    rows = []
    for epoch, digest in EXPECTED_CHECKPOINTS.items():
        state = load_file(str(checkpoint_path(epoch)), device="cpu")
        need(set(state) == expected, f"R0 epoch {epoch} trainable tensor set differs")
        incompatible = model.load_state_dict(state, strict=False)
        need(not incompatible.unexpected_keys and not (expected & set(incompatible.missing_keys)), f"R0 epoch {epoch} load differs")
        adapter = score_root / "adapters" / f"epoch_{epoch:02d}"
        adapter.mkdir(parents=True, exist_ok=False)
        model.backbone.save_pretrained(
            adapter, safe_serialization=True, selected_adapters=["default"],
        )
        head = {
            "weight": model.regression_head.weight.detach().float().cpu().contiguous(),
            "bias": model.regression_head.bias.detach().float().cpu().contiguous(),
        }
        head_path = score_root / "heads" / f"epoch_{epoch:02d}.safetensors"
        head_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(head, str(head_path), metadata={"schema_version": "mal2026-r0-three-axis-raw-regression-head-v1"})
        adapter_model = adapter / "adapter_model.safetensors"
        need(adapter_model.is_file(), f"R0 epoch {epoch} adapter export failed")
        rows.append({
            "epoch": epoch,
            "source_state_sha256": digest,
            "adapter_model_sha256": file_sha256(adapter_model),
            "head_sha256": file_sha256(head_path),
        })
    return [{"initialization": initialization}, *rows]


def stage() -> dict[str, Any]:
    audit = preflight()
    existing = [path for path in BUNDLE.iterdir() if path.name != "README.md"]
    need(not existing, "runtime_bundle_r0 already contains staged or failed artifacts")
    started = time.time()
    try:
        training = read_json(TRAINING_COMPLETE, "R0 training completion")
        score_root = BUNDLE / "score"
        rationale_root = BUNDLE / "rationale"
        score_root.mkdir(parents=True, exist_ok=False)
        rationale_root.mkdir(parents=True, exist_ok=False)
        hardlink_tree(QWEN_BASE, score_root / "base")
        score_export = export_score_adapters(training, score_root)
        hardlink_tree(AX_BASE, rationale_root / "base")
        hardlink_tree(BLIND_ADAPTER, rationale_root / "adapters" / "rank2_ax4_random1")
        hardlink_tree(FINAL_ADAPTER, rationale_root / "adapters" / "final_dpo")
        shutil.copytree(
            PEFT_SOURCE, BUNDLE / "python" / "peft",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(DEPLOYMENT / "runtime_manifest.r0.template.json", BUNDLE / "manifest.json")
        complete = {
            **audit,
            "status": "completed",
            "duration_seconds": time.time() - started,
            "score_export": score_export,
        }
        (BUNDLE / "bundle_complete.json").write_text(
            json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        return complete
    except Exception as exc:
        failure = {
            "status": "failed", "type": type(exc).__name__, "message": str(exc),
            "duration_seconds": time.time() - started,
        }
        (BUNDLE / "bundle_failed.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    try:
        report = stage() if args.stage else preflight()
    except Exception as exc:
        print(json.dumps({"status": "failed", "type": type(exc).__name__, "message": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
