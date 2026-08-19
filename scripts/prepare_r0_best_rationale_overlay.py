#!/usr/bin/env python3
"""Stage a hash-bound overlay: R0 score ensemble + best latest rationale SFT."""
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
OVERLAY = ROOT / "deployment/runtime_overlay_r0_best_rationale"
SOURCE_BUNDLE = ROOT / "deployment/runtime_bundle_r0"
SOURCE_COMPLETE = SOURCE_BUNDLE / "bundle_complete.json"
SOURCE_MANIFEST = SOURCE_BUNDLE / "manifest.json"
PROMPT = ROOT / "Rationale_evaluation_training.txt"
ADAPTER = ROOT / (
    "outputs/rationale-pipeline-sft-v1/"
    "rationale-pipeline-sft-v1-mal-direct-lora-full-20260807-001/adapter"
)
TRAINING_COMPLETE = ADAPTER.parent / "training_complete.json"
FINAL_REPORT = ROOT / (
    "outputs/rationale-pipeline-final-report-v2/"
    "rationale-pipeline-final-report-v2-20260811-001/aggregate.json"
)
COMPLETION_AUDIT = FINAL_REPORT.parent / "completion_audit.json"

EXPECTED = {
    "source_complete": "58c029d014b80ac9530a1e6e2535a235bb8e90f45b1713246b0919c11045e80f",
    "source_manifest": "a44760a93d7f84d31aff2a6531cd5895044db64d394a373ffd154273c25b641d",
    "prompt": "c7d18cdfceb82cba9d355e9f98b0cea7cc60f500bd6e56494ac87be0d3160285",
    "adapter_config": "ffba1f524ee191b1f91bb6ce03ae9bdc88b636739a1cdb0d732a7b0e386409f7",
    "adapter_model": "48eff87081928adb08ae483e2ceb40c67615777a3d9a1f1e69f4a0c9382d5dfb",
    "final_report": "a80e134a4266b2db1828f596fc6111aedd2e04fe91dea3ae9f82cc369c2e8b11",
    "completion_audit": "dec604e8926f6874c6781c98ffb0e4df362f8c7313275c775138e70a6ad61a48",
}


class OverlayError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise OverlayError(message)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"JSON object expected: {path}")
    return value


def preflight() -> dict[str, Any]:
    paths = {
        "source_complete": SOURCE_COMPLETE,
        "source_manifest": SOURCE_MANIFEST,
        "prompt": PROMPT,
        "adapter_config": ADAPTER / "adapter_config.json",
        "adapter_model": ADAPTER / "adapter_model.safetensors",
        "final_report": FINAL_REPORT,
        "completion_audit": COMPLETION_AUDIT,
    }
    observed = {}
    for key, path in paths.items():
        need(path.is_file() and not path.is_symlink(), f"artifact unavailable: {path}")
        observed[key] = digest(path)
        need(observed[key] == EXPECTED[key], f"artifact checksum differs: {key}")

    source_complete = read_json(SOURCE_COMPLETE)
    need(source_complete.get("status") == "completed", "source R0 bundle is incomplete")
    need(abs(float(source_complete.get("validation_continuous_macro_rmse")) - 0.5582937519204271) < 1e-12, "R0 score RMSE differs")
    need(abs(float(source_complete.get("validation_integer_macro_rmse")) - 0.6158981882311673) < 1e-12, "R0 integer RMSE differs")

    training = read_json(TRAINING_COMPLETE)
    need(training.get("status") == "completed", "latest rationale training is incomplete")
    need(training.get("human_or_reference_score_read_or_prompted") is False, "latest rationale policy is not score-blind")
    need(training.get("model_revision") == "ba21c20ea1b31ded1ec3e2fb432335077dc4be98", "rationale base revision differs")

    report = read_json(FINAL_REPORT)
    audit = read_json(COMPLETION_AUDIT)
    best = audit.get("best_rationale")
    need(report.get("status") == "completed" and audit.get("all_requirements_proven") is True, "latest final report is incomplete")
    need(isinstance(best, dict) and best.get("candidate") == "mal-direct-lora-epoch2", "selected rationale candidate differs")
    need(abs(float(best.get("judge_macro")) - 4.926875) < 1e-12, "selected rationale judge macro differs")
    return {
        "status": "passed",
        "candidate": "r0_prediction_ensemble_plus_mal_direct_lora_epoch2",
        "score_validation_continuous_macro_rmse": source_complete["validation_continuous_macro_rmse"],
        "score_validation_integer_macro_rmse": source_complete["validation_integer_macro_rmse"],
        "rationale_judge_macro": best["judge_macro"],
        "rationale_judge_worst_cell": best["judge_worst_cell"],
        "rationale_score_blind": True,
        "artifact_sha256": observed,
        "restricted_rows_read": False,
        "external_download": False,
    }


def hardlink_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=os.link, symlinks=False)


def stage() -> dict[str, Any]:
    result = preflight()
    existing = [path for path in OVERLAY.iterdir() if path.name != "README.md"]
    need(not existing, "runtime overlay already contains staged or failed artifacts")
    adapter_destination = OVERLAY / "rationale/adapters/mal_direct_lora_epoch2"
    prompt_destination = OVERLAY / "rationale/prompts/Rationale_evaluation_training.txt"
    hardlink_tree(ADAPTER, adapter_destination)
    prompt_destination.parent.mkdir(parents=True, exist_ok=False)
    os.link(PROMPT, prompt_destination)

    manifest = read_json(SOURCE_MANIFEST)
    manifest["pipeline_kind"] = "legacy_r0_prediction_ensemble_to_latest_score_blind_rationale"
    manifest["served_model_name"] = "mal2026-r0-ensemble-best-rationale-v1"
    rationale = manifest["rationale"]
    need(isinstance(rationale, dict), "source rationale manifest differs")
    rationale.update({
        "final_adapter_path": "rationale/adapters/mal_direct_lora_epoch2",
        "final_prompt_kind": "latest_score_blind_v1",
        "final_prompt_path": "rationale/prompts/Rationale_evaluation_training.txt",
        "final_prompt_sha256": EXPECTED["prompt"],
    })
    (OVERLAY / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    complete = {**result, "status": "completed"}
    (OVERLAY / "overlay_complete.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    try:
        result = stage() if args.stage else preflight()
    except Exception as exc:
        print(json.dumps({"status": "failed", "type": type(exc).__name__, "message": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
