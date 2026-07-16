#!/usr/bin/env python3
"""Create the ignored AI-Hub human-feedback train/dev/refit JSONL files.

The command is intentionally offline: it loads the already-reviewed local Qwen
snapshot solely to apply the frozen common target-token eligibility gate.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

from mal2026.human_feedback_data import (
    QWEN_CHAT_TEMPLATE_SHA256,
    QWEN_REVISION,
    HumanFeedbackDataError,
    discover_training_archives,
    prepare_human_feedback_data,
    write_prepared_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = ROOT / "outputs" / "model-cache" / f"Qwen--Qwen2.5-7B-Instruct-{QWEN_REVISION}"
DEFAULT_RAW_ROOT = ROOT / "data" / "raw" / "aihub"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "aihub_human_feedback_v1"
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "aihub_human_feedback_v1.json"


def load_pinned_tokenizer(path: Path):
    if not path.is_dir():
        raise HumanFeedbackDataError(f"local pinned tokenizer snapshot does not exist: {path}")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise HumanFeedbackDataError("transformers is required for the pinned tokenizer eligibility gate") from exc
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise HumanFeedbackDataError("pinned tokenizer has no chat template")
    observed = sha256(template.encode("utf-8")).hexdigest()
    if observed != QWEN_CHAT_TEMPLATE_SHA256:
        raise HumanFeedbackDataError("pinned tokenizer chat template SHA-256 mismatch")
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    tokenizer = load_pinned_tokenizer(args.tokenizer_path)
    prepared = prepare_human_feedback_data(discover_training_archives(args.raw_root), tokenizer)
    manifest = write_prepared_dataset(prepared, args.output_root, args.manifest_path)
    print(f"wrote restricted rows: {args.output_root.relative_to(ROOT)}")
    print(f"wrote aggregate manifest: {args.manifest_path.relative_to(ROOT)}")
    print(f"eligible records: {manifest['eligibility']['eligible_records']}")


if __name__ == "__main__":
    main()
