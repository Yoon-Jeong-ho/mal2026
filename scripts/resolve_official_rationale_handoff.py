#!/usr/bin/env python3
"""Resolve the handoff template from explicit completed artifact bindings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_rationale_candidate_evaluation import resolve_handoff  # noqa: E402
from mal2026.official_rationale_handoff import HandoffConfig, file_sha256, read_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=ROOT / "configs/official_rationale_handoff.v1.json")
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-pending-evaluations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    template = read_json(args.template, "handoff template")
    bindings = read_json(args.bindings, "candidate bindings")
    resolved = resolve_handoff(template, bindings, require_evaluations=not args.allow_pending_evaluations)
    checked = HandoffConfig(resolved)
    if not args.allow_pending_evaluations:
        checked.validate_dependencies()
    summary = {
        "status": "resolved", "candidate_count": len(resolved["candidates"]),
        "evaluations_required": not args.allow_pending_evaluations,
        "all_required_placeholders_removed": "REQUIRED_" not in json.dumps(resolved),
    }
    if args.dry_run:
        print(json.dumps(summary, sort_keys=True)); return
    if args.output is None or args.output.exists():
        raise RuntimeError("a fresh --output is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary.update({"output": str(args.output.resolve()), "output_sha256": file_sha256(args.output)})
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
