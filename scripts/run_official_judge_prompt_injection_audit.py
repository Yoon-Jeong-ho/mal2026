#!/usr/bin/env python3
"""Queue/run the frozen Q4 injection audit and combine it with the directional gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_judge_injection_gate import compare_records, need  # noqa: E402


CONFIG = ROOT / "configs/official_q4_judge_prompt_injection_gate.v1.json"
CONTRACT = ROOT / "src/mal2026/official_writing_contract.py"
CONTRACT_SHA256 = "7b04149227a44852ca78bd65f5ec70245b284503256374debf2735f17ca69e50"
CONTRASTIVE = ROOT / "outputs/official-prompt-alignment-v1/judge-contrastive/official-judge-contrastive-train32-001/aggregate_contrastive_gate.json"
JUDGE_AGG_ROOT = ROOT / "outputs/official-prompt-alignment-v1/q4-judge"
JUDGE_RECORD_ROOT = ROOT / "data/processed/restricted/official_prompt_alignment_v1/q4_judge"
LAUNCHER = ROOT / "scripts/run_official_q4_judge_prompt_injection.sh"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_records(run_id: str, count: int) -> list[dict[str, Any]]:
    path = JUDGE_RECORD_ROOT / run_id / "judge_records.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda row: str(row["source_id"]))
    need(len(rows) == count and all(row.get("judge_output") is not None for row in rows), "injection judge records differ")
    return rows


def wait_for_contrastive() -> dict[str, Any]:
    for _ in range(1440):
        if CONTRASTIVE.is_file():
            value = json.loads(CONTRASTIVE.read_text(encoding="utf-8"))
            need(value.get("status") in {"passed", "failed_gates"}, "directional contrastive state differs")
            return value
        time.sleep(30)
    raise RuntimeError("directional contrastive gate wait timeout")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8")); run_id = cfg["run_id"]
    from hashlib import sha256
    need(sha256(CONTRACT.read_bytes()).hexdigest() == CONTRACT_SHA256, "frozen judge prompt contract changed")
    count = int(cfg["source"]["records"])
    participant_root = ROOT / "data/processed/restricted/official_prompt_alignment_v1/judge_prompt_injection" / run_id
    run_root = ROOT / "outputs/official-prompt-alignment-v1/judge-prompt-injection" / run_id
    need((run_root / "preparation_report.json").is_file(), "injection preparation report is unavailable")
    need(set(cfg["variants"]) == {"base", "rationale_injection", "essay_injection"}, "injection variants differ")
    for variant in cfg["variants"]: need((participant_root / f"{variant}.jsonl").is_file(), f"injection input unavailable: {variant}")
    if args.dry_run:
        print(json.dumps({"status": "ready_not_launched", "run_id": run_id, "waits_for": str(CONTRASTIVE.relative_to(ROOT)),
                          "gpu_scope_when_launched": [0, 1, 2, 3], "variants": sorted(cfg["variants"]),
                          "judge_prompt_modified": False, "failure_exit_code": 2}, sort_keys=True))
        return
    contrastive = wait_for_contrastive()
    logs = run_root / "logs"; logs.mkdir()
    run_ids = {name: f"official-q4-judge-prompt-injection-{name.replace('_', '-')}-train32-001" for name in cfg["variants"]}
    for name, judge_run_id in run_ids.items():
        report = JUDGE_AGG_ROOT / judge_run_id / "aggregate_judge_report.json"
        need(not report.parent.exists(), f"injection judge output exists: {name}")
        command = [str(LAUNCHER), judge_run_id, str(participant_root / f"{name}.jsonl"), str(count)]
        with (logs / f"judge-{name}.log").open("x", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
        need(completed.returncode == 0, f"injection Q4 process failed: {name}")
        aggregate = json.loads(report.read_text(encoding="utf-8"))
        need(aggregate.get("status") == "completed" and all(aggregate.get("hard_gates", {}).values()), f"injection Q4 aggregate failed: {name}")
    base = read_records(run_ids["base"], count)
    tests = {name: compare_records(base, read_records(run_ids[name], count), cfg["thresholds_frozen_before_results"])
             for name in ("rationale_injection", "essay_injection")}
    injection_passed = all(test["passed"] for test in tests.values())
    injection_report = {
        "schema_version": "mal2026-official-proxy-judge-prompt-injection-gate-v1",
        "status": "passed" if injection_passed else "failed_gates", "run_id": run_id, "completed_at": now(),
        "records_per_variant": count, "thresholds_frozen_before_results": cfg["thresholds_frozen_before_results"],
        "tests": tests, "judge_prompt_modified": False,
        "judge_prompt_kind": "single_frozen_public-spec-aligned_proxy_not_verbatim_organizer_prompt",
        "human_or_reference_score_read_or_prompted": False,
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_predictions_or_injection_payloads",
    }
    injection_path = run_root / "aggregate_prompt_injection_gate.json"; need(not injection_path.exists(), "injection aggregate exists")
    atomic_json(injection_path, injection_report)
    directional_passed = contrastive.get("status") == "passed" and contrastive.get("rl_with_this_proxy_judge_allowed") is True
    combined_passed = directional_passed and injection_passed
    combined = {
        "schema_version": "mal2026-official-proxy-judge-rl-safety-gate-v1", "status": "passed" if combined_passed else "failed_gates",
        "completed_at": now(), "directional_contrastive_gate_passed": directional_passed,
        "prompt_injection_gate_passed": injection_passed, "rl_allowed": combined_passed,
        "failure_policy": "preserve all artifacts and exit 2; do not run RL with this proxy judge",
    }
    combined_path = run_root / "aggregate_rl_safety_gate.json"; atomic_json(combined_path, combined)
    print(json.dumps(combined, sort_keys=True))
    if not combined_passed: raise SystemExit(2)


if __name__ == "__main__": main()
