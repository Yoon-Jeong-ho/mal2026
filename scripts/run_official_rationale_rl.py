#!/usr/bin/env python3
"""Plan or execute the declared official rationale RL stages.

The runner never starts model servers implicitly.  Server launch/attestation
is an explicit experiment-runner responsibility, which keeps the sequential
TP4 DPO rollout/Q4 judge stages and the GRPO 2+1+1 GPU partition auditable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_rationale_rl import RLSettings, TASKS, validate_runtime_versions  # noqa: E402


DPO_CONFIG = ROOT / "configs/official_rationale_dpo.v1.json"
GRPO_CONFIG = ROOT / "configs/official_rationale_grpo.v1.json"
DPO_TRAINER = ROOT / "scripts/train_official_rationale_dpo.py"
GRPO_TRAINER = ROOT / "scripts/train_official_rationale_grpo.py"
OUTPUT_ROOT = ROOT / "outputs/official-rationale-rl-v1"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def task_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        task, separator, raw_path = value.partition("=")
        need(separator == "=" and task in TASKS and task not in result and bool(raw_path), "task path must be unique TASK=PATH")
        result[task] = Path(raw_path)
    return result


def require_gpu_idle(gpu: int) -> None:
    """Read-only conflict gate; never changes or terminates an existing process."""
    need(gpu in {0, 1, 2, 3}, "GPU is outside the authorized scope")
    completed = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        cwd=ROOT, text=True, capture_output=True,
    )
    need(completed.returncode == 0, f"GPU{gpu} availability query failed")
    pids = [line.strip() for line in completed.stdout.splitlines() if line.strip() and line.strip() != "[N/A]"]
    need(not pids, f"GPU{gpu} has pre-existing compute processes; refusing launch")


def validate_smoke_completion(path: Path, settings: RLSettings) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(OUTPUT_ROOT.resolve()), "DPO smoke completion is unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    gate = settings.gate_evidence()
    need(value.get("schema_version") == "mal2026-official-rationale-dpo-complete-v1" and value.get("status") == "completed", "DPO smoke completion status differs")
    need(value.get("global_step") == 1, "DPO smoke must be exactly one update")
    need(value.get("contrastive_gate_sha256") == gate["directional"]["sha256"] and value.get("rl_safety_gate_sha256") == gate["combined_safety"]["sha256"], "DPO smoke gate binding differs")
    return value


def plan() -> dict[str, Any]:
    versions = validate_runtime_versions()
    dpo = RLSettings.from_json(DPO_CONFIG)
    grpo = RLSettings.from_json(GRPO_CONFIG)
    gate_files = [ROOT / str(dpo.gate[key]) for key in ("path", "safety_path")]
    gates_ready = all(path.is_file() for path in gate_files)
    gate_evidence: Any = False
    if gates_ready:
        try:
            gate_evidence = dpo.gate_evidence()
        except Exception as exc:
            gate_evidence = {"ready": False, "failure_type": type(exc).__name__}
    legacy_execution = [
        {
            "name": item["name"],
            "smoke_command": f"run_official_rationale_rl.py --execute-dpo --smoke --legacy-arm {item['name']} --preference bundle=<restricted-jsonl> --preference-report bundle=<aggregate-json>",
            "full_command": f"run_official_rationale_rl.py --execute-dpo --legacy-arm {item['name']} --smoke-completion <training_complete.json> --preference bundle=<restricted-jsonl> --preference-report bundle=<aggregate-json>",
        }
        for item in dpo.legacy_ablations
    ]
    return {
        "schema_version": "mal2026-official-rationale-rl-plan-v1",
        "end_to_end_experiment_runner": "scripts/run_official_rationale_rl_experiment.py",
        "runtime_versions": versions,
        "status": "ready" if isinstance(gate_evidence, dict) and "directional" in gate_evidence else "waiting_for_hard_gates",
        "hard_prerequisites": {
            "directional_contrastive_gate": str(gate_files[0]),
            "combined_prompt_injection_safety_gate": str(gate_files[1]),
            "evidence": gate_evidence,
        },
        "dpo": {
            "preference_stages": ["external_vllm_0.25.1_rollout", "exact_pinned_q4_judge", "tie_excluding_12_cell_assembly"],
            "training_arms": ["bundle", "content", "organization", "expression"],
            "official_execution": [
                {"task": task, "full_gpu": gpu, "preference_projection": "sum_of_all_12_integer_cells" if task == "bundle" else f"sum_of_4_integer_cells_for_{task}"}
                for gpu, task in enumerate(TASKS)
            ],
            "launch_order": "GPU0 one-task one-update smoke; only after its completion, read-only GPU conflict checks then four official full tasks in parallel",
            "trainer": dpo.policy["trainer"],
            "offline": True,
            "gpu_schedule": "preference rollout TP4 GPUs0-3, release; Q4 judge GPUs0-3, release; four DPO tasks one GPU each",
        },
        "grpo": {
            "trainer": grpo.policy["trainer"],
            "rollout_backend": grpo.policy["rollout_backend"],
            "integrated_vllm": False,
            "gpu_schedule": "one task at a time: rollout TP2 GPUs0-1, policy GPU2, exact Q4 reward GPU3",
        },
        "legacy_top3": list(dpo.legacy_ablations),
        "legacy_execution": legacy_execution,
        "legacy_interpretation": "method-replication ablations only; never direct official arms",
        "legacy_rank_check": "run the three pinned legacy warm starts through offline DPO on the same new official bundle preferences and fixed Q4 prompt; DPO isolates the warm-start method comparison without adding another online rollout protocol",
        "validation_policy": "validation is prohibited for preference construction and RL reward; evaluate only after frozen training",
    }


def execute_dpo(args: argparse.Namespace) -> None:
    preferences = task_paths(args.preference)
    reports = task_paths(args.preference_report)
    settings = RLSettings.from_json(DPO_CONFIG)
    settings.gate_evidence()
    if args.legacy_arm is not None:
        names = {item["name"] for item in settings.legacy_ablations}
        need(args.legacy_arm in names and set(preferences) == {"bundle"} and set(reports) == {"bundle"}, "legacy DPO requires one pinned arm and bundle preference/report")
        if not args.smoke:
            need(args.smoke_completion is not None, "legacy full DPO requires a completed one-update smoke")
            smoke_record = validate_smoke_completion(args.smoke_completion, settings)
            need(smoke_record.get("legacy_arm") == args.legacy_arm, "legacy smoke arm differs")
        require_gpu_idle(0)
        output = OUTPUT_ROOT / f"official-rationale-dpo-v1-legacy-{args.legacy_arm}-{'gpu0-smoke-001' if args.smoke else 'full-001'}"
        command = [
            str(ROOT / ".venv-standard/bin/python"), str(DPO_TRAINER), "--config", str(DPO_CONFIG),
            "--legacy-arm", args.legacy_arm, "--preferences", str(preferences["bundle"]),
            "--preference-report", str(reports["bundle"]), "--output-dir", str(output),
        ]
        if args.smoke:
            command += ["--max-steps", "1", "--train-limit", "2"]
        completed = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, text=True)
        need(completed.returncode == 0, "legacy DPO task failed")
        return
    if args.smoke:
        task = args.task or "bundle"
        need(set(preferences) == {task} and set(reports) == {task}, "DPO smoke requires only the selected task preference/report")
        require_gpu_idle(0)
        output = OUTPUT_ROOT / f"official-rationale-dpo-v1-{task}-gpu0-smoke-001"
        command = [
            str(ROOT / ".venv-standard/bin/python"), str(DPO_TRAINER), "--config", str(DPO_CONFIG),
            "--task", task, "--preferences", str(preferences[task]), "--preference-report", str(reports[task]),
            "--output-dir", str(output), "--max-steps", "1", "--train-limit", "2",
        ]
        completed = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, text=True)
        need(completed.returncode == 0, "DPO GPU0 smoke failed")
        return
    need(args.task is None and set(preferences) == set(TASKS) and set(reports) == set(TASKS), "official full DPO requires all four tasks")
    need(args.smoke_completion is not None, "official full DPO requires a completed one-update smoke")
    smoke_record = validate_smoke_completion(args.smoke_completion, settings)
    need(smoke_record.get("legacy_arm") is None and smoke_record.get("task") in TASKS, "official smoke lineage differs")
    for gpu in range(4):
        require_gpu_idle(gpu)
    processes: list[subprocess.Popen[str]] = []
    for gpu, task in enumerate(TASKS):
        require_gpu_idle(gpu)
        output = OUTPUT_ROOT / f"official-rationale-dpo-v1-{task}-full-001"
        command = [
            str(ROOT / ".venv-standard/bin/python"), str(DPO_TRAINER),
            "--config", str(DPO_CONFIG), "--task", task,
            "--preferences", str(preferences[task]), "--output-dir", str(output),
            "--preference-report", str(reports[task]),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        processes.append(subprocess.Popen(command, cwd=ROOT, env=environment, text=True))
    codes = [process.wait() for process in processes]
    need(all(code == 0 for code in codes), f"DPO task failures: {codes}")


def execute_grpo(args: argparse.Namespace) -> None:
    need(args.task in TASKS and args.output_dir and args.rollout_endpoint and args.rollout_model and args.rollout_attestation and args.judge_endpoint and args.judge_attestation, "GRPO runtime arguments are incomplete")
    settings = RLSettings.from_json(GRPO_CONFIG)
    settings.gate_evidence()
    command = [
        str(ROOT / ".venv-standard/bin/python"), str(GRPO_TRAINER),
        "--config", str(GRPO_CONFIG), "--task", args.task,
        "--output-dir", str(args.output_dir), "--rollout-endpoint", args.rollout_endpoint,
        "--rollout-model", args.rollout_model, "--rollout-attestation", str(args.rollout_attestation),
        "--judge-attestation", str(args.judge_attestation),
    ]
    for endpoint in args.judge_endpoint:
        command += ["--judge-endpoint", endpoint]
    if args.smoke:
        command += ["--max-steps", "1", "--train-limit", "8"]
    completed = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "2"}, text=True)
    need(completed.returncode == 0, "GRPO task failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-dpo", action="store_true")
    parser.add_argument("--execute-grpo", action="store_true")
    parser.add_argument("--preference", action="append", default=[])
    parser.add_argument("--preference-report", action="append", default=[])
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--legacy-arm")
    parser.add_argument("--smoke-completion", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rollout-endpoint")
    parser.add_argument("--rollout-model")
    parser.add_argument("--rollout-attestation", type=Path)
    parser.add_argument("--judge-endpoint", action="append", default=[])
    parser.add_argument("--judge-attestation", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    need(sum((args.dry_run, args.execute_dpo, args.execute_grpo)) == 1, "select exactly one runner mode")
    if args.dry_run:
        print(json.dumps(plan(), ensure_ascii=False, indent=2, sort_keys=True))
    elif args.execute_dpo:
        execute_dpo(args)
    else:
        execute_grpo(args)


if __name__ == "__main__":
    main()
