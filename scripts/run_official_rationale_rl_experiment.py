#!/usr/bin/env python3
"""Durable end-to-end executor for the official rationale RL experiment.

No server is adopted from an earlier process.  Every model server is started
inside a scoped context, attested after health checks, and stopped using only
the Popen objects and ownership token created by this runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import sha256_file  # noqa: E402
from mal2026.official_rationale_rl import (  # noqa: E402
    AXES,
    RLSettings,
    TASKS,
    legacy_ablation,
    legacy_grpo_producer_spec,
    output_fresh,
    restricted_fresh,
    validate_runtime_versions,
)
from mal2026.official_rl_servers import (  # noqa: E402
    PYTHON,
    assert_gpus_idle,
    q4_judge_servers,
    verify_server_prerequisites,
    vllm_policy_server,
)


DPO_CONFIG = ROOT / "configs/official_rationale_dpo.v1.json"
GRPO_CONFIG = ROOT / "configs/official_rationale_grpo.v1.json"
PREFERENCE = ROOT / "scripts/generate_official_dpo_preferences.py"
DPO_TRAINER = ROOT / "scripts/train_official_rationale_dpo.py"
GRPO_TRAINER = ROOT / "scripts/train_official_rationale_grpo.py"
OUTPUT_BASE = ROOT / "outputs/official-rationale-rl-v1/orchestration"
RESTRICTED_BASE = ROOT / "data/processed/restricted/official_rationale_rl_v1"
ALIASES = {task: f"official-rl-{task}" for task in TASKS}
LEGACY_GRPO_ARMS = (
    "midm_bundle_random1_v8_replication",
    "ax4_bundle_random1_v8_replication",
    "ax4_bundle_all5_v8_replication",
)
DPO_SMOKE_GROUPS = 32


class OfficialRLExperimentError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialRLExperimentError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class DurableRun:
    def __init__(
        self,
        run_id: str,
        scope: str,
        grpo_tasks: Sequence[str],
        grpo_phases: Sequence[str],
        legacy_grpo_arms: Sequence[str],
        dpo_config: Path = DPO_CONFIG,
        grpo_config: Path = GRPO_CONFIG,
    ):
        need(re.fullmatch(r"official-rationale-rl-experiment-v1-[a-z0-9][a-z0-9-]{5,100}", run_id) is not None, "run ID differs")
        self.run_id = run_id
        self.root = OUTPUT_BASE / run_id
        self.restricted = RESTRICTED_BASE / run_id
        self.ledger_path = self.root / "ledger.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.scope = scope
        self.grpo_tasks = tuple(grpo_tasks)
        self.grpo_phases = tuple(grpo_phases)
        self.legacy_grpo_arms = tuple(legacy_grpo_arms)
        self.dpo_config = dpo_config.resolve()
        self.grpo_config = grpo_config.resolve()
        self.dpo = RLSettings.from_json(self.dpo_config)
        self.grpo = RLSettings.from_json(self.grpo_config)
        need(self.dpo.judge["prompt_sha256"] == self.grpo.judge["prompt_sha256"], "DPO/GRPO judge prompt bindings differ")
        need(self.dpo.gate == self.grpo.gate, "DPO/GRPO gate declarations differ")
        self.judge_prompt_sha256 = str(self.dpo.judge["prompt_sha256"])
        self.gates = self.dpo.gate_evidence()
        validate_runtime_versions()

    def initialize(self) -> None:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        expected = {
            "schema_version": "mal2026-official-rationale-rl-experiment-v1",
            "run_id": self.run_id,
            "git_sha": git_sha,
            "command": [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *sys.argv[1:]],
            "config_sha256": {
                "dpo": sha256_file(self.dpo_config),
                "grpo": sha256_file(self.grpo_config),
            },
            "runtime_versions": validate_runtime_versions(),
            "scope": self.scope,
            "grpo_tasks": list(self.grpo_tasks),
            "grpo_phases": list(self.grpo_phases),
            "legacy_grpo_arms": list(self.legacy_grpo_arms),
            "gpu_scope": [0, 1, 2, 3],
            "contrastive_gate_sha256": self.gates["directional"]["sha256"],
            "rl_safety_gate_sha256": self.gates["combined_safety"]["sha256"],
            "judge_prompt_sha256": self.judge_prompt_sha256,
            "dpo_smoke_groups": DPO_SMOKE_GROUPS,
            "validation_used_for_preferences_or_reward": False,
        }
        if self.manifest_path.is_file():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            need(all(existing.get(key) == value for key, value in expected.items()), "resume manifest differs")
            existing.update({"status": "running", "resumed_at": now()})
            atomic_json(self.manifest_path, existing)
            return
        need(not self.root.exists() and not self.restricted.exists(), "fresh run roots differ")
        self.root.mkdir(mode=0o700, parents=True)
        self.restricted.mkdir(mode=0o700, parents=True)
        for name in ("logs", "attestations", "stages", "aggregates", "models"):
            (self.root / name).mkdir(mode=0o700)
        atomic_json(self.manifest_path, {**expected, "status": "running", "created_at": now()})
        self.event("experiment", "started", {"resume": False})

    def event(self, stage: str, event: str, evidence: Mapping[str, Any]) -> None:
        row = {
            "timestamp": now(), "run_id": self.run_id, "stage": stage, "event": event,
            "gpu_scope": [0, 1, 2, 3],
            "gpu_authorization": "repository default GPUs0-3 and explicit user request; GPUs4-7 never queried or used",
            "validation_used_for_preferences_or_reward": False,
            "evidence": dict(evidence),
        }
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def attempt(self, stage: str) -> int:
        if not self.ledger_path.is_file():
            return 1
        count = 0
        with self.ledger_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                count += int(row.get("stage") == stage and row.get("event") == "attempt_started")
        return count + 1

    def run_stage(self, stage: str, function: Callable[[int], Mapping[str, Any]]) -> dict[str, Any]:
        report_path = self.root / "stages" / f"{stage}.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            need(report.get("status") == "completed" and report.get("stage") == stage, f"completed stage report differs: {stage}")
            expected_report_sha: str | None = None
            with self.ledger_path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("stage") == stage and row.get("event") in {"completed", "recovered_atomic_completed_report"}:
                        expected_report_sha = row.get("evidence", {}).get("report_sha256")
            self.verify_evidence(report.get("evidence"))
            actual_report_sha = sha256_file(report_path)
            if expected_report_sha is None:
                self.event(stage, "recovered_atomic_completed_report", {"report_sha256": actual_report_sha})
            else:
                need(expected_report_sha == actual_report_sha, f"completed stage report digest differs: {stage}")
            self.event(stage, "resume_skip_completed", {"report_sha256": sha256_file(report_path)})
            return report
        attempt = self.attempt(stage)
        self.event(stage, "attempt_started", {"attempt": attempt})
        try:
            evidence = dict(function(attempt))
            report = {
                "schema_version": "mal2026-official-rationale-rl-stage-v1", "status": "completed",
                "stage": stage, "attempt": attempt, "completed_at": now(),
                "contrastive_gate_sha256": self.gates["directional"]["sha256"],
                "rl_safety_gate_sha256": self.gates["combined_safety"]["sha256"],
                "judge_prompt_sha256": self.judge_prompt_sha256,
                "evidence": evidence,
            }
            atomic_json(report_path, report)
            self.event(stage, "completed", {"attempt": attempt, "report_sha256": sha256_file(report_path)})
            return report
        except Exception as exc:
            self.event(stage, "failed", {"attempt": attempt, "failure_type": type(exc).__name__})
            raise

    def verify_evidence(self, value: Any) -> None:
        if isinstance(value, dict):
            for path_key, sha_key in (("raw", "raw_sha256"), ("raw", "sha256"), ("preferences", "preference_sha256"), ("preferences", "sha256"), ("completion", "completion_sha256")):
                if path_key in value and sha_key in value:
                    path = Path(str(value[path_key]))
                    need(path.is_file() and sha256_file(path) == value[sha_key], f"resume artifact digest differs: {path}")
            if "aggregate" in value:
                need(Path(str(value["aggregate"])).is_file(), "resume aggregate artifact is unavailable")
            for item in value.values():
                self.verify_evidence(item)
        elif isinstance(value, list):
            for item in value:
                self.verify_evidence(item)

    def command(self, stage: str, attempt: int, command: Sequence[str], *, gpus: Sequence[int] | None = None) -> None:
        log = self.root / "logs" / f"{stage}-attempt-{attempt:03d}.log"
        need(not log.exists(), "stage command log must be fresh")
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        if gpus is not None:
            visible = ",".join(str(gpu) for gpu in gpus)
            environment.update({"CUDA_VISIBLE_DEVICES": visible, "MAL2026_RESERVED_PHYSICAL_GPUS": visible})
        self.event(stage, "command", {"attempt": attempt, "argv": list(command), "physical_gpus": None if gpus is None else list(gpus)})
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True)
        need(completed.returncode == 0, f"stage command failed: {stage}")

    def finish(self, stage_reports: Mapping[str, Any]) -> None:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        payload.update({
            "status": "completed", "completed_at": now(),
            "completed_stages": sorted(stage_reports),
            "stage_report_sha256": {stage: sha256_file(self.root / "stages" / f"{stage}.json") for stage in stage_reports},
        })
        atomic_json(self.manifest_path, payload)
        completion_paths: set[Path] = set()
        preference_reports: set[Path] = set()

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                if "completion" in value:
                    completion_paths.add(Path(str(value["completion"])))
                if "aggregate" in value:
                    candidate = Path(str(value["aggregate"]))
                    if candidate.is_file():
                        parsed = json.loads(candidate.read_text(encoding="utf-8"))
                        if parsed.get("schema_version") == "mal2026-official-rationale-preference-aggregate-v1":
                            preference_reports.add(candidate)
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        for report in stage_reports.values():
            collect(report.get("evidence"))
        model_summaries = []
        for path in sorted(completion_paths):
            value = json.loads(path.read_text(encoding="utf-8"))
            model_summaries.append({
                "run_id": value.get("run_id"), "task": value.get("task"), "legacy_arm": value.get("legacy_arm"),
                "classification": value.get("classification"), "global_step": value.get("global_step"),
                "contract_shift": value.get("contract_shift"), "producer_status": value.get("producer_status"),
                "handoff_eligible": value.get("handoff_eligible"), "model_id": value.get("model_id"),
                "model_revision": value.get("model_revision"), "warm_start_adapter": value.get("warm_start_adapter"),
                "model_config_sha256": value.get("model_config_sha256"),
                "warm_start_adapter_model_sha256": value.get("warm_start_adapter_model_sha256"),
                "legacy_completion_sha256": value.get("legacy_completion_sha256"),
                "output_adapter": value.get("output_adapter"),
                "output_adapter_config_sha256": value.get("output_adapter_config_sha256"),
                "output_adapter_model_sha256": value.get("output_adapter_model_sha256"),
                "train_rows": value.get("train_rows"), "trainer": value.get("trl_trainer"),
                "train_loss": value.get("trainer_metrics", {}).get("train_loss"),
                "reward_summary": value.get("reward_summary"), "completion_sha256": sha256_file(path),
            })
        preference_summaries = []
        for path in sorted(preference_reports):
            value = json.loads(path.read_text(encoding="utf-8"))
            preference_summaries.append({
                "arm": value.get("arm"), "groups": value.get("groups"), "preference_rows": value.get("preference_rows"),
                "per_task_reward_variance": value.get("per_task_reward_variance"), "report_sha256": sha256_file(path),
            })
        aggregate = {
            "schema_version": "mal2026-official-rationale-rl-experiment-aggregate-v1",
            "status": "completed", "run_id": self.run_id, "scope": self.scope,
            "completed_stages": sorted(stage_reports),
            "contrastive_gate_sha256": self.gates["directional"]["sha256"],
            "rl_safety_gate_sha256": self.gates["combined_safety"]["sha256"],
            "judge_prompt_sha256": self.judge_prompt_sha256,
            "model_summaries": model_summaries,
            "preference_summaries": preference_summaries,
            "validation_used_for_preferences_or_reward": False,
            "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_scores_evidence_or_predictions",
        }
        atomic_json(self.root / "aggregate_experiment.json", aggregate)

    def fail(self, exc: Exception) -> None:
        if self.manifest_path.is_file():
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            payload.update({"status": "failed", "failed_at": now(), "failure_type": type(exc).__name__, "resume_policy": "completed stage reports are reusable; failed attempt artifacts are preserved; retry uses the next attempt number"})
            atomic_json(self.manifest_path, payload)


def wait_idle(gpus: Sequence[int], timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            assert_gpus_idle(gpus)
            return
        except Exception:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)


def policy_adapters(settings: RLSettings, tasks: Sequence[str]) -> dict[str, Path]:
    return {task: Path(settings.warm_starts[task]) for task in tasks}


def preference_command(
    run: DurableRun, stage: str, attempt: int, *, phase: str, arm: str,
    output: Path, aggregate: Path, input_path: Path | None = None,
    endpoint: str | None = None, attestation: Path | None = None,
    aliases: Mapping[str, str] | None = None, judge_endpoints: Sequence[str] = (), limit: int | None = None,
) -> list[str]:
    command = [str(PYTHON), str(PREFERENCE), "--config", str(run.dpo_config), "--stage", phase, "--arm", arm, "--output", str(output), "--aggregate-output", str(aggregate), "--max-inflight", "128"]
    if input_path is not None:
        command += ["--input", str(input_path)]
    if phase == "rollout":
        need(endpoint is not None and attestation is not None and aliases is not None, "rollout server arguments differ")
        command += ["--policy-endpoint", endpoint, "--policy-attestation", str(attestation)]
        for task, alias in aliases.items():
            command += ["--model", f"{task}={alias}"]
    elif phase == "judge":
        need(attestation is not None and bool(judge_endpoints), "judge server arguments differ")
        command += ["--judge-attestation", str(attestation)]
        for value in judge_endpoints:
            command += ["--judge-endpoint", value]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def dpo_pipeline(run: DurableRun) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}

    def smoke_rollout(attempt: int) -> Mapping[str, Any]:
        wait_idle((0,))
        raw = run.restricted / f"dpo-smoke-rollout-attempt-{attempt:03d}.jsonl"
        aggregate = run.root / "aggregates" / f"dpo-smoke-rollout-attempt-{attempt:03d}.json"
        with vllm_policy_server(runtime_root=run.root, label=f"dpo-smoke-rollout-a{attempt:03d}", gpus=(0,), port=19320,
                                  adapters=policy_adapters(run.dpo, ("bundle",)), aliases={"bundle": ALIASES["bundle"]},
                                  max_num_seqs=32, max_num_batched_tokens=8192, dynamic_updates=False) as (endpoint, attestation):
            run.command("dpo-smoke-rollout", attempt, preference_command(run, "dpo-smoke-rollout", attempt, phase="rollout", arm="bundle", output=raw, aggregate=aggregate,
                                                                 endpoint=endpoint, attestation=attestation, aliases={"bundle": ALIASES["bundle"]}, limit=DPO_SMOKE_GROUPS))
        return {"raw": str(raw), "raw_sha256": sha256_file(raw), "aggregate": str(aggregate), "groups": DPO_SMOKE_GROUPS}

    reports["dpo-smoke-rollout"] = run.run_stage("dpo-smoke-rollout", smoke_rollout)
    smoke_rollout_raw = Path(reports["dpo-smoke-rollout"]["evidence"]["raw"])

    def smoke_judge(attempt: int) -> Mapping[str, Any]:
        wait_idle((0,))
        raw = run.restricted / f"dpo-smoke-judge-attempt-{attempt:03d}.jsonl"
        aggregate = run.root / "aggregates" / f"dpo-smoke-judge-attempt-{attempt:03d}.json"
        with q4_judge_servers(runtime_root=run.root, label=f"dpo-smoke-a{attempt:03d}", gpus=(0,), ports=(19420,), judge_prompt_sha256=run.judge_prompt_sha256) as (endpoints, attestation):
            run.command("dpo-smoke-judge", attempt, preference_command(run, "dpo-smoke-judge", attempt, phase="judge", arm="bundle", output=raw, aggregate=aggregate,
                                                               input_path=smoke_rollout_raw, attestation=attestation, judge_endpoints=endpoints))
        return {"raw": str(raw), "raw_sha256": sha256_file(raw), "aggregate": str(aggregate), "judgments": 4 * DPO_SMOKE_GROUPS}

    reports["dpo-smoke-judge"] = run.run_stage("dpo-smoke-judge", smoke_judge)
    smoke_judged = Path(reports["dpo-smoke-judge"]["evidence"]["raw"])

    def smoke_assemble(attempt: int) -> Mapping[str, Any]:
        raw = run.restricted / f"dpo-smoke-preferences-attempt-{attempt:03d}.jsonl"
        aggregate = run.root / "aggregates" / f"dpo-smoke-preferences-attempt-{attempt:03d}.json"
        run.command("dpo-smoke-assemble", attempt, preference_command(run, "dpo-smoke-assemble", attempt, phase="assemble", arm="bundle", output=raw, aggregate=aggregate, input_path=smoke_judged))
        return {"preferences": str(raw), "preference_sha256": sha256_file(raw), "aggregate": str(aggregate)}

    reports["dpo-smoke-assemble"] = run.run_stage("dpo-smoke-assemble", smoke_assemble)
    smoke_preferences = Path(reports["dpo-smoke-assemble"]["evidence"]["preferences"])
    smoke_preference_report = Path(reports["dpo-smoke-assemble"]["evidence"]["aggregate"])

    def smoke_train(attempt: int) -> Mapping[str, Any]:
        wait_idle((0,))
        output = run.root / "models" / f"dpo-bundle-smoke-attempt-{attempt:03d}"
        command = [str(PYTHON), str(DPO_TRAINER), "--config", str(run.dpo_config), "--task", "bundle",
                   "--preferences", str(smoke_preferences), "--preference-report", str(smoke_preference_report),
                   "--output-dir", str(output), "--max-steps", "1", "--train-limit", "1"]
        run.command("dpo-smoke-train", attempt, command, gpus=(0,))
        complete = output / "training_complete.json"
        value = json.loads(complete.read_text(encoding="utf-8"))
        need(value.get("status") == "completed" and value.get("global_step") == 1, "DPO real smoke completion differs")
        return {"output": str(output), "completion": str(complete), "completion_sha256": sha256_file(complete)}

    reports["dpo-smoke-train"] = run.run_stage("dpo-smoke-train", smoke_train)

    def full_rollout(attempt: int) -> Mapping[str, Any]:
        wait_idle((0, 1, 2, 3))
        adapters = policy_adapters(run.dpo, TASKS)
        with vllm_policy_server(runtime_root=run.root, label=f"dpo-full-rollout-a{attempt:03d}", gpus=(0, 1, 2, 3), port=19321,
                                  adapters=adapters, aliases=ALIASES, max_num_seqs=256, max_num_batched_tokens=32768, dynamic_updates=False) as (endpoint, attestation):
            result: dict[str, Any] = {}
            for arm, tasks in (("bundle", ("bundle",)), ("axis_triplet", AXES)):
                raw = run.restricted / f"dpo-full-rollout-{arm}-attempt-{attempt:03d}.jsonl"
                aggregate = run.root / "aggregates" / f"dpo-full-rollout-{arm}-attempt-{attempt:03d}.json"
                run.command(f"dpo-full-rollout-{arm}", attempt, preference_command(run, f"dpo-full-rollout-{arm}", attempt, phase="rollout", arm=arm, output=raw, aggregate=aggregate,
                                                                                     endpoint=endpoint, attestation=attestation, aliases={task: ALIASES[task] for task in tasks}))
                result[arm] = {"raw": str(raw), "sha256": sha256_file(raw), "aggregate": str(aggregate)}
        return result

    reports["dpo-full-rollout"] = run.run_stage("dpo-full-rollout", full_rollout)

    def full_judge(attempt: int) -> Mapping[str, Any]:
        wait_idle((0, 1, 2, 3))
        with q4_judge_servers(runtime_root=run.root, label=f"dpo-full-a{attempt:03d}", gpus=(0, 1, 2, 3), ports=(19420, 19421, 19422, 19423), judge_prompt_sha256=run.judge_prompt_sha256) as (endpoints, attestation):
            result: dict[str, Any] = {}
            for arm in ("bundle", "axis_triplet"):
                source = Path(reports["dpo-full-rollout"]["evidence"][arm]["raw"])
                raw = run.restricted / f"dpo-full-judge-{arm}-attempt-{attempt:03d}.jsonl"
                aggregate = run.root / "aggregates" / f"dpo-full-judge-{arm}-attempt-{attempt:03d}.json"
                run.command(f"dpo-full-judge-{arm}", attempt, preference_command(run, f"dpo-full-judge-{arm}", attempt, phase="judge", arm=arm, output=raw, aggregate=aggregate,
                                                                                 input_path=source, attestation=attestation, judge_endpoints=endpoints))
                result[arm] = {"raw": str(raw), "sha256": sha256_file(raw), "aggregate": str(aggregate)}
        return result

    reports["dpo-full-judge"] = run.run_stage("dpo-full-judge", full_judge)

    def full_assemble(attempt: int) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for arm in ("bundle", "axis_triplet"):
            source = Path(reports["dpo-full-judge"]["evidence"][arm]["raw"])
            raw = run.restricted / f"dpo-full-preferences-{arm}-attempt-{attempt:03d}.jsonl"
            aggregate = run.root / "aggregates" / f"dpo-full-preferences-{arm}-attempt-{attempt:03d}.json"
            run.command(f"dpo-full-assemble-{arm}", attempt, preference_command(run, f"dpo-full-assemble-{arm}", attempt, phase="assemble", arm=arm, output=raw, aggregate=aggregate, input_path=source))
            value = json.loads(aggregate.read_text(encoding="utf-8"))
            need(value.get("status") == "completed" and value.get("reward_variance_gate_passed") is True, f"DPO full preference gate failed: {arm}")
            result[arm] = {"preferences": str(raw), "sha256": sha256_file(raw), "aggregate": str(aggregate)}
        return result

    reports["dpo-full-assemble"] = run.run_stage("dpo-full-assemble", full_assemble)
    bundle = reports["dpo-full-assemble"]["evidence"]["bundle"]
    axis = reports["dpo-full-assemble"]["evidence"]["axis_triplet"]

    def official_train(attempt: int) -> Mapping[str, Any]:
        wait_idle((0, 1, 2, 3))
        processes: list[tuple[str, subprocess.Popen[str], Any, Path]] = []
        for gpu, task in enumerate(TASKS):
            assert_gpus_idle((gpu,))
            source = bundle if task == "bundle" else axis
            output = run.root / "models" / f"dpo-official-{task}-attempt-{attempt:03d}"
            log = run.root / "logs" / f"dpo-official-{task}-attempt-{attempt:03d}.log"
            command = [str(PYTHON), str(DPO_TRAINER), "--config", str(run.dpo_config), "--task", task,
                       "--preferences", source["preferences"], "--preference-report", source["aggregate"], "--output-dir", str(output)]
            environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": str(gpu), "MAL2026_RESERVED_PHYSICAL_GPUS": str(gpu)}
            handle = log.open("x", encoding="utf-8")
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True)
            processes.append((task, process, handle, output))
        failures: dict[str, int] = {}
        outputs: dict[str, Any] = {}
        for task, process, handle, output in processes:
            code = process.wait(); handle.close()
            if code != 0:
                failures[task] = code
            else:
                complete = output / "training_complete.json"
                outputs[task] = {"output": str(output), "completion": str(complete), "completion_sha256": sha256_file(complete)}
        need(not failures, f"official parallel DPO failures: {failures}")
        return outputs

    reports["dpo-official-train"] = run.run_stage("dpo-official-train", official_train)

    for legacy in run.dpo.legacy_ablations:
        name = str(legacy["name"])

        def legacy_smoke(attempt: int, legacy_name: str = name) -> Mapping[str, Any]:
            wait_idle((0,))
            output = run.root / "models" / f"dpo-legacy-{legacy_name}-smoke-attempt-{attempt:03d}"
            command = [str(PYTHON), str(DPO_TRAINER), "--config", str(run.dpo_config), "--legacy-arm", legacy_name,
                       "--preferences", bundle["preferences"], "--preference-report", bundle["aggregate"],
                       "--output-dir", str(output), "--max-steps", "1", "--train-limit", "2"]
            run.command(f"dpo-legacy-{legacy_name}-smoke", attempt, command, gpus=(0,))
            complete = output / "training_complete.json"
            value = json.loads(complete.read_text(encoding="utf-8"))
            need(value.get("legacy_arm") == legacy_name and value.get("global_step") == 1, "legacy DPO smoke differs")
            return {"output": str(output), "completion": str(complete), "completion_sha256": sha256_file(complete)}

        smoke_stage = f"dpo-legacy-{name}-smoke"
        reports[smoke_stage] = run.run_stage(smoke_stage, legacy_smoke)

        def legacy_full(attempt: int, legacy_name: str = name) -> Mapping[str, Any]:
            wait_idle((0,))
            output = run.root / "models" / f"dpo-legacy-{legacy_name}-full-attempt-{attempt:03d}"
            command = [str(PYTHON), str(DPO_TRAINER), "--config", str(run.dpo_config), "--legacy-arm", legacy_name,
                       "--preferences", bundle["preferences"], "--preference-report", bundle["aggregate"], "--output-dir", str(output)]
            run.command(f"dpo-legacy-{legacy_name}-full", attempt, command, gpus=(0,))
            complete = output / "training_complete.json"
            return {"output": str(output), "completion": str(complete), "completion_sha256": sha256_file(complete),
                    "bound": "one frozen DPO epoch over the same official bundle preference file"}

        full_stage = f"dpo-legacy-{name}-full"
        reports[full_stage] = run.run_stage(full_stage, legacy_full)
    return reports


def failed_legacy_grpo_producer(
    run: DurableRun, *, stage: str, phase: str, attempt: int, legacy_name: str,
    failure_stage: str, exc: Exception,
) -> Path:
    """Persist a non-handoff producer record before propagating the failure."""
    legacy = legacy_ablation(run.grpo, legacy_name)
    path = run.root / "aggregates" / f"{stage}-producer-status-attempt-{attempt:03d}.json"
    need(not path.exists(), "legacy GRPO producer status must be fresh")
    atomic_json(path, {
        "schema_version": "mal2026-official-rationale-grpo-producer-status-v1",
        "status": "failed_producer",
        "producer_status": "failed_producer",
        "handoff_eligible": False,
        "run_id": run.run_id,
        "stage": stage,
        "phase": phase,
        "task": "bundle",
        "legacy_arm": legacy_name,
        "classification": legacy["classification"],
        "contract_shift": legacy["contract_shift"],
        "model_id": legacy["model_id"],
        "model_revision": legacy["model_revision"],
        "model_path": legacy["model_path"],
        "warm_start_adapter": legacy["adapter_path"],
        "warm_start_adapter_model_sha256": legacy["adapter_model_sha256"],
        "legacy_completion_path": legacy["completion_path"],
        "legacy_completion_sha256": legacy["completion_sha256"],
        "failure_type": type(exc).__name__,
        "failure_stage": failure_stage,
        "failure_detail": str(exc),
        "training_contract": "public_spec_aligned_score_conditioned_rationale_only_descriptive_no_improvement_advice",
        "validation_used_for_reward_or_training": False,
    })
    run.event(stage, "producer_failed_closed", {
        "attempt": attempt,
        "producer_status": str(path),
        "producer_status_sha256": sha256_file(path),
        "failure_type": type(exc).__name__,
        "handoff_eligible": False,
    })
    return path


def grpo_one(
    run: DurableRun,
    task: str,
    phase: str,
    reports: dict[str, dict[str, Any]],
    *,
    legacy_name: str | None = None,
) -> None:
    max_steps = 1 if phase == "smoke" else int(run.grpo.policy["pilot_max_steps"] if phase == "pilot" else run.grpo.policy["max_steps"])
    train_limit = 8 if phase == "smoke" else int(run.grpo.policy["pilot_train_limit"] if phase == "pilot" else run.grpo.policy["full_train_limit"])
    stage = f"grpo-legacy-{legacy_name}-{phase}" if legacy_name is not None else f"grpo-{task}-{phase}"

    def execute(attempt: int) -> Mapping[str, Any]:
        producer: Mapping[str, Any] | None = None
        try:
            producer = legacy_grpo_producer_spec(run.grpo, legacy_name) if legacy_name is not None else None
            wait_idle((0, 1, 2, 3))
            output_name = f"grpo-legacy-{legacy_name}-{phase}" if legacy_name is not None else f"grpo-{task}-{phase}"
            output = run.root / "models" / f"{output_name}-attempt-{attempt:03d}"
            adapter = Path(str(producer["warm_start_adapter"])) if producer else Path(run.grpo.warm_starts[task])
            alias = f"official-rl-legacy-{legacy_name}" if legacy_name is not None else ALIASES[task]
            model_path = Path(str(producer["model_path"])) if producer else None
            model_id = str(producer["model_id"]) if producer else None
            model_revision = str(producer["model_revision"]) if producer else None
            server_kwargs: dict[str, Any] = {}
            if producer:
                server_kwargs = {"model_path": model_path, "model_id": model_id, "model_revision": model_revision}
            with vllm_policy_server(runtime_root=run.root, label=f"{output_name}-rollout-a{attempt:03d}", gpus=(0, 1), port=19330,
                                      adapters={task: adapter}, aliases={task: alias}, max_num_seqs=192, max_num_batched_tokens=65536,
                                      dynamic_updates=True, **server_kwargs) as (rollout_endpoint, rollout_attestation):
                with q4_judge_servers(runtime_root=run.root, label=f"{output_name}-reward-a{attempt:03d}", gpus=(3,), ports=(19430,), judge_prompt_sha256=run.judge_prompt_sha256) as (judge_endpoints, judge_attestation):
                    assert_gpus_idle((2,))
                    selector = ["--legacy-arm", legacy_name] if legacy_name is not None else ["--task", task]
                    command = [str(PYTHON), str(GRPO_TRAINER), "--config", str(run.grpo_config), *selector,
                               "--output-dir", str(output), "--rollout-endpoint", rollout_endpoint, "--rollout-model", alias,
                               "--rollout-attestation", str(rollout_attestation), "--judge-attestation", str(judge_attestation),
                               "--judge-endpoint", judge_endpoints[0], "--train-limit", str(train_limit), "--max-steps", str(max_steps)]
                    run.command(stage, attempt, command, gpus=(2,))
            complete = output / "training_complete.json"
            value = json.loads(complete.read_text(encoding="utf-8"))
            need(
                value.get("status") == "completed"
                and value.get("producer_status") == "completed"
                and value.get("handoff_eligible") is True
                and value.get("global_step") == max_steps
                and value.get("integrated_vllm") is False,
                "GRPO completion differs",
            )
            output_adapter = output / "adapter"
            need(value.get("output_adapter") == str(output_adapter.resolve()), "GRPO output-adapter handoff path differs")
            need(sha256_file(output_adapter / "adapter_config.json") == value.get("output_adapter_config_sha256"), "GRPO output adapter-config digest differs")
            need(sha256_file(output_adapter / "adapter_model.safetensors") == value.get("output_adapter_model_sha256"), "GRPO output adapter-model digest differs")
            if legacy_name is not None:
                need(value.get("legacy_arm") == legacy_name and value.get("model_id") == producer["model_id"], "legacy GRPO completion identity differs")
                need(value.get("legacy_completion_sha256") == producer["legacy_completion_sha256"], "legacy GRPO source completion binding differs")
                need(value.get("warm_start_adapter_model_sha256") == producer["warm_start_adapter_model_sha256"], "legacy GRPO warm-start adapter binding differs")
            return {
                "output": str(output), "completion": str(complete), "completion_sha256": sha256_file(complete),
                "producer_status": "completed", "handoff_eligible": True,
                "max_steps": max_steps, "train_limit": train_limit,
                "gpu_topology": {"rollout": [0, 1], "trainer": [2], "reward": [3]}, "integrated_vllm": False,
                "producer": producer,
            }
        except Exception as exc:
            if legacy_name is not None:
                failed_legacy_grpo_producer(
                    run, stage=stage, phase=phase, attempt=attempt, legacy_name=legacy_name,
                    failure_stage="static_compatibility" if producer is None else "real_producer_execution",
                    exc=exc,
                )
            raise

    reports[stage] = run.run_stage(stage, execute)


def grpo_pipeline(run: DurableRun) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for task in run.grpo_tasks:
        grpo_one(run, task, "smoke", reports)
        for phase in run.grpo_phases:
            grpo_one(run, task, phase, reports)
    for legacy_name in run.legacy_grpo_arms:
        grpo_one(run, "bundle", "smoke", reports, legacy_name=legacy_name)
        for phase in run.grpo_phases:
            grpo_one(run, "bundle", phase, reports, legacy_name=legacy_name)
    return reports


def dry_plan(args: argparse.Namespace) -> dict[str, Any]:
    dpo_config = Path(args.dpo_config).resolve()
    grpo_config = Path(args.grpo_config).resolve()
    dpo, grpo = RLSettings.from_json(dpo_config), RLSettings.from_json(grpo_config)
    need(dpo.judge["prompt_sha256"] == grpo.judge["prompt_sha256"], "DPO/GRPO judge prompt bindings differ")
    gate_paths = (
        [ROOT / str(dpo.gate[key]) for key in ("authorization_path", "legacy_directional_path", "legacy_failed_safety_path")]
        if dpo.gate.get("mode") == "explicit_user_authorization_after_preserved_v1_failure"
        else [ROOT / str(dpo.gate[key]) for key in ("path", "safety_path")]
    )
    requested_legacy = list(getattr(args, "legacy_grpo_arm", None) or LEGACY_GRPO_ARMS)
    legacy_producers = [legacy_grpo_producer_spec(grpo, name) for name in requested_legacy]
    sequence = ["smoke", *args.grpo_phase]
    producer_stage_plan = {
        "official:bundle": [f"grpo-bundle-{phase}" for phase in sequence],
        **{name: [f"grpo-legacy-{name}-{phase}" for phase in sequence] for name in requested_legacy},
    }
    return {
        "schema_version": "mal2026-official-rationale-rl-experiment-plan-v1",
        "run_id": args.run_id, "scope": args.scope,
        "config_sha256": {"dpo": sha256_file(dpo_config), "grpo": sha256_file(grpo_config)},
        "judge_prompt_sha256": dpo.judge["prompt_sha256"],
        "judge_prompt_kind": dpo.judge["prompt_kind"],
        "gate_mode": dpo.gate.get("mode", "legacy_combined_safety_gate"),
        "gate_files_present": all(path.is_file() for path in gate_paths),
        "gpu_scope": [0, 1, 2, 3], "gpu_queries_in_dry_run": False,
        "dpo_stages": [
            f"GPU0 {DPO_SMOKE_GROUPS}-group policy rollout", f"GPU0 {4 * DPO_SMOKE_GROUPS}-candidate exact-Q4 judgment", "bundle preference assembly",
            "GPU0 one-update DPO trainer smoke", "TP4 full bundle+axis rollout", "four-replica exact-Q4 full judgment",
            "bundle 12-cell and per-axis 4-cell assembly", "four official DPO tasks parallel", "three pinned legacy DPO smoke+full pairs",
        ],
        "grpo": {
            "official_best": "bundle",
            "official_tasks": args.grpo_task,
            "legacy_top3": requested_legacy,
            "per_producer_sequence": ["real_one_update_smoke", *args.grpo_phase],
            "producer_stage_plan": producer_stage_plan,
            "phases_after_each_real_smoke": args.grpo_phase,
            "topology": {"rollout_tp2": [0, 1], "trainer": [2], "q4_reward": [3]},
            "integrated_vllm": False,
            "training_contract": "public_spec_aligned_score_conditioned_rationale_only_descriptive_no_improvement_advice",
            "exact_q4_reward": True,
            "legacy_producers": legacy_producers,
        },
        "server_contract": {"enforce_eager": False, "vllm_max_num_seqs_dpo": 256, "vllm_max_batched_tokens_dpo": 32768, "q4_parallel_per_gpu": 4,
                            "owned_process_stop_only": True, "sequential_policy_then_q4_for_dpo": True},
        "legacy_arms": [item["name"] for item in dpo.legacy_ablations],
        "validation_used_for_preferences_or_reward": False,
        "resume": "append-only attempts; completed stage reports skip; failed attempt artifacts preserved",
        "runtime_versions": validate_runtime_versions(),
        "grpo_bounds": {key: grpo.policy[key] for key in ("pilot_max_steps", "pilot_train_limit", "max_steps", "full_train_limit")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dpo-config", type=Path, default=DPO_CONFIG)
    parser.add_argument("--grpo-config", type=Path, default=GRPO_CONFIG)
    parser.add_argument("--scope", choices=("dpo", "grpo", "all"), default="all")
    parser.add_argument("--grpo-task", action="append", choices=TASKS)
    parser.add_argument("--grpo-phase", action="append", choices=("pilot", "full"))
    parser.add_argument("--legacy-grpo-arm", action="append", choices=LEGACY_GRPO_ARMS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.grpo_task = args.grpo_task or ["bundle"]
    args.grpo_phase = args.grpo_phase or ["pilot", "full"]
    args.legacy_grpo_arm = args.legacy_grpo_arm or list(LEGACY_GRPO_ARMS)
    need(
        len(set(args.grpo_task)) == len(args.grpo_task)
        and len(set(args.grpo_phase)) == len(args.grpo_phase)
        and len(set(args.legacy_grpo_arm)) == len(args.legacy_grpo_arm),
        "GRPO declarations contain duplicates",
    )
    if args.dry_run:
        print(json.dumps(dry_plan(args), ensure_ascii=False, indent=2, sort_keys=True))
        return
    verify_server_prerequisites()
    run = DurableRun(
        args.run_id,
        args.scope,
        args.grpo_task,
        args.grpo_phase,
        args.legacy_grpo_arm,
        args.dpo_config,
        args.grpo_config,
    )
    reports: dict[str, dict[str, Any]] = {}
    try:
        run.initialize()
        if args.scope in {"dpo", "all"}:
            reports.update(dpo_pipeline(run))
        if args.scope in {"grpo", "all"}:
            reports.update(grpo_pipeline(run))
        run.finish(reports)
        print(json.dumps({"status": "completed", "run_id": args.run_id, "stages": len(reports)}, sort_keys=True))
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
