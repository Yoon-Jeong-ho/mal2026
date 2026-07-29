#!/usr/bin/env python3
"""Compare the frozen SFT and bundle-DPO rationale adapters on validation.

This runner deliberately evaluates only the bundled three-axis participant
contract used by the final judge.  It does not create or rank axis-triplet
arms.  Row-level generations and judge records stay under restricted data;
only aggregate comparison evidence is written under outputs/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import sha256_file  # noqa: E402
from mal2026.official_rationale_rl import (  # noqa: E402
    EXACT_JUDGE_PROMPT_PATH,
    EXACT_JUDGE_PROMPT_SHA256,
    MODEL_ID,
)
from mal2026.official_rl_servers import (  # noqa: E402
    PYTHON,
    assert_gpus_idle,
    q4_judge_servers,
    vllm_policy_server,
)
from mal2026.official_writing_contract import AXES, JUDGE_DIMENSIONS  # noqa: E402


DEFAULT_RUN_ID = "official-rationale-dpo-bundle-validation-exact-judge-20260729-020"
RESTRICTED_BASE = ROOT / "data/processed/restricted/official_rationale_rl_v1"
OUTPUT_BASE = ROOT / "outputs/official-rationale-rl-v1/evaluation"
Q4_RESTRICTED_BASE = ROOT / "data/processed/restricted/official_prompt_alignment_v1/q4_judge"
Q4_OUTPUT_BASE = ROOT / "outputs/official-prompt-alignment-v1/q4-judge"

SCORES = ROOT / (
    "data/processed/restricted/official_prompt_alignment_v1/score_predictions/"
    "official-score-essay-only-full-20260727-002/essay_only_epoch_04.jsonl"
)
BASELINE_GENERATION = ROOT / (
    "data/processed/restricted/official_prompt_alignment_v1/rationale_generation/"
    "official-rationale-generation-v1-ax4-bundle-validation-004"
)
BASELINE_PARTICIPANT = ROOT / (
    "data/processed/restricted/official_prompt_alignment_v1/participants/"
    "official-rationale-ax4-bundle-validation-001.jsonl"
)
DPO_ADAPTER = ROOT / (
    "outputs/official-rationale-rl-v1/orchestration/"
    "official-rationale-rl-experiment-v1-exact-judge-20260729-019/models/"
    "dpo-official-bundle-ddp4-full-user-aligned-001/adapter"
)
DPO_COMPLETION = DPO_ADAPTER.parent / "training_complete.json"
GENERATOR = ROOT / "scripts/generate_official_rationales_vllm.py"
COMPOSER = ROOT / "scripts/compose_official_participants.py"
JUDGE = ROOT / "scripts/evaluate_official_q4_judge.py"
MODEL_ALIAS = "official-dpo-bundle"
EXPECTED = 400


class ValidationError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_command(command: Sequence[str], log_path: Path) -> None:
    need(not log_path.exists(), f"log must be fresh: {log_path}")
    log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    need(completed.returncode == 0, f"command failed ({completed.returncode}): {log_path}")


def wait_for_owned_gpu_release(gpus: Sequence[int], timeout_seconds: int = 120) -> None:
    """Wait out CUDA teardown after an owned server exits.

    The initial launch still uses the immediate conflict gate.  This retry is
    only used between two stages owned by this runner, where CUDA process
    accounting can briefly outlive the parent process.
    """
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            assert_gpus_idle(gpus)
            return
        except Exception as exc:  # exact exception type is intentionally not hidden at timeout
            last_error = exc
            time.sleep(1)
    raise ValidationError(f"owned GPU contexts did not release: {last_error}")


def copy_first_jsonl_row(source: Path, destination: Path) -> None:
    need(not destination.exists(), f"smoke input must be fresh: {destination}")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as handle:
        first = handle.readline()
    need(bool(first.strip()), f"canonical input is empty: {source}")
    # Parse before copying so a malformed line cannot become smoke evidence.
    json.loads(first)
    destination.write_text(first if first.endswith("\n") else first + "\n", encoding="utf-8")


def generator_attestation(
    *, destination: Path, endpoint: str, policy_attestation: Path
) -> Path:
    need(not destination.exists(), "generator attestation must be fresh")
    atomic_json(
        destination,
        {
            "schema_version": "mal2026-official-rationale-vllm-server-attestation-v1",
            "created_at": now(),
            "endpoint": endpoint,
            "adapter_aliases": {MODEL_ALIAS: "bundle"},
            "base_server_attestation_sha256": sha256_file(policy_attestation),
            "data_split": "validation",
            "train_split_only": False,
        },
    )
    return destination


def generation_command(
    *, run_id: str, score_file: Path, output_dir: Path, endpoint: str,
    attestation: Path, expected: int, max_inflight: int,
) -> list[str]:
    return [
        str(PYTHON), str(GENERATOR), "--run-id", run_id,
        "--task", "bundle", "--split", "validation", "--expected", str(expected),
        "--score-file", str(score_file), "--output-dir", str(output_dir),
        "--endpoint", endpoint, "--model", MODEL_ALIAS,
        "--server-attestation", str(attestation), "--max-inflight", str(max_inflight),
    ]


def compose_command(
    *, run_id: str, score_file: Path, generation_dir: Path,
    output_file: Path, aggregate_file: Path, expected: int,
) -> list[str]:
    return [
        str(PYTHON), str(COMPOSER), "--run-id", run_id,
        "--score-file", str(score_file), "--generation-dir", str(generation_dir),
        "--output-file", str(output_file), "--aggregate-file", str(aggregate_file),
        "--expected", str(expected),
    ]


def judge_command(
    *, run_id: str, participant: Path, endpoints: Sequence[str],
    attestation: Path, expected: int,
) -> list[str]:
    command = [
        str(PYTHON), str(JUDGE), "--run-id", run_id,
        "--participant-file", str(participant), "--expected", str(expected),
        "--split", "validation", "--max-inflight", "16",
        "--server-attestation", str(attestation),
        "--system-prompt-file", str(EXACT_JUDGE_PROMPT_PATH),
    ]
    for endpoint in endpoints:
        command += ["--endpoint", endpoint]
    return command


def load_judge_rows(run_id: str, expected: int) -> dict[str, Mapping[str, Any]]:
    path = Q4_RESTRICTED_BASE / run_id / "judge_records.jsonl"
    rows: dict[str, Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            need(
                raw.get("failure_category") is None and isinstance(raw.get("judge_output"), dict),
                f"judge row failed in {run_id}",
            )
            source_id = str(raw["source_id"])
            need(source_id not in rows, f"duplicate judge source in {run_id}")
            rows[source_id] = raw["judge_output"]
    need(len(rows) == expected, f"judge population differs in {run_id}")
    return rows


def score_cells(output: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    cells: dict[str, dict[str, int]] = {}
    for axis in AXES:
        cells[axis] = {}
        for dimension in JUDGE_DIMENSIONS:
            value = output[axis][dimension]["score"]
            need(type(value) is int and 1 <= value <= 5, "judge cell score differs")
            cells[axis][dimension] = value
    return cells


def sign_test_two_sided(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def comparison(baseline_run: str, dpo_run: str) -> dict[str, Any]:
    baseline_report_path = Q4_OUTPUT_BASE / baseline_run / "aggregate_judge_report.json"
    dpo_report_path = Q4_OUTPUT_BASE / dpo_run / "aggregate_judge_report.json"
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    dpo_report = json.loads(dpo_report_path.read_text(encoding="utf-8"))
    need(
        baseline_report.get("status") == dpo_report.get("status") == "completed",
        "full judge hard gates did not complete",
    )
    need(
        baseline_report.get("judge_system_prompt_sha256")
        == dpo_report.get("judge_system_prompt_sha256")
        == EXACT_JUDGE_PROMPT_SHA256,
        "judge prompt binding differs",
    )
    baseline = load_judge_rows(baseline_run, EXPECTED)
    dpo = load_judge_rows(dpo_run, EXPECTED)
    need(set(baseline) == set(dpo), "paired validation IDs differ")

    totals: dict[str, list[int]] = {"baseline": [], "dpo": []}
    cell_values: dict[str, dict[str, dict[str, list[int]]]] = {
        name: {axis: {dimension: [] for dimension in JUDGE_DIMENSIONS} for axis in AXES}
        for name in ("baseline", "dpo")
    }
    differences: list[int] = []
    for source_id in sorted(baseline):
        paired = {
            "baseline": score_cells(baseline[source_id]),
            "dpo": score_cells(dpo[source_id]),
        }
        row_totals: dict[str, int] = {}
        for name in ("baseline", "dpo"):
            row_totals[name] = sum(
                paired[name][axis][dimension]
                for axis in AXES for dimension in JUDGE_DIMENSIONS
            )
            totals[name].append(row_totals[name])
            for axis in AXES:
                for dimension in JUDGE_DIMENSIONS:
                    cell_values[name][axis][dimension].append(paired[name][axis][dimension])
        differences.append(row_totals["dpo"] - row_totals["baseline"])

    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    ties = sum(value == 0 for value in differences)
    cell_means = {
        name: {
            axis: {
                dimension: statistics.fmean(cell_values[name][axis][dimension])
                for dimension in JUDGE_DIMENSIONS
            }
            for axis in AXES
        }
        for name in ("baseline", "dpo")
    }
    axis_means = {
        name: {
            axis: statistics.fmean(cell_means[name][axis].values()) for axis in AXES
        }
        for name in ("baseline", "dpo")
    }
    macro = {
        name: statistics.fmean(
            cell_means[name][axis][dimension]
            for axis in AXES for dimension in JUDGE_DIMENSIONS
        )
        for name in ("baseline", "dpo")
    }
    delta_macro = macro["dpo"] - macro["baseline"]
    selected = "dpo" if delta_macro > 0 else "baseline_sft"
    return {
        "schema_version": "mal2026-official-bundle-dpo-validation-comparison-v1",
        "status": "completed",
        "completed_at": now(),
        "contract": "single_bundled_three_axis_participant_json",
        "axis_triplet_used_for_training_or_selection": False,
        "validation_records": EXPECTED,
        "judge_prompt_sha256": EXACT_JUDGE_PROMPT_SHA256,
        "baseline": {
            "run_id": baseline_run,
            "participant_sha256": baseline_report["participant_sha256"],
            "judge_records_sha256": baseline_report["judge_records_sha256"],
            "macro_mean_5point": macro["baseline"],
            "mean_total_60point": statistics.fmean(totals["baseline"]),
            "perfect_total_rate": sum(value == 60 for value in totals["baseline"]) / EXPECTED,
            "axis_means": axis_means["baseline"],
            "cell_means": cell_means["baseline"],
        },
        "dpo": {
            "run_id": dpo_run,
            "participant_sha256": dpo_report["participant_sha256"],
            "judge_records_sha256": dpo_report["judge_records_sha256"],
            "macro_mean_5point": macro["dpo"],
            "mean_total_60point": statistics.fmean(totals["dpo"]),
            "perfect_total_rate": sum(value == 60 for value in totals["dpo"]) / EXPECTED,
            "axis_means": axis_means["dpo"],
            "cell_means": cell_means["dpo"],
        },
        "paired": {
            "dpo_wins": wins,
            "ties": ties,
            "dpo_losses": losses,
            "mean_total_delta_dpo_minus_baseline": statistics.fmean(differences),
            "median_total_delta_dpo_minus_baseline": statistics.median(differences),
            "two_sided_exact_sign_test_p": sign_test_two_sided(wins, losses),
            "macro_delta_dpo_minus_baseline": delta_macro,
            "axis_delta_dpo_minus_baseline": {
                axis: axis_means["dpo"][axis] - axis_means["baseline"][axis]
                for axis in AXES
            },
        },
        "selection": {
            "selected": selected,
            "rule": "higher frozen-validation exact-bundle-judge macro; retain SFT on a tie",
            "dpo_improved": delta_macro > 0,
        },
        "input_sha256": {
            "scores": sha256_file(SCORES),
            "baseline_participant": sha256_file(BASELINE_PARTICIPANT),
            "dpo_adapter_config": sha256_file(DPO_ADAPTER / "adapter_config.json"),
            "dpo_training_completion": sha256_file(DPO_COMPLETION),
        },
        "privacy": "aggregate_only_no_rows_prompts_essays_rationales_ids_or_predictions",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--resume-after-generation",
        action="store_true",
        help="Resume the same run after completed generation/composition artifacts.",
    )
    args = parser.parse_args()
    run_id = args.run_id
    need(
        run_id.startswith("official-rationale-dpo-bundle-validation-exact-judge-")
        and len(run_id) <= 110,
        "run ID differs",
    )
    restricted = RESTRICTED_BASE / run_id
    output = OUTPUT_BASE / run_id
    if args.resume_after_generation:
        need(restricted.is_dir() and output.is_dir(), "resume roots are unavailable")
    else:
        need(not restricted.exists() and not output.exists(), "validation run roots must be fresh")
    required_files = (
        SCORES, BASELINE_PARTICIPANT, DPO_COMPLETION, DPO_ADAPTER / "adapter_config.json",
        EXACT_JUDGE_PROMPT_PATH, GENERATOR, COMPOSER, JUDGE,
        BASELINE_GENERATION / "aggregate_generation_report.json",
    )
    need(all(path.is_file() and not path.is_symlink() for path in required_files), "canonical validation prerequisite is unavailable")
    need(sha256_file(EXACT_JUDGE_PROMPT_PATH) == EXACT_JUDGE_PROMPT_SHA256, "exact judge prompt changed")
    completion = json.loads(DPO_COMPLETION.read_text(encoding="utf-8"))
    need(
        completion.get("status") == "completed"
        and completion.get("task") == "bundle"
        and completion.get("split") == "train"
        and completion.get("validation_used_for_preferences_or_training") is False,
        "DPO training completion lineage differs",
    )

    if args.resume_after_generation:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        need(
            manifest.get("schema_version") == "mal2026-official-bundle-dpo-validation-run-v1"
            and manifest.get("status") == "running"
            and manifest.get("run_id") == run_id
            and manifest.get("score_file_sha256") == sha256_file(SCORES)
            and manifest.get("dpo_training_completion_sha256") == sha256_file(DPO_COMPLETION)
            and manifest.get("judge_prompt_sha256") == EXACT_JUDGE_PROMPT_SHA256,
            "resume manifest differs",
        )
        manifest["resumed_after_generation_at"] = now()
        manifest["resume_reason"] = (
            "runner_exited_between_completed_generation_and_q4_launch; "
            "terminal exception text was unavailable, so the exact cause remains uncertain"
        )
        atomic_json(output / "manifest.json", manifest)
    else:
        restricted.mkdir(mode=0o700, parents=True)
        output.mkdir(mode=0o700, parents=True)
        for name in ("logs", "attestations", "aggregates"):
            (output / name).mkdir(mode=0o700)
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        manifest = {
            "schema_version": "mal2026-official-bundle-dpo-validation-run-v1",
            "status": "running",
            "run_id": run_id,
            "created_at": now(),
            "git_sha": git_sha,
            "command": [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *sys.argv[1:]],
            "gpu_scope": [0, 1, 2, 3],
            "gpu_authorization": "repository default GPUs0-3 and explicit user request; GPUs4-7 not queried or used",
            "split": "validation",
            "validation_used_for_training_or_reward": False,
            "contract": "single_bundled_three_axis_participant_json",
            "axis_triplet_used_for_training_or_selection": False,
            "expected": EXPECTED,
            "seed": 42,
            "score_file_sha256": sha256_file(SCORES),
            "baseline_participant_sha256": sha256_file(BASELINE_PARTICIPANT),
            "dpo_training_completion_sha256": sha256_file(DPO_COMPLETION),
            "judge_prompt_sha256": EXACT_JUDGE_PROMPT_SHA256,
        }
        atomic_json(output / "manifest.json", manifest)

    smoke_scores = restricted / "smoke_score.jsonl"
    if args.resume_after_generation:
        need(smoke_scores.is_file() and not smoke_scores.is_symlink(), "resume smoke score is unavailable")
    else:
        copy_first_jsonl_row(SCORES, smoke_scores)
    smoke_generation = restricted / "generation-smoke"
    full_generation = restricted / "generation-dpo-validation400"
    smoke_participant = restricted / "participant-dpo-smoke.jsonl"
    full_participant = restricted / "participant-dpo-validation400.jsonl"

    if not args.resume_after_generation:
        assert_gpus_idle((0, 1, 2, 3))
        with vllm_policy_server(
            runtime_root=output,
            label="dpo-validation-tp4",
            gpus=(0, 1, 2, 3),
            port=19331,
            adapters={"bundle": DPO_ADAPTER},
            aliases={"bundle": MODEL_ALIAS},
            max_num_seqs=256,
            max_num_batched_tokens=32768,
            dynamic_updates=False,
            max_model_len=4096,
            data_split="validation",
        ) as (endpoint, policy_attestation):
            attestation = generator_attestation(
                destination=output / "attestations/generator-dpo-validation.json",
                endpoint=endpoint,
                policy_attestation=policy_attestation,
            )
            run_command(
                generation_command(
                    run_id=f"{run_id}-generation-smoke", score_file=smoke_scores,
                    output_dir=smoke_generation, endpoint=endpoint,
                    attestation=attestation, expected=1, max_inflight=1,
                ),
                output / "logs/generation-smoke.log",
            )
            smoke_report = json.loads((smoke_generation / "aggregate_generation_report.json").read_text(encoding="utf-8"))
            need(smoke_report.get("status") == "completed", "DPO validation generation smoke failed")
            run_command(
                generation_command(
                    run_id=f"{run_id}-generation-full", score_file=SCORES,
                    output_dir=full_generation, endpoint=endpoint,
                    attestation=attestation, expected=EXPECTED, max_inflight=256,
                ),
                output / "logs/generation-full.log",
            )

        run_command(
            compose_command(
                run_id=f"{run_id}-compose-smoke", score_file=smoke_scores,
                generation_dir=smoke_generation, output_file=smoke_participant,
                aggregate_file=output / "aggregates/participant-smoke.json", expected=1,
            ),
            output / "logs/compose-smoke.log",
        )
        run_command(
            compose_command(
                run_id=f"{run_id}-compose-full", score_file=SCORES,
                generation_dir=full_generation, output_file=full_participant,
                aggregate_file=output / "aggregates/participant-full.json", expected=EXPECTED,
            ),
            output / "logs/compose-full.log",
        )
    else:
        smoke_generation_report = json.loads((smoke_generation / "aggregate_generation_report.json").read_text(encoding="utf-8"))
        full_generation_report = json.loads((full_generation / "aggregate_generation_report.json").read_text(encoding="utf-8"))
        smoke_participant_report = json.loads((output / "aggregates/participant-smoke.json").read_text(encoding="utf-8"))
        full_participant_report = json.loads((output / "aggregates/participant-full.json").read_text(encoding="utf-8"))
        need(
            smoke_generation_report.get("status") == full_generation_report.get("status") == "completed"
            and smoke_participant_report.get("status") == full_participant_report.get("status") == "completed"
            and sha256_file(smoke_participant) == smoke_participant_report.get("participant_file_sha256")
            and sha256_file(full_participant) == full_participant_report.get("participant_file_sha256"),
            "completed generation/composition resume evidence differs",
        )

    smoke_judge_run = f"{run_id}-judge-smoke"
    baseline_judge_run = f"{run_id}-baseline"
    dpo_judge_run = f"{run_id}-dpo"
    for judge_run in (smoke_judge_run, baseline_judge_run, dpo_judge_run):
        need(
            not (Q4_RESTRICTED_BASE / judge_run).exists()
            and not (Q4_OUTPUT_BASE / judge_run).exists(),
            f"judge output must be fresh: {judge_run}",
        )

    wait_for_owned_gpu_release((0, 1, 2, 3))
    with q4_judge_servers(
        runtime_root=output,
        label="exact-bundle-validation",
        gpus=(0, 1, 2, 3),
        ports=(19431, 19432, 19433, 19434),
        judge_prompt_sha256=EXACT_JUDGE_PROMPT_SHA256,
    ) as (endpoints, judge_attestation):
        run_command(
            judge_command(
                run_id=smoke_judge_run, participant=smoke_participant,
                endpoints=endpoints, attestation=judge_attestation, expected=1,
            ),
            output / "logs/judge-smoke.log",
        )
        smoke_judge = json.loads(
            (Q4_OUTPUT_BASE / smoke_judge_run / "aggregate_judge_report.json").read_text(encoding="utf-8")
        )
        need(smoke_judge.get("status") == "completed", "exact bundle judge smoke failed")
        run_command(
            judge_command(
                run_id=baseline_judge_run, participant=BASELINE_PARTICIPANT,
                endpoints=endpoints, attestation=judge_attestation, expected=EXPECTED,
            ),
            output / "logs/judge-baseline.log",
        )
        run_command(
            judge_command(
                run_id=dpo_judge_run, participant=full_participant,
                endpoints=endpoints, attestation=judge_attestation, expected=EXPECTED,
            ),
            output / "logs/judge-dpo.log",
        )

    report = comparison(baseline_judge_run, dpo_judge_run)
    report["run_id"] = run_id
    report["generation"] = {
        "dpo_generated_rationales_sha256": sha256_file(full_generation / "generated_rationales.jsonl"),
        "dpo_participant_sha256": sha256_file(full_participant),
        "baseline_generation_report_sha256": sha256_file(BASELINE_GENERATION / "aggregate_generation_report.json"),
    }
    comparison_path = output / "aggregate_bundle_dpo_validation_comparison.json"
    atomic_json(comparison_path, report)
    manifest.update(
        {
            "status": "completed",
            "completed_at": now(),
            "comparison_sha256": sha256_file(comparison_path),
            "selected": report["selection"]["selected"],
        }
    )
    atomic_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "completed",
                "run_id": run_id,
                "baseline_macro": report["baseline"]["macro_mean_5point"],
                "dpo_macro": report["dpo"]["macro_mean_5point"],
                "paired": report["paired"],
                "selected": report["selection"]["selected"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
