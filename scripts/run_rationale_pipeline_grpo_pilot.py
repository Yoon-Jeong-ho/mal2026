#!/usr/bin/env python3
"""Own the GPU0--3 server topology for the bounded score-blind GRPO pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.api_rationale_data import sha256_file  # noqa: E402
from mal2026.official_rl_servers import PYTHON, assert_gpus_idle, q4_judge_servers, vllm_policy_server  # noqa: E402


TRAINER = ROOT / "scripts/train_rationale_pipeline_grpo.py"
OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-grpo-orchestration-v1"
JUDGE_SHA = "91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d"


def need(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--base-model", type=Path, required=True); parser.add_argument("--warm-start-adapter", type=Path, required=True); parser.add_argument("--warm-start-completion", type=Path, required=True); parser.add_argument("--variance-report", type=Path, required=True); args = parser.parse_args()
    setproctitle(f"mal2026:rationale-grpo-orchestrator:{args.run_id}"[:255]); assert_gpus_idle((0, 1, 2, 3))
    gate = json.loads(args.variance_report.read_text(encoding="utf-8")); need(gate.get("status") == "passed", "GRPO orchestration variance gate did not pass")
    output = OUTPUT_PARENT / args.run_id; need(not output.exists(), "GRPO orchestration output must be fresh"); output.mkdir(parents=True)
    alias = "mal2026-score-blind-grpo-policy"
    with vllm_policy_server(runtime_root=output, label="rollout", gpus=(0, 1), port=19700, adapters={"bundle": args.warm_start_adapter}, aliases={"bundle": alias}, max_num_seqs=64, max_num_batched_tokens=32768, dynamic_updates=True, max_model_len=4096, model_path=args.base_model, model_id="skt/A.X-4.0-Light", model_revision="ba21c20ea1b31ded1ec3e2fb432335077dc4be98", data_split="train") as (rollout_endpoint, rollout_attestation):
        with q4_judge_servers(runtime_root=output, label="reward", gpus=(3,), ports=(19710,), judge_prompt_sha256=JUDGE_SHA) as (judge_endpoints, judge_attestation):
            command = [str(PYTHON), str(TRAINER), "--run-id", args.run_id, "--base-model", str(args.base_model), "--warm-start-adapter", str(args.warm_start_adapter), "--warm-start-completion", str(args.warm_start_completion), "--variance-report", str(args.variance_report), "--rollout-endpoint", rollout_endpoint, "--rollout-alias", alias, "--judge-endpoint", judge_endpoints[0], "--max-steps", "20", "--train-limit", "160"]
            log = (output / "trainer.log").open("x", encoding="utf-8"); result = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "2", "PYTHONPATH": str(ROOT / "src") + ":" + str(ROOT / "scripts"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, stdout=log, stderr=subprocess.STDOUT, text=True); log.close()
            need(result.returncode in {0, 2}, "GRPO trainer integration failed")
    model_output = ROOT / "outputs/rationale-pipeline-grpo-v1" / args.run_id
    completion = model_output / "training_complete.json"; failed = model_output / "training_failed_gate.json"
    need(completion.is_file() != failed.is_file(), "GRPO completion artifact differs")
    artifact = completion if completion.is_file() else failed; value = json.loads(artifact.read_text(encoding="utf-8"))
    report = {"schema_version": "mal2026-rationale-pipeline-grpo-orchestration-v1", "status": value["status"], "run_id": args.run_id, "gpu_scope": [0, 1, 2, 3], "topology": {"rollout_tp2": [0, 1], "policy": [2], "exact_q4_reward": [3]}, "rollout_attestation_sha256": sha256_file(rollout_attestation), "judge_attestation_sha256": sha256_file(judge_attestation), "variance_report_sha256": sha256_file(args.variance_report), "training_artifact_path": str(artifact.resolve()), "training_artifact_sha256": sha256_file(artifact), "validation_used": False, "average_used": False}
    (output / "aggregate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); print(json.dumps({"status": value["status"], "run_id": args.run_id}, sort_keys=True), flush=True)
    if value["status"] != "completed": raise SystemExit(2)


if __name__ == "__main__": main()
