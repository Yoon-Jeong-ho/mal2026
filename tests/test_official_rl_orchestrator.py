from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mal2026.official_rationale_rl import RLSettings
import mal2026.official_rl_servers as servers
import scripts.run_official_rationale_rl_experiment as experiment


class OfficialRLOrchestratorTest(unittest.TestCase):
    def test_policy_commands_keep_eager_off_and_high_batching(self) -> None:
        adapters = {task: Path(f"/adapter/{task}") for task in ("bundle", "content", "organization", "expression")}
        aliases = {task: f"alias-{task}" for task in adapters}
        static = servers.vllm_policy_command(
            gpus=(0, 1, 2, 3), port=19321, adapters=adapters, aliases=aliases,
            max_num_seqs=256, max_num_batched_tokens=32768, dynamic_updates=False,
            max_model_len=8192,
        )
        self.assertNotIn("--enforce-eager", static)
        self.assertEqual(static[static.index("--tensor-parallel-size") + 1], "4")
        self.assertEqual(static[static.index("--max-num-seqs") + 1], "256")
        self.assertEqual(static[static.index("--max-num-batched-tokens") + 1], "32768")
        self.assertEqual(static[static.index("--max-model-len") + 1], "8192")
        structured = static[static.index("--structured-outputs-config") + 1]
        self.assertEqual(json.loads(structured), {"backend": "xgrammar", "disable_any_whitespace": True})
        self.assertIn("--lora-modules", static)
        self.assertEqual(sum(value.startswith("alias-") for value in static), 4)
        dynamic = servers.vllm_policy_command(
            gpus=(0, 1), port=19330, adapters={"bundle": adapters["bundle"]}, aliases={"bundle": aliases["bundle"]},
            max_num_seqs=192, max_num_batched_tokens=65536, dynamic_updates=True,
        )
        self.assertNotIn("--lora-modules", dynamic)
        self.assertEqual(dynamic[dynamic.index("--tensor-parallel-size") + 1], "2")

        root = Path(__file__).resolve().parents[1]
        grpo = RLSettings.from_json(root / "configs/official_rationale_grpo.v1.json")
        midm = next(item for item in grpo.legacy_ablations if item["name"].startswith("midm_"))
        midm_command = servers.vllm_policy_command(
            gpus=(0, 1), port=19330,
            adapters={"bundle": Path(midm["adapter_path"])},
            aliases={"bundle": "official-rl-legacy-midm"},
            max_num_seqs=192, max_num_batched_tokens=65536, dynamic_updates=True,
            model_path=Path(midm["model_path"]), model_id=midm["model_id"],
        )
        self.assertEqual(midm_command[2], midm["model_path"])
        self.assertEqual(midm_command[midm_command.index("--served-model-name") + 1], midm["model_id"])
        self.assertNotIn("--lora-modules", midm_command)

    def test_q4_command_is_pinned_high_throughput_shape(self) -> None:
        command = servers.q4_server_command(19420)
        self.assertEqual(command[command.index("--parallel") + 1], "4")
        self.assertEqual(command[command.index("--batch-size") + 1], "2048")
        self.assertEqual(command[command.index("--ubatch-size") + 1], "512")
        self.assertEqual(command[command.index("--reasoning") + 1], "off")

    def test_gpu_conflict_check_is_read_only_and_fail_closed(self) -> None:
        result = type("Result", (), {"returncode": 0, "stdout": "991\n", "stderr": ""})()
        with patch.object(servers.subprocess, "run", return_value=result) as mocked:
            with self.assertRaisesRegex(servers.OfficialRLServerError, "pre-existing GPU"):
                servers.assert_gpus_idle((0,))
        command = mocked.call_args.args[0]
        self.assertEqual(command[:3], ["nvidia-smi", "-i", "0"])
        self.assertNotIn("kill", command)

    def test_durable_stage_skips_completed_and_increments_failed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = object.__new__(experiment.DurableRun)
            run.run_id = "official-rationale-rl-experiment-v1-synthetic"
            run.root = root
            run.ledger_path = root / "ledger.jsonl"
            run.gates = {"directional": {"sha256": "a" * 64}, "combined_safety": {"sha256": "b" * 64}}
            run.judge_prompt_sha256 = "c" * 64
            (root / "stages").mkdir()
            calls = []

            def complete(attempt):
                calls.append(attempt)
                return {"synthetic": True}

            first = run.run_stage("complete", complete)
            second = run.run_stage("complete", complete)
            self.assertEqual(calls, [1])
            self.assertEqual(first, second)

            def fail(attempt):
                raise ValueError(str(attempt))

            with self.assertRaisesRegex(ValueError, "1"):
                run.run_stage("retry", fail)
            observed = []

            def retry(attempt):
                observed.append(attempt)
                return {"recovered": True}

            run.run_stage("retry", retry)
            self.assertEqual(observed, [2])

    def test_cpu_dry_plan_covers_full_dpo_and_grpo_topology(self) -> None:
        args = argparse.Namespace(
            run_id="official-rationale-rl-experiment-v1-synthetic",
            scope="all", grpo_task=["bundle"], grpo_phase=["pilot", "full"],
            legacy_grpo_arm=list(experiment.LEGACY_GRPO_ARMS),
            dpo_config=Path(__file__).resolve().parents[1] / "configs/official_rationale_dpo.llm_as_judge_txt.v2.json",
            grpo_config=Path(__file__).resolve().parents[1] / "configs/official_rationale_grpo.llm_as_judge_txt.v2.json",
        )
        with patch.object(servers, "gpu_compute_pids", side_effect=AssertionError("dry-run queried GPU")):
            plan = experiment.dry_plan(args)
        self.assertFalse(plan["gpu_queries_in_dry_run"])
        self.assertEqual(plan["grpo"]["topology"], {"rollout_tp2": [0, 1], "trainer": [2], "q4_reward": [3]})
        self.assertFalse(plan["grpo"]["integrated_vllm"])
        self.assertEqual(len(plan["legacy_arms"]), 3)
        self.assertEqual(plan["grpo"]["official_best"], "bundle")
        self.assertEqual(plan["grpo"]["legacy_top3"], list(experiment.LEGACY_GRPO_ARMS))
        self.assertEqual(plan["grpo"]["per_producer_sequence"], ["real_one_update_smoke", "pilot", "full"])
        self.assertEqual(set(plan["grpo"]["producer_stage_plan"]), {"official:bundle", *experiment.LEGACY_GRPO_ARMS})
        self.assertTrue(all(len(stages) == 3 for stages in plan["grpo"]["producer_stage_plan"].values()))
        self.assertTrue(plan["grpo"]["exact_q4_reward"])
        self.assertTrue(all(item["static_compatibility_status"] == "supported_pending_real_one_update_smoke" for item in plan["grpo"]["legacy_producers"]))
        self.assertIn("TP4 full bundle+axis rollout", plan["dpo_stages"])

    def test_legacy_grpo_failure_is_explicitly_not_handoff_eligible(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            run = object.__new__(experiment.DurableRun)
            run.run_id = "official-rationale-rl-experiment-v1-synthetic"
            run.root = Path(directory)
            (run.root / "aggregates").mkdir()
            run.grpo = RLSettings.from_json(root / "configs/official_rationale_grpo.v1.json")
            events = []
            run.event = lambda stage, event, evidence: events.append((stage, event, evidence))
            run.run_stage = lambda stage, function: function(1)
            arm = experiment.LEGACY_GRPO_ARMS[0]
            with patch.object(experiment, "legacy_grpo_producer_spec", side_effect=RuntimeError("synthetic unsupported architecture")):
                with self.assertRaisesRegex(RuntimeError, "unsupported architecture"):
                    experiment.grpo_one(run, "bundle", "smoke", {}, legacy_name=arm)
            path = next((run.root / "aggregates").glob("*producer-status*.json"))
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value["producer_status"], "failed_producer")
            self.assertFalse(value["handoff_eligible"])
            self.assertEqual(value["legacy_arm"], arm)
            self.assertEqual(value["failure_type"], "RuntimeError")
            self.assertEqual(value["failure_stage"], "static_compatibility")
            self.assertEqual(events[0][1], "producer_failed_closed")

    def test_configs_remain_valid_with_frozen_grpo_bounds(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dpo = RLSettings.from_json(root / "configs/official_rationale_dpo.v1.json")
        grpo = RLSettings.from_json(root / "configs/official_rationale_grpo.v1.json")
        self.assertEqual(dpo.algorithm, "dpo")
        self.assertEqual((grpo.policy["pilot_max_steps"], grpo.policy["max_steps"]), (80, 480))
        specs = [experiment.legacy_grpo_producer_spec(grpo, name) for name in experiment.LEGACY_GRPO_ARMS]
        self.assertEqual([item["model_architecture"] for item in specs], ["LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen2ForCausalLM"])
        self.assertTrue(all(item["legacy_completion_sha256"] for item in specs))
        self.assertTrue(all(item["warm_start_adapter_model_sha256"] for item in specs))


if __name__ == "__main__":
    unittest.main()
