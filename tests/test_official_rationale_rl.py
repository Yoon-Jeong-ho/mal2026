from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mal2026.official_rationale_rl as rl
from mal2026.official_writing_contract import OfficialContractError
import scripts.generate_official_dpo_preferences as preferences
import scripts.run_official_rationale_rl as runner
import scripts.run_official_rationale_rl_experiment as experiment_runner


def judge(value: int) -> dict:
    return {
        axis: {dimension: {"evidence": "근거", "score": value} for dimension in rl.JUDGE_DIMENSIONS}
        for axis in rl.AXES
    }


class OfficialRationaleRLTest(unittest.TestCase):
    def test_configs_pin_offline_dpo_and_external_grpo(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dpo = rl.RLSettings.from_json(root / "configs/official_rationale_dpo.v1.json")
        grpo = rl.RLSettings.from_json(root / "configs/official_rationale_grpo.v1.json")
        self.assertEqual(dpo.policy["trainer"], "trl.DPOTrainer")
        self.assertIs(dpo.policy["offline_preferences"], True)
        self.assertEqual(grpo.policy["trainer"], "trl.GRPOTrainer")
        self.assertEqual(grpo.policy["rollout_backend"], "external_vllm_http_rollout_func")
        self.assertIs(grpo.policy["use_vllm"], False)
        self.assertTrue(all(item["classification"].endswith("not_direct_official_arm") for item in dpo.legacy_ablations))

    def test_preference_uses_exact_12_cell_integer_total_and_excludes_tie(self) -> None:
        selected = rl.select_preference([("low", judge(2)), ("high", judge(3))])
        self.assertEqual(selected, {
            "chosen": "high",
            "rejected": "low",
            "chosen_judge_total": 36,
            "rejected_judge_total": 24,
            "judge_total_difference": 12,
        })
        self.assertIsNone(rl.select_preference([("a", judge(3)), ("b", judge(3))]))
        with self.assertRaisesRegex(rl.OfficialRationaleRLError, "selection contract"):
            rl.select_preference([("a", judge(2)), ("b", judge(3))], minimum_total_difference=2)

    def test_axis_preference_does_not_inherit_global_credit(self) -> None:
        globally_high_content_low = judge(5)
        for dimension in rl.JUDGE_DIMENSIONS:
            globally_high_content_low["content"][dimension]["score"] = 1
        globally_low_content_high = judge(1)
        for dimension in rl.JUDGE_DIMENSIONS:
            globally_low_content_high["content"][dimension]["score"] = 5
        scored = [
            ("global-high-content-low", globally_high_content_low),
            ("global-low-content-high", globally_low_content_high),
        ]
        self.assertEqual(rl.select_preference(scored)["chosen"], "global-high-content-low")
        local = rl.select_axis_preference(scored, "content")
        self.assertEqual(local["chosen"], "global-low-content-high")
        self.assertEqual(local["chosen_axis_judge_total"], 20)
        self.assertEqual(local["rejected_axis_judge_total"], 4)
        self.assertEqual(local["axis_judge_total_difference"], 16)

    def test_participant_preserves_frozen_integer_score_vector(self) -> None:
        scores = {"content": 1, "organization": 3, "expression": 5}
        result = rl.participant(scores, {axis: f"{axis} 생성 근거" for axis in rl.AXES})
        self.assertEqual({axis: result[axis]["score"] for axis in rl.AXES}, scores)
        with self.assertRaisesRegex(rl.OfficialRationaleRLError, "integer"):
            rl.participant({"content": 1.0, "organization": 3, "expression": 5}, {axis: "근거" for axis in rl.AXES})

    def test_both_gate_reports_are_hard_prerequisites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            directional = root / "directional.json"
            directional.write_text(json.dumps({
                "schema_version": "mal2026-official-proxy-judge-contrastive-gate-v1",
                "status": "passed",
                "rl_with_this_proxy_judge_allowed": True,
                "thresholds_frozen_before_results": {"minimum_target_mean_decrease": 0.25, "minimum_paired_decrease_rate": 0.5},
                "tests": {name: {"passed": True} for name in ("a", "b", "c")},
            }), encoding="utf-8")
            safety = root / "safety.json"
            safety.write_text(json.dumps({
                "schema_version": "mal2026-official-proxy-judge-rl-safety-gate-v1",
                "status": "passed",
                "directional_contrastive_gate_passed": True,
                "prompt_injection_gate_passed": True,
                "rl_allowed": True,
                "failure_policy": "preserve all artifacts and exit 2; do not run RL with this proxy judge",
            }), encoding="utf-8")
            self.assertEqual(len(rl.assert_contrastive_gate(directional)["sha256"]), 64)
            self.assertEqual(len(rl.assert_rl_safety_gate(safety)["sha256"]), 64)
            value = json.loads(safety.read_text(encoding="utf-8"))
            value["prompt_injection_gate_passed"] = False
            safety.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(rl.OfficialRationaleRLError, "prompt-injection"):
                rl.assert_rl_safety_gate(safety)

    def test_preference_loader_rejects_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "bad.jsonl"
            path.write_text(json.dumps({
                "schema_version": "mal2026-official-rationale-preference-v1",
                "split": "validation",
                "task": "bundle",
                "score_kind": "frozen_api_emitted_integer_prediction",
                "judge_total_difference": 1,
                "prompt": [{"role": "user", "content": "private"}],
                "chosen": [{"role": "assistant", "content": "좋음"}],
                "rejected": [{"role": "assistant", "content": "나쁨"}],
            }) + "\n", encoding="utf-8")
            with patch.object(rl, "RESTRICTED_ROOT", root):
                with self.assertRaisesRegex(rl.OfficialRationaleRLError, "split/task"):
                    rl.load_preferences(path, "bundle")

    def test_preference_loader_enforces_axis_projection_and_local_arithmetic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "axis.jsonl"
            row = {
                "schema_version": "mal2026-official-rationale-preference-v1",
                "split": "train",
                "arm": "axis_triplet",
                "task": "content",
                "score_kind": "frozen_api_emitted_integer_prediction",
                "chosen_axis_judge_total": 17,
                "rejected_axis_judge_total": 12,
                "axis_judge_total_difference": 5,
                "selection_projection": "sum_of_4_integer_cells_for_content",
                "prompt": [{"role": "user", "content": "private"}],
                "chosen": [{"role": "assistant", "content": "좋음"}],
                "rejected": [{"role": "assistant", "content": "나쁨"}],
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with patch.object(rl, "RESTRICTED_ROOT", root):
                rows, _ = rl.load_preferences(path, "content")
                self.assertEqual(len(rows), 1)
                row["selection_projection"] = "sum_of_all_12_integer_cells"
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(rl.OfficialRationaleRLError, "axis preference projection"):
                    rl.load_preferences(path, "content")

    def test_axis_reward_calls_full_judge_but_projects_four_target_cells(self) -> None:
        root = Path(__file__).resolve().parents[1]
        settings = rl.RLSettings.from_json(root / "configs/official_rationale_grpo.v1.json")
        captured = {}

        def fake_score(endpoint, model, prompt_text, essay_text, candidate, *, system_prompt=None):
            captured["candidate"] = candidate
            captured["system_prompt"] = system_prompt
            output = judge(2)
            for dimension in rl.JUDGE_DIMENSIONS:
                output["content"][dimension]["score"] = 4
            return output

        reward = rl.ExactQ4Reward(settings, "content", ["http://127.0.0.1:9999"], "judge")
        completion = json.dumps({"content": {"rationale": "내용 영역의 구체적인 생성 근거"}}, ensure_ascii=False)
        with patch.object(rl, "q4_score", fake_score):
            values = reward(
                prompts=[[{"role": "user", "content": "x"}]],
                completions=[[{"role": "assistant", "content": completion}]],
                prompt_text=["문제"], essay_text=["학생 글"],
                scores=[{"content": 2, "organization": 3, "expression": 4}],
                frozen_rationales=[{"content": "기존", "organization": "조직 근거", "expression": "표현 근거"}],
            )
        self.assertEqual(values, [4.0])
        self.assertEqual(captured["candidate"]["content"]["score"], 2)
        self.assertEqual(captured["candidate"]["organization"]["rationale"], "조직 근거")
        self.assertEqual(captured["system_prompt"], rl.FROZEN_PROXY_JUDGE_SYSTEM_PROMPT)
        self.assertEqual(reward.aggregate()["projection"], "content_4_cell_mean_from_full_12_cell_judgment")

    def test_q4_judge_accepts_length_only_after_complete_12_cell_parse(self) -> None:
        complete = json.dumps(judge(4), ensure_ascii=False)
        response = {"choices": [{"finish_reason": "length", "message": {"content": complete}}]}
        candidate = {
            axis: {"score": 3, "rationale": "판단 근거"}
            for axis in rl.AXES
        }
        with patch.object(rl, "http_json", return_value=response):
            parsed = rl.q4_score(
                "http://127.0.0.1:9999", "judge", "문제", "학생 글", candidate,
            )
        self.assertEqual(parsed, judge(4))

        truncated = {"choices": [{"finish_reason": "length", "message": {"content": complete[:-1]}}]}
        with patch.object(rl, "http_json", return_value=truncated):
            with self.assertRaises(OfficialContractError):
                rl.q4_score(
                    "http://127.0.0.1:9999", "judge", "문제", "학생 글", candidate,
                )

    def test_exact_user_judge_configs_bind_file_and_preserve_failed_gate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for algorithm in ("dpo", "grpo"):
            settings = rl.RLSettings.from_json(root / f"configs/official_rationale_{algorithm}.llm_as_judge_txt.v2.json")
            self.assertEqual(settings.judge_system_prompt(), (root / "llm_as_judge.txt").read_text(encoding="utf-8"))
            evidence = settings.gate_evidence()
            self.assertEqual(evidence["combined_safety"]["status"], "user_authorized_exact_prompt")
            self.assertIs(evidence["combined_safety"]["legacy_failed_gate_preserved"], True)

    def test_rollout_accepts_only_control_character_json_relaxation_then_revalidates_schema(self) -> None:
        root = Path(__file__).resolve().parents[1]
        settings = rl.RLSettings.from_json(root / "configs/official_rationale_dpo.llm_as_judge_txt.v2.json")
        response = {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"content":{"rationale":"첫 줄\n둘째 줄"}}'},
            }],
        }
        with patch.object(preferences, "http_json", return_value=response):
            outputs, relaxed, complete_length, retry_requests, retry_candidates = preferences.policy_request(
                "http://127.0.0.1:9999",
                "policy",
                "content",
                [{"role": "user", "content": "입력"}],
                1,
                settings,
                42,
            )
        self.assertEqual(relaxed, 1)
        self.assertEqual(complete_length, 0)
        self.assertEqual(retry_requests, 0)
        self.assertEqual(retry_candidates, 0)
        self.assertEqual(outputs, [{"content": "첫 줄\n둘째 줄"}])

    def test_rollout_accepts_only_schema_complete_length_finish(self) -> None:
        root = Path(__file__).resolve().parents[1]
        settings = rl.RLSettings.from_json(root / "configs/official_rationale_dpo.llm_as_judge_txt.v2.json")
        valid = {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": '{"content":{"rationale":"완결된 판단 근거"}}'},
            }],
        }
        with patch.object(preferences, "http_json", return_value=valid):
            outputs, relaxed, complete_length, retry_requests, retry_candidates = preferences.policy_request(
                "http://127.0.0.1:9999",
                "policy",
                "content",
                [{"role": "user", "content": "입력"}],
                1,
                settings,
                42,
            )
        self.assertEqual(outputs, [{"content": "완결된 판단 근거"}])
        self.assertEqual(relaxed, 0)
        self.assertEqual(complete_length, 1)
        self.assertEqual(retry_requests, 0)
        self.assertEqual(retry_candidates, 0)

        truncated = {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": '{"content":{"rationale":"잘린 판단 근거"'},
            }],
        }
        extended = {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"content":{"rationale":"잘린 판단 근거"}}'},
            }],
        }
        with patch.object(preferences, "http_json", side_effect=[truncated, extended]) as mocked:
            outputs, relaxed, complete_length, retry_requests, retry_candidates = preferences.policy_request(
                "http://127.0.0.1:9999",
                "policy",
                "content",
                [{"role": "user", "content": "입력"}],
                1,
                settings,
                42,
            )
        self.assertEqual(outputs, [{"content": "잘린 판단 근거"}])
        self.assertEqual((relaxed, complete_length, retry_requests, retry_candidates), (0, 0, 1, 1))
        self.assertEqual(mocked.call_args_list[0].args[1]["max_tokens"], 1200)
        self.assertEqual(mocked.call_args_list[1].args[1]["max_tokens"], 1200)
        self.assertEqual(
            mocked.call_args_list[0].args[1]["response_format"]["json_schema"]["schema"]
            ["properties"]["content"]["properties"]["rationale"]["maxLength"],
            384,
        )

        initial_two = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"content":{"rationale":"기존 완성 근거"}}'}},
                {"finish_reason": "length", "message": {"content": '{"content":{"rationale":"잘린 근거"'}},
            ],
        }
        retry_two = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"content":{"rationale":"무시할 재생성 근거"}}'}},
                {"finish_reason": "stop", "message": {"content": '{"content":{"rationale":"교체된 완성 근거"}}'}},
            ],
        }
        with patch.object(preferences, "http_json", side_effect=[initial_two, retry_two]):
            outputs, _, _, retry_requests, retry_candidates = preferences.policy_request(
                "http://127.0.0.1:9999",
                "policy",
                "content",
                [{"role": "user", "content": "입력"}],
                2,
                settings,
                42,
            )
        self.assertEqual(outputs, [
            {"content": "기존 완성 근거"},
            {"content": "교체된 완성 근거"},
        ])
        self.assertEqual((retry_requests, retry_candidates), (1, 1))

    def test_bundle_rollout_ceiling_covers_frozen_three_field_character_bound(self) -> None:
        root = Path(__file__).resolve().parents[1]
        settings = rl.RLSettings.from_json(root / "configs/official_rationale_dpo.llm_as_judge_txt.v2.json")
        response = {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": json.dumps({
                    axis: {"rationale": "완결된 판단 근거"}
                    for axis in rl.AXES
                }, ensure_ascii=False)},
            }],
        }
        with patch.object(preferences, "http_json", return_value=response) as mocked:
            preferences.policy_request(
                "http://127.0.0.1:9999", "policy", "bundle",
                [{"role": "user", "content": "입력"}], 1, settings, 42,
            )
        self.assertEqual(mocked.call_args.args[1]["max_tokens"], 4000)

    def test_gpu_conflict_gate_is_read_only_and_fail_closed(self) -> None:
        busy = type("Result", (), {"returncode": 0, "stdout": "12345\n"})()
        with patch.object(runner.subprocess, "run", return_value=busy) as mocked:
            with self.assertRaisesRegex(RuntimeError, "pre-existing compute"):
                runner.require_gpu_idle(0)
        command = mocked.call_args.args[0]
        self.assertEqual(command[:3], ["nvidia-smi", "-i", "0"])
        self.assertNotIn("kill", command)
        idle = type("Result", (), {"returncode": 0, "stdout": ""})()
        with patch.object(runner.subprocess, "run", return_value=idle):
            runner.require_gpu_idle(3)

    def test_dpo_smoke_uses_enough_groups_to_measure_saturated_reward_variance(self) -> None:
        self.assertEqual(experiment_runner.DPO_SMOKE_GROUPS, 32)


if __name__ == "__main__":
    unittest.main()
