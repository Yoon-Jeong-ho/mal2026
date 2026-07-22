"""GPU-free contracts for the train-only RLAIF prompt-ensemble study."""
from __future__ import annotations

import json
import importlib.util
import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mal2026.rlaif_evaluation import _response_schema
from mal2026.api_rationale_data import ROOT
from mal2026.rlaif_grpo import JUDGE, QwenPointReward, StructuredVLLMRollout, RLAIFSettings, _call_reward_judge, _policy_response_schema, canonical_completion, canonical_completion_text, random_prompt_index


class RLAIFGRPOContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = RLAIFSettings.from_json()
        self.axes = ("content", "organization", "expression")
        self.valid = json.dumps({
            "schema_version": "rationale-only-v1",
            "content": {"rationale": "주장과 근거의 연결이 구체적으로 드러난다."},
            "organization": {"rationale": "서론과 결론이 자연스럽게 이어진다."},
            "expression": {"rationale": "문장이 명확하고 표현이 자연스럽다."},
        }, ensure_ascii=False)

    def test_fixed_template_is_score_blind_and_has_five_forms(self) -> None:
        fixed = self.settings.fixed_prompt_template()
        template = fixed["protocol"]
        self.assertFalse(template["reference_score_in_prompt"])
        self.assertTrue(template["candidate_isolated"])
        self.assertEqual(len(template["prompt_types"]), 5)
        self.assertFalse(fixed["runtime"]["enforce_eager"])
        self.assertEqual(self.settings.arms, ("all5", "random1"))

    def test_canonical_completion_requires_exact_bounded_korean_schema(self) -> None:
        parsed = canonical_completion(self.valid, self.axes, 192)
        self.assertIsNotNone(parsed)
        extra = json.loads(self.valid); extra["score"] = 5
        self.assertIsNone(canonical_completion(json.dumps(extra, ensure_ascii=False), self.axes, 192))
        english = json.loads(self.valid); english["content"]["rationale"] = "Specific supporting detail is present."
        self.assertIsNone(canonical_completion(json.dumps(english, ensure_ascii=False), self.axes, 192))
        too_long = json.loads(self.valid); too_long["content"]["rationale"] = "가" * 193
        self.assertIsNone(canonical_completion(json.dumps(too_long, ensure_ascii=False), self.axes, 192))

    def test_random_one_form_is_stable_and_reaches_all_forms(self) -> None:
        canonical = canonical_completion_text(self.valid, self.axes, 192)
        self.assertIsNotNone(canonical)
        self.assertEqual(random_prompt_index(2026072209, "random1", "opaque", canonical), random_prompt_index(2026072209, "random1", "opaque", canonical))
        indices = {random_prompt_index(2026072209, "random1", f"opaque-{index}", canonical) for index in range(100)}
        self.assertEqual(indices, {0, 1, 2, 3, 4})

    def test_reward_transport_limit_is_forwarded_to_the_fixed_client(self) -> None:
        valid = {"failure_category": None, "scored": True, "schema_valid": True, "scores": {axis: 3 for axis in self.axes}, "attempts": 2}
        task = {"opaque_request_key": "opaque"}
        with patch("mal2026.rlaif_grpo.JUDGE.call_with_transport_attempts", return_value=valid) as call:
            result, retries = _call_reward_judge("http://127.0.0.1:1", task, 3)
        self.assertEqual(result, valid)
        self.assertEqual(retries, 1)
        call.assert_called_once_with("http://127.0.0.1:1", task, 3)

    def test_vllm_error_finish_retries_but_length_does_not(self) -> None:
        scores = {axis: 3 for axis in self.axes}
        with patch.object(JUDGE, "request_once", side_effect=[(None, "envelope_error"), (scores, None)]) as request, patch.object(JUDGE.time, "sleep"):
            recovered = JUDGE.judge_call("http://127.0.0.1:1", {}, "required_scores_only_v1", max_attempts=3)
        self.assertEqual(recovered, {"scores": scores, "attempts": 2, "failure": None})
        self.assertEqual(request.call_count, 2)
        with patch.object(JUDGE, "request_once", return_value=(None, "envelope_length")) as request:
            incomplete = JUDGE.judge_call("http://127.0.0.1:1", {}, "required_scores_only_v1", max_attempts=3)
        self.assertEqual(incomplete["failure"], "envelope_length")
        self.assertEqual(incomplete["attempts"], 1)
        self.assertEqual(request.call_count, 1)

    def test_vllm_error_finish_retry_is_bounded(self) -> None:
        with patch.object(JUDGE, "request_once", return_value=(None, "envelope_error")) as request, patch.object(JUDGE.time, "sleep"):
            failed = JUDGE.judge_call("http://127.0.0.1:1", {}, "required_scores_only_v1", max_attempts=3)
        self.assertEqual(failed["failure"], "envelope_error")
        self.assertEqual(failed["attempts"], 3)
        self.assertEqual(request.call_count, 3)

    def test_generation_schema_has_no_score_field(self) -> None:
        schema = _response_schema(self.axes, 192)
        self.assertEqual(set(schema["properties"]), {"schema_version", *self.axes})
        self.assertNotIn("score", schema["properties"])
        self.assertEqual(schema["properties"]["content"]["properties"]["rationale"]["maxLength"], 192)

    def test_v2_binds_structured_rollout_without_changing_reward_contract(self) -> None:
        v2 = RLAIFSettings.from_json(ROOT / "configs" / "rlaif_grpo_prompt_ensemble.v2.json")
        self.assertEqual(v2.policy["rollout_backend"], "vllm_structured_outputs_http_v1")
        self.assertEqual(v2.runtime["full_rollout_gpus"], [0])
        self.assertEqual(v2.runtime["full_policy_gpus"], [1, 2])
        self.assertEqual(v2.runtime["full_reward_gpus"], [3])
        self.assertEqual(v2.judge, self.settings.judge)
        self.assertEqual(v2.reward, self.settings.reward)
        schema = _policy_response_schema(self.axes, 192)
        self.assertEqual(schema["required"], ["schema_version", *self.axes])
        self.assertFalse(schema["additionalProperties"])

    def test_v3_keeps_the_population_and_reward_judge_but_bounds_policy_rationales(self) -> None:
        v3 = RLAIFSettings.from_json(ROOT / "configs" / "rlaif_grpo_prompt_ensemble.v3.json")
        self.assertEqual(v3.policy["rollout_backend"], "vllm_structured_outputs_http_v1")
        self.assertEqual(v3.inputs, RLAIFSettings.from_json(ROOT / "configs" / "rlaif_grpo_prompt_ensemble.v2.json").inputs)
        self.assertEqual(v3.judge, self.settings.judge)
        self.assertEqual(v3.reward["field_character_limit"], 128)
        self.assertEqual(_policy_response_schema(self.axes, 128)["properties"]["content"]["properties"]["rationale"]["maxLength"], 128)

    def test_v4_leaves_only_the_redundant_schema_length_cap_unconstrained(self) -> None:
        v3 = RLAIFSettings.from_json(ROOT / "configs" / "rlaif_grpo_prompt_ensemble.v3.json")
        v4 = RLAIFSettings.from_json(ROOT / "configs" / "rlaif_grpo_prompt_ensemble.v4.json")
        self.assertFalse(v4.policy["rollout_json_schema_enforces_field_limit"])
        self.assertEqual(v4.inputs, v3.inputs)
        self.assertEqual(v4.judge, v3.judge)
        self.assertEqual(v4.reward, v3.reward)
        rationale = _policy_response_schema(self.axes, 128, enforce_character_limit=False)["properties"]["content"]["properties"]["rationale"]
        self.assertEqual(rationale, {"type": "string", "minLength": 1})
        too_long = json.loads(self.valid); too_long["content"]["rationale"] = "가" * 129
        self.assertIsNone(canonical_completion(json.dumps(too_long, ensure_ascii=False), self.axes, int(v4.reward["field_character_limit"])))

    def test_v5_json_object_mode_keeps_canonical_contract_at_frozen_limit(self) -> None:
        v5 = RLAIFSettings.from_json(ROOT / "configs" / "rlaif_grpo_prompt_ensemble.v5.json")
        self.assertEqual(v5.policy["rollout_structured_output_mode"], "json_object")
        self.assertFalse(v5.policy["rollout_json_schema_enforces_field_limit"])
        self.assertEqual(v5.reward["field_character_limit"], 192)
        too_long = json.loads(self.valid); too_long["content"]["rationale"] = "가" * 193
        self.assertIsNone(canonical_completion(json.dumps(too_long, ensure_ascii=False), self.axes, int(v5.reward["field_character_limit"])))

    def test_v6_keeps_global_rollout_group_with_tensor_parallel_policy(self) -> None:
        v6 = RLAIFSettings.from_json(ROOT / "configs" / "rlaif_grpo_prompt_ensemble.v6.json")
        self.assertEqual(v6.runtime["full_rollout_gpus"], [0, 1])
        self.assertEqual(v6.runtime["rollout_tensor_parallel_size"], 2)
        self.assertEqual(v6.runtime["full_policy_gpus"], [2])
        self.assertEqual(v6.policy["per_device_train_batch_size_full"] * v6.policy["num_generations"], v6.policy["generation_batch_size_full"])

    def test_v7_is_only_the_documented_tp2_allocator_repair(self) -> None:
        v6 = RLAIFSettings.from_json(ROOT / "configs" / "rlaif_grpo_prompt_ensemble.v6.json")
        v7 = RLAIFSettings.from_json(ROOT / "configs" / "rlaif_grpo_prompt_ensemble.v7.json")
        self.assertEqual(v7.judge, v6.judge)
        self.assertEqual(v7.policy, v6.policy)
        self.assertEqual(v7.inputs, v6.inputs)
        self.assertEqual(v7.reward, v6.reward)
        runtime = dict(v7.runtime)
        self.assertEqual(runtime.pop("policy_training_cuda_alloc_conf"), "expandable_segments:True")
        self.assertEqual(runtime, v6.runtime)

    def test_v8_discards_an_unscorable_judge_generation_group_without_low_quality_label(self) -> None:
        settings = RLAIFSettings.from_json(ROOT / "configs" / "rlaif_grpo_prompt_ensemble.v8.json")
        self.assertEqual(settings.reward["unscorable_judge_group_policy"], "discard_generation_group")
        reward = QwenPointReward(settings, SimpleNamespace(task="content", arm="all5", run_id="rlaif-contract", reward_endpoint="http://127.0.0.1:1"))
        completion = json.dumps({"schema_version": "rationale-only-v1", "content": {"rationale": "근거와 주장의 연결이 글의 문장을 근거로 설명된다."}}, ensure_ascii=False)

        def judge(*_args, **kwargs):
            task = _args[1]
            if task["prompt_type_id"] == "balanced_rationale":
                return {"failure_category": "envelope_length", "scored": False, "schema_valid": False, "scores": None, "attempts": 1}, 0
            return {"failure_category": None, "scored": True, "schema_valid": True, "scores": {axis: 3 for axis in self.axes}, "attempts": 1}, 0

        with patch("mal2026.rlaif_grpo._call_reward_judge", side_effect=judge):
            values = reward([{"role": "user", "content": "x"}] * 4, [completion] * 4, ["id"] * 4, ["same-source"] * 4,
                            [{"sentences": ["문장입니다."]}] * 4)
        self.assertEqual(values, [0.0] * 4)
        summary = reward.aggregate()
        self.assertEqual(summary["judge_requests"], 20)
        self.assertEqual(summary["judge_calls"], 16)
        self.assertEqual(summary["judge_unscorable"], 4)
        self.assertEqual(summary["judge_failure_categories"], {"envelope_length": 4})
        self.assertEqual(summary["discarded_reward_groups"], 1)
        self.assertEqual(summary["discarded_reward_completions"], 4)

    def test_tp2_full_runner_reuses_policy_and_gpu3_judge_for_both_arms(self) -> None:
        """The later full matrix must not silently fall back to the v2 layout."""
        runner_path = ROOT / "scripts" / "run_rlaif_grpo_prompt_ensemble_v1.py"
        with patch.dict(os.environ, {"MAL2026_RLAIF_CONFIG": "configs/rlaif_grpo_prompt_ensemble.v7.json", "MAL2026_RLAIF_RUNTIME_ID": "20260722-099"}, clear=False):
            spec = importlib.util.spec_from_file_location("rlaif_v7_runner_contract", runner_path)
            self.assertIsNotNone(spec and spec.loader)
            runner = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(runner)
        calls: list[tuple] = []

        @contextmanager
        def rollout_server(**kwargs):
            calls.append(("rollout", kwargs))
            yield Path("/tmp/rlaif-policy-attestation.json")

        @contextmanager
        def judge_server(**kwargs):
            calls.append(("judge", kwargs))
            yield Path("/tmp/rlaif-reward-attestation.json")

        def train(*args, **kwargs):
            calls.append(("train", args, kwargs))
            return {}

        def evaluate(*args, **kwargs):
            calls.append(("evaluate", args, kwargs))
            return {}

        runner.policy_rollout_server = rollout_server
        runner.reward_server = judge_server
        runner.train_arm = train
        runner.evaluate_arm = evaluate
        self.assertFalse(runner.fixed_v6_enforce_eager(runner.config()))
        runner.run_full_task_arms("midm2_base", "bundle", 18520)

        self.assertEqual(calls[0], ("rollout", {"base_key": "midm2_base", "task": "bundle", "port": 18570, "label": "midm2_base-bundle-two-arm"}))
        self.assertEqual(calls[1], ("judge", {"gpus": [3], "data_parallel_size": 1, "port": 18520, "label": "midm2_base-bundle-two-arm"}))
        trains = [entry for entry in calls if entry[0] == "train"]
        self.assertEqual(len(trains), 2)
        self.assertTrue(all(entry[1][3] == "full" and entry[1][6] == "2" for entry in trains))
        self.assertTrue(all(entry[2]["rollout_endpoint"] == "http://127.0.0.1:18570" for entry in trains))
        self.assertEqual([entry[0] for entry in calls[-2:]], ["evaluate", "evaluate"])

    def test_v7_runner_exports_the_declared_allocator_only_to_the_trainer(self) -> None:
        runner_path = ROOT / "scripts" / "run_rlaif_grpo_prompt_ensemble_v1.py"
        with patch.dict(os.environ, {"MAL2026_RLAIF_CONFIG": "configs/rlaif_grpo_prompt_ensemble.v7.json", "MAL2026_RLAIF_RUNTIME_ID": "20260722-099"}, clear=False):
            spec = importlib.util.spec_from_file_location("rlaif_v7_allocator_contract", runner_path)
            self.assertIsNotNone(spec and spec.loader)
            runner = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(runner)
        captured: dict[str, str] = {}
        runner.arm_dir = lambda *_: ROOT / "outputs" / "rlaif-grpo-contract-no-output"
        runner.write_config = lambda *_: Path("/tmp/rlaif-grpo-contract-runtime.json")
        runner.run_stage = lambda _name, _command, environment: captured.update(environment)
        runner.verify_training = lambda *_: {}
        runner.ledger = lambda *_: None
        runner.train_arm("midm2_base", "bundle", "all5", "full", "http://127.0.0.1:18520", Path("/tmp/rlaif-reward-attestation.json"), "2",
                         rollout_endpoint="http://127.0.0.1:18570", rollout_attestation=Path("/tmp/rlaif-policy-attestation.json"))
        self.assertEqual(captured["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(captured["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(captured["MAL2026_RLAIF_ROLLOUT_ENDPOINT"], "http://127.0.0.1:18570")

    def test_custom_rollout_coalesces_repeat_sampler_groups_without_multiplying_rows(self) -> None:
        class Tokenizer:
            eos_token_id = 2

            def apply_chat_template(self, messages, **_):
                return [10, len(messages[0]["content"])]

            def encode(self, text, **_):
                return [len(text)]

        rollout = StructuredVLLMRollout.__new__(StructuredVLLMRollout)
        rollout.settings = RLAIFSettings.from_json(ROOT / "configs" / "rlaif_grpo_prompt_ensemble.v2.json")
        rollout.run = SimpleNamespace(seed=17)
        rollout.tokenizer = Tokenizer()
        rollout.last_synced_step = 0  # bypass I/O; exercise only the batch contract.
        rollout.request_count = 0
        rollout.completion_count = 0
        calls = []

        def request(messages, seed, num_choices):
            calls.append((messages, seed, num_choices))
            return ["{}"] * num_choices

        rollout._request = request
        first = [{"role": "user", "content": "가"}]
        second = [{"role": "user", "content": "나다"}]
        result = rollout([first] * 4 + [second] * 4, SimpleNamespace(state=SimpleNamespace(global_step=0)))
        self.assertEqual([count for _, _, count in calls], [4, 4])
        self.assertEqual(len(result["prompt_ids"]), 8)
        self.assertEqual(len(result["completion_ids"]), 8)
        self.assertEqual(rollout.request_count, 2)
        self.assertEqual(rollout.completion_count, 8)


if __name__ == "__main__":
    unittest.main()
