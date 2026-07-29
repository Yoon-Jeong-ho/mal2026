from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mal2026.solar_target_augmentation import AugmentationTask, SourceRow


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_solar_target_augmentation.py"
SPEC = importlib.util.spec_from_file_location("solar_target_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def task() -> AugmentationTask:
    source = SourceRow(
        identifier="source-1",
        document_id="document-1",
        prompt="논제",
        essay="충분히 긴 한국어 논증문 원문이다. 근거와 결론을 포함한다.",
        score=(3.5, 4.0, 2.5),
    )
    return AugmentationTask("source-1::solar-target::content::1", source, "content", 1)


class SolarTargetRunnerTests(unittest.TestCase):
    def test_official_docker_tp4_has_no_pull_or_eager_mode(self) -> None:
        command = MODULE.server_command(19420)
        self.assertEqual(command[:2], ["docker", "run"])
        self.assertNotIn("pull", command)
        self.assertEqual(command[command.index("--name") + 1], "mal2026-solar-target-19420")
        self.assertEqual(command[command.index("--gpus") + 1], '"device=0,1,2,3"')
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "4")
        self.assertIn("--enable-expert-parallel", command)
        self.assertIn("--enable-prefix-caching", command)
        self.assertIn("--enable-chunked-prefill", command)
        self.assertNotIn("--enforce-eager", command)
        self.assertIn("HF_HUB_OFFLINE=1", command)
        self.assertIn("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0", command)
        self.assertTrue(any("dst=/root/.cache/vllm" in item for item in command))

    def test_external_binding_includes_host_offline_and_cache_settings(self) -> None:
        container_name = "bound-solar"
        command = MODULE.server_command(19420, container_name)
        item = {
            "Id": "container-id",
            "Image": MODULE.DOCKER_IMAGE_ID,
            "State": {"Running": True, "StartedAt": "fixed"},
            "Config": {
                "Image": MODULE.DOCKER_IMAGE,
                "Cmd": command[command.index(MODULE.DOCKER_IMAGE) + 1:],
                "Env": [
                    "HF_HUB_OFFLINE=1",
                    "TRANSFORMERS_OFFLINE=1",
                    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0",
                ],
            },
            "Mounts": [
                {"Source": str(MODULE.RUNTIME_MODEL),
                 "Destination": MODULE.CONTAINER_MODEL, "RW": False},
                {"Source": str(MODULE.VLLM_CACHE_DIR),
                 "Destination": "/root/.cache/vllm", "RW": True},
            ],
            "HostConfig": {
                "NetworkMode": "host",
                "IpcMode": "host",
                "AutoRemove": True,
                "DeviceRequests": [{
                    "DeviceIDs": ["0", "1", "2", "3"],
                    "Capabilities": [["gpu"]],
                }],
            },
        }

        def inspect(value: dict) -> dict:
            with patch.object(
                MODULE.subprocess, "check_output", return_value=json.dumps([value])
            ):
                return MODULE.external_server_binding(container_name, 19420)

        binding = inspect(item)
        self.assertEqual(binding["network_mode"], "host")
        self.assertEqual(binding["ipc_mode"], "host")
        for mutate in (
            lambda value: value["HostConfig"].update(NetworkMode="bridge"),
            lambda value: value["HostConfig"].update(IpcMode="private"),
            lambda value: value["HostConfig"].update(AutoRemove=False),
            lambda value: value["Config"].update(Env=["HF_HUB_OFFLINE=1"]),
            lambda value: value["Mounts"].pop(),
        ):
            invalid = copy.deepcopy(item)
            mutate(invalid)
            with self.assertRaises(MODULE.SolarTargetRunError):
                inspect(invalid)

    def test_schemas_are_closed_and_integer_score_only(self) -> None:
        verifier = MODULE.verifier_output_schema()
        self.assertFalse(verifier["additionalProperties"])
        self.assertEqual(set(verifier["required"]), {"content", "organization", "expression"})
        self.assertEqual(verifier["properties"]["content"]["properties"]["score"]["type"], "integer")
        fidelity = MODULE.fidelity_output_schema()
        self.assertFalse(fidelity["additionalProperties"])
        self.assertEqual(fidelity["properties"]["source_based"]["type"], "boolean")

    def test_request_contract_injects_response_schema_into_solar_template(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"response_format": response_format', source)
        self.assertIn('"chat_template_kwargs": {', source)

    def test_label_replaces_only_target_and_has_no_average(self) -> None:
        self.assertEqual(
            MODULE.label_scores(task()),
            {"content": 1, "organization": 4.0, "expression": 2.5},
        )
        self.assertNotIn("average", MODULE.label_scores(task()))

    def test_seed_is_stable_and_stage_specific(self) -> None:
        value = MODULE.stable_seed(task(), "editor", 1)
        self.assertEqual(value, MODULE.stable_seed(task(), "editor", 1))
        self.assertNotEqual(value, MODULE.stable_seed(task(), "verifier", 1))
        self.assertNotEqual(value, MODULE.stable_seed(task(), "editor", 2))

    def test_blind_seed_depends_only_on_visible_messages(self) -> None:
        messages = [
            {"role": "system", "content": "blind"},
            {"role": "user", "content": "same prompt and essay"},
        ]
        value = MODULE.stable_blind_seed("verifier", messages)
        self.assertEqual(value, MODULE.stable_blind_seed("verifier", list(messages)))
        changed = [*messages[:-1], {"role": "user", "content": "different essay"}]
        self.assertNotEqual(value, MODULE.stable_blind_seed("verifier", changed))
        self.assertNotEqual(value, MODULE.stable_blind_seed("fidelity", messages))

    def test_runner_has_four_independent_families_and_no_rationale_feedback_mapper(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertEqual(MODULE.RETRIES, 4)
        self.assertNotIn("target_deficiency_labels", source)
        self.assertNotIn("previous_blind_scores", source)
        self.assertNotIn("target_deficiency:", source)

    def test_full_mode_fails_closed_without_scientific_authorization(self) -> None:
        MODULE.validate_execution_gate("smoke", {"execution_gate": {
            "full_run_authorized": False
        }})
        with self.assertRaisesRegex(
            MODULE.SolarTargetRunError, "not scientifically authorized"
        ):
            MODULE.validate_execution_gate("full", {"execution_gate": {
                "full_run_authorized": False
            }})
        MODULE.validate_execution_gate("full", {"execution_gate": {
            "full_run_authorized": True
        }})

    def test_full_approval_requires_lead_and_subagent(self) -> None:
        bindings = {"prompt": "fixed"}
        result = {
            "schema_version": "mal2026-solar-axis-target-result-v2",
            "status": "completed",
            "mode": "smoke",
            "source_records": 5,
            "blind_source_records": 5,
            "records": 75,
            "records_expected": 75,
            "variants_per_source": 15,
            "axis_counts": {axis: 25 for axis in ("content", "organization", "expression")},
            "target_score_counts": {
                axis: {str(score): 5 for score in range(1, 6)}
                for axis in ("content", "organization", "expression")
            },
            "failures": {},
            "binding_sha256": MODULE.canonical_sha(bindings),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            augmented_path = root / "augmented.train.jsonl"
            augmented_rows = []
            for source_index in range(5):
                for axis in ("content", "organization", "expression"):
                    for score in range(1, 6):
                        augmented_rows.append({
                            "source_id": f"source-{source_index}",
                            "augmented_id": f"source-{source_index}::{axis}::{score}",
                            "target_axis": axis,
                            "target_score": score,
                        })
            augmented_path.write_text(
                "".join(json.dumps(row) + "\n" for row in augmented_rows), encoding="utf-8"
            )
            result["augmented_train_path"] = str(augmented_path)
            result["augmented_train_sha256"] = MODULE.file_sha256(augmented_path)
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            review_path = root / "review.json"
            review = {
                "schema_version": "mal2026-solar-smoke-review-v1",
                "status": "approved",
                "all_records_reviewed": True,
                "result_sha256": MODULE.file_sha256(result_path),
                "augmented_train_sha256": MODULE.file_sha256(augmented_path),
                "reviewed_record_count": 75,
                "reviewed_cell_counts": result["target_score_counts"],
                "unresolved_findings": 0,
                "reviewers": ["lead"],
            }
            review_path.write_text(json.dumps(review), encoding="utf-8")
            previous_root = MODULE.RESTRICTED_ROOT
            MODULE.RESTRICTED_ROOT = root
            try:
                with self.assertRaises(MODULE.SolarTargetRunError):
                    MODULE.validate_smoke_approval(result_path, review_path, bindings)
                review["reviewers"].append("subagent:quality")
                review_path.write_text(json.dumps(review), encoding="utf-8")
                approval = MODULE.validate_smoke_approval(result_path, review_path, bindings)
                self.assertEqual(approval["smoke_result_sha256"], MODULE.file_sha256(result_path))
                self.assertEqual(approval["smoke_augmented_train_sha256"],
                                 MODULE.file_sha256(augmented_path))
            finally:
                MODULE.RESTRICTED_ROOT = previous_root


if __name__ == "__main__":
    unittest.main()
