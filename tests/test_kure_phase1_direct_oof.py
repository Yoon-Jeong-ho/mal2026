from __future__ import annotations

from dataclasses import replace
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

import mal2026.kure_phase1_direct_oof as module
from mal2026.kure_phase1_direct_oof import (
    KUREPhase1DirectOOFConfig, KUREPhase1DirectOOFError, _atomic_private_jsonl,
    _held_gold_after_persist, _validate_outer_public, aggregate, direct_coral_expected_score,
    load_phase1_state, prediction_band_diagnostics, run, summarize_gpu_telemetry,
)


class KUREPhase1DirectOOFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = json.loads(Path("configs/kure_phase1_direct_oof.v1.json").read_text(encoding="utf-8"))

    def config(self) -> KUREPhase1DirectOOFConfig:
        return KUREPhase1DirectOOFConfig.from_mapping(self.raw)

    def authorized_for_spy(self) -> KUREPhase1DirectOOFConfig:
        return replace(self.config(), status="authorized", execution_authorized=True,
                       task_card_sha256="a" * 64, task_card_commit="b" * 40,
                       preparer_sha256="c" * 64, preparer_commit="d" * 40,
                       preparation_request_config_sha256="g" * 64,
                       label_free_projection_sha256="e" * 64, label_free_manifest_sha256="f" * 64)

    def pending_config(self) -> KUREPhase1DirectOOFConfig:
        raw = dict(self.raw)
        raw.update({"status": "pending_scientific_authorization", "execution_authorized": False})
        for key in ("task_card_sha256", "task_card_commit", "preparer_sha256", "preparer_commit",
                    "preparation_request_config_sha256", "label_free_projection_sha256",
                    "label_free_manifest_sha256"):
            raw[key] = ""
        return KUREPhase1DirectOOFConfig.from_mapping(raw)

    def test_exact_fifteen_checkpoints_and_five_memberships(self) -> None:
        config = self.config(); config.validate(require_dependencies=False)
        self.assertEqual(len(config.checkpoint_bindings), 15)
        self.assertEqual([(item.outer_fold, item.axis) for item in config.checkpoint_bindings],
                         [(fold, axis) for fold in range(5) for axis in ("content", "organization", "expression")])
        self.assertEqual([item.outer_fold for item in config.fold_membership_bindings], list(range(5)))
        self.assertTrue(all(len(item.sha256) == 64 for item in config.fold_membership_bindings))
        self.assertTrue(all(module.file_sha256(Path(item.path)) == item.sha256 for item in config.checkpoint_bindings))
        self.assertTrue(all(module.file_sha256(Path(item.path)) == item.sha256 for item in config.fold_membership_bindings))
        self.assertEqual((config.smoke_gpu, config.full_gpu_scope), (0, (0, 1, 2, 3)))
        self.assertEqual(config.fold_gpu_mapping, {"0":0,"1":1,"2":2,"3":3,"4":0})
        self.assertEqual(config.telemetry_columns, ("timestamp","index","uuid","name","memory.total","driver_version","utilization.gpu","memory.used"))
        self.assertEqual((config.smoke_minimum_samples, config.full_minimum_samples, config.telemetry_interval_seconds), (1,2,30))
        self.assertEqual(config.source_stage3_aggregate_sha256,
                         "eb13d63d28258f331ebcefb2b79f4364ddcc9ff38eec38da533665222706e0e3")

    def test_pending_config_is_fail_closed_for_actual_execution(self) -> None:
        config = self.pending_config()
        self.assertEqual((config.status, config.execution_authorized), ("pending_scientific_authorization", False))
        self.assertEqual((config.label_free_projection_sha256, config.task_card_sha256,
                          config.preparation_request_config_sha256), ("", "", ""))
        self.assertIn("preparation_request_config_sha256", inspect.getsource(module._load_label_free_projection))
        validated = run(config, outer_fold=0, validate_only=True)
        self.assertEqual((validated["status"], validated["gpu_used"]), ("validated", False))
        with self.assertRaisesRegex(KUREPhase1DirectOOFError, "not authorized"): run(config, outer_fold=0)
        with self.assertRaisesRegex(KUREPhase1DirectOOFError, "not authorized"): aggregate(config)

    def test_preinference_dependency_spy_never_touches_gold_paths(self) -> None:
        config = self.authorized_for_spy(); touched = []
        def record(path, digest, label, *, private): touched.append(str(path))
        completed = [subprocess.CompletedProcess([], 0, stdout=b"card", stderr=b""),
                     subprocess.CompletedProcess([], 0, stdout=b"preparer", stderr=b""),
                     subprocess.CompletedProcess([], 0, stdout=b"request", stderr=b"")]
        def fake_sha(payload=b""):
            values = {b"card": "a" * 64, b"preparer": "c" * 64, b"request": "g" * 64}
            return mock.Mock(hexdigest=lambda: values[payload])
        with mock.patch.object(module, "_verify_ordinary_file", side_effect=record), \
             mock.patch.object(module, "_validate_stage3_contract"), \
             mock.patch.object(module, "_load_label_free_projection", return_value=({}, {})), \
             mock.patch.object(module, "validate_backbone_without_validation"), \
             mock.patch.object(module, "_source_config", return_value=mock.Mock(backbone=mock.Mock())), \
             mock.patch.object(module.subprocess, "run", side_effect=completed), \
             mock.patch.object(module, "sha256", side_effect=fake_sha):
            config.validate_safe_dependencies()
        forbidden = {config.train_path, config.fold_manifest_path, config.fold_rows_path, config.r0_oof_prediction_path}
        self.assertTrue(touched)
        self.assertFalse(forbidden & set(touched))

    def test_checkpoint_loader_explicitly_ignores_crt_head(self) -> None:
        checked = []
        for binding in self.config().checkpoint_bindings:
            state, disclosure = load_phase1_state(Path(binding.path), binding.sha256)
            self.assertIn("score.weight", state); self.assertIn("cut_gaps", state)
            self.assertFalse(any(key.startswith("head.") for key in state))
            self.assertEqual(disclosure["ignored_crt_tensors"], ["head.bias", "head.weight"])
            self.assertGreater(disclosure["loaded_lora_tensor_count"], 0)
            checked.append((binding.outer_fold, binding.axis))
        self.assertEqual(len(checked), 15)

    def test_direct_coral_decode_is_expected_score_not_argmax(self) -> None:
        logits = torch.tensor([[3.0, 1.5, -0.5, -2.0], [0.0, 0.0, 0.0, 0.0]])
        score = direct_coral_expected_score(logits)
        self.assertEqual(tuple(score.shape), (2,)); self.assertTrue(torch.all((1 <= score) & (score <= 5)))
        self.assertFalse(torch.allclose(score, score.round()))

    def test_private_publish_no_clobber_acl_and_symlink_rejection(self) -> None:
        anchor = Path("data/processed/restricted")
        with tempfile.TemporaryDirectory(dir=anchor) as temporary:
            path = Path(temporary) / "nested" / "predictions.jsonl"
            digest = _atomic_private_jsonl(path, [{"source_id": "x"}])
            self.assertEqual(len(digest), 64); self.assertEqual(path.stat().st_mode & 0o007, 0)
            with self.assertRaisesRegex(KUREPhase1DirectOOFError, "overwrite"): _atomic_private_jsonl(path, [])
            link = Path(temporary) / "link.jsonl"; link.symlink_to(path)
            with self.assertRaisesRegex(KUREPhase1DirectOOFError, "ordinary"):
                module._assert_private_file(link)

    def valid_outer(self, private: Path) -> dict:
        config = self.config(); bindings = [item for item in config.checkpoint_bindings if item.outer_fold == 0]
        return {"schema_version": module.SCHEMA_VERSION, "status": "completed", "mode": "outer_fold",
                "nonselectable": False, "run_id": config.run_id, "outer_fold": 0, "records": 400,
                "method": module.METHOD, "source_method": module.SOURCE_METHOD,
                "decode": "saved_phase1_CORAL_PMF_expected_score", "training_performed": False,
                "calibration_performed": False, "selection_performed": False,
                "config_sha256": module.config_sha256(config), "config_file_sha256": module._config_file_sha256(),
                "task_card_sha256": config.task_card_sha256, "task_card_commit": config.task_card_commit,
                "source_stage3_aggregate_sha256": config.source_stage3_aggregate_sha256,
                "fold_manifest_sha256": config.fold_manifest_sha256, "fold_rows_sha256": config.fold_rows_sha256,
                "r0_oof_prediction_sha256": config.r0_oof_prediction_sha256,
                "restricted_prediction_sha256": module.file_sha256(private), "validation_rows_loaded": False,
                "average_target_used": False, "axis_bindings": [
                    {"axis": item.axis, "checkpoint_sha256": item.sha256,
                     "decode": "saved_phase1_CORAL_PMF_expected_score", "ignored_crt_tensors": ["head.bias", "head.weight"],
                     "ignored_crt_tensor_count": 2, "loaded_coral_tensors": ["cut_base", "cut_gaps", "score.bias", "score.weight"],
                     "loaded_lora_tensor_count": 288, "loaded_tensor_count": 292,
                     "lineage": {"arm": "aihub_full_backbone", "pooling": "cls_l2",
                                 "artifact_sha256": "ffdc985d56c655c03e8964927b127b24f0c5bb7fdde8d89e944941f5419cf25a"}}
                    for item in bindings]}

    def test_tampered_outer_axis_and_checkpoint_are_rejected(self) -> None:
        with tempfile.NamedTemporaryFile() as private:
            path = Path(private.name); valid = self.valid_outer(path)
            _validate_outer_public(valid, self.config(), 0, path)
            bad = json.loads(json.dumps(valid)); bad["axis_bindings"][0]["axis"] = "expression"
            with self.assertRaisesRegex(KUREPhase1DirectOOFError, "axis order"): _validate_outer_public(bad, self.config(), 0, path)
            bad = json.loads(json.dumps(valid)); bad["axis_bindings"][1]["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(KUREPhase1DirectOOFError, "checkpoint"): _validate_outer_public(bad, self.config(), 0, path)

    def test_prediction_band_counts_and_collapse_rate(self) -> None:
        values = np.asarray([[1.49, 2.5, 3.49], [3.5, 4.49, 4.5], [5.0, 3.0, 2.0]])
        result = prediction_band_diagnostics(values)
        self.assertEqual(result["total_axis_predictions"], 9)
        self.assertEqual(sum(result["half_up_band_counts"]["content"].values()), 3)
        self.assertEqual(result["band_3_4_count"], 5)
        self.assertAlmostEqual(result["band_3_4_collapse_rate"], 5 / 9)

    def test_telemetry_missing_gpu_and_minimum_samples_rejected(self) -> None:
        header = "timestamp,index,uuid,name,memory.total,driver_version,utilization.gpu,memory.used\n"
        row = "t,0,u,H100,81920,999,50,1000\n"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.csv"; path.write_text(header + row)
            with self.assertRaisesRegex(KUREPhase1DirectOOFError, "minimum-sample"):
                summarize_gpu_telemetry(path, (0, 1), 1)
            with self.assertRaisesRegex(KUREPhase1DirectOOFError, "minimum-sample"):
                summarize_gpu_telemetry(path, (0,), 2)
            summary = summarize_gpu_telemetry(path, (0,), 1)
            self.assertEqual(summary["gpus"][0]["peak_utilization_percent"], 50)
            for invalid in (
                "t,0,u,H100,81920,999,nan,1000\n", "t,0,u,H100,81920,999,101,1000\n",
                "t,0,u,H100,81920,999,50,90000\n", ",0,u,H100,81920,999,50,1000\n",
            ):
                path.write_text(header + invalid)
                with self.assertRaises(KUREPhase1DirectOOFError): summarize_gpu_telemetry(path, (0,), 1)

    def test_held_gold_requires_persisted_predictions_and_order(self) -> None:
        config = self.config()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(KUREPhase1DirectOOFError, "remain unavailable"):
                _held_gold_after_persist(Path(config.train_path), config.train_sha256, {"x"}, Path(temporary) / "missing")
        source = inspect.getsource(run)
        self.assertLess(source.index("_atomic_private_jsonl"), source.index("_held_gold_after_persist"))
        prefix = source[:source.index("_atomic_private_jsonl")]
        self.assertNotIn("train_path", prefix); self.assertNotIn("fold_rows_path", prefix); self.assertNotIn("r0_oof_prediction_path", prefix)

    def test_aggregate_authenticates_predictions_before_gold(self) -> None:
        source = inspect.getsource(aggregate)
        self.assertIn("decision = promotion_gate(", source)
        self.assertIn('"automatic_stage6_deployment_eligible": False', source)
        self.assertLess(source.index("predictions.update"), source.index("load_embedding_artifact"))
        self.assertLess(source.index("predictions.update"), source.index("load_raw_axis_gold"))
        self.assertLess(source.index("predictions.update"), source.index("verify_post_prediction_gold_dependencies"))
        self.assertNotIn("verify_post_prediction_gold_dependencies", inspect.getsource(run))

    def test_post_prediction_gold_dependency_verifier_checks_current_bindings(self) -> None:
        module.verify_post_prediction_gold_dependencies(self.config())

    def test_pending_launcher_dynamic_sentinel_before_nvidia(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            sentinel = temporary_path / "nvidia-called"; fake = temporary_path / "nvidia-smi"
            fake.write_text(f"#!/bin/sh\ntouch {sentinel}\nexit 99\n"); fake.chmod(0o755)
            pending = dict(self.raw)
            pending.update({"status": "pending_scientific_authorization", "execution_authorized": False})
            for key in ("task_card_sha256", "task_card_commit", "preparer_sha256", "preparer_commit",
                        "preparation_request_config_sha256", "label_free_projection_sha256",
                        "label_free_manifest_sha256"):
                pending[key] = ""
            pending_path = temporary_path / "pending.json"; pending_path.write_text(json.dumps(pending))
            launcher = Path("scripts/run_kure_phase1_direct_oof_gpu0_3.sh").read_text()
            launcher = launcher.replace('CONFIG="$ROOT/configs/kure_phase1_direct_oof.v1.json"',
                                        f'CONFIG="{pending_path}"')
            copied = Path("scripts") / f".test-direct-pending-{os.getpid()}.sh"
            try:
                copied.write_text(launcher); copied.chmod(0o700)
                result = subprocess.run(["bash", str(copied), "smoke"],
                                        env={**os.environ, "PATH": f"{temporary}:{os.environ['PATH']}"},
                                        capture_output=True, text=True)
            finally:
                copied.unlink(missing_ok=True)
            self.assertNotEqual(result.returncode, 0); self.assertFalse(sentinel.exists())
            self.assertIn("not authorized", result.stderr)

    def test_scheduler_state_parser_is_fail_closed(self) -> None:
        launched = {"schema_version": "mal2026-vllm-idle-arm-state-v1", "status": "launched",
                    "physical_gpus": [0, 1, 2, 3], "run_id": "vllm-soak-gpu0-3-120h-20260803-004"}
        self.assertIn("launched", module.scheduler_state_conflict(launched, (0,), age_seconds=0))
        armed = {"schema_version": "mal2026-vllm-idle-arm-state-v1", "status": "armed",
                 "physical_gpus": [0, 1, 2, 3], "planned_run_id": "vllm-soak-gpu0-3-120h-20260803-004",
                 "idle_required_seconds": 1800, "consecutive_idle_seconds": 1000}
        self.assertIn("stale", module.scheduler_state_conflict(armed, (0,), age_seconds=121))
        armed["consecutive_idle_seconds"] = 1501
        self.assertIn("five-minute", module.scheduler_state_conflict(armed, (0,), age_seconds=1))
        stopped = {**launched, "status": "launched_then_stopped_by_user"}
        self.assertIsNone(module.scheduler_state_conflict(stopped, (0,), age_seconds=999))
        with self.assertRaises(KUREPhase1DirectOOFError):
            module.scheduler_state_conflict({**launched, "status": "mystery"}, (0,), age_seconds=0)
        with self.assertRaises(KUREPhase1DirectOOFError):
            module.scheduler_state_conflict({**launched, "schema_version": "bad"}, (0,), age_seconds=0)
        delayed_005 = {"schema_version": "mal2026-vllm-idle-arm-state-v1", "status": "delayed",
                       "physical_gpus": [0, 1, 2, 3],
                       "planned_run_id": "vllm-soak-gpu0-3-120h-20260803-005",
                       "idle_required_seconds": 1800, "consecutive_idle_seconds": 0}
        self.assertIsNone(module.scheduler_state_conflict(
            delayed_005, (0, 1, 2, 3), age_seconds=1,
            expected_run_id="vllm-soak-gpu0-3-120h-20260803-005"))
        with self.assertRaises(KUREPhase1DirectOOFError):
            module.scheduler_state_conflict(
                delayed_005, (0,), age_seconds=1,
                expected_run_id="vllm-soak-gpu0-3-120h-20260803-004")

    def test_setproctitle_exposes_direct_stage(self) -> None:
        import setproctitle
        previous = setproctitle.getproctitle()
        try:
            self.assertEqual(module.set_process_title("smoke:f0:content"),
                             "mal2026:direct:smoke:f0:content")
            self.assertEqual(setproctitle.getproctitle(), "mal2026:direct:smoke:f0:content")
            with self.assertRaises(KUREPhase1DirectOOFError):
                module.set_process_title("bad stage")
        finally:
            setproctitle.setproctitle(previous)

    def test_preparer_membership_read_only_and_projection_helper(self) -> None:
        spec = importlib.util.spec_from_file_location("phase1_preparer", "scripts/prepare_kure_phase1_direct_input.py")
        preparer = importlib.util.module_from_spec(spec); spec.loader.exec_module(preparer)
        aggregate = json.loads(Path(self.raw["source_stage3_aggregate_path"]).read_text())
        paths = [Path(item["path"]) for item in self.raw["fold_membership_bindings"]]
        before = [(path.stat().st_mtime_ns, preparer.digest(path)) for path in paths]
        folds, evidence = preparer.load_memberships(self.raw, aggregate)
        after = [(path.stat().st_mtime_ns, preparer.digest(path)) for path in paths]
        self.assertEqual(before, after); self.assertEqual((len(folds), len(evidence)), (2000, 5))
        preparer_source = inspect.getsource(preparer.main)
        self.assertIn('"labels_present": False', preparer_source)
        self.assertIn('"preparation_request_config_sha256"', preparer_source)
        self.assertNotIn('"config_sha256":', preparer_source)
        with tempfile.TemporaryDirectory(dir="data/processed/restricted") as temporary:
            train = Path(temporary) / "train.jsonl"; rows=[]; expected={}
            for fold in range(5):
                for index in range(2):
                    identifier=f"id-{fold}-{index}"; expected[identifier]=fold
                    rows.append({"id":identifier,"document_id":"d","prompt_num":"p","prompt":"q","essay":"e",
                                 "score":{"content":3,"organization":3,"expression":3,"average":3}})
            train.write_text("".join(json.dumps(row)+"\n" for row in rows)); checksum=preparer.digest(train)
            projection, counts=preparer.create_projection_rows(train, checksum, expected, expected_per_fold=2)
            self.assertEqual(counts, {str(fold):2 for fold in range(5)})
            self.assertTrue(all(not ({"score","average","gold"} & set(row)) for row in projection))
            output=Path(temporary)/"out"/"rows.jsonl"
            preparer.publish_private(output, (json.dumps(row)+"\n" for row in projection), Path("data/processed/restricted"))
            with self.assertRaises(RuntimeError): preparer.publish_private(output, [], Path("data/processed/restricted"))

    def test_cleanup_signals_only_live_owned_tracked_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "completed"
            external = subprocess.Popen(["sleep", "3"])
            script = f'''set -Eeuo pipefail
PIDS=()
remove_tracked_pid() {{ target="$1"; remaining=(); for pid in "${{PIDS[@]:-}}"; do [[ "$pid" == "$target" ]] || remaining+=("$pid"); done; PIDS=("${{remaining[@]}}"); }}
wait_tracked_pid() {{ pid="$1"; rc=0; wait "$pid" || rc=$?; remove_tracked_pid "$pid"; return "$rc"; }}
cleanup() {{ rc=$?; trap - EXIT INT TERM; declare -A live=(); output="$(jobs -pr)"; while IFS= read -r pid; do [[ -n "$pid" ]] && live["$pid"]=1; done <<<"$output"; for pid in "${{PIDS[@]:-}}"; do [[ -n "${{live[$pid]:-}}" ]] && kill "$pid" 2>/dev/null || :; done; exit "$rc"; }}
trap cleanup EXIT; trap 'exit 143' TERM
(sleep 0.02) & PIDS+=("$!"); wait_tracked_pid "${{PIDS[0]}}"
PIDS+=("{external.pid}")
(sleep 2; touch "{marker}") & PIDS+=("$!")
(sleep 0.1; kill -TERM $$) &
wait_tracked_pid "${{PIDS[1]}}"
'''
            try:
                started=time.monotonic(); result=subprocess.run(["bash","-c",script]); elapsed=time.monotonic()-started
                self.assertNotEqual(result.returncode,0); self.assertLess(elapsed,1.5); self.assertFalse(marker.exists())
                self.assertIsNone(external.poll(), "non-job/external PID was signaled")
            finally:
                external.terminate(); external.wait(timeout=2)

    def test_empty_tracked_array_cleanup_reaches_telemetry_wait_and_failure_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stop = Path(temporary) / "stop"; waited = Path(temporary) / "telemetry-waited"
            ledger = Path(temporary) / "ledger"
            script = f'''set -Eeuo pipefail
PIDS=(); TELEMETRY_STOP="{stop}"; FAILED_ARMED=1
append_event() {{ echo "$1" >>"{ledger}"; }}
(while [[ ! -e "$TELEMETRY_STOP" ]]; do sleep 0.01; done; echo done >"{waited}") & TELEMETRY_PID="$!"
cleanup() {{ rc=$?; trap - EXIT INT TERM; live_output="$(jobs -pr)"; declare -A live_jobs=(); while IFS= read -r pid; do [[ -n "$pid" ]] && live_jobs["$pid"]=1; done <<<"$live_output"; for pid in "${{PIDS[@]}}"; do [[ -n "${{live_jobs[$pid]:-}}" ]] && kill "$pid" || :; done; touch "$TELEMETRY_STOP"; wait "$TELEMETRY_PID"; if (( FAILED_ARMED )); then append_event stage_failed; fi; exit "$rc"; }}
trap cleanup EXIT
false
'''
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stderr)
            self.assertEqual(waited.read_text().strip(), "done")
            self.assertEqual(ledger.read_text().strip(), "stage_failed")

    def test_task_card_and_launcher_contract_text(self) -> None:
        card = Path("docs/experiment_records/kure_phase1_direct_oof_20260803_001.md").read_text()
        self.assertIn("data-steward", card); self.assertIn("must not open, stat/hash", card)
        launcher = Path("scripts/run_kure_phase1_direct_oof_gpu0_3.sh").read_text()
        self.assertNotIn("|| true", launcher); self.assertIn("flock -n", launcher)
        self.assertIn("stage_failed", launcher)
        self.assertIn('outputs/reservations/gpu0-3-watchdog-coordination-v1', launcher)
        self.assertIn('vllm-idle-arm-gpu0-3-20260803-004/state.json', launcher)
        self.assertIn('vllm-idle-arm-gpu0-3-20260803-005/state.json', launcher)
        self.assertIn('--scheduler-run-id "$run_id"', launcher)
        self.assertIn('setproctitle.setproctitle', launcher)
        self.assertNotIn('glob("*/state.json")', launcher)
        self.assertIn("telemetry_summary_sha256", launcher); self.assertIn("evidence_sha256", launcher)
        self.assertIn("task_card_sha256", launcher); self.assertIn("config_sha256", launcher)
        self.assertIn("sleep(30)", launcher); self.assertIn('MIN_SAMPLES=1', launcher); self.assertIn('MIN_SAMPLES=2', launcher)
        card = Path("docs/experiment_records/kure_phase1_direct_oof_20260803_001.md").read_text()
        self.assertIn("fold 0→GPU0", card); self.assertIn("fold 4→GPU0", card)
        smoke_line = next(line for line in launcher.splitlines() if "--smoke >" in line)
        aggregate_line = next(line for line in launcher.splitlines() if "--aggregate >" in line)
        self.assertIn('& PIDS+=("$!")', smoke_line); self.assertIn('& PIDS+=("$!")', aggregate_line)
        self.assertIn('live_output="$(jobs -pr)"', launcher)
        self.assertIn('wait_tracked_pid "${PIDS[0]}"', launcher)
        self.assertIn('remove_tracked_pid "$pid"', launcher)
        self.assertNotIn('/proc/', launcher)


if __name__ == "__main__": unittest.main()
