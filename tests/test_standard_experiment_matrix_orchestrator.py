from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_standard_experiment_matrix.sh"
HASH = "0" * 64


class StandardExperimentMatrixShellTests(unittest.TestCase):
    def test_shell_parses(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True, cwd=ROOT)

    def test_dry_run_is_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "new-runtime"
            command = [
                str(SCRIPT), "--dry-run", "--runtime-root", str(runtime), "--run-prefix", "dry-run-test",
                "--prepared-manifest", "/tmp/manifest.json", "--validation-sha256", HASH,
                "--qwen-model", "/tmp/qwen", "--qwen3-model", "/tmp/qwen3", "--nv-model", "/tmp/nv",
                "--nv-review-json", "/tmp/nv-review.json",
            ]
            completed = subprocess.run(command, check=True, cwd=ROOT, text=True, capture_output=True)
            self.assertIn("DRY RUN: no paths checked, files created, or jobs launched.", completed.stdout)
            self.assertIn("Stable decoder settings: per-device batch=1, accumulation=16", completed.stdout)
            self.assertIn("CUDA_VISIBLE_DEVICES=0,1,2,3; DDP/vLLM GPUs=4", completed.stdout)
            self.assertFalse(runtime.exists())

    def test_rejects_batch_override_that_changes_global_effective_batch(self) -> None:
        command = [
            str(SCRIPT), "--dry-run", "--runtime-root", "/tmp/matrix", "--run-prefix", "bad-batch",
            "--prepared-manifest", "/tmp/manifest.json", "--validation-sha256", HASH,
            "--qwen-model", "/tmp/qwen", "--qwen3-model", "/tmp/qwen3", "--nv-model", "/tmp/nv",
            "--nv-review-json", "/tmp/nv-review.json", "--decoder-grad-accum", "8",
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        self.assertNotEqual(0, completed.returncode)
        self.assertIn("decoder settings must preserve global effective batch 64", completed.stderr)

    def test_gpu_preflight_ignores_busy_unselected_devices_but_rejects_selected_ones(self) -> None:
        """Mock nvidia-smi: only UUIDs for CUDA_VISIBLE_DEVICES=0,1,2,3 block."""
        import json
        import os
        import shutil
        import uuid

        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            for name in ("qwen", "qwen3", "nv"):
                (fixture / name).mkdir()
            (fixture / "manifest.json").write_text("{}", encoding="utf-8")
            review = {
                "model_id": "nvidia/NV-Embed-v2", "revision": "3fa59658547db50a1e8e3346cf057fd0c77ed6ef",
                "license_acknowledged": True, "use_case": "research_noncommercial", "reviewer": "test",
                "outcome": "approved", "reviewed_files": {"modeling_nvembed.py": "0" * 64},
            }
            (fixture / "review.json").write_text(json.dumps(review), encoding="utf-8")
            nvidia = fixture / "nvidia-smi"
            nvidia.write_text("#!/bin/sh\ncase \"$1\" in\n  --query-gpu=*) printf '0, UUID-0\\n1, UUID-1\\n2, UUID-2\\n3, UUID-3\\n4, UUID-4\\n5, UUID-5\\n6, UUID-6\\n7, UUID-7\\n' ;;\n  --query-compute-apps=*) printf '%s\\n' \"${MOCK_APPS:-}\" ;;\nesac\n", encoding="utf-8")
            torchrun = fixture / "torchrun"
            torchrun.write_text("#!/bin/sh\nexit 73\n", encoding="utf-8")
            nvidia.chmod(0o755); torchrun.chmod(0o755)

            def invoke(apps: str) -> tuple[subprocess.CompletedProcess[str], Path]:
                prefix = "gpu-preflight-" + uuid.uuid4().hex
                runtime = ROOT / "outputs" / "experiment-matrix" / prefix
                command = [
                    str(SCRIPT), "--runtime-root", str(runtime), "--run-prefix", prefix,
                    "--prepared-manifest", str(fixture / "manifest.json"), "--validation-sha256", HASH,
                    "--qwen-model", str(fixture / "qwen"), "--qwen3-model", str(fixture / "qwen3"),
                    "--nv-model", str(fixture / "nv"), "--nv-review-json", str(fixture / "review.json"),
                ]
                env = dict(os.environ, MAL2026_NVIDIA_SMI=str(nvidia), MAL2026_TORCHRUN=str(torchrun), MOCK_APPS=apps)
                return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=env), runtime

            other_busy, runtime = invoke("999, UUID-4")
            try:
                self.assertEqual(1, other_busy.returncode)  # mocked torchrun fails after preflight passes
                self.assertIn("failed step preserved", other_busy.stderr)
                self.assertTrue(runtime.is_dir())
            finally:
                shutil.rmtree(runtime, ignore_errors=True)
            selected_busy, runtime = invoke("100, UUID-0")
            try:
                self.assertEqual(1, selected_busy.returncode)
                self.assertIn("selected GPU UUID UUID-0 has active compute PID 100", selected_busy.stderr)
                entries = [json.loads(line) for line in (runtime / "matrix_ledger.jsonl").read_text().splitlines()]
                self.assertIn({"status": "failed", "step": "decoder-direct-selection", "detail": "selected_gpu_busy_or_preflight_refusal", "privacy": "aggregate_only"}, [{key: value for key, value in entry.items() if key != "time_utc"} for entry in entries])
            finally:
                shutil.rmtree(runtime, ignore_errors=True)

    def test_gpu_preflight_fails_closed_when_nvidia_smi_override_is_missing_or_invalid(self) -> None:
        """A preflight tool that cannot be executed must never permit torchrun."""
        import json
        import os
        import shutil
        import uuid

        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            for name in ("qwen", "qwen3", "nv"):
                (fixture / name).mkdir()
            (fixture / "manifest.json").write_text("{}", encoding="utf-8")
            review = {
                "model_id": "nvidia/NV-Embed-v2", "revision": "3fa59658547db50a1e8e3346cf057fd0c77ed6ef",
                "license_acknowledged": True, "use_case": "research_noncommercial", "reviewer": "test",
                "outcome": "approved", "reviewed_files": {"modeling_nvembed.py": "0" * 64},
            }
            (fixture / "review.json").write_text(json.dumps(review), encoding="utf-8")
            invoked = fixture / "torchrun-invoked"
            torchrun = fixture / "torchrun"
            torchrun.write_text(f"#!/bin/sh\nprintf invoked > {invoked}\nexit 73\n", encoding="utf-8")
            torchrun.chmod(0o755)

            for override in ("missing-nvidia-smi-for-mal2026-test", str(fixture / "not-an-executable")):
                prefix = "preflight-unavailable-" + uuid.uuid4().hex
                runtime = ROOT / "outputs" / "experiment-matrix" / prefix
                command = [
                    str(SCRIPT), "--runtime-root", str(runtime), "--run-prefix", prefix,
                    "--prepared-manifest", str(fixture / "manifest.json"), "--validation-sha256", HASH,
                    "--qwen-model", str(fixture / "qwen"), "--qwen3-model", str(fixture / "qwen3"),
                    "--nv-model", str(fixture / "nv"), "--nv-review-json", str(fixture / "review.json"),
                ]
                try:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                        env=dict(
                            os.environ,
                            MAL2026_NVIDIA_SMI=override,
                            MAL2026_TORCHRUN=str(torchrun),
                        ),
                    )
                    self.assertNotEqual(0, completed.returncode)
                    self.assertIn("unable to resolve nvidia-smi", completed.stderr)
                    self.assertTrue(runtime.is_dir())
                    self.assertFalse(invoked.exists(), "torchrun must not run after preflight refusal")
                    entries = [json.loads(line) for line in (runtime / "matrix_ledger.jsonl").read_text().splitlines()]
                    stage_entries = [entry for entry in entries if entry["step"] == "decoder-direct-selection"]
                    self.assertEqual(["started", "failed"], [entry["status"] for entry in stage_entries])
                    self.assertEqual("selected_gpu_busy_or_preflight_refusal", stage_entries[-1]["detail"])
                finally:
                    invoked.unlink(missing_ok=True)
                    shutil.rmtree(runtime, ignore_errors=True)

    def test_rejects_gpu_count_or_device_outside_current_boundary(self) -> None:
        base = [
            str(SCRIPT), "--dry-run", "--runtime-root", "/tmp/matrix", "--run-prefix", "gpu-boundary",
            "--prepared-manifest", "/tmp/manifest.json", "--validation-sha256", HASH,
            "--qwen-model", "/tmp/qwen", "--qwen3-model", "/tmp/qwen3", "--nv-model", "/tmp/nv",
            "--nv-review-json", "/tmp/nv-review.json",
        ]
        too_many = subprocess.run(base + ["--num-gpus", "5"], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(2, too_many.returncode)
        self.assertIn("restricted to at most GPUs 0,1,2,3", too_many.stderr)
        outside = subprocess.run(base + ["--cuda-visible-devices", "0,1,2,4"], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(2, outside.returncode)
        self.assertIn("restricted to physical GPUs 0,1,2,3", outside.stderr)

    def test_human_feedback_refit_uses_the_same_hyphenated_filename_it_reads(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('local config_mode="${mode//_/-}"', source)
        self.assertIn('"$CONFIGS/decoder-$config_mode-refit.json"', source)
        self.assertNotIn('"$CONFIGS/decoder-$mode-refit.json"', source)

    def test_nonfinite_artifact_is_ledgered_failed_not_completed(self) -> None:
        import json
        import os
        import shutil
        import uuid

        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            for name in ("qwen", "qwen3", "nv"):
                (fixture / name).mkdir()
            (fixture / "manifest.json").write_text("{}", encoding="utf-8")
            review = {"model_id": "nvidia/NV-Embed-v2", "revision": "3fa59658547db50a1e8e3346cf057fd0c77ed6ef", "license_acknowledged": True, "use_case": "research_noncommercial", "reviewer": "test", "outcome": "approved", "reviewed_files": {"modeling_nvembed.py": "0" * 64}}
            (fixture / "review.json").write_text(json.dumps(review), encoding="utf-8")
            nvidia = fixture / "nvidia-smi"
            nvidia.write_text("#!/bin/sh\ncase \"$1\" in --query-gpu=*) printf '0, UUID-0\\n1, UUID-1\\n2, UUID-2\\n3, UUID-3\\n' ;; --query-compute-apps=*) : ;; esac\n", encoding="utf-8")
            torchrun = fixture / "torchrun"
            torchrun.write_text(r'''#!/bin/sh
while [ "$#" -gt 0 ]; do if [ "$1" = --config ]; then config="$2"; break; fi; shift; done
out=$(sed -n 's/.*"output_dir": "\([^"]*\)".*/\1/p' "$config")
mkdir -p "$out"
printf '{"status": "completed", "best_metric": NaN}\n' > "$out/standard_training_complete.json"
''', encoding="utf-8")
            nvidia.chmod(0o755); torchrun.chmod(0o755)
            prefix = "nonfinite-ledger-" + uuid.uuid4().hex
            runtime = ROOT / "outputs" / "experiment-matrix" / prefix
            selection = ROOT / "outputs" / "standard-runs" / f"{prefix}-decoder-direct-selection"
            command = [str(SCRIPT), "--runtime-root", str(runtime), "--run-prefix", prefix, "--prepared-manifest", str(fixture / "manifest.json"), "--validation-sha256", HASH, "--qwen-model", str(fixture / "qwen"), "--qwen3-model", str(fixture / "qwen3"), "--nv-model", str(fixture / "nv"), "--nv-review-json", str(fixture / "review.json")]
            try:
                completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=dict(os.environ, MAL2026_NVIDIA_SMI=str(nvidia), MAL2026_TORCHRUN=str(torchrun)))
                self.assertEqual(1, completed.returncode)
                entries = [json.loads(line) for line in (runtime / "matrix_ledger.jsonl").read_text().splitlines()]
                stage_entries = [entry for entry in entries if entry["step"] == "decoder-direct-selection"]
                self.assertEqual(["started", "failed"], [entry["status"] for entry in stage_entries])
                self.assertEqual("artifact_or_provenance_validation_failed", stage_entries[-1]["detail"])
            finally:
                shutil.rmtree(runtime, ignore_errors=True)
                shutil.rmtree(selection, ignore_errors=True)

    def test_status_only_decoder_completion_is_ledgered_failed_not_completed(self) -> None:
        """A successful process cannot ledger a status-only training JSON as complete."""
        import json
        import os
        import shutil
        import uuid

        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            for name in ("qwen", "qwen3", "nv"):
                (fixture / name).mkdir()
            (fixture / "manifest.json").write_text("{}", encoding="utf-8")
            review = {"model_id": "nvidia/NV-Embed-v2", "revision": "3fa59658547db50a1e8e3346cf057fd0c77ed6ef", "license_acknowledged": True, "use_case": "research_noncommercial", "reviewer": "test", "outcome": "approved", "reviewed_files": {"modeling_nvembed.py": "0" * 64}}
            (fixture / "review.json").write_text(json.dumps(review), encoding="utf-8")
            nvidia = fixture / "nvidia-smi"
            nvidia.write_text("#!/bin/sh\ncase \"$1\" in --query-gpu=*) printf '0, UUID-0\\n1, UUID-1\\n2, UUID-2\\n3, UUID-3\\n' ;; --query-compute-apps=*) : ;; esac\n", encoding="utf-8")
            torchrun = fixture / "torchrun"
            torchrun.write_text(r'''#!/bin/sh
while [ "$#" -gt 0 ]; do if [ "$1" = --config ]; then config="$2"; break; fi; shift; done
out=$(sed -n 's/.*"output_dir": "\([^"]*\)".*/\1/p' "$config")
mkdir -p "$out"
printf '{"status": "completed"}\n' > "$out/standard_training_complete.json"
''', encoding="utf-8")
            nvidia.chmod(0o755); torchrun.chmod(0o755)
            prefix = "status-only-decoder-" + uuid.uuid4().hex
            runtime = ROOT / "outputs" / "experiment-matrix" / prefix
            selection = ROOT / "outputs" / "standard-runs" / f"{prefix}-decoder-direct-selection"
            command = [str(SCRIPT), "--runtime-root", str(runtime), "--run-prefix", prefix, "--prepared-manifest", str(fixture / "manifest.json"), "--validation-sha256", HASH, "--qwen-model", str(fixture / "qwen"), "--qwen3-model", str(fixture / "qwen3"), "--nv-model", str(fixture / "nv"), "--nv-review-json", str(fixture / "review.json")]
            try:
                completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=dict(os.environ, MAL2026_NVIDIA_SMI=str(nvidia), MAL2026_TORCHRUN=str(torchrun)))
                self.assertEqual(1, completed.returncode)
                entries = [json.loads(line) for line in (runtime / "matrix_ledger.jsonl").read_text().splitlines()]
                stage_entries = [entry for entry in entries if entry["step"] == "decoder-direct-selection"]
                self.assertEqual(["started", "failed"], [entry["status"] for entry in stage_entries])
                self.assertEqual("artifact_or_provenance_validation_failed", stage_entries[-1]["detail"])
            finally:
                shutil.rmtree(runtime, ignore_errors=True)
                shutil.rmtree(selection, ignore_errors=True)

    def test_embedded_validator_rejects_status_only_or_missing_primary_aggregate(self) -> None:
        """The release gate requires the evaluator's primary metric, not status alone."""
        source = SCRIPT.read_text(encoding="utf-8")
        begin = 'validate_stage_artifacts() { "$PYTHON" - "$@" <<\'PY\'\n'
        end = "\nPY\n}\nverify_stage()"
        self.assertIn(begin, source)
        self.assertIn(end, source)
        validator = source.split(begin, 1)[1].split(end, 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validator_path = root / "stage_validator.py"
            validator_path.write_text(validator, encoding="utf-8")
            artifact = root / "standard-evals" / "decoder" / "aggregate_metrics.json"
            artifact.parent.mkdir(parents=True)
            for payload in (
                {"status": "completed"},
                {"status": "completed", "metrics": {"record_count": 1.0}},
            ):
                artifact.write_text(json.dumps(payload), encoding="utf-8")
                completed = subprocess.run(
                    [sys.executable, str(validator_path), "decoder-direct-final", str(artifact)],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(0, completed.returncode)
                self.assertIn("invalid artifact", completed.stderr)

    def test_embedded_validator_accepts_zero_mae_selected_checkpoint(self) -> None:
        """Zero MAE is a valid optimum; the gate must only reject negative/nonfinite values."""
        source = SCRIPT.read_text(encoding="utf-8")
        begin = 'validate_stage_artifacts() { "$PYTHON" - "$@" <<\'PY\'\n'
        end = "\nPY\n}\nverify_stage()"
        validator = source.split(begin, 1)[1].split(end, 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validator_path = root / "stage_validator.py"
            validator_path.write_text(validator, encoding="utf-8")
            artifact = root / "selected_checkpoint.json"
            artifact.write_text(json.dumps({
                "status": "completed", "phase": "selection",
                "selected_primary_macro_mae": 0.0, "selected_global_step": 1,
                "candidates": [{"global_step": 1, "primary_macro_mae": 0.0}],
            }), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(validator_path), "decoder-direct-select-checkpoint", str(artifact)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)

    def test_resume_skips_verified_selection_and_preserves_attempt_history(self) -> None:
        """Retry a downstream failure without repeating a verified selection stage."""
        import hashlib
        import os
        import shutil
        import uuid

        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            for name in ("qwen", "qwen3", "nv"):
                (fixture / name).mkdir()
            (fixture / "manifest.json").write_text("{}", encoding="utf-8")
            (fixture / "review.json").write_text(json.dumps({
                "model_id": "nvidia/NV-Embed-v2", "revision": "3fa59658547db50a1e8e3346cf057fd0c77ed6ef",
                "license_acknowledged": True, "use_case": "research_noncommercial", "reviewer": "test",
                "outcome": "approved", "reviewed_files": {"modeling_nvembed.py": "0" * 64},
            }), encoding="utf-8")
            count = fixture / "preflight-count"
            nvidia = fixture / "nvidia-smi"
            nvidia.write_text("\n".join([
                "#!/bin/sh", "case \"$1\" in",
                "  --query-gpu=*) printf '0, UUID-0\\n1, UUID-1\\n2, UUID-2\\n3, UUID-3\\n' ;;",
                "  --query-compute-apps=*)",
                f"    n=$(cat {count} 2>/dev/null || echo 0); n=$((n + 1)); printf '%s' \"$n\" > {count}",
                "    if [ \"$n\" -ge 2 ]; then printf '777, UUID-0\\n'; fi ;;",
                "esac", "",
            ]), encoding="utf-8")
            invoked = fixture / "torchrun-invocations"
            torchrun = fixture / "torchrun"
            torchrun.write_text("\n".join([
                "#!/bin/sh", f"printf x >> {invoked}",
                "while [ \"$#\" -gt 0 ]; do", "  if [ \"$1\" = --config ]; then config=\"$2\"; break; fi", "  shift", "done",
                "out=$(sed -n 's/.*\"output_dir\": \"\\([^\"]*\\)\".*/\\1/p' \"$config\")",
                "mkdir -p \"$out/adapter\"",
                "printf '{\"status\":\"completed\",\"run_id\":\"test\",\"phase\":\"selection\",\"global_step\":1,\"best_metric\":0.2,\"train_metrics\":{\"train_loss\":0.2},\"selection_candidate_steps\":[1]}\\n' > \"$out/standard_training_complete.json\"",
                "printf '{}\\n' > \"$out/adapter/adapter_config.json\"", "",
            ]), encoding="utf-8")
            nvidia.chmod(0o755)
            torchrun.chmod(0o755)
            prefix = "resume-verified-" + uuid.uuid4().hex
            runtime = ROOT / "outputs" / "experiment-matrix" / prefix
            selection = ROOT / "outputs" / "standard-runs" / f"{prefix}-decoder-direct-selection"
            command = [
                str(SCRIPT), "--runtime-root", str(runtime), "--run-prefix", prefix,
                "--prepared-manifest", str(fixture / "manifest.json"), "--validation-sha256", HASH,
                "--qwen-model", str(fixture / "qwen"), "--qwen3-model", str(fixture / "qwen3"),
                "--nv-model", str(fixture / "nv"), "--nv-review-json", str(fixture / "review.json"),
            ]
            env = dict(os.environ, MAL2026_NVIDIA_SMI=str(nvidia), MAL2026_TORCHRUN=str(torchrun), MAL2026_PYTHON=sys.executable)
            try:
                initial = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=env)
                self.assertEqual(1, initial.returncode, initial.stderr)
                self.assertEqual("x", invoked.read_text(encoding="utf-8"))
                completion = selection / "standard_training_complete.json"
                completion_hash = hashlib.sha256(completion.read_bytes()).hexdigest()
                resumed = subprocess.run(command + ["--resume-run-prefix", prefix], cwd=ROOT, text=True, capture_output=True, env=env)
                self.assertEqual(1, resumed.returncode, resumed.stderr)
                self.assertEqual("x", invoked.read_text(encoding="utf-8"), "verified selection must not rerun")
                self.assertEqual(completion_hash, hashlib.sha256(completion.read_bytes()).hexdigest())
                entries = [json.loads(line) for line in (runtime / "matrix_ledger.jsonl").read_text().splitlines()]
                self.assertEqual(["started", "completed", "skipped_verified"], [entry["status"] for entry in entries if entry["step"] == "decoder-direct-selection"])
                self.assertEqual(["started", "failed", "started", "failed"], [entry["status"] for entry in entries if entry["step"] == "decoder-direct-dev-step-1"])
                lineage = [json.loads(line) for line in (runtime / "resume_lineage.jsonl").read_text().splitlines()]
                self.assertEqual("resume", lineage[-1]["event"])
                self.assertEqual(prefix, lineage[-1]["run_prefix"])
                self.assertFalse((ROOT / "outputs" / "standard-evals" / f"{prefix}-decoder-direct-dev-step-1").exists())
                changed_input = subprocess.run(
                    command + ["--resume-run-prefix", prefix, "--wandb-project", "different-project"],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    env=env,
                )
                self.assertEqual(1, changed_input.returncode)
                self.assertIn("immutable selection config disagrees", changed_input.stderr)
            finally:
                shutil.rmtree(runtime, ignore_errors=True)
                shutil.rmtree(selection, ignore_errors=True)

    def test_bounded_continuation_reuses_verified_parent_and_rejects_hash_mismatch(self) -> None:
        """A new continuation must preserve its failed parent and never launch on bad evidence."""
        import hashlib
        import os
        import shutil
        import uuid

        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            for name in ("qwen", "qwen3", "nv"):
                (fixture / name).mkdir()
            manifest_input = fixture / "manifest.json"
            manifest_input.write_text("{}", encoding="utf-8")
            review = fixture / "review.json"
            review.write_text(json.dumps({
                "model_id": "nvidia/NV-Embed-v2", "revision": "3fa59658547db50a1e8e3346cf057fd0c77ed6ef",
                "license_acknowledged": True, "use_case": "research_noncommercial", "reviewer": "test",
                "outcome": "approved", "reviewed_files": {"modeling_nvembed.py": "0" * 64},
            }), encoding="utf-8")
            parent_prefix = "continuation-parent-" + uuid.uuid4().hex
            parent = ROOT / "outputs" / "standard-runs" / f"{parent_prefix}-decoder-direct-selection"
            parent_runtime = ROOT / "outputs" / "experiment-matrix" / parent_prefix
            parent_config_path = parent_runtime / "configs" / "decoder-direct-selection.json"
            failure_step = "decoder-direct-dev-step-1"
            failure_log = parent_runtime / "logs" / f"{failure_step}.log"
            parent_config = {
                "run_id": parent.name, "phase": "selection", "mode": "direct",
                "model_path": str(fixture / "qwen"), "tokenizer_path": str(fixture / "qwen"),
                "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                "tokenizer_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                "prepared_manifest": str(manifest_input), "output_dir": str(parent), "seed": 2026,
                "max_length": 2048, "learning_rate": 2e-5, "num_train_epochs": 12.0,
                "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 16, "eval_steps": 100, "save_steps": 100,
                "logging_steps": 5, "early_stopping_patience": 4, "lora_r": 32, "lora_alpha": 64,
                "lora_dropout": 0.05, "selection_summary_path": None, "selected_global_step": None,
                "wandb_project": "parent-project", "wandb_entity": None,
            }
            parent_ledger_before = ""
            try:
                (parent_runtime / "configs").mkdir(parents=True)
                failure_log.parent.mkdir(parents=True)
                parent.mkdir(parents=True)
                parent_config_path.write_text(json.dumps(parent_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                (parent_runtime / "matrix_manifest.json").write_text(json.dumps({
                    "status": "started", "run_prefix": parent_prefix,
                    "git_sha": "dfa3c343b6fb58bf33835eacf730df1306ac0b5a",
                    "prepared_manifest": str(manifest_input), "validation_sha256": HASH,
                    "cuda_visible_devices": "0,1,2,3", "num_gpus": 4,
                    "selection_configs": {"decoder-direct-selection.json": hashlib.sha256(parent_config_path.read_bytes()).hexdigest()},
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                (parent / "standard_training_complete.json").write_text(json.dumps({
                    "status": "completed", "run_id": parent.name, "phase": "selection", "mode": "direct",
                    "global_step": 1, "best_metric": 0.2, "train_metrics": {"train_loss": 0.2},
                    "selection_candidate_steps": [1],
                }) + "\n", encoding="utf-8")
                for adapter in (parent / "adapter", parent / "checkpoint-1"):
                    adapter.mkdir(parents=True, exist_ok=True)
                    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
                    (adapter / "adapter_model.safetensors").write_bytes(b"test")
                failure_log.write_text("spawn import failure\n", encoding="utf-8")
                parent_ledger_before = json.dumps({"status": "failed", "step": failure_step, "detail": str(failure_log), "privacy": "aggregate_only"}) + "\n"
                (parent_runtime / "matrix_ledger.jsonl").write_text(parent_ledger_before, encoding="utf-8")
                nvidia_calls = fixture / "nvidia-calls"
                nvidia = fixture / "nvidia-smi"
                nvidia.write_text("\n".join([
                    "#!/bin/sh", f"printf x >> {nvidia_calls}", "case \"$1\" in",
                    "  --query-gpu=*) printf '0, UUID-0\\n1, UUID-1\\n2, UUID-2\\n3, UUID-3\\n' ;;",
                    "  --query-compute-apps=*) printf '99, UUID-0\\n' ;;", "esac", "",
                ]), encoding="utf-8")
                invoked = fixture / "torchrun-invoked"
                torchrun = fixture / "torchrun"
                torchrun.write_text(f"#!/bin/sh\nprintf invoked > {invoked}\nexit 99\n", encoding="utf-8")
                nvidia.chmod(0o755)
                torchrun.chmod(0o755)
                prefix = "continuation-new-" + uuid.uuid4().hex
                runtime = ROOT / "outputs" / "experiment-matrix" / prefix
                command = [
                    str(SCRIPT), "--continue-from-decoder-selection-run", str(parent),
                    "--runtime-root", str(runtime), "--run-prefix", prefix,
                    "--prepared-manifest", str(manifest_input), "--validation-sha256", HASH,
                    "--qwen-model", str(fixture / "qwen"), "--qwen3-model", str(fixture / "qwen3"),
                    "--nv-model", str(fixture / "nv"), "--nv-review-json", str(review),
                ]
                env = dict(os.environ, MAL2026_NVIDIA_SMI=str(nvidia), MAL2026_TORCHRUN=str(torchrun), MAL2026_PYTHON=sys.executable)
                continued = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=env)
                self.assertEqual(1, continued.returncode, continued.stderr)
                self.assertTrue((runtime / "matrix_manifest.json").is_file(), continued.stderr)
                self.assertFalse(invoked.exists(), "parent reuse must not relaunch the direct selection")
                new_manifest = json.loads((runtime / "matrix_manifest.json").read_text())
                self.assertEqual(3, new_manifest["schema_version"])
                self.assertEqual(str(parent), new_manifest["continuation"]["parent_selection_run"])
                self.assertEqual(failure_step, new_manifest["continuation"]["original_failed_vllm_stage"])
                self.assertEqual([1], new_manifest["continuation"]["parent_artifacts"]["candidate_checkpoints"])
                entries = [json.loads(line) for line in (runtime / "matrix_ledger.jsonl").read_text().splitlines()]
                self.assertEqual(["reused_verified_parent"], [entry["status"] for entry in entries if entry["step"] == "decoder-direct-selection"])
                self.assertEqual(parent_ledger_before, (parent_runtime / "matrix_ledger.jsonl").read_text())
                self.assertFalse((parent / "selected_checkpoint.json").exists())

                parent_config["seed"] = 7
                parent_config_path.write_text(json.dumps(parent_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                bad_prefix = "continuation-bad-" + uuid.uuid4().hex
                bad_runtime = ROOT / "outputs" / "experiment-matrix" / bad_prefix
                before_calls = nvidia_calls.read_text(encoding="utf-8")
                bad_command = list(command)
                bad_command[4] = str(bad_runtime)
                bad_command[6] = bad_prefix
                rejected = subprocess.run(bad_command, cwd=ROOT, text=True, capture_output=True, env=env)
                self.assertEqual(1, rejected.returncode)
                self.assertIn("config hash", rejected.stderr)
                self.assertEqual(before_calls, nvidia_calls.read_text(encoding="utf-8"), "bad parent evidence must fail before GPU preflight")
            finally:
                for path in (parent_runtime, parent, locals().get("runtime"), locals().get("bad_runtime")):
                    if path is not None:
                        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
