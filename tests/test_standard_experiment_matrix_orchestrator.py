from __future__ import annotations

from pathlib import Path
import subprocess
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
            self.assertEqual(2, selected_busy.returncode)
            self.assertIn("selected GPU UUID UUID-0 has active compute PID 100", selected_busy.stderr)
            self.assertFalse(runtime.exists())


if __name__ == "__main__":
    unittest.main()
