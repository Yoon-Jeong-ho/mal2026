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


if __name__ == "__main__":
    unittest.main()
