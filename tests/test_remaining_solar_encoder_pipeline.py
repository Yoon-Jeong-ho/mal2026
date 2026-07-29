from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RemainingSolarEncoderPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "scripts/run_remaining_solar_encoder_pipeline.py"
        spec = importlib.util.spec_from_file_location("remaining_pipeline_contract", path)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def test_plan_covers_augmentation_both_encoders_and_final_fit(self) -> None:
        phases = self.module.phases()
        self.assertEqual(
            [phase.name for phase in phases],
            [
                "solar_augmentation", "augmented_bundle_rationales",
                "qwen_augmented_smoke", "qwen_augmented_full",
                "kure_augmented_smoke", "kure_augmented_full",
                "final_winner_smoke", "final_winner_full",
            ],
        )
        self.assertTrue(all(set(phase.gpus) <= {0, 1, 2, 3} for phase in phases))
        self.assertEqual(phases[-1].gpus, (0, 1, 2, 3))

    def test_no_phase_installs_or_pulls(self) -> None:
        command = "\n".join(" ".join(phase.command) for phase in self.module.phases())
        self.assertNotIn("docker pull", command)
        self.assertNotIn("pip install", command)
        self.assertNotIn("uv pip", command)


if __name__ == "__main__":
    unittest.main()
