"""Static safeguards for the restricted OpenAI candidate workflow."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("generate_openai_rationales", ROOT / "scripts" / "generate_openai_rationales.py")
assert SPEC and SPEC.loader
GENERATOR = module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class OpenAIValidationIsolationTests(unittest.TestCase):
    def test_openai_smoke_rejects_validation_before_any_api_call(self) -> None:
        args = type("Args", (), {"model": GENERATOR.DEFAULT_MODEL, "split": "validation", "run_id": "synthetic", "candidate": 1})()
        with self.assertRaisesRegex(ValueError, "train-only"):
            GENERATOR.smoke(args)

    def test_openai_smoke_cli_has_no_validation_choice(self) -> None:
        with self.assertRaises(SystemExit):
            GENERATOR.parser().parse_args(["smoke", "--run-id", "synthetic", "--split", "validation"])


if __name__ == "__main__":
    unittest.main()
