from pathlib import Path
import json
import tempfile
import unittest

from mal2026.official_rationale_handoff import (
    HandoffConfig,
    candidate_identity_sha256,
    combine_rationales,
    convert_bootstrap_scores,
    file_sha256,
    select_candidate,
)


class OfficialRationaleHandoffTests(unittest.TestCase):
    def test_config_declares_all_required_methods_and_fails_closed(self) -> None:
        config = HandoffConfig.from_json(Path("configs/official_rationale_handoff.v1.json"))
        self.assertEqual({candidate["method"] for candidate in config.candidates}, {"official_sft", "aihub_sft", "dpo", "grpo"})
        self.assertEqual(sum(candidate["method"] == "dpo" for candidate in config.candidates), 3)
        self.assertEqual(sum(candidate["method"] == "grpo" for candidate in config.candidates), 3)
        self.assertEqual({candidate["structure"] for candidate in config.candidates}, {"bundle", "axis_triplet"})
        self.assertEqual(len({candidate_identity_sha256(candidate) for candidate in config.candidates}), len(config.candidates))
        with self.assertRaisesRegex(Exception, "bootstrap selection SHA differs"):
            config.validate_dependencies()

    def test_fixed_q4_candidate_ranking(self) -> None:
        rows = [
            {"key": "a", "macro_mean": 4.8, "worst_cell": 4.6, "strict_parse_rate": 1.0},
            {"key": "b", "macro_mean": 4.9, "worst_cell": 4.0, "strict_parse_rate": 1.0},
            {"key": "c", "macro_mean": 4.9, "worst_cell": 4.1, "strict_parse_rate": 0.9},
            {"key": "d", "macro_mean": 4.9, "worst_cell": 4.1, "strict_parse_rate": 1.0},
        ]
        self.assertEqual(select_candidate(rows)["key"], "d")

    def test_bootstrap_score_conversion_reads_only_emitted_integers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.jsonl"; output = root / "output.jsonl"
            source.write_text(json.dumps({"source_id": "x", "split": "train", "arm": "bootstrap", "scores": {"content": 1, "organization": 3, "expression": 5}}) + "\n")
            digest = convert_bootstrap_scores(source, output, 1, "train")
            value = json.loads(output.read_text())
            self.assertEqual(digest, file_sha256(output))
            self.assertEqual(value, {"source_id": "x", "emitted_integer_prediction": {"content": 1, "organization": 3, "expression": 5}})
            self.assertNotIn("average", value["emitted_integer_prediction"])

    def test_bundle_and_exact_axis_combination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle.jsonl"
            rationales = {"content": "c", "organization": "o", "expression": "e"}
            bundle.write_text(json.dumps({"source_id": "x", "rationales": rationales}) + "\n")
            bundle_output = root / "bundle-out.jsonl"
            combine_rationales({"bundle": bundle}, bundle_output, 1, "bundle")
            self.assertEqual(json.loads(bundle_output.read_text())["rationales"], rationales)
            axes = {}
            for axis in rationales:
                path = root / f"{axis}.jsonl"; path.write_text(json.dumps({"source_id": "x", "rationales": {axis: rationales[axis]}}) + "\n"); axes[axis] = path
            axis_output = root / "axis-out.jsonl"
            combine_rationales(axes, axis_output, 1, "axis_triplet")
            self.assertEqual(json.loads(axis_output.read_text())["rationales"], rationales)


if __name__ == "__main__":
    unittest.main()
