from pathlib import Path
import json
import tempfile
import unittest

from mal2026.official_rationale_handoff import (
    ALLOWED_HISTORICAL_CONTINUATIONS,
    HISTORICAL_CLASSIFICATION,
    HISTORICAL_CONTRACT_SHIFT,
    HISTORICAL_RANKING_CAVEAT,
    HandoffConfig,
    candidate_identity_sha256,
    combine_rationales,
    convert_bootstrap_scores,
    file_sha256,
    select_candidate,
    validate_training_completion,
)


class OfficialRationaleHandoffTests(unittest.TestCase):
    def test_config_declares_all_required_methods_and_fails_closed(self) -> None:
        config = HandoffConfig.from_json(Path("configs/official_rationale_handoff.v1.json"))
        self.assertEqual({candidate["method"] for candidate in config.candidates}, {"official_sft", "aihub_sft", "dpo", "grpo"})
        self.assertEqual(sum(candidate["method"] == "dpo" for candidate in config.candidates), 3)
        self.assertEqual(sum(candidate["method"] == "grpo" for candidate in config.candidates), 3)
        self.assertEqual(
            {candidate["key"] for candidate in config.candidates if candidate["method"] in {"dpo", "grpo"}},
            {key for _, key in ALLOWED_HISTORICAL_CONTINUATIONS},
        )
        self.assertTrue(all(candidate["final_winner_eligible"] for candidate in config.candidates))
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

    def test_ineligible_candidate_cannot_win(self) -> None:
        rows = [
            {"key": "descriptive", "macro_mean": 5.0, "worst_cell": 5.0, "strict_parse_rate": 1.0, "final_winner_eligible": False},
            {"key": "eligible", "macro_mean": 4.0, "worst_cell": 4.0, "strict_parse_rate": 1.0, "final_winner_eligible": True},
        ]
        self.assertEqual(select_candidate(rows)["key"], "eligible")

    def test_named_historical_continuation_is_allowed_and_sha_bound(self) -> None:
        method, key = "dpo", "dpo_historical_midm_random1_bundle"
        source = ALLOWED_HISTORICAL_CONTINUATIONS[(method, key)]
        candidate = {
            "key": key, "method": method,
            "origin_classification": "public_spec_score_conditioned_historical_method_continuation",
            "historical_method": source["legacy_arm"], "historical_source_sha256": source["source_completion_sha256"], "final_winner_eligible": True,
            "ranking_caveat": HISTORICAL_RANKING_CAVEAT,
        }
        completion = {
            "schema_version": "mal2026-official-rationale-dpo-complete-v1", "status": "completed",
            "run_id": "official-rationale-dpo-historical-full-001", "task": "bundle", "split": "train",
            "legacy_arm": source["legacy_arm"], "classification": HISTORICAL_CLASSIFICATION,
            "contract_shift": HISTORICAL_CONTRACT_SHIFT, "legacy_completion_sha256": source["source_completion_sha256"],
            "human_or_reference_score_read_or_prompted": False,
        }
        validate_training_completion(candidate, "bundle", completion)
        completion["legacy_completion_sha256"] = "0" * 64
        with self.assertRaisesRegex(Exception, "source SHA differs"):
            validate_training_completion(candidate, "bundle", completion)

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
