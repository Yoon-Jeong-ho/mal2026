from pathlib import Path
import json
import tempfile
import unittest

from mal2026.official_rationale_candidate_evaluation import (
    REPEAT_INTERPRETATION,
    aggregate_deterministic_repeats,
    compose_participants,
    resolve_handoff,
)
from mal2026.official_rationale_handoff import AXES, file_sha256, validate_training_completion
from mal2026.official_writing_contract import JUDGE_DIMENSIONS


ROOT = Path(__file__).resolve().parents[1]
RESTRICTED = ROOT / "data/processed/restricted"


class OfficialRationaleCandidateEvaluationTests(unittest.TestCase):
    def test_smoke_adapter_cannot_enter_final_handoff(self) -> None:
        candidate = {"key": "official_sft_bundle", "method": "official_sft"}
        completion = {
            "schema_version": "mal2026-official-rationale-sft-complete-v1", "status": "completed",
            "run_id": "official-rationale-sft-gpu0-smoke-001", "phase": "gpu0_smoke", "task": "bundle",
            "candidate_provenance": {"human_or_reference_score_read_or_prompted": False},
        }
        with self.assertRaisesRegex(Exception, "non-full candidate"):
            validate_training_completion(candidate, "bundle", completion)

    def test_resolver_hash_binds_completed_full_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bootstrap = root / "bootstrap.json"; directional = root / "directional.json"; injection = root / "injection.json"
            for path in (bootstrap, directional, injection): path.write_text("{}\n")
            model = root / "model"; model.mkdir(); (model / "config.json").write_text("{}\n")
            binding = root / "binding.json"; binding.write_text("{}\n")
            run = root / "full-run"; adapter = run / "adapter"; adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text('{"r":32}\n'); (adapter / "adapter_model.safetensors").write_bytes(b"state")
            completion = run / "training_complete.json"
            completion.write_text(json.dumps({
                "schema_version": "mal2026-official-rationale-sft-complete-v1", "status": "completed",
                "run_id": "official-full", "phase": "full", "task": "bundle", "model_id": "model/id",
                "model_revision": "revision", "candidate_provenance": {"human_or_reference_score_read_or_prompted": False},
            }) + "\n")
            template = {
                "bootstrap_selection_path": str(bootstrap), "bootstrap_selection_sha256": "REQUIRED_SHA256",
                "judge": {"directional_gate_path": str(directional), "directional_gate_sha256": "REQUIRED_SHA256", "injection_gate_path": str(injection), "injection_gate_sha256": "REQUIRED_SHA256"},
                "candidates": [{
                    "key": "one", "method": "official_sft", "structure": "bundle", "model_id": "model/id", "model_revision": "revision",
                    "model_path": "REQUIRED", "model_config_sha256": "REQUIRED", "model_binding_path": "REQUIRED", "model_binding_sha256": "REQUIRED",
                    "adapters": {"bundle": {}}, "evaluation_path": "REQUIRED", "evaluation_sha256": "REQUIRED",
                }],
            }
            bindings = {"schema_version": "mal2026-official-rationale-candidate-bindings-v1", "candidates": {"one": {
                "model_path": str(model), "model_binding_path": str(binding), "adapters": {"bundle": str(completion)}, "evaluation_path": str(root / "evaluation.json"),
            }}}
            resolved = resolve_handoff(template, bindings, require_evaluations=False)
            candidate = resolved["candidates"][0]
            self.assertEqual(resolved["bootstrap_selection_sha256"], file_sha256(bootstrap))
            self.assertEqual(candidate["adapters"]["bundle"]["training_completion_sha256"], file_sha256(completion))
            self.assertEqual(candidate["evaluation_sha256"], "PENDING_EVALUATION_SHA256")

    def test_compose_copies_emitted_scores_without_mismatch(self) -> None:
        RESTRICTED.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RESTRICTED) as directory:
            root = Path(directory); scores = root / "scores.jsonl"; rationales = root / "rationales.jsonl"; output = root / "participants.jsonl"
            emitted = {"content": 1, "organization": 3, "expression": 5}
            scores.write_text(json.dumps({"source_id": "x", "emitted_integer_prediction": emitted}) + "\n")
            rationales.write_text(json.dumps({"source_id": "x", "rationales": {"content": "내용 근거", "organization": "구성 근거", "expression": "표현 근거"}}, ensure_ascii=False) + "\n")
            digest = compose_participants(scores, rationales, output, 1)
            value = json.loads(output.read_text())
            self.assertEqual(digest, file_sha256(output))
            self.assertEqual({axis: value["participant_output"][axis]["score"] for axis in AXES}, emitted)
            self.assertNotIn("average", value["participant_output"])

    def test_ten_fixed_repeats_report_agreement_not_independence(self) -> None:
        RESTRICTED.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RESTRICTED) as directory:
            root = Path(directory); pairs = []
            judge_output = {
                axis: {dimension: {"evidence": "근거", "score": 4} for dimension in JUDGE_DIMENSIONS}
                for axis in AXES
            }
            for repeat in range(10):
                report = root / f"report-{repeat}.json"; records = root / f"records-{repeat}.jsonl"
                records.write_text(json.dumps({"source_id": "x", "judge_output": judge_output, "failure_category": None, "attempts": 1}, ensure_ascii=False) + "\n")
                report.write_text(json.dumps({
                    "schema_version": "mal2026-official-q4-judge-aggregate-v1", "status": "completed",
                    "temperature": 0.0, "seed": 42, "human_or_reference_score_read_or_prompted": False,
                    "counts": {"expected": 1, "records": 1, "valid": 1}, "macro_mean": 4.0,
                    "worst_cell_mean": 4.0, "judge_records_sha256": file_sha256(records),
                }) + "\n")
                pairs.append((report, records))
            result = aggregate_deterministic_repeats(pairs, expected=1, repeats=10)
            self.assertEqual(result["metrics"], {"macro_mean": 4.0, "worst_cell": 4.0, "strict_parse_rate": 1.0})
            self.assertEqual(result["repeat_diagnostics"]["exact_agreement_rate"], 1.0)
            self.assertEqual(result["repeat_diagnostics"]["mean_population_variance"], 0.0)
            self.assertTrue(result["repeat_diagnostics"]["zero_variance_is_not_independence_evidence"])
            self.assertEqual(result["repeat_diagnostics"]["interpretation"], REPEAT_INTERPRETATION)


if __name__ == "__main__":
    unittest.main()
