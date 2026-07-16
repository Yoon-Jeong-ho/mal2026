from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from mal2026.human_feedback_data import (
    FEEDBACK_FIELDS,
    HumanFeedbackDataError,
    TARGET_TOKEN_CAP,
    derive_scores,
    discover_training_archives,
    parse_label_record,
    prepare_human_feedback_data,
    render_human_feedback_target,
    split_records,
    write_prepared_dataset,
)


class FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False):
        return {"input_ids": list(range(len(text)))}

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        assert not tokenize and not add_generation_prompt
        return "<user>" + messages[0]["content"] + "</user><assistant>" + messages[1]["content"] + "</assistant>"


def raw_record(question_id: str = "q1", answer_id: int = 1, prompt: str = "Question one", subject: str = "국어", base: int = 3) -> dict:
    analytic = {}
    for field in FEEDBACK_FIELDS[1:]:
        analytic[field] = {"score": [base, base], "feedback": f"{field} feedback"}
    return {
        "essay_question": {"id": question_id, "prompt": prompt, "subject": subject},
        "essay_answer": {"id": answer_id, "text": f"artificial answer {answer_id}"},
        "score": {"personal": {"holistic": {"score": [2, 2], "feedback": "holistic feedback"}, "analytic": analytic}},
    }


class HumanFeedbackDataTests(unittest.TestCase):
    def test_score_aggregation_and_exact_target_order(self):
        analytic = raw_record()["score"]["personal"]["analytic"]
        analytic["content_1"]["score"] = [1, 2]
        analytic["content_2"]["score"] = [2, 3]
        analytic["content_3"]["score"] = [3, 4]
        analytic["organization_1"]["score"] = [4, 5]
        analytic["organization_2"]["score"] = [4, 4]
        analytic["expression_1"]["score"] = [2, 2]
        analytic["expression_2"]["score"] = [3, 3]
        scores = derive_scores(analytic)
        self.assertEqual(Decimal("2.50"), scores["content"])
        self.assertEqual(Decimal("4.25"), scores["organization"])
        self.assertEqual(Decimal("2.50"), scores["expression"])
        self.assertEqual(Decimal("3.08"), scores["average"])
        record = parse_label_record(raw_record(), "descriptive")
        rendered = render_human_feedback_target(record)
        self.assertEqual(
            '{"feedback":{"holistic":"holistic feedback","content_1":"content_1 feedback","content_2":"content_2 feedback","content_3":"content_3 feedback","organization_1":"organization_1 feedback","organization_2":"organization_2 feedback","expression_1":"expression_1 feedback","expression_2":"expression_2 feedback","task_1":"task_1 feedback"},"scores":{"content":3.00,"organization":3.00,"expression":3.00,"average":3.00}}',
            rendered,
        )

    def test_invalid_rater_vector_and_feedback_fail_closed(self):
        invalid = raw_record()
        invalid["score"]["personal"]["analytic"]["content_1"]["score"] = [3]
        with self.assertRaises(HumanFeedbackDataError):
            parse_label_record(invalid, "descriptive")
        blank = raw_record()
        blank["score"]["personal"]["analytic"]["task_1"]["feedback"] = " "
        with self.assertRaises(HumanFeedbackDataError):
            parse_label_record(blank, "descriptive")

    def test_common_eligibility_split_and_safe_manifest(self):
        records = tuple(
            parse_label_record(raw_record(f"q{index}", index, f"Question {index}", "국어" if index < 3 else "과학"), "descriptive")
            for index in range(1, 5)
        )
        prepared = prepare_human_feedback_data((), FakeTokenizer(), expected_source_records=None) if False else None
        # DP selection is deterministic, group-disjoint, and ties by sorted hashes.
        train, dev, audit = split_records(records)
        self.assertTrue(train and dev)
        self.assertFalse({row.group_hash for row in train} & {row.group_hash for row in dev})
        self.assertEqual(4, audit["normalized_question_groups"])

        # Exercise the all-experiment common gate without a real archive.
        from mal2026 import human_feedback_data as module
        old = module.iter_training_records
        try:
            module.iter_training_records = lambda archives: iter(records)  # type: ignore[method-assign]
            prepared = prepare_human_feedback_data((), FakeTokenizer(), expected_source_records=None)
        finally:
            module.iter_training_records = old  # type: ignore[method-assign]
        self.assertEqual(4, prepared.manifest["eligibility"]["eligible_records"])
        self.assertEqual(TARGET_TOKEN_CAP, prepared.manifest["eligibility"]["assistant_target_token_cap"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "processed" / "fixture"
            manifest = root / "manifest.json"
            written = write_prepared_dataset(prepared, output, manifest)
            self.assertEqual(4, written["files"]["refit_train"]["record_count"])
            safe = manifest.read_text(encoding="utf-8")
            self.assertNotIn("Question 1", safe)
            self.assertNotIn("artificial answer", safe)
            self.assertNotIn("descriptive:q1:1", safe)
            self.assertTrue((output / "selection_train.jsonl").is_file())

    def test_discovery_rejects_validation_and_reads_only_tl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in ("025_descriptive_writing_evaluation", "026_argumentative_writing_evaluation"):
                path = root / dataset / "Training" / "02.라벨링데이터" / "TL_fixture.zip"
                path.parent.mkdir(parents=True)
                with ZipFile(path, "w") as zf:
                    zf.writestr("row.json", json.dumps(raw_record()))
                validation = root / dataset / "Validation" / "02.라벨링데이터" / "VL_fixture.zip"
                validation.parent.mkdir(parents=True)
                with ZipFile(validation, "w") as zf:
                    zf.writestr("row.json", json.dumps(raw_record()))
            found = discover_training_archives(root)
            self.assertEqual(2, len(found))
            self.assertTrue(all("Training" in item.relative_path and "Validation" not in item.relative_path for item in found))


if __name__ == "__main__":
    unittest.main()
