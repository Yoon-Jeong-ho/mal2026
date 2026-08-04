from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from humaneval.core import (
    AXES,
    ALLOWED_USERS,
    HumanValidationConflict,
    ResponseStore,
    AXIS_BAND_TARGET,
    axis_band_counts,
    build_study,
    load_rationales,
)


NOTICE = "[유의 사항] △ 서론, 본론, 결론을 갖춘 글을 쓸 것."


def evaluation_row(identifier: str, band: int) -> dict:
    return {
        "id": identifier,
        "document_id": f"doc-{identifier}",
        "prompt_num": "Q1",
        "prompt": f"주제 {identifier}에 관해 쓰시오. {NOTICE}",
        "essay": f"학생 글 {identifier} " + ("문장입니다. " * 10),
        "score": {"content": float(band), "organization": float(band), "expression": float(band), "average": float(band)},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class HumanValidationTests(unittest.TestCase):
    def fixture(self, root: Path):
        train, validation = root / "train.jsonl", root / "validation.jsonl"
        rows = [evaluation_row(f"row-{band}-{index}", band) for band in range(1, 6) for index in range(10)]
        write_jsonl(train, rows[:35])
        write_jsonl(validation, rows[35:])
        rationale_rows = [
            {"source_id": row["id"], "rationales": {axis: f"{axis} 모델 설명 {row['id']}" for axis in AXES}}
            for row in rows
        ]
        model = root / "model.jsonl"
        write_jsonl(model, rationale_rows)
        api_rows = []
        for row in rows:
            for candidate in (1, 2):
                api_rows.append({
                    "source_id": row["id"], "candidate": candidate,
                    "rationale": {**{axis: f"{axis} API 설명 {candidate} {row['id']}" for axis in AXES}, "schema_version": "fixture"},
                })
        api = root / "api.jsonl"
        write_jsonl(api, api_rows)
        rubric = root / "evaluation.txt"
        rubric.write_text("""[평가 기준 정의]
1. content
- 내용 기준 하나

2. organization
- 구성 기준 하나

3. expression
- 표현 기준 하나

[점수 기준]
- 5점: 매우 우수함.
- 4점: 우수함.
- 3점: 보통.
- 2점: 미흡함.
- 1점: 매우 미흡함.
""", encoding="utf-8")
        judge_guide = root / "llm_as_judge.txt"
        judge_guide.write_text("""domain_match
score_rationale_consistency
specificity
groundedness
generic한 총평
essay_text에 없는 내용
""", encoding="utf-8")
        return train, validation, api, model, rubric, judge_guide

    def test_build_is_deterministic_and_separates_notice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, validation, api, model, rubric, judge_guide = self.fixture(root)
            kwargs = {
                "split_paths": {"train": train, "validation": validation},
                "rubric_path": rubric,
                "judge_guide_path": judge_guide,
                "api_rationale_paths": [api], "model_rationale_paths": [model],
                "seed": 17,
            }
            first, second = build_study(**kwargs), build_study(**kwargs)
            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertEqual([item.source_id for item in first.items], [item.source_id for item in second.items])
            self.assertEqual({axis: AXIS_BAND_TARGET for axis in AXES}, axis_band_counts(first.items))
            self.assertEqual(NOTICE, first.common_notice)
            self.assertTrue(all(NOTICE not in item.topic_prompt for item in first.items))
            self.assertTrue(all("API 설명 1" in item.api_rationale["content"] for item in first.items))

    def test_rationale_loader_rejects_duplicate_after_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api.jsonl"
            bundle = {
                axis: {
                    "evidence_sentence_ids": [1],
                    "diagnosis": f"{axis} 진단",
                    "next_step": f"{axis} 숨겨야 할 개선 제안",
                }
                for axis in AXES
            }
            write_jsonl(path, [
                {"source_id": "x", "candidate": 1, "rationale": bundle},
                {"source_id": "x", "candidate": 2, "rationale": bundle},
            ])
            loaded = load_rationales([path], preferred_candidate=1)
            self.assertEqual(["x"], list(loaded))
            self.assertEqual("content 진단", loaded["x"]["content"])
            self.assertNotIn("개선 제안", json.dumps(loaded, ensure_ascii=False))

    def test_per_user_resume_and_blind_sequential_rationales(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, validation, api, model, rubric, judge_guide = self.fixture(root)
            study = build_study(
                split_paths={"train": train, "validation": validation}, rubric_path=rubric,
                judge_guide_path=judge_guide,
                api_rationale_paths=[api], model_rationale_paths=[model], seed=9,
            )
            database = root / "responses.sqlite3"
            store = ResponseStore(database, study)
            user = ALLOWED_USERS[0]
            initial = store.state(user)
            self.assertEqual("score", initial["phase"])
            self.assertNotIn("target_band", json.dumps(initial))
            reasons = {axis: f"{axis}에서 1점보다 근거가 있음" for axis in AXES}
            scored = store.record_scores(user, 0, {axis: 2 for axis in AXES}, reasons)
            self.assertEqual("rationale_a", scored["phase"])
            self.assertEqual({"label", "texts"}, set(scored["rationale"]))
            resumed = ResponseStore(database, study).state(user)
            self.assertEqual("rationale_a", resumed["phase"])
            verdicts_a = {axis: "partial" for axis in AXES}
            rationale_reasons_a = {axis: f"{axis} 일부만 타당" for axis in AXES}
            after_a = store.record_rationale(user, 0, verdicts_a, rationale_reasons_a)
            self.assertEqual("rationale_b", after_a["phase"])
            verdicts_b = {axis: "appropriate" for axis in AXES}
            rationale_reasons_b = {axis: f"{axis} 타당" for axis in AXES}
            next_item = store.record_rationale(user, 0, verdicts_b, rationale_reasons_b)
            self.assertEqual("score", next_item["phase"])
            self.assertEqual(1, next_item["progress"]["completed"])
            self.assertEqual("score", store.state(ALLOWED_USERS[1])["phase"])
            with self.assertRaises(HumanValidationConflict):
                store.record_scores(user, 0, {axis: 3 for axis in AXES}, reasons)
            exported = root / "responses.jsonl"
            self.assertEqual(1, store.export_jsonl(exported))
            exported_text = exported.read_text(encoding="utf-8")
            self.assertNotIn("학생 글", exported_text)
            self.assertNotIn("API 설명", exported_text)
            self.assertIn("content_reason", exported_text)


if __name__ == "__main__":
    unittest.main()
