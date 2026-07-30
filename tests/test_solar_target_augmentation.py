from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

from mal2026.solar_target_augmentation import (
    AXES,
    TARGET_SCORES,
    AugmentationTask,
    SolarTargetAugmentationError,
    SourceRow,
    build_tasks,
    editor_output_schema,
    make_task,
    parse_editor_output,
    parse_fidelity_output,
    parse_verifier_output,
    prompt_config,
    render_editor_messages,
    render_fidelity_messages,
    render_verifier_messages,
    select_smoke_sources,
    source_row_from_mapping,
    task_count,
    validate_candidate,
    validate_actual_label_candidate,
    validate_train_validation_disjoint,
)
from mal2026.evaluation_prompt_matrix import evaluation_sections


ROOT = Path(__file__).resolve().parents[1]


def row(index: int) -> SourceRow:
    return SourceRow(
        identifier=f"id-{index:04d}",
        document_id=f"doc-{index:04d}",
        prompt=f"논제 {index}",
        essay=(f"이 글은 논제 {index}에 찬성한다. 근거를 설명하고 결론을 제시한다. " * 3).strip(),
        score=(3.5, 4.0, 3.0),
    )


class NoDerivedScoreAccess(dict):
    def get(self, key, default=None):  # type: ignore[no-untyped-def]
        if key not in AXES:
            raise AssertionError(f"forbidden score access: {key}")
        return super().get(key, default)


class SolarTargetAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = [row(index) for index in range(2000)]

    def test_exact_rubric_checksum_and_v2_config_binding(self) -> None:
        config = prompt_config()
        expected = "1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145"
        self.assertEqual(config["provenance"]["rubric_source_sha256"], expected)
        self.assertEqual(config["verifier_contract"]["rubric_source_sha256"], expected)

    def test_matrix_is_30000_and_fifteen_per_canonical_source(self) -> None:
        tasks = build_tasks(self.rows)
        self.assertEqual(task_count(self.rows), 30000)
        self.assertEqual(len(tasks), 30000)
        self.assertEqual(len([task for task in tasks if task.source.identifier == "id-0000"]), 15)
        self.assertEqual({(task.target_axis, task.target_score) for task in tasks[:15]},
                         set((axis, score) for axis in AXES for score in TARGET_SCORES))
        self.assertEqual(tasks[0].task_id, "id-0000::solar-target::content::1")

    def test_every_variant_renders_from_immutable_canonical_source(self) -> None:
        source = row(1)
        for axis in AXES:
            for score in TARGET_SCORES:
                task = make_task(source, axis, score)
                payload = render_editor_messages(task)[1]["content"]
                self.assertIn(source.prompt, payload)
                for sentence in (item.strip() for item in source.essay.split(".") if item.strip()):
                    self.assertIn(sentence, payload)
                self.assertNotIn('"essay_text"', payload)
                self.assertNotIn("previous_augmented", payload)
        with self.assertRaises((AttributeError, TypeError)):
            source.essay = "mutated"  # type: ignore[misc]

    def test_editor_payload_has_three_axis_baseline_and_no_derived_score(self) -> None:
        payload = render_editor_messages(make_task(row(2), "organization", 1))[1]["content"]
        self.assertIn('"baseline_score":{"content":3.5,"organization":4.0,"expression":3.0}', payload)
        self.assertNotIn('"average"', payload)

    def test_candidate_families_are_independent_and_never_include_judge_feedback(self) -> None:
        task = make_task(row(2), "organization", 1)
        config = prompt_config()
        rendered = []
        for family in range(4):
            messages = render_editor_messages(task, family)
            self.assertEqual(len(messages), 2)
            payload = messages[-1]["content"]
            self.assertIn(config["axis_operation_families"]["organization"][family], payload)
            self.assertNotIn("previous_blind_scores", payload)
            self.assertNotIn("target_blind_feedback", payload)
            self.assertNotIn("target_deficiency", payload)
            self.assertNotIn("previous_augmented", payload)
            rendered.append(payload)
        self.assertEqual(len(set(rendered)), 4)
        with self.assertRaises(SolarTargetAugmentationError):
            render_editor_messages(task, 4)

    def test_editor_renders_exact_axis_score_rubric_and_target_boundary(self) -> None:
        config = prompt_config()
        for axis in AXES:
            payload = render_editor_messages(make_task(row(20), axis, 3))[1]["content"]
            for rubric_text in config["rubric"].values():
                self.assertIn(rubric_text, payload)
            self.assertIn(config["axis_edit_boundaries"][axis]["allowed"], payload)
            self.assertIn(config["axis_edit_boundaries"][axis]["forbidden"], payload)
            self.assertIn(config["axis_target_recipes"][axis]["3"], payload)
            self.assertIn('"target_axis_typed_edit_contract"', payload)
            self.assertIn('"output_length_chars"', payload)
            self.assertIn("[현재 작업의 필수 편집]", payload)
            self.assertIn('"target_score":3', payload)

    def test_editor_and_parser_use_the_same_axis_score_length_contract(self) -> None:
        config = prompt_config()
        source = row(22)
        for axis in AXES:
            for score in TARGET_SCORES:
                messages = render_editor_messages(make_task(source, axis, score))
                payload = json.loads(messages[1]["content"].splitlines()[1])["canonical_source"]
                lower, upper = config["quality_gates"]["prompt_length_ratio_by_axis"][axis]
                lower, upper = config["quality_gates"].get(
                    "prompt_length_ratio_by_axis_score", {}
                ).get(f"{axis}:{score}", [lower, upper])
                length = len(source.essay.strip())
                self.assertEqual(payload["output_length_chars"]["minimum"],
                                 max(20, math.ceil(length * lower)))
                self.assertEqual(payload["output_length_chars"]["maximum"],
                                 math.floor(length * upper))

    def test_axis_typed_edit_gates_reject_cross_axis_rewrites(self) -> None:
        source = SourceRow(
            identifier="typed-source",
            document_id="typed-document",
            prompt="논제",
            essay="첫 번째 주장을 제시한다. 두 번째 근거를 설명한다. 마지막 결론을 제시한다.",
            score=(3.0, 3.0, 3.0),
        )
        reordered = "마지막 결론을 제시한다. 첫 번째 주장을 제시한다. 두 번째 근거를 설명한다."
        plan = {"sentence_order": [2, 0, 1], "paragraph_break_after": [],
                "connector_actions": []}
        self.assertEqual(parse_editor_output(json.dumps(plan), source, "organization"), reordered)
        invalid_plan = {"sentence_order": [2, 0, 0], "paragraph_break_after": [],
                        "connector_actions": []}
        with self.assertRaises(SolarTargetAugmentationError):
            parse_editor_output(json.dumps(invalid_plan), source, "organization", 1)
        fixed_order_plan = {
            "paragraph_break_after": [0],
            "connector_actions": [{"position": 1, "connector": "따라서"}],
        }
        fixed_order_essay = parse_editor_output(
            json.dumps(fixed_order_plan), source, "organization", 4
        )
        self.assertLess(fixed_order_essay.index("첫 번째"), fixed_order_essay.index("두 번째"))
        self.assertLess(fixed_order_essay.index("두 번째"), fixed_order_essay.index("마지막"))
        added_sentence = {"sentence_actions": [
            {"apply": True, "replacement": "XXXXXXXXXXXXXXXXXXXXXXXX."},
            {"apply": True, "replacement": "YYYYYYYYYYYYYYYYYYYYYYYY."},
            {"apply": True, "replacement": "ZZZZZZZZZZZZZZZZZZZZZZZZ."},
        ]}
        with self.assertRaises(SolarTargetAugmentationError):
            parse_editor_output(json.dumps(added_sentence), source, "content")

        numbered = SourceRow(
            identifier="numbered-source",
            document_id="numbered-document",
            prompt="논제",
            essay="2026년에 첫 근거를 말한다. 두 번째 근거도 분명하게 설명한다.",
            score=(3.0, 3.0, 3.0),
        )
        changed_number = {"sentence_actions": [
            {"apply": True, "replacement": "2027년에 첫 근거를 말한다."},
            {"apply": False, "replacement": ""},
        ]}
        with self.assertRaises(SolarTargetAugmentationError):
            parse_editor_output(json.dumps(changed_number), numbered, "expression")

    def test_actual_label_protocol_keeps_blind_triplet_not_requested_target(self) -> None:
        source = SourceRow(
            identifier="actual-source", document_id="actual-document", prompt="논제",
            essay="입장을 제시한다. 근거를 구체적으로 설명한다. 결론을 분명히 정리한다.",
            score=(3.0, 3.0, 3.0),
        )
        task = make_task(source, "content", 1)
        output = {"sentence_actions": [
            {"apply": True, "replacement": "입장은 필요하다는 점을 제시한다."},
            {"apply": True, "replacement": "근거는 필요하기 때문에 필요하다고 설명한다."},
            {"apply": False, "replacement": ""},
        ]}
        essay = parse_editor_output(json.dumps(output), source, "content", 1)
        verifier = {
            "content": {"score": 2, "rationale": "내용 근거가 약하다."},
            "organization": {"score": 2, "rationale": "전개도 일부 약해졌다."},
            "expression": {"score": 3, "rationale": "표현은 대체로 자연스럽다."},
        }
        fidelity = {
            "source_based": True, "topic": True, "stance": True, "genre": True,
            "new_external_facts_added": False,
        }
        record = validate_actual_label_candidate(task, essay, verifier, fidelity)
        self.assertEqual(record["score"], {"content": 2, "organization": 2, "expression": 3})
        self.assertEqual(record["requested_target_score"], 1)
        self.assertNotEqual(record["score"]["content"], record["requested_target_score"])
        self.assertEqual(record["score_provenance"], "target_blind_solar_actual_triplet")
        fidelity["new_external_facts_added"] = True
        with self.assertRaises(SolarTargetAugmentationError):
            validate_actual_label_candidate(task, essay, verifier, fidelity)

    def test_actual_label_edit_count_can_ignore_requested_score_fraction(self) -> None:
        source = SourceRow(
            identifier="actual-edit-count-source",
            document_id="actual-edit-count-document",
            prompt="논제",
            essay=(
                "입장을 제시한다. 첫 번째 근거를 설명한다. 두 번째 근거를 설명한다. "
                "세 번째 근거를 설명한다. 결론을 정리한다."
            ),
            score=(3.0, 3.0, 3.0),
        )
        output = {"sentence_actions": [
            {"apply": True, "replacement": "입장은 필요하다고 제시한다."},
            {"apply": False, "replacement": ""},
            {"apply": False, "replacement": ""},
            {"apply": False, "replacement": ""},
            {"apply": False, "replacement": ""},
        ]}
        raw = json.dumps(output, ensure_ascii=False)
        with self.assertRaisesRegex(
            SolarTargetAugmentationError, "substantive sentence edit count"
        ):
            parse_editor_output(raw, source, "content", 1)
        essay = parse_editor_output(
            raw,
            source,
            "content",
            1,
            enforce_score_specific_edit_count=False,
        )
        self.assertIn("입장은 필요하다고", essay)

        no_op = {"sentence_actions": [
            {"apply": False, "replacement": ""} for _ in range(5)
        ]}
        with self.assertRaisesRegex(
            SolarTargetAugmentationError, "substantive sentence edit count"
        ):
            parse_editor_output(
                json.dumps(no_op),
                source,
                "content",
                1,
                enforce_score_specific_edit_count=False,
            )

    def test_content5_uses_source_grounded_evidence_ledger(self) -> None:
        source = SourceRow(
            identifier="content5-source", document_id="content5-document", prompt="논제",
            essay=("입장을 분명히 제시하고 주제를 직접 설명한다.\n\n"
                   "근거를 구체적으로 말하면서 그 이유를 함께 밝히고 마지막에는 결론을 정리한다."),
            score=(3.0, 3.0, 3.0),
        )
        output = {"evidence_additions": [{
            "source_sentence_index": 1,
            "addition_type": "causal_bridge",
            "addition_text": "그 결과 주장의 필요성이 더 분명해진다.",
        }]}
        essay = parse_editor_output(json.dumps(output), source, "content", 5)
        self.assertEqual(len(essay.split("\n\n")), 2)
        self.assertTrue(essay.startswith("입장을 분명히 제시하고"))
        bad = {"evidence_additions": [
            output["evidence_additions"][0], output["evidence_additions"][0]
        ]}
        with self.assertRaises(SolarTargetAugmentationError):
            parse_editor_output(json.dumps(bad), source, "content", 5)

        too_short = {"evidence_additions": [{
            "source_sentence_index": 1,
            "addition_type": "causal_bridge",
            "addition_text": "그러하다.",
        }]}
        with self.assertRaisesRegex(SolarTargetAugmentationError, "length ratio"):
            parse_editor_output(json.dumps(too_short), source, "content", 5)

        too_long = {"evidence_additions": [{
            "source_sentence_index": 1,
            "addition_type": "causal_bridge",
            "addition_text": (
                "이 근거가 주장을 뒷받침하는 과정과 그 결과를 같은 의미로 "
                "매우 길고 자세하게 거듭 설명하면서 결론과의 연결까지 반복하여 밝힌다."
            ),
        }]}
        with self.assertRaisesRegex(SolarTargetAugmentationError, "length ratio"):
            parse_editor_output(json.dumps(too_long), source, "content", 5)

    def test_expression5_uses_positioned_sentence_actions(self) -> None:
        source = SourceRow(
            identifier="diagnostic-source", document_id="diagnostic-document", prompt="논제",
            essay="이 문장은 자연스럽게 쓰여 있다. 다음 문장도 근거를 분명하게 설명한다.",
            score=(3.0, 3.0, 3.0),
        )
        output = {"sentence_actions": [
            {"apply": True, "replacement": "이 문장은 자연스럽게 작성되어 있다."},
            {"apply": False, "replacement": ""},
        ]}
        essay = parse_editor_output(json.dumps(output), source, "expression", 5)
        self.assertIn("작성되어", essay)
        invalid = {"sentence_actions": [
            {"apply": False, "replacement": ""},
            {"apply": False, "replacement": ""},
        ]}
        with self.assertRaises(SolarTargetAugmentationError):
            parse_editor_output(json.dumps(invalid), source, "expression", 5)

        task = make_task(source, "expression", 5)
        schema = editor_output_schema(task)
        self.assertEqual(schema["required"], ["sentence_actions"])
        self.assertNotIn("sentence_diagnostics", schema["properties"])
        payload = json.loads(render_editor_messages(task)[1]["content"].splitlines()[1])[
            "canonical_source"
        ]
        self.assertEqual(
            payload["required_editor_output"],
            prompt_config()["editor_output_contracts"]["content_expression"],
        )

    def test_expression_spacing_edits_are_real_but_axis_length_is_enforced(self) -> None:
        source = SourceRow(
            identifier="spacing-source", document_id="spacing-document", prompt="논제",
            essay="이 문장은 충분히 자연스럽게 쓰여 있다. 다음 문장도 내용을 분명하게 설명한다.",
            score=(3.0, 3.0, 3.0),
        )
        modest = {"sentence_actions": [
            {"apply": True, "replacement": "이 문장은 충분히 자연스럽게쓰여 있다."},
            {"apply": True, "replacement": "다음 문장도 내용을 분명하게설명한다."},
        ]}
        self.assertNotEqual(parse_editor_output(json.dumps(modest), source, "expression", 1), source.essay)
        severe = {"sentence_actions": [
            {"apply": True, "replacement": "이문장은충분히자연스럽게쓰여있다."},
            {"apply": True, "replacement": "다음문장도내용을분명하게설명한다."},
        ]}
        with self.assertRaises(SolarTargetAugmentationError):
            parse_editor_output(json.dumps(severe), source, "expression", 1)

    def test_new_evaluation_metadata_is_rejected(self) -> None:
        source = row(21)
        leaked = {"sentence_actions": [
            {"apply": True, "replacement": "이 글은 논제 21에 찬성하며 목표 점수는 1이다."},
            {"apply": True, "replacement": "근거를 막연하게 언급하고 결론을 제시한다."},
            *[{"apply": False, "replacement": ""} for _ in range(4)],
        ]}
        with self.assertRaises(SolarTargetAugmentationError):
            parse_editor_output(json.dumps(leaked), source, "content")

    def test_editor_accepts_only_augmented_essay(self) -> None:
        source = row(3)
        edited = source.essay.replace("찬성한다", "찬성하는 편이다")
        self.assertEqual(parse_editor_output(json.dumps({"augmented_essay": edited}), source), edited)
        with self.assertRaises(SolarTargetAugmentationError):
            parse_editor_output(json.dumps({"augmented_essay": edited, "score": {}}), source)
        with self.assertRaises(SolarTargetAugmentationError):
            parse_editor_output(json.dumps({"augmented_essay": source.essay}), source)
        with self.assertRaises(SolarTargetAugmentationError):
            parse_editor_output(json.dumps({"augmented_essay": source.essay + "!"}), source)

    def test_score_verifier_is_target_blind_and_uses_exact_evaluation_contract(self) -> None:
        source = row(4)
        augmented = source.essay.replace("찬성한다", "찬성한다는 입장이다")
        messages = render_verifier_messages(source.prompt, augmented)
        serialized = json.dumps(messages, ensure_ascii=False)
        expected_system, expected_user = evaluation_sections()
        self.assertEqual(messages[0]["content"], expected_system)
        self.assertEqual(
            messages[1]["content"],
            expected_user.replace("{주제 지문}", source.prompt).replace("{논증적 글 본문}", augmented),
        )
        self.assertNotIn("[유저 프롬프트]", messages[0]["content"])
        self.assertIn(source.prompt, serialized)
        self.assertIn(augmented, serialized)
        self.assertNotIn("target_axis", serialized)
        self.assertNotIn("target_score", serialized)
        self.assertNotIn("baseline_score", serialized)
        self.assertNotIn(source.essay, serialized)

    def test_fidelity_audit_is_target_blind(self) -> None:
        source = row(5)
        augmented = source.essay + " 원래 논지를 다시 확인한다."
        serialized = json.dumps(render_fidelity_messages(source.essay, augmented), ensure_ascii=False)
        self.assertIn(source.essay, serialized)
        self.assertIn(augmented, serialized)
        self.assertNotIn("target_axis", serialized)
        self.assertNotIn("target_score", serialized)
        self.assertNotIn("baseline_score", serialized)

    def test_parsers_require_integer_scores_rationales_and_boolean_fidelity(self) -> None:
        verifier = {axis: {"score": 3, "rationale": f"{axis} 근거"} for axis in AXES}
        self.assertEqual(parse_verifier_output(json.dumps(verifier))["content"]["score"], 3)
        verifier["content"]["score"] = 3.5
        with self.assertRaises(SolarTargetAugmentationError):
            parse_verifier_output(json.dumps(verifier))
        fidelity = {"source_based": True, "topic": True, "stance": True, "genre": True,
                    "new_external_facts_added": False}
        self.assertEqual(parse_fidelity_output(json.dumps(fidelity)), fidelity)
        fidelity["topic"] = "yes"
        with self.assertRaises(SolarTargetAugmentationError):
            parse_fidelity_output(json.dumps(fidelity))

    def test_hard_gates_accept_exact_target_and_reject_mutations(self) -> None:
        source = row(6)
        task = make_task(source, "content", 2)
        essay = source.essay.replace("근거를 설명하고", "근거를 막연히 언급하고")
        verifier = {
            "content": {"score": 2, "rationale": "근거가 부족하다."},
            "organization": {"score": 4, "rationale": "구조가 유지된다."},
            "expression": {"score": 3, "rationale": "표현이 유지된다."},
        }
        fidelity = {"source_based": True, "topic": True, "stance": True, "genre": True,
                    "new_external_facts_added": False}
        source_verifier = {
            "content": {"score": 4, "rationale": "원문 내용"},
            "organization": {"score": 4, "rationale": "원문 구조"},
            "expression": {"score": 3, "rationale": "원문 표현"},
        }
        self.assertEqual(validate_candidate(task, essay, verifier, source_verifier, fidelity)["target_score"], 2)
        wrong_target = {axis: dict(value) for axis, value in verifier.items()}
        wrong_target["content"]["score"] = 3
        with self.assertRaises(SolarTargetAugmentationError):
            validate_candidate(task, essay, wrong_target, source_verifier, fidelity)
        wrong_non_target = {axis: dict(value) for axis, value in verifier.items()}
        wrong_non_target["organization"]["score"] = 3
        with self.assertRaises(SolarTargetAugmentationError):
            validate_candidate(task, essay, wrong_non_target, source_verifier, fidelity)
        bad_fidelity = dict(fidelity, new_external_facts_added=True)
        with self.assertRaises(SolarTargetAugmentationError):
            validate_candidate(task, essay, verifier, source_verifier, bad_fidelity)

    def test_non_target_gate_uses_blind_source_score_not_gold_float(self) -> None:
        source = row(6)
        task = make_task(source, "content", 2)
        essay = source.essay.replace("근거를 설명하고", "근거를 막연히 언급하고")
        verifier = {axis: {"score": score, "rationale": "근거"} for axis, score in
                    {"content": 2, "organization": 3, "expression": 4}.items()}
        source_verifier = {axis: {"score": score, "rationale": "원문 근거"} for axis, score in
                           {"content": 4, "organization": 3, "expression": 4}.items()}
        fidelity = {"source_based": True, "topic": True, "stance": True, "genre": True,
                    "new_external_facts_added": False}
        validate_candidate(task, essay, verifier, source_verifier, fidelity)

    def test_train_validation_disjoint_on_both_identity_fields(self) -> None:
        train = [row(7), row(8)]
        validate_train_validation_disjoint(train, [{"id": "v", "document_id": "vd"}])
        with self.assertRaises(SolarTargetAugmentationError):
            validate_train_validation_disjoint(train, [{"id": train[0].identifier, "document_id": "vd"}])
        with self.assertRaises(SolarTargetAugmentationError):
            validate_train_validation_disjoint(train, [{"id": "v", "document_id": train[0].document_id}])

    def test_smoke_selection_is_deterministic_order_independent_and_not_first_n(self) -> None:
        population = self.rows[:100]
        selected = select_smoke_sources(population, count=5, seed="fixed")
        reversed_selected = select_smoke_sources(list(reversed(population)), count=5, seed="fixed")
        self.assertEqual([item.identifier for item in selected], [item.identifier for item in reversed_selected])
        self.assertNotEqual([item.identifier for item in selected], [item.identifier for item in population[:5]])
        with self.assertRaises(SolarTargetAugmentationError):
            select_smoke_sources(population, count=4, seed="fixed")

    def test_source_parser_never_reads_derived_score(self) -> None:
        raw = {
            "id": "safe", "document_id": "safe-doc", "prompt": "논제", "essay": "충분히 긴 본문입니다.",
            "score": NoDerivedScoreAccess(content=3.0, organization=4.0, expression=2.0),
        }
        self.assertEqual(source_row_from_mapping(raw).score, (3.0, 4.0, 2.0))


if __name__ == "__main__":
    unittest.main()
