from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from mal2026.solar_target_augmentation import SourceRow, make_task


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_solar_actual_label_smoke.py"
SPEC = importlib.util.spec_from_file_location("solar_actual_label_smoke", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(
    family: int,
    actual_target: int,
    *,
    non_target_l1: int,
    target_progress: int,
) -> dict:
    return {
        "task_id": "source::solar-target::content::1",
        "candidate_id": f"candidate-{family}",
        "candidate_family_index": family,
        "candidate_essay_sha256": f"hash-{family}",
        "requested_target_axis": "content",
        "requested_target_score": 1,
        "score": {"content": actual_target, "organization": 3, "expression": 3},
        "movement": {
            "candidate_target_distance": abs(actual_target - 1),
            "target_progress": target_progress,
            "non_target_l1_drift": non_target_l1,
            "target_exact": actual_target == 1,
            "direction_followed": actual_target < 3,
            "progress_class": (
                "closer" if target_progress > 0 else
                ("same" if target_progress == 0 else "farther")
            ),
            "non_target_exact": non_target_l1 == 0,
        },
    }


class SolarActualLabelSmokeTests(unittest.TestCase):
    def test_movement_uses_blind_actual_scores_not_gold_or_requested_label(self) -> None:
        source = SourceRow(
            identifier="source", document_id="document", prompt="논제",
            essay="입장을 제시한다. 근거를 설명한다. 결론을 정리한다.",
            score=(4.7, 4.5, 4.25),
        )
        task = make_task(source, "content", 1)
        metrics = MODULE.movement_metrics(
            task,
            {"content": 2, "organization": 2, "expression": 3},
            {"content": 3, "organization": 3, "expression": 3},
        )
        self.assertEqual(metrics["source_blind_target_score"], 3)
        self.assertEqual(metrics["actual_target_score"], 2)
        self.assertEqual(metrics["target_progress"], 1)
        self.assertEqual(metrics["progress_class"], "closer")
        self.assertEqual(metrics["non_target_l1_drift"], 1)
        self.assertFalse(metrics["target_exact"])

    def test_selection_is_independent_of_actual_scores_and_target_distance(self) -> None:
        candidates = [
            record(0, 2, non_target_l1=0, target_progress=1),
            record(1, 1, non_target_l1=2, target_progress=2),
            record(2, 1, non_target_l1=0, target_progress=2),
        ]
        original = MODULE.select_one_per_task(candidates)
        self.assertEqual(len(original), 1)
        mutated = []
        for candidate in candidates:
            value = dict(candidate)
            value["score"] = {"content": 5, "organization": 1, "expression": 5}
            value["movement"] = dict(candidate["movement"])
            value["movement"].update({
                "candidate_target_distance": 4,
                "target_progress": -3,
                "non_target_l1_drift": 8,
            })
            mutated.append(value)
        repeated = MODULE.select_one_per_task(mutated)
        self.assertEqual(
            original[0]["candidate_family_index"], repeated[0]["candidate_family_index"]
        )
        self.assertTrue(original[0]["selected_for_task_diagnostic"])
        self.assertIn("independent_of_scores", original[0]["selection_rule"])

    def test_requested_cell_metrics_report_actual_distribution(self) -> None:
        candidates = [
            record(0, 2, non_target_l1=0, target_progress=1),
            record(1, 1, non_target_l1=1, target_progress=2),
        ]
        metrics = MODULE.requested_cell_metrics(candidates)["content:1"]
        self.assertEqual(metrics["valid"], 2)
        self.assertEqual(metrics["actual_target_distribution"]["1"], 1)
        self.assertEqual(metrics["actual_target_distribution"]["2"], 1)
        self.assertEqual(metrics["target_exact"], 1)
        self.assertEqual(metrics["non_target_exact"], 1)

    def test_modal_label_requires_unique_five_draw_majority(self) -> None:
        stable, support = MODULE.modal_triplet([
            {"content": 2, "organization": 3, "expression": 3},
            {"content": 2, "organization": 3, "expression": 3},
            {"content": 2, "organization": 3, "expression": 3},
            {"content": 2, "organization": 2, "expression": 3},
            {"content": 3, "organization": 3, "expression": 3},
        ])
        self.assertEqual(stable, {"content": 2, "organization": 3, "expression": 3})
        self.assertEqual(support, 3)
        unstable, support = MODULE.modal_triplet([
            {"content": 1, "organization": 3, "expression": 3},
            {"content": 1, "organization": 2, "expression": 3},
            {"content": 2, "organization": 3, "expression": 3},
            {"content": 2, "organization": 2, "expression": 3},
            {"content": 3, "organization": 3, "expression": 3},
        ])
        self.assertIsNone(unstable)
        self.assertEqual(support, 1)

    def test_runner_has_no_full_mode_and_preserves_all_families(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn('choices=("smoke", "full")', source)
        self.assertEqual(MODULE.CANDIDATE_FAMILIES, 4)
        self.assertEqual(MODULE.TASK_COUNT, 75)
        self.assertEqual(MODULE.CANDIDATE_COUNT, 300)
        self.assertEqual(MODULE.MODAL_LABEL_DRAWS, 5)
        self.assertEqual(MODULE.ACTUAL_EDITOR_MAX_TOKENS, 1600)
        self.assertIn('"requested_target_is_label": False', source)
        self.assertIn('"actual_blind_triplet_is_label": True', source)
        self.assertIn("enforce_score_specific_edit_count=False", source)


if __name__ == "__main__":
    unittest.main()
