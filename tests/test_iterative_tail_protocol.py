from __future__ import annotations

from collections import Counter
from copy import deepcopy
import unittest

from mal2026.iterative_tail_protocol import (
    CONFIG_PATH,
    FINAL_GATE,
    PROMOTION_GATE,
    ROUND_METHODS,
    ROUND_NAMES,
    RUN_ID,
    IterativeTailProtocolError,
    build_protocol_summary,
    build_task_card,
    load_bound_training_rows,
    load_protocol,
    validate_protocol_mapping,
)


class IterativeTailProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(CONFIG_PATH)

    def test_exact_run_round_gpu_and_isolation_contract(self):
        protocol = self.protocol
        self.assertEqual(RUN_ID, protocol.run_id)
        self.assertEqual(ROUND_NAMES, tuple(item["name"] for item in protocol.rounds))
        self.assertEqual(ROUND_METHODS, tuple((item["method_family"], item["hyperparameters"]) for item in protocol.rounds))
        self.assertEqual(list(range(1, 21)), [item["number"] for item in protocol.rounds])
        self.assertEqual([0, 1, 2, 3], protocol.raw["execution"]["authorized_gpus"])
        self.assertEqual(0, protocol.raw["execution"]["smoke_gpu"])
        self.assertTrue(protocol.raw["execution"]["fresh_initialization_per_round"])
        self.assertFalse(protocol.raw["data_contract"]["contains_average_target"])
        self.assertFalse(protocol.raw["data_contract"]["validation_loaded"])
        self.assertFalse(protocol.raw["data_contract"]["validation_selection"])
        self.assertFalse(protocol.raw["optional_api"]["enabled"])
        self.assertEqual(PROMOTION_GATE, protocol.raw["promotion_gate"])
        self.assertEqual(FINAL_GATE, protocol.raw["final_gate"])
        self.assertEqual(0.005, protocol.raw["promotion_gate"]["macro_rmse_min_improvement"])
        self.assertEqual(0.010, protocol.raw["promotion_gate"]["equal_group_rmse_min_improvement"])
        self.assertEqual(0.01, protocol.raw["final_gate"]["macro_rmse_min_improvement"])
        self.assertEqual(10000, protocol.raw["final_gate"]["bootstrap_resamples"])
        self.assertEqual(2026080101, protocol.raw["final_gate"]["bootstrap_seed"])

    def test_protocol_rejects_binding_round_and_gate_drift(self):
        for mutate in (
            lambda raw: raw["bindings"].__setitem__("embedding_rows_sha256", "0" * 64),
            lambda raw: raw["rounds"][4].__setitem__("name", "changed"),
            lambda raw: raw["rounds"][1]["hyperparameters"]["alpha_grid"].append(1000.0),
            lambda raw: raw["bindings"].__setitem__("score_blind_rationale_train_sha256", "0" * 64),
            lambda raw: raw["promotion_gate"].__setitem__("fixed", False),
            lambda raw: raw["promotion_gate"].__setitem__("max_axis_rmse_worsening", 0.02),
            lambda raw: raw["final_gate"].__setitem__("validation_selection", True),
            lambda raw: raw["final_gate"].__setitem__("bootstrap_resamples", 9999),
            lambda raw: raw["optional_api"].__setitem__("enabled", True),
        ):
            with self.subTest(mutate=mutate):
                raw = deepcopy(self.protocol.raw)
                mutate(raw)
                with self.assertRaises(IterativeTailProtocolError):
                    validate_protocol_mapping(raw)

    def test_canonical_train_embeddings_are_exact_oof_2000_and_balanced(self):
        rows = load_bound_training_rows(self.protocol)
        self.assertEqual(2000, len(rows))
        self.assertEqual(Counter({fold: 400 for fold in range(5)}), Counter(row.oof_fold for row in rows))
        self.assertNotIn(None, {row.oof_fold for row in rows})

    def test_task_card_and_summary_expose_aggregate_fields_only(self):
        card = build_task_card(self.protocol)
        self.assertEqual(2000, card["train_records"])
        self.assertEqual(20, card["round_count"])
        self.assertFalse(card["validation_selection"])
        summary = build_protocol_summary(self.protocol, [{"count": 2000, "macro_axis_rmse": 0.56, "promoted": False}])
        self.assertEqual(1, summary["completed_rounds"])
        self.assertEqual("baseline", summary["rounds"][0]["name"])
        with self.assertRaisesRegex(IterativeTailProtocolError, "row-level"):
            build_protocol_summary(self.protocol, [{"source_id": "restricted"}])
        with self.assertRaisesRegex(IterativeTailProtocolError, "scalar"):
            build_protocol_summary(self.protocol, [{"fold_rmse": [0.5] * 5}])


if __name__ == "__main__":
    unittest.main()
