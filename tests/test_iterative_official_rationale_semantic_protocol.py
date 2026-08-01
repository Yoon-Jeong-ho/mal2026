from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from mal2026.iterative_official_rationale_embedding_data import (
    RUN_ID as FEATURE_RUN_ID,
    SCHEMA_VERSION as FEATURE_SCHEMA,
    matrix_sha256,
    rademacher_projection,
)
from mal2026.iterative_official_rationale_semantic_protocol import (
    BOUND,
    OfficialRationaleSemanticProtocol,
    OfficialRationaleSemanticProtocolError,
    load_protocol,
    validate_bound_inputs,
    validate_protocol_mapping,
)


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return sha256(value).hexdigest()


def _json(path: Path, value: object) -> str:
    return _write(path, (json.dumps(value, sort_keys=True) + "\n").encode())


def _bound_fixture(tmp_path: Path) -> OfficialRationaleSemanticProtocol:
    raw = deepcopy(load_protocol().raw)
    raw["binding_state"] = BOUND
    lineage = raw["lineage"]

    ids = [f"source-{index:04d}" for index in range(2000)]
    canonical = b"".join((json.dumps({"id": source_id}) + "\n").encode() for source_id in ids)
    folds = {source_id: index % 5 for index, source_id in enumerate(ids)}
    baseline = b"".join(
        (json.dumps({"source_id": source_id, "fold": folds[source_id]}) + "\n").encode()
        for source_id in reversed(ids)
    )
    fingerprint = sha256("\n".join(f"{key}:{folds[key]}" for key in sorted(folds)).encode()).hexdigest()

    def bind(name: str, relative: str, content: bytes) -> Path:
        path = tmp_path / relative
        lineage[f"{name}_path"] = relative
        lineage[f"{name}_sha256"] = _write(path, content)
        return path

    bind("canonical_train", "canonical.jsonl", canonical)
    bind("baseline_oof", "baseline.jsonl", baseline)
    lineage["fold_assignment_fingerprint"] = fingerprint

    for source, model in (("terra", "gpt-5.6-terra"), ("luna", "gpt-5.6-luna")):
        rows = bind(f"{source}_candidate_rows", f"{source}/rows.jsonl", b"restricted rationale rows\n")
        manifest = {
            "schema_version": "mal2026-official-openai-candidate-v1", "status": "validated",
            "model": model, "split": "train", "train_rows": 2000,
            "candidates_per_essay": 3, "requests": 6000, "accepted": 6000,
            "human_or_reference_score_read_or_prompted": False,
            "official_system_prompt_sha256": lineage["official_system_prompt_sha256"],
            "candidates_sha256": sha256(rows.read_bytes()).hexdigest(),
        }
        path = tmp_path / f"{source}/manifest.json"
        lineage[f"{source}_candidate_manifest_path"] = str(path.relative_to(tmp_path))
        lineage[f"{source}_candidate_manifest_sha256"] = _json(path, manifest)

    bind("v11_config", "v11/config.json", b"{}\n")
    bind("v11_aggregate", "v11/aggregate.json", (json.dumps({
        "schema_version": "mal2026-iterative-official-balanced-boundary-aggregate-v11",
        "status": "completed", "final_gate_pass": False,
    }) + "\n").encode())
    bind("v11_completion", "v11/completion.json", (json.dumps({
        "schema_version": "mal2026-iterative-official-balanced-boundary-completion-v11",
        "status": "completed_no_promotion_baseline_retained", "final_gate_pass": False,
    }) + "\n").encode())

    lineage["qwen_model_path"] = "model"
    bind("qwen_model_config", "model/config.json", (json.dumps({
        "model_type": "qwen3", "hidden_size": 4096, "architectures": ["Qwen3ForCausalLM"],
    }) + "\n").encode())

    feature_bytes = b"".join(
        (json.dumps({"source_id": source_id, "features": [0.0] * 201}, separators=(",", ":")) + "\n").encode()
        for source_id in ids
    )
    feature_rows = bind("generated_feature_rows", "features/rows.jsonl", feature_bytes)
    render_contract = {
        "kind": "participant_axis_rationale_text_alone_v1", "source_order": ["terra", "luna"],
        "candidate_order": [1, 2, 3], "axis_order": ["content", "organization", "expression"],
        "essay_included": False, "prompt_included": False,
        "participant_score_included": False, "gold_included": False,
    }
    render_hash = sha256(json.dumps(render_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    feature_manifest = {
        "schema_version": FEATURE_SCHEMA, "status": "completed", "run_id": FEATURE_RUN_ID,
        "split_role": "train", "records": 2000, "feature_dim": 201,
        "embedding_dim": 4096, "projection_dim": 32, "projection_seed": 2026080212,
        "model_id": lineage["qwen_model_id"], "model_revision": lineage["qwen_model_revision"],
        "model_config_sha256": lineage["qwen_model_config_sha256"],
        "projection_matrix_sha256": matrix_sha256(rademacher_projection()),
        "feature_rows_sha256": sha256(feature_rows.read_bytes()).hexdigest(),
        "render_contract": render_contract, "render_contract_sha256": render_hash,
        "candidate_bindings": {
            "canonical_train_sha256": lineage["canonical_train_sha256"],
            "terra_manifest_sha256": lineage["terra_candidate_manifest_sha256"],
            "luna_manifest_sha256": lineage["luna_candidate_manifest_sha256"],
            "official_system_prompt_sha256": lineage["official_system_prompt_sha256"],
            "render_contract_sha256": render_hash,
        },
        "validation_loaded": False,
        "candidate_score_in_embedding_text": False, "essay_in_embedding_text": False,
        "prompt_in_embedding_text": False,
    }
    manifest_bytes = (json.dumps(feature_manifest, sort_keys=True) + "\n").encode()
    bind("generated_feature_manifest", "features/manifest.json", manifest_bytes)
    bind("generated_feature_public_manifest", "public/manifest.json", manifest_bytes)
    return OfficialRationaleSemanticProtocol(tmp_path / "protocol.json", raw)


class OfficialRationaleSemanticProtocolTest(unittest.TestCase):
    def test_preregistered_unbound_state_is_fail_closed_after_binding_transition(self) -> None:
        live = load_protocol()
        raw = deepcopy(live.raw)
        raw["binding_state"] = "awaiting_generated_feature_artifact"
        raw["lineage"]["generated_feature_manifest_sha256"] = None
        raw["lineage"]["generated_feature_rows_sha256"] = None
        raw["lineage"]["generated_feature_public_manifest_sha256"] = None
        protocol = validate_protocol_mapping(raw)
        self.assertEqual(protocol.raw["binding_state"], "awaiting_generated_feature_artifact")
        self.assertEqual(protocol.raw["semantic_feature_contract"]["semantic_dimensions"], 201)
        self.assertEqual([item["variant_id"] for item in protocol.raw["candidates"]], [
            "rationale-semantic201-ridge-a10-cap050",
            "rationale-fusion297-ridge-a10-cap050",
            "rationale-fusion297-balanced-3v4-l2-001-c055-w020",
        ])
        with self.assertRaisesRegex(OfficialRationaleSemanticProtocolError, "awaiting checksum-only binding"):
            validate_bound_inputs(protocol)

    def test_protocol_rejects_scientific_or_fixed_lineage_drift(self) -> None:
        raw = deepcopy(load_protocol().raw)
        raw["semantic_feature_contract"]["projection_seed"] += 1
        with self.assertRaisesRegex(OfficialRationaleSemanticProtocolError, "feature contract"):
            validate_protocol_mapping(raw)
        raw = deepcopy(load_protocol().raw)
        raw["lineage"]["official_system_prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(OfficialRationaleSemanticProtocolError, "fixed lineage"):
            validate_protocol_mapping(raw)

    def test_bound_protocol_validates_target_blind_features_in_canonical_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = validate_bound_inputs(_bound_fixture(root), root=root)
        self.assertEqual(audit.records, 2000)
        self.assertEqual(audit.folds, {0: 400, 1: 400, 2: 400, 3: 400, 4: 400})
        self.assertEqual(audit.semantic_dimensions, 201)


if __name__ == "__main__":
    unittest.main()
