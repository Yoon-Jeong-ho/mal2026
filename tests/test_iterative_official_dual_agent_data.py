from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from mal2026.iterative_official_dual_agent_data import (
    EXPECTED_CANDIDATES,
    EXPECTED_ESSAYS,
    SCHEMA_VERSION,
    SOURCE_SHA256,
    SOURCE_SPECS,
    OfficialDualAgentDataError,
    load_dual_candidates,
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _participant(number: int) -> dict[str, dict[str, object]]:
    return {
        axis: {"score": 2 + (number + offset) % 3, "rationale": f"{axis} 후보 {number} 근거"}
        for offset, axis in enumerate(("content", "organization", "expression"))
    }


def _write_source(root: Path, source: str, run_id: str, model: str,
                  essay_hashes: dict[str, str]) -> tuple[Path, Path]:
    candidates = root / f"{source}.candidates.jsonl"
    with candidates.open("w", encoding="utf-8") as handle:
        for index, (source_id, essay_sha) in enumerate(essay_hashes.items()):
            for number in (1, 2, 3):
                row = {
                    "custom_id": f"{run_id}:train:{number}:{index:08d}",
                    "source_id": source_id, "split": "train", "candidate": number,
                    "essay_sha256": essay_sha, "model": model,
                    "schema_version": SCHEMA_VERSION, "participant_output": _participant(number),
                    "api_response_id": f"resp-{source}-{index}-{number}",
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = root / f"{source}.manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "status": "validated", "run_id": run_id,
        "model": model, "split": "train", "train_rows": EXPECTED_ESSAYS,
        "candidates_per_essay": 3, "requests": EXPECTED_CANDIDATES,
        "accepted": EXPECTED_CANDIDATES, "source_sha256": SOURCE_SHA256,
        "human_or_reference_score_read_or_prompted": False,
        "candidates_sha256": _sha(candidates),
    }, sort_keys=True), encoding="utf-8")
    return manifest, candidates


class DualAgentDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.essay_hashes = {
            f"essay-{index:04d}": sha256(f"canonical essay {index}".encode()).hexdigest()
            for index in range(EXPECTED_ESSAYS)
        }
        cls.paths = []
        for source, run_id, model in SOURCE_SPECS:
            cls.paths.extend(_write_source(cls.root, source, run_id, model, cls.essay_hashes))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _load(self, paths: list[Path] | None = None):
        return load_dual_candidates(*(paths or self.paths), essay_sha256_by_source=self.essay_hashes)

    def test_valid_dual_sources_return_rows_and_aggregate_only_provenance(self) -> None:
        rows, provenance = self._load()
        self.assertEqual(len(rows), 12000)
        self.assertEqual({row.agent_source for row in rows}, {"terra", "luna"})
        self.assertEqual({row.candidate_number for row in rows}, {1, 2, 3})
        self.assertEqual(provenance["candidate_count"], 12000)
        self.assertEqual(provenance["essay_count"], 2000)
        self.assertEqual(provenance["source_count"], 2)
        self.assertFalse(provenance["row_content_in_provenance"])
        self.assertEqual([item["candidate_count"] for item in provenance["sources"]], [6000, 6000])
        serialized = json.dumps(provenance)
        self.assertNotIn("rationale", serialized)
        self.assertNotIn("source_id", serialized)

    def test_manifest_must_be_validated_and_exactly_identify_model_run_source_and_count(self) -> None:
        for key, value in (
            ("status", "validating"), ("model", "wrong-model"), ("run_id", "wrong-run"),
            ("source_sha256", "0" * 64), ("accepted", 5999),
        ):
            with self.subTest(key=key):
                altered = self.root / f"altered-{key}.manifest.json"
                raw = json.loads(self.paths[2].read_text(encoding="utf-8"))
                raw[key] = value
                altered.write_text(json.dumps(raw), encoding="utf-8")
                paths = [self.paths[0], self.paths[1], altered, self.paths[3]]
                with self.assertRaises(OfficialDualAgentDataError):
                    self._load(paths)

    def _mutated_luna(self, mutate) -> list[Path]:
        rows = self.paths[3].read_text(encoding="utf-8").splitlines()
        value = json.loads(rows[0])
        mutate(value)
        rows[0] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        candidates = self.root / f"mutated-{sha256(rows[0].encode()).hexdigest()[:10]}.jsonl"
        candidates.write_text("\n".join(rows) + "\n", encoding="utf-8")
        manifest_raw = json.loads(self.paths[2].read_text(encoding="utf-8"))
        manifest_raw["candidates_sha256"] = _sha(candidates)
        manifest = candidates.with_suffix(".manifest.json")
        manifest.write_text(json.dumps(manifest_raw), encoding="utf-8")
        return [self.paths[0], self.paths[1], manifest, candidates]

    def test_candidate_essay_sha_and_per_essay_coverage_fail_closed(self) -> None:
        for mutate in (
            lambda row: row.__setitem__("essay_sha256", "0" * 64),
            lambda row: row.__setitem__("candidate", 2),
        ):
            with self.assertRaises(OfficialDualAgentDataError):
                self._load(self._mutated_luna(mutate))

    def test_participant_and_exact_row_schema_fail_closed(self) -> None:
        def invalid_participant(row):
            row["participant_output"]["content"]["score"] = 2.5

        def extra_field(row):
            row["unexpected"] = True

        for mutate in (invalid_participant, extra_field):
            with self.assertRaises(OfficialDualAgentDataError):
                self._load(self._mutated_luna(mutate))

    def test_candidate_checksum_and_canonical_essay_population_are_required(self) -> None:
        candidates = self.root / "checksum-drift.jsonl"
        candidates.write_bytes(self.paths[3].read_bytes() + b"\n")
        with self.assertRaises(OfficialDualAgentDataError):
            self._load([self.paths[0], self.paths[1], self.paths[2], candidates])
        incomplete = dict(self.essay_hashes)
        incomplete.pop(next(iter(incomplete)))
        with self.assertRaises(OfficialDualAgentDataError):
            load_dual_candidates(*self.paths, essay_sha256_by_source=incomplete)


if __name__ == "__main__":
    unittest.main()
