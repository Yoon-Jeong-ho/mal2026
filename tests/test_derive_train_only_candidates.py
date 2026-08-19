"""Synthetic-only checks for the restricted train-only lineage transformer."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("derive_train_only_candidates", ROOT / "scripts" / "derive_train_only_candidates.py")
assert SPEC and SPEC.loader
DERIVE = module_from_spec(SPEC)
SPEC.loader.exec_module(DERIVE)


def test_canonical_set_hash_is_order_independent_and_does_not_include_members() -> None:
    first = DERIVE.canonical_set_sha256({"synthetic-a", "synthetic-b"})
    second = DERIVE.canonical_set_sha256({"synthetic-b", "synthetic-a"})
    assert first == second
    assert "synthetic-a" not in first and "synthetic-b" not in first


def test_mapping_index_rejects_cross_split_source_overlap(tmp_path: Path) -> None:
    path = tmp_path / "source_map.jsonl"
    path.write_text(
        '{"custom_id":"synthetic-1","source_id":"synthetic-source","split":"train","candidate":1}\n'
        '{"custom_id":"synthetic-2","source_id":"synthetic-source","split":"validation","candidate":1}\n',
        encoding="utf-8",
    )
    try:
        DERIVE.mapping_index(path, 2)
    except RuntimeError as exc:
        assert str(exc) in {"source map has duplicate routing keys", "source map does not prove complete disjoint split routing"}
    else:
        raise AssertionError("cross-split source overlap was accepted")
