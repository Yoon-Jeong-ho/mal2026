#!/usr/bin/env python3
"""One-shot data-steward creation of the label-free phase1-direct input."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable, Mapping, Any

ROOT = Path(__file__).resolve().parents[1]
AXES = ("content", "organization", "expression")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"ordinary file required: {path}")
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def secure_tree(parent: Path, anchor: Path) -> None:
    anchor = anchor.resolve(); parent = parent.resolve()
    need(anchor == parent or anchor in parent.parents, "restricted output is outside its anchor")
    anchor.mkdir(parents=True, exist_ok=True)
    chain = [anchor, *reversed([item for item in parent.parents if anchor in item.parents]), parent]
    seen = set()
    for directory in chain:
        if directory in seen: continue
        seen.add(directory); directory.mkdir(exist_ok=True); os.chmod(directory, 0o770)
        need(directory.stat().st_mode & 0o007 == 0, f"world-accessible restricted parent: {directory}")


def verify_private(path: Path, anchor: Path) -> None:
    need(path.is_file() and not path.is_symlink() and path.stat().st_mode & 0o007 == 0, "restricted source file ACL differs")
    anchor = anchor.resolve(); cursor = path.resolve().parent
    need(anchor == cursor or anchor in cursor.parents, "restricted source is outside anchor")
    while True:
        need(cursor.is_dir() and not cursor.is_symlink() and cursor.stat().st_mode & 0o007 == 0,
             f"restricted source parent ACL differs: {cursor}")
        if cursor == anchor: break
        cursor = cursor.parent


def publish_private(path: Path, lines: Iterable[str], anchor: Path) -> str:
    need(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    secure_tree(path.parent, anchor)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for line in lines: stream.write(line)
            stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o660)
        os.link(temporary_path, path)  # atomic EEXIST no-clobber publication
        temporary_path.unlink(); os.chmod(path, 0o660)
        need(path.stat().st_mode & 0o007 == 0 and not path.is_symlink(), "restricted output ACL differs")
        parent_fd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(parent_fd)
        finally: os.close(parent_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return digest(path)


def load_memberships(config: Mapping[str, Any], aggregate: Mapping[str, Any]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    bindings = config.get("fold_membership_bindings")
    need(isinstance(bindings, list) and len(bindings) == 5, "five membership bindings required")
    coral = next((item for item in aggregate.get("methods", ()) if item.get("method") == "coral-natural"), None)
    need(isinstance(coral, Mapping), "Stage3 coral-natural entry missing")
    reported = {int(item["outer_fold"]): item["restricted_prediction_sha256"] for item in coral["fold_bindings"]}
    result: dict[str, int] = {}; evidence = []
    anchor = ROOT / "data/processed/restricted"
    for expected_fold, binding in enumerate(bindings):
        need(set(binding) == {"outer_fold", "path", "sha256"} and binding["outer_fold"] == expected_fold,
             "ordered membership binding differs")
        path = ROOT / binding["path"]
        need(digest(path) == binding["sha256"] == reported[expected_fold], "membership/Stage3 binding differs")
        verify_private(path, anchor)
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                need(set(row) == {"source_id", "outer_fold", "prediction"} and row["outer_fold"] == expected_fold,
                     "membership row schema/fold differs")
                identifier = row["source_id"]
                need(isinstance(identifier, str) and identifier not in result, "membership ID differs")
                need(isinstance(row["prediction"], Mapping) and set(row["prediction"]) == set(AXES),
                     "membership prediction axes differ")
                result[identifier] = expected_fold; count += 1
        need(count == 400, "membership fold size differs")
        evidence.append({"outer_fold": expected_fold, "path": binding["path"], "sha256": binding["sha256"], "records": count})
    need(len(result) == 2000, "membership population differs")
    return result, evidence


def create_projection_rows(train: Path, expected_sha256: str, folds: Mapping[str, int],
                           *, expected_per_fold: int = 400) -> tuple[list[dict[str, Any]], dict[str, int]]:
    need(digest(train) == expected_sha256, "canonical train checksum differs")
    projection_rows = []; seen = set(); fold_counts = {str(fold): 0 for fold in range(5)}
    with train.open(encoding="utf-8") as stream:
        for line in stream:
            source = json.loads(line)
            need(set(source) == {"id", "document_id", "prompt_num", "prompt", "essay", "score"}, "canonical schema differs")
            identifier = source["id"]
            need(identifier in folds and identifier not in seen, "canonical ID/membership differs")
            seen.add(identifier); fold = folds[identifier]; fold_counts[str(fold)] += 1
            row = {"id": identifier, "document_id": source["document_id"], "prompt_num": source["prompt_num"],
                   "prompt": source["prompt"], "essay": source["essay"], "outer_fold": fold}
            need(not ({"score", "average", "gold"} & set(row)), "label field entered projection")
            projection_rows.append(row)
    expected_counts = {str(fold): expected_per_fold for fold in range(5)}
    need(seen == set(folds) and fold_counts == expected_counts, "projection population/folds differ")
    return projection_rows, fold_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve(); config = json.loads(config_path.read_text(encoding="utf-8"))
    train = ROOT / config["train_path"]
    aggregate_path = ROOT / config["source_stage3_aggregate_path"]
    need(digest(aggregate_path) == config["source_stage3_aggregate_sha256"], "Stage3 aggregate checksum differs")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    need(aggregate.get("status") == "completed" and aggregate.get("run_id") == "kure-ordinal-oof-v1-20260803-001",
         "Stage3 aggregate identity differs")
    folds, membership_evidence = load_memberships(config, aggregate)
    projection_rows, fold_counts = create_projection_rows(train, config["train_sha256"], folds)
    projection_path = ROOT / config["label_free_projection_path"]
    manifest_path = ROOT / config["label_free_manifest_path"]
    need(projection_path != manifest_path, "projection and manifest paths must differ")
    anchor = ROOT / "data/processed/restricted"
    projection_sha = publish_private(projection_path, (
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in projection_rows), anchor)
    generator = Path(__file__).resolve()
    manifest = {
        "schema_version": "mal2026-kure-phase1-direct-input-manifest-v1", "status": "completed",
        "records": 2000, "fold_counts": fold_counts, "projection_path": str(projection_path.relative_to(ROOT)),
        "projection_sha256": projection_sha, "projection_schema": ["id", "document_id", "prompt_num", "prompt", "essay", "outer_fold"],
        "labels_present": False, "average_present": False, "gold_present": False,
        "source_train_path": config["train_path"], "source_train_sha256": config["train_sha256"],
        "source_stage3_aggregate_path": config["source_stage3_aggregate_path"],
        "source_stage3_aggregate_sha256": config["source_stage3_aggregate_sha256"],
        "fold_membership_bindings": membership_evidence,
        "generator_path": str(generator.relative_to(ROOT)), "generator_sha256": digest(generator),
        "generator_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_path": str(config_path.relative_to(ROOT)),
        "preparation_request_config_sha256": digest(config_path),
    }
    manifest_sha = publish_private(manifest_path, [json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"], anchor)
    print(json.dumps({"status": "completed", "projection_sha256": projection_sha, "manifest_sha256": manifest_sha}, sort_keys=True))


if __name__ == "__main__": main()
