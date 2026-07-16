"""Prepare the restricted AI-Hub human-feedback training corpus.

This module deliberately keeps student writing, prompts, feedback, and record
identifiers in memory.  Only :func:`write_prepared_dataset` writes them, and it
requires an ignored ``data/processed`` destination.  The accompanying manifest
contains aggregate counts, hashes, and archive fingerprints only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Protocol, Sequence
from zipfile import ZipFile

from .data_contract import DataContractError, file_sha256, normalize_prompt, stable_hash


DATASET_ID = "aihub_human_feedback_v1"
SCHEMA_VERSION = 1
DEV_FRACTION = Decimal("0.20")
TARGET_TOKEN_CAP = 1536
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
QWEN_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
QWEN_CHAT_TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"

# Order is part of the decoder target contract; do not replace with a set.
FEEDBACK_FIELDS = (
    "holistic",
    "content_1",
    "content_2",
    "content_3",
    "organization_1",
    "organization_2",
    "expression_1",
    "expression_2",
    "task_1",
)
SCORE_FIELDS = ("content", "organization", "expression", "average")
_ANALYTIC_BY_COMPONENT = {
    "content": ("content_1", "content_2", "content_3"),
    "organization": ("organization_1", "organization_2"),
    "expression": ("expression_1", "expression_2"),
}
_WS = re.compile(r"\s+")


class HumanFeedbackDataError(DataContractError):
    """Raised if a source archive violates the frozen human-feedback contract."""


class TokenizerLike(Protocol):
    def __call__(self, text: str, *, add_special_tokens: bool = False) -> Mapping[str, Any]: ...

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool = False, add_generation_prompt: bool = False
    ) -> str: ...


@dataclass(frozen=True)
class HumanFeedbackRecord:
    """Validated restricted record; never serialize this to tracked files."""

    id: str
    prompt: str
    essay: str
    scores: Mapping[str, Decimal]
    feedback: Mapping[str, str]
    source: str
    subject: str
    question_id: str
    group_hash: str


@dataclass(frozen=True)
class ArchiveInput:
    source: str
    path: Path
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class PreparedHumanFeedbackData:
    selection_train: tuple[HumanFeedbackRecord, ...]
    selection_dev: tuple[HumanFeedbackRecord, ...]
    refit_train: tuple[HumanFeedbackRecord, ...]
    manifest: Mapping[str, Any]


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HumanFeedbackDataError(f"{field} must be a nonblank string")
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HumanFeedbackDataError(f"{field} must be an object")
    return value


def _decimal_score(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise HumanFeedbackDataError(f"{field} score must be numeric")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise HumanFeedbackDataError(f"{field} score is invalid") from exc
    if not decimal.is_finite() or not Decimal("1") <= decimal <= Decimal("5"):
        raise HumanFeedbackDataError(f"{field} score must be finite and in [1, 5]")
    return decimal


def _two_scores(value: Any, field: str) -> tuple[Decimal, Decimal]:
    if not isinstance(value, list) or len(value) != 2:
        raise HumanFeedbackDataError(f"{field}.score must contain exactly two rater values")
    return (_decimal_score(value[0], field), _decimal_score(value[1], field))


def _mean(values: Iterable[Decimal]) -> Decimal:
    values = tuple(values)
    if not values:
        raise HumanFeedbackDataError("cannot calculate an empty score mean")
    return sum(values, Decimal("0")) / Decimal(len(values))


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def derive_scores(analytic: Mapping[str, Any]) -> dict[str, Decimal]:
    """Aggregate two-rater analytic labels exactly, then round emitted targets."""
    ratings: dict[str, tuple[Decimal, Decimal]] = {}
    for field in FEEDBACK_FIELDS[1:]:
        item = _mapping(analytic.get(field), f"analytic.{field}")
        ratings[field] = _two_scores(item.get("score"), f"analytic.{field}")
    components = {
        component: _mean(score for field in fields for score in ratings[field])
        for component, fields in _ANALYTIC_BY_COMPONENT.items()
    }
    # This is intentionally based on unrounded component means.
    average = _mean(components.values())
    return {**{field: _quantize(components[field]) for field in _ANALYTIC_BY_COMPONENT}, "average": _quantize(average)}


def _feedback_from_personal(personal: Mapping[str, Any]) -> dict[str, str]:
    holistic = _mapping(personal.get("holistic"), "personal.holistic")
    analytic = _mapping(personal.get("analytic"), "personal.analytic")
    feedback: dict[str, str] = {"holistic": _nonblank(holistic.get("feedback"), "holistic.feedback")}
    for field in FEEDBACK_FIELDS[1:]:
        item = _mapping(analytic.get(field), f"analytic.{field}")
        feedback[field] = _nonblank(item.get("feedback"), f"analytic.{field}.feedback")
    return feedback


def parse_label_record(raw: Any, source: str) -> HumanFeedbackRecord:
    """Validate one descriptive/argumentative AI-Hub Training label record."""
    root = _mapping(raw, "label")
    question = _mapping(root.get("essay_question"), "essay_question")
    answer = _mapping(root.get("essay_answer"), "essay_answer")
    personal = _mapping(_mapping(root.get("score"), "score").get("personal"), "score.personal")
    question_id = _nonblank(question.get("id"), "essay_question.id")
    prompt = _nonblank(question.get("prompt"), "essay_question.prompt")
    subject = _nonblank(question.get("subject"), "essay_question.subject")
    essay = _nonblank(answer.get("text"), "essay_answer.text")
    # Retain the source namespace to avoid cross-corpus answer-ID collisions.
    answer_id = answer.get("id")
    if isinstance(answer_id, bool) or answer_id is None or not str(answer_id).strip():
        raise HumanFeedbackDataError("essay_answer.id must be nonblank")
    analytic = _mapping(personal.get("analytic"), "score.personal.analytic")
    scores = derive_scores(analytic)
    feedback = _feedback_from_personal(personal)
    normalized = normalize_prompt(prompt)
    return HumanFeedbackRecord(
        id=f"{source}:{question_id}:{answer_id}",
        prompt=prompt,
        essay=essay,
        scores=scores,
        feedback=feedback,
        source=source,
        subject=subject,
        question_id=question_id,
        group_hash=stable_hash(normalized),
    )


def discover_training_archives(raw_root: str | Path) -> tuple[ArchiveInput, ...]:
    """Discover only TL archives in upstream Training directories (never Validation)."""
    root = Path(raw_root)
    datasets = {
        "descriptive": root / "025_descriptive_writing_evaluation",
        "argumentative": root / "026_argumentative_writing_evaluation",
    }
    archives: list[ArchiveInput] = []
    for source, dataset_root in datasets.items():
        if not dataset_root.is_dir():
            raise HumanFeedbackDataError(f"missing source dataset root: {dataset_root}")
        candidates = sorted(dataset_root.rglob("TL_*.zip"))
        if not candidates:
            raise HumanFeedbackDataError(f"no Training TL archives found under {dataset_root}")
        for path in candidates:
            parts = set(path.parts)
            if "Training" not in parts or "Validation" in parts or not path.name.startswith("TL_"):
                raise HumanFeedbackDataError(f"non-Training archive selected: {path}")
            archives.append(ArchiveInput(source, path, path.relative_to(root).as_posix(), file_sha256(path)))
    return tuple(sorted(archives, key=lambda item: (item.source, item.relative_path)))


def iter_training_records(archives: Sequence[ArchiveInput]) -> Iterable[HumanFeedbackRecord]:
    """Read only JSON labels in the ordered TL input archives."""
    for archive in archives:
        try:
            with ZipFile(archive.path) as zf:
                members = sorted(name for name in zf.namelist() if name.endswith(".json") and not name.endswith("/"))
                if not members:
                    raise HumanFeedbackDataError(f"TL archive contains no JSON labels: {archive.relative_path}")
                for member in members:
                    try:
                        raw = json.loads(zf.read(member).decode("utf-8-sig"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise HumanFeedbackDataError(f"invalid JSON label in {archive.relative_path}") from exc
                    yield parse_label_record(raw, archive.source)
        except HumanFeedbackDataError:
            raise
        except Exception as exc:  # zip corruption and I/O are contract failures.
            raise HumanFeedbackDataError(f"cannot read TL archive: {archive.relative_path}") from exc


def render_score_target(scores: Mapping[str, Decimal]) -> str:
    """Render exact, ordered score JSON with two decimal places."""
    if tuple(scores) != SCORE_FIELDS:
        raise HumanFeedbackDataError("scores must use the exact four-field ordering")
    parts = []
    for field in SCORE_FIELDS:
        value = scores[field]
        if not isinstance(value, Decimal) or value != _quantize(value):
            raise HumanFeedbackDataError("emitted scores must be two-decimal Decimal values")
        parts.append(f'"{field}":{value:.2f}')
    return "{" + ",".join(parts) + "}"


def render_human_feedback_target(record: HumanFeedbackRecord) -> str:
    """Render the exact ordered assistant target; feedback text is never altered."""
    if tuple(record.feedback) != FEEDBACK_FIELDS:
        raise HumanFeedbackDataError("feedback must use the exact nine-field ordering")
    feedback_json = json.dumps(dict(record.feedback), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return '{"feedback":' + feedback_json + ',"scores":' + render_score_target(record.scores) + "}"


def human_feedback_target_token_count(tokenizer: TokenizerLike, record: HumanFeedbackRecord) -> int:
    """Validate the complete assistant render and return raw assistant-token count."""
    target = render_human_feedback_target(record)
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "입력"}, {"role": "assistant", "content": target}],
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = tokenizer(target, add_special_tokens=False)
    except Exception as exc:
        raise HumanFeedbackDataError("pinned tokenizer cannot render/tokenize assistant target") from exc
    if not isinstance(rendered, str) or target not in rendered:
        raise HumanFeedbackDataError("pinned tokenizer did not render a complete assistant target")
    token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if not isinstance(token_ids, list) or any(not isinstance(token, int) for token in token_ids):
        raise HumanFeedbackDataError("tokenizer must return integer input_ids for assistant target")
    return len(token_ids)


def _validate_source_consistency(records: Sequence[HumanFeedbackRecord]) -> None:
    seen_ids: set[str] = set()
    question_normalizations: dict[tuple[str, str], str] = {}
    for record in records:
        if record.id in seen_ids:
            raise HumanFeedbackDataError("duplicate canonical source record id")
        seen_ids.add(record.id)
        key = (record.source, record.question_id)
        normalized = normalize_prompt(record.prompt)
        old = question_normalizations.setdefault(key, normalized)
        if old != normalized:
            raise HumanFeedbackDataError("one (source, question_id) maps to inconsistent normalized questions")


def _primary_strata(records: Sequence[HumanFeedbackRecord]) -> dict[str, tuple[str, str]]:
    candidates: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        candidates[record.group_hash].add((record.source, record.subject))
    return {group: min(strata) for group, strata in candidates.items()}


def _choose_groups_by_dp(groups: Sequence[tuple[str, int]], total: int) -> tuple[str, ...]:
    """Closest-count subset; for ties use lexicographically sorted group hashes."""
    # sum -> lexicographically least sorted sequence attaining that sum
    states: dict[int, tuple[str, ...]] = {0: ()}
    for group_hash, count in sorted(groups):
        if count <= 0:
            raise HumanFeedbackDataError("group record count must be positive")
        additions: dict[int, tuple[str, ...]] = {}
        for subtotal, selected in states.items():
            candidate_sum = subtotal + count
            candidate = tuple(sorted((*selected, group_hash)))
            prior = states.get(candidate_sum, additions.get(candidate_sum))
            if prior is None or candidate < prior:
                additions[candidate_sum] = candidate
        for subtotal, candidate in additions.items():
            prior = states.get(subtotal)
            if prior is None or candidate < prior:
                states[subtotal] = candidate
    target = Decimal(total) * DEV_FRACTION
    group_count = len(groups)
    # A development partition must contain a whole group.  When a primary
    # stratum has multiple groups, preserve at least one for optimization too.
    candidates = [
        selected
        for selected in states.values()
        if selected and (group_count == 1 or len(selected) < group_count)
    ]
    if not candidates:
        raise HumanFeedbackDataError("primary stratum cannot yield a development subset")
    group_sizes = dict(groups)
    return min(candidates, key=lambda selected: (abs(Decimal(sum(group_sizes[group] for group in selected)) - target), selected))


def split_records(records: Sequence[HumanFeedbackRecord]) -> tuple[tuple[HumanFeedbackRecord, ...], tuple[HumanFeedbackRecord, ...], dict[str, Any]]:
    """Create a deterministic group-disjoint 80:20 development split."""
    if not records:
        raise HumanFeedbackDataError("cannot split an empty eligible dataset")
    group_rows: dict[str, list[HumanFeedbackRecord]] = defaultdict(list)
    for record in records:
        group_rows[record.group_hash].append(record)
    if len(group_rows) < 2:
        raise HumanFeedbackDataError("at least two normalized-question groups are required")
    primary = _primary_strata(records)
    by_primary: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for group_hash, group in group_rows.items():
        by_primary[primary[group_hash]].append((group_hash, len(group)))

    selected_groups: set[str] = set()
    algorithm_strata: dict[str, Any] = {}
    for stratum, groups in sorted(by_primary.items()):
        total = sum(count for _, count in groups)
        selected = _choose_groups_by_dp(groups, total)
        selected_groups.update(selected)
        algorithm_strata[f"{stratum[0]}::{stratum[1]}"] = {
            "group_count": len(groups),
            "record_count": total,
            "target_dev_records": str(Decimal(total) * DEV_FRACTION),
            "selected_dev_records": sum(count for group, count in groups if group in set(selected)),
            "selected_group_hashes": list(selected),
        }
    dev = tuple(record for record in records if record.group_hash in selected_groups)
    train = tuple(record for record in records if record.group_hash not in selected_groups)
    if not dev or not train:
        raise HumanFeedbackDataError("group split produced an empty partition")
    if {record.group_hash for record in dev} & {record.group_hash for record in train}:
        raise HumanFeedbackDataError("normalized-question group leaked across split")

    original_counts: dict[str, dict[str, int]] = {}
    for record in records:
        key = f"{record.source}::{record.subject}"
        bucket = original_counts.setdefault(key, {"eligible_records": 0, "selection_train_records": 0, "selection_dev_records": 0})
        bucket["eligible_records"] += 1
        bucket["selection_dev_records" if record.group_hash in selected_groups else "selection_train_records"] += 1
    collisions = sum(1 for group, strata in ((group, {(row.source, row.subject) for row in rows}) for group, rows in group_rows.items()) if len(strata) > 1)
    audit = {
        "selection_algorithm": "per_primary_dataset_subject_exact_record_count_dp_then_lexicographic_group_hash_sequence",
        "requested_dev_fraction": str(DEV_FRACTION),
        "normalized_question_groups": len(group_rows),
        "cross_corpus_or_stratum_group_collisions": collisions,
        "primary_strata": algorithm_strata,
        "per_original_stratum_records": dict(sorted(original_counts.items())),
        "selection_train_group_hashes": sorted(set(group_rows) - selected_groups),
        "selection_dev_group_hashes": sorted(selected_groups),
    }
    return train, dev, audit


def _safe_row(record: HumanFeedbackRecord) -> dict[str, Any]:
    """This row is safe only for the ignored processed-data destination."""
    return {
        "id": record.id,
        "prompt": record.prompt,
        "essay": record.essay,
        "score": {field: float(record.scores[field]) for field in SCORE_FIELDS},
        "feedback": {field: record.feedback[field] for field in FEEDBACK_FIELDS},
    }


def _jsonl_sha256(path: Path) -> str:
    return file_sha256(path)


def _write_jsonl(path: Path, records: Sequence[HumanFeedbackRecord]) -> dict[str, Any]:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_safe_row(record), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    return {"records": len(records), "sha256": _jsonl_sha256(path)}


def _assert_ignored_processed_destination(output_root: Path) -> None:
    # A path-level guard remains effective even if a future .gitignore edit is wrong.
    if "data" not in output_root.parts or "processed" not in output_root.parts:
        raise HumanFeedbackDataError("restricted records may only be written under data/processed")


def prepare_human_feedback_data(
    archives: Sequence[ArchiveInput], tokenizer: TokenizerLike, *, expected_source_records: int | None = 48_016
) -> PreparedHumanFeedbackData:
    """Validate, commonly filter, and split all upstream Training label rows."""
    records = tuple(iter_training_records(archives))
    if expected_source_records is not None and len(records) != expected_source_records:
        raise HumanFeedbackDataError(f"expected {expected_source_records} Training label records, found {len(records)}")
    _validate_source_consistency(records)
    eligible: list[HumanFeedbackRecord] = []
    rejected_over_budget = 0
    for record in records:
        if human_feedback_target_token_count(tokenizer, record) > TARGET_TOKEN_CAP:
            rejected_over_budget += 1
        else:
            eligible.append(record)
    train, dev, split_audit = split_records(eligible)
    archive_fingerprint = stable_hash("\n".join(f"{item.source}\t{item.relative_path}\t{item.sha256}" for item in archives))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "source": {
            "included_corpora": ["descriptive", "argumentative"],
            "included_split": "AI-Hub upstream Training TL archives only",
            "excluded": ["essay corpus", "AI-Hub upstream Validation", "frozen eval/validation.jsonl"],
            "archive_counts_by_source": dict(sorted(Counter(item.source for item in archives).items())),
            # Checksums are retained without raw archive paths or names.  Their
            # ordered source/path fingerprint above remains reproducible from
            # the local discovery procedure without exposing raw locations.
            "archive_sha256_by_source": {
                source: [item.sha256 for item in archives if item.source == source]
                for source in sorted({item.source for item in archives})
            },
            "archive_list_sha256": archive_fingerprint,
            "source_records": len(records),
        },
        "eligibility": {
            "common_to_all_four_experiments": True,
            "assistant_target_token_cap": TARGET_TOKEN_CAP,
            "over_budget_rejections": rejected_over_budget,
            "eligible_records": len(eligible),
            "eligible_record_id_sha256": stable_hash("\n".join(sorted(record.id for record in eligible))),
            "tokenizer": {
                "id": QWEN_MODEL_ID,
                "revision": QWEN_REVISION,
                "chat_template_sha256": QWEN_CHAT_TEMPLATE_SHA256,
                "assistant_target_render": "complete_assistant_chat_template_then_raw_assistant_token_count",
            },
        },
        "score_contract": {
            "analytic_raters_per_criterion": 2,
            "analytic_score_range": [1, 5],
            "component_aggregation": "mean of all constituent analytic rater scores using Decimal",
            "average_aggregation": "mean of unrounded content, organization, expression component means using Decimal",
            "emitted_quantization": "0.01 ROUND_HALF_UP",
            "fields": list(SCORE_FIELDS),
        },
        "feedback_contract": {"ordered_fields": list(FEEDBACK_FIELDS), "task_1_used_for_score": False},
        "split": split_audit,
    }
    return PreparedHumanFeedbackData(train, dev, tuple(eligible), manifest)


def write_prepared_dataset(prepared: PreparedHumanFeedbackData, output_root: str | Path, manifest_path: str | Path) -> Mapping[str, Any]:
    """Write restricted JSONL to ignored storage and an aggregate-only manifest."""
    output = Path(output_root)
    manifest_file = Path(manifest_path)
    _assert_ignored_processed_destination(output)
    if output.exists():
        raise HumanFeedbackDataError(f"refusing to overwrite processed dataset: {output}")
    if manifest_file.exists():
        raise HumanFeedbackDataError(f"refusing to overwrite manifest: {manifest_file}")
    output.mkdir(parents=True)
    try:
        files = {
            "selection_train": {"filename": "selection_train.jsonl", **_write_jsonl(output / "selection_train.jsonl", prepared.selection_train)},
            "selection_dev": {"filename": "selection_dev.jsonl", **_write_jsonl(output / "selection_dev.jsonl", prepared.selection_dev)},
            "refit_train": {"filename": "refit_train.jsonl", **_write_jsonl(output / "refit_train.jsonl", prepared.refit_train)},
        }
        # Canonical consumers use record_count; keep no parallel count spelling.
        for details in files.values():
            details["record_count"] = details.pop("records")
        manifest = dict(prepared.manifest)
        manifest["files"] = files
        # Defense-in-depth against accidentally adding restricted fields to the manifest.
        rendered = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        forbidden = ("\"prompt\"", "\"essay\"", "\"record_id\"")
        if any(field in rendered for field in forbidden):
            raise HumanFeedbackDataError("aggregate manifest contains a restricted row field")
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest
    except Exception:
        # Do not remove possibly useful restricted data automatically.  The user can inspect it.
        raise
