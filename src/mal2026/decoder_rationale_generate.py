"""Score-blind frozen-teacher generation of train-only synthetic rationales.

This runner deliberately has a narrower interface than decoder SFT.  It never
reads a score field, never receives IDs in the teacher prompt, and writes only
restricted local artifacts under one ignored ``outputs/runs/<run-id>`` root.
Synthetic evidence is not a human rationale or a faithful-explanation claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

from .decoder import (
    CANONICAL_TRAIN_SHA256,
    ContractError,
    require_canonical_dataset,
    require_immutable_revision,
    resolve_run_output_dir,
    template_sha256,
)
from .data_contract import DatasetRecord, load_and_validate_jsonl, split_prompt_groups, stable_hash
from .rationale import RationaleValidationError, validate_rationale_payload


TEACHER_SYSTEM_MESSAGE = (
    "당신은 한국어 글의 원문 근거를 추출하는 도우미입니다. "
    "요청한 JSON 이외의 텍스트를 출력하지 마세요."
)
TEACHER_USER_INSTRUCTION = (
    "과제와 학생 글을 읽고 CONTENT, ORGANIZATION, EXPRESSION 각각에 대해 "
    "글 원문의 quote와 그 문자 start/end, 관찰 observation을 JSON으로 만드세요. "
    "observation에는 숫자, 등급, 우열, 채점 관련 표현을 쓰지 마세요."
)
TEACHER_TEMPLATE_SHA256 = template_sha256(TEACHER_SYSTEM_MESSAGE, TEACHER_USER_INSTRUCTION)


@dataclass(frozen=True)
class TeacherRationaleConfig:
    run_id: str
    phase: str  # selection derives optimization-train; refit uses all canonical train.
    teacher_id: str
    teacher_revision: str
    tokenizer_revision: str
    train_path: str
    output_dir: str
    canonical_config_path: str
    train_sha256: str = CANONICAL_TRAIN_SHA256
    seed: int = 20260716
    max_new_tokens: int = 512
    max_retries: int = 2
    wandb_project: str = "mal2026-korean-writing-scoring"
    wandb_entity: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TeacherRationaleConfig":
        known = {field.name for field in fields(cls)}
        extra = sorted(set(raw) - known)
        if extra:
            raise ContractError(f"unknown synthetic-rationale config fields: {extra}")
        return cls(**dict(raw))

    def validate(self) -> None:
        if self.phase not in {"selection", "refit"}:
            raise ContractError("synthetic rationale phase must be selection or refit")
        require_immutable_revision(self.teacher_revision, "teacher_revision")
        require_immutable_revision(self.tokenizer_revision, "tokenizer_revision")
        require_canonical_dataset(self.train_path, "train", self.train_sha256)
        resolve_run_output_dir(self.run_id, self.output_dir)
        if self.max_new_tokens != 512 or self.max_retries != 2:
            raise ContractError("frozen teacher generation uses 512 new tokens and two retries")


def load_json_config(path: str) -> TeacherRationaleConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("unable to load synthetic-rationale JSON config") from exc
    if not isinstance(raw, Mapping):
        raise ContractError("synthetic-rationale config must be an object")
    config = TeacherRationaleConfig.from_mapping(raw)
    config.validate()
    _validate_canonical_contract(config)
    return config


def _validate_canonical_contract(config: TeacherRationaleConfig) -> tuple[dict[str, Any], str]:
    from .config import ConfigError, load_experiment_config

    try:
        contract, contract_hash = load_experiment_config(config.canonical_config_path)
    except ConfigError as exc:
        raise ContractError(f"invalid canonical rationale config: {exc}") from exc
    if contract["run_kind"] != "decoder-rationale-score":
        raise ContractError("teacher generation requires decoder-rationale-score canonical config")
    teacher = contract["teacher"]
    if (teacher["id"], teacher["revision"]) != (config.teacher_id, config.teacher_revision):
        raise ContractError("runtime teacher ID/revision does not match canonical config")
    if teacher["prompt_template_sha256"] != TEACHER_TEMPLATE_SHA256:
        raise ContractError("canonical teacher prompt hash does not match frozen score-blind template")
    if (teacher["seed"], teacher["max_new_tokens"], teacher["max_retries"]) != (config.seed, config.max_new_tokens, config.max_retries):
        raise ContractError("runtime generation settings do not match canonical config")
    return contract, contract_hash


def teacher_request(record: DatasetRecord) -> list[dict[str, str]]:
    """Construct the teacher request from prompt/essay only, never a full row."""
    # Do not refactor this to accept Mapping: DatasetRecord makes accidental
    # access to ``scores``, IDs, or split metadata visibly unnecessary here.
    user = (
        TEACHER_USER_INSTRUCTION
        + "\n<writing_prompt>\n" + record.prompt + "\n</writing_prompt>"
        + "\n<student_essay>\n" + record.essay + "\n</student_essay>"
    )
    return [{"role": "system", "content": TEACHER_SYSTEM_MESSAGE}, {"role": "user", "content": user}]


def _partition_records(config: TeacherRationaleConfig) -> tuple[Sequence[DatasetRecord], dict[str, Any]]:
    rows = load_and_validate_jsonl(config.train_path, expected_sha256=config.train_sha256)
    if config.phase == "selection":
        split = split_prompt_groups(rows, 0.10)
        return split.optimization_train, split.manifest
    return rows, {
        "schema_version": 1,
        "selection_algorithm": "refit_all_canonical_train_records",
        "optimization_train_records": len(rows),
        "development_records": 0,
        "optimization_record_id_sha256": stable_hash("\n".join(sorted(row.id for row in rows))),
    }


def _parse_teacher_output(text: str, essay: str) -> list[dict[str, Any]]:
    parsed = json.loads(text)
    result = validate_rationale_payload(parsed, essay=essay)
    if not result.nonempty_valid:
        raise RationaleValidationError("teacher response may not use empty rationale as a successful candidate")
    return parsed["rationale"]


def generate(config: TeacherRationaleConfig) -> None:
    """Generate, validate, and gate synthetic rationales with a frozen teacher."""
    config.validate()
    _, canonical_config_hash = _validate_canonical_contract(config)
    run_dir = resolve_run_output_dir(config.run_id, config.output_dir)
    if run_dir.exists():
        raise ContractError("refusing to overwrite synthetic rationale output")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_id, revision=config.tokenizer_revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.teacher_id, revision=config.teacher_revision, torch_dtype=torch.bfloat16)
    model.eval()
    records, partition = _partition_records(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact = run_dir / "synthetic-rationales.jsonl"
    valid_count = 0
    failed_count = 0
    with artifact.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for record in records:
            # ``teacher_request`` is the only model-input construction and has
            # no score, id, document_id, prompt_num, or split argument.
            messages = teacher_request(record)
            prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(model.device)
            rationale: list[dict[str, Any]] = []
            for _attempt in range(config.max_retries + 1):
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=config.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
                try:
                    rationale = _parse_teacher_output(text, record.essay)
                    valid_count += 1
                    break
                except (json.JSONDecodeError, RationaleValidationError, TypeError, KeyError, ValueError):
                    rationale = []
            if not rationale:
                failed_count += 1
            # ID is unavoidable for the local one-to-one SFT join, but no text,
            # scores, teacher output, or retry error is retained outside this
            # ignored artifact.
            handle.write(json.dumps({"id": record.id, "rationale": rationale}, ensure_ascii=False, separators=(",", ":")) + "\n")
    rate = valid_count / len(records)
    provenance = {
        "schema_version": 1,
        "status": "passed" if rate >= 0.85 else "failed",
        "source_train_sha256": config.train_sha256,
        "partition_record_id_sha256": partition["optimization_record_id_sha256"],
        "partition_record_count": len(records),
        "nonempty_valid_count": valid_count,
        "invalid_or_exhausted_count": failed_count,
        "nonempty_valid_rate": rate,
        "nonempty_valid_rate_gate": 0.85,
        "teacher_id": config.teacher_id,
        "teacher_revision": config.teacher_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "teacher_template_sha256": TEACHER_TEMPLATE_SHA256,
        "generation_do_sample": False,
        "generation_max_new_tokens": config.max_new_tokens,
        "generation_max_retries": config.max_retries,
        "seed": config.seed,
        "canonical_config_hash": canonical_config_hash,
        "artifact_sha256": _file_sha256(artifact),
    }
    (run_dir / "rationale_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rate < 0.85:
        _write_manifest(run_dir, config, canonical_config_hash, provenance)
        raise ContractError("synthetic rationale nonempty-valid rate is below the frozen 85% no-go gate")
    _write_manifest(run_dir, config, canonical_config_hash, provenance)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(run_dir: Path, config: TeacherRationaleConfig, canonical_config_hash: str, provenance: Mapping[str, Any]) -> None:
    from .provenance import aggregate_only_payload, build_run_manifest

    config_hash = hashlib.sha256(json.dumps(asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    manifest = build_run_manifest(
        run_id=config.run_id,
        config_hash=config_hash,
        data_contract={
            "train_sha256": config.train_sha256,
            "partition_record_count": provenance["partition_record_count"],
            "partition_record_id_sha256": provenance["partition_record_id_sha256"],
        },
        command=" ".join(sys.argv),
        output_path=str(run_dir),
        extra={
            "canonical_config_hash": canonical_config_hash,
            "phase": config.phase,
            "teacher_revision": config.teacher_revision,
            "teacher_template_sha256": TEACHER_TEMPLATE_SHA256,
            "nonempty_valid_count": provenance["nonempty_valid_count"],
            "invalid_or_exhausted_count": provenance["invalid_or_exhausted_count"],
            "nonempty_valid_rate": provenance["nonempty_valid_rate"],
            "nonempty_valid_rate_gate": 0.85,
            "status": provenance["status"],
            "deviations": "none",
        },
    )
    (run_dir / "run_manifest.json").write_text(json.dumps(aggregate_only_payload(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="generate score-blind synthetic rationale artifacts for decoder SFT")
    parser.add_argument("--config", required=True, help="non-secret frozen-teacher generation JSON config")
    args = parser.parse_args(argv)
    generate(load_json_config(args.config))


if __name__ == "__main__":  # pragma: no cover
    main()
