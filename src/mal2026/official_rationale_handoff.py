"""Fail-closed selection and generation handoff for the final rationale model.

Selection is permitted only after the fixed proxy judge's directional and
prompt-injection gates pass and every declared candidate has a completed,
hash-bound evaluation under that same judge.  The generator is conditioned
only on bootstrap-emitted integer predictions, never human/reference scores.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
AXES = ("content", "organization", "expression")
METHODS = ("official_sft", "aihub_sft", "dpo", "grpo")
STRUCTURES = ("bundle", "axis_triplet")
JUDGE_PROMPT_KIND = "single_frozen_public-spec-aligned_proxy_not_verbatim_organizer_prompt"


class RationaleHandoffError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RationaleHandoffError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RationaleHandoffError(f"{label} is unreadable") from exc
    need(isinstance(value, dict), f"{label} must be an object")
    return value


def canonical_sha(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def candidate_identity(candidate: Mapping[str, Any]) -> dict[str, Any]:
    adapters = candidate["adapters"]
    return {
        "key": candidate["key"], "method": candidate["method"], "structure": candidate["structure"],
        "model_id": candidate["model_id"], "model_revision": candidate["model_revision"],
        "model_config_sha256": candidate["model_config_sha256"], "model_binding_sha256": candidate["model_binding_sha256"],
        "adapters": {task: {
            "adapter_config_sha256": value["adapter_config_sha256"],
            "adapter_model_sha256": value["adapter_model_sha256"],
            "training_completion_sha256": value["training_completion_sha256"],
        } for task, value in sorted(adapters.items())},
    }


def candidate_identity_sha256(candidate: Mapping[str, Any]) -> str:
    return canonical_sha(candidate_identity(candidate))


def select_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Frozen Q4 ranking: macro, worst cell, parse rate, then key."""
    need(bool(rows), "candidate evaluations are empty")
    return min(rows, key=lambda row: (
        -float(row["macro_mean"]), -float(row["worst_cell"]),
        -float(row["strict_parse_rate"]), str(row["key"]),
    ))


class HandoffConfig:
    """Strict nested config kept as mappings to support a variable candidate set."""

    TOP_LEVEL = {
        "schema_version", "run_id", "bootstrap_selection_path", "bootstrap_selection_sha256",
        "judge", "candidates", "selection_output_path", "restricted_output_root", "runtime_output_root",
        "vllm", "generation",
    }

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self.raw = dict(raw)
        self.validate_structure()

    @classmethod
    def from_json(cls, path: Path) -> "HandoffConfig":
        return cls(read_json(path, "rationale handoff config"))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def candidates(self) -> list[dict[str, Any]]:
        return self.raw["candidates"]

    def validate_structure(self) -> None:
        need(set(self.raw) == self.TOP_LEVEL, "handoff config fields differ")
        need(self.raw["schema_version"] == "mal2026-official-rationale-handoff-config-v1", "handoff config schema differs")
        need(self.raw["run_id"] == "official-rationale-handoff-v1-20260727-001", "handoff run identity differs")
        judge = self.raw["judge"]
        need(isinstance(judge, dict) and set(judge) == {"contract_path", "contract_sha256", "model_path", "model_sha256", "directional_gate_path", "directional_gate_sha256", "injection_gate_path", "injection_gate_sha256", "prompt_kind", "repeats_per_validation_record"}, "judge config differs")
        need(judge["prompt_kind"] == JUDGE_PROMPT_KIND and judge["repeats_per_validation_record"] == 10, "fixed judge protocol differs")
        candidates = self.raw["candidates"]
        need(isinstance(candidates, list) and candidates, "candidate declarations are empty")
        keys: set[str] = set(); methods: set[str] = set()
        for candidate in candidates:
            expected = {"key", "method", "structure", "model_id", "model_revision", "model_path", "model_config_sha256", "model_binding_path", "model_binding_sha256", "adapters", "evaluation_path", "evaluation_sha256"}
            need(isinstance(candidate, dict) and set(candidate) == expected, "candidate declaration fields differ")
            need(isinstance(candidate["key"], str) and candidate["key"] not in keys, "candidate key differs")
            keys.add(candidate["key"]); methods.add(candidate["method"])
            need(candidate["method"] in METHODS and candidate["structure"] in STRUCTURES, "candidate method/structure differs")
            tasks = {"bundle"} if candidate["structure"] == "bundle" else set(AXES)
            need(isinstance(candidate["adapters"], dict) and set(candidate["adapters"]) == tasks, "candidate adapters do not match its structure")
            for adapter in candidate["adapters"].values():
                need(isinstance(adapter, dict) and set(adapter) == {"path", "adapter_config_sha256", "adapter_model_sha256", "training_completion_path", "training_completion_sha256"}, "adapter declaration differs")
        need(methods == set(METHODS), "official SFT, AI-Hub SFT, DPO, and GRPO candidates must all be declared")
        generation = self.raw["generation"]
        need(generation == {"temperature": 0.0, "top_p": 1.0, "seed": 42, "train_records": 2000, "validation_records": 400, "smoke_records": 1, "max_inflight": 128}, "generation contract differs")
        vllm = self.raw["vllm"]
        need(isinstance(vllm, dict) and vllm.get("gpu_scope") == [0, 1, 2, 3] and vllm.get("smoke_gpu_scope") == [0] and vllm.get("tensor_parallel_size_full") == 4 and vllm.get("enforce_eager") is False, "vLLM resource contract differs")
        restricted = (ROOT / "data" / "processed" / "restricted").resolve()
        need(Path(self.raw["restricted_output_root"]).resolve().is_relative_to(restricted), "handoff output must be restricted")
        need(Path(self.raw["runtime_output_root"]).resolve().is_relative_to((ROOT / "outputs").resolve()), "runtime output must be ignored")

    def validate_dependencies(self) -> dict[str, Any]:
        bootstrap_path = Path(self.raw["bootstrap_selection_path"])
        need(bootstrap_path.is_file() and file_sha256(bootstrap_path) == self.raw["bootstrap_selection_sha256"], "bootstrap selection SHA differs")
        bootstrap = read_json(bootstrap_path, "bootstrap selection")
        need(bootstrap.get("status") == "stage_a_completed" and bootstrap.get("selection_source") == "train_internal_dev_only" and bootstrap.get("canonical_validation_used_for_selection") is False, "bootstrap selection contract differs")
        selected_scores = bootstrap.get("selected_score_files")
        need(isinstance(selected_scores, dict), "bootstrap score binding differs")
        restricted = (ROOT / "data" / "processed" / "restricted").resolve()
        for split, count in (("train", 2000), ("validation", 400)):
            path = Path(selected_scores.get(f"{split}_path", ""))
            need(path.is_file() and path.resolve().is_relative_to(restricted) and file_sha256(path) == selected_scores.get(f"{split}_sha256") and selected_scores.get(f"{split}_records") == count, f"bootstrap {split} score dependency differs")

        judge = self.raw["judge"]
        need(Path(judge["contract_path"]).resolve() == (ROOT / "src" / "mal2026" / "official_writing_contract.py").resolve(), "fixed judge contract path differs")
        judge_model = Path(judge["model_path"])
        need(judge_model.is_file() and not judge_model.is_symlink() and file_sha256(judge_model) == judge["model_sha256"], "fixed Q4 judge model SHA differs")
        for key in ("contract", "directional_gate", "injection_gate"):
            path = Path(judge[f"{key}_path"])
            need(path.is_file() and file_sha256(path) == judge[f"{key}_sha256"], f"judge {key} SHA differs")
        directional = read_json(Path(judge["directional_gate_path"]), "directional judge gate")
        injection = read_json(Path(judge["injection_gate_path"]), "injection judge gate")
        need(directional.get("schema_version") == "mal2026-official-proxy-judge-contrastive-gate-v1" and directional.get("status") == "passed" and directional.get("rl_with_this_proxy_judge_allowed") is True, "directional judge gate has not passed")
        need(injection.get("schema_version") == "mal2026-official-proxy-judge-rl-safety-gate-v1" and injection.get("status") == "passed" and injection.get("directional_contrastive_gate_passed") is True and injection.get("prompt_injection_gate_passed") is True and injection.get("rl_allowed") is True, "combined injection judge gate has not passed")

        evaluated: list[dict[str, Any]] = []
        for candidate in self.candidates:
            model = Path(candidate["model_path"])
            need(model.is_dir() and not model.is_symlink() and file_sha256(model / "config.json") == candidate["model_config_sha256"], f"candidate model differs: {candidate['key']}")
            model_binding = Path(candidate["model_binding_path"])
            need(model_binding.is_file() and not model_binding.is_symlink() and file_sha256(model_binding) == candidate["model_binding_sha256"], f"candidate model binding differs: {candidate['key']}")
            for task, adapter in candidate["adapters"].items():
                root = Path(adapter["path"]); completion = Path(adapter["training_completion_path"])
                need(root.is_dir() and not root.is_symlink() and completion.is_file(), f"candidate adapter unavailable: {candidate['key']}/{task}")
                need(file_sha256(root / "adapter_config.json") == adapter["adapter_config_sha256"], f"candidate adapter config differs: {candidate['key']}/{task}")
                adapter_config = read_json(root / "adapter_config.json", f"candidate adapter config {candidate['key']}/{task}")
                need(type(adapter_config.get("r")) is int and 0 < adapter_config["r"] <= self.raw["vllm"]["max_lora_rank"], f"candidate adapter rank differs: {candidate['key']}/{task}")
                model_file = root / "adapter_model.safetensors"
                need(model_file.is_file() and file_sha256(model_file) == adapter["adapter_model_sha256"], f"candidate adapter state differs: {candidate['key']}/{task}")
                need(file_sha256(completion) == adapter["training_completion_sha256"], f"candidate training completion differs: {candidate['key']}/{task}")
                completion_value = read_json(completion, f"candidate training completion {candidate['key']}/{task}")
                need(completion_value.get("status") == "completed", f"candidate training is incomplete: {candidate['key']}/{task}")
            evaluation_path = Path(candidate["evaluation_path"])
            need(evaluation_path.is_file() and file_sha256(evaluation_path) == candidate["evaluation_sha256"], f"candidate evaluation incomplete: {candidate['key']}")
            evaluation = read_json(evaluation_path, f"candidate evaluation {candidate['key']}")
            identity_sha = candidate_identity_sha256(candidate)
            need(evaluation.get("schema_version") == "mal2026-official-rationale-candidate-evaluation-v1" and evaluation.get("status") == "completed", "candidate evaluation status differs")
            need(evaluation.get("candidate_key") == candidate["key"] and evaluation.get("candidate_identity_sha256") == identity_sha, "candidate evaluation identity differs")
            need(evaluation.get("judge_contract_sha256") == judge["contract_sha256"] and evaluation.get("judge_model_sha256") == judge["model_sha256"] and evaluation.get("judge_prompt_kind") == judge["prompt_kind"], "candidate fixed judge differs")
            need(evaluation.get("validation_records") == 400 and evaluation.get("repeats_per_record") == judge["repeats_per_validation_record"], "candidate repeated validation protocol differs")
            metrics = evaluation.get("metrics")
            need(isinstance(metrics, dict) and set(metrics) == {"macro_mean", "worst_cell", "strict_parse_rate"}, "candidate evaluation metrics differ")
            need(all(isinstance(metrics[key], (int, float)) and math.isfinite(float(metrics[key])) for key in metrics), "candidate evaluation metric is nonnumeric")
            need(1 <= float(metrics["macro_mean"]) <= 5 and 1 <= float(metrics["worst_cell"]) <= 5 and 0 <= float(metrics["strict_parse_rate"]) <= 1, "candidate evaluation metric range differs")
            evaluated.append({"key": candidate["key"], "method": candidate["method"], "structure": candidate["structure"], **metrics, "candidate_identity_sha256": identity_sha, "evaluation_sha256": candidate["evaluation_sha256"]})
        winner = select_candidate(evaluated)
        return {"bootstrap": bootstrap, "evaluated": evaluated, "winner": winner, "candidate": next(value for value in self.candidates if value["key"] == winner["key"])}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            need(bool(line.strip()), f"blank JSONL line {line_number}")
            value = json.loads(line)
            need(isinstance(value, dict), f"JSONL line {line_number} differs")
            yield value


def convert_bootstrap_scores(source: Path, output: Path, expected: int, split: str) -> str:
    """Convert matrix score schema to the existing generator's strict schema."""
    need(not output.exists(), "adapted score output already exists")
    records = list(iter_jsonl(source))
    need(len(records) == expected, "bootstrap score count differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for raw in records:
            need(set(raw) == {"source_id", "split", "arm", "scores"} and raw["split"] == split, "bootstrap score schema differs")
            scores = raw["scores"]
            need(isinstance(scores, dict) and set(scores) == set(AXES) and all(type(scores[axis]) is int and 1 <= scores[axis] <= 5 for axis in AXES), "bootstrap score is not an official integer")
            handle.write(json.dumps({"source_id": raw["source_id"], "emitted_integer_prediction": scores}, ensure_ascii=False, sort_keys=True) + "\n")
    return file_sha256(output)


def combine_rationales(inputs: Mapping[str, Path], output: Path, expected: int, structure: str) -> str:
    """Normalize bundle or exact-axis generator outputs to one strict JSONL."""
    need(structure in STRUCTURES and not output.exists(), "rationale combine request differs")
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for task, path in inputs.items():
        rows = {str(row["source_id"]): row for row in iter_jsonl(path)}
        need(len(rows) == expected, f"rationale population differs: {task}")
        by_task[task] = rows
    tasks = {"bundle"} if structure == "bundle" else set(AXES)
    need(set(by_task) == tasks, "rationale tasks differ from winner structure")
    ids = set(next(iter(by_task.values())))
    need(all(set(rows) == ids for rows in by_task.values()), "axis rationale IDs differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for source_id in sorted(ids):
            if structure == "bundle":
                rationales = by_task["bundle"][source_id].get("rationales")
            else:
                rationales = {axis: by_task[axis][source_id].get("rationales", {}).get(axis) for axis in AXES}
            need(isinstance(rationales, dict) and set(rationales) == set(AXES) and all(isinstance(rationales[axis], str) and rationales[axis].strip() for axis in AXES), "strict rationale-only output differs")
            handle.write(json.dumps({"source_id": source_id, "rationales": rationales}, ensure_ascii=False, sort_keys=True) + "\n")
    return file_sha256(output)
