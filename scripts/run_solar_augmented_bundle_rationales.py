#!/usr/bin/env python3
"""Generate bundle-only rationales for all 6,000 Solar augmented essays."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.augmented_bundle_rationale import (  # noqa: E402
    AXES,
    CONFIG_PATH,
    SOLAR_RESULT,
    AugmentedBundleRationaleError,
    AugmentedRow,
    config,
    load_completed_solar,
    output_schema,
    parse_output,
    render_messages,
)
from mal2026.official_rationale_rl import MODEL_ID, MODEL_PATH, MODEL_REVISION  # noqa: E402
from mal2026.official_rl_servers import assert_gpus_idle, vllm_policy_server  # noqa: E402
from mal2026.solar_axis_augmentation import file_sha256  # noqa: E402


RUN_ID = "official-dpo-bundle-solar-augmented-rationales-v1-20260729-001"
ALIAS = "mal2026-selected-dpo-bundle"
DPO_ADAPTER = ROOT / (
    "outputs/official-rationale-rl-v1/orchestration/"
    "official-rationale-rl-experiment-v1-exact-judge-20260729-019/models/"
    "dpo-official-bundle-ddp4-full-user-aligned-001/adapter"
)
DPO_COMPLETION = DPO_ADAPTER.parent / "training_complete.json"
SELECTION = ROOT / (
    "outputs/official-rationale-rl-v1/evaluation/"
    "official-rationale-dpo-bundle-validation-exact-judge-20260729-020/"
    "aggregate_bundle_dpo_validation_comparison.json"
)
HANDOFF = ROOT / (
    "data/processed/restricted/official_prompt_alignment_v1/final_rationale_handoff/"
    "official-rationale-dpo-selected-handoff-exact-bundle-20260729-021/"
    "aggregate_handoff_manifest.json"
)
OUTPUT_ROOT = ROOT / "outputs/solar-augmented-bundle-rationale-v1"
RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_augmented_bundle_rationale_v1"


class AugmentedRationaleRunError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise AugmentedRationaleRunError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_sha(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"JSON object differs: {path}")
    return value


def verify_selected_model() -> dict[str, Any]:
    required = (
        DPO_ADAPTER / "adapter_config.json", DPO_ADAPTER / "adapter_model.safetensors",
        DPO_COMPLETION, SELECTION, HANDOFF, MODEL_PATH / "config.json",
    )
    need(all(path.is_file() and not path.is_symlink() for path in required), "selected DPO model binding is unavailable")
    completion, selection, handoff = read_json(DPO_COMPLETION), read_json(SELECTION), read_json(HANDOFF)
    need(completion.get("status") == "completed" and completion.get("task") == "bundle", "DPO completion differs")
    need(selection.get("status") == "completed" and selection.get("selection", {}).get("selected") == "dpo", "DPO selection differs")
    need(selection.get("axis_triplet_used_for_training_or_selection") is False, "axis-triplet selection lineage is forbidden")
    need(handoff.get("status") == "completed" and handoff.get("structure") == "bundle", "DPO handoff differs")
    need(handoff.get("axis_triplet_used_for_training_or_selection") is False, "axis-triplet handoff lineage is forbidden")
    need(handoff.get("human_or_reference_score_read_or_prompted") is False, "selected DPO handoff read a protected score")
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": file_sha256(MODEL_PATH / "config.json"),
        "adapter_config_sha256": file_sha256(DPO_ADAPTER / "adapter_config.json"),
        "adapter_model_sha256": file_sha256(DPO_ADAPTER / "adapter_model.safetensors"),
        "training_completion_sha256": file_sha256(DPO_COMPLETION),
        "winner_selection_sha256": file_sha256(SELECTION),
        "original_handoff_sha256": file_sha256(HANDOFF),
    }


def token_length_audit(rows: Sequence[AugmentedRow]) -> dict[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, revision=MODEL_REVISION, local_files_only=True,
        trust_remote_code=True, use_fast=True,
    )
    lengths = [
        len(tokenizer.apply_chat_template(render_messages(row), tokenize=True, add_generation_prompt=True))
        for row in rows
    ]
    need(len(lengths) == 6000 and max(lengths) + int(config()["generation"]["max_tokens"]) <= 4096, "augmented rationale input would truncate")
    ordered = sorted(lengths)
    return {
        "records": len(ordered), "maximum": ordered[-1],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "max_model_len": 4096, "max_completion_tokens": int(config()["generation"]["max_tokens"]),
        "truncated_records": 0,
    }


def request_body(row: AugmentedRow, attempt: int) -> dict[str, Any]:
    generation = config()["generation"]
    return {
        "model": ALIAS,
        "messages": render_messages(row),
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "seed": generation["seed"] + attempt - 1,
        "max_tokens": generation["max_tokens"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "mal2026_solar_augmented_bundle_rationale", "strict": True, "schema": output_schema()},
        },
    }


def request_one(endpoint: str, row: AugmentedRow, retries: int) -> tuple[dict[str, Any] | None, str | None]:
    last: str | None = None
    for attempt in range(1, retries + 1):
        body = request_body(row, attempt)
        request = Request(
            endpoint.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=300) as response:
                outer = json.loads(response.read().decode())
            choice = outer["choices"][0]
            need(choice.get("finish_reason") == "stop", "augmented rationale did not stop cleanly")
            rationales = parse_output(choice["message"]["content"])
            return {
                "source_id": row.identifier,
                "source_train_id": row.source_id,
                "target_axis": row.target_axis,
                "rationales": rationales,
                "attempts": attempt,
            }, None
        except HTTPError as exc:
            last = "http_429" if exc.code == 429 else ("http_5xx" if exc.code >= 500 else "http_4xx")
        except (URLError, TimeoutError):
            last = "transport"
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError, AugmentedRationaleRunError, AugmentedBundleRationaleError):
            last = "schema_or_finish"
        if last not in {"http_429", "http_5xx", "transport"}:
            break
        time.sleep(attempt)
    return None, last


def generate(endpoint: str, rows: Sequence[AugmentedRow], inflight: int, retries: int, partial: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    need(not partial.exists(), "augmented rationale partial output must be fresh")
    records: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    with partial.open("x", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=inflight) as pool:
        pending: dict[Any, AugmentedRow] = {}
        iterator = iter(rows)
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < inflight:
                try:
                    row = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(request_one, endpoint, row, retries)] = row
            if not pending:
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                record, failure = future.result()
                if record is None:
                    failures[failure or "unknown"] += 1
                else:
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            if len(records) % 100 == 0:
                handle.flush()
    return records, failures


def wait_gpu_release(gpus: Sequence[int], seconds: int = 120) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            assert_gpus_idle(gpus)
            return
        except Exception:
            time.sleep(1)
    raise AugmentedRationaleRunError("owned GPU contexts did not release")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    need(args.run_id == RUN_ID, "augmented rationale run identity differs")

    prompt = config()
    rows, solar_result = load_completed_solar()
    model_binding = verify_selected_model()
    token_audit = token_length_audit(rows)
    output, restricted = OUTPUT_ROOT / args.run_id, RESTRICTED_ROOT / args.run_id
    need(not output.exists() and not restricted.exists(), "augmented rationale outputs must be fresh")
    output.mkdir(mode=0o700, parents=True)
    restricted.mkdir(mode=0o700, parents=True)
    (output / "logs").mkdir(mode=0o700)
    (output / "attestations").mkdir(mode=0o700)
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "mal2026-solar-augmented-bundle-rationale-run-v1",
        "status": "running", "run_id": args.run_id, "created_at": now(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *sys.argv[1:]],
        "gpu_scope": [0, 1, 2, 3],
        "gpu_authorization": "repository default GPUs0-3 and explicit user request; GPUs4-7 not queried or used",
        "records_expected": 6000, "source_split": "train_only",
        "structure": "bundle", "axis_triplet_used": False,
        "solar_result_sha256": file_sha256(SOLAR_RESULT),
        "solar_augmented_train_sha256": solar_result["augmented_train_sha256"],
        "prompt_config_sha256": file_sha256(CONFIG_PATH),
        "token_length_audit": token_audit,
        "model_binding": model_binding,
        "score_kind": "solar_synthetic_continuous_quarter_step",
        "synthetic_scores_prompted": True,
        "human_or_reference_score_read_or_prompted": False,
        "validation_used_for_generation_or_selection": False,
        "average_read_or_used": False,
    }
    # The exact result file and its canonical payload are both bound without
    # exposing any generated row.
    manifest["solar_result_contract_sha256"] = canonical_sha(solar_result)
    atomic_json(manifest_path, manifest)

    generation = prompt["generation"]
    try:
        assert_gpus_idle((0,))
        with vllm_policy_server(
            runtime_root=output, label="solar-augmented-rationale-smoke", gpus=(0,), port=19430,
            adapters={"bundle": DPO_ADAPTER}, aliases={"bundle": ALIAS},
            max_num_seqs=16, max_num_batched_tokens=8192, dynamic_updates=False,
            max_model_len=4096, data_split="train",
        ) as (endpoint, _):
            record, failure = request_one(endpoint, rows[0], int(generation["retries"]))
            need(record is not None and failure is None, f"augmented rationale one-row smoke failed: {failure}")
            (restricted / "smoke.jsonl").write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        wait_gpu_release((0,))

        with vllm_policy_server(
            runtime_root=output, label="solar-augmented-rationale-full", gpus=(0, 1, 2, 3), port=19431,
            adapters={"bundle": DPO_ADAPTER}, aliases={"bundle": ALIAS},
            max_num_seqs=256, max_num_batched_tokens=32768, dynamic_updates=False,
            max_model_len=4096, data_split="train",
        ) as (endpoint, policy_attestation):
            records, failures = generate(
                endpoint, rows, int(generation["max_inflight"]), int(generation["retries"]),
                restricted / "rationales.partial.jsonl",
            )
        need(not failures and len(records) == 6000, f"augmented rationale hard gates failed: valid={len(records)} failures={dict(failures)}")
        records.sort(key=lambda row: str(row["source_id"]))
        final_path = restricted / "rationales.train-augmented.jsonl"
        with final_path.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        rationale_sha = file_sha256(final_path)
        handoff = {
            "schema_version": "mal2026-solar-augmented-bundle-rationale-handoff-v1",
            "status": "completed", "run_id": args.run_id,
            "structure": "bundle", "axis_triplet_used_for_training_or_selection": False,
            "records": 6000, "source_split": "train_only",
            "solar_augmented_train_sha256": solar_result["augmented_train_sha256"],
            "rationale_train_augmented_sha256": rationale_sha,
            "rationale_model_key": "official_bundle_dpo_exact_judge_20260729_019",
            "model_binding": model_binding,
            "prompt_config_sha256": file_sha256(CONFIG_PATH),
            "score_kind": "solar_synthetic_continuous_quarter_step",
            "synthetic_scores_prompted": True,
            "human_or_reference_score_read_or_prompted": False,
            "validation_used_for_generation_or_selection": False,
            "average_read_or_used": False,
            "generation": {key: generation[key] for key in ("temperature", "top_p", "seed", "max_tokens", "retries")},
            "policy_server_attestation_sha256": file_sha256(policy_attestation),
            "privacy": "aggregate manifest contains no essay, rationale, score, prediction, or identifier rows",
        }
        handoff_path = restricted / "aggregate_handoff_manifest.json"
        atomic_json(handoff_path, handoff)
        result = {
            "schema_version": "mal2026-solar-augmented-bundle-rationale-result-v1",
            "status": "completed", "run_id": args.run_id, "completed_at": now(),
            "records": 6000, "parse_valid": 6000, "failure_categories": {},
            "rationale_path": str(final_path.resolve()), "rationale_sha256": rationale_sha,
            "handoff_path": str(handoff_path.resolve()), "handoff_sha256": file_sha256(handoff_path),
            "structure": "bundle", "axis_triplet_used": False,
            "token_length_audit": token_audit,
            "human_or_reference_score_read_or_prompted": False,
            "privacy": "aggregate result contains no essay, rationale, score, prediction, or identifier rows",
        }
        result_path = output / "result.json"
        atomic_json(result_path, result)
        manifest.update({"status": "completed", "completed_at": now(), "result_sha256": file_sha256(result_path)})
        atomic_json(manifest_path, manifest)
        print(json.dumps(result, sort_keys=True))
    except BaseException as exc:
        manifest.update({"status": "failed", "failed_at": now(), "failure_category": type(exc).__name__})
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
