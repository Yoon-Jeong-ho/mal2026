#!/usr/bin/env python3
"""Score a frozen random Solar-consensus control with the pinned Qwen3.6 Q4 judge."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mal2026.official_rl_servers import Q4_MODEL, q4_judge_servers  # noqa: E402
from mal2026.official_score_matrix import AXES, file_sha256, score_metrics  # noqa: E402
from mal2026.official_score_prompt import (  # noqa: E402
    EVALUATION_PROMPT_SHA256,
    USER_SUPPLIED_EVALUATION,
    query_text,
    system_prompt,
)


SELECTION_RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_consensus_selection_v1"
OUTPUT_ROOT = ROOT / "outputs/solar-consensus-qwen36-control-v1"
RESTRICTED_ROOT = ROOT / "data/processed/restricted/solar_consensus_qwen36_control_v1"
MODEL_ALIAS = "qwen36-35b-a3b-q4_k_m"
GPUS = (0, 1, 2, 3)
PORTS = (19420, 19421, 19422, 19423)
MAX_INFLIGHT = 16
SEED = 2026073004


class Qwen36ControlError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Qwen36ControlError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    need(path.is_file() and not path.is_symlink(), "Qwen3.6 control input is unavailable")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    need(rows and all(isinstance(row, dict) for row in rows), "Qwen3.6 control input differs")
    identifiers = [row.get("candidate_id") for row in rows]
    need(all(isinstance(value, str) for value in identifiers) and
         len(set(identifiers)) == len(identifiers), "Qwen3.6 control identity differs")
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    need(not path.exists(), "Qwen3.6 control output must be fresh")
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return file_sha256(path)


def output_schema() -> dict[str, Any]:
    cell = {
        "type": "object",
        "additionalProperties": False,
        "required": ["score", "rationale"],
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "rationale": {"type": "string", "minLength": 1},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(AXES),
        "properties": {axis: cell for axis in AXES},
    }


def parse_output(text: str) -> dict[str, dict[str, Any]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Qwen36ControlError("Qwen3.6 output is not JSON") from exc
    need(isinstance(value, dict) and set(value) == set(AXES),
         "Qwen3.6 output axes differ")
    result: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        cell = value[axis]
        need(isinstance(cell, dict) and set(cell) == {"score", "rationale"} and
             type(cell["score"]) is int and 1 <= cell["score"] <= 5 and
             isinstance(cell["rationale"], str) and cell["rationale"].strip(),
             "Qwen3.6 output cell differs")
        result[axis] = {"score": cell["score"], "rationale": cell["rationale"].strip()}
    return result


def http_json(endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    error: BaseException | None = None
    for _ in range(2):
        try:
            request = Request(
                endpoint + "/v1/chat/completions", data=wire,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urlopen(request, timeout=900) as response:
                value = json.loads(response.read().decode())
            need(isinstance(value, dict), "Qwen3.6 response envelope differs")
            return value
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError,
                Qwen36ControlError) as exc:
            error = exc
    assert error is not None
    raise Qwen36ControlError("Qwen3.6 local request failed") from error


def score_one(endpoint: str, row: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = str(row["candidate_id"])
    seed = int.from_bytes(
        sha256(f"{SEED}\0{candidate_id}".encode()).digest()[:4], "big"
    )
    response_format = {"type": "json_object", "schema": output_schema()}
    payload = {
        "model": MODEL_ALIAS,
        "messages": [
            {"role": "system", "content": system_prompt(USER_SUPPLIED_EVALUATION)},
            {"role": "user", "content": query_text(
                str(row["prompt"]), str(row["essay"]),
                kind=USER_SUPPLIED_EVALUATION,
            )},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "max_tokens": 1536,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": response_format,
    }
    outer = http_json(endpoint, payload)
    try:
        choice = outer["choices"][0]
        need(choice.get("finish_reason") in {"stop", "length"},
             "Qwen3.6 finish reason differs")
        parsed = parse_output(choice["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise Qwen36ControlError("Qwen3.6 response envelope differs") from exc
    return {
        "candidate_id": candidate_id,
        "control_stratum": row["control_stratum"],
        "seed": seed,
        "qwen36": parsed,
        "qwen36_score": {axis: parsed[axis]["score"] for axis in AXES},
        "solar_modal_score": row["score"],
        "oof_encoder_consensus": row["oof_encoder_consensus"],
    }


def agreement_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    need(rows, "agreement population is empty")
    labels = [[float(row["solar_modal_score"][axis]) for axis in AXES] for row in rows]
    qwen = [[float(row["qwen36_score"][axis]) for axis in AXES] for row in rows]
    integers = [[int(value) for value in values] for values in qwen]
    metrics = score_metrics(
        labels, qwen, integers, [[[False], [False], [False]] for _ in rows]
    )
    metrics["triplet_exact_agreement"] = sum(
        all(int(row["qwen36_score"][axis]) == int(row["solar_modal_score"][axis])
            for axis in AXES)
        for row in rows
    ) / len(rows)
    metrics["axis_exact_agreement"] = {
        axis: sum(int(row["qwen36_score"][axis]) ==
                  int(row["solar_modal_score"][axis]) for row in rows) / len(rows)
        for axis in AXES
    }
    metrics["qwen36_score_counts"] = {
        axis: dict(sorted(Counter(
            int(row["qwen36_score"][axis]) for row in rows
        ).items())) for axis in AXES
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--selection-run-id", required=True)
    args = parser.parse_args()
    output = OUTPUT_ROOT / args.run_id
    restricted = RESTRICTED_ROOT / args.run_id
    need(not output.exists() and not restricted.exists(), "Qwen3.6 outputs must be fresh")
    output.mkdir(mode=0o700, parents=True)
    restricted.mkdir(mode=0o700, parents=True)
    input_path = SELECTION_RESTRICTED_ROOT / args.selection_run_id / "qwen36_random_control.jsonl"
    rows = read_jsonl(input_path)
    runtime = output / "runtime"
    runtime.mkdir(mode=0o700)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with q4_judge_servers(
        runtime_root=runtime,
        label="solar-consensus-control",
        gpus=GPUS,
        ports=PORTS,
        judge_prompt_sha256=EVALUATION_PROMPT_SHA256,
    ) as (endpoints, attestation):
        with ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as pool:
            futures = {
                pool.submit(score_one, endpoints[index % len(endpoints)], row): row
                for index, row in enumerate(rows)
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    records.append(future.result())
                except Exception as exc:
                    failures.append({
                        "candidate_id": str(row["candidate_id"]),
                        "category": type(exc).__name__,
                    })
    records.sort(key=lambda row: str(row["candidate_id"]))
    failures.sort(key=lambda row: row["candidate_id"])
    need(not failures and len(records) == len(rows), "Qwen3.6 control has scoring failures")
    records_path = restricted / "qwen36_control_scores.jsonl"
    records_hash = write_jsonl(records_path, records)
    strata = sorted({str(row["control_stratum"]) for row in records})
    result = {
        "schema_version": "mal2026-solar-consensus-qwen36-control-result-v1",
        "status": "completed",
        "completed_at": now(),
        "run_id": args.run_id,
        "selection_run_id": args.selection_run_id,
        "records": len(records),
        "failed": 0,
        "agreement_with_solar_pseudo_label": agreement_summary(records),
        "agreement_by_stratum": {
            stratum: agreement_summary([
                row for row in records if row["control_stratum"] == stratum
            ]) for stratum in strata
        },
        "protocol": {
            "physical_gpus": list(GPUS),
            "parallel_requests": MAX_INFLIGHT,
            "temperature": 0.0,
            "validation_used": False,
            "exact_evaluation_txt_prompt": True,
            "solar_modal_score_is_pseudo_label_not_ground_truth": True,
            "control_selection_was_frozen_before_qwen36_scoring": True,
        },
        "bindings": {
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "worktree_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()),
            "evaluation_txt_sha256": EVALUATION_PROMPT_SHA256,
            "q4_model_path": str(Q4_MODEL.resolve()),
            "q4_model_sha256": file_sha256(Q4_MODEL),
            "control_input_sha256": file_sha256(input_path),
            "server_attestation_sha256": file_sha256(attestation),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "records_sha256": records_hash,
        },
        "privacy": "aggregate contains no essay, prompt, rationale, identifier, or individual prediction",
    }
    need(math.isfinite(float(
        result["agreement_with_solar_pseudo_label"]["macro_continuous_rmse"]
    )), "Qwen3.6 aggregate metric is non-finite")
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "completed", "records": len(records),
        "macro_rmse_vs_solar": result["agreement_with_solar_pseudo_label"][
            "macro_continuous_rmse"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
