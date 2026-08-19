#!/usr/bin/env python3
"""Recompute a judge aggregate while discarding all raw request/response payloads."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import ijson

from judge_feedback_candidates_v2 import aggregate, config, gate_results, sha256


REQUEST_FIELDS = {"opaque_request_key", "opaque_logical_key", "opaque_group_key", "kind", "repeat", "candidate_1_label", "cell", "pair_key", "pointwise_group_candidate_1", "pointwise_group_candidate_2"}
RESPONSE_FIELDS = {"opaque_request_key", "resolved_verdict", "transport_or_schema_failure"}


def restricted_fields(path: Path, fields: set[str], *, need_order: bool) -> list[dict[str, Any]]:
    """Stream JSON events and retain only aggregation metadata, never payloads."""
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            row: dict[str, Any] = {}
            order: list[str] = []
            # Do not deserialize the JSONL object.  This event stream retains
            # only top-level aggregation metadata and ignores body/response.
            for prefix, event, value in ijson.parse(io.BytesIO(line)):
                if prefix in fields and event in {"string", "number", "boolean", "null"}:
                    row[prefix] = value
                elif need_order and prefix == "display_order.item" and event == "string":
                    order.append(value)
            if need_order:
                row["display_order"] = order
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    requests = restricted_fields(run / "pilot_requests.jsonl", REQUEST_FIELDS, need_order=True)
    responses = restricted_fields(run / "pilot_raw_responses.jsonl", RESPONSE_FIELDS, need_order=False)
    metrics = aggregate(requests, responses, config())
    sample = manifest.get("sample_essays")
    if not isinstance(sample, int):
        raise SystemExit("manifest sample count is invalid")
    metrics["sample_essays"] = sample
    gates = gate_results(metrics, sample, config())
    output = {"schema_version": "mal2026-judge-aggregate-audit-v1", "status": "passed" if all(gates.values()) else "failed_gates", "metrics": metrics, "hard_gates": gates, "request_sha256": sha256(run / "pilot_requests.jsonl"), "raw_response_sha256": sha256(run / "pilot_raw_responses.jsonl"), "raw_payloads_retained": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "request_count": metrics["counts"]["requests"], "raw_payloads_retained": False}, sort_keys=True))


if __name__ == "__main__":
    main()
