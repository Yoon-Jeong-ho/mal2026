#!/usr/bin/env python3
"""Write the orchestration contract's aggregate-only transition record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True); p.add_argument("--from-state", required=True)
    p.add_argument("--to-state", required=True); p.add_argument("--gate", required=True)
    p.add_argument("--decision", choices=("pass", "fail", "blocked"), required=True)
    p.add_argument("--data-scope", choices=("none", "train_only", "validation_only"), required=True)
    p.add_argument("--gpus", nargs="*", type=int, default=[]); p.add_argument("--evidence-ref", required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--technical", default="{}")
    p.add_argument("--semantic", default="{}"); p.add_argument("--immutable", default="not_run")
    p.add_argument("--one-variable-change", default="none")
    a = p.parse_args()
    for value in (a.technical, a.semantic):
        if not isinstance(json.loads(value), dict): raise SystemExit("failure categories must be JSON objects")
    record = {"run_id": a.run_id, "from_state": a.from_state, "to_state": a.to_state,
              "gate": a.gate, "decision": a.decision,
              "technical_failures": json.loads(a.technical), "semantic_abstentions": json.loads(a.semantic),
              "immutable_regressions": a.immutable, "data_scope": a.data_scope, "gpu_scope": a.gpus,
              "one_variable_change": a.one_variable_change, "evidence_ref": a.evidence_ref}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists(): raise SystemExit("refusing to overwrite gate summary")
    a.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
