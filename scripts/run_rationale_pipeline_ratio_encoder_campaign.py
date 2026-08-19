#!/usr/bin/env python3
"""Continue all predeclared ratio encoder matrices after an active first stage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

from setproctitle import setproctitle


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-standard/bin/python"
MATRIX = ROOT / "scripts/run_rationale_pipeline_score_encoder_matrix.py"
MATRIX_OUTPUT = ROOT / "outputs/rationale-pipeline-score-encoder-matrix-v1"
OUTPUT_PARENT = ROOT / "outputs/rationale-pipeline-ratio-encoder-campaign-v1"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stage(model: str, ratio: str, suffix: str) -> dict[str, Any]:
    slug = "qwen3-embedding-8b" if model == "qwen3_embedding_8b" else "kure-v1"
    run_id = f"rationale-pipeline-score-encoder-matrix-{ratio}-{slug}-{suffix}"
    configs = [
        ROOT / "configs" / f"rationale-pipeline-score-encoder-{ratio}-{slug}-{objective}-{initialization}-{suffix}.json"
        for objective in ("bounded-regression", "categorical-5class")
        for initialization in ("base", "aihub")
    ]
    return {"model_key": model, "ratio": ratio, "run_id": run_id, "configs": configs}


def completed(item: Mapping[str, Any]) -> dict[str, Any] | None:
    path = MATRIX_OUTPUT / str(item["run_id"]) / "aggregate.json"
    if not path.is_file(): return None
    value = json.loads(path.read_text(encoding="utf-8"))
    need(value.get("status") == "completed" and value.get("model_key") == item["model_key"] and value.get("rationale_ratio") == item["ratio"], "ratio encoder campaign completed stage differs")
    return value


def pid_alive(pid: int, run_id: str) -> bool:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.is_file(): return False
    return run_id in path.read_bytes().replace(b"\0", b" ").decode(errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--active-first-pid", type=int, required=True)
    args = parser.parse_args()
    setproctitle(f"mal2026:ratio-encoder-campaign:{args.run_id}"[:255])
    output = OUTPUT_PARENT / args.run_id; need(not output.exists(), "ratio encoder campaign output must be fresh"); output.mkdir(parents=True)
    stages = [stage(model, ratio, args.suffix) for model in ("qwen3_embedding_8b", "kure_v1") for ratio in ("1to1", "1to2", "1to3")]
    need(all(path.is_file() for item in stages for path in item["configs"]), "ratio encoder campaign config unavailable")
    state: dict[str, Any] = {
        "schema_version": "mal2026-rationale-pipeline-ratio-encoder-campaign-v1",
        "status": "running", "run_id": args.run_id, "created_at": now(), "suffix": args.suffix,
        "gpu_scope": [0, 1, 2, 3],
        "user_authorization": "2026-08-08: user explicitly requested OpenAI:SFT 1:1, 1:2, and 1:3 encoder training for Qwen3-Embedding-8B and KURE, with and without AI-Hub, regression and classification.",
        "stages": [{"run_id": item["run_id"], "model_key": item["model_key"], "ratio": item["ratio"], "status": "pending"} for item in stages],
        "average_used": False,
    }
    atomic_json(output / "state.json", state)

    first = stages[0]
    while completed(first) is None:
        need(pid_alive(args.active_first_pid, first["run_id"]), "active first ratio encoder stage exited without aggregate")
        time.sleep(60)
    state["stages"][0]["status"] = "completed"; state["stages"][0]["completed_at"] = now(); atomic_json(output / "state.json", state)

    for index, item in enumerate(stages[1:], 1):
        existing = completed(item)
        if existing is None:
            destination = MATRIX_OUTPUT / item["run_id"]
            need(not destination.exists(), f"ratio encoder campaign stage has partial output: {item['run_id']}")
            state["stages"][index]["status"] = "running"; state["stages"][index]["started_at"] = now(); atomic_json(output / "state.json", state)
            log = (output / f"{index:02d}-{item['run_id']}.log").open("x", encoding="utf-8")
            command = [str(PYTHON), str(MATRIX), "--run-id", item["run_id"]]
            for path in item["configs"]: command.extend(("--config", str(path)))
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True); log.close()
            need(result.returncode == 0, f"ratio encoder campaign stage failed: {item['run_id']}")
            existing = completed(item); need(existing is not None, "ratio encoder campaign stage aggregate unavailable")
        state["stages"][index]["status"] = "completed"; state["stages"][index]["completed_at"] = now(); atomic_json(output / "state.json", state)

    summaries = []
    for item in stages:
        value = completed(item); need(value is not None, "ratio encoder campaign completion differs")
        summaries.append({"run_id": item["run_id"], "model_key": item["model_key"], "ratio": item["ratio"], "winner": value["winner"]})
    state.update({"status": "completed", "completed_at": now(), "summaries": summaries})
    atomic_json(output / "aggregate.json", state); atomic_json(output / "state.json", state)
    print(json.dumps({"status": "completed", "run_id": args.run_id, "stages": len(stages)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
