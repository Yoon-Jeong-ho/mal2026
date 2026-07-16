"""Select a retained Trainer checkpoint using aggregate vLLM source-dev metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .standard_decoder_data import ROOT, StandardDecoderContractError


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StandardDecoderContractError(f"cannot read {description}") from exc
    if not isinstance(value, dict):
        raise StandardDecoderContractError(f"{description} must be a JSON object")
    return value


def select_checkpoint(selection_run_dir: Path, evaluation_results: Sequence[Path]) -> dict[str, Any]:
    """Write aggregate-only selected_checkpoint.json after all candidates scored.

    Trainer checkpoints are candidates only.  The maintained vLLM evaluator is
    the sole source of the predeclared source-dev macro-MAE selection metric.
    Ties are resolved deterministically by lower global update count.
    """
    runs_root = (ROOT / "outputs" / "standard-runs").resolve()
    run_dir = selection_run_dir.resolve()
    if run_dir.parent != runs_root:
        raise StandardDecoderContractError("selection run must be a direct standard-runs child")
    completion = _load_object(run_dir / "standard_training_complete.json", "selection training completion")
    if completion.get("status") != "completed" or completion.get("phase") != "selection":
        raise StandardDecoderContractError("selection run is not a completed selection Trainer run")
    expected_steps = completion.get("selection_candidate_steps")
    if not isinstance(expected_steps, list) or not expected_steps or any(not isinstance(step, int) or step <= 0 for step in expected_steps):
        raise StandardDecoderContractError("selection run has no retained Trainer checkpoint steps")
    if (run_dir / "selected_checkpoint.json").exists():
        raise StandardDecoderContractError("refusing to overwrite immutable selected checkpoint summary")
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for result_path in evaluation_results:
        result = _load_object(result_path.resolve(), "vLLM aggregate evaluation")
        if result.get("status") != "completed" or result.get("source") != "selection_dev":
            raise StandardDecoderContractError("all selection results must be completed source-dev vLLM evaluations")
        if result.get("mode") != completion.get("mode") or result.get("model_revision") != completion.get("model_revision"):
            raise StandardDecoderContractError("vLLM result mode/model revision differs from selection run")
        adapter = Path(result.get("adapter_path", "")).resolve()
        step = result.get("adapter_global_step")
        expected_adapter = run_dir / f"checkpoint-{step}"
        metrics = result.get("metrics")
        if not isinstance(step, int) or step not in expected_steps or step in seen or adapter != expected_adapter.resolve():
            raise StandardDecoderContractError("vLLM result does not correspond to one retained selection checkpoint")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("primary_macro_mae"), (float, int)):
            raise StandardDecoderContractError("vLLM result lacks numeric primary_macro_mae")
        metric = float(metrics["primary_macro_mae"])
        if metric != metric or metric == float("inf") or metric == float("-inf"):
            raise StandardDecoderContractError("vLLM macro MAE must be finite")
        seen.add(step)
        candidates.append({"global_step": step, "primary_macro_mae": metric, "evaluation_metrics_path": str(result_path.resolve())})
    if seen != set(expected_steps):
        raise StandardDecoderContractError("every retained Trainer checkpoint must have exactly one vLLM source-dev metric")
    selected = min(candidates, key=lambda item: (item["primary_macro_mae"], item["global_step"]))
    summary = {
        "status": "completed", "phase": "selection", "selection_run_id": completion.get("run_id"),
        "mode": completion.get("mode"), "model_revision": completion.get("model_revision"),
        "tokenizer_revision": completion.get("tokenizer_revision"),
        "selection_metric": "vllm_source_dev_primary_macro_mae", "trainer_monitor": "eval_loss_only",
        "selected_global_step": selected["global_step"], "selected_primary_macro_mae": selected["primary_macro_mae"],
        "candidates": sorted(candidates, key=lambda item: item["global_step"]),
        "privacy": "aggregate_only_no_rows_prompts_essays_feedback_ids_or_model_outputs_persisted",
    }
    (run_dir / "selected_checkpoint.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
