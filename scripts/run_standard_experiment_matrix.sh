#!/usr/bin/env bash
# Run the approved decoder/encoder matrix sequentially with standard project runners.
# This launcher never stores restricted rows or credentials. It must be started from
# tmux (or a scheduler) after validated smoke tests, and it refuses busy GPUs.
set -Eeuo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_DEFAULT="$ROOT/.venv-standard/bin/python"
TORCHRUN_DEFAULT="$ROOT/.venv-standard/bin/torchrun"
QWEN_REV="a09a35458c702b33eeacc393d103063234e8bc28"
QWEN3_REV="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
NV_REV="3fa59658547db50a1e8e3346cf057fd0c77ed6ef"

usage() {
  cat <<USAGE
Usage: $(basename "$0") --runtime-root ABS_DIR --run-prefix PREFIX \\
  --prepared-manifest ABS_PATH --validation-sha256 SHA256 \\
  --qwen-model ABS_DIR --qwen3-model ABS_DIR --nv-model ABS_DIR --nv-review-json ABS_PATH [options]

Runs, strictly sequentially: direct decoder selection -> vLLM source-dev checkpoint
selection -> refit -> frozen final evaluation; human_feedback decoder through the same
lifecycle; Qwen3 and reviewed NV encoder selection -> refit -> frozen final evaluation.

Required paths may alternatively be supplied as MAL2026_{RUNTIME_ROOT,RUN_PREFIX,
PREPARED_MANIFEST,VALIDATION_SHA256,QWEN_MODEL,QWEN3_MODEL,NV_MODEL,NV_REVIEW_JSON}.
All generated configs, logs, and aggregate-only ledger entries are written below the
new ignored runtime root. Training/evaluation outputs remain direct children of the
canonical ignored outputs/standard-*-{runs,evals} roots.

Options:
  --num-gpus N                    DDP/vLLM GPU count (default: 4; GPUs 0-3 only)
  --cuda-visible-devices LIST     CUDA devices (default: 0,...,N-1)
  --wandb-project NAME            W&B scalar project (default: mal2026-korean-writing-scoring)
  --wandb-entity NAME             optional W&B entity
  --decoder-batch-size N          stable decoder per-device batch (default: 1)
  --decoder-grad-accum N          decoder accumulation; defaults to global batch 64
  --encoder-batch-size N          encoder per-device batch (default: 1)
  --encoder-grad-accum N          encoder accumulation; defaults to global batch 64
  --dry-run                       print planned outputs; do not check paths or write/run anything
  -h, --help                      show this text

Do not use --dry-run as a smoke test. Before a real launch, verify the approved
validated smoke evidence compatible with the selected 0--3 GPU allocation and ensure no other process owns those GPUs.
USAGE
}

RUNTIME_ROOT="${MAL2026_RUNTIME_ROOT:-}"
RUN_PREFIX="${MAL2026_RUN_PREFIX:-}"
MANIFEST="${MAL2026_PREPARED_MANIFEST:-}"
VALIDATION_SHA256="${MAL2026_VALIDATION_SHA256:-}"
QWEN_MODEL="${MAL2026_QWEN_MODEL:-}"
QWEN3_MODEL="${MAL2026_QWEN3_MODEL:-}"
NV_MODEL="${MAL2026_NV_MODEL:-}"
NV_REVIEW_JSON="${MAL2026_NV_REVIEW_JSON:-}"
NUM_GPUS=4
CUDA_VISIBLE=""
WANDB_PROJECT="mal2026-korean-writing-scoring"
WANDB_ENTITY=""
DECODER_BATCH=1
DECODER_ACCUM=""
ENCODER_BATCH=1
ENCODER_ACCUM=""
DRY_RUN=false

need_value() { [[ $# -ge 2 && -n "$2" ]] || { echo "missing value for $1" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime-root) need_value "$@"; RUNTIME_ROOT="$2"; shift 2;;
    --run-prefix) need_value "$@"; RUN_PREFIX="$2"; shift 2;;
    --prepared-manifest) need_value "$@"; MANIFEST="$2"; shift 2;;
    --validation-sha256) need_value "$@"; VALIDATION_SHA256="$2"; shift 2;;
    --qwen-model) need_value "$@"; QWEN_MODEL="$2"; shift 2;;
    --qwen3-model) need_value "$@"; QWEN3_MODEL="$2"; shift 2;;
    --nv-model) need_value "$@"; NV_MODEL="$2"; shift 2;;
    --nv-review-json) need_value "$@"; NV_REVIEW_JSON="$2"; shift 2;;
    --num-gpus) need_value "$@"; NUM_GPUS="$2"; shift 2;;
    --cuda-visible-devices) need_value "$@"; CUDA_VISIBLE="$2"; shift 2;;
    --wandb-project) need_value "$@"; WANDB_PROJECT="$2"; shift 2;;
    --wandb-entity) need_value "$@"; WANDB_ENTITY="$2"; shift 2;;
    --decoder-batch-size) need_value "$@"; DECODER_BATCH="$2"; shift 2;;
    --decoder-grad-accum) need_value "$@"; DECODER_ACCUM="$2"; shift 2;;
    --encoder-batch-size) need_value "$@"; ENCODER_BATCH="$2"; shift 2;;
    --encoder-grad-accum) need_value "$@"; ENCODER_ACCUM="$2"; shift 2;;
    --dry-run) DRY_RUN=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

is_positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_absolute() { [[ "$1" = /* ]]; }
for item in "$NUM_GPUS" "$DECODER_BATCH" "$ENCODER_BATCH"; do
  is_positive_integer "$item" || { echo "positive integer required: $item" >&2; exit 2; }
done
(( NUM_GPUS <= 4 )) || { echo "current allocation is restricted to at most GPUs 0,1,2,3" >&2; exit 2; }
# Keep the research protocol's global effective batch fixed at 64.  With the
# safe b1 defaults this derives 16 accumulation steps on the permitted four
# GPUs. Overrides
# are accepted only when they preserve the same global batch exactly.
if [[ -z "$DECODER_ACCUM" ]]; then
  (( 64 % (NUM_GPUS * DECODER_BATCH) == 0 )) || { echo "decoder batch/GPU product must divide global batch 64" >&2; exit 2; }
  DECODER_ACCUM=$((64 / (NUM_GPUS * DECODER_BATCH)))
fi
if [[ -z "$ENCODER_ACCUM" ]]; then
  (( 64 % (NUM_GPUS * ENCODER_BATCH) == 0 )) || { echo "encoder batch/GPU product must divide global batch 64" >&2; exit 2; }
  ENCODER_ACCUM=$((64 / (NUM_GPUS * ENCODER_BATCH)))
fi
for item in "$DECODER_ACCUM" "$ENCODER_ACCUM"; do
  is_positive_integer "$item" || { echo "positive integer required: $item" >&2; exit 2; }
done
(( NUM_GPUS * DECODER_BATCH * DECODER_ACCUM == 64 )) || { echo "decoder settings must preserve global effective batch 64" >&2; exit 2; }
(( NUM_GPUS * ENCODER_BATCH * ENCODER_ACCUM == 64 )) || { echo "encoder settings must preserve global effective batch 64" >&2; exit 2; }
[[ "$RUN_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || { echo "run prefix must be a safe 1-80 char identifier" >&2; exit 2; }
[[ "$VALIDATION_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "validation SHA-256 must be 64 lowercase hexadecimal characters" >&2; exit 2; }
for path in "$RUNTIME_ROOT" "$MANIFEST" "$QWEN_MODEL" "$QWEN3_MODEL" "$NV_MODEL" "$NV_REVIEW_JSON"; do
  is_absolute "$path" || { echo "absolute path required: $path" >&2; exit 2; }
done
if [[ -z "$CUDA_VISIBLE" ]]; then
  CUDA_VISIBLE="$(seq -s, 0 $((NUM_GPUS - 1)))"
fi
IFS=, read -r -a CUDA_IDS <<< "$CUDA_VISIBLE"
[[ ${#CUDA_IDS[@]} -eq $NUM_GPUS ]] || { echo "--cuda-visible-devices must list exactly --num-gpus devices" >&2; exit 2; }
declare -A REQUESTED_GPU_INDEX=()
for gpu in "${CUDA_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "--cuda-visible-devices must use numeric physical GPU indices" >&2; exit 2; }
  (( 10#$gpu <= 3 )) || { echo "current allocation is restricted to physical GPUs 0,1,2,3" >&2; exit 2; }
  [[ -z "${REQUESTED_GPU_INDEX[$gpu]+x}" ]] || { echo "--cuda-visible-devices contains a duplicate GPU index: $gpu" >&2; exit 2; }
  REQUESTED_GPU_INDEX[$gpu]=1
done

RUNS="$ROOT/outputs/standard-runs"
EVALS="$ROOT/outputs/standard-evals"
ENCODER_RUNS="$ROOT/outputs/standard-encoder-runs"
ENCODER_EVALS="$ROOT/outputs/standard-encoder-evals"
PYTHON="${MAL2026_PYTHON:-$PYTHON_DEFAULT}"
TORCHRUN="${MAL2026_TORCHRUN:-$TORCHRUN_DEFAULT}"
NVIDIA_SMI="${MAL2026_NVIDIA_SMI:-nvidia-smi}"

run_name() { printf '%s-%s' "$RUN_PREFIX" "$1"; }
DIRECT_SELECTION="$RUNS/$(run_name decoder-direct-selection)"
DIRECT_REFIT="$RUNS/$(run_name decoder-direct-refit)"
FEEDBACK_SELECTION="$RUNS/$(run_name decoder-human-feedback-selection)"
FEEDBACK_REFIT="$RUNS/$(run_name decoder-human-feedback-refit)"
QWEN3_SELECTION="$ENCODER_RUNS/$(run_name encoder-qwen3-selection)"
QWEN3_REFIT="$ENCODER_RUNS/$(run_name encoder-qwen3-refit)"
NV_SELECTION="$ENCODER_RUNS/$(run_name encoder-nvembed-selection)"
NV_REFIT="$ENCODER_RUNS/$(run_name encoder-nvembed-refit)"

if "$DRY_RUN"; then
  cat <<PLAN
DRY RUN: no paths checked, files created, or jobs launched.
Runtime root: $RUNTIME_ROOT
GPU contract: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE; DDP/vLLM GPUs=$NUM_GPUS
Stable decoder settings: per-device batch=$DECODER_BATCH, accumulation=$DECODER_ACCUM
Encoder settings: per-device batch=$ENCODER_BATCH, accumulation=$ENCODER_ACCUM
Selection run outputs:
  $DIRECT_SELECTION
  $FEEDBACK_SELECTION
  $QWEN3_SELECTION
  $NV_SELECTION
PLAN
  exit 0
fi

for command in "$PYTHON" "$TORCHRUN" git; do [[ -x "$command" || -n "$(command -v "$command" 2>/dev/null || true)" ]] || { echo "required command unavailable: $command" >&2; exit 2; }; done
for path in "$MANIFEST" "$NV_REVIEW_JSON"; do [[ -f "$path" ]] || { echo "required file missing: $path" >&2; exit 2; }; done
for path in "$QWEN_MODEL" "$QWEN3_MODEL" "$NV_MODEL"; do [[ -d "$path" ]] || { echo "required model directory missing: $path" >&2; exit 2; }; done
[[ ! -e "$RUNTIME_ROOT" ]] || { echo "runtime root already exists; refusing overwrite: $RUNTIME_ROOT" >&2; exit 2; }
[[ "$RUNTIME_ROOT" == "$ROOT/outputs/"* ]] || { echo "runtime root must be inside ignored $ROOT/outputs" >&2; exit 2; }
git -C "$ROOT" check-ignore -q "$RUNTIME_ROOT" || { echo "runtime root is not ignored by Git: $RUNTIME_ROOT" >&2; exit 2; }
for path in "$DIRECT_SELECTION" "$DIRECT_REFIT" "$FEEDBACK_SELECTION" "$FEEDBACK_REFIT" "$QWEN3_SELECTION" "$QWEN3_REFIT" "$NV_SELECTION" "$NV_REFIT"; do
  [[ ! -e "$path" ]] || { echo "run output already exists; choose a new prefix: $path" >&2; exit 2; }
done
"$PYTHON" - "$NV_REVIEW_JSON" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
required = {"model_id", "revision", "license_acknowledged", "use_case", "reviewer", "outcome", "reviewed_files"}
if not isinstance(payload, dict) or set(payload) != required or payload["model_id"] != "nvidia/NV-Embed-v2" or payload["revision"] != "3fa59658547db50a1e8e3346cf057fd0c77ed6ef" or payload["outcome"] != "approved" or not payload["license_acknowledged"] or not isinstance(payload["reviewed_files"], dict) or not payload["reviewed_files"]:
    raise SystemExit("NV review JSON is not an approved immutable NV-Embed-v2 review")
PY
# Map selected indices to UUIDs immediately before each process stage. Other
# users may legitimately occupy unselected GPUs, but a selected GPU is never
# shared or terminated. Returning rather than exiting lets run_step ledger the
# refusal before the matrix aborts.
assert_selected_gpus_idle() {
  # This is an ownership gate, not an optional diagnostic: without a working
  # nvidia-smi executable the launcher cannot prove the selected GPUs are
  # unshared.  Refuse before every stage rather than risking a shared launch.
  if ! command -v "$NVIDIA_SMI" >/dev/null 2>&1 && [[ ! -x "$NVIDIA_SMI" ]]; then
    echo "unable to resolve nvidia-smi for selected-GPU ownership preflight; refusing to launch" >&2
    return 1
  fi
  local gpu_uuid_table gpu_process_table raw_index raw_uuid index uuid raw_pid pid
  gpu_uuid_table="$($NVIDIA_SMI --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)" || {
    echo "unable to identify selected GPU UUIDs; refusing to launch" >&2; return 1;
  }
  local -A requested_gpu_uuid=()
  while IFS=, read -r raw_index raw_uuid; do
    index="$(printf '%s' "$raw_index" | tr -d '[:space:]')"
    uuid="$(printf '%s' "$raw_uuid" | tr -d '[:space:]')"
    if [[ -n "${REQUESTED_GPU_INDEX[$index]+x}" && -n "$uuid" ]]; then requested_gpu_uuid[$uuid]=1; fi
  done <<< "$gpu_uuid_table"
  [[ ${#requested_gpu_uuid[@]} -eq $NUM_GPUS ]] || {
    echo "could not map every selected GPU index to a UUID; refusing to launch" >&2; return 1;
  }
  gpu_process_table="$($NVIDIA_SMI --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null)" || {
    echo "unable to inspect selected GPU processes; refusing to launch" >&2; return 1;
  }
  while IFS=, read -r raw_pid raw_uuid; do
    pid="$(printf '%s' "$raw_pid" | tr -d '[:space:]')"
    uuid="$(printf '%s' "$raw_uuid" | tr -d '[:space:]')"
    if [[ -n "$pid" && -n "${requested_gpu_uuid[$uuid]+x}" ]]; then
      echo "selected GPU UUID $uuid has active compute PID $pid; refusing to share or terminate it" >&2
      return 1
    fi
  done <<< "$gpu_process_table"
}

CONFIGS="$RUNTIME_ROOT/configs"; LOGS="$RUNTIME_ROOT/logs"; LEDGER="$RUNTIME_ROOT/matrix_ledger.jsonl"
mkdir -p "$CONFIGS" "$LOGS"
export ROOT RUNTIME_ROOT RUN_PREFIX MANIFEST VALIDATION_SHA256 QWEN_MODEL QWEN3_MODEL NV_MODEL NV_REVIEW_JSON NUM_GPUS CUDA_VISIBLE WANDB_PROJECT WANDB_ENTITY DECODER_BATCH DECODER_ACCUM ENCODER_BATCH ENCODER_ACCUM QWEN_REV QWEN3_REV NV_REV DIRECT_SELECTION DIRECT_REFIT FEEDBACK_SELECTION FEEDBACK_REFIT QWEN3_SELECTION QWEN3_REFIT NV_SELECTION NV_REFIT
"$PYTHON" - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
root = Path(os.environ["RUNTIME_ROOT"]); configs = root / "configs"
def dump(name, payload):
    path = configs / name
    with path.open("x", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True); f.write("\n")
def decoder(mode, output):
    return {"run_id": Path(output).name, "phase":"selection", "mode":mode, "model_path":os.environ["QWEN_MODEL"], "tokenizer_path":os.environ["QWEN_MODEL"], "model_revision":os.environ["QWEN_REV"], "tokenizer_revision":os.environ["QWEN_REV"], "prepared_manifest":os.environ["MANIFEST"], "output_dir":output, "seed":2026, "max_length":2048 if mode == "direct" else 4096, "learning_rate":2e-5, "num_train_epochs":12.0, "per_device_train_batch_size":int(os.environ["DECODER_BATCH"]), "per_device_eval_batch_size":int(os.environ["DECODER_BATCH"]), "gradient_accumulation_steps":int(os.environ["DECODER_ACCUM"]), "eval_steps":100, "save_steps":100, "logging_steps":5, "early_stopping_patience":4, "lora_r":32, "lora_alpha":64, "lora_dropout":0.05, "selection_summary_path":None, "selected_global_step":None, "wandb_project":os.environ["WANDB_PROJECT"], "wandb_entity":os.environ["WANDB_ENTITY"] or None}
def encoder(backbone, output):
    nv = backbone == "nv_embed_v2"; model = os.environ["NV_MODEL"] if nv else os.environ["QWEN3_MODEL"]; revision = os.environ["NV_REV"] if nv else os.environ["QWEN3_REV"]
    return {"run_id":Path(output).name, "phase":"selection", "backbone":backbone, "model_id":"nvidia/NV-Embed-v2" if nv else "Qwen/Qwen3-Embedding-8B", "model_revision":revision, "tokenizer_revision":revision, "model_path":model, "prepared_manifest":os.environ["MANIFEST"], "output_dir":output, "max_length":2048, "seed":2026, "learning_rate":1e-4, "weight_decay":0.01, "warmup_ratio":0.05, "num_train_epochs":20.0, "per_device_train_batch_size":int(os.environ["ENCODER_BATCH"]), "per_device_eval_batch_size":int(os.environ["ENCODER_BATCH"]), "gradient_accumulation_steps":int(os.environ["ENCODER_ACCUM"]), "eval_steps":100, "save_steps":100, "logging_steps":5, "early_stopping_patience":3, "lora_r":16, "lora_alpha":32, "lora_dropout":0.05, "lora_target_modules":["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], "nv_snapshot_dir":model if nv else None, "nv_review":json.load(open(os.environ["NV_REVIEW_JSON"], encoding="utf-8")) if nv else None, "selection_metadata_path":None, "wandb_project":os.environ["WANDB_PROJECT"], "wandb_entity":os.environ["WANDB_ENTITY"] or None}
dump("decoder-direct-selection.json", decoder("direct", os.environ["DIRECT_SELECTION"]))
dump("decoder-human-feedback-selection.json", decoder("human_feedback", os.environ["FEEDBACK_SELECTION"]))
dump("encoder-qwen3-selection.json", encoder("qwen3_embedding", os.environ["QWEN3_SELECTION"]))
dump("encoder-nvembed-selection.json", encoder("nv_embed_v2", os.environ["NV_SELECTION"]))
manifest = {"status":"started", "run_prefix":os.environ["RUN_PREFIX"], "git_sha":subprocess.check_output(["git","-C",os.environ["ROOT"],"rev-parse","HEAD"], text=True).strip(), "prepared_manifest":os.environ["MANIFEST"], "validation_sha256":os.environ["VALIDATION_SHA256"], "cuda_visible_devices":os.environ["CUDA_VISIBLE"], "num_gpus":int(os.environ["NUM_GPUS"]), "privacy":"aggregate_only_no_restricted_rows_or_credentials", "selection_configs":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in configs.glob("*.json")}}
with (root / "matrix_manifest.json").open("x", encoding="utf-8") as f: json.dump(manifest, f, indent=2, sort_keys=True); f.write("\n")
PY

ledger() { local status="$1" step="$2" detail="${3:-}"; STATUS="$status" STEP="$step" DETAIL="$detail" "$PYTHON" - "$LEDGER" <<'PY'
import json, os, sys, time
entry={"time_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status":os.environ["STATUS"], "step":os.environ["STEP"], "detail":os.environ["DETAIL"], "privacy":"aggregate_only"}
with open(sys.argv[1], "a", encoding="utf-8") as f: f.write(json.dumps(entry, ensure_ascii=True, sort_keys=True)+"\n")
PY
}
run_step() {
  local step="$1"; shift
  local log="$LOGS/$step.log"
  ledger started "$step" "$*"
  if ! assert_selected_gpus_idle; then
    ledger failed "$step" "selected_gpu_busy_or_preflight_refusal"
    return 1
  fi
  if env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE" PYTHONPATH="$ROOT/src" TOKENIZERS_PARALLELISM=false "$@" >"$log" 2>&1; then
    return 0
  fi
  ledger failed "$step" "$log"
  echo "failed step preserved at $log" >&2
  return 1
}
validate_stage_artifacts() { "$PYTHON" - "$@" <<'PY'
"""Reject incomplete, numerically corrupt, or schema-inadequate stage artifacts.

The launcher is intentionally a last-line release gate rather than a second
implementation of every runner's provenance contract.  It verifies the
minimal completion fields that make an artifact usable by the next matrix
stage, before the aggregate-only ledger can call that stage completed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


step, *raw_paths = sys.argv[1:]
if not step or not raw_paths:
    raise SystemExit("stage validator requires a step and at least one artifact path")


def fail(path: Path, message: str) -> "None":
    raise SystemExit(f"stage {step}: invalid artifact {path}: {message}")


def object_field(payload: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        fail(path, f"{key} must be a mapping")
    return value


def finite_number(value: Any, field: str, path: Path, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(path, f"{field} must be a finite numeric value")
    numeric = float(value)
    if not math.isfinite(numeric):
        fail(path, f"{field} must be finite")
    if nonnegative and numeric < 0:
        fail(path, f"{field} must be nonnegative")
    return numeric


def positive_int(value: Any, field: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(path, f"{field} must be a positive integer")
    return value


def nonempty_string(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(path, f"{field} must be a nonempty string")
    return value


def reject_nonfinite(value: Any, path: Path, location: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            fail(path, f"non-finite numeric value at {location}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_nonfinite(item, path, f"{location}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            reject_nonfinite(item, path, f"{location}.{key}")


def require_completed(payload: dict[str, Any], path: Path) -> None:
    if payload.get("status") != "completed":
        fail(path, "status must be completed")


def validate_decoder_completion(payload: dict[str, Any], path: Path) -> None:
    require_completed(payload, path)
    nonempty_string(payload.get("run_id"), "run_id", path)
    phase = payload.get("phase")
    if phase not in {"selection", "refit"}:
        fail(path, "phase must be selection or refit")
    positive_int(payload.get("global_step"), "global_step", path)
    if "selected_global_step" in payload and payload["selected_global_step"] is not None:
        positive_int(payload["selected_global_step"], "selected_global_step", path)
    train_metrics = object_field(payload, "train_metrics", path)
    finite_number(train_metrics.get("train_loss"), "train_metrics.train_loss", path)
    if phase == "selection":
        finite_number(payload.get("best_metric"), "best_metric", path)
        candidates = payload.get("selection_candidate_steps")
        if not isinstance(candidates, list) or not candidates:
            fail(path, "selection_candidate_steps must be a nonempty list")
        for index, candidate in enumerate(candidates):
            positive_int(candidate, f"selection_candidate_steps[{index}]", path)
    else:
        positive_int(payload.get("selected_global_step"), "selected_global_step", path)
        nonempty_string(payload.get("selection_summary_path"), "selection_summary_path", path)
    if not (path.parent / "adapter" / "adapter_config.json").is_file():
        fail(path, "decoder adapter/adapter_config.json is missing")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_encoder_completion(payload: dict[str, Any], path: Path) -> None:
    require_completed(payload, path)
    nonempty_string(payload.get("run_id"), "run_id", path)
    phase = payload.get("phase")
    if phase not in {"selection", "refit"}:
        fail(path, "phase must be selection or refit")
    positive_int(payload.get("selected_global_step"), "selected_global_step", path)
    positive_int(payload.get("trainer_global_step"), "trainer_global_step", path)
    train_metrics = object_field(payload, "train_metrics", path)
    finite_number(train_metrics.get("train_loss"), "train_metrics.train_loss", path)
    model_hash = payload.get("model_state_sha256")
    if not isinstance(model_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", model_hash):
        fail(path, "model_state_sha256 must be a lowercase SHA-256 hex digest")
    state_path = path.parent / "final_model" / "model.safetensors"
    if not state_path.is_file():
        fail(path, "encoder final_model/model.safetensors is missing")
    if sha256(state_path) != model_hash:
        fail(path, "model_state_sha256 does not match final_model/model.safetensors")
    if phase == "selection":
        selection_metrics = object_field(payload, "selection_metrics", path)
        finite_number(selection_metrics.get("eval_primary_macro_mae"), "selection_metrics.eval_primary_macro_mae", path)
    else:
        config = object_field(payload, "config", path)
        selection_path = nonempty_string(config.get("selection_metadata_path"), "config.selection_metadata_path", path)
        candidate = Path(selection_path)
        if not candidate.is_absolute() or candidate.name != "standard_encoder_training_complete.json" or not candidate.is_file():
            fail(path, "config.selection_metadata_path must name an existing absolute encoder completion artifact")


def validate_aggregate(payload: dict[str, Any], path: Path) -> None:
    require_completed(payload, path)
    metrics = object_field(payload, "metrics", path)
    if "standard-encoder-evals" in path.parts:
        finite_number(metrics.get("eval_primary_macro_mae"), "metrics.eval_primary_macro_mae", path)
    elif "standard-evals" in path.parts:
        finite_number(metrics.get("primary_macro_mae"), "metrics.primary_macro_mae", path)
    else:
        fail(path, "aggregate_metrics.json must be below standard-evals or standard-encoder-evals")


def validate_selected_checkpoint(payload: dict[str, Any], path: Path) -> None:
    require_completed(payload, path)
    if payload.get("phase") != "selection":
        fail(path, "phase must be selection")
    finite_number(payload.get("selected_primary_macro_mae"), "selected_primary_macro_mae", path, nonnegative=True)
    positive_int(payload.get("selected_global_step"), "selected_global_step", path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        fail(path, "candidates must be a nonempty list")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            fail(path, f"candidates[{index}] must be a mapping")
        positive_int(candidate.get("global_step"), f"candidates[{index}].global_step", path)
        finite_number(candidate.get("primary_macro_mae"), f"candidates[{index}].primary_macro_mae", path, nonnegative=True)


for raw_path in raw_paths:
    path = Path(raw_path)
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(path, f"cannot read JSON: {exc}")
    if not isinstance(payload, dict):
        fail(path, "top-level JSON value must be an object")
    reject_nonfinite(payload, path)
    if path.name == "standard_training_complete.json":
        validate_decoder_completion(payload, path)
    elif path.name == "standard_encoder_training_complete.json":
        validate_encoder_completion(payload, path)
    elif path.name == "aggregate_metrics.json":
        validate_aggregate(payload, path)
    elif path.name == "selected_checkpoint.json":
        validate_selected_checkpoint(payload, path)
    else:
        fail(path, "unrecognized stage artifact filename")
PY
}
verify_stage() {
  local step="$1"; shift
  if validate_stage_artifacts "$step" "$@"; then
    ledger completed "$step" "$*"
  else
    ledger failed "$step" "artifact_or_provenance_validation_failed"
    return 1
  fi
}

write_decoder_refit() {
  local mode="$1"; local selection="$2"; local refit="$3"
  local config_mode="${mode//_/-}"
  MODE="$mode" SELECTION="$selection" REFIT="$refit" "$PYTHON" - "$CONFIGS/decoder-$config_mode-refit.json" <<'PY'
import json, os, sys
selection=json.load(open(os.environ["SELECTION"]+"/selected_checkpoint.json", encoding="utf-8")); base=json.load(open(f"{os.environ['RUNTIME_ROOT']}/configs/decoder-{os.environ['MODE'].replace('_','-')}-selection.json", encoding="utf-8")
base.update({"run_id":os.path.basename(os.environ["REFIT"]), "phase":"refit", "output_dir":os.environ["REFIT"], "eval_steps":0, "save_steps":0, "selection_summary_path":os.environ["SELECTION"]+"/selected_checkpoint.json", "selected_global_step":selection["selected_global_step"]})
with open(sys.argv[1],"x",encoding="utf-8") as f: json.dump(base,f,indent=2,sort_keys=True);f.write("\n")
PY
}
write_decoder_eval() { local mode="$1" source="$2" adapter="$3" name="$4"; MODE="$mode" SOURCE="$source" ADAPTER="$adapter" NAME="$name" "$PYTHON" - "$CONFIGS/$name.json" <<'PY'
import json, os, sys
mode=os.environ["MODE"]; source=os.environ["SOURCE"]
p={"run_id":os.environ["NAME"],"mode":mode,"model_path":os.environ["QWEN_MODEL"],"model_revision":os.environ["QWEN_REV"],"adapter_path":os.environ["ADAPTER"],"source":source,"prepared_manifest":os.environ["MANIFEST"],"validation_sha256":"" if source=="selection_dev" else os.environ["VALIDATION_SHA256"],"output_dir":os.environ["ROOT"]+"/outputs/standard-evals/"+os.environ["NAME"],"tensor_parallel_size":int(os.environ["NUM_GPUS"]),"max_model_len":2048 if mode=="direct" else 4096,"max_new_tokens":256 if mode=="direct" else 1536,"gpu_memory_utilization":0.9,"wandb_project":os.environ["WANDB_PROJECT"],"wandb_entity":os.environ["WANDB_ENTITY"] or None}
with open(sys.argv[1],"x",encoding="utf-8") as f: json.dump(p,f,indent=2,sort_keys=True);f.write("\n")
PY
}
run_decoder() {
  local mode="$1"; local selection="$2"; local refit="$3"; local config_mode="${mode//_/-}"
  local selection_step="decoder-$config_mode-selection"
  run_step "$selection_step" "$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" "$ROOT/scripts/train_standard_decoder_sft.py" --config "$CONFIGS/decoder-$config_mode-selection.json"
  verify_stage "$selection_step" "$selection/standard_training_complete.json"
  local -a steps metric_args=()
  mapfile -t steps < <("$PYTHON" - "$selection/standard_training_complete.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(*p["selection_candidate_steps"],sep="\n")
PY
)
  [[ ${#steps[@]} -gt 0 ]] || { echo "no retained decoder checkpoints" >&2; return 1; }
  local step name eval_step
  for step in "${steps[@]}"; do
    name="$(run_name decoder-$config_mode-dev-step-$step)"; eval_step="decoder-$config_mode-dev-step-$step"
    write_decoder_eval "$mode" selection_dev "$selection/checkpoint-$step" "$name"
    run_step "$eval_step" "$PYTHON" "$ROOT/scripts/evaluate_standard_decoder_vllm.py" --config "$CONFIGS/$name.json"
    verify_stage "$eval_step" "$ROOT/outputs/standard-evals/$name/aggregate_metrics.json"
    metric_args+=(--evaluation-metrics "$ROOT/outputs/standard-evals/$name/aggregate_metrics.json")
  done
  local select_step="decoder-$config_mode-select-checkpoint"
  run_step "$select_step" "$PYTHON" "$ROOT/scripts/select_standard_decoder_checkpoint.py" --selection-run-dir "$selection" "${metric_args[@]}"
  verify_stage "$select_step" "$selection/selected_checkpoint.json"
  write_decoder_refit "$mode" "$selection" "$refit"
  local refit_step="decoder-$config_mode-refit"
  run_step "$refit_step" "$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" "$ROOT/scripts/train_standard_decoder_sft.py" --config "$CONFIGS/decoder-$config_mode-refit.json"
  verify_stage "$refit_step" "$refit/standard_training_complete.json"
  local final="$(run_name decoder-$config_mode-final)"; local final_step="decoder-$config_mode-final"
  write_decoder_eval "$mode" frozen_validation "$refit/adapter" "$final"
  run_step "$final_step" "$PYTHON" "$ROOT/scripts/evaluate_standard_decoder_vllm.py" --config "$CONFIGS/$final.json"
  verify_stage "$final_step" "$ROOT/outputs/standard-evals/$final/aggregate_metrics.json"
}
write_encoder_refit() { local tag="$1" selection="$2" refit="$3"; TAG="$tag" SELECTION="$selection" REFIT="$refit" "$PYTHON" - "$CONFIGS/encoder-$tag-refit.json" <<'PY'
import json,os,sys
base=json.load(open(f"{os.environ['RUNTIME_ROOT']}/configs/encoder-{os.environ['TAG']}-selection.json",encoding="utf-8"));base.update({"run_id":os.path.basename(os.environ["REFIT"]),"phase":"refit","output_dir":os.environ["REFIT"],"eval_steps":0,"save_steps":0,"selection_metadata_path":os.environ["SELECTION"]+"/standard_encoder_training_complete.json"})
with open(sys.argv[1],"x",encoding="utf-8") as f:json.dump(base,f,indent=2,sort_keys=True);f.write("\n")
PY
}
run_encoder() {
  local tag="$1"; local selection="$2"; local refit="$3"; local selection_step="encoder-$tag-selection"
  run_step "$selection_step" "$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" "$ROOT/scripts/train_standard_encoder.py" --config "$CONFIGS/encoder-$tag-selection.json"
  verify_stage "$selection_step" "$selection/standard_encoder_training_complete.json"
  write_encoder_refit "$tag" "$selection" "$refit"
  local refit_step="encoder-$tag-refit"
  run_step "$refit_step" "$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" "$ROOT/scripts/train_standard_encoder.py" --config "$CONFIGS/encoder-$tag-refit.json"
  verify_stage "$refit_step" "$refit/standard_encoder_training_complete.json"
  local final="$(run_name encoder-$tag-final)"; FINAL="$final" REFIT="$refit" "$PYTHON" - "$CONFIGS/$final.json" <<'PY'
import json,os,sys
p={"run_id":os.environ["FINAL"],"source":"frozen_validation","training_metadata_path":os.environ["REFIT"]+"/standard_encoder_training_complete.json","prepared_manifest":os.environ["MANIFEST"],"validation_sha256":os.environ["VALIDATION_SHA256"],"output_dir":os.environ["ROOT"]+"/outputs/standard-encoder-evals/"+os.environ["FINAL"],"per_device_eval_batch_size":int(os.environ["ENCODER_BATCH"]),"wandb_project":os.environ["WANDB_PROJECT"],"wandb_entity":os.environ["WANDB_ENTITY"] or None}
with open(sys.argv[1],"x",encoding="utf-8") as f:json.dump(p,f,indent=2,sort_keys=True);f.write("\n")
PY
  local final_step="encoder-$tag-final"
  run_step "$final_step" "$PYTHON" "$ROOT/scripts/evaluate_standard_encoder.py" --config "$CONFIGS/$final.json"
  verify_stage "$final_step" "$ROOT/outputs/standard-encoder-evals/$final/aggregate_metrics.json"
}

ledger started matrix "approved sequential matrix"
run_decoder direct "$DIRECT_SELECTION" "$DIRECT_REFIT"
run_decoder human_feedback "$FEEDBACK_SELECTION" "$FEEDBACK_REFIT"
run_encoder qwen3 "$QWEN3_SELECTION" "$QWEN3_REFIT"
run_encoder nvembed "$NV_SELECTION" "$NV_REFIT"
ledger completed matrix "$RUNTIME_ROOT/matrix_manifest.json"
echo "matrix completed; aggregate-only ledger: $LEDGER"
