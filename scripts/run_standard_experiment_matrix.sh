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
  --resume-run-prefix PREFIX      resume this exact failed matrix runtime; requires
                                  --run-prefix PREFIX and immutable inputs unchanged
  --continue-from-decoder-selection-run ABS_DIR
                                  start a new, provenance-linked continuation from
                                  a verified failed direct-selection parent
  --continue-from-direct-evaluation-runtime ABS_DIR
                                  start a new direct-refit recovery from a verified
                                  failed continuation that completed direct evaluation
  --dry-run                       print planned outputs; do not check paths or write/run anything
  -h, --help                      show this text

Do not use --dry-run as a smoke test. Before a real launch, verify the approved
validated smoke evidence compatible with the selected 0--3 GPU allocation and ensure no other process owns those GPUs.

Resume is intentionally narrow: it reads the existing manifest, verifies its
Git SHA, all immutable selection-config hashes, and every recorded completed
stage's strict artifact contract.  It never rewrites configs, logs, ledgers,
or stage outputs.  Only a recorded-and-verified completed stage is skipped;
failed or not-started stages are retried only when their output path is absent.
An incomplete pre-existing output is refused rather than overwritten.

Continuation is a one-purpose, append-only exception for a verified direct
selection whose downstream vLLM evaluation failed before producing output.  It
creates a new runtime/prefix, records the parent/deviation evidence, never
rewrites the parent, and appends its selected_checkpoint.json only after all
new-prefix source-dev candidate evaluations complete.

Direct-evaluation recovery is narrower still: it accepts only the recorded
cont4-style failed continuation, verifies every completed direct evaluation and
the selected checkpoint, and begins at a new direct-refit output.  It never
resumes or alters the failed runtime or its original selection parent.
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
RESUME_RUN_PREFIX=""
CONTINUE_PARENT_SELECTION=""
CONTINUE_DIRECT_EVALUATION_RUNTIME=""

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
    --resume-run-prefix) need_value "$@"; RESUME_RUN_PREFIX="$2"; shift 2;;
    --continue-from-decoder-selection-run) need_value "$@"; CONTINUE_PARENT_SELECTION="$2"; shift 2;;
    --continue-from-direct-evaluation-runtime) need_value "$@"; CONTINUE_DIRECT_EVALUATION_RUNTIME="$2"; shift 2;;
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
if [[ -n "$RESUME_RUN_PREFIX" ]]; then
  [[ "$RESUME_RUN_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || { echo "resume run prefix must be a safe 1-80 char identifier" >&2; exit 2; }
  [[ "$RESUME_RUN_PREFIX" == "$RUN_PREFIX" ]] || { echo "--resume-run-prefix must exactly match --run-prefix" >&2; exit 2; }
  [[ "$DRY_RUN" == false ]] || { echo "--resume-run-prefix cannot be combined with --dry-run" >&2; exit 2; }
fi
if [[ -n "$CONTINUE_PARENT_SELECTION" ]]; then
  is_absolute "$CONTINUE_PARENT_SELECTION" || { echo "--continue-from-decoder-selection-run requires an absolute path" >&2; exit 2; }
  [[ -z "$RESUME_RUN_PREFIX" ]] || { echo "--continue-from-decoder-selection-run cannot be combined with --resume-run-prefix" >&2; exit 2; }
  [[ "$DRY_RUN" == false ]] || { echo "--continue-from-decoder-selection-run cannot be combined with --dry-run" >&2; exit 2; }
fi
if [[ -n "$CONTINUE_DIRECT_EVALUATION_RUNTIME" ]]; then
  is_absolute "$CONTINUE_DIRECT_EVALUATION_RUNTIME" || { echo "--continue-from-direct-evaluation-runtime requires an absolute path" >&2; exit 2; }
  [[ -z "$RESUME_RUN_PREFIX" && -z "$CONTINUE_PARENT_SELECTION" ]] || {
    echo "--continue-from-direct-evaluation-runtime cannot be combined with resume or selection continuation" >&2; exit 2;
  }
  [[ "$DRY_RUN" == false ]] || { echo "--continue-from-direct-evaluation-runtime cannot be combined with --dry-run" >&2; exit 2; }
fi
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
NEW_DIRECT_SELECTION="$RUNS/$(run_name decoder-direct-selection)"
DIRECT_SELECTION="$NEW_DIRECT_SELECTION"
CONTINUATION_MODE=false
DIRECT_EVALUATION_CONTINUATION_MODE=false
PARENT_RUNTIME_ROOT=""
PARENT_PREFIX=""
if [[ -n "$CONTINUE_PARENT_SELECTION" ]]; then
  [[ -d "$CONTINUE_PARENT_SELECTION" ]] || { echo "continuation parent selection run is missing or not a directory: $CONTINUE_PARENT_SELECTION" >&2; exit 2; }
  CONTINUE_PARENT_SELECTION="$(cd "$CONTINUE_PARENT_SELECTION" && pwd -P)"
  [[ "$(dirname "$CONTINUE_PARENT_SELECTION")" == "$RUNS" ]] || {
    echo "continuation parent selection run must be a direct child of $RUNS" >&2; exit 2;
  }
  parent_base="$(basename "$CONTINUE_PARENT_SELECTION")"
  [[ "$parent_base" == *-decoder-direct-selection ]] || {
    echo "continuation parent must be a direct decoder selection run" >&2; exit 2;
  }
  PARENT_PREFIX="${parent_base%-decoder-direct-selection}"
  [[ "$PARENT_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ && "$parent_base" == "$PARENT_PREFIX-decoder-direct-selection" ]] || {
    echo "continuation parent selection basename is not a canonical run identifier" >&2; exit 2;
  }
  PARENT_RUNTIME_ROOT="$ROOT/outputs/experiment-matrix/$PARENT_PREFIX"
  CONTINUATION_MODE=true
  DIRECT_SELECTION="$CONTINUE_PARENT_SELECTION"
fi
DIRECT_EVALUATION_RUNTIME=""
DIRECT_EVALUATION_PREFIX=""
if [[ -n "$CONTINUE_DIRECT_EVALUATION_RUNTIME" ]]; then
  [[ -d "$CONTINUE_DIRECT_EVALUATION_RUNTIME" ]] || {
    echo "direct-evaluation continuation runtime is missing or not a directory: $CONTINUE_DIRECT_EVALUATION_RUNTIME" >&2; exit 2;
  }
  DIRECT_EVALUATION_RUNTIME="$(cd "$CONTINUE_DIRECT_EVALUATION_RUNTIME" && pwd -P)"
  [[ "$(dirname "$DIRECT_EVALUATION_RUNTIME")" == "$ROOT/outputs/experiment-matrix" ]] || {
    echo "direct-evaluation continuation runtime must be a direct child of $ROOT/outputs/experiment-matrix" >&2; exit 2;
  }
  DIRECT_EVALUATION_PREFIX="$(basename "$DIRECT_EVALUATION_RUNTIME")"
  [[ "$DIRECT_EVALUATION_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || {
    echo "direct-evaluation continuation runtime basename is not a canonical run identifier" >&2; exit 2;
  }
  DIRECT_EVALUATION_CONTINUATION_MODE=true
fi
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
[[ "$RUNTIME_ROOT" == "$ROOT/outputs/"* ]] || { echo "runtime root must be inside ignored $ROOT/outputs" >&2; exit 2; }
git -C "$ROOT" check-ignore -q "$RUNTIME_ROOT" || { echo "runtime root is not ignored by Git: $RUNTIME_ROOT" >&2; exit 2; }
if [[ "$DIRECT_EVALUATION_CONTINUATION_MODE" == true ]]; then
  DIRECT_SELECTION="$($PYTHON - "$DIRECT_EVALUATION_RUNTIME/matrix_manifest.json" <<'PY'
import json, sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    selection = payload["continuation"]["parent_selection_run"]
except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"cannot read direct-evaluation continuation parent selection: {exc}")
if not isinstance(selection, str) or not selection.startswith("/"):
    raise SystemExit("direct-evaluation continuation parent selection must be an absolute path")
print(selection)
PY
  )"
fi
"$PYTHON" - "$NV_REVIEW_JSON" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
required = {"model_id", "revision", "license_acknowledged", "use_case", "reviewer", "outcome", "reviewed_files"}
if not isinstance(payload, dict) or set(payload) != required or payload["model_id"] != "nvidia/NV-Embed-v2" or payload["revision"] != "3fa59658547db50a1e8e3346cf057fd0c77ed6ef" or payload["outcome"] != "approved" or not payload["license_acknowledged"] or not isinstance(payload["reviewed_files"], dict) or not payload["reviewed_files"]:
    raise SystemExit("NV review JSON is not an approved immutable NV-Embed-v2 review")
PY
CONFIGS="$RUNTIME_ROOT/configs"; LOGS="$RUNTIME_ROOT/logs"; LEDGER="$RUNTIME_ROOT/matrix_ledger.jsonl"
RESUME_MODE=false
if [[ -n "$RESUME_RUN_PREFIX" ]]; then
  RESUME_MODE=true
  [[ -d "$RUNTIME_ROOT" ]] || { echo "resume runtime root is missing or not a directory: $RUNTIME_ROOT" >&2; exit 2; }
  [[ -f "$RUNTIME_ROOT/matrix_manifest.json" && -f "$LEDGER" && -d "$CONFIGS" && -d "$LOGS" ]] || {
    echo "resume runtime root must contain matrix_manifest.json, matrix_ledger.jsonl, configs/, and logs/" >&2; exit 2;
  }
else
  [[ ! -e "$RUNTIME_ROOT" ]] || { echo "runtime root already exists; use --resume-run-prefix for the exact failed run or choose a new root: $RUNTIME_ROOT" >&2; exit 2; }
  output_paths=("$NEW_DIRECT_SELECTION" "$DIRECT_REFIT" "$FEEDBACK_SELECTION" "$FEEDBACK_REFIT" "$QWEN3_SELECTION" "$QWEN3_REFIT" "$NV_SELECTION" "$NV_REFIT")
  for path in "${output_paths[@]}"; do
    [[ ! -e "$path" ]] || { echo "run output already exists; choose a new prefix: $path" >&2; exit 2; }
  done
  mkdir -p "$CONFIGS" "$LOGS"
fi
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

export ROOT RUNTIME_ROOT RUN_PREFIX MANIFEST VALIDATION_SHA256 QWEN_MODEL QWEN3_MODEL NV_MODEL NV_REVIEW_JSON NUM_GPUS CUDA_VISIBLE WANDB_PROJECT WANDB_ENTITY DECODER_BATCH DECODER_ACCUM ENCODER_BATCH ENCODER_ACCUM QWEN_REV QWEN3_REV NV_REV DIRECT_SELECTION DIRECT_REFIT FEEDBACK_SELECTION FEEDBACK_REFIT QWEN3_SELECTION QWEN3_REFIT NV_SELECTION NV_REFIT RESUME_MODE CONTINUATION_MODE CONTINUE_PARENT_SELECTION PARENT_RUNTIME_ROOT PARENT_PREFIX DIRECT_EVALUATION_CONTINUATION_MODE DIRECT_EVALUATION_RUNTIME DIRECT_EVALUATION_PREFIX
"$PYTHON" - <<'PY'
import hashlib, json, math, os, subprocess
from pathlib import Path
root = Path(os.environ["RUNTIME_ROOT"]); configs = root / "configs"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump_new(name, payload):
    path = configs / name
    with path.open("x", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True); f.write("\n")

def decoder(mode, output):
    return {"run_id": Path(output).name, "phase":"selection", "mode":mode, "model_path":os.environ["QWEN_MODEL"], "tokenizer_path":os.environ["QWEN_MODEL"], "model_revision":os.environ["QWEN_REV"], "tokenizer_revision":os.environ["QWEN_REV"], "prepared_manifest":os.environ["MANIFEST"], "output_dir":output, "seed":2026, "max_length":2048 if mode == "direct" else 4096, "learning_rate":2e-5, "num_train_epochs":12.0, "per_device_train_batch_size":int(os.environ["DECODER_BATCH"]), "per_device_eval_batch_size":int(os.environ["DECODER_BATCH"]), "gradient_accumulation_steps":int(os.environ["DECODER_ACCUM"]), "eval_steps":100, "save_steps":100, "logging_steps":5, "early_stopping_patience":4, "lora_r":32, "lora_alpha":64, "lora_dropout":0.05, "selection_summary_path":None, "selected_global_step":None, "wandb_project":os.environ["WANDB_PROJECT"], "wandb_entity":os.environ["WANDB_ENTITY"] or None}

def encoder(backbone, output):
    nv = backbone == "nv_embed_v2"; model = os.environ["NV_MODEL"] if nv else os.environ["QWEN3_MODEL"]; revision = os.environ["NV_REV"] if nv else os.environ["QWEN3_REV"]
    return {"run_id":Path(output).name, "phase":"selection", "backbone":backbone, "model_id":"nvidia/NV-Embed-v2" if nv else "Qwen/Qwen3-Embedding-8B", "model_revision":revision, "tokenizer_revision":revision, "model_path":model, "prepared_manifest":os.environ["MANIFEST"], "output_dir":output, "max_length":2048, "seed":2026, "learning_rate":1e-4, "weight_decay":0.01, "warmup_ratio":0.05, "num_train_epochs":20.0, "per_device_train_batch_size":int(os.environ["ENCODER_BATCH"]), "per_device_eval_batch_size":int(os.environ["ENCODER_BATCH"]), "gradient_accumulation_steps":int(os.environ["ENCODER_ACCUM"]), "eval_steps":100, "save_steps":100, "logging_steps":5, "early_stopping_patience":3, "lora_r":16, "lora_alpha":32, "lora_dropout":0.05, "lora_target_modules":["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], "nv_snapshot_dir":model if nv else None, "nv_review":json.load(open(os.environ["NV_REVIEW_JSON"], encoding="utf-8")) if nv else None, "selection_metadata_path":None, "wandb_project":os.environ["WANDB_PROJECT"], "wandb_entity":os.environ["WANDB_ENTITY"] or None}


def read_object(path, label):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"continuation cannot read {label}: {path}: {exc}")
    if not isinstance(payload, dict):
        raise SystemExit(f"continuation {label} must be a JSON object: {path}")
    return payload


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SystemExit(f"continuation parent {label} must be finite")


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemExit(f"continuation parent {label} must be a positive integer")


def validate_continuation_parent():
    """Validate the bounded reusable parent before any stage can reach GPUs."""
    parent = Path(os.environ["CONTINUE_PARENT_SELECTION"])
    parent_runtime = Path(os.environ["PARENT_RUNTIME_ROOT"])
    parent_config_path = parent_runtime / "configs" / "decoder-direct-selection.json"
    parent_manifest_path = parent_runtime / "matrix_manifest.json"
    parent_ledger_path = parent_runtime / "matrix_ledger.jsonl"
    parent_completion_path = parent / "standard_training_complete.json"
    for path, label in ((parent_runtime, "runtime root"), (parent_config_path, "direct selection config"),
                        (parent_manifest_path, "runtime manifest"), (parent_ledger_path, "runtime ledger"),
                        (parent_completion_path, "training completion")):
        if not path.is_file() and not path.is_dir():
            raise SystemExit(f"continuation parent {label} is missing: {path}")
    parent_manifest = read_object(parent_manifest_path, "runtime manifest")
    expected_parent_git = "dfa3c343b6fb58bf33835eacf730df1306ac0b5a"
    if parent_manifest.get("run_prefix") != os.environ["PARENT_PREFIX"] or parent_manifest.get("git_sha") != expected_parent_git:
        raise SystemExit("continuation parent runtime is not the approved pre-fix direct-selection run")
    for key, expected in {
        "prepared_manifest": os.environ["MANIFEST"], "validation_sha256": os.environ["VALIDATION_SHA256"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE"], "num_gpus": int(os.environ["NUM_GPUS"]),
    }.items():
        if parent_manifest.get(key) != expected:
            raise SystemExit(f"continuation parent manifest identity mismatch for {key}")
    manifest_path = Path(os.environ["MANIFEST"])
    try:
        relative_manifest = manifest_path.relative_to(Path(os.environ["ROOT"]))
        expected_manifest_bytes = subprocess.check_output(
            ["git", "-C", os.environ["ROOT"], "show", f"{expected_parent_git}:{relative_manifest.as_posix()}"],
        )
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"continuation cannot prove parent data manifest identity from approved Git content: {exc}")
    expected_manifest_sha = hashlib.sha256(expected_manifest_bytes).hexdigest()
    if sha256(manifest_path) != expected_manifest_sha:
        raise SystemExit("continuation parent data manifest differs from the approved pre-fix Git content")
    parent_config = read_object(parent_config_path, "direct selection config")
    selection_hashes = parent_manifest.get("selection_configs")
    if not isinstance(selection_hashes, dict) or selection_hashes.get("decoder-direct-selection.json") != sha256(parent_config_path):
        raise SystemExit("continuation parent direct selection config hash does not match its manifest")
    if not isinstance(parent_config.get("wandb_project"), str) or not parent_config["wandb_project"] or parent_config.get("wandb_entity") not in (None, "") and not isinstance(parent_config.get("wandb_entity"), str):
        raise SystemExit("continuation parent W&B identity is malformed")
    expected_parent_config = decoder("direct", str(parent))
    expected_parent_config["wandb_project"] = parent_config["wandb_project"]
    expected_parent_config["wandb_entity"] = parent_config.get("wandb_entity")
    if parent_config != expected_parent_config:
        raise SystemExit("continuation parent direct config differs from canonical architecture/data/model/batch/seed identity")
    selected_checkpoint = parent / "selected_checkpoint.json"
    if selected_checkpoint.exists():
        raise SystemExit("continuation parent already has selected_checkpoint.json; refusing to overwrite parent selection")
    completion = read_object(parent_completion_path, "training completion")
    if completion.get("status") != "completed" or completion.get("phase") != "selection" or completion.get("mode") != "direct" or completion.get("run_id") != parent.name:
        raise SystemExit("continuation parent completion is not a completed direct selection")
    positive(completion.get("global_step"), "global_step")
    finite(completion.get("best_metric"), "best_metric")
    metrics = completion.get("train_metrics")
    if not isinstance(metrics, dict):
        raise SystemExit("continuation parent train_metrics must be an object")
    finite(metrics.get("train_loss"), "train_metrics.train_loss")
    if completion.get("config") != parent_config:
        raise SystemExit("continuation parent completion config does not exactly bind to its selection config")
    identity = json.loads(json.dumps(parent_config, ensure_ascii=False))
    for key in (
        "run_id", "phase", "output_dir", "selection_summary_path", "selected_global_step",
        "num_train_epochs", "eval_steps", "save_steps", "logging_steps", "early_stopping_patience",
        "wandb_project", "wandb_entity",
    ):
        identity.pop(key)
    if completion.get("identity") != identity:
        raise SystemExit("continuation parent completion identity does not bind to its selection config")
    if completion.get("model_revision") != parent_config["model_revision"] or completion.get("tokenizer_revision") != parent_config["tokenizer_revision"]:
        raise SystemExit("continuation parent completion model/tokenizer revisions do not match its config")
    steps = completion.get("selection_candidate_steps")
    if not isinstance(steps, list) or not steps or len(set(steps)) != len(steps):
        raise SystemExit("continuation parent selection_candidate_steps must be a nonempty unique list")
    adapter_configs = {}
    for step in steps:
        positive(step, "selection_candidate_steps entry")
        if step > completion["global_step"]:
            raise SystemExit("continuation parent candidate checkpoint exceeds completed global_step")
        checkpoint = parent / f"checkpoint-{step}"
        config = checkpoint / "adapter_config.json"
        model = checkpoint / "adapter_model.safetensors"
        if not checkpoint.is_dir() or not config.is_file() or not model.is_file():
            raise SystemExit(f"continuation parent checkpoint adapter is incomplete: {checkpoint}")
        new_eval = Path(os.environ["ROOT"]) / "outputs" / "standard-evals" / f"{os.environ['RUN_PREFIX']}-decoder-direct-dev-step-{step}"
        if new_eval.exists():
            raise SystemExit(f"continuation new-prefix direct evaluation output already exists: {new_eval}")
        adapter_configs[str(step)] = sha256(config)
    old_eval_roots = sorted((Path(os.environ["ROOT"]) / "outputs" / "standard-evals").glob(f"{os.environ['PARENT_PREFIX']}-decoder-direct-dev-step-*"))
    if old_eval_roots:
        raise SystemExit(f"continuation parent already has direct source-dev evaluator output: {old_eval_roots[0]}")
    final_adapter_config = parent / "adapter" / "adapter_config.json"
    final_adapter_model = parent / "adapter" / "adapter_model.safetensors"
    if not final_adapter_config.is_file() or not final_adapter_model.is_file():
        raise SystemExit("continuation parent final adapter is incomplete")
    first_candidate_step = f"decoder-direct-dev-step-{steps[0]}"
    saw_first_candidate_start = False
    failed = None
    try:
        lines = parent_ledger_path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise ValueError("not an object")
            if entry.get("step") == first_candidate_step:
                if entry.get("status") == "started":
                    saw_first_candidate_start = True
                elif entry.get("status") == "failed" and saw_first_candidate_start:
                    failed = entry
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"continuation cannot parse parent ledger: {exc}")
    if failed is None or not isinstance(failed.get("detail"), str):
        raise SystemExit("continuation parent lacks the required started-then-failed first direct vLLM candidate record")
    failure_log = Path(failed["detail"])
    expected_log = parent_runtime / "logs" / f"{failed['step']}.log"
    if failure_log != expected_log or not failure_log.is_file():
        raise SystemExit("continuation parent failed vLLM log is missing or does not match its ledger")
    fix_commit = "68a3b4ff13ea062aad5b7f636b183f4efff11fda"
    base_commit = "c46cf0e"
    try:
        changed = subprocess.check_output(["git", "-C", os.environ["ROOT"], "diff", "--name-only", f"{expected_parent_git}..{fix_commit}"], text=True).splitlines()
        diff = subprocess.check_output(["git", "-C", os.environ["ROOT"], "diff", "--no-ext-diff", f"{expected_parent_git}..{fix_commit}", "--", "scripts/evaluate_standard_decoder_vllm.py"], text=True)
        subprocess.run(["git", "-C", os.environ["ROOT"], "merge-base", "--is-ancestor", fix_commit, current_git_sha], check=True)
        subprocess.run(["git", "-C", os.environ["ROOT"], "merge-base", "--is-ancestor", base_commit, current_git_sha], check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"continuation cannot verify the approved evaluator fix lineage: {exc}")
    if set(changed) != {"scripts/evaluate_standard_decoder_vllm.py", "tests/test_standard_decoder_stack.py"} or not diff:
        raise SystemExit("continuation evaluator fix diff is not the approved narrow entrypoint-only production change plus regression test")
    return parent_config, {
        "kind": "bounded_decoder_direct_selection_reuse",
        "rationale": "The parent completed direct selection; its first vLLM source-dev evaluation failed before output because spawn re-imported an unguarded CLI entry point. The continuation reuses only the verified selection and evaluates all candidates under the guarded entry point.",
        "parent_runtime_root": str(parent_runtime), "parent_selection_run": str(parent),
        "parent_git_sha": expected_parent_git, "current_git_sha": current_git_sha,
        "continuation_base_git_sha": base_commit, "entrypoint_fix_git_sha": fix_commit,
        "entrypoint_fix_changed_files": changed,
        "entrypoint_fix_diff": diff, "entrypoint_fix_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "original_failed_vllm_stage": failed["step"], "original_failed_vllm_log": str(failure_log),
        "parent_artifacts": {
            "completion_path": str(parent_completion_path), "completion_sha256": sha256(parent_completion_path),
            "selection_config_path": str(parent_config_path), "selection_config_sha256": sha256(parent_config_path),
            "data_manifest_sha256": expected_manifest_sha,
            "data_manifest_identity_source": f"git:{expected_parent_git}:{relative_manifest.as_posix()}",
            "candidate_checkpoints": steps, "candidate_adapter_config_sha256": adapter_configs,
            "final_adapter_config_sha256": sha256(final_adapter_config),
        },
    }


def validate_direct_evaluation_continuation():
    """Reuse only a fully verified cont4-style direct-evaluation prefix.

    This is intentionally not a generic resume path.  The known parent is a
    continuation that reached checkpoint selection and then stopped before it
    could create a direct-refit config because the historical heredoc Python
    source has a syntax error.  Every parent file remains read-only.
    """
    continuation_runtime = Path(os.environ["DIRECT_EVALUATION_RUNTIME"])
    continuation_prefix = os.environ["DIRECT_EVALUATION_PREFIX"]
    parent = Path(os.environ["DIRECT_SELECTION"])
    matrix_root = Path(os.environ["ROOT"]) / "outputs" / "experiment-matrix"
    runs = Path(os.environ["ROOT"]) / "outputs" / "standard-runs"
    evals = Path(os.environ["ROOT"]) / "outputs" / "standard-evals"
    manifest_path = continuation_runtime / "matrix_manifest.json"
    ledger_path = continuation_runtime / "matrix_ledger.jsonl"
    configs_dir = continuation_runtime / "configs"
    logs_dir = continuation_runtime / "logs"
    for path, label in ((continuation_runtime, "direct-evaluation runtime"), (manifest_path, "runtime manifest"),
                        (ledger_path, "runtime ledger"), (configs_dir, "runtime configs"), (logs_dir, "runtime logs")):
        if not path.is_file() and not path.is_dir():
            raise SystemExit(f"direct-evaluation continuation {label} is missing: {path}")
    if continuation_runtime.parent != matrix_root or continuation_runtime.name != continuation_prefix:
        raise SystemExit("direct-evaluation continuation runtime is not a canonical experiment-matrix child")
    manifest = read_object(manifest_path, "direct-evaluation runtime manifest")
    expected_cont4_git = "4c6ed913d755c78ee2a3dabbc02c4cccc9ed0c99"
    required_manifest = {
        "schema_version": 3,
        "status": "started",
        "run_prefix": continuation_prefix,
        "git_sha": expected_cont4_git,
        "prepared_manifest": os.environ["MANIFEST"],
        "validation_sha256": os.environ["VALIDATION_SHA256"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE"],
        "num_gpus": int(os.environ["NUM_GPUS"]),
        "privacy": "aggregate_only_no_restricted_rows_or_credentials",
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise SystemExit(f"direct-evaluation continuation manifest identity mismatch for {key}")
    if manifest.get("prepared_manifest_sha256") != sha256(Path(os.environ["MANIFEST"])):
        raise SystemExit("direct-evaluation continuation prepared manifest checksum mismatch")
    if manifest.get("nv_review_sha256") != sha256(Path(os.environ["NV_REVIEW_JSON"])):
        raise SystemExit("direct-evaluation continuation NV review checksum mismatch")
    continuation = manifest.get("continuation")
    if not isinstance(continuation, dict) or continuation.get("kind") != "bounded_decoder_direct_selection_reuse":
        raise SystemExit("direct-evaluation continuation lacks the approved direct-selection reuse lineage")
    if continuation.get("current_git_sha") != expected_cont4_git or continuation.get("parent_selection_run") != str(parent):
        raise SystemExit("direct-evaluation continuation lineage does not bind to the recorded parent selection")
    parent_runtime = Path(continuation.get("parent_runtime_root", ""))
    if parent_runtime.parent != matrix_root or not parent_runtime.is_dir() or parent.parent != runs:
        raise SystemExit("direct-evaluation continuation parent paths are not canonical")
    parent_config_path = configs_dir / "decoder-direct-selection.json"
    parent_config = read_object(parent_config_path, "direct-evaluation direct selection config")
    selection_hashes = manifest.get("selection_configs")
    expected_selection_config_names = {
        "decoder-direct-selection.json", "decoder-human-feedback-selection.json",
        "encoder-qwen3-selection.json", "encoder-nvembed-selection.json",
    }
    if not isinstance(selection_hashes, dict) or set(selection_hashes) != expected_selection_config_names:
        raise SystemExit("direct-evaluation continuation selection config hash set is incomplete")
    for name in expected_selection_config_names:
        path = configs_dir / name
        if not path.is_file() or selection_hashes[name] != sha256(path):
            raise SystemExit(f"direct-evaluation continuation selection config hash mismatch: {path}")
    if not isinstance(parent_config.get("wandb_project"), str) or not parent_config["wandb_project"]:
        raise SystemExit("direct-evaluation continuation direct config W&B project is malformed")
    if parent_config.get("wandb_entity") is not None and not isinstance(parent_config["wandb_entity"], str):
        raise SystemExit("direct-evaluation continuation direct config W&B entity is malformed")
    expected_parent_config = decoder("direct", str(parent))
    expected_parent_config["wandb_project"] = parent_config["wandb_project"]
    expected_parent_config["wandb_entity"] = parent_config.get("wandb_entity")
    if parent_config != expected_parent_config:
        raise SystemExit("direct-evaluation continuation direct config differs from canonical model/data/batch/seed identity")
    artifacts = continuation.get("parent_artifacts")
    if not isinstance(artifacts, dict):
        raise SystemExit("direct-evaluation continuation parent artifacts are missing")
    original_config_path = Path(artifacts.get("selection_config_path", ""))
    completion_path = parent / "standard_training_complete.json"
    if original_config_path != parent_runtime / "configs" / "decoder-direct-selection.json" or not original_config_path.is_file():
        raise SystemExit("direct-evaluation continuation original selection config path is invalid")
    if artifacts.get("selection_config_sha256") != sha256(original_config_path) or original_config_path.read_bytes() != parent_config_path.read_bytes():
        raise SystemExit("direct-evaluation continuation original selection config is not immutable")
    if artifacts.get("completion_path") != str(completion_path) or not completion_path.is_file() or artifacts.get("completion_sha256") != sha256(completion_path):
        raise SystemExit("direct-evaluation continuation parent selection completion is not immutable")
    completion = read_object(completion_path, "parent selection completion")
    if completion.get("status") != "completed" or completion.get("phase") != "selection" or completion.get("mode") != "direct" or completion.get("run_id") != parent.name:
        raise SystemExit("direct-evaluation continuation parent completion is not a completed direct selection")
    if completion.get("config") != parent_config:
        raise SystemExit("direct-evaluation continuation parent completion does not bind to the direct config")
    steps = completion.get("selection_candidate_steps")
    if not isinstance(steps, list) or not steps or len(set(steps)) != len(steps):
        raise SystemExit("direct-evaluation continuation candidate step set is invalid")
    for step in steps:
        positive(step, "selection_candidate_steps entry")
        if step > completion.get("global_step", 0):
            raise SystemExit("direct-evaluation continuation candidate exceeds selection global step")
    if artifacts.get("candidate_checkpoints") != steps:
        raise SystemExit("direct-evaluation continuation lineage candidate steps differ from parent completion")
    adapter_hashes = artifacts.get("candidate_adapter_config_sha256")
    if not isinstance(adapter_hashes, dict) or set(adapter_hashes) != {str(step) for step in steps}:
        raise SystemExit("direct-evaluation continuation candidate adapter hash set is invalid")
    for step in steps:
        checkpoint = parent / f"checkpoint-{step}"
        adapter_config = checkpoint / "adapter_config.json"
        adapter_model = checkpoint / "adapter_model.safetensors"
        if not checkpoint.is_dir() or not adapter_config.is_file() or not adapter_model.is_file() or adapter_hashes[str(step)] != sha256(adapter_config):
            raise SystemExit(f"direct-evaluation continuation parent checkpoint is incomplete or changed: {checkpoint}")
    final_adapter_config = parent / "adapter" / "adapter_config.json"
    final_adapter_model = parent / "adapter" / "adapter_model.safetensors"
    if not final_adapter_config.is_file() or not final_adapter_model.is_file() or artifacts.get("final_adapter_config_sha256") != sha256(final_adapter_config):
        raise SystemExit("direct-evaluation continuation parent final adapter is incomplete or changed")

    try:
        ledger_entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"direct-evaluation continuation cannot parse runtime ledger: {exc}")
    if not all(isinstance(entry, dict) for entry in ledger_entries):
        raise SystemExit("direct-evaluation continuation ledger contains a non-object entry")
    def step_statuses(name):
        return [entry.get("status") for entry in ledger_entries if entry.get("step") == name]

    metric_provenance = []
    selected_candidates = []
    for step in steps:
        name = f"{continuation_prefix}-decoder-direct-dev-step-{step}"
        stage = f"decoder-direct-dev-step-{step}"
        config_path = configs_dir / f"{name}.json"
        checkpoint = parent / f"checkpoint-{step}"
        output_dir = evals / name
        metrics_path = output_dir / "aggregate_metrics.json"
        expected_config = {
            "run_id": name, "mode": "direct", "model_path": os.environ["QWEN_MODEL"],
            "model_revision": os.environ["QWEN_REV"], "adapter_path": str(checkpoint),
            "source": "selection_dev", "prepared_manifest": os.environ["MANIFEST"],
            "validation_sha256": "", "output_dir": str(output_dir),
            "tensor_parallel_size": int(os.environ["NUM_GPUS"]), "max_model_len": 2048,
            "max_new_tokens": 256, "gpu_memory_utilization": 0.9, "enforce_eager": True,
            "disable_flashinfer_sampler": True, "max_lora_rank": 32,
            "wandb_project": parent_config["wandb_project"], "wandb_entity": parent_config.get("wandb_entity"),
        }
        if not config_path.is_file() or read_object(config_path, f"direct evaluation config {step}") != expected_config:
            raise SystemExit(f"direct-evaluation continuation config is inconsistent for candidate {step}")
        result = read_object(metrics_path, f"direct evaluation metrics {step}")
        if step_statuses(stage) != ["started", "completed"]:
            raise SystemExit(f"direct-evaluation continuation ledger is incomplete for candidate {step}")
        if result.get("status") != "completed" or result.get("run_id") != name or result.get("mode") != "direct" or result.get("source") != "selection_dev":
            raise SystemExit(f"direct-evaluation continuation metrics provenance is incomplete for candidate {step}")
        if result.get("model_revision") != os.environ["QWEN_REV"] or result.get("adapter_path") != str(checkpoint.resolve()) or result.get("adapter_global_step") != step:
            raise SystemExit(f"direct-evaluation continuation metrics adapter provenance is inconsistent for candidate {step}")
        if result.get("config") != expected_config:
            raise SystemExit(f"direct-evaluation continuation metrics config provenance is inconsistent for candidate {step}")
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise SystemExit(f"direct-evaluation continuation metrics are missing for candidate {step}")
        primary = metrics.get("primary_macro_mae")
        finite(primary, f"candidate {step} primary_macro_mae")
        if float(primary) < 0:
            raise SystemExit(f"direct-evaluation continuation candidate {step} primary_macro_mae must be nonnegative")
        metric_provenance.append({
            "global_step": step, "evaluation_config_path": str(config_path), "evaluation_config_sha256": sha256(config_path),
            "aggregate_metrics_path": str(metrics_path), "aggregate_metrics_sha256": sha256(metrics_path),
            "primary_macro_mae": float(primary),
        })
        selected_candidates.append({"global_step": step, "primary_macro_mae": float(primary), "evaluation_metrics_path": str(metrics_path)})

    selected_path = parent / "selected_checkpoint.json"
    if step_statuses("decoder-direct-select-checkpoint") != ["started", "completed"]:
        raise SystemExit("direct-evaluation continuation selection ledger is incomplete")
    selected = read_object(selected_path, "parent selected checkpoint")
    if selected.get("status") != "completed" or selected.get("phase") != "selection" or selected.get("selection_run_id") != parent.name or selected.get("mode") != "direct":
        raise SystemExit("direct-evaluation continuation selected checkpoint provenance is incomplete")
    if selected.get("model_revision") != os.environ["QWEN_REV"] or selected.get("tokenizer_revision") != os.environ["QWEN_REV"]:
        raise SystemExit("direct-evaluation continuation selected checkpoint model identity mismatch")
    if selected.get("candidates") != selected_candidates:
        raise SystemExit("direct-evaluation continuation selected checkpoint candidates do not exactly match aggregate metrics")
    winner = min(selected_candidates, key=lambda item: (item["primary_macro_mae"], item["global_step"]))
    if selected.get("selected_global_step") != winner["global_step"] or selected.get("selected_primary_macro_mae") != winner["primary_macro_mae"]:
        raise SystemExit("direct-evaluation continuation selected checkpoint is not the deterministic aggregate-metric winner")

    previous_refit = runs / f"{continuation_prefix}-decoder-direct-refit"
    if previous_refit.exists() or (configs_dir / "decoder-direct-refit.json").exists() or step_statuses("decoder-direct-refit"):
        raise SystemExit("direct-evaluation continuation has pre-existing direct-refit artifacts or launch records")
    try:
        historical_script = subprocess.check_output(
            ["git", "-C", os.environ["ROOT"], "show", f"{expected_cont4_git}:scripts/run_standard_experiment_matrix.sh"], text=True,
        )
        subprocess.run(["git", "-C", os.environ["ROOT"], "merge-base", "--is-ancestor", expected_cont4_git, current_git_sha], check=True)
        historical_refit = historical_script.split("write_decoder_refit() {", 1)[1].split("\nPY\n}", 1)[0].split("<<'PY'\n", 1)[1]
        compile(historical_refit, f"git:{expected_cont4_git}:write_decoder_refit", "exec")
    except SyntaxError as exc:
        if "was never closed" not in str(exc):
            raise SystemExit(f"direct-evaluation continuation historical refit failure is not the approved SyntaxError: {exc}")
        syntax_error = f"{exc.__class__.__name__}: {exc.msg}"
    except (IndexError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"direct-evaluation continuation cannot prove historical refit failure: {exc}")
    else:
        raise SystemExit("direct-evaluation continuation historical refit generator is unexpectedly valid")
    return parent_config, {
        "kind": "bounded_direct_evaluation_refit_recovery",
        "rationale": "The recorded continuation completed all direct source-dev evaluations and deterministic checkpoint selection, then stopped before direct refit because its historical shell-generated refit config Python was syntactically invalid. This new runtime reuses only those verified completed aggregate artifacts and starts at a new direct refit.",
        "direct_evaluation_runtime": str(continuation_runtime),
        "direct_evaluation_runtime_manifest_sha256": sha256(manifest_path),
        "direct_evaluation_runtime_ledger_sha256": sha256(ledger_path),
        "direct_evaluation_runtime_git_sha": expected_cont4_git,
        "parent_selection_run": str(parent),
        "parent_selected_checkpoint_path": str(selected_path),
        "parent_selected_checkpoint_sha256": sha256(selected_path),
        "selected_global_step": winner["global_step"],
        "selected_primary_macro_mae": winner["primary_macro_mae"],
        "completed_direct_evaluations": metric_provenance,
        "historical_refit_failure": {
            "kind": "shell_generated_decoder_refit_config_syntax_error",
            "historical_script_git_sha": expected_cont4_git,
            "exception": syntax_error,
            "historical_script_sha256": hashlib.sha256(historical_script.encode("utf-8")).hexdigest(),
            "parent_refit_config_absent": True,
            "parent_refit_output_absent": True,
            "parent_refit_ledger_absent": True,
        },
    }

current_git_sha = subprocess.check_output(["git", "-C", os.environ["ROOT"], "rev-parse", "HEAD"], text=True).strip()
continuation = None
direct_selection_config = decoder("direct", os.environ["DIRECT_SELECTION"])
if os.environ["CONTINUATION_MODE"] == "true":
    direct_selection_config, continuation = validate_continuation_parent()
elif os.environ["DIRECT_EVALUATION_CONTINUATION_MODE"] == "true":
    direct_selection_config, continuation = validate_direct_evaluation_continuation()
expected_configs = {
    "decoder-direct-selection.json": direct_selection_config,
    "decoder-human-feedback-selection.json": decoder("human_feedback", os.environ["FEEDBACK_SELECTION"]),
    "encoder-qwen3-selection.json": encoder("qwen3_embedding", os.environ["QWEN3_SELECTION"]),
    "encoder-nvembed-selection.json": encoder("nv_embed_v2", os.environ["NV_SELECTION"]),
}
if os.environ["RESUME_MODE"] == "false":
    for name, payload in expected_configs.items():
        dump_new(name, payload)
    manifest = {
        "schema_version": 4 if os.environ["DIRECT_EVALUATION_CONTINUATION_MODE"] == "true" else (3 if continuation else 2),
        "status":"started", "run_prefix":os.environ["RUN_PREFIX"], "git_sha":current_git_sha,
        "prepared_manifest":os.environ["MANIFEST"], "prepared_manifest_sha256":sha256(Path(os.environ["MANIFEST"])),
        "nv_review_sha256":sha256(Path(os.environ["NV_REVIEW_JSON"])),
        "validation_sha256":os.environ["VALIDATION_SHA256"], "cuda_visible_devices":os.environ["CUDA_VISIBLE"],
        "num_gpus":int(os.environ["NUM_GPUS"]), "privacy":"aggregate_only_no_restricted_rows_or_credentials",
        "selection_configs":{name:sha256(configs / name) for name in expected_configs},
    }
    if continuation:
        manifest["continuation"] = continuation
    with (root / "matrix_manifest.json").open("x", encoding="utf-8") as f: json.dump(manifest, f, indent=2, sort_keys=True); f.write("\n")
else:
    manifest_path = root / "matrix_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read resume matrix manifest: {exc}")
    if not isinstance(manifest, dict):
        raise SystemExit("resume matrix manifest must be a JSON object")
    required = {
        "status":"started", "run_prefix":os.environ["RUN_PREFIX"], "git_sha":current_git_sha,
        "prepared_manifest":os.environ["MANIFEST"], "validation_sha256":os.environ["VALIDATION_SHA256"],
        "cuda_visible_devices":os.environ["CUDA_VISIBLE"], "num_gpus":int(os.environ["NUM_GPUS"]),
        "privacy":"aggregate_only_no_restricted_rows_or_credentials",
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise SystemExit(f"resume immutable manifest mismatch for {key}: refusing changed config/model/data/Git input")
    if "prepared_manifest_sha256" in manifest and manifest["prepared_manifest_sha256"] != sha256(Path(os.environ["MANIFEST"])):
        raise SystemExit("resume prepared manifest checksum changed: refusing changed data input")
    if "nv_review_sha256" in manifest and manifest["nv_review_sha256"] != sha256(Path(os.environ["NV_REVIEW_JSON"])):
        raise SystemExit("resume NV review checksum changed: refusing changed model-review input")
    recorded_hashes = manifest.get("selection_configs")
    if not isinstance(recorded_hashes, dict) or set(recorded_hashes) != set(expected_configs):
        raise SystemExit("resume selection config hash set is incomplete or changed")
    for name, expected in expected_configs.items():
        path = configs / name
        if not path.is_file() or not isinstance(recorded_hashes[name], str) or len(recorded_hashes[name]) != 64:
            raise SystemExit(f"resume immutable selection config is missing or malformed: {path}")
        if sha256(path) != recorded_hashes[name]:
            raise SystemExit(f"resume immutable selection config hash mismatch: {path}")
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"resume immutable selection config is invalid JSON: {path}: {exc}")
        if observed != expected:
            raise SystemExit(f"resume immutable selection config disagrees with supplied inputs: {path}")
PY

ledger() { local status="$1" step="$2" detail="${3:-}"; STATUS="$status" STEP="$step" DETAIL="$detail" "$PYTHON" - "$LEDGER" <<'PY'
import json, os, sys, time
entry={"time_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status":os.environ["STATUS"], "step":os.environ["STEP"], "detail":os.environ["DETAIL"], "privacy":"aggregate_only"}
with open(sys.argv[1], "a", encoding="utf-8") as f: f.write(json.dumps(entry, ensure_ascii=True, sort_keys=True)+"\n")
PY
}
record_resume_lineage() {
  [[ "$RESUME_MODE" == true ]] || return 0
  "$PYTHON" - "$RUNTIME_ROOT/resume_lineage.jsonl" "$RUNTIME_ROOT/matrix_manifest.json" "$LEDGER" <<'PY'
import hashlib, json, os, subprocess, sys, time
lineage, manifest, ledger = map(os.fsencode, sys.argv[1:])
def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()
entry = {
    "event": "resume",
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "run_prefix": os.environ["RUN_PREFIX"],
    "immutable_manifest_sha256": digest(manifest),
    "immutable_git_sha": json.load(open(manifest, encoding="utf-8"))["git_sha"],
    "resume_git_sha": subprocess.check_output(["git", "-C", os.environ["ROOT"], "rev-parse", "HEAD"], text=True).strip(),
    "ledger_sha256_before_resume": digest(ledger),
    "privacy": "aggregate_only",
}
with open(lineage, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")
PY
  ledger resumed matrix "$RUNTIME_ROOT/resume_lineage.jsonl"
}
stage_log_path() {
  local step="$1" base="$LOGS/$step.log" attempt=1
  if [[ ! -e "$base" ]]; then
    printf '%s' "$base"
    return 0
  fi
  while [[ -e "$LOGS/$step.attempt-$attempt.log" ]]; do ((attempt++)); done
  printf '%s' "$LOGS/$step.attempt-$attempt.log"
}
assert_stage_output_absent() {
  local step="$1"; shift
  local path
  for path in "$@"; do
    [[ ! -e "$path" ]] || {
      echo "stage $step has a pre-existing incomplete output; refusing to overwrite or delete: $path" >&2
      return 1
    }
  done
}
run_step() {
  local step="$1"; shift
  local log
  log="$(stage_log_path "$step")"
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
maybe_skip_verified_stage() {
  local step="$1"; shift
  if [[ "$CONTINUATION_MODE" == true && "$step" == "decoder-direct-selection" ]]; then
    if ! validate_stage_artifacts "$step" "$@"; then
      echo "continuation parent direct selection failed strict artifact validation" >&2
      return 2
    fi
    ledger reused_verified_parent "$step" "$*"
    return 0
  fi
  [[ "$RESUME_MODE" == true ]] || return 1
  local ledger_result status
  if ledger_result="$("$PYTHON" - "$LEDGER" "$step" <<'PY'
import json, sys
ledger, step = sys.argv[1:]
latest = None
try:
    with open(ledger, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            if not isinstance(entry, dict) or not isinstance(entry.get("step"), str) or not isinstance(entry.get("status"), str):
                raise SystemExit(f"malformed ledger entry at line {number}")
            if entry["step"] == step:
                latest = entry["status"]
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"cannot read resume ledger: {exc}")
if latest in {"completed", "skipped_verified"}:
    print("verified")
    raise SystemExit(0)
raise SystemExit(1)
PY
  )"; then
    status=0
  else
    status=$?
  fi
  if (( status == 1 )); then
    return 1
  elif (( status != 0 )); then
    echo "cannot safely determine resume status for $step; refusing to rerun it" >&2
    return 2
  fi
  if ! validate_stage_artifacts "$step" "$@"; then
    ledger failed "$step" "resume_recorded_completed_artifact_validation_failed"
    echo "recorded completed stage $step failed strict resume validation" >&2
    return 2
  fi
  ledger skipped_verified "$step" "$*"
  return 0
}
execute_stage() {
  local step="$1" artifact="$2" output="$3" skip_status
  shift 3
  if maybe_skip_verified_stage "$step" "$artifact"; then
    return 0
  else
    skip_status=$?
  fi
  if (( skip_status != 1 )); then
    return 1
  fi
  if ! assert_stage_output_absent "$step" "$output"; then
    ledger failed "$step" "preexisting_incomplete_output_refusal:$output"
    return 1
  fi
  run_step "$step" "$@"
  verify_stage "$step" "$artifact"
}

write_decoder_refit() {
  local mode="$1"; local selection="$2"; local refit="$3"
  local config_mode selection_config
  config_mode="${mode//_/-}"
  selection_config="$CONFIGS/decoder-$config_mode-selection.json"
  MODE="$mode" SELECTION="$selection" REFIT="$refit" SELECTION_CONFIG="$selection_config" CONFIG_PATH="$CONFIGS/decoder-$config_mode-refit.json" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

selection_dir = Path(os.environ["SELECTION"])
selection_path = selection_dir / "selected_checkpoint.json"
selection = json.loads(selection_path.read_text(encoding="utf-8"))
base = json.loads(Path(os.environ["SELECTION_CONFIG"]).read_text(encoding="utf-8"))
refit_dir = Path(os.environ["REFIT"])
base.update({
    "run_id": refit_dir.name,
    "phase": "refit",
    "output_dir": str(refit_dir),
    "eval_steps": 0,
    "save_steps": 0,
    "selection_summary_path": str(selection_path),
    "selected_global_step": selection["selected_global_step"],
    "wandb_project": os.environ["WANDB_PROJECT"],
    "wandb_entity": os.environ["WANDB_ENTITY"] or None,
})
path = Path(os.environ["CONFIG_PATH"])
if path.exists():
    if json.loads(path.read_text(encoding="utf-8")) != base:
        raise SystemExit(f"immutable generated config mismatch; refusing overwrite: {path}")
else:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(base, handle, indent=2, sort_keys=True)
        handle.write("\n")
PY
}
write_decoder_eval() { local mode="$1" source="$2" adapter="$3" name="$4"; MODE="$mode" SOURCE="$source" ADAPTER="$adapter" NAME="$name" "$PYTHON" - "$CONFIGS/$name.json" <<'PY'
import json, os, sys
mode=os.environ["MODE"]; source=os.environ["SOURCE"]
p={"run_id":os.environ["NAME"],"mode":mode,"model_path":os.environ["QWEN_MODEL"],"model_revision":os.environ["QWEN_REV"],"adapter_path":os.environ["ADAPTER"],"source":source,"prepared_manifest":os.environ["MANIFEST"],"validation_sha256":"" if source=="selection_dev" else os.environ["VALIDATION_SHA256"],"output_dir":os.environ["ROOT"]+"/outputs/standard-evals/"+os.environ["NAME"],"tensor_parallel_size":int(os.environ["NUM_GPUS"]),"max_model_len":2048 if mode=="direct" else 4096,"max_new_tokens":256 if mode=="direct" else 1536,"gpu_memory_utilization":0.9,"enforce_eager":True,"disable_flashinfer_sampler":True,"max_lora_rank":32,"wandb_project":os.environ["WANDB_PROJECT"],"wandb_entity":os.environ["WANDB_ENTITY"] or None}
path = os.path.abspath(sys.argv[1])
if os.path.exists(path):
    if json.load(open(path, encoding="utf-8")) != p:
        raise SystemExit(f"immutable generated config mismatch; refusing overwrite: {path}")
else:
    with open(path,"x",encoding="utf-8") as f: json.dump(p,f,indent=2,sort_keys=True);f.write("\n")
PY
}
reuse_direct_evaluation_stage() {
  local step="$1" artifact="$2" status="$3"
  if ! validate_stage_artifacts "$step" "$artifact"; then
    ledger failed "$step" "direct_evaluation_reuse_artifact_validation_failed"
    echo "direct-evaluation recovery artifact failed strict validation: $artifact" >&2
    return 1
  fi
  ledger "$status" "$step" "$artifact"
}
run_direct_refit_recovery() {
  local selection="$1" refit="$2" step name artifact
  reuse_direct_evaluation_stage "decoder-direct-selection" "$selection/standard_training_complete.json" "reused_verified_parent"
  local -a steps=()
  mapfile -t steps < <("$PYTHON" - "$selection/standard_training_complete.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(*payload["selection_candidate_steps"], sep="\n")
PY
)
  [[ ${#steps[@]} -gt 0 ]] || { echo "direct-evaluation recovery has no retained decoder checkpoints" >&2; return 1; }
  for step in "${steps[@]}"; do
    name="$DIRECT_EVALUATION_PREFIX-decoder-direct-dev-step-$step"
    artifact="$ROOT/outputs/standard-evals/$name/aggregate_metrics.json"
    reuse_direct_evaluation_stage "decoder-direct-dev-step-$step" "$artifact" "reused_verified_direct_evaluation"
  done
  reuse_direct_evaluation_stage "decoder-direct-select-checkpoint" "$selection/selected_checkpoint.json" "reused_verified_direct_evaluation"
  write_decoder_refit direct "$selection" "$refit"
  execute_stage "decoder-direct-refit" "$refit/standard_training_complete.json" "$refit" "$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" "$ROOT/scripts/train_standard_decoder_sft.py" --config "$CONFIGS/decoder-direct-refit.json"
  local final="$(run_name decoder-direct-final)"
  write_decoder_eval direct frozen_validation "$refit/adapter" "$final"
  execute_stage "decoder-direct-final" "$ROOT/outputs/standard-evals/$final/aggregate_metrics.json" "$ROOT/outputs/standard-evals/$final" "$PYTHON" "$ROOT/scripts/evaluate_standard_decoder_vllm.py" --config "$CONFIGS/$final.json"
}
run_decoder() {
  local mode="$1"; local selection="$2"; local refit="$3"; local config_mode="${mode//_/-}"
  local selection_step="decoder-$config_mode-selection"
  execute_stage "$selection_step" "$selection/standard_training_complete.json" "$selection" "$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" "$ROOT/scripts/train_standard_decoder_sft.py" --config "$CONFIGS/decoder-$config_mode-selection.json"
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
    execute_stage "$eval_step" "$ROOT/outputs/standard-evals/$name/aggregate_metrics.json" "$ROOT/outputs/standard-evals/$name" "$PYTHON" "$ROOT/scripts/evaluate_standard_decoder_vllm.py" --config "$CONFIGS/$name.json"
    metric_args+=(--evaluation-metrics "$ROOT/outputs/standard-evals/$name/aggregate_metrics.json")
  done
  local select_step="decoder-$config_mode-select-checkpoint"
  execute_stage "$select_step" "$selection/selected_checkpoint.json" "$selection/selected_checkpoint.json" "$PYTHON" "$ROOT/scripts/select_standard_decoder_checkpoint.py" --selection-run-dir "$selection" "${metric_args[@]}"
  write_decoder_refit "$mode" "$selection" "$refit"
  local refit_step="decoder-$config_mode-refit"
  execute_stage "$refit_step" "$refit/standard_training_complete.json" "$refit" "$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" "$ROOT/scripts/train_standard_decoder_sft.py" --config "$CONFIGS/decoder-$config_mode-refit.json"
  local final="$(run_name decoder-$config_mode-final)"; local final_step="decoder-$config_mode-final"
  write_decoder_eval "$mode" frozen_validation "$refit/adapter" "$final"
  execute_stage "$final_step" "$ROOT/outputs/standard-evals/$final/aggregate_metrics.json" "$ROOT/outputs/standard-evals/$final" "$PYTHON" "$ROOT/scripts/evaluate_standard_decoder_vllm.py" --config "$CONFIGS/$final.json"
}
write_encoder_refit() { local tag="$1" selection="$2" refit="$3"; TAG="$tag" SELECTION="$selection" REFIT="$refit" "$PYTHON" - "$CONFIGS/encoder-$tag-refit.json" <<'PY'
import json,os,sys
base=json.load(open(f"{os.environ['RUNTIME_ROOT']}/configs/encoder-{os.environ['TAG']}-selection.json",encoding="utf-8"));base.update({"run_id":os.path.basename(os.environ["REFIT"]),"phase":"refit","output_dir":os.environ["REFIT"],"eval_steps":0,"save_steps":0,"selection_metadata_path":os.environ["SELECTION"]+"/standard_encoder_training_complete.json"})
path = os.path.abspath(sys.argv[1])
if os.path.exists(path):
    if json.load(open(path, encoding="utf-8")) != base:
        raise SystemExit(f"immutable generated config mismatch; refusing overwrite: {path}")
else:
    with open(path,"x",encoding="utf-8") as f:json.dump(base,f,indent=2,sort_keys=True);f.write("\n")
PY
}
run_encoder() {
  local tag="$1"; local selection="$2"; local refit="$3"; local selection_step="encoder-$tag-selection"
  execute_stage "$selection_step" "$selection/standard_encoder_training_complete.json" "$selection" "$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" "$ROOT/scripts/train_standard_encoder.py" --config "$CONFIGS/encoder-$tag-selection.json"
  write_encoder_refit "$tag" "$selection" "$refit"
  local refit_step="encoder-$tag-refit"
  execute_stage "$refit_step" "$refit/standard_encoder_training_complete.json" "$refit" "$TORCHRUN" --standalone --nproc_per_node="$NUM_GPUS" "$ROOT/scripts/train_standard_encoder.py" --config "$CONFIGS/encoder-$tag-refit.json"
  local final="$(run_name encoder-$tag-final)"; FINAL="$final" REFIT="$refit" "$PYTHON" - "$CONFIGS/$final.json" <<'PY'
import json,os,sys
p={"run_id":os.environ["FINAL"],"source":"frozen_validation","training_metadata_path":os.environ["REFIT"]+"/standard_encoder_training_complete.json","prepared_manifest":os.environ["MANIFEST"],"validation_sha256":os.environ["VALIDATION_SHA256"],"output_dir":os.environ["ROOT"]+"/outputs/standard-encoder-evals/"+os.environ["FINAL"],"per_device_eval_batch_size":int(os.environ["ENCODER_BATCH"]),"wandb_project":os.environ["WANDB_PROJECT"],"wandb_entity":os.environ["WANDB_ENTITY"] or None}
path = os.path.abspath(sys.argv[1])
if os.path.exists(path):
    if json.load(open(path, encoding="utf-8")) != p:
        raise SystemExit(f"immutable generated config mismatch; refusing overwrite: {path}")
else:
    with open(path,"x",encoding="utf-8") as f:json.dump(p,f,indent=2,sort_keys=True);f.write("\n")
PY
  local final_step="encoder-$tag-final"
  execute_stage "$final_step" "$ROOT/outputs/standard-encoder-evals/$final/aggregate_metrics.json" "$ROOT/outputs/standard-encoder-evals/$final" "$PYTHON" "$ROOT/scripts/evaluate_standard_encoder.py" --config "$CONFIGS/$final.json"
}

if [[ "$RESUME_MODE" == true ]]; then
  record_resume_lineage
else
  ledger started matrix "approved sequential matrix"
fi
if [[ "$DIRECT_EVALUATION_CONTINUATION_MODE" == true ]]; then
  run_direct_refit_recovery "$DIRECT_SELECTION" "$DIRECT_REFIT"
else
  run_decoder direct "$DIRECT_SELECTION" "$DIRECT_REFIT"
fi
run_decoder human_feedback "$FEEDBACK_SELECTION" "$FEEDBACK_REFIT"
run_encoder qwen3 "$QWEN3_SELECTION" "$QWEN3_REFIT"
run_encoder nvembed "$NV_SELECTION" "$NV_REFIT"
ledger completed matrix "$RUNTIME_ROOT/matrix_manifest.json"
echo "matrix completed; aggregate-only ledger: $LEDGER"
