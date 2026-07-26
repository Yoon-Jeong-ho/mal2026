# AI-Hub integer three-axis score pretraining contract

This experiment trains **all parameters** of `Qwen/Qwen3-Embedding-8B` with
one of two task heads: bounded sigmoid regression (`3` logits) or ordinal
cumulative classification (`3 x 4` logits). This is the required
`AI-Hub full-parameter tuning -> MAL2026 LoRA` arm; it is not an AI-Hub LoRA
proxy. A LoRA-on-AI-Hub variant, if run later, is an optional ablation only.

## Data and selection

- The only inputs are the canonical manifest
  `data/manifests/aihub_human_feedback_v1.json` and its `selection_train`,
  `selection_dev`, and `refit_train` files.
- Targets are `content`, `organization`, and `expression`. Each source decimal
  is converted with Decimal `ROUND_HALF_UP` and constrained to integer `1..5`.
- The source `average` member is validated as a schema member but is never
  indexed, used as a feature, or used as a target. A sentinel unit test enforces
  this boundary.
- Model selection uses only AI-Hub `selection_dev`, every 100 optimizer steps,
  with a maximum of 20 epochs and patience 3. The ordering is lowest macro
  integer RMSE, highest macro integer Spearman, lowest continuous RMSE, then
  earlier global step.
- Refit starts from the same seed and fresh base initialization, uses all
  48,016 `refit_train` rows, and runs for exactly the selected global-step
  count. It retains the selection run's original scheduler horizon and stops
  at the selected optimizer step, so the learning-rate trajectory is also
  replayed rather than compressed. It never opens `eval/validation.jsonl`.
- The historical 2026-07-17 four-axis continuous run's 1,900 selected steps
  inform the schedule provenance only. Its weights are not loaded and it is not
  a primary initialization.

## State and completion schemas

Each refit writes a complete `full_model/`, `full_model_state.json`, and
`training_complete.json` under the ignored output root. The state schema is
`mal2026-aihub-integer-score-full-state-v2`. Every regular file in the model
directory is SHA-256 hashed, and a canonical inventory hash hard-binds the
whole artifact. The model contains the complete fully tuned backbone and its
matched score head. The matched head tensors also receive a separate
dtype/shape/value SHA-256 so the aggregate can hard-bind the selected head
without loading or duplicating the 17 GB backbone.

Downstream score-matrix consumers may use:

- `matched_full`: only for the exact same head kind, axes, model revision,
  and tensor shapes;
- `full_backbone_and_matched_head_then_mal_lora`: stream every `backbone.*`
  tensor and the matched `score_head.*` tensors into a fresh pinned Qwen3
  score model, then attach a **fresh** MAL2026 LoRA. During MAL2026 adaptation,
  the new LoRA and retained matched head remain trainable while the fully tuned
  backbone is frozen by PEFT.

The completion schema is
`mal2026-aihub-integer-score-pretrain-completion-v2`. It records checksums,
selected step/metrics, split lineage, initialization replay, state semantics,
and aggregate-only privacy status; it contains no row text, IDs, feedback, or
predictions.

Config, completion, state metadata, and aggregate artifacts all carry the
following exact downstream contract fields:

```json
{
  "integer_target_used": true,
  "target_projection": "official_half_up",
  "score_fields": ["content", "organization", "expression"],
  "average_target_used": false
}
```

Each aggregate head result hard-binds `completion_path` and
`completion_sha256` as well as `artifact_path`, `artifact_sha256`,
`state_metadata_path`, and `state_metadata_sha256`.

## Runner

```bash
# Inspection only; starts no GPU process.
.venv-standard/bin/python scripts/orchestrate_official_aihub_score_pretrain.py \
  --config configs/official_aihub_integer_score_pretrain.v1.json --dry-run

# When GPUs 0--3 are available: GPU0 one-update selection+refit smoke, then
# FSDP full-shard selection and exact-step fresh refit for each head.
.venv-standard/bin/python scripts/orchestrate_official_aihub_score_pretrain.py \
  --config configs/official_aihub_integer_score_pretrain.v1.json
```
