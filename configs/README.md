# Experiment configuration templates

Templates are intentionally not runnable. Copy one into an ignored run/config
location, replace each `REQUIRED_*` value with an immutable revision/hash, and
set the runtime config to the aggregate-only prepared-data manifest. Do not put
credentials, prompts, student writing, identifiers, labels, or W&B tokens in
configuration files.

Use `decoder-direct.template.json` for score-only Qwen SFT and
`decoder-human-feedback-score.template.json` for human-feedback→score Qwen
SFT. The shared validator fixes their token budgets, split fraction, and
prepared-data schema version.

## Encoder human-feedback input

Encoder selection/refit require the absolute canonical safe manifest
`data/manifests/aihub_human_feedback_v1.json` through `--prepared-manifest`.
It identifies the three fixed ignored prepared files with keys
`selection_train`, `selection_dev`, and `refit_train`; each has exactly
`filename`, `sha256`, and `record_count`. Files reside under
`data/processed/aihub_human_feedback_v1/`. Encoder input consumes only prompt
and essay, and its regression target consumes only the four-field score; the
required human feedback is validated but never passed to the encoder.

The prepared source development split is approximately 20%; selection uses
four-score macro MAE. Final evaluation accepts only frozen
`eval/validation.jsonl` and refit lineage.
