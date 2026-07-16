# Experiment configuration templates

These JSON files are intentionally **not runnable**. Before a smoke run, copy
the relevant file to an ignored run directory and replace every `REQUIRED_*`
value with an immutable commit SHA or approved architecture value. The shared
config validator fails closed for branch/tag revisions, unset pooling, or empty
adapter target modules. Do not put credentials, prompts, writing text, IDs, or
W&B tokens in these files.


## Human-feedback encoder data interface

Encoder `selection` and `refit` runs do **not** accept `eval/train.jsonl`.
They require an absolute ignored `data/processed/.../manifest.json` through
`--prepared-manifest`. The manifest is aggregate-only and uses protocol
`aihub_human_feedback_score_v1`; it names exactly the relative files
`selection_train.jsonl`, `selection_dev.jsonl`, and `refit_train.jsonl`, each
with a SHA-256 and record count. It also has an aggregate source fingerprint,
eligibility summary, and group-split summary. It must not contain writing text,
feedback, IDs, or row tables. Prepared row files are restricted and contain
`id`, `prompt`, `essay`, four-field `score`, and complete `feedback`; encoder
input uses only prompt and essay and targets only score.

Source selection is the prepared approximately 20% development split and
uses four-score macro MAE. `final-eval` remains limited to the canonical
`eval/validation.jsonl` and verifies its refit lineage without accepting any
training-data argument.
