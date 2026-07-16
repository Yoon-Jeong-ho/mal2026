# Qwen2.5 decoder experiment interface

`src/mal2026/decoder_train.py` and `decoder_eval.py` implement two decoder
regimes: `direct` emits four scores, while `human_feedback` emits the source
human feedback followed by those scores. Both receive **only** the prompt and
student response in their user message. IDs, split metadata, scores, and
feedback never enter the model input or W&B.

## Prepared source-data contract

Preparation writes restricted JSONL files under ignored
`data/processed/aihub_human_feedback_v1/` and an aggregate-only tracked
manifest at `data/manifests/aihub_human_feedback_v1.json`. A runtime config
binds both the directory and the manifest SHA-256. The manifest has exactly:

```json
{
  "schema_version": 1,
  "dataset_id": "aihub_human_feedback_v1",
  "files": {
    "selection_train": {"filename":"selection_train.jsonl","sha256":"...","record_count":1},
    "selection_dev": {"filename":"selection_dev.jsonl","sha256":"...","record_count":1},
    "refit_train": {"filename":"refit_train.jsonl","sha256":"...","record_count":1}
  }
}
```

Rows have exactly ordered keys `id`, `prompt`, `essay`, `score`, `feedback`.
`score` has ordered `content`, `organization`, `expression`, `average` numeric
values in `[1,5]` quantized to two decimals. `feedback` has ordered nonblank
human strings `holistic`, `content_1`, `content_2`, `content_3`,
`organization_1`, `organization_2`, `expression_1`, `expression_2`, `task_1`.
The loader rejects duplicate JSON keys, key reordering, blank values, invalid
scores, file digest/count mismatches, and train/dev ID overlap. Selection reads
the preparation-owned `selection_train`/`selection_dev` split; refit reads
`refit_train`. The runner does not resplit source data.

The human-feedback assistant target is exactly:

```json
{"feedback":{"holistic":"...","content_1":"...","content_2":"...","content_3":"...","organization_1":"...","organization_2":"...","expression_1":"...","expression_2":"...","task_1":"..."},"scores":{"content":3.20,"organization":3.50,"expression":3.75,"average":3.48}}
```

The direct target is only the ordered `scores` object. Decoder parsing rejects
prose/markdown, duplicate or reordered keys, missing/blank feedback, non-numeric
or non-two-decimal scores, and out-of-range scores. Invalid output gets the
saved selection-train mean; no partial numeric extraction occurs.

## Budgets, selection, and final evaluation

Both modes use full pinned chat-template accounting and deterministic 75:25
head:tail truncation of **input only**. Targets are never truncated. Direct uses
a 2,048-token rendered-chat budget and `max_new_tokens=256`; human-feedback
uses 4,096 and `max_new_tokens=1536`. Preparation applies the common
human-feedback target eligibility gate (`<=1536` pinned-token tokens) before
splitting, so the four experiments have the same eligible rows.

Selection chooses the lowest source-dev four-score macro-MAE only among
checkpoints with strict parse validity at least `0.99`; otherwise selection
fails. Refit must cryptographically bind to that selected update and fallback
mean. Frozen `eval/validation.jsonl` is used only once after refit; it has no
feedback, so human-feedback output is schema-validated but never compared to a
gold explanation. Final metrics report component MAE/RMSE and macro MAE plus
parse-failure rate.

Use filled, ignored copies of `configs/decoder-direct.template.json` or
`configs/decoder-human-feedback-score.template.json`. Model/tokenizer revisions
must be immutable commit SHAs. Runs use BF16 Accelerate/DDP, LoRA rank/alpha/
dropout `32/64/0.05`, one example/GPU, accumulation 8, and aggregate-only W&B.
For example after smoke gates:

```bash
PYTHONPATH=src accelerate launch --num_processes 8 scripts/train_decoder_sft.py --config /secure/configs/decoder-human-feedback-selection.json
```
